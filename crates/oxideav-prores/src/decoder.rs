//! ProRes 422 decoder (Proxy / LT / Standard).
//!
//! Wire format: see `frame.rs` for the frame/picture header layout
//! and `slice.rs` for the per-slice entropy format. This module glues
//! them together into a `VideoFrame` of `PixelFormat::Yuv422P`.

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
};

use crate::dct::idct8x8;
use crate::frame::{parse_frame_header, parse_picture_header};
use crate::slice::{decode_slice, BLOCKS_PER_MB};

const MB_WIDTH_PX: usize = 16; // 4:2:2 luma MB spans 16 px wide (2 blocks)
const MB_HEIGHT_PX: usize = 16; // and 16 px tall (2 blocks)

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(ProResDecoder {
        codec_id: params.codec_id.clone(),
        pending: None,
        eof: false,
    }))
}

struct ProResDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for ProResDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "prores decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_packet(&pkt.data, pkt.pts, pkt.time_base)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

pub fn decode_packet(data: &[u8], pts: Option<i64>, time_base: TimeBase) -> Result<VideoFrame> {
    let (fh, after_frame) = parse_frame_header(data)?;
    let (ph, after_pic) = parse_picture_header(after_frame)?;

    let width = fh.width as usize;
    let height = fh.height as usize;
    if width == 0 || height == 0 {
        return Err(Error::invalid("prores: zero-sized frame"));
    }

    // Compute macroblock grid.
    let mbs_x = width.div_ceil(MB_WIDTH_PX);
    let mbs_y = height.div_ceil(MB_HEIGHT_PX);
    let slice_mb_width = 1usize << ph.log2_slice_mb_width;
    let slices_per_row = mbs_x.div_ceil(slice_mb_width);
    let expected_slice_count = slices_per_row * mbs_y;
    if ph.slice_count as usize != expected_slice_count {
        return Err(Error::invalid(format!(
            "prores: slice count mismatch (got {}, expected {expected_slice_count})",
            ph.slice_count
        )));
    }

    // Read slice-size table.
    let sizes_bytes = (ph.slice_count as usize) * 2;
    if after_pic.len() < sizes_bytes {
        return Err(Error::invalid("prores: slice-size table truncated"));
    }
    let mut slice_sizes = Vec::with_capacity(ph.slice_count as usize);
    for i in 0..ph.slice_count as usize {
        let off = i * 2;
        slice_sizes.push(u16::from_be_bytes(after_pic[off..off + 2].try_into().unwrap()) as usize);
    }
    let mut cursor = &after_pic[sizes_bytes..];

    // Allocate planes (aligned to MB grid — we'll crop on output).
    let padded_w = mbs_x * MB_WIDTH_PX;
    let padded_h = mbs_y * MB_HEIGHT_PX;
    let padded_c_w = padded_w / 2;
    let mut y_plane = vec![0u8; padded_w * padded_h];
    let mut cb_plane = vec![0u8; padded_c_w * padded_h];
    let mut cr_plane = vec![0u8; padded_c_w * padded_h];

    let mut slice_idx = 0usize;
    for my in 0..mbs_y {
        let mut mx = 0usize;
        while mx < mbs_x {
            let slice_bytes = slice_sizes[slice_idx];
            if cursor.len() < slice_bytes {
                return Err(Error::invalid("prores: slice payload truncated"));
            }
            let slice_data = &cursor[..slice_bytes];
            cursor = &cursor[slice_bytes..];
            slice_idx += 1;

            let decoded = decode_slice(slice_data)?;
            let expected_mbs_this_slice = slice_mb_width.min(mbs_x - mx);
            if decoded.mb_count as usize != expected_mbs_this_slice {
                return Err(Error::invalid(format!(
                    "prores: slice at ({mx},{my}) had mb_count {} but grid wants {expected_mbs_this_slice}",
                    decoded.mb_count
                )));
            }

            // Dequant + IDCT + paste.
            for mb_within in 0..decoded.mb_count as usize {
                let mb_x = mx + mb_within;
                let base = mb_within * BLOCKS_PER_MB;
                // 4 luma blocks — positions within MB: (0,0), (1,0), (0,1), (1,1)
                for (i, (bx, by)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                    let mut blk_f = dequant_to_f32(
                        &decoded.blocks[base + i],
                        &fh.luma_qmat,
                        decoded.quant_index,
                    );
                    idct8x8(&mut blk_f);
                    paste_block(
                        &mut y_plane,
                        padded_w,
                        mb_x * MB_WIDTH_PX + bx * 8,
                        my * MB_HEIGHT_PX + by * 8,
                        &blk_f,
                    );
                }
                // 2 Cb blocks — stacked vertically: (0,0), (0,1); chroma MB is 8x16
                for (i, by) in [0usize, 1].iter().enumerate() {
                    let mut blk_f = dequant_to_f32(
                        &decoded.blocks[base + 4 + i],
                        &fh.chroma_qmat,
                        decoded.quant_index,
                    );
                    idct8x8(&mut blk_f);
                    paste_block(
                        &mut cb_plane,
                        padded_c_w,
                        mb_x * 8,
                        my * MB_HEIGHT_PX + by * 8,
                        &blk_f,
                    );
                }
                // 2 Cr blocks — same layout as Cb.
                for (i, by) in [0usize, 1].iter().enumerate() {
                    let mut blk_f = dequant_to_f32(
                        &decoded.blocks[base + 6 + i],
                        &fh.chroma_qmat,
                        decoded.quant_index,
                    );
                    idct8x8(&mut blk_f);
                    paste_block(
                        &mut cr_plane,
                        padded_c_w,
                        mb_x * 8,
                        my * MB_HEIGHT_PX + by * 8,
                        &blk_f,
                    );
                }
            }

            mx += decoded.mb_count as usize;
        }
    }

    // Crop padded buffers back to the declared picture size.
    let c_w = width.div_ceil(2);
    let y_cropped = crop_plane(&y_plane, padded_w, width, height);
    let cb_cropped = crop_plane(&cb_plane, padded_c_w, c_w, height);
    let cr_cropped = crop_plane(&cr_plane, padded_c_w, c_w, height);

    Ok(VideoFrame {
        format: PixelFormat::Yuv422P,
        width: width as u32,
        height: height as u32,
        pts,
        time_base,
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_cropped,
            },
            VideoPlane {
                stride: c_w,
                data: cb_cropped,
            },
            VideoPlane {
                stride: c_w,
                data: cr_cropped,
            },
        ],
    })
}

fn dequant_to_f32(blk: &[i32; 64], qmat: &[u8; 64], quant_index: u8) -> [f32; 64] {
    let qi = quant_index.max(1) as i32;
    let mut out = [0.0f32; 64];
    for k in 0..64 {
        out[k] = (blk[k] * qmat[k] as i32 * qi) as f32;
    }
    out
}

fn paste_block(plane: &mut [u8], stride: usize, x0: usize, y0: usize, blk: &[f32; 64]) {
    for j in 0..8 {
        for i in 0..8 {
            let v = blk[j * 8 + i] + 128.0;
            let px = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v.round() as u8
            };
            plane[(y0 + j) * stride + x0 + i] = px;
        }
    }
}

fn crop_plane(src: &[u8], src_stride: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h];
    for y in 0..dst_h {
        out[y * dst_w..y * dst_w + dst_w]
            .copy_from_slice(&src[y * src_stride..y * src_stride + dst_w]);
    }
    out
}
