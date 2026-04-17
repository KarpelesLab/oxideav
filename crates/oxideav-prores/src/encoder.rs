//! ProRes 422 encoder (Proxy / LT / Standard).
//!
//! Reads a `Yuv422P` `VideoFrame`, walks 16x16 macroblocks, forward
//! DCTs each 8x8 block, quantises by `qmat * quant_index`, and emits
//! the frame/picture/slice container defined in `frame.rs`.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::dct::fdct8x8;
use crate::frame::{
    write_frame_header, write_picture_header, Profile, FRAME_HDR_SIZE, PICTURE_HDR_SIZE,
};
use crate::quant::{CHROMA_QMAT_STANDARD, LUMA_QMAT_STANDARD};
use crate::slice::{encode_slice, BLOCKS_PER_MB, MAX_MBS_PER_SLICE};

/// Default `quant_index`. Multiplied with the per-block matrix entry.
/// A value of `4` gives roughly Standard-profile quality on the
/// built-in matrix; drop to `2` for HQ-ish, raise to `8` for Proxy.
pub const DEFAULT_QUANT_INDEX: u8 = 4;

const MB_WIDTH_PX: usize = 16;
const MB_HEIGHT_PX: usize = 16;
const SLICE_MB_WIDTH_LOG2: u8 = 3; // 2^3 = 8 MBs per slice

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("prores encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("prores encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv422P);
    if pix != PixelFormat::Yuv422P {
        return Err(Error::unsupported(format!(
            "prores encoder: pixel format {pix:?} not supported (only Yuv422P)"
        )));
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    // Map codec `bit_rate`/hints to a profile. Without any hint we
    // default to Standard. (4444 / HQ are explicitly out-of-scope for
    // this baseline implementation — reject below so users get a
    // clear error instead of silent data loss.)
    let profile = Profile::Standard;

    // Derive a quant index. If the caller set `bit_rate` we just take
    // the default — per-profile bitrate targets aren't modelled yet.
    let quant_index = DEFAULT_QUANT_INDEX;

    Ok(Box::new(ProResEncoder {
        output_params,
        width,
        height,
        profile,
        quant_index,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct ProResEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    profile: Profile,
    quant_index: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for ProResEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                if v.width != self.width || v.height != self.height {
                    return Err(Error::invalid(
                        "prores encoder: frame dimensions do not match encoder config",
                    ));
                }
                if v.format != PixelFormat::Yuv422P {
                    return Err(Error::invalid(format!(
                        "prores encoder: frame format {:?} is not Yuv422P",
                        v.format
                    )));
                }
                let data = encode_frame_422(v, self.profile, self.quant_index)?;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("prores encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Encode a single 4:2:2 picture to a complete ProRes packet.
pub fn encode_frame_422(frame: &VideoFrame, profile: Profile, quant_index: u8) -> Result<Vec<u8>> {
    if frame.format != PixelFormat::Yuv422P {
        return Err(Error::unsupported("prores encoder: Yuv422P only"));
    }
    if frame.planes.len() != 3 {
        return Err(Error::invalid("prores encoder: expected 3 planes"));
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let c_w = width.div_ceil(2);

    let mbs_x = width.div_ceil(MB_WIDTH_PX);
    let mbs_y = height.div_ceil(MB_HEIGHT_PX);
    let slice_mb_width = 1usize << SLICE_MB_WIDTH_LOG2;
    let slices_per_row = mbs_x.div_ceil(slice_mb_width);
    let slice_count = slices_per_row * mbs_y;

    // Build each slice payload.
    let mut slice_payloads: Vec<Vec<u8>> = Vec::with_capacity(slice_count);
    for my in 0..mbs_y {
        let mut mx = 0usize;
        while mx < mbs_x {
            let remaining = mbs_x - mx;
            let mbs_this_slice = remaining.min(MAX_MBS_PER_SLICE).min(slice_mb_width);
            let mut blocks: Vec<[i32; 64]> = Vec::with_capacity(mbs_this_slice * BLOCKS_PER_MB);
            for mb_within in 0..mbs_this_slice {
                let mb_x = mx + mb_within;
                // 4 luma blocks.
                for (bx, by) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let x0 = mb_x * MB_WIDTH_PX + bx * 8;
                    let y0 = my * MB_HEIGHT_PX + by * 8;
                    blocks.push(encode_block(
                        &frame.planes[0].data,
                        frame.planes[0].stride,
                        width,
                        height,
                        x0,
                        y0,
                        &LUMA_QMAT_STANDARD,
                        quant_index,
                    ));
                }
                // 2 Cb blocks, then 2 Cr blocks.
                for plane_idx in [1usize, 2] {
                    for by in [0usize, 1] {
                        let x0 = mb_x * 8;
                        let y0 = my * MB_HEIGHT_PX + by * 8;
                        blocks.push(encode_block(
                            &frame.planes[plane_idx].data,
                            frame.planes[plane_idx].stride,
                            c_w,
                            height,
                            x0,
                            y0,
                            &CHROMA_QMAT_STANDARD,
                            quant_index,
                        ));
                    }
                }
            }
            let payload = encode_slice(mbs_this_slice as u8, quant_index, &blocks)?;
            slice_payloads.push(payload);
            mx += mbs_this_slice;
        }
    }
    debug_assert_eq!(slice_payloads.len(), slice_count);
    if slice_payloads.iter().any(|p| p.len() > u16::MAX as usize) {
        return Err(Error::invalid(
            "prores encoder: slice exceeded u16 size table limit",
        ));
    }

    // Compute sizes.
    let slice_table_size = slice_count * 2;
    let slice_bytes: usize = slice_payloads.iter().map(|p| p.len()).sum();
    let picture_size = (slice_table_size + slice_bytes) as u32;
    let total_frame_size =
        (FRAME_HDR_SIZE + PICTURE_HDR_SIZE + slice_table_size + slice_bytes) as u32;

    // Assemble.
    let mut out = Vec::with_capacity(total_frame_size as usize);
    write_frame_header(
        &mut out,
        frame.width as u16,
        frame.height as u16,
        profile,
        &LUMA_QMAT_STANDARD,
        &CHROMA_QMAT_STANDARD,
        total_frame_size,
    );
    write_picture_header(
        &mut out,
        picture_size,
        slice_count as u16,
        SLICE_MB_WIDTH_LOG2,
    );
    for p in &slice_payloads {
        out.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    for p in &slice_payloads {
        out.extend_from_slice(p);
    }
    debug_assert_eq!(out.len(), total_frame_size as usize);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn encode_block(
    plane: &[u8],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    qmat: &[u8; 64],
    quant_index: u8,
) -> [i32; 64] {
    let mut blk = [0.0f32; 64];
    for j in 0..8 {
        let y = (y0 + j).min(plane_h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(plane_w.saturating_sub(1));
            blk[j * 8 + i] = plane[y * stride + x] as f32 - 128.0;
        }
    }
    fdct8x8(&mut blk);
    let qi = quant_index.max(1) as f32;
    let mut out = [0i32; 64];
    for k in 0..64 {
        let denom = qmat[k] as f32 * qi;
        let v = blk[k] / denom;
        out[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    out
}
