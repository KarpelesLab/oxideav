//! Baseline-JPEG encoder.
//!
//! Produces a standalone, self-contained JPEG packet per video frame:
//! SOI, JFIF APP0, DQT, SOF0, DHT (Annex K tables), SOS, entropy scan, EOI.
//! Handles 4:2:0 / 4:2:2 / 4:4:4 YUV planar input. No restart markers are
//! emitted (decoder still supports reading them).

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::jpeg::dct::fdct8x8;
use crate::jpeg::huffman::{
    DefaultHuffman, HuffTable, STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS, STD_AC_LUMA_BITS,
    STD_AC_LUMA_VALS, STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS,
};
use crate::jpeg::markers;
use crate::jpeg::quant::{scale_for_quality, DEFAULT_CHROMA_Q50, DEFAULT_LUMA_Q50};
use crate::jpeg::zigzag::ZIGZAG;

/// Quality factor 1..=100 (libjpeg style). 75 is a sensible default.
pub const DEFAULT_QUALITY: u8 = 75;

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("MJPEG encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("MJPEG encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    match pix {
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {}
        _ => {
            return Err(Error::unsupported(format!(
                "MJPEG encoder: pixel format {:?} not supported",
                pix
            )))
        }
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    Ok(Box::new(MjpegEncoder {
        output_params,
        width,
        height,
        pix,
        quality: DEFAULT_QUALITY,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct MjpegEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for MjpegEncoder {
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
                        "MJPEG encoder: frame dimensions do not match encoder config",
                    ));
                }
                if v.format != self.pix {
                    return Err(Error::invalid(format!(
                        "MJPEG encoder: frame format {:?} does not match encoder format {:?}",
                        v.format, self.pix
                    )));
                }
                let data = encode_jpeg(v, self.quality)?;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("MJPEG encoder: video frames only")),
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

// ---- Encoding ------------------------------------------------------------

fn encode_jpeg(frame: &VideoFrame, quality: u8) -> Result<Vec<u8>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (h_factor, v_factor) = match frame.format {
        PixelFormat::Yuv444P => (1u8, 1u8),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv420P => (2, 2),
        _ => {
            return Err(Error::unsupported(
                "MJPEG encoder: unsupported pixel format",
            ))
        }
    };
    if frame.planes.len() != 3 {
        return Err(Error::invalid("MJPEG encoder: expected 3 planes"));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);

    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // JFIF APP0.
    write_jfif_app0(&mut out);
    // DQT (both tables, precision 0 = 8-bit).
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    // SOF0.
    write_sof0(&mut out, width as u16, height as u16, h_factor, v_factor);
    // DHT: all 4 default tables.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    // SOS.
    write_sos(&mut out);
    // Scan data.
    write_scan(
        &mut out, frame, width, height, h_factor, v_factor, &luma_q, &chroma_q, &huff,
    )?;
    // EOI.
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

fn write_length_prefix(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let len = (payload.len() + 2) as u16;
    out.push(0xFF);
    out.push(marker);
    out.push((len >> 8) as u8);
    out.push(len as u8);
    out.extend_from_slice(payload);
}

fn write_jfif_app0(out: &mut Vec<u8>) {
    // "JFIF\0" + version 1.01 + density unit 0 (aspect) + 1x1 density + 0 thumbnail.
    let payload = [
        b'J', b'F', b'I', b'F', 0, // identifier
        1, 1, // version
        0, // units
        0, 1, // Xdensity
        0, 1, // Ydensity
        0, 0, // thumbnail w, h
    ];
    write_length_prefix(out, markers::APP0, &payload);
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, nat_order: &[u16; 64]) {
    let mut payload = Vec::with_capacity(1 + 64);
    payload.push(table_id & 0x0F); // precision=0, id=table_id
    for k in 0..64 {
        payload.push(nat_order[ZIGZAG[k]].min(255) as u8);
    }
    write_length_prefix(out, markers::DQT, &payload);
}

fn write_sof0(out: &mut Vec<u8>, width: u16, height: u16, h: u8, v: u8) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // components
                     // Y
    payload.push(1);
    payload.push((h << 4) | v);
    payload.push(0);
    // Cb
    payload.push(2);
    payload.push(0x11);
    payload.push(1);
    // Cr
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF0, &payload);
}

fn write_dht(out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]) {
    let mut payload = Vec::with_capacity(1 + 16 + values.len());
    payload.push(((class & 0x01) << 4) | (id & 0x0F));
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_length_prefix(out, markers::DHT, &payload);
}

fn write_sos(out: &mut Vec<u8>) {
    let payload: [u8; 10] = [
        3, // components
        1, 0x00, // Y uses DC=0 AC=0
        2, 0x11, // Cb uses DC=1 AC=1
        3, 0x11, // Cr uses DC=1 AC=1
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

#[allow(clippy::too_many_arguments)]
fn write_scan(
    out: &mut Vec<u8>,
    frame: &VideoFrame,
    width: usize,
    height: usize,
    h_factor: u8,
    v_factor: u8,
    luma_q: &[u16; 64],
    chroma_q: &[u16; 64],
    huff: &DefaultHuffman,
) -> Result<()> {
    let mut bw = BitWriter::new(out);
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    let y_plane = &frame.planes[0];
    let cb_plane = &frame.planes[1];
    let cr_plane = &frame.planes[2];

    let (c_w, c_h) = match frame.format {
        PixelFormat::Yuv444P => (width, height),
        PixelFormat::Yuv422P => (width.div_ceil(2), height),
        PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
        _ => unreachable!(),
    };

    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Luma blocks.
            for by in 0..v_factor as usize {
                for bx in 0..h_factor as usize {
                    let x0 = mx * mcu_w_px + bx * 8;
                    let y0 = my * mcu_h_px + by * 8;
                    let mut blk = [0.0f32; 64];
                    fill_block(
                        &mut blk,
                        &y_plane.data,
                        y_plane.stride,
                        width,
                        height,
                        x0,
                        y0,
                    );
                    encode_block(
                        &mut bw,
                        &mut blk,
                        luma_q,
                        &mut prev_dc_y,
                        &huff.luma_dc,
                        &huff.luma_ac,
                    );
                }
            }
            // Chroma: one block per component per MCU.
            let cb_x0 = mx * 8;
            let cb_y0 = my * 8;
            let mut blk_cb = [0.0f32; 64];
            fill_block(
                &mut blk_cb,
                &cb_plane.data,
                cb_plane.stride,
                c_w,
                c_h,
                cb_x0,
                cb_y0,
            );
            encode_block(
                &mut bw,
                &mut blk_cb,
                chroma_q,
                &mut prev_dc_cb,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );
            let mut blk_cr = [0.0f32; 64];
            fill_block(
                &mut blk_cr,
                &cr_plane.data,
                cr_plane.stride,
                c_w,
                c_h,
                cb_x0,
                cb_y0,
            );
            encode_block(
                &mut bw,
                &mut blk_cr,
                chroma_q,
                &mut prev_dc_cr,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );
        }
    }

    bw.finish();
    Ok(())
}

/// Fill an 8×8 f32 block from a plane, repeating edge pixels for any MCU
/// blocks that extend past the picture boundary. Subtracts 128 (level shift).
fn fill_block(
    dst: &mut [f32; 64],
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
) {
    for j in 0..8 {
        let y = (y0 + j).min(h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(w.saturating_sub(1));
            let v = plane[y * stride + x] as i32;
            dst[j * 8 + i] = (v - 128) as f32;
        }
    }
}

fn encode_block(
    bw: &mut BitWriter<'_>,
    block: &mut [f32; 64],
    quant: &[u16; 64],
    prev_dc: &mut i32,
    dc_huff: &HuffTable,
    ac_huff: &HuffTable,
) {
    fdct8x8(block);
    // Quantise.
    let mut q = [0i32; 64];
    for k in 0..64 {
        let v = block[k] / quant[k] as f32;
        q[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    // DC.
    let dc_diff = q[0] - *prev_dc;
    *prev_dc = q[0];
    let (size, bits) = category(dc_diff);
    let hc = dc_huff.encode[size as usize];
    bw.write_bits(hc.code as u32, hc.len as u32);
    if size > 0 {
        bw.write_bits(bits, size as u32);
    }
    // AC in zigzag order.
    let mut run: u32 = 0;
    for k in 1..64 {
        let v = q[ZIGZAG[k]];
        if v == 0 {
            run += 1;
        } else {
            while run >= 16 {
                // ZRL.
                let zc = ac_huff.encode[0xF0];
                bw.write_bits(zc.code as u32, zc.len as u32);
                run -= 16;
            }
            let (sz, bv) = category(v);
            let rs = ((run as u8) << 4) | sz;
            let ac = ac_huff.encode[rs as usize];
            bw.write_bits(ac.code as u32, ac.len as u32);
            if sz > 0 {
                bw.write_bits(bv, sz as u32);
            }
            run = 0;
        }
    }
    if run > 0 {
        let eob = ac_huff.encode[0x00];
        bw.write_bits(eob.code as u32, eob.len as u32);
    }
}

/// JPEG category: `(size, bits)` where `size` = min bits to hold |v|, and
/// `bits` is the value encoded per Annex F (positive → plain, negative →
/// bitwise complement fitted to `size` bits).
fn category(v: i32) -> (u8, u32) {
    if v == 0 {
        return (0, 0);
    }
    let abs = v.unsigned_abs();
    let size = 32 - abs.leading_zeros();
    debug_assert!(size <= 16);
    let bits = if v > 0 {
        abs
    } else {
        (abs ^ ((1u32 << size) - 1)) & ((1u32 << size) - 1)
    };
    // For negative v, re-derive per spec: bits = v + ((1<<size) - 1) mod 2^size.
    let bits = if v > 0 {
        bits
    } else {
        (((1u32 << size).wrapping_sub(1)) as i32 + v) as u32
    };
    (size as u8, bits)
}

// ---- Bit writer with 0xFF stuffing --------------------------------------

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    buf: u32,
    nbits: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            buf: 0,
            nbits: 0,
        }
    }

    fn write_bits(&mut self, value: u32, len: u32) {
        if len == 0 {
            return;
        }
        // Mask value to `len` bits.
        let v = value & ((1u32 << len) - 1);
        self.buf = (self.buf << len) | v;
        self.nbits += len;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let b = ((self.buf >> self.nbits) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    fn finish(&mut self) {
        if self.nbits > 0 {
            // Pad with 1s per Annex F, then flush last byte.
            let pad = 8 - self.nbits;
            self.buf = (self.buf << pad) | ((1u32 << pad) - 1);
            self.nbits = 0;
            let b = (self.buf & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_zero() {
        assert_eq!(category(0), (0, 0));
    }

    #[test]
    fn category_small() {
        // v=+1: size=1, bits=1
        assert_eq!(category(1), (1, 1));
        // v=-1: size=1, bits=0
        assert_eq!(category(-1), (1, 0));
        // v=+5: size=3, bits=5
        assert_eq!(category(5), (3, 5));
        // v=-5: size=3, bits=(-5 + 7) = 2
        assert_eq!(category(-5), (3, 2));
    }

    #[test]
    fn bit_writer_stuffs_ff() {
        let mut buf = Vec::new();
        let mut bw = BitWriter::new(&mut buf);
        bw.write_bits(0xFF, 8);
        bw.finish();
        assert_eq!(buf, vec![0xFF, 0x00]);
    }
}
