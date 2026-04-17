// Parallel-array index loops are idiomatic in codec/DCT code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! Apple ProRes 422 codec — minimal pure-Rust decoder + encoder.
//!
//! Scope: **ProRes 422 Proxy / LT / Standard** for 4:2:2 Y'CbCr 8-bit
//! input (`PixelFormat::Yuv422P`). HQ, 4444, and 4444 XQ are out of
//! scope for this baseline and will be rejected at encoder construction
//! time.
//!
//! The wire format follows SMPTE RDD 36 structurally (frame header
//! with `icpf` magic, picture header, per-slice table, slices holding
//! 8x8 DCT coefficients for 4 luma + 2 Cb + 2 Cr blocks per
//! macroblock) but uses a simplified entropy layer (unsigned /
//! signed exp-Golomb on zig-zag-scanned coefficients) that is
//! internally round-trip-exact but **not** bit-compatible with Apple
//! ProRes in the wild. See the module docs of [`bitstream`] and
//! [`slice`] for the deviation.
//!
//! ### Module layout
//!
//! * [`bitstream`] — MSB-first bit reader/writer, unsigned/signed exp-Golomb.
//! * [`dct`]       — Textbook f32 8x8 forward/inverse DCT.
//! * [`quant`]     — Hard-coded luma/chroma quant matrices + zig-zag table.
//! * [`slice`]     — Per-slice pack/unpack of coefficient blocks.
//! * [`frame`]     — Frame + picture header layouts.
//! * [`decoder`]   — `Packet -> VideoFrame` (Yuv422P).
//! * [`encoder`]   — `VideoFrame` (Yuv422P) -> `Packet`.

pub mod bitstream;
pub mod dct;
pub mod decoder;
pub mod encoder;
pub mod frame;
pub mod quant;
pub mod slice;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId, PixelFormat};

/// Public codec id.
pub const CODEC_ID_STR: &str = "prores";

/// Register the ProRes 422 decoder + encoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("prores_422_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_pixel_format(PixelFormat::Yuv422P);
    reg.register_decoder_impl(
        CodecId::new(CODEC_ID_STR),
        caps.clone(),
        decoder::make_decoder,
    );
    reg.register_encoder_impl(CodecId::new(CODEC_ID_STR), caps, encoder::make_encoder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{
        CodecId, CodecParameters, Frame, MediaType, PixelFormat, TimeBase, VideoFrame,
    };

    /// Build a 64x48 gradient in Yuv422P. Values are deliberately
    /// smooth so the codec's lossy path can hit reasonable PSNR at
    /// `quant_index = 4`.
    fn synthetic_gradient(width: u32, height: u32) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        let cw = w / 2;
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * h];
        let mut cr = vec![0u8; cw * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i + j) * 255 / (w + h)).min(255) as u8;
            }
        }
        for j in 0..h {
            for i in 0..cw {
                cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-64, 64)) as u8;
                cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
            }
        }
        VideoFrame {
            format: PixelFormat::Yuv422P,
            width,
            height,
            pts: Some(0),
            time_base: TimeBase::new(1, 30),
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: cb,
                },
                VideoPlane {
                    stride: cw,
                    data: cr,
                },
            ],
        }
    }

    fn psnr(orig: &[u8], decoded: &[u8]) -> f64 {
        assert_eq!(orig.len(), decoded.len());
        let mut mse = 0.0f64;
        for (a, b) in orig.iter().zip(decoded.iter()) {
            let d = *a as f64 - *b as f64;
            mse += d * d;
        }
        mse /= orig.len() as f64;
        if mse == 0.0 {
            return 120.0;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    #[test]
    fn encoder_decoder_roundtrip_psnr() {
        // 64x48 fits exactly 4x3 MBs of 16x16 — no padding edge cases.
        let width = 64u32;
        let height = 48u32;
        let original = synthetic_gradient(width, height);

        let mut enc_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        enc_params.media_type = MediaType::Video;
        enc_params.width = Some(width);
        enc_params.height = Some(height);
        enc_params.pixel_format = Some(PixelFormat::Yuv422P);

        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        let mut encoder = reg.make_encoder(&enc_params).expect("make_encoder");
        encoder
            .send_frame(&Frame::Video(original.clone()))
            .expect("send_frame");
        let pkt = encoder.receive_packet().expect("receive_packet");

        let dec_params = enc_params.clone();
        let mut decoder = reg.make_decoder(&dec_params).expect("make_decoder");
        decoder.send_packet(&pkt).expect("send_packet");
        let frame = decoder.receive_frame().expect("receive_frame");
        let decoded = match frame {
            Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };

        assert_eq!(decoded.format, PixelFormat::Yuv422P);
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.planes.len(), 3);

        for (i, (o, d)) in original
            .planes
            .iter()
            .zip(decoded.planes.iter())
            .enumerate()
        {
            assert_eq!(o.data.len(), d.data.len(), "plane {i} size mismatch");
            let p = psnr(&o.data, &d.data);
            assert!(p > 30.0, "plane {i} PSNR too low: {p:.2} dB (want > 30)");
            eprintln!("plane {i} PSNR = {p:.2} dB");
        }
    }

    #[test]
    fn decoder_registered() {
        let mut reg = oxideav_codec::CodecRegistry::new();
        register(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(reg.has_encoder(&CodecId::new(CODEC_ID_STR)));
    }
}
