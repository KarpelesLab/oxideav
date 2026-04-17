//! Integration tests for the VP8-lossy WebP encoder path
//! ([`oxideav_webp::encoder_vp8`]).
//!
//! The encoder takes a `Yuv420P` frame, runs it through the pure-Rust
//! `oxideav-vp8` keyframe encoder, and wraps the resulting bitstream in
//! a RIFF/WEBP container with a single `VP8 ` chunk. We verify the
//! full pipeline by:
//!
//! 1. Building a 128×128 YUV420P test pattern.
//! 2. Feeding it through [`encoder_vp8::make_encoder`] → the
//!    registered `Encoder` trait → a `.webp` packet.
//! 3. Checking the RIFF magic + `VP8 ` FourCC at the expected offsets.
//! 4. Decoding the packet bytes via [`oxideav_webp::decode_webp`]
//!    (the read path already handles RIFF/WEBP with a VP8 chunk).
//! 5. Converting the reconstructed RGBA back to YUV420P and asserting
//!    PSNR > 30 dB on the Y plane.

use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_webp::{decode_webp, encoder_vp8, CODEC_ID_VP8};

const W: u32 = 128;
const H: u32 = 128;

/// Build a deterministic YUV420P test pattern:
///   * Y = smooth diagonal luma gradient.
///   * U = horizontal chroma ramp.
///   * V = vertical chroma ramp.
fn build_test_pattern() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![0u8; (W * H) as usize];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            // Luma: smooth diagonal gradient in [32..=223].
            let t = ((row + col) * 255) / (W as usize + H as usize - 2);
            y[row * W as usize + col] = (32 + (t * 191) / 255) as u8;
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (64 + (col * 127) / cw.max(1)) as u8;
            v[row * cw + col] = (64 + (row * 127) / ch.max(1)) as u8;
        }
    }
    (y, u, v)
}

fn make_yuv420_frame(y: &[u8], u: &[u8], v: &[u8]) -> VideoFrame {
    let cw = (W / 2) as usize;
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
        pts: Some(0),
        time_base: TimeBase::new(1, 1000),
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: u.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: v.to_vec(),
            },
        ],
    }
}

fn make_encoder_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

/// BT.601 limited-range RGB → Y conversion (same transform the WebP
/// decoder's YUV→RGB path uses, inverted). Matches the decoder's
/// BT.601 reverse cast closely enough for PSNR purposes.
fn rgb_to_y(r: u8, g: u8, b: u8) -> u8 {
    // Y  = 0.257 R + 0.504 G + 0.098 B + 16 (BT.601 limited range).
    // Use 8-bit fixed-point; rounds to the nearest integer.
    let y = (66 * r as i32 + 129 * g as i32 + 25 * b as i32 + 128) >> 8;
    (y + 16).clamp(0, 255) as u8
}

fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
fn vp8_lossy_webp_roundtrip_psnr_above_30() {
    let (y, u, v) = build_test_pattern();
    let frame = make_yuv420_frame(&y, &u, &v);

    let params = make_encoder_params();
    let mut enc = encoder_vp8::make_encoder(&params).expect("make vp8 encoder");
    enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive_packet");
    let webp_bytes = pkt.data;

    // --- Container sanity: RIFF + WEBP + VP8  marker at the expected offsets.
    assert!(
        webp_bytes.len() >= 20,
        "packet too small: {}",
        webp_bytes.len()
    );
    assert_eq!(&webp_bytes[0..4], b"RIFF", "missing RIFF magic");
    assert_eq!(&webp_bytes[8..12], b"WEBP", "missing WEBP form type");
    assert_eq!(
        &webp_bytes[12..16],
        b"VP8 ",
        "expected VP8 chunk FourCC (with trailing space), got {:?}",
        &webp_bytes[12..16]
    );

    // --- Decode through the full WebP container pipeline.
    let image = decode_webp(&webp_bytes).expect("decode_webp");
    assert_eq!(image.width, W);
    assert_eq!(image.height, H);
    assert_eq!(image.frames.len(), 1);
    let rgba = &image.frames[0].rgba;
    assert_eq!(rgba.len(), (W * H * 4) as usize);

    // Convert decoded RGBA back to Y samples and compute PSNR against the
    // source luma plane. The VP8 encoder at the default qindex (~50) on
    // this smooth test pattern clears ~35 dB comfortably — we assert >30.
    let mut dec_y = vec![0u8; (W * H) as usize];
    for j in 0..H as usize {
        for i in 0..W as usize {
            let p = &rgba[(j * W as usize + i) * 4..(j * W as usize + i) * 4 + 3];
            dec_y[j * W as usize + i] = rgb_to_y(p[0], p[1], p[2]);
        }
    }
    let psnr = psnr_y(&y, &dec_y);
    eprintln!("VP8 lossy WebP Y-plane PSNR: {psnr:.2} dB");
    assert!(
        psnr > 30.0,
        "VP8 lossy WebP PSNR too low: {psnr:.2} dB (expected >30)"
    );
}

#[test]
fn vp8_encoder_rejects_rgba_input() {
    use oxideav_core::Error;

    // An Rgba input should bounce out of the VP8 lossy path with an
    // Error::Unsupported — callers are expected to use the webp_vp8l
    // (lossless) encoder for Rgba frames.
    let mut p = make_encoder_params();
    p.pixel_format = Some(PixelFormat::Rgba);
    let err = encoder_vp8::make_encoder(&p)
        .err()
        .expect("rgba params should be rejected at construction");
    match err {
        Error::Unsupported(msg) => {
            assert!(
                msg.to_lowercase().contains("yuv420p") || msg.to_lowercase().contains("rgba"),
                "unexpected Unsupported message: {msg}"
            );
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}

#[test]
fn vp8_encoder_rejects_rgba_frame_at_send_time() {
    use oxideav_core::Error;

    // If the encoder was built with Yuv420P params but fed an Rgba frame,
    // send_frame must reject it (not silently corrupt).
    let params = make_encoder_params();
    let mut enc = encoder_vp8::make_encoder(&params).expect("make vp8 encoder");
    let rgba = vec![0u8; (W * H * 4) as usize];
    let rgba_frame = VideoFrame {
        format: PixelFormat::Rgba,
        width: W,
        height: H,
        pts: Some(0),
        time_base: TimeBase::new(1, 1000),
        planes: vec![VideoPlane {
            stride: (W as usize) * 4,
            data: rgba,
        }],
    };
    let err = enc
        .send_frame(&Frame::Video(rgba_frame))
        .err()
        .expect("rgba frame should be rejected by a VP8 encoder");
    match err {
        Error::Unsupported(msg) => {
            assert!(
                msg.to_lowercase().contains("yuv420p") || msg.to_lowercase().contains("rgba"),
                "unexpected Unsupported message: {msg}"
            );
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}
