//! AMV video encoder roundtrip tests.
//!
//! Synthesises a 64×64 YUV420P test pattern (solid block + diagonal
//! gradient + pseudo-random noise), feeds it through the `AmvVideoEncoder`,
//! then decodes each emitted packet with the existing `AmvVideoDecoder`.
//! Baseline JPEG at Q50 on smooth-ish synthetic content comfortably hits
//! PSNR > 30 dB; we assert that floor.
//!
//! Also asserts the encoder output starts with `FF D8` and ends with
//! `FF D9`, matching the AMV frame-envelope the decoder expects.

use oxideav_codec::CodecRegistry;
use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Frame, PixelFormat, Rational, TimeBase, VideoFrame,
};

const W: u32 = 64;
const H: u32 = 64;

/// Build a deterministic 64×64 YUV420P frame whose planes mix:
/// - top-left quadrant: solid mid-grey (Y=128, Cb=Cr=128)
/// - top-right quadrant: horizontal luma gradient 0..255
/// - bottom-left quadrant: vertical luma gradient, coloured chroma
/// - bottom-right quadrant: deterministic 8-bit "noise"
fn make_test_frame() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; cw * ch];
    let mut cr = vec![128u8; cw * ch];

    for j in 0..h {
        for i in 0..w {
            let (in_top, in_left) = (j < h / 2, i < w / 2);
            y[j * w + i] = match (in_top, in_left) {
                (true, true) => 128,
                (true, false) => ((i - w / 2) * 255 / (w / 2 - 1)).min(255) as u8,
                (false, true) => (j * 255 / (h - 1)).min(255) as u8,
                (false, false) => {
                    // Deterministic low-amplitude "noise" around mid-grey.
                    // We keep the amplitude small (±32) so the JPEG-Q50
                    // noise floor doesn't swamp the PSNR on the noise
                    // quadrant — baseline JPEG ruthlessly quantises
                    // high-frequency content in DCT space.
                    let mut x =
                        (j as u32).wrapping_mul(2654435761) ^ (i as u32).wrapping_mul(40503);
                    x ^= x >> 13;
                    x = x.wrapping_mul(0x5bd1e995);
                    x ^= x >> 15;
                    (128_i32 + ((x & 0x3F) as i32 - 32)).clamp(0, 255) as u8
                }
            };
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            let in_top = j < ch / 2;
            let in_left = i < cw / 2;
            if !in_top && in_left {
                // Bottom-left: tint toward blue-ish.
                cb[j * cw + i] = 180;
                cr[j * cw + i] = 100;
            } else if !in_top && !in_left {
                // Bottom-right: slight warm tint.
                cb[j * cw + i] = 110;
                cr[j * cw + i] = 160;
            }
        }
    }

    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
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

/// Compute PSNR across all three YUV planes, treating each byte as a sample.
fn psnr_planar(a: &VideoFrame, b: &VideoFrame) -> f64 {
    assert_eq!(a.planes.len(), b.planes.len());
    let mut sse: f64 = 0.0;
    let mut n: usize = 0;
    for (pa, pb) in a.planes.iter().zip(b.planes.iter()) {
        // Walk each plane's *visible* region — the planes we constructed
        // have stride == visible-width, but to be safe scan row-by-row.
        let plane_h = pa.data.len() / pa.stride.max(1);
        let plane_h_b = pb.data.len() / pb.stride.max(1);
        let rows = plane_h.min(plane_h_b);
        let cols = pa.stride.min(pb.stride);
        for r in 0..rows {
            for c in 0..cols {
                let av = pa.data[r * pa.stride + c] as f64;
                let bv = pb.data[r * pb.stride + c] as f64;
                let e = av - bv;
                sse += e * e;
                n += 1;
            }
        }
    }
    if n == 0 || sse <= 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / n as f64;
    let peak = 255.0f64;
    10.0 * (peak * peak / mse).log10()
}

#[test]
fn amv_video_roundtrip_psnr_above_30db() {
    let mut reg = CodecRegistry::new();
    oxideav_amv::register_codecs(&mut reg);

    let id = CodecId::new(oxideav_amv::VIDEO_CODEC_ID_STR);
    assert!(reg.has_decoder(&id), "video decoder not registered");
    assert!(reg.has_encoder(&id), "video encoder not registered");

    let mut params = CodecParameters::video(id.clone());
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = reg.make_encoder(&params).expect("make AMV video encoder");
    let mut dec = reg.make_decoder(&params).expect("make AMV video decoder");

    let input = make_test_frame();
    enc.send_frame(&Frame::Video(input.clone())).unwrap();
    let pkt = enc.receive_packet().expect("encoder produced packet");

    // Frame-envelope sanity: starts with FF D8 (SOI), ends with FF D9 (EOI).
    assert!(pkt.data.len() >= 4, "AMV packet absurdly short");
    assert_eq!(
        &pkt.data[0..2],
        &[0xFF, 0xD8],
        "AMV packet should start with SOI"
    );
    let n = pkt.data.len();
    assert_eq!(
        &pkt.data[n - 2..n],
        &[0xFF, 0xD9],
        "AMV packet should end with EOI"
    );
    // And it should be noticeably shorter than a full-header JPEG: at 64×64
    // the standard headers alone are ~600+ bytes of DQT/DHT/SOF/SOS/APP0;
    // asserting "payload-only" is tricky, but check the stripped form lacks
    // a DQT marker directly after SOI (the full-header form would have one).
    assert_ne!(
        &pkt.data[2..4],
        &[0xFF, 0xDB],
        "AMV packet should not carry DQT segment"
    );

    dec.send_packet(&pkt).unwrap();
    let Frame::Video(decoded) = dec.receive_frame().expect("decoded frame") else {
        panic!("expected video frame");
    };
    assert_eq!(decoded.width, W);
    assert_eq!(decoded.height, H);
    assert_eq!(decoded.format, PixelFormat::Yuv420P);

    let psnr_db = psnr_planar(&input, &decoded);
    eprintln!("AMV video roundtrip PSNR = {:.2} dB", psnr_db);
    assert!(
        psnr_db >= 30.0,
        "PSNR {:.2} dB below expected 30 dB floor",
        psnr_db
    );
}

#[test]
fn amv_video_encoder_rejects_wrong_format() {
    let mut reg = CodecRegistry::new();
    oxideav_amv::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_amv::VIDEO_CODEC_ID_STR);

    let mut params = CodecParameters::video(id);
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    // make_encoder should fail with unsupported format.
    let r = reg.make_encoder(&params);
    assert!(r.is_err(), "encoder must reject non-4:2:0 input");
}
