//! VP8 I-frame encoder → decoder round-trip tests.
//!
//! These cover the v1 encoder (keyframes only, DC_PRED, fixed qindex,
//! loop filter disabled):
//!   * Solid-colour 128×128 image round-trips with very high PSNR.
//!   * A mid-complexity YUV test pattern round-trips above the 25 dB
//!     PSNR bar the task brief specifies.
//!   * The compressed output starts with the correct VP8 keyframe tag
//!     (show_frame set, frame_type=0) followed by the 3-byte start code
//!     `9d 01 2a`.

use oxideav_core::{PixelFormat, TimeBase, VideoFrame, VideoPlane};
use oxideav_vp8::encoder::encode_keyframe;
use oxideav_vp8::{decode_frame, parse_header, FrameType};

const W: u32 = 128;
const H: u32 = 128;
const QINDEX: u8 = 50;

fn make_frame(y: &[u8], u: &[u8], v: &[u8]) -> VideoFrame {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    assert_eq!(y.len(), (W * H) as usize);
    assert_eq!(u.len(), cw * ch);
    assert_eq!(v.len(), cw * ch);
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
        pts: None,
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
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
fn keyframe_starts_with_correct_start_code() {
    let y = vec![200u8; (W * H) as usize];
    let u = vec![100u8; ((W / 2) * (H / 2)) as usize];
    let v = vec![150u8; ((W / 2) * (H / 2)) as usize];
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    assert!(encoded.len() >= 10, "encoded stream too short");
    // Parse the frame tag — frame_type must be Key, show_frame=true.
    let parsed = parse_header(&encoded).expect("parse");
    assert!(matches!(parsed.tag.frame_type, FrameType::Key));
    assert!(parsed.tag.show_frame);
    // Sync code at offset 3..6.
    assert_eq!(
        &encoded[3..6],
        &[0x9d, 0x01, 0x2a],
        "sync code mismatch: got {:02x?}",
        &encoded[3..6]
    );
    // Width/height little-endian at 6..10.
    let w = u16::from_le_bytes([encoded[6], encoded[7]]) & 0x3fff;
    let h = u16::from_le_bytes([encoded[8], encoded[9]]) & 0x3fff;
    assert_eq!(w as u32, W);
    assert_eq!(h as u32, H);
}

#[test]
fn roundtrip_solid_color_high_psnr() {
    // Solid grey frame — should come back essentially losslessly since
    // only DC energy is present and DC quant is coarse but stable.
    let y = vec![128u8; (W * H) as usize];
    let u = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    let decoded = decode_frame(&encoded).expect("decode");
    assert_eq!(decoded.width, W);
    assert_eq!(decoded.height, H);
    let py = psnr(&decoded.planes[0].data, &y);
    let pu = psnr(&decoded.planes[1].data, &u);
    let pv = psnr(&decoded.planes[2].data, &v);
    eprintln!("solid-grey PSNR Y={py:.2} U={pu:.2} V={pv:.2}");
    assert!(py >= 40.0, "solid-grey Y PSNR too low: {py:.2} dB");
    assert!(pu >= 40.0, "solid-grey U PSNR too low: {pu:.2} dB");
    assert!(pv >= 40.0, "solid-grey V PSNR too low: {pv:.2} dB");
}

#[test]
fn roundtrip_yuv_test_pattern_psnr_above_25() {
    // Mid-complexity YUV pattern: smooth diagonal luma gradient plus
    // horizontal / vertical chroma gradients.
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![0u8; (W * H) as usize];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            let base = ((row + col) as i32 * 255 / (W + H - 2) as i32) as u8;
            y[row * W as usize + col] = base;
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (64 + ((col * 255) / cw) as u8 / 2) as u8;
            v[row * cw + col] = (192 - ((row * 255) / ch) as u8 / 2) as u8;
        }
    }
    let frame = make_frame(&y, &u, &v);
    let encoded = encode_keyframe(W, H, QINDEX, &frame).expect("encode");
    let decoded = decode_frame(&encoded).expect("decode");
    let py = psnr(&decoded.planes[0].data, &y);
    let pu = psnr(&decoded.planes[1].data, &u);
    let pv = psnr(&decoded.planes[2].data, &v);
    eprintln!(
        "yuv-pattern PSNR Y={py:.2} U={pu:.2} V={pv:.2} (encoded size {} bytes)",
        encoded.len()
    );
    assert!(py > 25.0, "Y PSNR too low: {py:.2} dB");
    assert!(pu > 25.0, "U PSNR too low: {pu:.2} dB");
    assert!(pv > 25.0, "V PSNR too low: {pv:.2} dB");
}
