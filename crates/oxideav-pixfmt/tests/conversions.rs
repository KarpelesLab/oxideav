//! Exact-roundtrip tests for the RGB-family swizzles and bit-depth
//! conversions. Every pair tested here must be lossless.

use oxideav_core::{PixelFormat, TimeBase, VideoFrame, VideoPlane};
use oxideav_pixfmt::{convert, ConvertOptions};

fn tb() -> TimeBase {
    TimeBase::new(1, 25)
}

fn synth_rgba(w: u32, h: u32) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
            data.push(((x + y) * 5) as u8);
        }
    }
    VideoFrame {
        format: PixelFormat::Rgba,
        width: w,
        height: h,
        pts: None,
        time_base: tb(),
        planes: vec![VideoPlane {
            stride: (w * 4) as usize,
            data,
        }],
    }
}

fn synth_rgb24(w: u32, h: u32) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 13 + y * 7) as u8);
            data.push((x * 3 + y * 31) as u8);
            data.push((x * 29 + y * 17) as u8);
        }
    }
    VideoFrame {
        format: PixelFormat::Rgb24,
        width: w,
        height: h,
        pts: None,
        time_base: tb(),
        planes: vec![VideoPlane {
            stride: (w * 3) as usize,
            data,
        }],
    }
}

#[test]
fn rgb_family_4byte_roundtrips() {
    let opts = ConvertOptions::default();
    let src = synth_rgba(32, 16);
    for fmt in [PixelFormat::Bgra, PixelFormat::Argb, PixelFormat::Abgr] {
        let stage = convert(&src, fmt, &opts).expect("swizzle");
        let back = convert(&stage, PixelFormat::Rgba, &opts).expect("swizzle back");
        assert_eq!(back.planes[0].data, src.planes[0].data, "roundtrip {fmt:?}");
    }
}

#[test]
fn rgb_family_3byte_roundtrips() {
    let opts = ConvertOptions::default();
    let src = synth_rgb24(32, 16);
    let bgr = convert(&src, PixelFormat::Bgr24, &opts).unwrap();
    let back = convert(&bgr, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb24_to_rgba_and_back_preserves_colour() {
    let opts = ConvertOptions::default();
    let src = synth_rgb24(16, 8);
    let rgba = convert(&src, PixelFormat::Rgba, &opts).unwrap();
    let back = convert(&rgba, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgb48_rgb24_roundtrip() {
    let opts = ConvertOptions::default();
    let src = synth_rgb24(16, 8);
    let deep = convert(&src, PixelFormat::Rgb48Le, &opts).unwrap();
    let back = convert(&deep, PixelFormat::Rgb24, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn rgba64_rgba_roundtrip() {
    let opts = ConvertOptions::default();
    let src = synth_rgba(16, 8);
    let deep = convert(&src, PixelFormat::Rgba64Le, &opts).unwrap();
    let back = convert(&deep, PixelFormat::Rgba, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn gray8_gray16_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) {
        data.push((i * 5) as u8);
    }
    let src = VideoFrame {
        format: PixelFormat::Gray8,
        width: w,
        height: h,
        pts: None,
        time_base: tb(),
        planes: vec![VideoPlane {
            stride: w as usize,
            data,
        }],
    };
    let deep = convert(&src, PixelFormat::Gray16Le, &opts).unwrap();
    let back = convert(&deep, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, src.planes[0].data);
}

#[test]
fn mono_black_gray8_roundtrip() {
    let opts = ConvertOptions::default();
    let w = 16u32;
    let h = 8u32;
    let mut data = vec![0u8; (w * h) as usize];
    for i in 0..data.len() {
        data[i] = if i % 2 == 0 { 255 } else { 0 };
    }
    let src = VideoFrame {
        format: PixelFormat::Gray8,
        width: w,
        height: h,
        pts: None,
        time_base: tb(),
        planes: vec![VideoPlane {
            stride: w as usize,
            data: data.clone(),
        }],
    };
    let mono = convert(&src, PixelFormat::MonoBlack, &opts).unwrap();
    let back = convert(&mono, PixelFormat::Gray8, &opts).unwrap();
    assert_eq!(back.planes[0].data, data);
}

#[test]
fn swizzle_all_four_byte_pairs() {
    // Every 4-byte ↔ 4-byte pair must roundtrip exactly.
    let opts = ConvertOptions::default();
    let src = synth_rgba(32, 16);
    let formats = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    for a in formats {
        for b in formats {
            if a == b {
                continue;
            }
            let frame_a = convert(&src, a, &opts).unwrap();
            let frame_b = convert(&frame_a, b, &opts).unwrap();
            let frame_back = convert(&frame_b, a, &opts).unwrap();
            assert_eq!(
                frame_a.planes[0].data, frame_back.planes[0].data,
                "a=Rgba stage={a:?} then {b:?}"
            );
        }
    }
}
