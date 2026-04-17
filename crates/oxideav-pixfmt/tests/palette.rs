//! Palette generation + Pal8 quantise → dequantise tests.

use oxideav_core::{PixelFormat, TimeBase, VideoFrame, VideoPlane};
use oxideav_pixfmt::{
    convert, generate_palette, ConvertOptions, Dither, PaletteGenOptions, PaletteStrategy,
};

fn tb() -> TimeBase {
    TimeBase::new(1, 25)
}

fn deterministic_rgba(w: u32, h: u32, seed: u32) -> VideoFrame {
    // Cheap xorshift for repeatability without a random crate.
    let mut state = seed | 1;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        data.push((state & 0xff) as u8);
        data.push(((state >> 8) & 0xff) as u8);
        data.push(((state >> 16) & 0xff) as u8);
        data.push(((state >> 24) & 0xff) as u8);
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

fn gradient_rgb24(w: u32, h: u32) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as u32 * 255) / (w - 1).max(1)) as u8;
            let g = ((y as u32 * 255) / (h - 1).max(1)) as u8;
            let b = (((x + y) * 255) / ((w + h) - 2).max(1) as u32) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
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

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sq = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sq += d * d;
    }
    if sq == 0.0 {
        return f64::INFINITY;
    }
    let mse = sq / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
fn generate_palette_stays_under_256() {
    let frame = deterministic_rgba(256, 256, 0xDEADBEEF);
    let opts = PaletteGenOptions::default();
    let palette = generate_palette(&[&frame], &opts).unwrap();
    assert!(
        palette.colors.len() <= 256,
        "got {} colours",
        palette.colors.len()
    );
    assert!(!palette.colors.is_empty());
}

#[test]
fn uniform_palette_has_256_entries() {
    let frame = deterministic_rgba(64, 64, 0xB16B00B5);
    let opts = PaletteGenOptions {
        strategy: PaletteStrategy::Uniform,
        max_colors: 255, // u8 max
        transparency: None,
    };
    let palette = generate_palette(&[&frame], &opts).unwrap();
    assert_eq!(palette.colors.len(), 255);
}

#[test]
fn pal8_roundtrip_exceeds_24_db() {
    let src = gradient_rgb24(64, 64);
    let palette = generate_palette(
        &[&src],
        &PaletteGenOptions {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 64,
            transparency: None,
        },
    )
    .unwrap();

    let opts = ConvertOptions {
        dither: Dither::FloydSteinberg,
        palette: Some(palette.clone()),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };

    let pal8 = convert(&src, PixelFormat::Pal8, &opts).unwrap();
    let back = convert(
        &pal8,
        PixelFormat::Rgb24,
        &ConvertOptions {
            dither: Dither::None,
            palette: Some(palette),
            color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
        },
    )
    .unwrap();
    let psnr = psnr_rgb(&src.planes[0].data, &back.planes[0].data);
    println!("pal8 Floyd-Steinberg psnr = {psnr:.2}");
    assert!(psnr > 24.0, "pal8 psnr {psnr} below 24 dB");
}

#[test]
fn pal8_decode_missing_palette_errors() {
    let src = gradient_rgb24(8, 4);
    let palette = generate_palette(
        &[&src],
        &PaletteGenOptions {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 16,
            transparency: None,
        },
    )
    .unwrap();
    let opts = ConvertOptions {
        dither: Dither::None,
        palette: Some(palette),
        color_space: oxideav_pixfmt::ColorSpace::Bt601Limited,
    };
    let pal8 = convert(&src, PixelFormat::Pal8, &opts).unwrap();
    // Now omit the palette — must fail.
    let bare = ConvertOptions::default();
    let res = convert(&pal8, PixelFormat::Rgb24, &bare);
    assert!(res.is_err(), "palette omission must error");
}
