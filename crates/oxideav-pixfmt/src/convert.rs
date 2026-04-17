//! High-level `convert()` entry point.
//!
//! Every supported conversion flows through [`convert`], which dispatches
//! on `(src.format, dst_format)` to the appropriate helper in
//! [`crate::rgb`], [`crate::yuv`], [`crate::gray`], [`crate::palette`],
//! or [`crate::pal8`]. Anything that isn't wired up yet returns
//! `Error::Unsupported`.

use oxideav_core::{Error, PixelFormat, Result, VideoFrame, VideoPlane};

use crate::gray;
use crate::pal8;
use crate::palette::Palette;
use crate::rgb;
use crate::yuv::{self, YuvMatrix};

/// Dither strategy selected when down-quantising to a palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Dither {
    #[default]
    None,
    Bayer8x8,
    FloydSteinberg,
}

/// YUV / RGB matrix selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    #[default]
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
}

/// Options bundle passed to [`convert`].
#[derive(Clone, Debug, Default)]
pub struct ConvertOptions {
    pub dither: Dither,
    pub palette: Option<Palette>,
    pub color_space: ColorSpace,
}

/// Return `Some(src)` when the caller's destination format already
/// matches the source's format — useful to skip a pointless clone in
/// hot paths.
pub fn convert_in_place_if_same(src: &VideoFrame, dst_format: PixelFormat) -> Option<&VideoFrame> {
    if src.format == dst_format {
        Some(src)
    } else {
        None
    }
}

/// Convert `src` to `dst_format`, producing a newly allocated frame.
pub fn convert(
    src: &VideoFrame,
    dst_format: PixelFormat,
    opts: &ConvertOptions,
) -> Result<VideoFrame> {
    if src.format == dst_format {
        return Ok(src.clone());
    }
    let matrix = YuvMatrix::from_color_space(opts.color_space);
    match (src.format, dst_format) {
        // -----------------------------------------------------------------
        // RGB family swizzles (3- and 4-byte packed all-to-all).
        // -----------------------------------------------------------------
        (PixelFormat::Rgb24, PixelFormat::Bgr24) => {
            swizzle3(src, dst_format, rgb::RGB_POS, rgb::BGR_POS)
        }
        (PixelFormat::Bgr24, PixelFormat::Rgb24) => {
            swizzle3(src, dst_format, rgb::BGR_POS, rgb::RGB_POS)
        }

        (PixelFormat::Rgba, PixelFormat::Bgra) => {
            swizzle4(src, dst_format, rgb::RGBA_POS, rgb::BGRA_POS)
        }
        (PixelFormat::Bgra, PixelFormat::Rgba) => {
            swizzle4(src, dst_format, rgb::BGRA_POS, rgb::RGBA_POS)
        }
        (PixelFormat::Rgba, PixelFormat::Argb) => {
            swizzle4(src, dst_format, rgb::RGBA_POS, rgb::ARGB_POS)
        }
        (PixelFormat::Argb, PixelFormat::Rgba) => {
            swizzle4(src, dst_format, rgb::ARGB_POS, rgb::RGBA_POS)
        }
        (PixelFormat::Rgba, PixelFormat::Abgr) => {
            swizzle4(src, dst_format, rgb::RGBA_POS, rgb::ABGR_POS)
        }
        (PixelFormat::Abgr, PixelFormat::Rgba) => {
            swizzle4(src, dst_format, rgb::ABGR_POS, rgb::RGBA_POS)
        }
        (PixelFormat::Bgra, PixelFormat::Argb) => {
            swizzle4(src, dst_format, rgb::BGRA_POS, rgb::ARGB_POS)
        }
        (PixelFormat::Argb, PixelFormat::Bgra) => {
            swizzle4(src, dst_format, rgb::ARGB_POS, rgb::BGRA_POS)
        }
        (PixelFormat::Bgra, PixelFormat::Abgr) => {
            swizzle4(src, dst_format, rgb::BGRA_POS, rgb::ABGR_POS)
        }
        (PixelFormat::Abgr, PixelFormat::Bgra) => {
            swizzle4(src, dst_format, rgb::ABGR_POS, rgb::BGRA_POS)
        }
        (PixelFormat::Argb, PixelFormat::Abgr) => {
            swizzle4(src, dst_format, rgb::ARGB_POS, rgb::ABGR_POS)
        }
        (PixelFormat::Abgr, PixelFormat::Argb) => {
            swizzle4(src, dst_format, rgb::ABGR_POS, rgb::ARGB_POS)
        }

        // 3 ↔ 4 mixes (promote with opaque alpha, or drop alpha).
        (PixelFormat::Rgb24, PixelFormat::Rgba) => {
            promote3_to_4(src, dst_format, rgb::RGB_POS, rgb::RGBA_POS)
        }
        (PixelFormat::Rgb24, PixelFormat::Bgra) => {
            promote3_to_4(src, dst_format, rgb::RGB_POS, rgb::BGRA_POS)
        }
        (PixelFormat::Rgb24, PixelFormat::Argb) => {
            promote3_to_4(src, dst_format, rgb::RGB_POS, rgb::ARGB_POS)
        }
        (PixelFormat::Rgb24, PixelFormat::Abgr) => {
            promote3_to_4(src, dst_format, rgb::RGB_POS, rgb::ABGR_POS)
        }
        (PixelFormat::Bgr24, PixelFormat::Rgba) => {
            promote3_to_4(src, dst_format, rgb::BGR_POS, rgb::RGBA_POS)
        }
        (PixelFormat::Bgr24, PixelFormat::Bgra) => {
            promote3_to_4(src, dst_format, rgb::BGR_POS, rgb::BGRA_POS)
        }
        (PixelFormat::Bgr24, PixelFormat::Argb) => {
            promote3_to_4(src, dst_format, rgb::BGR_POS, rgb::ARGB_POS)
        }
        (PixelFormat::Bgr24, PixelFormat::Abgr) => {
            promote3_to_4(src, dst_format, rgb::BGR_POS, rgb::ABGR_POS)
        }

        (PixelFormat::Rgba, PixelFormat::Rgb24) => {
            demote4_to_3(src, dst_format, rgb::RGBA_POS, rgb::RGB_POS)
        }
        (PixelFormat::Rgba, PixelFormat::Bgr24) => {
            demote4_to_3(src, dst_format, rgb::RGBA_POS, rgb::BGR_POS)
        }
        (PixelFormat::Bgra, PixelFormat::Rgb24) => {
            demote4_to_3(src, dst_format, rgb::BGRA_POS, rgb::RGB_POS)
        }
        (PixelFormat::Bgra, PixelFormat::Bgr24) => {
            demote4_to_3(src, dst_format, rgb::BGRA_POS, rgb::BGR_POS)
        }
        (PixelFormat::Argb, PixelFormat::Rgb24) => {
            demote4_to_3(src, dst_format, rgb::ARGB_POS, rgb::RGB_POS)
        }
        (PixelFormat::Argb, PixelFormat::Bgr24) => {
            demote4_to_3(src, dst_format, rgb::ARGB_POS, rgb::BGR_POS)
        }
        (PixelFormat::Abgr, PixelFormat::Rgb24) => {
            demote4_to_3(src, dst_format, rgb::ABGR_POS, rgb::RGB_POS)
        }
        (PixelFormat::Abgr, PixelFormat::Bgr24) => {
            demote4_to_3(src, dst_format, rgb::ABGR_POS, rgb::BGR_POS)
        }

        // -----------------------------------------------------------------
        // Deeper packed RGB ↔ 8-bit.
        // -----------------------------------------------------------------
        (PixelFormat::Rgb48Le, PixelFormat::Rgb24) => do_rgb48_to_rgb24(src, dst_format),
        (PixelFormat::Rgb24, PixelFormat::Rgb48Le) => do_rgb24_to_rgb48(src, dst_format),
        (PixelFormat::Rgba64Le, PixelFormat::Rgba) => do_rgba64_to_rgba(src, dst_format),
        (PixelFormat::Rgba, PixelFormat::Rgba64Le) => do_rgba_to_rgba64(src, dst_format),

        // -----------------------------------------------------------------
        // Gray ↔ RGB / Gray16Le / Mono.
        // -----------------------------------------------------------------
        (PixelFormat::Gray8, PixelFormat::Rgb24) => gray_to_packed3(src, dst_format),
        (PixelFormat::Gray8, PixelFormat::Rgba) => gray_to_packed4(src, dst_format),
        (PixelFormat::Gray16Le, PixelFormat::Gray8) => do_gray16_to_gray8(src, dst_format),
        (PixelFormat::Gray8, PixelFormat::Gray16Le) => do_gray8_to_gray16(src, dst_format),
        (PixelFormat::MonoBlack, PixelFormat::Gray8) => do_mono_to_gray(src, dst_format, true),
        (PixelFormat::MonoWhite, PixelFormat::Gray8) => do_mono_to_gray(src, dst_format, false),
        (PixelFormat::Gray8, PixelFormat::MonoBlack) => do_gray_to_mono(src, dst_format, true),
        (PixelFormat::Gray8, PixelFormat::MonoWhite) => do_gray_to_mono(src, dst_format, false),

        // -----------------------------------------------------------------
        // YUV ↔ RGB.
        // -----------------------------------------------------------------
        (PixelFormat::Yuv420P, PixelFormat::Rgb24) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 2, 2, false)
        }
        (PixelFormat::Yuv422P, PixelFormat::Rgb24) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 2, 1, false)
        }
        (PixelFormat::Yuv444P, PixelFormat::Rgb24) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 1, 1, false)
        }
        (PixelFormat::Yuv420P, PixelFormat::Rgba) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 2, 2, true)
        }
        (PixelFormat::Yuv422P, PixelFormat::Rgba) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 2, 1, true)
        }
        (PixelFormat::Yuv444P, PixelFormat::Rgba) => {
            do_yuv_to_rgb(src, dst_format, matrix.with_range(true), 1, 1, true)
        }

        (PixelFormat::Rgb24, PixelFormat::Yuv420P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 2, 2, false)
        }
        (PixelFormat::Rgb24, PixelFormat::Yuv422P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 2, 1, false)
        }
        (PixelFormat::Rgb24, PixelFormat::Yuv444P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 1, 1, false)
        }
        (PixelFormat::Rgba, PixelFormat::Yuv420P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 2, 2, true)
        }
        (PixelFormat::Rgba, PixelFormat::Yuv422P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 2, 1, true)
        }
        (PixelFormat::Rgba, PixelFormat::Yuv444P) => {
            do_rgb_to_yuv(src, dst_format, matrix.with_range(true), 1, 1, true)
        }

        // YuvJ* ↔ Yuv* (range rescale).
        (PixelFormat::YuvJ420P, PixelFormat::Yuv420P) => {
            rescale_range(src, dst_format, 2, 2, false)
        }
        (PixelFormat::YuvJ422P, PixelFormat::Yuv422P) => {
            rescale_range(src, dst_format, 2, 1, false)
        }
        (PixelFormat::YuvJ444P, PixelFormat::Yuv444P) => {
            rescale_range(src, dst_format, 1, 1, false)
        }
        (PixelFormat::Yuv420P, PixelFormat::YuvJ420P) => rescale_range(src, dst_format, 2, 2, true),
        (PixelFormat::Yuv422P, PixelFormat::YuvJ422P) => rescale_range(src, dst_format, 2, 1, true),
        (PixelFormat::Yuv444P, PixelFormat::YuvJ444P) => rescale_range(src, dst_format, 1, 1, true),

        // NV12 / NV21 ↔ Yuv420P.
        (PixelFormat::Nv12, PixelFormat::Yuv420P) => nv_to_yuv420p(src, dst_format, true),
        (PixelFormat::Nv21, PixelFormat::Yuv420P) => nv_to_yuv420p(src, dst_format, false),
        (PixelFormat::Yuv420P, PixelFormat::Nv12) => yuv420p_to_nv(src, dst_format, true),
        (PixelFormat::Yuv420P, PixelFormat::Nv21) => yuv420p_to_nv(src, dst_format, false),

        // -----------------------------------------------------------------
        // Palette.
        // -----------------------------------------------------------------
        (PixelFormat::Pal8, PixelFormat::Rgb24) => pal8_to_rgb(src, dst_format, opts, false),
        (PixelFormat::Pal8, PixelFormat::Rgba) => pal8_to_rgb(src, dst_format, opts, true),
        (PixelFormat::Rgb24, PixelFormat::Pal8) => rgb_to_pal8(src, dst_format, opts, false),
        (PixelFormat::Rgba, PixelFormat::Pal8) => rgb_to_pal8(src, dst_format, opts, true),

        (s, d) => Err(Error::unsupported(format!(
            "pixfmt: conversion {s:?} → {d:?} not implemented"
        ))),
    }
}

// -------------------------------------------------------------------------
// Frame helpers.

fn make_frame(src: &VideoFrame, format: PixelFormat, planes: Vec<VideoPlane>) -> VideoFrame {
    VideoFrame {
        format,
        width: src.width,
        height: src.height,
        pts: src.pts,
        time_base: src.time_base,
        planes,
    }
}

fn tight_row(src: &[u8], stride: usize, row: usize, row_bytes: usize) -> &[u8] {
    let off = row * stride;
    &src[off..off + row_bytes]
}

fn gather_tight(src: &[u8], stride: usize, w_bytes: usize, h: usize) -> Vec<u8> {
    if stride == w_bytes {
        return src[..w_bytes * h].to_vec();
    }
    let mut out = Vec::with_capacity(w_bytes * h);
    for row in 0..h {
        out.extend_from_slice(tight_row(src, stride, row, w_bytes));
    }
    out
}

// -------------------------------------------------------------------------
// RGB family.

fn swizzle3(
    src: &VideoFrame,
    dst_format: PixelFormat,
    src_pos: rgb::Rgb3,
    dst_pos: rgb::Rgb3,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::swizzle3(
            sr,
            src_pos,
            &mut out[row * w * 3..row * w * 3 + w * 3],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn swizzle4(
    src: &VideoFrame,
    dst_format: PixelFormat,
    src_pos: rgb::Rgba4,
    dst_pos: rgb::Rgba4,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::swizzle4(
            sr,
            src_pos,
            &mut out[row * w * 4..row * w * 4 + w * 4],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn promote3_to_4(
    src: &VideoFrame,
    dst_format: PixelFormat,
    src_pos: rgb::Rgb3,
    dst_pos: rgb::Rgba4,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::rgb3_to_rgba4(
            sr,
            src_pos,
            &mut out[row * w * 4..row * w * 4 + w * 4],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn demote4_to_3(
    src: &VideoFrame,
    dst_format: PixelFormat,
    src_pos: rgb::Rgba4,
    dst_pos: rgb::Rgb3,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::rgba4_to_rgb3(
            sr,
            src_pos,
            &mut out[row * w * 3..row * w * 3 + w * 3],
            dst_pos,
            w,
        );
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// Deep RGB.

fn do_rgb48_to_rgb24(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 6);
        rgb::rgb48_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn do_rgb24_to_rgb48(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 6];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 3);
        rgb::rgb24_to_rgb48(sr, &mut out[row * w * 6..row * w * 6 + w * 6], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 6,
            data: out,
        }],
    ))
}

fn do_rgba64_to_rgba(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 8);
        rgb::rgba64_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn do_rgba_to_rgba64(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 8];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 4);
        rgb::rgba_to_rgba64(sr, &mut out[row * w * 8..row * w * 8 + w * 8], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 8,
            data: out,
        }],
    ))
}

// -------------------------------------------------------------------------
// Gray / Mono.

fn gray_to_packed3(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 3];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 3,
            data: out,
        }],
    ))
}

fn gray_to_packed4(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 4];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 4,
            data: out,
        }],
    ))
}

fn do_gray16_to_gray8(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w * 2);
        gray::gray16le_to_gray8(sr, &mut out[row * w..row * w + w], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

fn do_gray8_to_gray16(src: &VideoFrame, dst_format: PixelFormat) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h * 2];
    for row in 0..h {
        let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
        gray::gray8_to_gray16le(sr, &mut out[row * w * 2..row * w * 2 + w * 2], w);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 2,
            data: out,
        }],
    ))
}

fn do_mono_to_gray(
    src: &VideoFrame,
    dst_format: PixelFormat,
    black_is_zero: bool,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    // Mono strides are often `(w + 7) / 8`, but honour the provided
    // stride if it differs.
    let src_stride = in_plane.stride;
    let compact = gather_mono_rows(&in_plane.data, src_stride, (w + 7) / 8, h);
    gray::mono_to_gray8(&compact, &mut out, w, h, black_is_zero);
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}

fn do_gray_to_mono(
    src: &VideoFrame,
    dst_format: PixelFormat,
    black_is_zero: bool,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let packed_stride = (w + 7) / 8;
    let src_tight = gather_tight(&in_plane.data, in_plane.stride, w, h);
    let mut out = vec![0u8; packed_stride * h];
    gray::gray8_to_mono(&src_tight, &mut out, w, h, black_is_zero);
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: packed_stride,
            data: out,
        }],
    ))
}

fn gather_mono_rows(src: &[u8], stride: usize, packed: usize, h: usize) -> Vec<u8> {
    if stride == packed {
        return src[..packed * h].to_vec();
    }
    let mut out = Vec::with_capacity(packed * h);
    for row in 0..h {
        out.extend_from_slice(&src[row * stride..row * stride + packed]);
    }
    out
}

// -------------------------------------------------------------------------
// YUV ↔ RGB.

fn do_yuv_to_rgb(
    src: &VideoFrame,
    dst_format: PixelFormat,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: YUV source needs 3 planes"));
    }
    let w = src.width as usize;
    let h = src.height as usize;
    let cw = w / wsub;
    let ch = h / hsub;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);

    let mut rgb_buf = vec![0u8; w * h * 3];
    match (wsub, hsub) {
        (1, 1) => yuv::yuv444_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 1) => yuv::yuv422_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        (2, 2) => yuv::yuv420_to_rgb24(&yp, &up, &vp, &mut rgb_buf, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }

    if !alpha {
        return Ok(make_frame(
            src,
            dst_format,
            vec![VideoPlane {
                stride: w * 3,
                data: rgb_buf,
            }],
        ));
    }
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb_buf[i * 3];
        rgba[i * 4 + 1] = rgb_buf[i * 3 + 1];
        rgba[i * 4 + 2] = rgb_buf[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w * 4,
            data: rgba,
        }],
    ))
}

fn do_rgb_to_yuv(
    src: &VideoFrame,
    dst_format: PixelFormat,
    matrix: YuvMatrix,
    wsub: usize,
    hsub: usize,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let w = src.width as usize;
    let h = src.height as usize;
    if w % wsub != 0 || h % hsub != 0 {
        return Err(Error::invalid(
            "pixfmt: RGB → YUV requires dimensions divisible by subsampling",
        ));
    }
    let cw = w / wsub;
    let ch = h / hsub;

    let in_plane = &src.planes[0];
    // Project to a tight RGB24 buffer.
    let rgb24: Vec<u8> = if alpha_in {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let row_bytes = w * 4;
            let sr = tight_row(&in_plane.data, in_plane.stride, row, row_bytes);
            for i in 0..w {
                out.push(sr[i * 4]);
                out.push(sr[i * 4 + 1]);
                out.push(sr[i * 4 + 2]);
            }
        }
        out
    } else {
        gather_tight(&in_plane.data, in_plane.stride, w * 3, h)
    };

    let mut yp = vec![0u8; w * h];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    match (wsub, hsub) {
        (1, 1) => yuv::rgb24_to_yuv444(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 1) => yuv::rgb24_to_yuv422(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        (2, 2) => yuv::rgb24_to_yuv420(&rgb24, &mut yp, &mut up, &mut vp, w, h, matrix),
        _ => return Err(Error::unsupported("pixfmt: unsupported YUV subsampling")),
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn rescale_range(
    src: &VideoFrame,
    dst_format: PixelFormat,
    wsub: usize,
    hsub: usize,
    to_full: bool,
) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: YuvJ source needs 3 planes"));
    }
    let w = src.width as usize;
    let h = src.height as usize;
    let cw = w / wsub;
    let ch = h / hsub;
    let mut yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let mut up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let mut vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    if to_full {
        yuv::limited_to_full_luma(&mut yp);
        yuv::limited_to_full_chroma(&mut up);
        yuv::limited_to_full_chroma(&mut vp);
    } else {
        yuv::full_to_limited_luma(&mut yp);
        yuv::full_to_limited_chroma(&mut up);
        yuv::full_to_limited_chroma(&mut vp);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn nv_to_yuv420p(src: &VideoFrame, dst_format: PixelFormat, is_nv12: bool) -> Result<VideoFrame> {
    if src.planes.len() < 2 {
        return Err(Error::invalid("pixfmt: NV source needs 2 planes"));
    }
    let w = src.width as usize;
    let h = src.height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let uv = gather_tight(&src.planes[1].data, src.planes[1].stride, cw * 2, ch);
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    if is_nv12 {
        yuv::nv12_uv_split(&uv, &mut up, &mut vp, cw, ch);
    } else {
        yuv::nv21_vu_split(&uv, &mut up, &mut vp, cw, ch);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw,
                data: up,
            },
            VideoPlane {
                stride: cw,
                data: vp,
            },
        ],
    ))
}

fn yuv420p_to_nv(src: &VideoFrame, dst_format: PixelFormat, is_nv12: bool) -> Result<VideoFrame> {
    if src.planes.len() < 3 {
        return Err(Error::invalid("pixfmt: Yuv420P source needs 3 planes"));
    }
    let w = src.width as usize;
    let h = src.height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let yp = gather_tight(&src.planes[0].data, src.planes[0].stride, w, h);
    let up = gather_tight(&src.planes[1].data, src.planes[1].stride, cw, ch);
    let vp = gather_tight(&src.planes[2].data, src.planes[2].stride, cw, ch);
    let mut uv = vec![0u8; cw * ch * 2];
    if is_nv12 {
        yuv::nv12_uv_merge(&up, &vp, &mut uv, cw, ch);
    } else {
        yuv::nv21_vu_merge(&up, &vp, &mut uv, cw, ch);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![
            VideoPlane {
                stride: w,
                data: yp,
            },
            VideoPlane {
                stride: cw * 2,
                data: uv,
            },
        ],
    ))
}

// -------------------------------------------------------------------------
// Palette.

fn pal8_to_rgb(
    src: &VideoFrame,
    dst_format: PixelFormat,
    opts: &ConvertOptions,
    alpha: bool,
) -> Result<VideoFrame> {
    let palette = opts
        .palette
        .as_ref()
        .ok_or_else(|| Error::invalid("pixfmt: Pal8 → RGB requires ConvertOptions.palette"))?;
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    if alpha {
        let mut out = vec![0u8; w * h * 4];
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
            pal8::expand_row_to_rgba(sr, &mut out[row * w * 4..row * w * 4 + w * 4], palette, w);
        }
        Ok(make_frame(
            src,
            dst_format,
            vec![VideoPlane {
                stride: w * 4,
                data: out,
            }],
        ))
    } else {
        let mut out = vec![0u8; w * h * 3];
        for row in 0..h {
            let sr = tight_row(&in_plane.data, in_plane.stride, row, w);
            pal8::expand_row_to_rgb24(sr, &mut out[row * w * 3..row * w * 3 + w * 3], palette, w);
        }
        Ok(make_frame(
            src,
            dst_format,
            vec![VideoPlane {
                stride: w * 3,
                data: out,
            }],
        ))
    }
}

fn rgb_to_pal8(
    src: &VideoFrame,
    dst_format: PixelFormat,
    opts: &ConvertOptions,
    alpha_in: bool,
) -> Result<VideoFrame> {
    let palette = opts
        .palette
        .as_ref()
        .ok_or_else(|| Error::invalid("pixfmt: RGB → Pal8 requires ConvertOptions.palette"))?;
    let w = src.width as usize;
    let h = src.height as usize;
    let in_plane = &src.planes[0];
    let mut out = vec![0u8; w * h];
    if alpha_in {
        let tight = gather_tight(&in_plane.data, in_plane.stride, w * 4, h);
        pal8::quantise_rgba_to_pal8(&tight, &mut out, w, h, palette, opts.dither);
    } else {
        let tight = gather_tight(&in_plane.data, in_plane.stride, w * 3, h);
        pal8::quantise_rgb24_to_pal8(&tight, &mut out, w, h, palette, opts.dither);
    }
    Ok(make_frame(
        src,
        dst_format,
        vec![VideoPlane {
            stride: w,
            data: out,
        }],
    ))
}
