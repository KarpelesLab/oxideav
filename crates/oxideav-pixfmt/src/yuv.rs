//! YUV ↔ RGB conversions (BT.601 and BT.709, limited and full range),
//! planar chroma resampling (4:2:0 ↔ 4:2:2 ↔ 4:4:4), and NV12/NV21
//! ↔ Yuv420P bridging.
//!
//! The scalar floating-point inner loops here are fast enough for the
//! frame sizes the rest of the framework passes around. Callers that
//! need sub-frame latency should hoist a conversion around the top of
//! their decode loop, not inside it.

use crate::convert::ColorSpace;

/// BT.601 / BT.709 weight pair. The integer matrix used by the
/// converters below is built from these f32 values at runtime.
#[derive(Clone, Copy)]
pub struct YuvMatrix {
    pub kr: f32,
    pub kb: f32,
    pub limited: bool,
}

impl YuvMatrix {
    /// BT.601 weights.
    pub const BT601: Self = Self {
        kr: 0.299,
        kb: 0.114,
        limited: true,
    };
    /// BT.709 weights.
    pub const BT709: Self = Self {
        kr: 0.2126,
        kb: 0.0722,
        limited: true,
    };
    pub fn with_range(mut self, limited: bool) -> Self {
        self.limited = limited;
        self
    }

    pub fn from_color_space(cs: ColorSpace) -> Self {
        match cs {
            ColorSpace::Bt601Limited => Self::BT601.with_range(true),
            ColorSpace::Bt601Full => Self::BT601.with_range(false),
            ColorSpace::Bt709Limited => Self::BT709.with_range(true),
            ColorSpace::Bt709Full => Self::BT709.with_range(false),
        }
    }
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

/// Encode a single (R, G, B) pixel into (Y, U, V) per `matrix`.
///
/// For "limited" output, Y is in [16, 235] and Cb/Cr in [16, 240].
/// Full-range outputs use 0..=255 for both.
pub fn rgb_to_yuv(r: u8, g: u8, b: u8, matrix: YuvMatrix) -> (u8, u8, u8) {
    let kr = matrix.kr;
    let kb = matrix.kb;
    let kg = 1.0 - kr - kb;
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;

    // Y is the luma in [0, 255].
    let y_full = kr * rf + kg * gf + kb * bf;
    // Cb / Cr are centered on 128 with a full-range span of ±128.
    let cb_full = (bf - y_full) / (2.0 * (1.0 - kb));
    let cr_full = (rf - y_full) / (2.0 * (1.0 - kr));

    if matrix.limited {
        // Studio / "TV" range.
        // Y: scale 0..255 → 16..235 (scale 219/255).
        // C: scale ±128 → ±112 then centre on 128 (scale 224/255).
        let y = y_full * (219.0 / 255.0) + 16.0;
        let cb = cb_full * (224.0 / 255.0) + 128.0;
        let cr = cr_full * (224.0 / 255.0) + 128.0;
        (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
    } else {
        // Full range — JPEG / "J" YUV.
        let y = y_full;
        let cb = cb_full + 128.0;
        let cr = cr_full + 128.0;
        (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
    }
}

/// Decode a single (Y, U, V) pixel into (R, G, B).
pub fn yuv_to_rgb(y: u8, cb: u8, cr: u8, matrix: YuvMatrix) -> (u8, u8, u8) {
    let kr = matrix.kr;
    let kb = matrix.kb;
    let kg = 1.0 - kr - kb;

    let (yf, cbf, crf) = if matrix.limited {
        let y_lin = (y as f32 - 16.0) * (255.0 / 219.0);
        let cb_lin = (cb as f32 - 128.0) * (255.0 / 224.0);
        let cr_lin = (cr as f32 - 128.0) * (255.0 / 224.0);
        (y_lin, cb_lin, cr_lin)
    } else {
        (y as f32, cb as f32 - 128.0, cr as f32 - 128.0)
    };

    let r = yf + 2.0 * (1.0 - kr) * crf;
    let b = yf + 2.0 * (1.0 - kb) * cbf;
    let g = (yf - kr * r - kb * b) / kg;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

/// Convert a 4:4:4 planar triple (each plane tightly packed at `w×h`)
/// into a packed RGB24 output buffer (also tightly packed).
pub fn yuv444_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    debug_assert!(dst.len() >= w * h * 3);
    for row in 0..h {
        for col in 0..w {
            let (r, g, b) = yuv_to_rgb(
                yp[row * w + col],
                up[row * w + col],
                vp[row * w + col],
                matrix,
            );
            let o = (row * w + col) * 3;
            dst[o] = r;
            dst[o + 1] = g;
            dst[o + 2] = b;
        }
    }
}

/// Convert a 4:2:2 planar triple into packed RGB24. Chroma planes are
/// `(w/2)×h`; each pair of luma columns shares one chroma sample.
pub fn yuv422_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    for row in 0..h {
        for col in 0..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb(
                yp[row * w + col],
                up[row * cw + cc],
                vp[row * cw + cc],
                matrix,
            );
            let o = (row * w + col) * 3;
            dst[o] = r;
            dst[o + 1] = g;
            dst[o + 2] = b;
        }
    }
}

/// Convert a 4:2:0 planar triple into packed RGB24. Chroma planes are
/// `(w/2)×(h/2)`; each 2×2 luma block shares one chroma sample.
pub fn yuv420_to_rgb24(
    yp: &[u8],
    up: &[u8],
    vp: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    for row in 0..h {
        let cr = row / 2;
        for col in 0..w {
            let cc = col / 2;
            let (r, g, b) = yuv_to_rgb(
                yp[row * w + col],
                up[cr * cw + cc],
                vp[cr * cw + cc],
                matrix,
            );
            let o = (row * w + col) * 3;
            dst[o] = r;
            dst[o + 1] = g;
            dst[o + 2] = b;
        }
    }
}

// Encode: pack RGB → YUV 4:4:4 planar.
pub fn rgb24_to_yuv444(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, u, v) = rgb_to_yuv(src[o], src[o + 1], src[o + 2], matrix);
            yp[row * w + col] = y;
            up[row * w + col] = u;
            vp[row * w + col] = v;
        }
    }
}

/// RGB → YUV 4:2:2 (average chroma horizontally over 2 luma columns).
pub fn rgb24_to_yuv422(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, _u, _v) = rgb_to_yuv(src[o], src[o + 1], src[o + 2], matrix);
            yp[row * w + col] = y;
        }
        for cc in 0..cw {
            // Average of columns (cc*2) and (cc*2 + 1).
            let mut cbs = 0i32;
            let mut crs = 0i32;
            for dx in 0..2 {
                let col = cc * 2 + dx;
                let o = (row * w + col) * 3;
                let (_y, u, v) = rgb_to_yuv(src[o], src[o + 1], src[o + 2], matrix);
                cbs += u as i32;
                crs += v as i32;
            }
            up[row * cw + cc] = ((cbs + 1) / 2) as u8;
            vp[row * cw + cc] = ((crs + 1) / 2) as u8;
        }
    }
}

/// RGB → YUV 4:2:0 (average chroma over a 2×2 block).
pub fn rgb24_to_yuv420(
    src: &[u8],
    yp: &mut [u8],
    up: &mut [u8],
    vp: &mut [u8],
    w: usize,
    h: usize,
    matrix: YuvMatrix,
) {
    let cw = w / 2;
    let ch = h / 2;
    // Luma pass.
    for row in 0..h {
        for col in 0..w {
            let o = (row * w + col) * 3;
            let (y, _u, _v) = rgb_to_yuv(src[o], src[o + 1], src[o + 2], matrix);
            yp[row * w + col] = y;
        }
    }
    // Chroma pass.
    for cr in 0..ch {
        for cc in 0..cw {
            let mut cbs = 0i32;
            let mut crs = 0i32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let row = cr * 2 + dy;
                    let col = cc * 2 + dx;
                    let o = (row * w + col) * 3;
                    let (_y, u, v) = rgb_to_yuv(src[o], src[o + 1], src[o + 2], matrix);
                    cbs += u as i32;
                    crs += v as i32;
                }
            }
            up[cr * cw + cc] = ((cbs + 2) / 4) as u8;
            vp[cr * cw + cc] = ((crs + 2) / 4) as u8;
        }
    }
}

// ---------------------------------------------------------------------
// Planar ↔ planar subsample conversions.

/// Downsample a 4:4:4 chroma plane to 4:2:2 (average horizontally).
pub fn chroma_444_to_422(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        for cc in 0..cw {
            let a = src[row * w + cc * 2] as u16;
            let b = src[row * w + cc * 2 + 1] as u16;
            dst[row * cw + cc] = ((a + b + 1) / 2) as u8;
        }
    }
}

/// Upsample a 4:2:2 chroma plane to 4:4:4 (pixel replication).
pub fn chroma_422_to_444(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        for col in 0..w {
            let cc = col / 2;
            dst[row * w + col] = src[row * cw + cc];
        }
    }
}

/// Downsample 4:4:4 → 4:2:0 (average 2×2).
pub fn chroma_444_to_420(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    let ch = h / 2;
    for cr in 0..ch {
        for cc in 0..cw {
            let mut s = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    s += src[(cr * 2 + dy) * w + cc * 2 + dx] as u32;
                }
            }
            dst[cr * cw + cc] = ((s + 2) / 4) as u8;
        }
    }
}

/// Upsample 4:2:0 → 4:4:4 (nearest neighbour).
pub fn chroma_420_to_444(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        let cr = row / 2;
        for col in 0..w {
            let cc = col / 2;
            dst[row * w + col] = src[cr * cw + cc];
        }
    }
}

/// Downsample 4:2:2 → 4:2:0 (average vertically).
pub fn chroma_422_to_420(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    let ch = h / 2;
    for cr in 0..ch {
        for cc in 0..cw {
            let a = src[(cr * 2) * cw + cc] as u16;
            let b = src[(cr * 2 + 1) * cw + cc] as u16;
            dst[cr * cw + cc] = ((a + b + 1) / 2) as u8;
        }
    }
}

/// Upsample 4:2:0 → 4:2:2 (row replication).
pub fn chroma_420_to_422(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let cw = w / 2;
    for row in 0..h {
        let cr = row / 2;
        for cc in 0..cw {
            dst[row * cw + cc] = src[cr * cw + cc];
        }
    }
}

// ---------------------------------------------------------------------
// NV12 / NV21 ↔ Yuv420P.

/// Split an NV12 interleaved UV plane into distinct U and V planes.
pub fn nv12_uv_split(uv: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        up[i] = uv[i * 2];
        vp[i] = uv[i * 2 + 1];
    }
}

/// Same as [`nv12_uv_split`] but with V-first ordering (NV21).
pub fn nv21_vu_split(vu: &[u8], up: &mut [u8], vp: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        vp[i] = vu[i * 2];
        up[i] = vu[i * 2 + 1];
    }
}

pub fn nv12_uv_merge(up: &[u8], vp: &[u8], uv: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        uv[i * 2] = up[i];
        uv[i * 2 + 1] = vp[i];
    }
}

pub fn nv21_vu_merge(up: &[u8], vp: &[u8], vu: &mut [u8], cw: usize, ch: usize) {
    for i in 0..cw * ch {
        vu[i * 2] = vp[i];
        vu[i * 2 + 1] = up[i];
    }
}

// ---------------------------------------------------------------------
// Full/limited range plane conversion for YuvJ* ↔ Yuv*.

/// Scale a plane from studio ("limited") to full range in place.
pub fn limited_to_full_luma(plane: &mut [u8]) {
    for b in plane.iter_mut() {
        let v = *b as f32;
        let f = ((v - 16.0) * (255.0 / 219.0)).clamp(0.0, 255.0);
        *b = f.round() as u8;
    }
}

/// Scale a chroma plane from limited to full (centre stays at 128).
pub fn limited_to_full_chroma(plane: &mut [u8]) {
    for b in plane.iter_mut() {
        let v = *b as f32 - 128.0;
        let f = (v * (255.0 / 224.0) + 128.0).clamp(0.0, 255.0);
        *b = f.round() as u8;
    }
}

pub fn full_to_limited_luma(plane: &mut [u8]) {
    for b in plane.iter_mut() {
        let v = *b as f32;
        let f = (v * (219.0 / 255.0) + 16.0).clamp(0.0, 255.0);
        *b = f.round() as u8;
    }
}

pub fn full_to_limited_chroma(plane: &mut [u8]) {
    for b in plane.iter_mut() {
        let v = *b as f32 - 128.0;
        let f = (v * (224.0 / 255.0) + 128.0).clamp(0.0, 255.0);
        *b = f.round() as u8;
    }
}
