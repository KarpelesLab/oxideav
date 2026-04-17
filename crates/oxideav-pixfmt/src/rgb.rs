//! RGB / BGR family swizzles, plus bit-depth changes between packed
//! 8-bit and 16-bit representations.
//!
//! All functions in this module assume tightly packed input/output
//! (no stride padding). The caller is responsible for stripping stride
//! before handing buffers in and re-adding it afterwards.

/// Component index into a 4-byte packed pixel. Used to describe where
/// R, G, B, and A live for each of the 4-channel formats.
#[derive(Clone, Copy)]
pub struct Rgba4 {
    pub r: usize,
    pub g: usize,
    pub b: usize,
    pub a: usize,
}

/// Byte positions for each 4-channel packed format.
pub const RGBA_POS: Rgba4 = Rgba4 {
    r: 0,
    g: 1,
    b: 2,
    a: 3,
};
pub const BGRA_POS: Rgba4 = Rgba4 {
    r: 2,
    g: 1,
    b: 0,
    a: 3,
};
pub const ARGB_POS: Rgba4 = Rgba4 {
    r: 1,
    g: 2,
    b: 3,
    a: 0,
};
pub const ABGR_POS: Rgba4 = Rgba4 {
    r: 3,
    g: 2,
    b: 1,
    a: 0,
};

/// Component index into a 3-byte packed pixel.
#[derive(Clone, Copy)]
pub struct Rgb3 {
    pub r: usize,
    pub g: usize,
    pub b: usize,
}

pub const RGB_POS: Rgb3 = Rgb3 { r: 0, g: 1, b: 2 };
pub const BGR_POS: Rgb3 = Rgb3 { r: 2, g: 1, b: 0 };

/// Swizzle a packed 3-byte pixel stream between RGB and BGR (or any
/// two Rgb3 layouts).
pub fn swizzle3(src: &[u8], src_pos: Rgb3, dst: &mut [u8], dst_pos: Rgb3, pixels: usize) {
    debug_assert!(src.len() >= pixels * 3 && dst.len() >= pixels * 3);
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 3;
        let r = src[s + src_pos.r];
        let g = src[s + src_pos.g];
        let b = src[s + src_pos.b];
        dst[d + dst_pos.r] = r;
        dst[d + dst_pos.g] = g;
        dst[d + dst_pos.b] = b;
    }
}

/// Swizzle a packed 4-byte pixel stream between any two Rgba4 layouts.
pub fn swizzle4(src: &[u8], src_pos: Rgba4, dst: &mut [u8], dst_pos: Rgba4, pixels: usize) {
    debug_assert!(src.len() >= pixels * 4 && dst.len() >= pixels * 4);
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 4;
        let r = src[s + src_pos.r];
        let g = src[s + src_pos.g];
        let b = src[s + src_pos.b];
        let a = src[s + src_pos.a];
        dst[d + dst_pos.r] = r;
        dst[d + dst_pos.g] = g;
        dst[d + dst_pos.b] = b;
        dst[d + dst_pos.a] = a;
    }
}

/// Convert a 3-byte packed source to a 4-byte packed destination,
/// synthesising an opaque alpha (255).
pub fn rgb3_to_rgba4(src: &[u8], src_pos: Rgb3, dst: &mut [u8], dst_pos: Rgba4, pixels: usize) {
    for i in 0..pixels {
        let s = i * 3;
        let d = i * 4;
        let r = src[s + src_pos.r];
        let g = src[s + src_pos.g];
        let b = src[s + src_pos.b];
        dst[d + dst_pos.r] = r;
        dst[d + dst_pos.g] = g;
        dst[d + dst_pos.b] = b;
        dst[d + dst_pos.a] = 255;
    }
}

/// Drop the alpha channel, converting a 4-byte packed source to a
/// 3-byte packed destination.
pub fn rgba4_to_rgb3(src: &[u8], src_pos: Rgba4, dst: &mut [u8], dst_pos: Rgb3, pixels: usize) {
    for i in 0..pixels {
        let s = i * 4;
        let d = i * 3;
        let r = src[s + src_pos.r];
        let g = src[s + src_pos.g];
        let b = src[s + src_pos.b];
        dst[d + dst_pos.r] = r;
        dst[d + dst_pos.g] = g;
        dst[d + dst_pos.b] = b;
    }
}

/// Rgb48Le → Rgb24 (drop low 8 bits, keep the high byte of each LE word).
pub fn rgb48_to_rgb24(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        // 6 bytes in, 3 bytes out; LE → high byte = index 1, 3, 5.
        dst[i * 3] = src[i * 6 + 1];
        dst[i * 3 + 1] = src[i * 6 + 3];
        dst[i * 3 + 2] = src[i * 6 + 5];
    }
}

/// Rgb24 → Rgb48Le (left-shift 8 and replicate high byte into the low
/// byte for a proper scaling instead of losing bottom range).
pub fn rgb24_to_rgb48(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        for c in 0..3 {
            let b = src[i * 3 + c];
            // Replicate: value * 257 / 256 style — use (b << 8) | b.
            let v: u16 = (b as u16) << 8 | (b as u16);
            let off = i * 6 + c * 2;
            dst[off] = (v & 0xFF) as u8;
            dst[off + 1] = (v >> 8) as u8;
        }
    }
}

/// Rgba64Le → Rgba.
pub fn rgba64_to_rgba(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        // 8 bytes in, 4 bytes out; LE high byte = index 1,3,5,7.
        dst[i * 4] = src[i * 8 + 1];
        dst[i * 4 + 1] = src[i * 8 + 3];
        dst[i * 4 + 2] = src[i * 8 + 5];
        dst[i * 4 + 3] = src[i * 8 + 7];
    }
}

/// Rgba → Rgba64Le.
pub fn rgba_to_rgba64(src: &[u8], dst: &mut [u8], pixels: usize) {
    for i in 0..pixels {
        for c in 0..4 {
            let b = src[i * 4 + c];
            let v: u16 = (b as u16) << 8 | (b as u16);
            let off = i * 8 + c * 2;
            dst[off] = (v & 0xFF) as u8;
            dst[off + 1] = (v >> 8) as u8;
        }
    }
}
