//! Block-level decoding scaffold.
//!
//! For this session only the AC/DC prediction state and the zig-zag descan
//! helpers are implemented — the full Annex B texture-coefficient VLC walk
//! and the I-VOP MB decoder are the follow-up.

use crate::headers::vol::ZIGZAG;

/// Inverse zig-zag an 8×8 block. `raw_coeffs` is the linear stream as
/// decoded from the bitstream; the output is natural-order 8×8.
pub fn inverse_zigzag(raw_coeffs: &[i32; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        out[ZIGZAG[i]] = raw_coeffs[i];
    }
    out
}

/// Textbook 8×8 IDCT (float), used by the I-VOP path. Duplicated from
/// oxideav-mpeg1video to keep the crate stand-alone.
pub fn idct8x8(block: &mut [f32; 64]) {
    use std::f32::consts::PI;
    use std::sync::OnceLock;

    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    let cos = T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 {
                (1.0_f32 / 2.0_f32).sqrt()
            } else {
                1.0
            };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    });

    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += cos[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += cos[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Clamp a signed sample value to an 8-bit pixel.
pub fn clip_to_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_roundtrip() {
        // A flat DC-only block should come out as a roughly uniform plane.
        let mut b = [0.0f32; 64];
        b[0] = 8.0 * 128.0; // DC value representing 128 (after IDCT scale).
        idct8x8(&mut b);
        for v in &b {
            assert!((v - 128.0).abs() < 1.0, "got {v}, want ~128");
        }
    }
}
