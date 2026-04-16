//! 8×8 Inverse DCT for Theora (spec §7.9).
//!
//! Theora uses an integer IDCT similar to the one in VP3/VP4 with
//! fixed-point coefficients. This implementation uses the same generic
//! textbook IDCT as oxideav-mjpeg's `jpeg::dct`; matching libtheora's
//! bit-exact integer IDCT is a follow-up optimisation once the rest of the
//! pipeline is validated.

use std::f32::consts::PI;
use std::sync::OnceLock;

/// `t[k][n] = C(k)/2 * cos((2n+1)kπ/16)` with `C(0) = 1/√2`, `C(k>0) = 1`.
fn cos_table() -> &'static [[f32; 8]; 8] {
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 { (0.5f32).sqrt() } else { 1.0 };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    })
}

/// Inverse DCT of an 8×8 block (natural order, already dequantised). In-place.
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Theora's natural-order zig-zag scan mapping (spec Table 7.20).
/// `ZIGZAG[i]` = the natural-order coefficient position at scan index `i`.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_permutation() {
        let mut seen = [false; 64];
        for &p in &ZIGZAG {
            assert!(p < 64);
            assert!(!seen[p], "duplicate {p}");
            seen[p] = true;
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn idct_dc_spreads_uniformly() {
        // A pure DC coefficient should come out as a uniform block.
        let mut block = [0.0f32; 64];
        block[0] = 800.0; // matches mjpeg's DC-of-constant-block = 8 * value
        idct8x8(&mut block);
        for &v in &block {
            assert!((v - 100.0).abs() < 1e-2, "expected 100, got {v}");
        }
    }
}
