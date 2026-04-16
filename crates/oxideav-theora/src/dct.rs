//! 8×8 Inverse DCT for Theora (spec §7.9).
//!
//! Theora uses an integer IDCT similar to the one in VP3/VP4 with
//! fixed-point coefficients. This module provides both a textbook float
//! IDCT (`idct8x8`) and a spec-compliant 16-bit integer IDCT (`idct2d`).
//! The integer path is what the frame decoder uses, so output is bit-exact
//! with libtheora.

use std::f32::consts::PI;
use std::sync::OnceLock;

/// 16-bit approximations of sines/cosines used by the 1D IDCT (spec Table 7.65).
const C1: i32 = 64277;
const C2: i32 = 60547;
const C3: i32 = 54491;
const C4: i32 = 46341;
const C5: i32 = 36410;
const C6: i32 = 25080;
const C7: i32 = 12785;
// By symmetry, S_k = C_(8-k).
const S3: i32 = C5;
const S6: i32 = C2;
const S7: i32 = C1;

/// Truncate to 16-bit signed (wrapping, two's-complement).
#[inline]
fn t16(v: i32) -> i32 {
    (v as i16) as i32
}

/// Integer 1D inverse DCT (spec §7.9.3.1). Exactly bit-exact.
pub fn idct1d(y: &[i32; 8]) -> [i32; 8] {
    let mut t = [0i32; 8];
    // Steps 1-6.
    t[0] = t16(y[0] + y[4]);
    t[0] = (C4 * t[0]) >> 16;
    t[1] = t16(y[0] - y[4]);
    t[1] = (C4 * t[1]) >> 16;
    // Steps 7-8.
    t[2] = ((C6 * y[2]) >> 16) - ((S6 * y[6]) >> 16);
    t[3] = ((S6 * y[2]) >> 16) + ((C6 * y[6]) >> 16);
    // Steps 9-12.
    t[4] = ((C7 * y[1]) >> 16) - ((S7 * y[7]) >> 16);
    t[5] = ((C3 * y[5]) >> 16) - ((S3 * y[3]) >> 16);
    t[6] = ((S3 * y[5]) >> 16) + ((C3 * y[3]) >> 16);
    t[7] = ((S7 * y[1]) >> 16) + ((C7 * y[7]) >> 16);
    // Steps 13-22.
    let mut r = t[4] + t[5];
    t[5] = t[4] - t[5];
    t[5] = t16(t[5]);
    t[5] = (C4 * t[5]) >> 16;
    t[4] = r;

    r = t[7] + t[6];
    t[6] = t[7] - t[6];
    t[6] = t16(t[6]);
    t[6] = (C4 * t[6]) >> 16;
    t[7] = r;

    // Steps 23-31.
    r = t[0] + t[3];
    t[3] = t[0] - t[3];
    t[0] = r;

    r = t[1] + t[2];
    t[2] = t[1] - t[2];
    t[1] = r;

    r = t[6] + t[5];
    t[5] = t[6] - t[5];
    t[6] = r;

    // Output X[0..7].
    let mut x = [0i32; 8];
    x[0] = t16(t[0] + t[7]);
    x[1] = t16(t[1] + t[6]);
    x[2] = t16(t[2] + t[5]);
    x[3] = t16(t[3] + t[4]);
    x[4] = t16(t[3] - t[4]);
    x[5] = t16(t[2] - t[5]);
    x[6] = t16(t[1] - t[6]);
    x[7] = t16(t[0] - t[7]);
    x
}

/// Full 2D integer inverse DCT (spec §7.9.3.2). Input `dqc` is in natural
/// order (coefficients DQC[0..63] laid out as row-major, bottom-left origin —
/// but since we produce a symmetric residual the row indexing is preserved in
/// the output `res` as-is).
///
/// Produces an 8×8 residual where `res[ri * 8 + ci]` is the i-dct output at
/// row `ri` (bottom-to-top) and column `ci` (left-to-right). Callers that
/// write top-down into a buffer should map `ri` by flipping.
pub fn idct2d(dqc: &[i32; 64]) -> [i32; 64] {
    let mut tmp = [0i32; 64];
    // Row pass.
    for ri in 0..8 {
        let y = [
            dqc[ri * 8],
            dqc[ri * 8 + 1],
            dqc[ri * 8 + 2],
            dqc[ri * 8 + 3],
            dqc[ri * 8 + 4],
            dqc[ri * 8 + 5],
            dqc[ri * 8 + 6],
            dqc[ri * 8 + 7],
        ];
        let x = idct1d(&y);
        for ci in 0..8 {
            tmp[ri * 8 + ci] = x[ci];
        }
    }
    // Column pass + divide by 16 (rounded).
    let mut out = [0i32; 64];
    for ci in 0..8 {
        let y = [
            tmp[ci],
            tmp[8 + ci],
            tmp[16 + ci],
            tmp[24 + ci],
            tmp[32 + ci],
            tmp[40 + ci],
            tmp[48 + ci],
            tmp[56 + ci],
        ];
        let x = idct1d(&y);
        for ri in 0..8 {
            out[ri * 8 + ci] = (x[ri] + 8) >> 4;
        }
    }
    out
}

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

/// Theora's natural-order zig-zag scan mapping (spec Table 7.20, also
/// Figure 2.8). `ZIGZAG[scan] = natural_pos` — i.e. given a scan (zig-zag)
/// index 0..63, returns the natural row-major coefficient position.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse of [`ZIGZAG`]: `INV_ZIGZAG[natural] = scan_pos` — i.e. given a
/// natural row-major coefficient position, returns the zig-zag (scan) index.
pub const INV_ZIGZAG: [usize; 64] = {
    let mut inv = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        inv[ZIGZAG[i]] = i;
        i += 1;
    }
    inv
};

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
