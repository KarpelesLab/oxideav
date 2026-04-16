//! Forward 8x8 DCT (encoder side).
//!
//! Float-domain DCT-II using the textbook factorisation. The integer IDCT in
//! [`crate::dct`] is the bit-exact reference for decode; the encoder DCT does
//! not need to match it sample-for-sample, only to feed quantisation.
//!
//! Output layout matches `crate::dct::idct8x8` natural order (row-major,
//! `block[ri * 8 + ci]`).

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

/// In-place forward DCT of an 8×8 block (natural row-major order).
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    // Row pass: tmp[y][k] = sum_n t[k][n] * block[y][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    // Column pass: out[m][x] = sum_k t[m][k] * tmp[k][x]
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[m][k] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct::idct8x8;

    #[test]
    fn dct_idct_round_trip_random() {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = (i as f32 * 13.0 + 7.0) % 100.0 - 50.0;
        }
        let orig = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!(
                (block[i] - orig[i]).abs() < 1e-3,
                "mismatch at {i}: {} vs {}",
                block[i],
                orig[i]
            );
        }
    }

    #[test]
    fn dc_of_constant_block() {
        let mut block = [50.0f32; 64];
        fdct8x8(&mut block);
        // DC == sum / 8 with our normalisation.
        assert!((block[0] - 400.0).abs() < 1e-3, "DC = {}", block[0]);
        for i in 1..64 {
            assert!(block[i].abs() < 1e-3);
        }
    }
}
