//! 8×8 forward and inverse Discrete Cosine Transform.
//!
//! Straight textbook implementation over f32. Not the fastest algorithm out
//! there — AAN or the Loeffler factorisation would be faster — but it is
//! short, correct, and pure-Rust. Suitable for small-scale MJPEG workloads;
//! optimisation can come later.

use std::f32::consts::PI;
use std::sync::OnceLock;

/// `t[k][n] = C(k)/2 * cos((2n+1)kπ/16)`, stored row-major. With
/// `C(0) = 1/√2` and `C(k>0) = 1`. Applying this matrix along each axis
/// implements the standard JPEG FDCT/IDCT normalisation (factor of 1/4
/// after two 1-D passes, with the `C(u)C(v)` factor absorbed).
fn cos_table() -> &'static [[f32; 8]; 8] {
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
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
    })
}

/// Inverse DCT of an 8×8 block (natural order, already dequantised). In-place.
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];

    // Row-wise 1-D inverse: for each row y, tmp[y][n] = Σk t[k][n] * block[y][k]
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    // Column-wise 1-D inverse.
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

/// Forward DCT of an 8×8 block in natural order, in-place. Caller should
/// subtract 128 from sample values before calling so input is centred on 0.
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];

    // Row-wise 1-D forward: for each row y, tmp[y][k] = Σn t[k][n] * block[y][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    // Column-wise.
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_of_dct_is_identity() {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = ((i * 7) % 255) as f32 - 128.0;
        }
        let original = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < 1e-2,
                "mismatch at {i}: got {}, want {}",
                block[i],
                original[i]
            );
        }
    }

    #[test]
    fn dc_of_constant_block() {
        // All samples = 100. With our normalisation, DC = 100 * 8.
        let mut block = [100.0f32; 64];
        fdct8x8(&mut block);
        assert!((block[0] - 800.0).abs() < 1e-2, "DC = {}", block[0]);
        for i in 1..64 {
            assert!(block[i].abs() < 1e-2, "AC[{i}] = {}", block[i]);
        }
    }
}
