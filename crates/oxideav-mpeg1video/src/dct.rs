//! 8×8 inverse DCT wrapper. Uses the same textbook f32 IDCT as oxideav-mjpeg's
//! JPEG decoder (duplicated rather than re-exported to avoid a cross-crate
//! dependency — the implementation is 30 lines and perfectly self-contained).

use std::f32::consts::PI;
use std::sync::OnceLock;

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

/// Inverse DCT of an 8×8 block. Input is natural order (dequantised). In-place.
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
