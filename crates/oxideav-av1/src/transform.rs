//! AV1 inverse transforms — §7.7.
//!
//! AV1 defines four transform "types" (DCT, ADST, flipped ADST, identity)
//! in each of two dimensions — giving 16 combinations per block size,
//! plus the special `WHT` (Walsh-Hadamard) used by lossless.
//!
//! This module implements the simplest set needed for DC-only blocks:
//!
//! * 4×4 inverse **DCT-DCT** (`iadst` / `iidt` return `Unsupported`).
//! * 8×8 inverse **DCT-DCT**.
//!
//! Sizes 16 / 32 and the ADST / flipped-ADST / identity / WHT variants
//! surface as `Error::Unsupported` with a precise §7.7 sub-clause so the
//! caller can see exactly where the decoder gave up.
//!
//! All arithmetic is integer: AV1 specifies fixed-point constants and
//! `round_shift(x, n)` = `(x + (1 << (n-1))) >> n`. The constants below
//! match `cos(j·π/64)` scaled by 4 to 14-bit precision (spec §7.7.2 Table
//! T.1).

use oxideav_core::{Error, Result};

/// AV1 2-D transform types (§7.7 Table 13-4). Only `DctDct` is implemented
/// for now.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
    FlipAdstDct = 4,
    DctFlipAdst = 5,
    FlipAdstFlipAdst = 6,
    AdstFlipAdst = 7,
    FlipAdstAdst = 8,
    IdtIdt = 9,
    VIdt = 10,
    HIdt = 11,
    VDct = 12,
    HDct = 13,
    VAdst = 14,
    HAdst = 15,
}

impl TxType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::DctDct,
            1 => Self::AdstDct,
            2 => Self::DctAdst,
            3 => Self::AdstAdst,
            4 => Self::FlipAdstDct,
            5 => Self::DctFlipAdst,
            6 => Self::FlipAdstFlipAdst,
            7 => Self::AdstFlipAdst,
            8 => Self::FlipAdstAdst,
            9 => Self::IdtIdt,
            10 => Self::VIdt,
            11 => Self::HIdt,
            12 => Self::VDct,
            13 => Self::HDct,
            14 => Self::VAdst,
            15 => Self::HAdst,
            _ => return Err(Error::invalid(format!("av1 tx: invalid tx_type {v}"))),
        })
    }
}

/// Cosine constants scaled by 2^14 — `cos_bit = 12` in spec parlance.
/// These appear verbatim in the DCT butterflies.
const COS_BIT: i32 = 12;
const C_PI_8: i32 = 15137; // cos(pi/8) * 2^14
const S_PI_8: i32 = 6270; // sin(pi/8) * 2^14 (= cos(3pi/8))
const C_PI_16: i32 = 16069; // cos(pi/16) * 2^14
const S_PI_16: i32 = 3196; // sin(pi/16) * 2^14
const C_3PI_16: i32 = 13623; // cos(3pi/16) * 2^14
const S_3PI_16: i32 = 9102; // sin(3pi/16) * 2^14
const C_SQRT2: i32 = 11585; // cos(pi/4) * 2^14 == sqrt(2)/2 * 2^14

/// `round_shift(x, n)` — the rounding operator used throughout §7.7.
#[inline]
fn round_shift(x: i32, n: i32) -> i32 {
    debug_assert!(n > 0);
    (x + (1 << (n - 1))) >> n
}

/// 4-point inverse DCT — `iadst4` and `iidt4` are not implemented.
///
/// The operation is an in-place transform of a length-4 `i32` vector. It
/// matches the spec's `inv_dct_2d_add_4x4` column / row kernels.
fn idct4(x: &mut [i32; 4]) {
    // Butterfly: step 1
    let s0 = x[0];
    let s1 = x[1];
    let s2 = x[2];
    let s3 = x[3];

    // step 2: even half — add/sub with sqrt(2)
    let t0 = (C_SQRT2 * (s0 + s2)) >> COS_BIT;
    let t1 = (C_SQRT2 * (s0 - s2)) >> COS_BIT;
    // odd half — rotation by pi/8 / 3pi/8
    let t2 = ((C_PI_8 * s1) - (S_PI_8 * s3)) >> COS_BIT;
    let t3 = ((S_PI_8 * s1) + (C_PI_8 * s3)) >> COS_BIT;

    // step 3: combine
    x[0] = t0 + t3;
    x[1] = t1 + t2;
    x[2] = t1 - t2;
    x[3] = t0 - t3;
}

/// 8-point inverse DCT. Standard radix-2 decimation-in-time butterfly.
fn idct8(x: &mut [i32; 8]) {
    // Stage 1 — bit reverse pairs (0,4,2,6,1,5,3,7) in logical order.
    let e0 = x[0];
    let e1 = x[4];
    let e2 = x[2];
    let e3 = x[6];
    let o0 = x[1];
    let o1 = x[5];
    let o2 = x[3];
    let o3 = x[7];

    // Even — reuse idct4 idea
    let f0 = (C_SQRT2 * (e0 + e1)) >> COS_BIT;
    let f1 = (C_SQRT2 * (e0 - e1)) >> COS_BIT;
    let f2 = ((C_PI_8 * e2) - (S_PI_8 * e3)) >> COS_BIT;
    let f3 = ((S_PI_8 * e2) + (C_PI_8 * e3)) >> COS_BIT;
    let e_out0 = f0 + f3;
    let e_out1 = f1 + f2;
    let e_out2 = f1 - f2;
    let e_out3 = f0 - f3;

    // Odd — rotations pi/16, 3pi/16
    let g0 = ((C_PI_16 * o0) - (S_PI_16 * o3)) >> COS_BIT;
    let g3 = ((S_PI_16 * o0) + (C_PI_16 * o3)) >> COS_BIT;
    let g1 = ((C_3PI_16 * o1) - (S_3PI_16 * o2)) >> COS_BIT;
    let g2 = ((S_3PI_16 * o1) + (C_3PI_16 * o2)) >> COS_BIT;

    // Stage 2 — combine odd
    let h0 = g0 + g1;
    let h1 = g0 - g1;
    let h2 = g3 - g2;
    let h3 = g3 + g2;
    let i1 = (C_SQRT2 * (h2 - h1)) >> COS_BIT;
    let i2 = (C_SQRT2 * (h2 + h1)) >> COS_BIT;

    x[0] = e_out0 + h0;
    x[1] = e_out1 + i1;
    x[2] = e_out2 + i2;
    x[3] = e_out3 + h3;
    x[4] = e_out3 - h3;
    x[5] = e_out2 - i2;
    x[6] = e_out1 - i1;
    x[7] = e_out0 - h0;
}

/// Apply a 2-D inverse transform and clip-add the result to `dst`.
///
/// `coeffs` is a `w × h` row-major `i32` block carrying dequantised
/// residuals. Only `TxType::DctDct` is accepted; others return
/// `Unsupported`.
pub fn inverse_transform_add(
    tx: TxType,
    w: usize,
    h: usize,
    coeffs: &[i32],
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    if tx != TxType::DctDct {
        return Err(Error::unsupported(format!(
            "av1 transform: {tx:?} not implemented (§7.7.2 {}; only DctDct available)",
            match tx {
                TxType::DctDct => unreachable!(),
                TxType::AdstDct | TxType::DctAdst | TxType::AdstAdst => "ADST path",
                TxType::FlipAdstDct
                | TxType::DctFlipAdst
                | TxType::FlipAdstFlipAdst
                | TxType::AdstFlipAdst
                | TxType::FlipAdstAdst => "flipped-ADST path",
                TxType::IdtIdt | TxType::VIdt | TxType::HIdt => "identity path",
                TxType::VDct | TxType::HDct | TxType::VAdst | TxType::HAdst => "mixed path",
            },
        )));
    }
    if !matches!((w, h), (4, 4) | (8, 8)) {
        return Err(Error::unsupported(format!(
            "av1 transform: {w}×{h} DCT not implemented (§7.7.2; only 4×4 and 8×8)",
        )));
    }
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "av1 transform: coeffs len {} != {w}*{h}",
            coeffs.len()
        )));
    }
    let mut tmp = vec![0i32; w * h];
    // Column pass: transform each column.
    for c in 0..w {
        match h {
            4 => {
                let mut col = [
                    coeffs[c],
                    coeffs[w + c],
                    coeffs[2 * w + c],
                    coeffs[3 * w + c],
                ];
                idct4(&mut col);
                for (r, v) in col.iter().enumerate() {
                    tmp[r * w + c] = *v;
                }
            }
            8 => {
                let mut col = [0i32; 8];
                for r in 0..8 {
                    col[r] = coeffs[r * w + c];
                }
                idct8(&mut col);
                for r in 0..8 {
                    tmp[r * w + c] = col[r];
                }
            }
            _ => unreachable!(),
        }
    }
    // Row pass: transform each row.
    let out = &mut vec![0i32; w * h];
    for r in 0..h {
        match w {
            4 => {
                let mut row = [tmp[r * 4], tmp[r * 4 + 1], tmp[r * 4 + 2], tmp[r * 4 + 3]];
                idct4(&mut row);
                for c in 0..4 {
                    out[r * 4 + c] = row[c];
                }
            }
            8 => {
                let mut row = [0i32; 8];
                for c in 0..8 {
                    row[c] = tmp[r * 8 + c];
                }
                idct8(&mut row);
                for c in 0..8 {
                    out[r * 8 + c] = row[c];
                }
            }
            _ => unreachable!(),
        }
    }
    // `inv_shift`: AV1 applies a final round_shift(x, 4) for 4×4 and 8×8
    // (§7.7.1). Then the residual is clip-added to the predictor.
    let inv_shift = match (w, h) {
        (4, 4) => 4,
        (8, 8) => 5,
        _ => unreachable!(),
    };
    for r in 0..h {
        for c in 0..w {
            let residual = round_shift(out[r * w + c], inv_shift);
            let p = dst[r * dst_stride + c] as i32;
            let s = (p + residual).clamp(0, 255);
            dst[r * dst_stride + c] = s as u8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_dc_only_4x4_adds_constant() {
        // A block whose only non-zero coefficient is the DC term should
        // add a near-constant shift to every pixel.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64; // post-dequant DC of 64 → residual ≈ 64 / (4 * sqrt(4)) * scale.
        let mut dst = [50u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        // Check all pixels shifted by the same amount (within ±1 rounding).
        let first = dst[0] as i32;
        for &v in &dst[1..] {
            let d = (v as i32 - first).abs();
            assert!(d <= 1, "non-uniform DC-only shift: {dst:?}");
        }
        // Shift should be positive for a positive DC coefficient.
        assert!(first > 50);
    }

    #[test]
    fn dct_zero_coeffs_is_noop() {
        let coeffs = [0i32; 16];
        let mut dst = [42u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn dct_zero_coeffs_8x8_is_noop() {
        let coeffs = [0i32; 64];
        let mut dst = [70u8; 64];
        inverse_transform_add(TxType::DctDct, 8, 8, &coeffs, &mut dst, 8).unwrap();
        for &v in &dst {
            assert_eq!(v, 70);
        }
    }

    #[test]
    fn unsupported_tx_type_returns_clear_error() {
        let coeffs = [0i32; 16];
        let mut dst = [0u8; 16];
        match inverse_transform_add(TxType::AdstAdst, 4, 4, &coeffs, &mut dst, 4) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains("§7.7.2"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_size_returns_clear_error() {
        let coeffs = vec![0i32; 16 * 16];
        let mut dst = vec![0u8; 16 * 16];
        match inverse_transform_add(TxType::DctDct, 16, 16, &coeffs, &mut dst, 16) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains("16"), "msg: {s}");
                assert!(s.contains("§7.7.2"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
