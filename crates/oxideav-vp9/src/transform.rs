//! VP9 inverse transforms — §8.7.1.
//!
//! VP9 defines four transform types per block size:
//!
//! * `DCT_DCT`       — used almost everywhere.
//! * `DCT_ADST`      — row DCT, column ADST.
//! * `ADST_DCT`      — row ADST, column DCT.
//! * `ADST_ADST`     — both ADST.
//!
//! Plus, in the lossless mode (§8.7.1.1), 4×4 blocks use a
//! Walsh-Hadamard Transform (WHT) rather than the DCT.
//!
//! This module implements the minimal set needed for DC-only blocks:
//!
//! * 4×4 inverse **DCT-DCT** (ADST / WHT return `Unsupported`).
//! * 8×8 inverse **DCT-DCT**.
//!
//! Sizes 16 / 32 and the ADST / ADST-ADST / WHT variants surface as
//! `Error::Unsupported` with a precise §8.7 sub-clause so the caller can
//! see exactly where the decoder gave up.
//!
//! All arithmetic is integer: VP9 specifies fixed-point constants and a
//! `round_shift(x, n)` = `(x + (1 << (n-1))) >> n`. Constants below match
//! `cos(j·π/64)` scaled to 14-bit precision (spec §8.7.1.2 `cospi_*`
//! table). They differ slightly from AV1's because VP9 uses a 14-bit
//! constant base while AV1 uses 12-bit butterflies with bigger intermediate
//! rounding.
//!
//! Parallel to the AV1 `transform` module; the two implementations
//! structurally mirror each other so a future contributor can eyeball the
//! delta quickly.

use oxideav_core::{Error, Result};

/// VP9 2-D transform types (§7.4.2 Table 7-3). Values match the spec.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    /// `DCT_DCT` — row & column DCT. Used for inter predicted blocks and
    /// `DC_PRED` / `TM_PRED` intra.
    DctDct = 0,
    /// `ADST_DCT` — row DCT, column ADST. Used with `V_PRED` / `D*_PRED`.
    AdstDct = 1,
    /// `DCT_ADST` — row ADST, column DCT. Used with `H_PRED` / `D*_PRED`.
    DctAdst = 2,
    /// `ADST_ADST` — both ADST. Used with corner-directional intra modes.
    AdstAdst = 3,
    /// `WHT_WHT` — 4×4 Walsh-Hadamard used for lossless (§8.7.1.1).
    /// Note: this isn't a spec-assigned `TX_TYPE` number — it's selected
    /// implicitly via `lossless=1`. We surface it here so the API can
    /// reject it with a precise `Unsupported`.
    WhtWht = 4,
}

impl TxType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::DctDct,
            1 => Self::AdstDct,
            2 => Self::DctAdst,
            3 => Self::AdstAdst,
            4 => Self::WhtWht,
            _ => return Err(Error::invalid(format!("vp9 tx: invalid tx_type {v}"))),
        })
    }
}

/// Cosine constants scaled by 2^14 — `cospi_*_64` from VP9 §8.7.1.2. The
/// `C_SQRT2` constant corresponds to `cospi_16_64` = round(cos(π/4) · 2^14).
const COS_BIT: i32 = 14;
const COSPI_8_64: i32 = 15137; // cos(π/8) · 2^14
const COSPI_24_64: i32 = 6270; // cos(3π/8) · 2^14  == sin(π/8) · 2^14
const COSPI_4_64: i32 = 16069; // cos(π/16) · 2^14
const COSPI_28_64: i32 = 3196; // cos(7π/16) · 2^14 == sin(π/16) · 2^14
const COSPI_12_64: i32 = 13623; // cos(3π/16) · 2^14
const COSPI_20_64: i32 = 9102; // cos(5π/16) · 2^14
const COSPI_16_64: i32 = 11585; // cos(π/4) · 2^14  == √2/2 · 2^14

/// `round_shift(x, n)` — the rounding operator used throughout §8.7.
#[inline]
fn round_shift(x: i32, n: i32) -> i32 {
    debug_assert!(n > 0);
    (x + (1 << (n - 1))) >> n
}

/// 4-point inverse DCT (§8.7.1.3). In-place transform of a length-4 `i32`
/// vector — matches the spec's `inverse_transform_1d` with a 4-point DCT
/// kernel. `iadst4` and `iwht4` are not implemented.
fn idct4(x: &mut [i32; 4]) {
    let s0 = x[0];
    let s1 = x[1];
    let s2 = x[2];
    let s3 = x[3];

    // Even half — add/sub with cospi_16_64 (√2/2 · 2^14).
    let t0 = round_shift(COSPI_16_64 * (s0 + s2), COS_BIT);
    let t1 = round_shift(COSPI_16_64 * (s0 - s2), COS_BIT);
    // Odd half — rotation by π/8 / 3π/8.
    let t2 = round_shift(COSPI_24_64 * s1 - COSPI_8_64 * s3, COS_BIT);
    let t3 = round_shift(COSPI_8_64 * s1 + COSPI_24_64 * s3, COS_BIT);

    x[0] = t0 + t3;
    x[1] = t1 + t2;
    x[2] = t1 - t2;
    x[3] = t0 - t3;
}

/// 8-point inverse DCT (§8.7.1.4). Standard radix-2 decimation-in-time
/// butterfly matching the VP9 libvpx reference. Only used for DCT-DCT
/// 8×8 blocks in this scaffold.
fn idct8(x: &mut [i32; 8]) {
    // Stage 1 — reorder into even / odd halves.
    let e0 = x[0];
    let e1 = x[4];
    let e2 = x[2];
    let e3 = x[6];
    let o0 = x[1];
    let o1 = x[5];
    let o2 = x[3];
    let o3 = x[7];

    // Even — 4-point iDCT on (e0, e1, e2, e3).
    let f0 = round_shift(COSPI_16_64 * (e0 + e1), COS_BIT);
    let f1 = round_shift(COSPI_16_64 * (e0 - e1), COS_BIT);
    let f2 = round_shift(COSPI_24_64 * e2 - COSPI_8_64 * e3, COS_BIT);
    let f3 = round_shift(COSPI_8_64 * e2 + COSPI_24_64 * e3, COS_BIT);
    let e_out0 = f0 + f3;
    let e_out1 = f1 + f2;
    let e_out2 = f1 - f2;
    let e_out3 = f0 - f3;

    // Odd — rotations at π/16 / 3π/16.
    let g0 = round_shift(COSPI_28_64 * o0 - COSPI_4_64 * o3, COS_BIT);
    let g3 = round_shift(COSPI_4_64 * o0 + COSPI_28_64 * o3, COS_BIT);
    let g1 = round_shift(COSPI_12_64 * o1 - COSPI_20_64 * o2, COS_BIT);
    let g2 = round_shift(COSPI_20_64 * o1 + COSPI_12_64 * o2, COS_BIT);

    // Stage 2 — combine odd outputs.
    let h0 = g0 + g1;
    let h1 = g0 - g1;
    let h2 = g3 - g2;
    let h3 = g3 + g2;
    let i1 = round_shift(COSPI_16_64 * (h2 - h1), COS_BIT);
    let i2 = round_shift(COSPI_16_64 * (h2 + h1), COS_BIT);

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
/// residuals. Only `TxType::DctDct` is accepted today; others return
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
            "vp9 transform: {tx:?} not implemented (§8.7.1 {}; only DctDct available)",
            match tx {
                TxType::DctDct => unreachable!(),
                TxType::AdstDct | TxType::DctAdst | TxType::AdstAdst => "ADST path (§8.7.1.7)",
                TxType::WhtWht => "WHT path (§8.7.1.1, lossless only)",
            },
        )));
    }
    if !matches!((w, h), (4, 4) | (8, 8)) {
        return Err(Error::unsupported(format!(
            "vp9 transform: {w}×{h} DCT not implemented (§8.7.1; only 4×4 and 8×8)",
        )));
    }
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "vp9 transform: coeffs len {} != {w}*{h}",
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
    // VP9 applies a final `round_shift(x, 4)` for 4×4 and `round_shift(x, 5)`
    // for 8×8 (§8.7.1). Then the residual is clip-added to the predictor.
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
        coeffs[0] = 64;
        let mut dst = [50u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        let first = dst[0] as i32;
        for &v in &dst[1..] {
            let d = (v as i32 - first).abs();
            assert!(d <= 1, "non-uniform DC-only shift: {dst:?}");
        }
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
                assert!(s.contains("§8.7.1"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_wht_returns_clear_error() {
        let coeffs = [0i32; 16];
        let mut dst = [0u8; 16];
        match inverse_transform_add(TxType::WhtWht, 4, 4, &coeffs, &mut dst, 4) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains("WHT"), "msg should mention WHT: {s}");
                assert!(s.contains("§8.7.1"), "msg: {s}");
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
                assert!(s.contains("§8.7.1"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
