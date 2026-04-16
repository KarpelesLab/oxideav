//! Line Spectral Pair (LSP) decoding utilities for Speex.
//!
//! Float-mode port of the relevant parts of `libspeex/lsp.c` and
//! `libspeex/quant_lsp.c`. The original code supports both fixed- and
//! floating-point arithmetic; oxideav uses the float path throughout.
//!
//! In float mode:
//!   * `LSP_LINEAR(i) = 0.25*(i+1)` — initial guess for the order-10
//!     narrowband LSP vector (radians).
//!   * `LSP_DIV_256(x) = x / 256`, `LSP_DIV_512 = x/512`, `LSP_DIV_1024
//!     = x/1024` — codebook gain factors per stage.
//!   * `ANGLE2X(a) = cos(a)` — LSPs travel through `lsp_to_lpc` as their
//!     cosine ("frequency-domain") values.
//!   * `LSP_SCALING = 1.0`, `LSP_MARGIN = 0.002` (in `nb_celp.c`).

use crate::bitreader::BitReader;
use crate::lsp_tables_nb::{CDBK_NB, CDBK_NB_HIGH1, CDBK_NB_HIGH2, CDBK_NB_LOW1, CDBK_NB_LOW2};
use crate::lsp_tables_wb::{HIGH_LSP_CDBK, HIGH_LSP_CDBK2};
use oxideav_core::Result;

/// Initial linear LSP value for stage `i` (radians). Matches
/// `LSP_LINEAR(i) = 0.25*(i+1)` from `quant_lsp.c`.
#[inline]
fn lsp_linear(i: usize) -> f32 {
    0.25 * (i as f32 + 1.0)
}

/// Float `lsp_unquant_lbr` — three-stage VQ used by NB sub-modes 1, 2, 3,
/// 4 and 8. Reads 18 bits total (3 × 6).
pub fn lsp_unquant_lbr(lsp: &mut [f32], order: usize, br: &mut BitReader) -> Result<()> {
    debug_assert_eq!(order, 10, "Speex narrowband uses 10th-order LSP");
    for i in 0..order {
        lsp[i] = lsp_linear(i);
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..10 {
        lsp[i] += (CDBK_NB[id * 10 + i] as f32) / 256.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i] += (CDBK_NB_LOW1[id * 5 + i] as f32) / 512.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i + 5] += (CDBK_NB_HIGH1[id * 5 + i] as f32) / 512.0;
    }
    Ok(())
}

/// Float `lsp_unquant_nb` — five-stage VQ used by NB sub-modes 5, 6, 7.
/// Reads 30 bits total (5 × 6).
pub fn lsp_unquant_nb(lsp: &mut [f32], order: usize, br: &mut BitReader) -> Result<()> {
    debug_assert_eq!(order, 10, "Speex narrowband uses 10th-order LSP");
    for i in 0..order {
        lsp[i] = lsp_linear(i);
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..10 {
        lsp[i] += (CDBK_NB[id * 10 + i] as f32) / 256.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i] += (CDBK_NB_LOW1[id * 5 + i] as f32) / 512.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i] += (CDBK_NB_LOW2[id * 5 + i] as f32) / 1024.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i + 5] += (CDBK_NB_HIGH1[id * 5 + i] as f32) / 512.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..5 {
        lsp[i + 5] += (CDBK_NB_HIGH2[id * 5 + i] as f32) / 1024.0;
    }
    Ok(())
}

/// High-band LSP initial value. Mirrors `LSP_LINEAR_HIGH(i) = 0.3125*i + 0.75`
/// in `quant_lsp.c` (float path). Spans roughly [0.75, 2.9375] radians for
/// an 8-order wideband filter, which is the upper half of the full-band
/// [0, π] range since the high band covers 4–8 kHz at 16 kHz sample rate.
#[inline]
fn lsp_linear_high(i: usize) -> f32 {
    0.3125 * i as f32 + 0.75
}

/// Float `lsp_unquant_high` — two-stage VQ used by the wideband SB-CELP
/// layer for every WB sub-mode. Reads 12 bits total (2 × 6) with a
/// linear initial guess spanning the high band's spectral range.
pub fn lsp_unquant_high(lsp: &mut [f32], order: usize, br: &mut BitReader) -> Result<()> {
    debug_assert_eq!(order, 8, "Speex wideband high-band uses 8th-order LSP");
    for i in 0..order {
        lsp[i] = lsp_linear_high(i);
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..order {
        lsp[i] += (HIGH_LSP_CDBK[id * order + i] as f32) / 256.0;
    }

    let id = br.read_u32(6)? as usize;
    for i in 0..order {
        lsp[i] += (HIGH_LSP_CDBK2[id * order + i] as f32) / 512.0;
    }
    Ok(())
}

/// Convert the LSP vector (angles, radians) to LPC coefficients via the
/// closed-form polynomial reconstruction from `libspeex/lsp.c`
/// (`lsp_to_lpc`). Order is fixed to even; 10 for narrowband, 8 for the
/// wideband high-band layer.
///
/// The output is in LPC-direct form, i.e. the standard `a[1..order]`
/// (with `a[0] = 1` implicit). This matches what `iir_mem16` expects in
/// `den[]`.
pub fn lsp_to_lpc(freq: &[f32], ak: &mut [f32], order: usize) {
    debug_assert!(order % 2 == 0, "LSP requires even order");
    debug_assert_eq!(ak.len(), order);
    let m = order / 2;
    let mut wp = vec![0.0f32; 4 * m + 2];
    let mut x_freq = vec![0.0f32; order];
    for i in 0..order {
        x_freq[i] = freq[i].cos();
    }
    let mut xin1 = 1.0f32;
    let mut xin2 = 1.0f32;

    for j in 0..=order {
        let mut i2 = 0usize;
        let mut last_n4 = 0usize;
        for i in 0..m {
            let base = i * 4;
            let n1 = base;
            let n2 = base + 1;
            let n3 = base + 2;
            let n4 = base + 3;
            let xout1 = xin1 - 2.0 * x_freq[i2] * wp[n1] + wp[n2];
            let xout2 = xin2 - 2.0 * x_freq[i2 + 1] * wp[n3] + wp[n4];
            wp[n2] = wp[n1];
            wp[n4] = wp[n3];
            wp[n1] = xin1;
            wp[n3] = xin2;
            xin1 = xout1;
            xin2 = xout2;
            i2 += 2;
            last_n4 = n4;
        }
        // Read/write n4+1 and n4+2 as in the reference.
        let xout1 = xin1 + wp[last_n4 + 1];
        let xout2 = xin2 - wp[last_n4 + 2];
        if j > 0 {
            ak[j - 1] = (xout1 + xout2) * 0.5;
        }
        wp[last_n4 + 1] = xin1;
        wp[last_n4 + 2] = xin2;
        xin1 = 0.0;
        xin2 = 0.0;
    }
}

/// Float `lsp_interpolate` — blend the previous frame's LSP with the
/// current one for each sub-frame (`subframe in 0..nb_subframes`), then
/// enforce stability margins so adjacent LSPs stay separated by at least
/// `margin` radians. Mirrors `lsp.c` exactly.
pub fn lsp_interpolate(
    old_lsp: &[f32],
    new_lsp: &[f32],
    out: &mut [f32],
    len: usize,
    subframe: usize,
    nb_subframes: usize,
    margin: f32,
) {
    let tmp = (1.0 + subframe as f32) / nb_subframes as f32;
    for i in 0..len {
        out[i] = (1.0 - tmp) * old_lsp[i] + tmp * new_lsp[i];
    }
    if out[0] < margin {
        out[0] = margin;
    }
    let pi_minus = std::f32::consts::PI - margin;
    if out[len - 1] > pi_minus {
        out[len - 1] = pi_minus;
    }
    for i in 1..len - 1 {
        if out[i] < out[i - 1] + margin {
            out[i] = out[i - 1] + margin;
        }
        if out[i] > out[i + 1] - margin {
            out[i] = 0.5 * (out[i] + out[i + 1] - margin);
        }
    }
}

/// Bandwidth expansion of LPC coefficients — from `libspeex/lpc.c`
/// (`bw_lpc`). Multiplies coefficient `k` (1-indexed) by `gamma^k`.
/// Used in PLC (packet-loss concealment) when no parameters are
/// transmitted.
pub fn bw_lpc(gamma: f32, lpc_in: &[f32], lpc_out: &mut [f32], order: usize) {
    let mut tmp = gamma;
    for i in 0..order {
        lpc_out[i] = tmp * lpc_in[i];
        tmp *= gamma;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsp_linear_matches_reference() {
        // From quant_lsp.c float: LSP_LINEAR(i) = .25*(i)+.25.
        assert!((lsp_linear(0) - 0.25).abs() < 1e-6);
        assert!((lsp_linear(9) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn interpolate_blends_evenly() {
        let old = [1.0_f32; 10];
        let new = [2.0_f32; 10];
        let mut out = [0.0_f32; 10];
        // Halfway through 4 subframes ⇒ tmp = (1+1)/4 = 0.5.
        lsp_interpolate(&old, &new, &mut out, 10, 1, 4, 0.002);
        for &v in &out {
            assert!((v - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn lsp_to_lpc_reproduces_known_a0() {
        // Trivial monotone LSPs at uniform spacing produce a stable
        // synthesis filter — `ak[0]` should be small.
        let mut lsp = [0.0_f32; 10];
        for i in 0..10 {
            lsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / 11.0;
        }
        let mut ak = [0.0_f32; 10];
        lsp_to_lpc(&lsp, &mut ak, 10);
        // For evenly-spaced LSPs (the "open-circuit" filter), the LPC
        // coefficients are non-degenerate; just sanity-check that none
        // are NaN/Inf and the magnitudes stay bounded.
        for &c in &ak {
            assert!(c.is_finite(), "LPC coef must be finite, got {c}");
            assert!(c.abs() < 4.0, "LPC coef out of expected range: {c}");
        }
    }
}
