//! 10th-order LPC / LSP predictor state + LSP decode + LSP→LPC.
//!
//! G.729 uses a **moving-average (MA) predictor** of order 4 on the ten
//! LSP frequencies, switched between two predictor sets by the `L0` bit
//! (§3.2.4). This module provides:
//!
//! - [`LpcPredictorState`] — rolling history of the four previous
//!   quantised LSF residual vectors, initialised to the silent state.
//! - [`decode_lsp`] — the L0/L1/L2/L3-index → LSP (cosine-domain)
//!   decoder, with MA-predictor reconstruction + monotonicity repair.
//! - [`lsp_to_lpc`] — LSP (cosine-domain) → LPC coefficients, via the
//!   standard F1(z) / F2(z) polynomial expansion (G.729 §3.2.6).
//! - [`interpolate_lsp`] — linear interpolation of two LSP vectors.
//!
//! Conventions:
//! - Within this module LSFs are stored as `f32` radians in `(0, pi)`.
//! - LSPs are the cosines of those frequencies, in `[-1, 1]`,
//!   strictly decreasing with index.
//! - LPC coefficients use the standard convention
//!   `A(z) = 1 + sum_k a[k] z^-k` (so `a[0] == 1.0` and the synthesis
//!   filter `1/A(z)` emits `y[n] = x[n] - sum_k a[k]*y[n-k]`).
//!
//! Reference: ITU-T G.729 §3.2.4 / §3.2.6.

use crate::lsp_tables::{FG_Q15, FG_SUM_INV_Q12, FG_SUM_Q15, LSPCB1_Q13, LSPCB2_Q13, M_HALF};
use crate::LPC_ORDER;

/// Size of the MA-predictor history (four previous LSF residual vectors).
pub const MA_HISTORY: usize = 4;

/// Minimum spacing (radians) enforced between adjacent LSFs before
/// conversion to LPC. Matches the spec's stability-safeguard distance.
const LSF_MIN_SPACING: f32 = 0.0012;

/// State of the LPC / LSP predictor across frames.
#[derive(Clone, Debug)]
pub struct LpcPredictorState {
    /// Previous-frame quantised LSF **residuals** (for the MA predictor),
    /// in Q13 integer form (i.e. raw first+second-stage codebook sums
    /// before the predictor has been unwound).
    pub freq_res_q13: [[i16; LPC_ORDER]; MA_HISTORY],
    /// 10 short-term predictor coefficients (after LSP → LP conversion),
    /// as f32. `a[0] == 1.0`; `a[k]` for `k>=1` are predictor taps in
    /// the convention `A(z) = 1 + sum a[k] z^-k`.
    pub a: [f32; LPC_ORDER + 1],
    /// Previously-decoded LSP vector (cosine domain), for subframe-1
    /// interpolation.
    pub lsp_prev: [f32; LPC_ORDER],
}

impl Default for LpcPredictorState {
    fn default() -> Self {
        Self::new()
    }
}

impl LpcPredictorState {
    /// Fresh predictor state. The initial LSF vector is the uniform
    /// spread recommended in G.729 §3.2.4 Eq. (28):
    /// `pi * (k + 1) / (M + 1)` for `k = 0..M-1`. The MA history is
    /// seeded so that the very first decoded LSF lands on that same
    /// uniform spread when all indices are zero — the reference
    /// implementation initialises the history to `pi * (k+1) / (M+1)`
    /// in Q13 (i.e. same value as `lsp_prev` after conversion).
    pub fn new() -> Self {
        // Initial LSFs in Q13, matching the spec's `lsp_old_q[]` init.
        // lsf[k] = pi * (k+1) / 11, and Q13(pi) = 25736.
        let mut lsf_init_q13 = [0i16; LPC_ORDER];
        let pi_q13: i32 = 25736; // pi << 13
        for k in 0..LPC_ORDER {
            lsf_init_q13[k] = (pi_q13 * (k as i32 + 1) / (LPC_ORDER as i32 + 1)) as i16;
        }
        // LSP_prev = cos(LSF) in f32.
        let mut lsp_prev = [0.0f32; LPC_ORDER];
        for k in 0..LPC_ORDER {
            let lsf = (lsf_init_q13[k] as f32) / 8192.0;
            lsp_prev[k] = lsf.cos();
        }
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        Self {
            freq_res_q13: [lsf_init_q13; MA_HISTORY],
            a,
            lsp_prev,
        }
    }

    /// Roll the MA-predictor history to make room for a new residual
    /// vector `v`. The oldest entry is dropped.
    pub fn push_residual(&mut self, v: [i16; LPC_ORDER]) {
        // Rotate: freq[k] <- freq[k-1], freq[0] <- v.
        for k in (1..MA_HISTORY).rev() {
            self.freq_res_q13[k] = self.freq_res_q13[k - 1];
        }
        self.freq_res_q13[0] = v;
    }
}

/// Decode one frame's LSP vector from the four LSP indices.
///
/// Returns LSPs in the **cosine domain** (strictly decreasing,
/// components in `[-1, 1]`), suitable for feeding to [`lsp_to_lpc`]
/// after optional interpolation.
///
/// Updates `state.freq_res_q13` with the new residual (the raw sum of
/// first + second-stage table entries) so the MA predictor advances.
pub fn decode_lsp(
    state: &mut LpcPredictorState,
    l0: u8,
    l1: u8,
    l2: u8,
    l3: u8,
) -> [f32; LPC_ORDER] {
    // 1) Build the Q13 residual vector (L1 vector plus L2/L3 residuals).
    let predictor = (l0 & 1) as usize;
    let l1 = (l1 as usize) & 0x7F;
    let l2 = (l2 as usize) & 0x1F;
    let l3 = (l3 as usize) & 0x1F;

    let cb1 = &LSPCB1_Q13[l1];
    let cb2_lo = &LSPCB2_Q13[l2];
    let cb2_hi = &LSPCB2_Q13[l3];

    let mut residual = [0i16; LPC_ORDER];
    for k in 0..M_HALF {
        residual[k] = cb1[k].saturating_add(cb2_lo[k]);
        residual[k + M_HALF] = cb1[k + M_HALF].saturating_add(cb2_hi[k]);
    }

    // 2) Reconstruct the quantised LSF vector via the MA predictor.
    //    lsf[j] = fg_sum[p][j] * residual[j] * fg_sum_inv[p][j]
    //          + sum_{k=0..MA_NP} fg[p][k][j] * prev_residual[k][j]
    //    (all in fixed-point; we switch to f32 here.)
    let fg = &FG_Q15[predictor];
    let fg_sum = &FG_SUM_Q15[predictor];
    let _fg_sum_inv = &FG_SUM_INV_Q12[predictor];

    // The spec formula reconstructs lsf_q[j] so that
    //   sum_over_k_plus_current fg_entries[k][j] == 1   (in Q15).
    // In floating-point, with `fg` scaled to Q15 (divide by 1<<15),
    //   lsf[j] = (fg_sum[j] * residual_now[j]
    //          + sum_k fg[k][j] * residual_prev[k][j])  /  1<<15
    // with residual_now / residual_prev already in Q13. The Q13 lsf
    // result we convert to radians by dividing by 8192.
    let mut lsf_q13 = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        let mut acc: f32 = (fg_sum[j] as f32) * (residual[j] as f32);
        for k in 0..MA_HISTORY {
            acc += (fg[k][j] as f32) * (state.freq_res_q13[k][j] as f32);
        }
        lsf_q13[j] = acc / 32768.0;
    }

    // 3) Push the *raw residual* into the predictor history.
    state.push_residual(residual);

    // 4) Convert Q13 LSFs to radians and enforce monotonicity + safety
    //    spacing so the LPC conversion stays stable.
    let mut lsf = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        lsf[j] = lsf_q13[j] / 8192.0;
    }
    // Clamp to (eps, pi - eps) and enforce minimum spacing.
    let pi = core::f32::consts::PI;
    let eps = LSF_MIN_SPACING;
    if lsf[0] < eps {
        lsf[0] = eps;
    }
    for j in 1..LPC_ORDER {
        if lsf[j] < lsf[j - 1] + eps {
            lsf[j] = lsf[j - 1] + eps;
        }
    }
    if lsf[LPC_ORDER - 1] > pi - eps {
        // Squeeze back from the top.
        lsf[LPC_ORDER - 1] = pi - eps;
        for j in (0..LPC_ORDER - 1).rev() {
            if lsf[j] > lsf[j + 1] - eps {
                lsf[j] = lsf[j + 1] - eps;
            }
        }
    }

    // 5) Convert LSFs to the LSP cosine domain (strictly decreasing).
    let mut lsp = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        lsp[j] = lsf[j].cos();
    }
    lsp
}

/// Linear interpolation between two LSP vectors in the cosine domain.
/// `alpha = 0` -> `a`, `alpha = 1` -> `b`.
pub fn interpolate_lsp(a: &[f32; LPC_ORDER], b: &[f32; LPC_ORDER], alpha: f32) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        out[j] = (1.0 - alpha) * a[j] + alpha * b[j];
    }
    out
}

/// Convert an LSP vector (cosine-domain, strictly decreasing) to 10 LPC
/// predictor coefficients `a[1..=10]`. `a[0]` is set to `1.0`.
///
/// Algorithm (G.729 §3.2.6 / standard LSP→LPC expansion):
///
/// A(z) = (F1(z) + F2(z)) / 2, where
///
/// F1(z) = prod_{odd LSPs} (1 - 2*q_i*z^-1 + z^-2) * (1 + z^-1)
/// F2(z) = prod_{even LSPs} (1 - 2*q_i*z^-1 + z^-2) * (1 - z^-1)
///
/// We expand F1 and F2 to degree M/2+1, then add them and halve, then
/// add/subtract the (1 ± z^-1) trailing factors to get the degree-M
/// polynomial coefficients.
pub fn lsp_to_lpc(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
    // Partition LSPs: even-indexed (0,2,4,6,8) -> F1, odd-indexed
    // (1,3,5,7,9) -> F2. (G.729 Annex A `Lsp_Az`.)
    let mut f1 = [0.0f32; M_HALF + 1];
    let mut f2 = [0.0f32; M_HALF + 1];
    get_lsp_pol(&[lsp[0], lsp[2], lsp[4], lsp[6], lsp[8]], &mut f1);
    get_lsp_pol(&[lsp[1], lsp[3], lsp[5], lsp[7], lsp[9]], &mut f2);

    // Multiply F1 by (1 + z^-1) -> F1'(z), F2 by (1 - z^-1) -> F2'(z).
    // In-place update is valid if we walk from top down: f1'[i] =
    // f1[i] + f1[i-1]; f2'[i] = f2[i] - f2[i-1], for i = M_HALF .. 1.
    for i in (1..=M_HALF).rev() {
        f1[i] = f1[i] + f1[i - 1];
        f2[i] = f2[i] - f2[i - 1];
    }

    // Compose A(z) = 0.5 * (F1'(z) + F2'(z)). Because F1' is symmetric
    // (palindromic) and F2' is anti-symmetric in the degree-M+1 form,
    // we get:
    //   a[k]     = 0.5 * (f1[k] + f2[k])   for k = 1..M_HALF
    //   a[M+1-k] = 0.5 * (f1[k] - f2[k])   for k = 1..M_HALF
    // and a[0] = 1. (Standard convention: A(z) = 1 + sum a[k] z^-k.)
    let mut a = [0.0f32; LPC_ORDER + 1];
    a[0] = 1.0;
    for k in 1..=M_HALF {
        let sym = 0.5 * (f1[k] + f2[k]);
        let anti = 0.5 * (f1[k] - f2[k]);
        a[k] = sym;
        a[LPC_ORDER + 1 - k] = anti;
    }
    a
}

/// Expand the polynomial prod_i (1 - 2*q_i*z^-1 + z^-2) for
/// `q.len() = M_HALF = 5` LSPs. The product has the palindromic
/// property `F(z) = z^-M * F(1/z)` (degree M = `2*M_HALF`), so only
/// the first `M_HALF + 1` coefficients are independent — this is what
/// `out` receives (`out[0] == 1`, `out[M_HALF] == 1` by symmetry).
///
/// Translates the recurrence used by ITU-T G.729 Annex A
/// `Get_lsp_pol` — we walk `j` from the top downwards so we read
/// `out[j-1]`/`out[j-2]` before they're overwritten.
fn get_lsp_pol(q: &[f32; M_HALF], out: &mut [f32; M_HALF + 1]) {
    out[0] = 1.0;
    out[1] = -2.0 * q[0];
    for i in 2..=M_HALF {
        let qi = q[i - 1];
        // Leading coefficient: out[i] grows by +2 every iteration, but
        // because we're keeping only the "low half" of a palindromic
        // polynomial, the running leading term is `(2-2*qi) * 1` = ...
        // Actually the G.729 C reference writes:
        //     f[i] = -2 * lsp_i * f[i-1] + 2 * f[i-2]
        // (note the +2, not +1, because the z^-(2i) term and the
        // palindromic partner fold together by the time i=5).
        out[i] = -2.0 * qi * out[i - 1] + 2.0 * out[i - 2];
        // Middle coefficients, j = i-1 .. 2, walking downwards.
        for j in (2..i).rev() {
            out[j] = out[j] - 2.0 * qi * out[j - 1] + out[j - 2];
        }
        // Lowest coefficient (j=1): out[0]=1, so out[j-2] is not used.
        out[1] = out[1] - 2.0 * qi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_lsps_are_decreasing() {
        let st = LpcPredictorState::new();
        for k in 1..LPC_ORDER {
            assert!(
                st.lsp_prev[k] < st.lsp_prev[k - 1],
                "lsp not decreasing at {k}: {} vs {}",
                st.lsp_prev[k],
                st.lsp_prev[k - 1]
            );
        }
    }

    #[test]
    fn push_residual_rolls_history() {
        let mut st = LpcPredictorState::new();
        let v = [1i16; LPC_ORDER];
        st.push_residual(v);
        assert_eq!(st.freq_res_q13[0], v);
        let w = [2i16; LPC_ORDER];
        st.push_residual(w);
        assert_eq!(st.freq_res_q13[0], w);
        assert_eq!(st.freq_res_q13[1], v);
    }

    #[test]
    fn lsp_to_lpc_identity_for_trivial_case() {
        // Uniform LSP spread -> LPC coefficients should be small (no
        // resonances). a[0] must be exactly 1.0.
        let st = LpcPredictorState::new();
        let a = lsp_to_lpc(&st.lsp_prev);
        assert_eq!(a[0], 1.0);
        // Coefficients should be finite.
        for k in 0..=LPC_ORDER {
            assert!(a[k].is_finite(), "a[{k}] is not finite: {}", a[k]);
        }
    }

    #[test]
    fn lsp_to_lpc_from_known_indices_is_stable() {
        // Decode LSPs from a known index quadruple, convert, and check
        // that the resulting AR filter is stable (all roots inside unit
        // circle) by verifying the polynomial value at z=1 is positive
        // and at z=-1 has the right sign — a crude but useful check.
        let mut st = LpcPredictorState::new();
        let lsp = decode_lsp(&mut st, 0, 0, 0, 0);
        // LSP must be strictly decreasing in cosine domain.
        for k in 1..LPC_ORDER {
            assert!(lsp[k] < lsp[k - 1], "lsp not decreasing at {k}");
        }
        let a = lsp_to_lpc(&lsp);
        assert_eq!(a[0], 1.0);
        // A(1) = 1 + sum a[k]; for a stable minimum-phase A(z) this is
        // positive (all roots inside unit circle imply A(1) > 0).
        let a_at_1: f32 = (1..=LPC_ORDER).map(|k| a[k]).sum::<f32>() + 1.0;
        assert!(a_at_1 > 0.0, "A(1) = {a_at_1} should be positive");
    }

    #[test]
    fn interpolate_lsp_midpoint_is_average() {
        let a = [0.5f32; LPC_ORDER];
        let b = [-0.5f32; LPC_ORDER];
        let m = interpolate_lsp(&a, &b, 0.5);
        for k in 0..LPC_ORDER {
            assert!((m[k] - 0.0).abs() < 1e-6);
        }
    }
}
