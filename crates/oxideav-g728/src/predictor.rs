//! Backward-adaptive LPC + log-gain predictors for G.728.
//!
//! The ITU-T G.728 decoder re-estimates a 50th-order LPC synthesis filter
//! and a 10th-order log-gain predictor every 4 vectors (= 20 samples,
//! 2.5 ms) purely from the decoder's own reconstructed speech / gain
//! history. No side information is transmitted — the encoder and decoder
//! stay in sync by running the same analysis on the same signal.
//!
//! This module implements:
//!
//! - Windowed autocorrelation over the last `HISTORY_LEN` synthesis samples.
//! - Levinson-Durbin recursion for the 50th-order LPC coefficients.
//! - A short Levinson-Durbin (order 10) over the log-gain history for the
//!   gain predictor.
//! - Bandwidth-expansion (mu = 0.9883^k) applied to the LPC vector so the
//!   synthesis filter stays bounded even when the autocorrelation estimate
//!   is rank-deficient.
//!
//! Shortcuts vs. the reference code:
//!
//! - The spec uses a hybrid "logarithmic windowing" (Barnwell) to
//!   accumulate autocorrelations over a decaying window spanning several
//!   hundred samples. This module uses a fixed 100-sample Hamming window
//!   over the most recent history — simpler, slightly less stable during
//!   transients but free of the spec's weird recursive accumulators.
//! - Spectral smoothing via γ = 0.75 (bandwidth expansion) is applied in
//!   place of the spec's 15 Hz bandwidth-expansion table (§3.7.1) — same
//!   shape, different constant.
//! - The log-gain predictor uses the same path as the LPC predictor at
//!   order 10 rather than the spec's lattice formulation.

use crate::{GAIN_ORDER, LPC_ORDER};

/// Length of the history window we run autocorrelation over. The spec
/// uses a decaying recursive accumulator spanning ~150 samples; we use a
/// fixed tapered window of this many most-recent synthesis samples.
pub const HISTORY_LEN: usize = 100;

/// Bandwidth expansion factor applied to the LPC vector. Multiplies a[k]
/// by γ^k to pull the filter's poles inward from the unit circle, which
/// guarantees stability even when the autocorrelation estimate is noisy.
pub const BW_EXPANSION: f32 = 0.96;

/// Hamming window of length `HISTORY_LEN`.
fn hamming_window() -> [f32; HISTORY_LEN] {
    let mut w = [0.0_f32; HISTORY_LEN];
    let denom = (HISTORY_LEN - 1) as f32;
    for n in 0..HISTORY_LEN {
        let phase = 2.0 * core::f32::consts::PI * (n as f32) / denom;
        w[n] = 0.54 - 0.46 * phase.cos();
    }
    w
}

/// Compute the first `order+1` autocorrelation lags of a windowed buffer.
///
/// `history[0]` is the most recent sample; successively older samples
/// follow. The window is applied in time order (oldest sample gets the
/// leading-edge weight), matching the usual DSP convention.
pub fn autocorrelation<const N: usize>(history: &[f32; N], order: usize) -> Vec<f32> {
    assert!(order + 1 <= N, "order+1 must fit in history");
    let win = {
        // Build a Hamming window sized to N at runtime (HISTORY_LEN is fixed
        // but this fn is generic for unit tests with smaller N).
        let mut w = vec![0.0_f32; N];
        let denom = (N - 1) as f32;
        for n in 0..N {
            let phase = 2.0 * core::f32::consts::PI * (n as f32) / denom;
            w[n] = 0.54 - 0.46 * phase.cos();
        }
        w
    };
    // Apply window in time order: history is newest-first, so history[N-1]
    // is oldest and should get win[0] (leading edge).
    let mut x = vec![0.0_f32; N];
    for n in 0..N {
        x[n] = history[N - 1 - n] * win[n];
    }
    let mut r = vec![0.0_f32; order + 1];
    for k in 0..=order {
        let mut acc = 0.0_f32;
        for n in k..N {
            acc += x[n] * x[n - k];
        }
        r[k] = acc;
    }
    r
}

/// Levinson-Durbin recursion: autocorrelation `r[0..=order]` → predictor
/// coefficients `a[0..=order]` with `a[0] = 1`.
///
/// Returns `None` if the recursion encounters a non-positive prediction
/// error (numerically singular input). Callers should fall back to the
/// previous predictor in that case.
pub fn levinson_durbin(r: &[f32], order: usize) -> Option<Vec<f32>> {
    assert_eq!(r.len(), order + 1);
    if r[0] <= 0.0 {
        return None;
    }
    let mut a = vec![0.0_f32; order + 1];
    a[0] = 1.0;
    let mut e = r[0];

    let mut tmp = vec![0.0_f32; order + 1];

    for i in 1..=order {
        // Reflection coefficient k_i = -(r[i] + sum_{j=1..i-1} a[j]*r[i-j]) / e.
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / e;
        if !k.is_finite() || k.abs() >= 1.0 {
            // Non-minimum-phase: bail out.
            return None;
        }
        // a^(i)[j] = a^(i-1)[j] + k * a^(i-1)[i-j]
        tmp[..=i].copy_from_slice(&a[..=i]);
        for j in 1..i {
            a[j] = tmp[j] + k * tmp[i - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        if e <= 0.0 || !e.is_finite() {
            return None;
        }
    }
    Some(a)
}

/// Apply bandwidth expansion: a[k] := a[k] * γ^k for k = 1..=order.
pub fn bandwidth_expand(a: &mut [f32], gamma: f32) {
    let mut g = gamma;
    for k in 1..a.len() {
        a[k] *= g;
        g *= gamma;
    }
}

/// Update an LPC coefficient vector `a[0..=LPC_ORDER]` from the recent
/// synthesis history. On numerical failure the existing `a` is left
/// unchanged and `false` is returned.
pub fn update_lpc_from_history(a: &mut [f32; LPC_ORDER + 1], history: &[f32; HISTORY_LEN]) -> bool {
    let r = autocorrelation::<HISTORY_LEN>(history, LPC_ORDER);
    // Add a tiny white-noise floor to r[0] to keep the recursion robust
    // when the decoder has produced very quiet output (all zeros in the
    // first few vectors).
    let mut r = r;
    let floor = r[0] * 1e-4 + 1e-6;
    r[0] += floor;
    let Some(mut new_a) = levinson_durbin(&r, LPC_ORDER) else {
        return false;
    };
    bandwidth_expand(&mut new_a, BW_EXPANSION);
    for k in 0..=LPC_ORDER {
        a[k] = new_a[k];
    }
    true
}

/// Update a gain-predictor coefficient vector from the log-gain history.
/// Same shape as `update_lpc_from_history` but 10th-order.
pub fn update_gain_predictor(
    b: &mut [f32; GAIN_ORDER + 1],
    log_gain_history: &[f32; crate::predictor::GAIN_HISTORY_LEN],
) -> bool {
    let r = autocorrelation::<{ crate::predictor::GAIN_HISTORY_LEN }>(log_gain_history, GAIN_ORDER);
    let mut r = r;
    let floor = r[0] * 1e-4 + 1e-6;
    r[0] += floor;
    let Some(mut new_b) = levinson_durbin(&r, GAIN_ORDER) else {
        return false;
    };
    bandwidth_expand(&mut new_b, 0.90);
    for k in 0..=GAIN_ORDER {
        b[k] = new_b[k];
    }
    true
}

/// History length for the log-gain predictor autocorrelation.
pub const GAIN_HISTORY_LEN: usize = 40;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levinson_recovers_ar1_coefficient() {
        // y[n] = 0.8 * y[n-1] + x[n]   ⇒ A(z) = 1 - 0.8 z^-1
        // Autocorrelation for the output of this model driven by unit
        // white noise has r[0] = 1 / (1 - 0.8^2) = 2.7778 and
        // r[1] = 0.8 * r[0] = 2.2222.
        let r = vec![2.7778, 2.2222];
        let a = levinson_durbin(&r, 1).expect("recursion");
        assert!((a[0] - 1.0).abs() < 1e-6);
        // Our convention: y[n] = x[n] - sum(a[k] y[n-k]). So for the
        // given model we want a[1] = -0.8.
        assert!((a[1] + 0.8).abs() < 1e-3, "a[1] = {} expected ≈ -0.8", a[1]);
    }

    #[test]
    fn levinson_rejects_nonpositive_r0() {
        assert!(levinson_durbin(&[0.0, 0.5], 1).is_none());
    }

    #[test]
    fn bandwidth_expansion_shrinks_higher_order_taps() {
        let mut a = [1.0_f32, 0.5, 0.5, 0.5];
        bandwidth_expand(&mut a, 0.5);
        assert!((a[0] - 1.0).abs() < 1e-6);
        assert!((a[1] - 0.25).abs() < 1e-6);
        assert!((a[2] - 0.125).abs() < 1e-6);
        assert!((a[3] - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn autocorrelation_is_symmetric_in_construction() {
        let mut hist = [0.0_f32; 16];
        for n in 0..16 {
            hist[n] = ((n as f32) * 0.3).sin();
        }
        let r = autocorrelation::<16>(&hist, 4);
        assert_eq!(r.len(), 5);
        // r[0] is energy — must be non-negative.
        assert!(r[0] >= 0.0);
        // r[k] for k>0 should be bounded by r[0].
        for k in 1..5 {
            assert!(r[k].abs() <= r[0] + 1e-6);
        }
    }

    #[test]
    fn update_lpc_handles_zero_history() {
        // All-zero history: the tiny floor we add to r[0] should keep the
        // recursion alive and produce a trivial (near-identity) filter.
        let mut a = [0.0_f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let hist = [0.0_f32; HISTORY_LEN];
        let ok = update_lpc_from_history(&mut a, &hist);
        assert!(ok, "recursion on zero history should still succeed");
        // The filter must remain well-defined.
        for k in 0..=LPC_ORDER {
            assert!(a[k].is_finite(), "a[{k}] = {} not finite", a[k]);
        }
    }
}
