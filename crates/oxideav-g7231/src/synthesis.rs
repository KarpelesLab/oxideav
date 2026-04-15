//! LPC synthesis filter scaffold (10th order).
//!
//! The G.723.1 synthesis path (§3.7) reconstructs speech by running the
//! excitation through
//!
//! ```text
//!              1
//!     s(n) = -----  e(n)
//!            A(z)
//!
//!     A(z) = 1 + Σ_{k=1..10} a_k z^{-k}
//! ```
//!
//! This module hosts the direct-form IIR synthesis filter and its state;
//! upstream stages (LSP decode → interpolated LPCs per subframe, excitation
//! generation from adaptive + fixed codebooks, gain decoding, post-filter)
//! will land with the full decoder body.

use crate::tables::LPC_ORDER;

/// 10th-order all-pole synthesis filter state.
#[derive(Clone, Debug)]
pub struct LpcSynthesis {
    /// Filter memory (history of previously generated samples).
    mem: [f32; LPC_ORDER],
}

impl LpcSynthesis {
    pub fn new() -> Self {
        Self {
            mem: [0.0; LPC_ORDER],
        }
    }

    /// Reset the filter state to silence.
    pub fn reset(&mut self) {
        self.mem = [0.0; LPC_ORDER];
    }

    /// Run `s(n) = e(n) - Σ a_k s(n-k)` over `excitation`, writing the
    /// synthesised signal into `out` (same length). `lpc` is the per-frame
    /// (or per-subframe) LPC coefficient array `[a_1..a_10]` — the leading
    /// `a_0 = 1` is implied.
    pub fn filter(&mut self, lpc: &[f32; LPC_ORDER], excitation: &[f32], out: &mut [f32]) {
        debug_assert_eq!(excitation.len(), out.len());
        for (i, &e) in excitation.iter().enumerate() {
            let mut s = e;
            for (k, &a) in lpc.iter().enumerate() {
                s -= a * self.mem[k];
            }
            // Shift memory: mem[0] is newest, mem[LPC_ORDER-1] is oldest.
            for k in (1..LPC_ORDER).rev() {
                self.mem[k] = self.mem[k - 1];
            }
            self.mem[0] = s;
            out[i] = s;
        }
    }
}

impl Default for LpcSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_lpc_is_identity() {
        let mut syn = LpcSynthesis::new();
        let lpc = [0.0f32; LPC_ORDER];
        let excitation = [0.5, -0.25, 0.125, 0.0, 0.0];
        let mut out = [0.0f32; 5];
        syn.filter(&lpc, &excitation, &mut out);
        assert_eq!(out, excitation);
    }

    #[test]
    fn single_pole_decays() {
        // a_1 = -0.5 → pole at +0.5 (since denominator is 1 + a_1 z^{-1} ⇒ recursion s[n] = e[n] - a_1 s[n-1]).
        let mut syn = LpcSynthesis::new();
        let mut lpc = [0.0f32; LPC_ORDER];
        lpc[0] = -0.5;
        let excitation = [1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        syn.filter(&lpc, &excitation, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);
        assert!((out[3] - 0.125).abs() < 1e-6);
    }
}
