//! 10th-order LPC / LSP predictor state for G.729.
//!
//! G.729 uses a **moving-average (MA) predictor** of order 4 on the ten
//! LSP frequencies, switched between two predictor sets by the `L0` bit
//! (§3.2.4). This module holds:
//!
//! - `LpcPredictorState` — the rolling history of the four previous
//!   quantised LSF residual vectors (initialised to the silent state
//!   recommended by the spec).
//! - the 10 short-term predictor (LP) coefficients produced after LSP →
//!   LP conversion each subframe.
//!
//! LSP-to-LP conversion itself is left for a follow-up; this file
//! provides the state that conversion writes into.
//!
//! Reference: ITU-T G.729 §3.2.4 ("Computation of the LP coefficients").

use crate::LPC_ORDER;

/// Size of the MA-predictor history (four previous LSF residual vectors).
pub const MA_HISTORY: usize = 4;

/// State of the LPC / LSP predictor across frames.
///
/// All frequencies are stored as normalised LSF values in `[0, pi]` scaled
/// to `Q15` (`f32` here to keep the scaffold simple; the integer fixed-
/// point reference C code uses Q13). The decoder body will swap these
/// representations before shipping.
#[derive(Clone, Debug)]
pub struct LpcPredictorState {
    /// Previous-frame quantised LSF residuals (for the MA predictor).
    /// `freq[k]` holds the k-th-most-recent residual vector; indices are
    /// rotated on each frame.
    pub freq: [[f32; LPC_ORDER]; MA_HISTORY],
    /// 10 short-term predictor coefficients (after LSP → LP conversion).
    /// Populated each subframe; unused by the scaffold today.
    pub a: [f32; LPC_ORDER + 1],
    /// Previously decoded LSP frequencies, used as the starting point for
    /// LSP interpolation between subframes.
    pub lsp_prev: [f32; LPC_ORDER],
}

impl Default for LpcPredictorState {
    fn default() -> Self {
        Self::new()
    }
}

impl LpcPredictorState {
    /// Fresh predictor state, matching the G.729 reference initial values.
    ///
    /// The initial LSF vector is the uniform spread recommended in
    /// G.729 §3.2.4 Eq. (28), i.e. `pi * (k + 1) / (M + 1)` for
    /// `k = 0..M-1`.
    pub fn new() -> Self {
        let mut lsp_prev = [0.0f32; LPC_ORDER];
        // Spec uses cos(pi * k / (M + 1)); our scaffold stores raw LSFs.
        let step = core::f32::consts::PI / (LPC_ORDER as f32 + 1.0);
        for k in 0..LPC_ORDER {
            lsp_prev[k] = step * (k as f32 + 1.0);
        }
        Self {
            freq: [[0.0; LPC_ORDER]; MA_HISTORY],
            a: [0.0; LPC_ORDER + 1],
            lsp_prev,
        }
    }

    /// Roll the MA-predictor history to make room for a new residual
    /// vector `v`. The oldest entry is dropped.
    pub fn push_residual(&mut self, v: [f32; LPC_ORDER]) {
        // Rotate: freq[k] <- freq[k-1], freq[0] <- v.
        for k in (1..MA_HISTORY).rev() {
            self.freq[k] = self.freq[k - 1];
        }
        self.freq[0] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_lsps_are_monotone() {
        let st = LpcPredictorState::new();
        for k in 1..LPC_ORDER {
            assert!(st.lsp_prev[k] > st.lsp_prev[k - 1]);
        }
        assert!(st.lsp_prev[LPC_ORDER - 1] < core::f32::consts::PI);
    }

    #[test]
    fn push_residual_rolls_history() {
        let mut st = LpcPredictorState::new();
        let v = [1.0f32; LPC_ORDER];
        st.push_residual(v);
        assert_eq!(st.freq[0], v);
        let w = [2.0f32; LPC_ORDER];
        st.push_residual(w);
        assert_eq!(st.freq[0], w);
        assert_eq!(st.freq[1], v);
    }
}
