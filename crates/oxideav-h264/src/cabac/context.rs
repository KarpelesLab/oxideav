//! CABAC per-context probability state and initialization.
//!
//! Implements the formula in ITU-T H.264 (07/2019) §9.3.1.1:
//! "Initialization process for context variables".
//!
//! Given an `(m, n)` pair from Tables 9-12..9-33 and the slice's
//! `SliceQPY` value, the derived pair `(pStateIdx, valMPS)` is:
//!
//! ```text
//! preCtxState = Clip3(1, 126, ((m * Clip3(0, 51, SliceQPY)) >> 4) + n)
//!
//! if preCtxState <= 63:
//!     pStateIdx = 63 - preCtxState
//!     valMPS    = 0
//! else:
//!     pStateIdx = preCtxState - 64
//!     valMPS    = 1
//! ```
//!
//! `pStateIdx` indexes one of 64 probability states (each doubled so
//! that valMPS flips the "which symbol is MPS" bit), and drives the
//! state transition table used by the arithmetic decoder.

/// A single CABAC context variable.
///
/// Matches the `(pStateIdx, valMPS)` pair from §9.3.1.1. `pStateIdx`
/// is in `0..=63`; `valMPS` is `0` or `1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CabacContext {
    /// Probability state index (§Table 9-44): 0..=63.
    pub p_state_idx: u8,
    /// Most-probable symbol: 0 or 1.
    pub val_mps: u8,
}

impl CabacContext {
    /// Derive `(pStateIdx, valMPS)` from the `(m, n)` pair and
    /// `SliceQPY` per §9.3.1.1.
    ///
    /// `slice_qpy` is the luma QP for this slice
    /// (`pps.pic_init_qp_minus26 + 26 + slice_qp_delta`).
    pub fn init(m: i32, n: i32, slice_qpy: i32) -> Self {
        // Clip3(0, 51, SliceQPY)
        let qpy = clip3(0, 51, slice_qpy);
        // preCtxState = Clip3(1, 126, ((m * qpy) >> 4) + n)
        let pre = clip3(1, 126, ((m * qpy) >> 4) + n);
        if pre <= 63 {
            CabacContext {
                p_state_idx: (63 - pre) as u8,
                val_mps: 0,
            }
        } else {
            CabacContext {
                p_state_idx: (pre - 64) as u8,
                val_mps: 1,
            }
        }
    }
}

/// Implementation of Clip3 from §5.7 of the spec:
/// clips `x` into the inclusive range `[lo, hi]`.
#[inline]
fn clip3(lo: i32, hi: i32, x: i32) -> i32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::tables::{init_slice_contexts, NUM_CTX};

    /// Cross-check the §9.3.1.1 formula by hand, row-by-row.
    ///
    /// This doesn't need a spec table — the formula itself is exact,
    /// and we just verify `CabacContext::init` reproduces the steps
    /// for a handful of well-chosen `(m, n, SliceQPY)` triples.
    #[test]
    fn init_formula_matches_known_row() {
        // Case 1: tiny m, moderate n, mid-range QP.
        // m = 20, n = -15, SliceQPY = 28
        //   qpy = 28
        //   pre = Clip3(1,126, (20*28)>>4 + -15) = Clip3(1,126, 35 + -15) = 20
        //   20 <= 63 -> pStateIdx = 43, valMPS = 0
        let c = CabacContext::init(20, -15, 28);
        assert_eq!(c.p_state_idx, 43);
        assert_eq!(c.val_mps, 0);

        // Case 2: value falling above 63 triggers valMPS = 1.
        // m = 2, n = 54, SliceQPY = 26
        //   qpy = 26
        //   pre = Clip3(1,126, (2*26)>>4 + 54) = Clip3(1,126, 3 + 54) = 57
        //   57 <= 63 -> pStateIdx = 6, valMPS = 0
        let c = CabacContext::init(2, 54, 26);
        assert_eq!(c.p_state_idx, 6);
        assert_eq!(c.val_mps, 0);

        // Case 3: pre goes above 63 -> valMPS = 1.
        // m = 20, n = 58, SliceQPY = 40
        //   qpy = 40
        //   pre = Clip3(1,126, (20*40)>>4 + 58) = Clip3(1,126, 50 + 58) = 108
        //   108 > 63 -> pStateIdx = 44, valMPS = 1
        let c = CabacContext::init(20, 58, 40);
        assert_eq!(c.p_state_idx, 44);
        assert_eq!(c.val_mps, 1);

        // Case 4: Clip3 lower bound reached.
        // m = -20, n = -30, SliceQPY = 51
        //   qpy = 51
        //   pre = Clip3(1,126, (-20*51)>>4 + -30)
        //       = Clip3(1,126, -64 + -30) = Clip3(1,126, -94) = 1
        //   1 <= 63 -> pStateIdx = 62, valMPS = 0
        let c = CabacContext::init(-20, -30, 51);
        assert_eq!(c.p_state_idx, 62);
        assert_eq!(c.val_mps, 0);

        // Case 5: SliceQPY clipping below 0.
        // m = 10, n = 20, SliceQPY = -5 -> clipped to 0
        //   pre = Clip3(1,126, (10*0)>>4 + 20) = 20
        //   pStateIdx = 43, valMPS = 0
        let c = CabacContext::init(10, 20, -5);
        assert_eq!(c.p_state_idx, 43);
        assert_eq!(c.val_mps, 0);

        // Case 6: SliceQPY clipping above 51.
        // m = 1, n = 60, SliceQPY = 100 -> clipped to 51
        //   pre = Clip3(1,126, (1*51)>>4 + 60) = Clip3(1,126, 3 + 60) = 63
        //   63 <= 63 -> pStateIdx = 0, valMPS = 0
        let c = CabacContext::init(1, 60, 100);
        assert_eq!(c.p_state_idx, 0);
        assert_eq!(c.val_mps, 0);
    }

    /// An I-slice invocation must produce exactly `NUM_CTX` entries.
    #[test]
    fn init_slice_contexts_for_i_slice_has_460_entries() {
        let ctxs = init_slice_contexts(0, true, 26);
        assert_eq!(ctxs.len(), NUM_CTX);
        assert_eq!(NUM_CTX, 460);
        // All pStateIdx values must be in 0..=63.
        for c in &ctxs {
            assert!(c.p_state_idx <= 63);
            assert!(c.val_mps <= 1);
        }
    }

    /// Same for a P-slice using cabac_init_idc = 0.
    #[test]
    fn init_slice_contexts_for_p_slice_has_460_entries() {
        let ctxs = init_slice_contexts(0, false, 28);
        assert_eq!(ctxs.len(), NUM_CTX);
        for c in &ctxs {
            assert!(c.p_state_idx <= 63);
            assert!(c.val_mps <= 1);
        }
    }

    /// cabac_init_idc must be accepted in 0..=2 for P/B slices, and
    /// the three different idc values should yield different context
    /// vectors (at least somewhere).
    #[test]
    fn different_init_idc_gives_different_contexts() {
        let a = init_slice_contexts(0, false, 26);
        let b = init_slice_contexts(1, false, 26);
        let c = init_slice_contexts(2, false, 26);
        assert_eq!(a.len(), NUM_CTX);
        assert_eq!(b.len(), NUM_CTX);
        assert_eq!(c.len(), NUM_CTX);
        assert!(
            a != b || a != c,
            "all three idc tables collapsed to the same state"
        );
    }
}
