//! Wideband (sub-band CELP) sub-mode descriptors.
//!
//! Mirrors the four `wb_submodeN` records in `libspeex/modes_wb.c`.
//! Speex wideband (16 kHz) is layered on top of the narrowband decoder:
//! the low band is the NB decoder's output, and the high band (4 kHz –
//! 8 kHz) is an 8th-order SB-CELP layer with its own LSPs, optional
//! innovation codebook, and synthesis filter.
//!
//! Bit budget per sub-mode (from `modes_wb.c`):
//!
//! ```text
//!   submode id | label        | bits | innovation
//!   -----------+--------------+------+-----------------------------
//!   1          |  4 kbit ext  |   36 | none (spectral folding only)
//!   2          | ~8.8 kbit    |  112 | SplitCb 5×10 × 5-bit + sign
//!   3          | ~12 kbit     |  192 | SplitCb 5×8 × 7-bit + sign
//!   4          | ~20 kbit     |  352 | SplitCb 5×8 × 7-bit + sign (dbl cb)
//! ```

use crate::hexc_tables::{HEXC_10_32_TABLE, HEXC_TABLE};
use crate::submodes::SplitCbParams;

/// How the SB-CELP high-band excitation is recovered.
#[derive(Clone, Copy, Debug)]
pub enum WbInnov {
    /// Spectral folding — alternates and scales the NB innovation
    /// samples (`libspeex/sb_celp.c` `!innovation_unquant` branch).
    /// Uses only the NB innovation + a 5-bit gain index.
    Folding,
    /// Stochastic codebook (4-bit gain + split-VQ shape + sign).
    SplitCb(SplitCbParams),
}

/// One SB-CELP wideband sub-mode record.
#[derive(Clone, Copy, Debug)]
pub struct WbSubmode {
    /// Per-sub-frame gain bits (0, 1, or 3 in the narrowband layer; for
    /// SB-CELP only the stochastic modes consume 4 gain bits, and that
    /// logic is handled inside [`WbInnov::SplitCb`] — the folding
    /// branch uses 5-bit `g`).
    pub have_subframe_gain: u32,
    /// If true, run the innovation quantizer twice per sub-frame and
    /// sum the two outputs at 0.4× weight. Used by WB sub-mode 4.
    pub double_codebook: bool,
    /// Innovation shape (folding or split-VQ).
    pub innov: WbInnov,
    /// Total bits per encoded frame (incl. wideband-bit + SB_SUBMODE_BITS
    /// selector, which sum to 4).
    pub bits_per_frame: u32,
}

/// Split-VQ codebook for WB sub-mode 2 (4 subvects × 10 samples, 5-bit
/// shape, no sign) — mirrors `split_cb_high_lbr` in `modes_wb.c`.
pub const SPLIT_CB_HIGH_LBR: SplitCbParams = SplitCbParams {
    subvect_size: 10,
    nb_subvect: 4,
    shape_cb: &HEXC_10_32_TABLE,
    shape_bits: 5,
    have_sign: false,
};

/// Split-VQ codebook for WB sub-modes 3 and 4 (5 subvects × 8 samples,
/// 7-bit shape, 1-bit sign) — mirrors `split_cb_high` in `modes_wb.c`.
pub const SPLIT_CB_HIGH: SplitCbParams = SplitCbParams {
    subvect_size: 8,
    nb_subvect: 5,
    shape_cb: &HEXC_TABLE,
    shape_bits: 7,
    have_sign: true,
};

/// WB sub-mode 1 — 4 kbit/s layer. LSPs + 5-bit gain; excitation is
/// recovered by spectral folding of the narrowband innovation.
pub const WB_SUBMODE_1: WbSubmode = WbSubmode {
    have_subframe_gain: 0,
    double_codebook: false,
    innov: WbInnov::Folding,
    bits_per_frame: 36,
};

/// WB sub-mode 2 — ~8.8 kbit/s layer (`split_cb_high_lbr`).
pub const WB_SUBMODE_2: WbSubmode = WbSubmode {
    have_subframe_gain: 0,
    double_codebook: false,
    innov: WbInnov::SplitCb(SPLIT_CB_HIGH_LBR),
    bits_per_frame: 112,
};

/// WB sub-mode 3 — ~12 kbit/s layer (`split_cb_high`).
pub const WB_SUBMODE_3: WbSubmode = WbSubmode {
    have_subframe_gain: 0,
    double_codebook: false,
    innov: WbInnov::SplitCb(SPLIT_CB_HIGH),
    bits_per_frame: 192,
};

/// WB sub-mode 4 — ~20 kbit/s layer (`split_cb_high` × 2).
pub const WB_SUBMODE_4: WbSubmode = WbSubmode {
    have_subframe_gain: 0,
    double_codebook: true,
    innov: WbInnov::SplitCb(SPLIT_CB_HIGH),
    bits_per_frame: 352,
};

/// Look up a WB sub-mode by its 3-bit selector. `0` is "null submode"
/// (no high-band transmission — decoder outputs low-band only); 5..=7
/// are reserved.
pub fn wb_submode(id: u32) -> Option<&'static WbSubmode> {
    match id {
        1 => Some(&WB_SUBMODE_1),
        2 => Some(&WB_SUBMODE_2),
        3 => Some(&WB_SUBMODE_3),
        4 => Some(&WB_SUBMODE_4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wb_submode_bit_counts_match_reference() {
        assert_eq!(WB_SUBMODE_1.bits_per_frame, 36);
        assert_eq!(WB_SUBMODE_2.bits_per_frame, 112);
        assert_eq!(WB_SUBMODE_3.bits_per_frame, 192);
        assert_eq!(WB_SUBMODE_4.bits_per_frame, 352);
    }

    #[test]
    fn wb_submode_ids_resolve() {
        assert!(wb_submode(1).is_some());
        assert!(wb_submode(4).is_some());
        assert!(wb_submode(5).is_none());
    }
}
