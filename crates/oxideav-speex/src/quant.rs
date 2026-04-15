//! Codebook / quantisation table stubs.
//!
//! Speex's CELP decoder looks up three families of quantisation tables:
//!   * **LSP codebooks** — Line-Spectral-Pairs for the short-term predictor.
//!   * **Pitch codebooks** — gains/positions for the long-term predictor.
//!   * **Innovation codebooks** — the (sparse) residual excitation.
//!
//! The reference implementation (`libspeex/*_tables_nb.c`,
//! `*_tables_lbr.c`, `exc_*_table.c`, `gain_table.c`, `cb_search.c`) ships
//! several hundred kB of constant tables. Replicating those is a
//! mechanical port — they live here as named stubs so downstream code
//! can refer to them by the same names as the reference, but with the
//! actual numeric content deferred.
//!
//! The stubs are marked `[T; 0]` on purpose: any attempt to use them
//! without first populating the arrays will fail to compile, which is
//! the signal we want. The synthesis code in `decoder.rs` therefore
//! returns `Unsupported` until those arrays are filled in.

/// LSP codebook used for sub-mode 1 (and its siblings). 10-stage VQ,
/// 64-entry inner codebooks per the reference.
pub const LSP_CDBK_NB: &[i16; 0] = &[];
/// Second-stage LSP codebook for narrowband.
pub const LSP_CDBK_NB_LOW: &[i16; 0] = &[];
/// High-band LSP codebook for wideband sub-modes.
pub const LSP_CDBK_HB: &[i16; 0] = &[];

/// Pitch gain codebook (three-tap LTP, 32 entries).
pub const GAIN_CDBK_NB: &[i16; 0] = &[];
/// Pitch gain codebook for low-bitrate NB sub-modes.
pub const GAIN_CDBK_LBR: &[i16; 0] = &[];
/// High-band pitch/excitation gain codebook.
pub const GAIN_CDBK_HB: &[i16; 0] = &[];

/// Innovation (fixed) codebook — 5 bits per pulse, 40-sample sub-frame.
pub const EXC_5_64_TABLE: &[i16; 0] = &[];
/// Innovation codebook for the 2.6 kbps mode.
pub const EXC_5_256_TABLE: &[i16; 0] = &[];
/// Innovation codebook for the 8 kbps mode.
pub const EXC_8_128_TABLE: &[i16; 0] = &[];
/// Innovation codebook for the 11 kbps mode.
pub const EXC_10_32_TABLE: &[i16; 0] = &[];
/// High-band innovation codebook.
pub const EXC_20_32_TABLE: &[i16; 0] = &[];

/// Pitch-lag codebook indices (narrowband, sub-frame length = 40).
/// The smallest encoded pitch period is 17 samples; the largest is 143.
pub const PITCH_LAG_MIN_NB: u32 = 17;
pub const PITCH_LAG_MAX_NB: u32 = 143;

/// Sub-frame size (samples), identical across all NB sub-modes.
pub const SUBFRAME_SIZE_NB: u32 = 40;
/// Sub-frames per NB frame (160 / 40 = 4).
pub const SUBFRAMES_PER_FRAME_NB: u32 = 4;

/// Sub-frame size for the wideband high-band layer (also 40).
pub const SUBFRAME_SIZE_HB: u32 = 40;

/// LPC analysis order.
pub const LPC_ORDER_NB: usize = 10;
pub const LPC_ORDER_HB: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_sane() {
        assert_eq!(SUBFRAME_SIZE_NB * SUBFRAMES_PER_FRAME_NB, 160);
        assert_eq!(PITCH_LAG_MIN_NB, 17);
        assert_eq!(PITCH_LAG_MAX_NB, 143);
    }

    #[test]
    fn stub_tables_are_linkable() {
        // Ensures the zero-length arrays remain addressable constants.
        let _: &[i16] = LSP_CDBK_NB;
        let _: &[i16] = GAIN_CDBK_NB;
        let _: &[i16] = EXC_5_64_TABLE;
    }
}
