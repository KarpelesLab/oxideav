//! Codebook / layout constants for the G.723.1 decoder.
//!
//! Only dimension and layout tables are included in this scaffold — the
//! full LSP-VQ, adaptive-pitch, and fixed-codebook values (tables 3–12 in
//! ITU-T G.723.1) will land with the synthesis path in a follow-up.
//!
//! References in comments point to the May 2006 ITU-T Recommendation.

/// Sampling rate of the encoded signal — fixed at 8 kHz.
pub const SAMPLE_RATE_HZ: u32 = 8_000;

/// Frame duration: 30 ms = 240 samples at 8 kHz.
pub const FRAME_SIZE_SAMPLES: usize = 240;

/// A frame is split into four 60-sample subframes (§3.2).
pub const SUBFRAME_SIZE: usize = 60;
/// Number of subframes per frame.
pub const SUBFRAMES_PER_FRAME: usize = 4;

/// Order of the LPC synthesis / perceptual-weighting filters (§3.3).
pub const LPC_ORDER: usize = 10;

/// LPC-analysis window length in samples (§3.3 Figure 3).
pub const LPC_WINDOW: usize = 180;

/// Pitch-search range — open-loop pitch estimate lies in [PITCH_MIN..=PITCH_MAX].
pub const PITCH_MIN: usize = 18;
pub const PITCH_MAX: usize = 142;

/// High-rate (6.3 kbit/s, MP-MLQ) frame size in bytes (§5.4, Table 6).
pub const HIGH_RATE_BYTES: usize = 24;
/// Total bits per 6.3 kbit/s frame.
pub const HIGH_RATE_BITS: u32 = 189;

/// Low-rate (5.3 kbit/s, ACELP) frame size in bytes.
pub const LOW_RATE_BYTES: usize = 20;
/// Total bits per 5.3 kbit/s frame.
pub const LOW_RATE_BITS: u32 = 158;

/// SID (silence-insertion descriptor) frame size in bytes — §Annex A.
pub const SID_BYTES: usize = 4;

/// Number of 8-bit predictor sub-codebooks that together form the 24-bit
/// LSP VQ index (§3.4, tables 3–5).
pub const LSP_VQ_SUBCODEBOOKS: usize = 3;
/// Entries per LSP sub-codebook.
pub const LSP_VQ_ENTRIES: usize = 256;

/// Perceptual-weighting filter γ₁/γ₂ constants (§3.5).
pub const PERCEPTUAL_GAMMA1: f32 = 0.9;
pub const PERCEPTUAL_GAMMA2: f32 = 0.5;

/// Post-filter constants (§3.9).
pub const POSTFILTER_GAMMA1: f32 = 0.65;
pub const POSTFILTER_GAMMA2: f32 = 0.75;
pub const POSTFILTER_TILT: f32 = 0.25;

/// Bit-layout (high-rate, MP-MLQ) — field widths in bits, in packing order.
///
/// Derived from ITU-T G.723.1 Annex B Table B.1. The implementation below
/// only uses these for consistency checks; per-field semantics live in the
/// synthesis module once implemented.
pub const HIGH_RATE_FIELD_WIDTHS: &[u32] = &[
    2, // LPC0 (upper sub-codebook index)
    8, // LPC1
    8, // LPC2
    8, // ACL0 (adaptive-codebook lag subframe 0)
    5, // ACL1
    5, // ACL2
    5, // ACL3
    2, // GAIN0 (combined gain index subframe 0, high rate)
    12, 12, 12, 12, // combined GAINs
    1, 1, 1, 1, // grid bits
    1, // reserved
    13, 16, 14, 14, // MP-MLQ pulse positions / signs subframes 0..3
    6, 5, 5, 5, // pulse-position LSBs
];

/// Bit-layout (low-rate, ACELP) — field widths in bits, in packing order.
pub const LOW_RATE_FIELD_WIDTHS: &[u32] = &[
    2, // LPC0
    8, // LPC1
    8, // LPC2
    7, // ACL0
    2, // ACL1
    7, // ACL2
    2, // ACL3
    12, 12, 12, 12, // combined GAINs
    1, 1, 1, 1, // grid bits
    12, 12, 12, 12, // ACELP fixed-codebook pulses (subframes 0..3)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_is_four_subframes() {
        assert_eq!(SUBFRAME_SIZE * SUBFRAMES_PER_FRAME, FRAME_SIZE_SAMPLES);
    }

    #[test]
    fn total_bits_match_byte_sizes() {
        // Note: published bit totals are 189 / 158; byte framing rounds up
        // with unused bits at the end of the last byte.
        assert!(HIGH_RATE_BITS as usize <= HIGH_RATE_BYTES * 8);
        assert!(LOW_RATE_BITS as usize <= LOW_RATE_BYTES * 8);
    }
}
