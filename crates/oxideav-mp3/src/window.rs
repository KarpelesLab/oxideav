//! MPEG-1 Layer III synthesis window D[] and IMDCT windows.
//!
//! D[] is the literal 512-tap polyphase synthesis window from
//! ISO/IEC 11172-3:1993 Annex B (informative) Table 3-B.3 / Annex D.1.
//! These are floating-point values quantised to 1/65536 (`intwinbase[j] /
//! 65536` in the libmpg123 `tabinit.c` representation). The same numbers
//! appear in the public-domain `dist10` reference decoder, in libmpg123
//! and in `puremp3` (MIT).
//!
//! Reference indices that are easy to spot-check:
//!   D[  0] =  0.000000000
//!   D[  1] = -0.000015259   (= -1/65536)
//!   D[ 64] = +0.003250122
//!   D[256] = +1.144989014   (= 75038/65536, the table peak)
//!   D[511] = +0.000015259
//!
//! To avoid shipping the same 512-entry table twice in the workspace,
//! the canonical copy lives in [`oxideav_mp1::window`] and MP3 simply
//! re-exports it. MP1 is the natural home: it is the simplest layer
//! (no Huffman / IMDCT machinery) and has no other crate dependencies,
//! so a minimal build of `oxideav-mp1` does not need any of the MP3
//! pieces. MP3's own IMDCT windows — which are Layer III specific —
//! remain here.

/// 512-tap synthesis window D[] (ISO/IEC 11172-3 Table 3-B.3).
///
/// Re-exported from the MP1 crate; see [`oxideav_mp1::window`].
/// Sum of |D[i]| ≈ 85.06; the polyphase synthesis matrix and this table
/// together produce unity-amplitude reconstruction (no extra scaling).
pub use oxideav_mp1::window::SYNTHESIS_WINDOW as SYNTH_WINDOW_D;

/// Return the 512-entry synthesis window D[] (ISO 11172-3 Annex B / D.1).
#[inline]
pub fn synthesis_window() -> &'static [f32; 512] {
    &SYNTH_WINDOW_D
}

/// IMDCT post-multiplication windows for the four block types.
/// ISO 11172-3 §2.4.3.4.10.3 / Figure A.4: window[block_type][n] for
/// n = 0..36 (long blocks) or n = 0..12 (short blocks).
pub fn imdct_window_long(block_type: u8) -> [f32; 36] {
    let mut w = [0.0f32; 36];
    let pi = std::f64::consts::PI;
    match block_type {
        0 => {
            // Normal: sin((n + 0.5) * pi / 36), n = 0..36.
            for (n, slot) in w.iter_mut().enumerate() {
                *slot = ((n as f64 + 0.5) * pi / 36.0).sin() as f32;
            }
        }
        1 => {
            // Start block: long-to-short transition.
            // n = 0..18: sin((n + 0.5) * pi / 36)
            // n = 18..24: 1.0
            // n = 24..30: sin((n - 18 + 0.5) * pi / 12)
            // n = 30..36: 0.0
            for (n, slot) in w.iter_mut().enumerate().take(18) {
                *slot = ((n as f64 + 0.5) * pi / 36.0).sin() as f32;
            }
            for slot in w.iter_mut().take(24).skip(18) {
                *slot = 1.0;
            }
            for (n, slot) in w.iter_mut().enumerate().take(30).skip(24) {
                *slot = ((n as f64 - 18.0 + 0.5) * pi / 12.0).sin() as f32;
            }
            // n = 30..36 stay 0.
        }
        3 => {
            // Stop block: short-to-long transition — mirror of type 1.
            // n = 0..6: 0.0
            // n = 6..12: sin((n - 6 + 0.5) * pi / 12)
            // n = 12..18: 1.0
            // n = 18..36: sin((n + 0.5) * pi / 36)
            for (n, slot) in w.iter_mut().enumerate().take(12).skip(6) {
                *slot = ((n as f64 - 6.0 + 0.5) * pi / 12.0).sin() as f32;
            }
            for slot in w.iter_mut().take(18).skip(12) {
                *slot = 1.0;
            }
            for (n, slot) in w.iter_mut().enumerate().take(36).skip(18) {
                *slot = ((n as f64 + 0.5) * pi / 36.0).sin() as f32;
            }
        }
        _ => {
            // Block type 2 is short — not used with this window fn; caller
            // should use imdct_window_short.
        }
    }
    w
}

pub fn imdct_window_short() -> [f32; 12] {
    let mut w = [0.0f32; 12];
    let pi = std::f64::consts::PI;
    for (n, slot) in w.iter_mut().enumerate() {
        *slot = ((n as f64 + 0.5) * pi / 12.0).sin() as f32;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_window_endpoints_match_iso_table() {
        // Cross-check the known reference values from ISO 11172-3
        // Table 3-B.3 / libmpg123 intwinbase / puremp3 SYNTH_DTBL.
        // SYNTH_WINDOW_D is a re-export of oxideav_mp1::window::SYNTHESIS_WINDOW
        // — this test verifies the re-export still resolves to the
        // canonical table bits.
        assert_eq!(SYNTH_WINDOW_D[0], 0.0);
        assert!((SYNTH_WINDOW_D[1] - -0.000015259).abs() < 1e-9);
        assert!((SYNTH_WINDOW_D[64] - 0.003250122).abs() < 1e-9);
        // The peak: 75038 / 65536 = 1.14498901367...
        assert!((SYNTH_WINDOW_D[256] - 1.144989014).abs() < 1e-7);
        assert!((SYNTH_WINDOW_D[511] - 0.000015259).abs() < 1e-9);
    }

    #[test]
    fn synthesis_window_sum_abs_matches_spec() {
        let s: f64 = SYNTH_WINDOW_D.iter().map(|v| v.abs() as f64).sum();
        // Sum of |D[i]| over 512 taps for the canonical ISO table is
        // approximately 85.064 — well above the analytic-sinc value of
        // ~32 we used previously.
        assert!(
            (s - 85.064).abs() < 0.5,
            "unexpected sum_abs {s}; check table transcription"
        );
    }

    #[test]
    fn synthesis_window_matches_mp1() {
        // Document the cross-crate dedup: MP3's SYNTH_WINDOW_D is a
        // `pub use` re-export of MP1's SYNTHESIS_WINDOW, so the two
        // symbols must observe the same bits. (`SYNTHESIS_WINDOW` is
        // `const`, so pointer equality is not guaranteed — compare by
        // value.)
        for i in 0..512 {
            assert_eq!(
                SYNTH_WINDOW_D[i],
                oxideav_mp1::window::SYNTHESIS_WINDOW[i],
                "divergence at index {i}"
            );
        }
    }

    #[test]
    fn imdct_long_is_symmetric_for_type_0() {
        let w = imdct_window_long(0);
        for i in 0..18 {
            let diff = (w[i] - w[35 - i]).abs();
            assert!(diff < 1e-5, "window type 0 asymmetric at {i}: {diff}");
        }
    }

    #[test]
    fn imdct_short_is_symmetric() {
        let w = imdct_window_short();
        for i in 0..6 {
            let diff = (w[i] - w[11 - i]).abs();
            assert!(diff < 1e-5);
        }
    }
}
