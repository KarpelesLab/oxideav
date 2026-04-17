//! Codebook tables for the G.728 shape + gain VQ.
//!
//! The ITU-T G.728 reference code ships two tables:
//!
//! - `CODEBK` (a.k.a. `Y`) — 128 entries × 5 samples, the shape codebook.
//! - `GB` / `GQ` — 8 positive magnitudes for the gain codebook. The sign
//!   bit in the 10-bit index flips polarity independently.
//!
//! This module does **not** reproduce the spec tables bit-exactly. Instead
//! it synthesises stable, deterministic codebooks that match the spec's
//! *shape* (128 × 5 unit-RMS shape vectors; 8 log-spaced gain magnitudes)
//! so the decoder produces structured output without needing the full
//! Annex A reference dump. A follow-up swap-in of the exact ITU tables is
//! a pure data change — no code touches them beyond `SHAPE_CB` / `GAIN_CB`.

use crate::{GAIN_CB_SIZE, SHAPE_CB_SIZE, VECTOR_SIZE};

// ---------------------------------------------------------------------------
// Shape codebook (128 × 5)
// ---------------------------------------------------------------------------

/// Deterministic Park-Miller LCG so the tables are reproducible across
/// builds without pulling in a random crate. Seeded from the per-entry
/// index so tests can depend on specific values.
const fn lcg_next(state: u32) -> u32 {
    // 48271 * state mod (2^31 - 1), as a 32-bit wrapping multiply then modulo.
    // const fn can't do i64/u64 branchless modulo, so we do it via the
    // classic Bays/Durham split.
    let hi = state / 44488;
    let lo = state % 44488;
    let t = 48271u32
        .wrapping_mul(lo)
        .wrapping_sub(3399u32.wrapping_mul(hi));
    if t == 0 {
        2147483646
    } else {
        t & 0x7FFF_FFFF
    }
}

/// Convert an LCG state `u32` into a `[-1, 1]` f32.
const fn lcg_to_unit(state: u32) -> f32 {
    // state is in (0, 2^31-1]; map to [-1, 1]
    let u = state as f32 / 2147483647.0_f32;
    2.0 * u - 1.0
}

const fn build_shape_cb() -> [[f32; VECTOR_SIZE]; SHAPE_CB_SIZE] {
    let mut out = [[0.0_f32; VECTOR_SIZE]; SHAPE_CB_SIZE];
    let mut i = 0usize;
    while i < SHAPE_CB_SIZE {
        // Seed the LCG from the row index so every row is independent.
        let mut state = (i as u32).wrapping_mul(2654435761).wrapping_add(1);
        if state == 0 {
            state = 1;
        }
        // Draw 5 samples, then normalise to unit RMS.
        let mut raw = [0.0_f32; VECTOR_SIZE];
        let mut k = 0usize;
        while k < VECTOR_SIZE {
            state = lcg_next(state);
            raw[k] = lcg_to_unit(state);
            k += 1;
        }
        // Sum of squares.
        let mut ss = 0.0_f32;
        let mut k2 = 0usize;
        while k2 < VECTOR_SIZE {
            ss += raw[k2] * raw[k2];
            k2 += 1;
        }
        // Newton-style inverse-sqrt fallback (const fn -> no sqrt intrinsic).
        // For ss in a reasonable range [0.5, 5.0] we can approximate:
        //   1/sqrt(x) ≈ 1 / (0.5 + 0.5x) for x≈1; refine with one Newton step.
        let rms = {
            let mut y = 1.0_f32 / (0.5 + 0.5 * ss);
            // One Newton step of f(y) = 1/y^2 - ss  ⇒  y += 0.5 * y * (1 - ss*y*y)
            let mut s = 0;
            while s < 4 {
                y += 0.5 * y * (1.0 - ss * y * y);
                s += 1;
            }
            // y is now ≈ 1/sqrt(ss); we want sqrt(ss/VECTOR_SIZE) as the RMS.
            // rms = sqrt(ss / 5); inv_rms = sqrt(5) * y.
            // sqrt(5) ≈ 2.2360679775
            let inv_rms = 2.2360679775 * y;
            inv_rms
        };
        let mut k3 = 0usize;
        while k3 < VECTOR_SIZE {
            out[i][k3] = raw[k3] * rms;
            k3 += 1;
        }
        i += 1;
    }
    out
}

/// 128 × 5 shape codebook, unit-RMS rows (deterministic placeholder, see
/// module docstring). Rows are independent across the full 128-entry set
/// so every index produces a distinct, non-zero excitation shape.
pub const SHAPE_CB: [[f32; VECTOR_SIZE]; SHAPE_CB_SIZE] = build_shape_cb();

// ---------------------------------------------------------------------------
// Gain codebook (8 magnitudes)
// ---------------------------------------------------------------------------

/// 4 positive gain magnitudes, log-spaced across the dynamic range
/// typical of G.728 quantised excitations. The 2-bit magnitude field in
/// the 10-bit bitstream selects one of these; the independent sign bit
/// flips polarity for a total of 8 signed levels.
///
/// Values span a 6-octave range so the backward-adaptive log-gain
/// predictor has enough per-step reach to ramp up from silence to a
/// loud target in a handful of vectors (each vector can change the log
/// gain by up to `ln(4.0) ≈ 1.39` nats).
pub const GAIN_CB: [f32; GAIN_CB_SIZE] = [0.25, 1.0, 2.5, 4.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_cb_rows_are_unit_rms() {
        for (i, row) in SHAPE_CB.iter().enumerate() {
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let rms = (ss / VECTOR_SIZE as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.05,
                "row {i} rms = {rms}, expected ~1.0"
            );
        }
    }

    #[test]
    fn shape_cb_rows_are_distinct() {
        // No two rows should be bit-identical.
        for i in 0..SHAPE_CB_SIZE {
            for j in (i + 1)..SHAPE_CB_SIZE {
                let mut diff = 0.0_f32;
                for k in 0..VECTOR_SIZE {
                    diff += (SHAPE_CB[i][k] - SHAPE_CB[j][k]).abs();
                }
                assert!(diff > 1e-6, "rows {i} and {j} are identical");
            }
        }
    }

    #[test]
    fn gain_cb_is_monotone_increasing() {
        for k in 1..GAIN_CB_SIZE {
            assert!(GAIN_CB[k] > GAIN_CB[k - 1]);
        }
    }
}
