//! LSP quantisation codebook tables for G.729.
//!
//! These are the static tables defined in ITU-T G.729 §3.2.4 and shipped
//! verbatim in the reference implementation (`TAB_LD8A.C`). All values
//! are in `Q13` or `Q15` / `Q12` fixed-point exactly as the spec defines
//! them — the decoder body consumes them unchanged.
//!
//! Layout / sizing:
//! - [`LSPCB1_Q13`]: first-stage codebook, `NC0 = 128` entries of 10
//!   components each (`L1` field, 7 bits).
//! - [`LSPCB2_Q13`]: second-stage codebook, `NC1 = 32` entries of 5
//!   components each (`L2` and `L3` fields, 5 bits each; each field
//!   indexes its own half of the LSF vector).
//! - [`FG_Q15`]: MA-predictor coefficients, `[2][MA_NP=4][M=10]`.
//! - [`FG_SUM_Q15`] / [`FG_SUM_INV_Q12`]: per-predictor column sums and
//!   inverses used to compute the quantised LSF vector.
//!
//! The `LSPCB1_Q13` and `LSPCB2_Q13` entries below have been populated
//! with the first rows from the reference package; the full contents of
//! the remaining rows are drop-in entries from §3.2.4 Tables 5/6 of the
//! recommendation and will land alongside the LSP-to-LP conversion code.
//! The placeholder rows are clearly marked with `[0; 10]` / `[0; 5]`
//! sentinels so tests cannot mistake them for real data.

use crate::LPC_ORDER;

/// First-stage LSP codebook size.
pub const NC0: usize = 128;
/// Second-stage LSP codebook size.
pub const NC1: usize = 32;
/// Number of MA-predictor taps (history depth).
pub const MA_NP: usize = 4;
/// LPC order alias for use in table dimensions.
pub const M: usize = LPC_ORDER;
/// Half of the LPC order — the L2 / L3 half-vectors.
pub const M_HALF: usize = LPC_ORDER / 2;

/// First-stage LSP codebook (Q13). `LSPCB1_Q13[i][j]` is the j-th LSF
/// component of the i-th codeword.
///
/// Real data is only populated for a prefix of rows; the remaining rows
/// are zero sentinels until the decoder body is wired up.
pub const LSPCB1_Q13: [[i16; M]; NC0] = {
    let mut t = [[0i16; M]; NC0];
    // Row 0 (spec Table 5, entry 0):
    t[0] = [
        1486, 2168, 3751, 9074, 12134, 13944, 17983, 19173, 21190, 21820,
    ];
    // Row 1 (spec Table 5, entry 1):
    t[1] = [
        1730, 2640, 3450, 4870, 6126, 7876, 15644, 17817, 20294, 21902,
    ];
    // Row 127 (spec Table 5, last entry, quoted verbatim):
    t[127] = [
        1721, 2577, 5553, 7195, 8651, 10686, 15069, 16953, 18703, 19929,
    ];
    t
};

/// Second-stage LSP codebook (Q13). Two 5-component half-vectors are
/// stored in one 5-wide row; `L2` indexes columns 0..5, `L3` indexes
/// columns 0..5 of the _other_ halves (added to the upper five of
/// `LSPCB1_Q13`). See G.729 §3.2.4 for the split.
///
/// Real data is only populated for a prefix of rows; the remaining rows
/// are zero sentinels until the decoder body is wired up.
pub const LSPCB2_Q13: [[i16; M_HALF]; NC1] = {
    let mut t = [[0i16; M_HALF]; NC1];
    // Row 0 (spec Table 6, entry 0, first half):
    t[0] = [-435, -815, -742, 1033, -518];
    // Row 1 (spec Table 6, entry 1, first half):
    t[1] = [-833, -891, 463, -8, -1251];
    // Row 31 (spec Table 6, last entry, first half):
    t[31] = [-163, 674, -11, -886, 531];
    t
};

/// MA-predictor coefficients (Q15), two predictor sets (indexed by
/// `L0`), each with `MA_NP` history taps of `M = 10` components.
///
/// Values verbatim from ITU-T G.729 §3.2.4 Table 7.
pub const FG_Q15: [[[i16; M]; MA_NP]; 2] = [
    [
        [8421, 9109, 9175, 8965, 9034, 9057, 8765, 8775, 9106, 8673],
        [7018, 7189, 7638, 7307, 7444, 7379, 7038, 6956, 6930, 6868],
        [5472, 4990, 5134, 5177, 5246, 5141, 5206, 5095, 4830, 5147],
        [4056, 3031, 2614, 3024, 2916, 2713, 3309, 3237, 2857, 3473],
    ],
    [
        [7733, 7880, 8188, 8175, 8247, 8490, 8637, 8601, 8359, 7569],
        [4210, 3031, 2552, 3473, 3876, 3853, 4184, 4154, 3909, 3968],
        [3214, 1930, 1313, 2143, 2493, 2385, 2755, 2706, 2542, 2919],
        [3024, 1592, 940, 1631, 1723, 1579, 2034, 2084, 1913, 2601],
    ],
];

/// Per-column sums of [`FG_Q15`] in Q15, used in the LSP reconstruction
/// formula in §3.2.4 Eq. (19).
pub const FG_SUM_Q15: [[i16; M]; 2] = [
    [7798, 8447, 8205, 8293, 8126, 8477, 8447, 8703, 9043, 8604],
    [
        14585, 18333, 19772, 17344, 16426, 16459, 15155, 15220, 16043, 15708,
    ],
];

/// Per-column inverses of [`FG_SUM_Q15`] in Q12. See §3.2.4 Eq. (20).
pub const FG_SUM_INV_Q12: [[i16; M]; 2] = [
    [
        17210, 15888, 16357, 16183, 16516, 15833, 15888, 15421, 14840, 15597,
    ],
    [9202, 7320, 6788, 7738, 8170, 8154, 8856, 8818, 8366, 8544],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_dimensions() {
        assert_eq!(LSPCB1_Q13.len(), 128);
        assert_eq!(LSPCB1_Q13[0].len(), 10);
        assert_eq!(LSPCB2_Q13.len(), 32);
        assert_eq!(LSPCB2_Q13[0].len(), 5);
        assert_eq!(FG_Q15.len(), 2);
        assert_eq!(FG_Q15[0].len(), 4);
        assert_eq!(FG_Q15[0][0].len(), 10);
        assert_eq!(FG_SUM_Q15[0].len(), 10);
        assert_eq!(FG_SUM_INV_Q12[0].len(), 10);
    }

    #[test]
    fn fg_sum_is_close_to_complement_of_fg_column_sums() {
        // The published `fg_sum[p][j]` is approximately `(1<<15) -
        // sum_k fg[p][k][j]`, rounded at Q15 by the spec's table
        // generator. We only check the relationship holds within a few
        // units of Q15 — treat this as a transcription sanity check.
        for p in 0..2 {
            for j in 0..M {
                let mut s: i32 = 0;
                for k in 0..MA_NP {
                    s += FG_Q15[p][k][j] as i32;
                }
                let reconstructed = (1i32 << 15) - s;
                let diff = (reconstructed - FG_SUM_Q15[p][j] as i32).abs();
                assert!(
                    diff < 16,
                    "fg_sum drift at predictor {p}, col {j}: \
                     expected ≈{reconstructed}, got {}",
                    FG_SUM_Q15[p][j]
                );
            }
        }
    }

    #[test]
    fn lspcb1_populated_rows_are_monotonic() {
        // LSF codewords are monotonically increasing within each row.
        for &row_idx in &[0usize, 1, 127] {
            let row = &LSPCB1_Q13[row_idx];
            for j in 1..M {
                assert!(
                    row[j] > row[j - 1],
                    "LSPCB1_Q13[{row_idx}] not monotonic at {j}"
                );
            }
        }
    }
}
