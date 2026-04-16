//! Fixed-point constants from ETSI EN 300 961 (GSM 06.10).
//!
//! Names match the spec tables so they are easy to cross-reference with
//! §5 of the standard (and with the public-domain libgsm reference).

/// LAR quantisation constants, one per coefficient (indexed 0..8).
/// `A[i]` is not kept directly — `INVA[i] = 2^16 / A[i]` replaces division.
pub const INVA: [i16; 8] = [13107, 13107, 13107, 13107, 19223, 17476, 31454, 29708];
pub const MIC: [i16; 8] = [-32, -32, -16, -16, -8, -8, -4, -4];
pub const MAC: [i16; 8] = [31, 31, 15, 15, 7, 7, 3, 3];
pub const B: [i16; 8] = [0, 0, 2048, -2560, 94, -1792, -341, -1144];

/// LTP gain quantiser output: `QLB[bc]` for bc ∈ 0..=3 (Q15).
pub const QLB: [i16; 4] = [3277, 11469, 21299, 32767];

/// RPE block amplitude dequantisation factor per spec §5.3.5 (FAC table).
pub const FAC: [i16; 8] = [18431, 20479, 22527, 24575, 26623, 28671, 30719, 32767];

/// Forward-quantisation helpers used by the analysis path.
/// `A[i] * 1024` and `B_TIMES_TWO[i]` from §5.2.8 (GSM 06.10 Table 5.1).
pub const A: [i16; 8] = [20480, 20480, 20480, 20480, 13964, 15360, 8534, 9036];
pub const B_TIMES_TWO: [i16; 8] = [0, 0, 4096, -5120, 188, -3584, -682, -2288];

/// LTP gain quantisation thresholds (§5.2.12 Table 5.4) — `DLB[0..3] * 32768`,
/// used to classify `R/S` into one of four bc indices.
pub const DLB: [i16; 4] = [6554, 16384, 26214, 32767];

/// RPE inverse-mantissa factor used in forward APCM quantisation (§5.2.14
/// Table 5.5). Derived from `1 / real_FAC[mant]`.
pub const NRFAC: [i16; 8] = [29128, 26215, 23832, 21846, 20165, 18725, 17476, 16384];

/// RPE weighting-filter impulse response (§5.2.13 Table 5.6), scaled so that
/// `H[i] = integer(real_H[i] * 8192)`. Symmetric: H[0..=10].
pub const H: [i16; 11] = [-134, -374, 0, 2054, 5741, 8192, 5741, 2054, 0, -374, -134];
