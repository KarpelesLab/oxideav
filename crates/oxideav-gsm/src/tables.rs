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
