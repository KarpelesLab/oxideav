//! Table B-10 — MCBPC (macroblock-type + coded-block-pattern for chroma)
//! VLC for I-VOPs. For I-VOPs the field encodes:
//! * macroblock type: Intra (value 3 / 4) vs Intra+Q (value 4 adds a
//!   dquant adjustment per §7.4.1.2);
//! * cbpc (2 bits): chroma block pattern.
//!
//! The `mb_type` / `cbpc` decomposition is performed by the caller; the VLC
//! decodes to an integer 0..=8 that the caller maps (Table B-10). Value 7 is
//! the stuffing codeword.

use std::sync::OnceLock;

use crate::tables::vlc::VlcEntry;

/// Table B-10 rows for I-VOPs. (bits, code, value).
const I_ROWS: [(u8, u32, u8); 9] = [
    (1, 0b1, 0),
    (3, 0b001, 1),
    (3, 0b010, 2),
    (3, 0b011, 3),
    (4, 0b0001, 4),
    (6, 0b00_0001, 5),
    (6, 0b00_0010, 6),
    (6, 0b00_0011, 7),
    (9, 0b000_000_001, 8), // stuffing
];

/// Intra MCBPC stuffing codeword value.
pub const STUFFING: u8 = 8;

pub fn i_table() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        I_ROWS
            .iter()
            .map(|&(b, c, v)| VlcEntry::new(b, c, v))
            .collect()
    })
    .as_slice()
}
