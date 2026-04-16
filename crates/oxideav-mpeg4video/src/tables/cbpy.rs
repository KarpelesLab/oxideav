//! Table B-9 — CBPY (coded block pattern for luma) VLC.
//!
//! Scaffold only: this table decodes the 4-bit CBPY field and is used by the
//! macroblock layer. Populated with the full table from ISO/IEC 14496-2 Annex
//! B so the I-VOP decoder can consume it. The intra path uses the "inverted"
//! reading (CBPY for intra MB is bit-inverted of the table output — §7.4.1.2).

use std::sync::OnceLock;

use crate::tables::vlc::VlcEntry;

/// (bits, code, value) — Table B-9 row 0..=15. `value` is the raw cbpy value
/// 0..=15 (not bit-inverted — the caller does that for intra MBs).
const ROWS: [(u8, u32, u8); 16] = [
    (4, 0b0011, 0),
    (5, 0b00101, 1),
    (5, 0b00100, 2),
    (4, 0b1001, 3),
    (5, 0b00011, 4),
    (4, 0b0111, 5),
    (6, 0b001010, 6),
    (4, 0b1011, 7),
    (5, 0b00010, 8),
    (6, 0b001011, 9),
    (4, 0b0101, 10),
    (4, 0b1010, 11),
    (4, 0b0100, 12),
    (4, 0b1000, 13),
    (4, 0b0110, 14),
    (2, 0b11, 15),
];

pub fn table() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        ROWS.iter()
            .map(|&(b, c, v)| VlcEntry::new(b, c, v))
            .collect()
    })
    .as_slice()
}
