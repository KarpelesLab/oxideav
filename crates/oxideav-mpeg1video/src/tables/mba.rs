//! Table B-1 — macroblock_address_increment VLC.
//!
//! Reproduced from libavcodec/mpeg12data.c `ff_mpeg12_mbAddrIncrTable`.

use crate::vlc::VlcEntry;

/// `macroblock_escape`. Adds 33 to the address.
pub const ESCAPE: u8 = 0xFF;
/// `macroblock_stuffing` (MPEG-1 only). Ignored.
pub const STUFFING: u8 = 0xFE;

// Parallel `(code, bits)` arrays with values 1..=33 then escape and stuffing.
const CODE: [u32; 35] = [
    0x1, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x7, 0x6, 0xb, 0xa, 0x9, 0x8, 0x7, 0x6, 0x17, 0x16, 0x15,
    0x14, 0x13, 0x12, 0x23, 0x22, 0x21, 0x20, 0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x8,
    0xf,
];
const BITS: [u8; 35] = [
    1, 3, 3, 4, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11,
];

pub fn table() -> &'static [VlcEntry<u8>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut v = Vec::with_capacity(35);
        for i in 0..33 {
            v.push(VlcEntry::new(BITS[i], CODE[i], (i + 1) as u8));
        }
        v.push(VlcEntry::new(BITS[33], CODE[33], ESCAPE));
        v.push(VlcEntry::new(BITS[34], CODE[34], STUFFING));
        v
    })
    .as_slice()
}
