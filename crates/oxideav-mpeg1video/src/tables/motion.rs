//! Table B-10 — motion_code VLC.
//!
//! The spec uses a symmetric ±16 alphabet. For |value| = 0, code is "1".
//! For |value| > 0, the codeword of the positive symbol is followed by a
//! 1-bit sign (0 = +, 1 = –). This module returns the absolute magnitude; the
//! caller must read the sign bit when |value| > 0.
//!
//! Code/bit pairs from libavcodec/mpeg12data.c `ff_mpeg12_mbMotionVectorTable`.

use crate::vlc::VlcEntry;

const CODE: [u32; 17] = [
    0x1, 0x1, 0x1, 0x1, 0x3, 0x5, 0x4, 0x3, 0xb, 0xa, 0x9, 0x11, 0x10, 0xf, 0xe, 0xd, 0xc,
];
const BITS: [u8; 17] = [1, 2, 3, 4, 6, 7, 7, 7, 9, 9, 9, 10, 10, 10, 10, 10, 10];

pub fn table() -> &'static [VlcEntry<u8>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        (0..17)
            .map(|i| VlcEntry::new(BITS[i], CODE[i], i as u8))
            .collect()
    })
    .as_slice()
}
