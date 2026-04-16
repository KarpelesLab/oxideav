//! Tables B-12 / B-13 — DCT DC size VLCs, separate tables for luminance and
//! chrominance. Decoded value is the DC size (0..=11 in MPEG-1; effectively
//! 0..=8 for 8-bit video). The call site then reads `size` further bits as the
//! sign-extended DC differential.
//!
//! Code/length pairs are taken from libavcodec/mpeg12data.c
//! (`ff_mpeg12_vlc_dc_lum_{code,bits}` and `ff_mpeg12_vlc_dc_chroma_{code,bits}`),
//! which matches ISO/IEC 11172-2 Tables B-12/B-13.

use crate::vlc::VlcEntry;

const LUMA_CODE: [u32; 12] = [
    0x4, 0x0, 0x1, 0x5, 0x6, 0xe, 0x1e, 0x3e, 0x7e, 0xfe, 0x1fe, 0x1ff,
];
const LUMA_BITS: [u8; 12] = [3, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9];

const CHROMA_CODE: [u32; 12] = [
    0x0, 0x1, 0x2, 0x6, 0xe, 0x1e, 0x3e, 0x7e, 0xfe, 0x1fe, 0x3fe, 0x3ff,
];
const CHROMA_BITS: [u8; 12] = [2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10];

pub fn luma() -> &'static [VlcEntry<u8>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        (0..12)
            .map(|i| VlcEntry::new(LUMA_BITS[i], LUMA_CODE[i], i as u8))
            .collect()
    })
    .as_slice()
}

pub fn chroma() -> &'static [VlcEntry<u8>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        (0..12)
            .map(|i| VlcEntry::new(CHROMA_BITS[i], CHROMA_CODE[i], i as u8))
            .collect()
    })
    .as_slice()
}
