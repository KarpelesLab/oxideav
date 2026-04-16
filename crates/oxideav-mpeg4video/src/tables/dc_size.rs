//! Table B-12 / B-13 — intra DC size VLCs for luma and chroma in MPEG-4.
//!
//! These tables match the H.263 DC-size tables (and ISO/IEC 14496-2 Tables
//! B-12 / B-13). The decoded value is the number of additional bits that
//! encode the DC differential.
//!
//! Source: ISO/IEC 14496-2 Annex B, cross-checked against FFmpeg's
//! libavcodec/mpeg4videodec.c `ff_mpeg4_dc_lum` / `ff_mpeg4_dc_chrom`.

use std::sync::OnceLock;

use crate::tables::vlc::VlcEntry;

// Luma: (code, bits) pairs for DC sizes 0..=12.
const LUMA_CODE: [u32; 13] = [3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const LUMA_BITS: [u8; 13] = [3, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11];
// Chroma: sizes 0..=12.
const CHROMA_CODE: [u32; 13] = [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const CHROMA_BITS: [u8; 13] = [2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

pub fn luma() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        (0..LUMA_CODE.len())
            .map(|i| VlcEntry::new(LUMA_BITS[i], LUMA_CODE[i], i as u8))
            .collect()
    })
    .as_slice()
}

pub fn chroma() -> &'static [VlcEntry<u8>] {
    static CELL: OnceLock<Vec<VlcEntry<u8>>> = OnceLock::new();
    CELL.get_or_init(|| {
        (0..CHROMA_CODE.len())
            .map(|i| VlcEntry::new(CHROMA_BITS[i], CHROMA_CODE[i], i as u8))
            .collect()
    })
    .as_slice()
}
