//! Table B-16 / B-17 — texture coefficient VLCs (intra / inter).
//!
//! **Scaffold only** for this session. The I-VOP decoder will be wired up in
//! the follow-up patch; the full Annex B tables are ~120 entries each and
//! need careful audit. The types here are stable so the follow-up can drop
//! tables in without further plumbing.

use crate::tables::vlc::VlcEntry;

/// One decoded texture coefficient symbol.
#[derive(Clone, Copy, Debug)]
pub enum TcoefSym {
    /// `(last, run, level)` in short form — sign is in a following bit.
    RunLevel { last: bool, run: u8, level_abs: u8 },
    /// Escape codeword — decoder reads additional bits to get the actual
    /// `(last, run, level)` triple. MPEG-4 has three escape modes (see
    /// §7.4.1.3.2 types 1/2/3).
    Escape,
}

/// Returns the intra tcoef table (Table B-16). Empty stub for this session;
/// the I-VOP decoder is not yet wired to use it.
pub fn intra_table() -> &'static [VlcEntry<TcoefSym>] {
    &[]
}

/// Returns the inter tcoef table (Table B-17). Placeholder for follow-up.
pub fn inter_table() -> &'static [VlcEntry<TcoefSym>] {
    &[]
}
