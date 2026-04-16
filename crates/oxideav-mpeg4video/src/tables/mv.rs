//! Table B-13 — motion vector VLC. Used by P- and B-VOPs (out of scope for
//! this session). Scaffold only so the module layout stabilises.

use crate::tables::vlc::VlcEntry;

/// Returns the motion VLC table. Empty stub for this session; the P-VOP
/// decoder lands in a follow-up session.
pub fn table() -> &'static [VlcEntry<i8>] {
    &[]
}
