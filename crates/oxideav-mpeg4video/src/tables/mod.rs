//! VLC tables from ISO/IEC 14496-2 Annex B.
//!
//! Only the subset needed by the I-VOP decode path is populated in this
//! session. Placeholder modules are provided for the tables the follow-up
//! (P/B-frame + motion compensation) will need, so the layout is stable.

pub mod cbpy;
pub mod dc_size;
pub mod mcbpc;
pub mod mv;
pub mod tcoef;
pub mod vlc;
