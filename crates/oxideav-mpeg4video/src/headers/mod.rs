//! Header parsers for MPEG-4 Part 2 (ISO/IEC 14496-2 §6.2).
//!
//! Layered structure:
//! * Visual Object Sequence (VOS) — §6.2.2.1
//! * Visual Object (VO)           — §6.2.2.2
//! * Video Object Layer (VOL)     — §6.2.3
//! * Video Object Plane (VOP)     — §6.2.5 (per-frame)

pub mod vol;
pub mod vop;
pub mod vos;

pub use vol::{AspectRatioInfo, ChromaFormat, ShapeType, VideoObjectLayer};
pub use vop::{VideoObjectPlane, VopCodingType};
pub use vos::{VisualObject, VisualObjectSequence, VoType};
