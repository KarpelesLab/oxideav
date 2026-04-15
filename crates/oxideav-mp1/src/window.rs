//! MPEG-1 Audio Layer I/II synthesis window — placeholder.
//!
//! The real 512-entry window is Table D.1 in ISO/IEC 11172-3. The
//! synthesis filter (`synthesis.rs`) is currently unreachable (decoder
//! returns `Unsupported`), so a zero-filled placeholder is adequate to
//! satisfy the compiler. Replace with the spec table when the full
//! decoder is wired up.

pub const SYNTHESIS_WINDOW: [f32; 512] = [0.0; 512];
