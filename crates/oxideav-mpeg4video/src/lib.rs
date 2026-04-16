//! Pure-Rust MPEG-4 Part 2 video (ISO/IEC 14496-2) decoder.
//!
//! Scope, current session:
//! * Visual Object Sequence / Visual Object / Video Object Layer / Video Object
//!   Plane header parsing for Advanced Simple Profile (ASP) levels 1-5.
//! * `CodecParameters` population from the VOL header (width, height, frame rate).
//! * I-VOP macroblock decode with AC/DC prediction + H.263 dequantisation +
//!   IDCT — scaffolded; registration returns a decoder that reports
//!   `Unsupported` for inter VOPs.
//!
//! Explicitly out of scope for this session (decoder reports `Unsupported`):
//! * P- and B-VOPs (motion compensation)
//! * GMC / sprites / S-VOPs
//! * Quarter-pel MC
//! * Interlaced video / field coding
//! * Data partitioning, reversible VLCs, resync-based error recovery
//! * MPEG-4 Studio / AVC Simple profiles
//! * Encoder
//!
//! The crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bitreader;
pub mod block;
pub mod decoder;
pub mod headers;
pub mod iq;
pub mod mb;
pub mod start_codes;
pub mod tables;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

/// The canonical oxideav codec id for MPEG-4 Part 2 video.
///
/// Note: this matches the ISO standard name. Container-level FourCCs like
/// `XVID`, `DIVX`, `DX50`, `MP4V`, `FMP4` are all this codec.
pub const CODEC_ID_STR: &str = "mpeg4video";

/// Register this decoder with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mpeg4video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(4096, 4096);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, decoder::make_decoder);
}
