//! Pure-Rust MPEG-4 Part 2 video (ISO/IEC 14496-2) decoder.
//!
//! Scope:
//! * VOS / Visual Object / Video Object Layer / Video Object Plane header
//!   parsing for Advanced Simple Profile (ASP) levels 1-5.
//! * `CodecParameters` population from the VOL.
//! * **I-VOP** decode — AC/DC prediction + H.263 / MPEG-4 dequantisation
//!   + IDCT.
//! * **P-VOP** decode — half-pel motion compensation, single-MV mode (4MV
//!   path is implemented but rarely triggered by typical encoders), inter
//!   texture reconstruction, MV-median prediction with first-slice-line
//!   special cases, and skipped-MB pass-through.
//! * Video-packet resync markers (§6.3.5.2) — detect-and-consume with
//!   forward-MB-num validation to avoid false positives.
//! * One reference frame held in the decoder; refreshed by each
//!   I-VOP/P-VOP.
//!
//! Out of scope (returns `Unsupported`):
//! * B-VOPs (bidirectional prediction).
//! * S-VOPs (sprites), GMC.
//! * Quarter-pel motion (`quarter_sample` rejected at VOL parse time).
//! * Interlaced field coding, scalability, data partitioning, reversible
//!   VLCs.
//! * MPEG-4 Studio / AVC Simple profiles.
//! * Encoder: I-VOPs only — P / B / S VOPs are out of scope (§6.2.5).
//!
//! The crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bitreader;
pub mod bitwriter;
pub mod block;
pub mod decoder;
pub mod encoder;
pub mod headers;
pub mod inter;
pub mod iq;
pub mod mb;
pub mod mc;
pub mod pvop;
pub mod resync;
pub mod start_codes;
pub mod tables;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

/// The canonical oxideav codec id for MPEG-4 Part 2 video.
///
/// Note: this matches the ISO standard name. Container-level FourCCs like
/// `XVID`, `DIVX`, `DX50`, `MP4V`, `FMP4` are all this codec.
pub const CODEC_ID_STR: &str = "mpeg4video";

/// Register this codec's decoder + I/P-VOP encoder with a registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mpeg4video_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(4096, 4096);
    let id = CodecId::new(CODEC_ID_STR);
    reg.register_decoder_impl(id.clone(), caps.clone(), decoder::make_decoder);
    // Encoder produces I-VOPs + P-VOPs; advertise non-intra-only.
    reg.register_encoder_impl(id, caps, encoder::make_encoder);
}
