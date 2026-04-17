//! AV1 (AOMedia Video 1) — pure-Rust parse-only crate for oxideav.
//!
//! Status: **parse-only plus tile-decode skeleton**. Full tile decode is
//! the bulk of an AV1 decoder (~20 KLOC of CDF + transforms + intra/inter
//! prediction + loop restoration) and lands incrementally. What this
//! crate does today:
//!
//! * Walks the OBU stream (§5.3) and surfaces typed `Obu` values.
//! * Parses sequence header OBUs (§5.5) — full color config, operating
//!   points, all enable flags, optional timing / decoder model info.
//! * Parses frame header OBUs (§5.9) up to (but not including)
//!   `tile_info()` — frame type, dimensions (with superres), render size,
//!   intrabc, interpolation filter, ref frame indices, etc.
//! * Parses the AV1CodecConfigurationRecord (`av1C`) used by MP4 and
//!   Matroska, including the embedded sequence-header config OBU.
//! * Registers a `Decoder` factory that ingests OBU streams and exposes
//!   header-level state via `Av1Decoder::sequence_header()` /
//!   `last_frame_header()`.
//! * Ships the core tile-decode primitives:
//!     - Symbol (range / arithmetic) decoder (§4.10.4 + §9.3).
//!     - Inverse 4×4 and 8×8 DCT-DCT transforms (§7.7).
//!     - `DC_PRED` / `V_PRED` / `H_PRED` intra predictors (§7.11.2).
//!     - A tile-decode skeleton that initialises the symbol decoder,
//!       computes the superblock grid, and hands back a precise
//!       `Error::Unsupported` pointing at the next unimplemented clause
//!       (currently: default CDF tables for partition decode).
//!
//! `receive_frame()` still returns `Error::Unsupported(...)` — the
//! remaining gap is default CDFs + per-block mode / coefficient decode.
//! Each missing piece surfaces a message that names its §ref.
//!
//! Spec references throughout follow the **AV1 Bitstream & Decoding Process
//! Specification (2019-01-08)**: <https://aomediacodec.github.io/av1-spec/av1-spec.pdf>.

pub mod bitreader;
pub mod decoder;
pub mod extradata;
pub mod frame_header;
pub mod intra;
pub mod obu;
pub mod sequence_header;
pub mod symbol;
pub mod tile_decode;
pub mod tile_group;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 decoder factory with a codec registry.
///
/// The implementation declares `av1_sw_parse` to make it visible in
/// `oxideav list` style output that this is the parse-only build, not the
/// future full software decoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw_parse")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, decoder::make_decoder);
}

pub use decoder::{make_decoder, Av1Decoder};
pub use extradata::Av1CodecConfig;
pub use frame_header::{parse_frame_header, FrameHeader, FrameType, ParseDepth};
pub use obu::{iter_obus, parse_config_obus, parse_obu_header, read_obu, Obu, ObuHeader, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingPoint, SequenceHeader,
    TimingInfo,
};
