//! Pure-Rust VP9 video decoder scaffold.
//!
//! Status:
//! * §6.2 uncompressed header — parsed (color_config, frame_size,
//!   render_size, loop_filter, quantization, segmentation, tile_info,
//!   header_size). Sufficient to populate `CodecParameters`
//!   (width/height/pixel_format).
//! * §9.2 boolean (range) decoder — implemented (init / boolean / literal /
//!   uniform).
//! * §6.3 compressed header — partially parsed: tx_mode, reference_mode.
//!   Coefficient / skip / inter-mode / mv probability sub-procedures are
//!   not yet decoded.
//! * §6.4 tile / partition / block decode — skeleton in `tile.rs`:
//!   initialises the bool decoder per tile, computes the tile grid /
//!   superblock count, and surfaces `Error::Unsupported` at the first
//!   partition symbol (§6.4.3) with a precise §ref.
//! * §8.5.1 intra prediction — `DC_PRED` / `V_PRED` / `H_PRED` available
//!   as primitives in `intra.rs`. Other modes (`TM_PRED`, directional)
//!   return `Error::Unsupported`.
//! * §8.7.1 inverse transforms — 4×4 and 8×8 inverse DCT-DCT available
//!   as primitives in `transform.rs`. ADST / WHT / larger sizes return
//!   `Error::Unsupported`.
//!
//! This scaffold is enough for higher layers (containers, the codec
//! registry, the CLI list output, MP4 demux) to recognise VP9 streams,
//! report stream dimensions, and surface a clean "decode not yet
//! implemented" error. The primitives mirror `oxideav-av1` — the two
//! crates are structurally parallel on purpose.
//!
//! Reference: VP9 Bitstream & Decoding Process Specification, version 0.7
//! (2017): <https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf>.

pub mod bitreader;
pub mod bool_decoder;
pub mod compressed_header;
pub mod decoder;
pub mod headers;
pub mod intra;
pub mod tile;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "vp9";

/// Register the VP9 decoder with the codec registry. The implementation
/// reports `intra_only=false` (VP9 has inter prediction) and `lossy=true`.
/// The factory returns a decoder which will currently fail with
/// `Error::Unsupported` at frame-pull time — but parses headers and
/// populates parameters successfully.
pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("vp9_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register_decoder_impl(cid, caps, decoder::make_decoder);
}

pub use compressed_header::{parse_compressed_header, CompressedHeader, ReferenceMode, TxMode};
pub use decoder::{
    codec_parameters_from_header, frame_rate_from_container, make_decoder,
    pixel_format_from_color_config, Vp9Decoder,
};
pub use headers::{
    parse_uncompressed_header, ColorConfig, ColorSpace, FrameType, LoopFilterParams,
    QuantizationParams, RefFrame, SegmentationParams, TileInfo, UncompressedHeader,
};
