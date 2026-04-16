// Parallel-array index loops read more naturally for block/IDCT code.
#![allow(clippy::needless_range_loop)]

//! Theora video codec (Xiph On2-VP3 descendant).
//!
//! Status:
//! * Header parsing (identification + comment + setup) — done.
//! * I-frame decode — done; bit-exact IDCT, integer-domain dequant, full
//!   loop filter.
//! * P-frame (inter) decode — done; macroblock modes, motion vectors
//!   (LAST/LAST2/golden), 4-MV mode, half-pel motion compensation,
//!   per-RFI DC prediction, and golden-frame reference are all wired up.

pub mod bitreader;
pub mod block;
pub mod coded_order;
pub mod dct;
pub mod decoder;
pub mod headers;
pub mod huffman;
pub mod inter;
pub mod quant;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "theora";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("theora_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    reg.register_decoder_impl(cid, caps, make_decoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

/// Public factory intended for tests / ad-hoc integration.
pub fn make_decoder_for_tests(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

pub use decoder::{classify_packet, codec_parameters_from_identification, FrameType, PacketKind};
pub use headers::{
    parse_comment_header, parse_headers_from_extradata, parse_identification_header,
    parse_setup_header, parse_xiph_extradata, Comment, Headers, Identification, PixelFormat, Setup,
};
