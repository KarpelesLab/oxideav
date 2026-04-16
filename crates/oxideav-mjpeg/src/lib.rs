// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! Baseline JPEG / Motion-JPEG codec, pure Rust.
//!
//! Each video packet is a standalone JPEG (one full SOI..EOI). The decoder
//! recognises baseline (SOF0) 8-bit JPEGs with 4:2:0 / 4:2:2 / 4:4:4 chroma
//! subsampling and outputs `VideoFrame`s in the corresponding `Yuv*P` pixel
//! format (or `Gray8` for 1-component streams). The encoder accepts the
//! same pixel formats and produces a standalone JPEG using the Annex K
//! "typical" Huffman tables, so its output is interoperable with any
//! compliant JPEG decoder.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Progressive JPEG (SOF2), hierarchical, arithmetic coding, lossless
//! - 12-bit precision
//! - CMYK / 4-component scans
//! - Non-interleaved scans (one component per SOS segment)

pub mod decoder;
pub mod encoder;
pub mod jpeg;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "mjpeg";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("mjpeg_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(16384, 16384);
    reg.register_both(cid, caps, decoder::make_decoder, encoder::make_encoder);
}
