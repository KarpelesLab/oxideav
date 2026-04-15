//! Opus audio codec (RFC 6716 bitstream, RFC 7845 in-Ogg mapping).
//!
//! Current scope: codec id + `OpusHead` parsing. Full decoder (SILK linear
//! prediction + CELT MDCT) is a substantial multi-session project and is
//! not yet implemented — building a decoder today returns `Unsupported`,
//! but identification and remuxing work end-to-end through `oxideav-ogg`.

pub mod header;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "opus";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::audio("opus_sw")
        .with_lossy(true)
        .with_max_channels(255)
        .with_max_sample_rate(48_000);
    // Both factories return Unsupported for now — registration just lets
    // probe + remux work; transcode through this codec will error.
    reg.register_both(cid, caps, make_decoder, make_encoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "Opus decoder not yet implemented in pure Rust — identification + remux are supported today",
    ))
}

fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::unsupported(
        "Opus encoder not yet implemented in pure Rust",
    ))
}

pub use header::{parse_opus_head, OpusHead};
