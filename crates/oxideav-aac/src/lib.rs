//! AAC (Advanced Audio Coding, AAC-LC profile) codec — scaffold.
//!
//! What's landed: MSB-first bit reader aligned with the ISO/IEC 14496-3
//! specification. The ADTS frame parser, AudioSpecificConfig parser,
//! Huffman code tables, TNS / M/S / intensity stereo, scalefactor
//! decoding, spectral-data unpacking, and inverse MDCT are a follow-up.
//!
//! The decoder is registered so the framework can probe/remux AAC
//! streams today (MP4/ADTS carriage is already supported); `make_decoder`
//! currently returns `Unsupported`.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "aac";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("aac_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_channels(8)
        .with_max_sample_rate(96_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "AAC-LC decoder is a scaffold — ADTS parsing, ASC, Huffman tables, and IMDCT pending",
    ))
}
