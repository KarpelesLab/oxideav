//! MPEG-1/2 Audio Layer III (MP3) codec — scaffold.
//!
//! What's landed: MSB-first bit reader and a frame-header parser that
//! handles all MPEG-1/2/2.5 version/layer/bitrate/samplerate/channel-mode
//! combinations. The full decoder (side info, bit reservoir, Huffman
//! tables, scalefactor decode, requantisation, stereo processing,
//! antialias, IMDCT, hybrid filterbank, polyphase synthesis) is a
//! multi-session follow-up.
//!
//! The decoder is registered so the framework can probe/remux MP3
//! streams today; `make_decoder` currently returns `Unsupported`.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod frame;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "mp3";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("mp3_sw")
        .with_lossy(true)
        .with_intra_only(false) // MP3 uses a bit reservoir — not intra-only
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "MP3 decoder is a scaffold — Huffman, requantisation, IMDCT, and hybrid filter bank pending",
    ))
}
