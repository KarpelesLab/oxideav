//! MPEG-1 Audio Layer II (MP2 / MUSICAM) codec — scaffold.
//!
//! What's landed: MSB-first bit reader and a `parse_header` for the
//! 32-bit MPEG audio frame header. The full Layer II decoder (bit
//! allocation tables B.2a–d, scalefactor/SCFSI decode, grouped-sample
//! unpack, polyphase subband synthesis) is a follow-up.
//!
//! The decoder is registered so the framework can probe/remux MP2
//! streams today; `make_decoder` currently returns `Unsupported`.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod header;
pub mod tables;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "mp2";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("mp2_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "MP2 decoder is a scaffold — bit allocation, scalefactors, and synthesis filter bank are pending",
    ))
}
