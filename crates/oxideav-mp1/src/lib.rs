//! MPEG-1 Audio Layer I (MP1) codec — scaffold.
//!
//! What's landed: MSB-first bit reader, frame-header parser (bitrate /
//! sample-rate / channel-mode / emphasis tables), and an initial
//! `SynthesisState` skeleton for the 32-band polyphase filter bank.
//! The full decoder body (bit-allocation table, scalefactor decode,
//! subband sample reconstruction, synthesis filter pass) is a follow-up.
//!
//! The decoder is registered so the framework can probe/remux MP1
//! streams today; `make_decoder` currently returns `Unsupported`.

// Scaffold-only — symbols will be used once the full decoder body lands.
// Lint allows come off when the decoder is exercised.
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod header;
pub mod synthesis;
pub mod window;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "mp1";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("mp1_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "MP1 decoder is a scaffold — bit-allocation + subband synthesis pending",
    ))
}
