//! CELT — the MDCT path of Opus (RFC 6716 §4.3) — scaffold.
//!
//! What's landed: a full RFC 6716 §4.1 range decoder (`ec_dec_bits`,
//! `ec_dec_icdf`, `ec_dec_uint`). The remaining CELT decoder stages —
//! band-energy decode, PVQ (pyramid vector quantisation), band-shape
//! reconstruction, anti-collapse, pitch prediction, MDCT, and overlap-
//! add — are a follow-up.
//!
//! The decoder is registered so the framework can detect CELT-carrying
//! streams today; `make_decoder` currently returns `Unsupported`.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::double_parens,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod range_decoder;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "celt";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("celt_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "CELT decoder is a scaffold — range decoder done; band decode + MDCT pending",
    ))
}
