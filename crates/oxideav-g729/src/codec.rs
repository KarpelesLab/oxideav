//! G.729 codec registration.
//!
//! G.729 produces 10 ms frames of mono 8 kHz PCM — 80 `S16` samples per
//! frame, coded into 80 bits (10 bytes). The bitstream is lossy.

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

use crate::{CODEC_ID_STR, SAMPLE_RATE};

/// Register the G.729 decoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("g729_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(SAMPLE_RATE);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    crate::decoder::make_decoder(params)
}
