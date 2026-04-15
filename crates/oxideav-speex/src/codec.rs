//! Speex codec registration.

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("speex_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(2)
        .with_max_sample_rate(32_000);
    reg.register_decoder_impl(CodecId::new(super::CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    super::decoder::make_decoder(params)
}
