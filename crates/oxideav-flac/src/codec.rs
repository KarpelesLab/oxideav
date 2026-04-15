//! FLAC codec registration.

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(super::CODEC_ID_STR);
    let caps = CodecCapabilities::audio("flac_sw")
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_channels(8)
        .with_max_sample_rate(655_350);
    reg.register_both(cid, caps, make_decoder, make_encoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    super::decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    super::encoder::make_encoder(params)
}
