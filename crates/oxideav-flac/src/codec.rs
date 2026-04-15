//! FLAC codec registration. Decoder is forthcoming.

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecId, CodecParameters, Error, Result};

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(super::CODEC_ID_STR);
    reg.register_decoder(cid.clone(), make_decoder);
    reg.register_encoder(cid, make_encoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "FLAC subframe decoder not yet implemented in pure Rust — only probe + remux today",
    ))
}

fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::unsupported(
        "FLAC encoder not yet implemented in pure Rust",
    ))
}
