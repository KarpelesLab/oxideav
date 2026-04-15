//! Vorbis audio codec.
//!
//! This crate currently provides the codec **identifier** (`vorbis`) and a
//! parser for the Vorbis Identification header (the first packet of every
//! Vorbis logical bitstream in Ogg). A full bit-exact decoder — codebooks,
//! floors, residues, MDCT — is a substantial follow-up project and is not yet
//! implemented; building a decoder still produces an `Error::Unsupported`.
//!
//! Even without decoding, registering this codec lets the framework:
//! - identify a Vorbis stream by id (e.g. for `oxideav probe` output),
//! - cleanly remux Vorbis streams across containers (no decode required).

pub mod identification;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "vorbis";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    reg.register_decoder(cid.clone(), make_decoder);
    reg.register_encoder(cid, make_encoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "Vorbis decoder not yet implemented in pure Rust — only identification + remux are supported today",
    ))
}

fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::unsupported(
        "Vorbis encoder not yet implemented in pure Rust",
    ))
}

pub use identification::{parse_identification_header, Identification};
