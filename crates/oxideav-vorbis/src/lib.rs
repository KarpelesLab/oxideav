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

pub mod audio_packet;
pub mod bitreader;
pub mod codebook;
pub mod dbtable;
pub mod decoder;
pub mod floor;
pub mod identification;
pub mod imdct;
pub mod residue;
pub mod setup;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "vorbis";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::audio("vorbis_sw")
        .with_lossy(true)
        .with_max_channels(255);
    // Decoder is in active development — register it so probing through the
    // pipeline works; init may still error on unsupported config.
    reg.register_decoder_impl(cid, caps, make_decoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

pub use identification::{parse_identification_header, Identification};
