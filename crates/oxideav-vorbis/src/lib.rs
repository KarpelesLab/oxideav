// Parallel-array index loops are idiomatic in audio codec code; let clippy
// nag elsewhere.
#![allow(clippy::needless_range_loop)]

//! Vorbis audio codec.
//!
//! Decoder is feature-complete for the common q3-q10 file shapes: matches
//! libvorbis / lewton output within float rounding on the test fixtures.
//!
//! Encoder is tier 2 — emits long blocks only, mono / stereo, with
//! sum/difference channel coupling, ATH-scaled floor1, and a 128-entry
//! residue VQ. Output decodes through both our own decoder and ffmpeg's
//! libvorbis. Quality is well above the 100× Goertzel-ratio acceptance
//! bar on synthesised tones (mono ~14000×, stereo ~8000×). Sample sizes
//! land at 3-7× libvorbis @ 128 kbps for tones; closer for noise. See
//! `encoder.rs` module-level doc for the deferred items (transient
//! detection, point-stereo, libvorbis-Annex-B reference books).

pub mod audio_packet;
pub mod bitreader;
pub mod bitwriter;
pub mod codebook;
pub mod dbtable;
pub mod decoder;
pub mod encoder;
pub mod floor;
pub mod identification;
pub mod imdct;
pub mod libvorbis_setup;
pub mod residue;
pub mod setup;
pub mod setup_writer;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "vorbis";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::audio("vorbis_sw")
        .with_lossy(true)
        .with_max_channels(255);
    reg.register_decoder_impl(cid.clone(), caps.clone(), make_decoder);
    reg.register_encoder_impl(cid, caps, make_encoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

pub use identification::{parse_identification_header, Identification};
