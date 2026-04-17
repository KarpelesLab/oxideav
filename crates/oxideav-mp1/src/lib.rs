//! MPEG-1 Audio Layer I (MP1) codec.
//!
//! Decoder: packet-in / `AudioFrame`-out, covering:
//!
//! - MPEG-1 sample rates: 32 000, 44 100, and 48 000 Hz.
//! - All Layer I channel modes: single-channel, stereo, dual-channel,
//!   joint-stereo (bound-based sample sharing; Layer I has no intensity
//!   stereo scaling — §2.4.2.3).
//! - Bit-allocation codes 0..14 (15 is forbidden and rejected).
//! - 6-bit scalefactor indices into the `SCALE[64]` table
//!   (ISO/IEC 11172-3 Table 3-B.1).
//! - 32-band polyphase synthesis filter per Annex B / Annex D.
//!
//! Encoder: minimum-viable CBR Layer I output, mirror of the decode path.
//! No psychoacoustic model — greedy energy-per-bit allocation. Mono or
//! plain stereo only, no joint stereo, no CRC. See [`encoder`].
//!
//! Not in scope (either direction): CRC verification, free-format frames
//! (bitrate index 0), intensity-stereo scaling.
//!
//! See [`decoder::make_decoder`] / [`encoder::make_encoder`] for entry
//! points and [`crate::bitalloc`] for the requantization math.

#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::too_many_arguments,
    clippy::doc_overindented_list_items
)]

pub mod analysis;
pub mod bitalloc;
pub mod bitreader;
pub mod bitwriter;
pub mod decoder;
pub mod encoder;
pub mod header;
pub mod synthesis;
pub mod window;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "mp1";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("mp1_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_both(
        CodecId::new(CODEC_ID_STR),
        caps,
        make_decoder,
        make_encoder,
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}
