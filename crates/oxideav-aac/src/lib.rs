//! AAC-LC decoder — ISO/IEC 14496-3 / ISO 13818-7.
//!
//! Implemented:
//! - ADTS frame header parser (§1.A.2 / 13818-7 §6.2)
//! - AudioSpecificConfig parser (§1.6.2.1)
//! - Huffman codebooks 1-11 + scalefactor (§4.A.1)
//! - SCE/CPE syntax (§4.6.2/§4.6.3)
//! - Section data + scalefactor reconstruction (§4.6.2.3)
//! - Inverse quantisation x = sign·|q|^(4/3) (§4.6.6)
//! - M/S stereo (§4.6.13)
//! - IMDCT 2048/256 with sine + KBD windows + overlap-add (§4.6.18 / §4.6.11)
//! - LongStart / LongStop / EightShort window sequences
//! - TNS bit-skip (no synthesis), pulse data flag check, fill / DSE elements
//!
//! Not implemented (returns `Error::Unsupported` or stubbed to zeros):
//! - Pulse data (§4.6.5), TNS synthesis (§4.6.9), gain control (§4.6.12)
//! - Intensity stereo (§4.6.14) — bands marked IS leave zeros
//! - PNS / perceptual noise substitution (§4.6.10) — ditto
//! - LFE / CCE / PCE elements
//! - HE-AAC SBR (§4.6.18.4) / PS — return Unsupported when detected
//! - Main / SSR / LTP profiles (§4.6.7-8) — only AAC-LC accepted

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::manual_memcpy,
    clippy::too_many_arguments,
    clippy::if_same_then_else
)]

pub mod adts;
pub mod asc;
pub mod bitreader;
pub mod decoder;
pub mod huffman;
pub mod huffman_tables;
pub mod ics;
pub mod imdct;
pub mod sfband;
pub mod syntax;
pub mod synth;
pub mod window;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "aac";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("aac_sw")
        .with_lossy(true)
        .with_intra_only(true)
        // We currently decode mono and stereo only; multi-channel returns
        // `Error::Unsupported`. The cap value is what we *advertise* to the
        // registry — keep at 2 until we wire 5.1.
        .with_max_channels(2)
        .with_max_sample_rate(96_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}
