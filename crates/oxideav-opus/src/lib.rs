#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::useless_vec,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::nonminimal_bool,
    clippy::manual_range_contains,
    clippy::needless_late_init,
    clippy::needless_return,
    clippy::let_unit_value,
    clippy::needless_borrow,
    unused_mut,
    unused_variables,
    unused_assignments,
    clippy::unnecessary_cast,
    clippy::manual_memcpy,
    clippy::neg_multiply,
    clippy::precedence
)]

//! Opus audio codec (RFC 6716 bitstream, RFC 7845 in-Ogg mapping).
//!
//! What's landed:
//!
//! * `OpusHead` identification-packet parsing (RFC 7845 §5.1).
//! * Full TOC byte + framing code 0/1/2/3 packet parser (RFC 6716 §3).
//! * Decoder that produces correct silence output for DTX / silence
//!   frames and for CELT frames whose silence flag is set.
//! * Clean `Unsupported` rejection (no panics, no garbage) for SILK-only
//!   and Hybrid frames, and for CELT frames that require the full
//!   band-energy + PVQ + inverse MDCT stack (not yet landed).
//!
//! Scope that remains for follow-up agents: the CELT decoder stages
//! (coarse+fine+final band energy, PVQ shape decode, anti-collapse,
//! inverse MDCT with overlap-add window, post-filter); and the SILK
//! decoder entirely. All of these are tracked in RFC 6716 §4.2 / §4.3.

pub mod decoder;
pub mod header;
pub mod toc;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "opus";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::audio("opus_sw")
        .with_lossy(true)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_both(cid, caps, make_decoder, make_encoder);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Err(Error::unsupported(
        "Opus encoder not yet implemented in pure Rust",
    ))
}

pub use header::{parse_opus_head, OpusHead};
pub use toc::{parse_packet, OpusBandwidth, OpusMode, OpusPacket, Toc};
