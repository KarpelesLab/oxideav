//! Amiga ProTracker / SoundTracker module ("MOD") support.
//!
//! MOD files are self-contained song data: a 20-byte title, 31 sample
//! descriptors, a pattern order list, a 4-character signature that
//! identifies the channel count, 64×N-channel patterns, then raw signed
//! 8-bit sample bodies.
//!
//! This crate registers:
//!
//! - A **container** (`mod`) that slurps the entire file and emits it as
//!   a single packet. The "packets" abstraction isn't natural for MOD —
//!   playback is driven by song position + effect state, not per-packet
//!   decode — so the container just delivers the bytes to the codec.
//! - A **codec** (`mod`) whose decoder parses the header + pattern +
//!   sample data and emits mixed stereo PCM. This initial version ships
//!   the full header parser and a stub decoder that reports correct
//!   duration but outputs silence; Paula channel emulation, effects, and
//!   sample mixing follow in a dedicated session.
//!
//! Per-channel (instrument) output is planned: see
//! `MEMORY.md → MOD multichannel` for the architectural sketch.

pub mod container;
pub mod decoder;
pub mod header;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;

pub const CODEC_ID_STR: &str = "mod";

pub fn register_codecs(reg: &mut CodecRegistry) {
    decoder::register(reg);
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}
