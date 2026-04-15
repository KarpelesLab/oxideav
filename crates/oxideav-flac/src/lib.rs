//! FLAC support: native container + codec identifier.
//!
//! - The **container** parses `fLaC` magic + metadata blocks and emits one
//!   packet per FLAC frame, scanning sync codes to find frame boundaries.
//! - The **codec** registers id `flac`. The decoder is not yet implemented;
//!   today this lets you probe and remux FLAC files. A pure-Rust subframe
//!   decoder is a substantial follow-up.

pub mod codec;
pub mod container;
pub mod metadata;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;

pub const CODEC_ID_STR: &str = "flac";

pub fn register_codecs(reg: &mut CodecRegistry) {
    codec::register(reg);
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}
