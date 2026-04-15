//! Pure-Rust Ogg container (RFC 3533).
//!
//! Implements the page layer (capture pattern, segment table, CRC32) and a
//! packet-reassembly demuxer / packet-splitting muxer. Codec-specific parsing
//! lives in dedicated crates (`oxideav-vorbis`, future `oxideav-opus`, …);
//! this crate only sniffs the first packet of each logical bitstream to set
//! `CodecParameters::codec_id` correctly so the registry can dispatch.

pub mod codec_id;
pub mod crc;
pub mod demux;
pub mod mux;
pub mod page;

use oxideav_container::ContainerRegistry;

/// Register the Ogg demuxer/muxer with a [`ContainerRegistry`].
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("ogg", demux::open);
    reg.register_muxer("ogg", mux::open);
    reg.register_extension("ogg", "ogg");
    reg.register_extension("oga", "ogg");
    reg.register_extension("ogv", "ogg");
    reg.register_extension("opus", "ogg");
}
