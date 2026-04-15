//! Pure-Rust Matroska (MKV/WebM) container.
//!
//! Implements the EBML primitives plus enough of the Matroska schema to
//! demux the audio codecs oxideav already understands (FLAC, Opus, Vorbis,
//! PCM). The muxer can write back any codec we can carry — there are no
//! codec-specific assumptions in the container layer.

pub mod codec_id;
pub mod demux;
pub mod ebml;
pub mod ids;
pub mod mux;

use oxideav_container::ContainerRegistry;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("matroska", demux::open);
    reg.register_muxer("matroska", mux::open);
    reg.register_extension("mkv", "matroska");
    reg.register_extension("mka", "matroska");
    reg.register_extension("webm", "matroska");
}
