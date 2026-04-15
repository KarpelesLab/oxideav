//! Pure-Rust MP4 / ISO Base Media File Format container.
//!
//! Scope: demuxer for probe + remux of audio and video tracks. A full muxer
//! is a separate piece of work (requires buffering or two-pass writing since
//! `moov` depends on the final sample tables).

pub mod boxes;
pub mod codec_id;
pub mod demux;

use oxideav_container::ContainerRegistry;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("mp4", demux::open);
    reg.register_extension("mp4", "mp4");
    reg.register_extension("m4a", "mp4");
    reg.register_extension("m4v", "mp4");
    reg.register_extension("mov", "mp4");
    reg.register_extension("3gp", "mp4");
}
