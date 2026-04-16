//! Pure-Rust AVI (RIFF/AVI) container: demuxer + muxer.
//!
//! AVI is Microsoft's legacy RIFF-based container, still ubiquitous for
//! Motion-JPEG output from security cameras and older capture hardware. This
//! crate parses and emits AVI 1.0 files. OpenDML extensions (`ix##`,
//! super-index, files > 2 GiB) are explicitly out of scope — see
//! `muxer::write_packet` which returns an error if the output approaches
//! 2 GiB.

pub mod codec_map;
pub mod demuxer;
pub mod muxer;
pub mod riff;
pub mod stream_format;

use oxideav_container::ContainerRegistry;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("avi", demuxer::open);
    reg.register_muxer("avi", muxer::open);
    reg.register_extension("avi", "avi");
}
