//! Container traits (demuxer + muxer) and a registry.

pub mod registry;

use oxideav_core::{Packet, Result, StreamInfo};
use std::io::{Read, Seek, Write};

/// Reads a container and emits packets per stream.
pub trait Demuxer: Send {
    /// Name of the container format (e.g., `"wav"`).
    fn format_name(&self) -> &str;

    /// Streams in this container. Stable across the lifetime of the demuxer.
    fn streams(&self) -> &[StreamInfo];

    /// Read the next packet from any stream. Returns `Error::Eof` at end.
    fn next_packet(&mut self) -> Result<Packet>;

    /// Hint that only the listed stream indices will be consumed by the
    /// pipeline. Demuxers that can efficiently skip inactive streams at
    /// the container level (e.g., MKV cluster-aware, MP4 trak-aware)
    /// should override this. The default is a no-op — the pipeline
    /// drops unwanted packets on the floor.
    fn set_active_streams(&mut self, _indices: &[u32]) {}

    /// Seek to the nearest keyframe at or before `pts` (in the given
    /// stream's time base). Returns the actual timestamp seeked to, or
    /// `Error::Unsupported` if this demuxer can't seek.
    fn seek_to(&mut self, _stream_index: u32, _pts: i64) -> Result<i64> {
        Err(oxideav_core::Error::unsupported(
            "this demuxer does not support seeking",
        ))
    }
}

/// Writes packets into a container.
pub trait Muxer: Send {
    fn format_name(&self) -> &str;

    /// Write the container header. Must be called after stream configuration
    /// and before the first `write_packet`.
    fn write_header(&mut self) -> Result<()>;

    fn write_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Finalize the file (write index, patch in total sizes, etc.).
    fn write_trailer(&mut self) -> Result<()>;
}

/// Factory that tries to open a stream as a particular container format.
///
/// Implementations should read the minimum needed to confirm the format and
/// return `Error::InvalidData` if the stream is not in this format.
pub type OpenDemuxerFn = fn(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>>;

/// Factory that creates a muxer for a set of streams.
pub type OpenMuxerFn =
    fn(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>>;

/// Convenience trait bundle for seekable readers.
pub trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

/// Convenience trait bundle for seekable writers.
pub trait WriteSeek: Write + Seek + Send {}
impl<T: Write + Seek + Send> WriteSeek for T {}

pub use registry::ContainerRegistry;

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::Error;

    struct DummyDemuxer;

    impl Demuxer for DummyDemuxer {
        fn format_name(&self) -> &str {
            "dummy"
        }
        fn streams(&self) -> &[StreamInfo] {
            &[]
        }
        fn next_packet(&mut self) -> Result<Packet> {
            Err(Error::Eof)
        }
    }

    #[test]
    fn default_seek_to_is_unsupported() {
        let mut d = DummyDemuxer;
        match d.seek_to(0, 0) {
            Err(Error::Unsupported(_)) => {}
            other => panic!(
                "expected default seek_to to return Unsupported, got {:?}",
                other
            ),
        }
    }
}
