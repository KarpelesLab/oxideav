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
