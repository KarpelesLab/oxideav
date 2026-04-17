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

    /// Container-level metadata as ordered (key, value) pairs.
    /// Keys follow a loose convention borrowed from Vorbis comments:
    /// `title`, `artist`, `album`, `comment`, `date`, `sample_name:<n>`,
    /// `channels`, `n_patterns`, etc. Demuxers that carry no metadata
    /// return an empty slice (the default).
    fn metadata(&self) -> &[(String, String)] {
        &[]
    }
    /// Container-level duration, if known. Default is `None` — callers
    /// may fall back to the longest per-stream duration. Expressed as
    /// microseconds for portability; convert to seconds at the edge.
    fn duration_micros(&self) -> Option<i64> {
        None
    }

    /// Attached pictures (cover art, artist photos, ...) embedded in
    /// the container. Returns an empty slice (the default) when the
    /// container carries none or doesn't support them. Containers that
    /// do — ID3v2 on MP3, `METADATA_BLOCK_PICTURE` on FLAC, `covr`
    /// atoms on MP4, etc. — override this to expose the images.
    fn attached_pictures(&self) -> &[oxideav_core::AttachedPicture] {
        &[]
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

/// Information passed to a content-based [`ProbeFn`].
///
/// `buf` holds the first few KB of the input — enough to recognise the
/// magic bytes of any container we know about. `ext` carries the file
/// extension as a hint (lowercase, no leading dot); some containers
/// (raw MP3 with no ID3v2, headerless tracker formats) need it to break
/// ties with otherwise weak signatures.
pub struct ProbeData<'a> {
    pub buf: &'a [u8],
    pub ext: Option<&'a str>,
}

/// Confidence score returned by a [`ProbeFn`]. `0` means no match.
/// Higher means more certain. Conventional values:
///
/// * `100` – unambiguous magic bytes at a known offset
/// * `75`  – signature match corroborated by file extension
/// * `50`  – signature match without extension corroboration
/// * `25`  – extension match only (no content signature available)
pub type ProbeScore = u8;

/// Maximum probe score (alias for `100`).
pub const MAX_PROBE_SCORE: ProbeScore = 100;
/// Default score returned when only the file extension matches.
pub const PROBE_SCORE_EXTENSION: ProbeScore = 25;

/// Content-based format detection function.
///
/// Returns a [`ProbeScore`] in `0..=100`. Implementations should be
/// pure (no I/O, no allocation beyond the stack) and fast — they may
/// be invoked once per registered demuxer on every input file.
pub type ProbeFn = fn(probe: &ProbeData) -> ProbeScore;

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
