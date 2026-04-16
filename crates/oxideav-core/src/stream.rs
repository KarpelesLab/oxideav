//! Stream metadata shared between containers and codecs.

use crate::format::{MediaType, PixelFormat, SampleFormat};
use crate::rational::Rational;
use crate::time::TimeBase;

/// A stable identifier for a codec. Codec crates register a `CodecId` so the
/// codec registry can look them up by name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CodecId(pub String);

impl CodecId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CodecId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::fmt::Display for CodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Codec-level parameters shared between demuxer/muxer and en/decoder.
#[derive(Clone, Debug)]
pub struct CodecParameters {
    pub codec_id: CodecId,
    pub media_type: MediaType,

    // Audio-specific
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub sample_format: Option<SampleFormat>,

    // Video-specific
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub pixel_format: Option<PixelFormat>,
    pub frame_rate: Option<Rational>,

    /// Per-codec setup bytes (e.g., SPS/PPS, OpusHead). Format defined by codec.
    pub extradata: Vec<u8>,

    pub bit_rate: Option<u64>,
}

impl CodecParameters {
    pub fn audio(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            media_type: MediaType::Audio,
            sample_rate: None,
            channels: None,
            sample_format: None,
            width: None,
            height: None,
            pixel_format: None,
            frame_rate: None,
            extradata: Vec::new(),
            bit_rate: None,
        }
    }

    /// True when `self` and `other` have the same codec_id and core
    /// format parameters (sample_rate/channels/sample_format for audio,
    /// width/height/pixel_format for video). Extradata and bitrate
    /// differences are tolerated — many containers rewrite extradata
    /// losslessly during a copy operation.
    pub fn matches_core(&self, other: &CodecParameters) -> bool {
        self.codec_id == other.codec_id
            && self.sample_rate == other.sample_rate
            && self.channels == other.channels
            && self.sample_format == other.sample_format
            && self.width == other.width
            && self.height == other.height
            && self.pixel_format == other.pixel_format
    }

    pub fn video(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            media_type: MediaType::Video,
            sample_rate: None,
            channels: None,
            sample_format: None,
            width: None,
            height: None,
            pixel_format: None,
            frame_rate: None,
            extradata: Vec::new(),
            bit_rate: None,
        }
    }
}

/// Description of a single stream inside a container.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub index: u32,
    pub time_base: TimeBase,
    pub duration: Option<i64>,
    pub start_time: Option<i64>,
    pub params: CodecParameters,
}
