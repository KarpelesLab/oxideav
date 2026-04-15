//! Codec capability description.
//!
//! Each codec implementation registered with the codec registry attaches one
//! of these structs to declare what it can do, what its constraints are, and
//! how the registry should rank it against alternative implementations of
//! the same codec id.
//!
//! The flag layout mirrors ffmpeg's `-codecs` output:
//!
//! ```text
//!  D..... = Decoding supported
//!  .E.... = Encoding supported
//!  ..V... = Video codec       ..A... = Audio       ..S... = Subtitle
//!  ..D... = Data              ..T... = Attachment
//!  ...I.. = Intra-frame-only codec
//!  ....L. = Lossy compression
//!  .....S = Lossless compression
//! ```

use std::fmt;

use crate::format::MediaType;

/// Default priority for software implementations. Lower numbers are preferred
/// at resolution time, so register hardware impls with a smaller value (e.g.
/// `10`) and software fallbacks with the default `100`.
pub const DEFAULT_PRIORITY: i32 = 100;

/// What an implementation can do plus how it ranks vs alternatives.
#[derive(Clone, Debug)]
pub struct CodecCapabilities {
    pub decode: bool,
    pub encode: bool,
    pub media_type: MediaType,
    pub intra_only: bool,
    pub lossy: bool,
    pub lossless: bool,
    /// Hardware-accelerated implementation (VAAPI/NVENC/QSV/VideoToolbox/...).
    pub hardware_accelerated: bool,
    /// Short identifier for this implementation, e.g. "flac_sw", "h264_qsv".
    pub implementation: String,
    /// Restrictions — `None` means "no constraint".
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub max_bitrate: Option<u64>,
    pub max_sample_rate: Option<u32>,
    pub max_channels: Option<u16>,
    /// Lower numbers are preferred. HW impls should be ~10, SW impls ~100.
    pub priority: i32,
}

impl CodecCapabilities {
    /// Construct a software audio decoder/encoder capability set with sensible
    /// defaults — adjust fields after creation.
    pub fn audio(implementation: impl Into<String>) -> Self {
        Self {
            decode: false,
            encode: false,
            media_type: MediaType::Audio,
            intra_only: true, // audio packets are independently decodable in most codecs
            lossy: false,
            lossless: false,
            hardware_accelerated: false,
            implementation: implementation.into(),
            max_width: None,
            max_height: None,
            max_bitrate: None,
            max_sample_rate: None,
            max_channels: None,
            priority: DEFAULT_PRIORITY,
        }
    }

    pub fn video(implementation: impl Into<String>) -> Self {
        Self {
            decode: false,
            encode: false,
            media_type: MediaType::Video,
            intra_only: false,
            lossy: false,
            lossless: false,
            hardware_accelerated: false,
            implementation: implementation.into(),
            max_width: None,
            max_height: None,
            max_bitrate: None,
            max_sample_rate: None,
            max_channels: None,
            priority: DEFAULT_PRIORITY,
        }
    }

    /// 6-character ffmpeg-style flag string. Useful for `oxideav list`-style
    /// output.
    pub fn flag_string(&self) -> String {
        let mut s = String::with_capacity(6);
        s.push(if self.decode { 'D' } else { '.' });
        s.push(if self.encode { 'E' } else { '.' });
        s.push(match self.media_type {
            MediaType::Video => 'V',
            MediaType::Audio => 'A',
            MediaType::Subtitle => 'S',
            MediaType::Data => 'D',
            MediaType::Unknown => '.',
        });
        s.push(if self.intra_only { 'I' } else { '.' });
        s.push(if self.lossy { 'L' } else { '.' });
        s.push(if self.lossless { 'S' } else { '.' });
        s
    }

    // Builder-style helpers so registrations stay compact.

    pub fn with_decode(mut self) -> Self {
        self.decode = true;
        self
    }
    pub fn with_encode(mut self) -> Self {
        self.encode = true;
        self
    }
    pub fn with_intra_only(mut self, v: bool) -> Self {
        self.intra_only = v;
        self
    }
    pub fn with_lossy(mut self, v: bool) -> Self {
        self.lossy = v;
        self
    }
    pub fn with_lossless(mut self, v: bool) -> Self {
        self.lossless = v;
        self
    }
    pub fn with_hardware(mut self, v: bool) -> Self {
        self.hardware_accelerated = v;
        self
    }
    pub fn with_priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }
    pub fn with_max_size(mut self, w: u32, h: u32) -> Self {
        self.max_width = Some(w);
        self.max_height = Some(h);
        self
    }
    pub fn with_max_bitrate(mut self, br: u64) -> Self {
        self.max_bitrate = Some(br);
        self
    }
    pub fn with_max_sample_rate(mut self, sr: u32) -> Self {
        self.max_sample_rate = Some(sr);
        self
    }
    pub fn with_max_channels(mut self, ch: u16) -> Self {
        self.max_channels = Some(ch);
        self
    }
}

impl fmt::Display for CodecCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.flag_string(), self.implementation)
    }
}

/// User preferences for codec selection — pass to the registry's resolve
/// methods to bias / restrict the choice.
#[derive(Clone, Debug, Default)]
pub struct CodecPreferences {
    /// Implementation names to prefer (boost their priority by `boost`).
    pub prefer: Vec<String>,
    /// Implementation names to skip entirely.
    pub exclude: Vec<String>,
    /// Forbid hardware-accelerated impls.
    pub no_hardware: bool,
    /// Boost amount for `prefer` impls (subtracted from priority).
    pub boost: i32,
}

impl CodecPreferences {
    pub fn excludes(&self, caps: &CodecCapabilities) -> bool {
        self.exclude.iter().any(|n| n == &caps.implementation)
            || (self.no_hardware && caps.hardware_accelerated)
    }

    pub fn effective_priority(&self, caps: &CodecCapabilities) -> i32 {
        if self.prefer.iter().any(|n| n == &caps.implementation) {
            caps.priority - self.boost.max(0)
        } else {
            caps.priority
        }
    }
}
