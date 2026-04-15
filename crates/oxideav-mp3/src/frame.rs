//! MPEG audio frame header parsing (Layer III focus).
//!
//! Spec: ISO/IEC 11172-3 §2.4.1 (MPEG-1) and ISO/IEC 13818-3 §2.4.1 (MPEG-2).
//!
//! Byte layout:
//! ```text
//!   AAAAAAAA AAABBCCD EEEEFFGH IIJJKLMM
//! ```
//! - A: 11-bit sync word (all ones)
//! - B: MPEG audio version id (2 bits)
//! - C: Layer description (2 bits) — Layer III = `01`
//! - D: Protection bit (1 bit, 0 → CRC-16 follows header)
//! - E: Bitrate index (4 bits)
//! - F: Sampling rate index (2 bits)
//! - G: Padding bit
//! - H: Private bit
//! - I: Channel mode (2 bits)
//! - J: Mode extension (2 bits — joint-stereo specific)
//! - K: Copyright
//! - L: Original
//! - M: Emphasis (2 bits)

use oxideav_core::{Error, Result};

/// MPEG audio version. Layer III is defined for all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpegVersion {
    /// MPEG-1 (ISO/IEC 11172-3) — sample rates 32/44.1/48 kHz.
    Mpeg1,
    /// MPEG-2 (ISO/IEC 13818-3 LSF) — sample rates 16/22.05/24 kHz.
    Mpeg2,
    /// MPEG-2.5 (unofficial Fraunhofer extension) — sample rates 8/11.025/12 kHz.
    Mpeg25,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMode {
    Stereo,
    JointStereo,
    DualChannel,
    Mono,
}

impl ChannelMode {
    pub fn channel_count(self) -> u16 {
        match self {
            Self::Mono => 1,
            _ => 2,
        }
    }
}

/// Layer description — only Layer III is decoded here. The header parser
/// accepts Layer I / II tags only for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Layer1,
    Layer2,
    Layer3,
}

/// Parsed MPEG audio frame header (all 32 bits worth of fields).
#[derive(Clone, Copy, Debug)]
pub struct FrameHeader {
    pub version: MpegVersion,
    pub layer: Layer,
    /// True if no CRC-16 follows the header (the spec's "protection" bit
    /// is inverted — 1 means no CRC).
    pub no_crc: bool,
    /// Raw bitrate index (0..=15). Index 0 = "free format", 15 = forbidden.
    pub bitrate_index: u8,
    /// Bitrate in kbps (0 for free-format).
    pub bitrate_kbps: u32,
    /// Raw sample-rate index (0..=3); 3 is forbidden.
    pub sample_rate_index: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    pub padding: bool,
    pub private_bit: bool,
    pub channel_mode: ChannelMode,
    /// Raw 2-bit mode extension (joint-stereo flags).
    pub mode_extension: u8,
    pub copyright: bool,
    pub original: bool,
    /// Raw 2-bit emphasis code.
    pub emphasis: u8,
}

impl FrameHeader {
    pub fn channels(&self) -> u16 {
        self.channel_mode.channel_count()
    }

    /// Number of PCM samples produced per channel per Layer III frame.
    /// MPEG-1 = 1152; MPEG-2/2.5 = 576 (one granule).
    pub fn samples_per_frame(&self) -> u32 {
        match self.version {
            MpegVersion::Mpeg1 => 1152,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
        }
    }

    /// Length of the side-information block in bytes.
    /// MPEG-1: 32 bytes (stereo) or 17 (mono).
    /// MPEG-2/2.5: 17 bytes (stereo) or 9 (mono).
    pub fn side_info_bytes(&self) -> usize {
        match (self.version, self.channel_mode) {
            (MpegVersion::Mpeg1, ChannelMode::Mono) => 17,
            (MpegVersion::Mpeg1, _) => 32,
            (_, ChannelMode::Mono) => 9,
            _ => 17,
        }
    }

    /// Total encoded frame length in bytes (header + optional CRC + side-info
    /// + main data for *this* frame). Returns `None` for free-format streams
    /// (bitrate_index == 0) since the length must be computed by sync search.
    pub fn frame_bytes(&self) -> Option<u32> {
        if self.bitrate_kbps == 0 {
            return None;
        }
        // Standard length formula, e.g. minimp3's hdr_frame_bytes():
        //   MPEG-1 L3: 144 * bitrate / sample_rate + padding
        //   MPEG-2 L3:  72 * bitrate / sample_rate + padding
        let numerator = match self.version {
            MpegVersion::Mpeg1 => 144u32,
            _ => 72u32,
        };
        let len = numerator * self.bitrate_kbps * 1000 / self.sample_rate;
        Some(len + if self.padding { 1 } else { 0 })
    }
}

/// Parse the first 4 bytes of `bytes` as an MPEG audio Layer III frame header.
/// Returns `Error::NeedMore` if the slice is shorter than 4 bytes,
/// `Error::InvalidData` otherwise.
pub fn parse_frame_header(bytes: &[u8]) -> Result<FrameHeader> {
    if bytes.len() < 4 {
        return Err(Error::NeedMore);
    }
    let h: u32 = (bytes[0] as u32) << 24
        | (bytes[1] as u32) << 16
        | (bytes[2] as u32) << 8
        | (bytes[3] as u32);
    parse_frame_header_u32(h)
}

/// Parse a frame header directly from the packed 32-bit big-endian value.
pub fn parse_frame_header_u32(h: u32) -> Result<FrameHeader> {
    // 11-bit sync. MPEG-1/2 officially use 0xFFE (11 ones). MPEG-2.5
    // reuses bits 20..=19 as a version marker (0 → 2.5), so we accept
    // 0xFFE here and branch on the version bits below.
    if (h & 0xFFE0_0000) != 0xFFE0_0000 {
        return Err(Error::invalid("MP3 frame: bad sync"));
    }
    let version_bits = (h >> 19) & 0x3;
    let version = match version_bits {
        0b00 => MpegVersion::Mpeg25,
        0b01 => return Err(Error::invalid("MP3 frame: reserved version bits")),
        0b10 => MpegVersion::Mpeg2,
        0b11 => MpegVersion::Mpeg1,
        _ => unreachable!(),
    };

    let layer_bits = (h >> 17) & 0x3;
    let layer = match layer_bits {
        0b00 => return Err(Error::invalid("MP3 frame: reserved layer bits")),
        0b01 => Layer::Layer3,
        0b10 => Layer::Layer2,
        0b11 => Layer::Layer1,
        _ => unreachable!(),
    };
    // We decode Layer III only; surface anything else early so the decoder
    // can reject the packet cleanly.
    if layer != Layer::Layer3 {
        return Err(Error::unsupported(
            "MP3 decoder: only Layer III is implemented",
        ));
    }

    let no_crc = ((h >> 16) & 0x1) != 0;
    let bitrate_index = ((h >> 12) & 0xF) as u8;
    let sample_rate_index = ((h >> 10) & 0x3) as u8;
    let padding = ((h >> 9) & 0x1) != 0;
    let private_bit = ((h >> 8) & 0x1) != 0;
    let mode_bits = ((h >> 6) & 0x3) as u8;
    let mode_extension = ((h >> 4) & 0x3) as u8;
    let copyright = ((h >> 3) & 0x1) != 0;
    let original = ((h >> 2) & 0x1) != 0;
    let emphasis = (h & 0x3) as u8;

    if sample_rate_index == 3 {
        return Err(Error::invalid("MP3 frame: reserved sample-rate index"));
    }
    if bitrate_index == 15 {
        return Err(Error::invalid("MP3 frame: forbidden bitrate index"));
    }

    let bitrate_kbps = lookup_bitrate(version, layer, bitrate_index)?;
    let sample_rate = lookup_sample_rate(version, sample_rate_index);

    let channel_mode = match mode_bits {
        0b00 => ChannelMode::Stereo,
        0b01 => ChannelMode::JointStereo,
        0b10 => ChannelMode::DualChannel,
        0b11 => ChannelMode::Mono,
        _ => unreachable!(),
    };

    Ok(FrameHeader {
        version,
        layer,
        no_crc,
        bitrate_index,
        bitrate_kbps,
        sample_rate_index,
        sample_rate,
        padding,
        private_bit,
        channel_mode,
        mode_extension,
        copyright,
        original,
        emphasis,
    })
}

/// Bitrate table, in kbps. Index 0 = free format (returned as 0). Index 15 is
/// forbidden and rejected before we get here.
///
/// Layout: `[version_group][layer_group][index]`. Per the spec:
/// - MPEG-1: Layer I, II, III have distinct tables.
/// - MPEG-2/2.5: Layer I uses one table; Layer II & III share another.
#[rustfmt::skip]
const BITRATE_TABLE: [[[u32; 16]; 3]; 2] = [
    // version_group 0: MPEG-1
    [
        // Layer I
        [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0],
        // Layer II
        [0, 32, 48, 56,  64,  80,  96, 112, 128, 160, 192, 224, 256, 320, 384, 0],
        // Layer III
        [0, 32, 40, 48,  56,  64,  80,  96, 112, 128, 160, 192, 224, 256, 320, 0],
    ],
    // version_group 1: MPEG-2 / MPEG-2.5
    [
        // Layer I
        [0, 32, 48, 56,  64,  80,  96, 112, 128, 144, 160, 176, 192, 224, 256, 0],
        // Layer II  (== Layer III)
        [0,  8, 16, 24,  32,  40,  48,  56,  64,  80,  96, 112, 128, 144, 160, 0],
        // Layer III
        [0,  8, 16, 24,  32,  40,  48,  56,  64,  80,  96, 112, 128, 144, 160, 0],
    ],
];

fn lookup_bitrate(version: MpegVersion, layer: Layer, index: u8) -> Result<u32> {
    let vg = match version {
        MpegVersion::Mpeg1 => 0,
        MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 1,
    };
    let lg = match layer {
        Layer::Layer1 => 0,
        Layer::Layer2 => 1,
        Layer::Layer3 => 2,
    };
    Ok(BITRATE_TABLE[vg][lg][index as usize])
}

/// Sample-rate lookup. MPEG-2 values are half MPEG-1; MPEG-2.5 is quarter.
fn lookup_sample_rate(version: MpegVersion, index: u8) -> u32 {
    const BASE: [u32; 4] = [44_100, 48_000, 32_000, 0];
    let b = BASE[index as usize & 0x3];
    match version {
        MpegVersion::Mpeg1 => b,
        MpegVersion::Mpeg2 => b / 2,
        MpegVersion::Mpeg25 => b / 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mpeg1_layer3_stereo_128k_44100() {
        // 11111111 11111011 10010000 00000000
        //          sync(+v1+L3+noCRC) 128kbps 44.1k noPad stereo
        // version = 11 (MPEG-1), layer = 01 (Layer III), protection = 1 (no CRC)
        // bitrate_index = 1001 (= 128 kbps), sr_index = 00 (44100), pad = 0
        // mode = 00 (stereo)
        let hdr = parse_frame_header(&[0xFF, 0xFB, 0x90, 0x00]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg1);
        assert_eq!(hdr.layer, Layer::Layer3);
        assert!(hdr.no_crc);
        assert_eq!(hdr.bitrate_kbps, 128);
        assert_eq!(hdr.sample_rate, 44_100);
        assert_eq!(hdr.channel_mode, ChannelMode::Stereo);
        assert_eq!(hdr.channels(), 2);
        assert_eq!(hdr.samples_per_frame(), 1152);
        // 144 * 128_000 / 44_100 = 417 bytes
        assert_eq!(hdr.frame_bytes(), Some(417));
        assert_eq!(hdr.side_info_bytes(), 32);
    }

    #[test]
    fn parse_mpeg2_layer3_mono() {
        // 11111111 11110011 10010000 11000000
        // version = 10 (MPEG-2), layer = 01 (Layer III), proto = 1
        // bitrate_index = 1001 → MPEG-2 L3 = 80 kbps, sr_index = 00 → 22050
        // mode = 11 (mono)
        let hdr = parse_frame_header(&[0xFF, 0xF3, 0x90, 0xC0]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.bitrate_kbps, 80);
        assert_eq!(hdr.sample_rate, 22_050);
        assert_eq!(hdr.channel_mode, ChannelMode::Mono);
        assert_eq!(hdr.samples_per_frame(), 576);
        assert_eq!(hdr.side_info_bytes(), 9);
    }

    #[test]
    fn rejects_bad_sync() {
        let err = parse_frame_header(&[0xFE, 0xFB, 0x90, 0x00]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_layer2() {
        // Layer II: layer_bits = 10
        let err = parse_frame_header(&[0xFF, 0xFD, 0x90, 0x00]).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn rejects_forbidden_bitrate() {
        // bitrate_index = 1111
        let err = parse_frame_header(&[0xFF, 0xFB, 0xF0, 0x00]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_reserved_sample_rate() {
        // sr_index = 11
        let err = parse_frame_header(&[0xFF, 0xFB, 0x9C, 0x00]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
