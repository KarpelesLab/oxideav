//! MPEG-1 audio frame header (ISO/IEC 11172-3 §2.4.1).
//!
//! A 32-bit frame header in big-endian bit order:
//!
//! ```text
//!  syncword          12  0xFFF
//!  ID                 1  1 = MPEG-1
//!  layer              2  `10` = Layer II
//!  protection_bit     1  0 = CRC-16 follows
//!  bitrate_index      4
//!  sampling_frequency 2
//!  padding_bit        1
//!  private_bit        1
//!  mode               2  00=stereo 01=JS 10=dual 11=mono
//!  mode_extension     2  (joint stereo only — bound subband index)
//!  copyright          1
//!  original           1
//!  emphasis           2
//! ```
//!
//! Only MPEG-1 Layer II at 32/44.1/48 kHz is handled by this decoder.

use oxideav_core::{Error, Result};

/// MPEG audio channel mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Stereo,
    JointStereo,
    DualChannel,
    Mono,
}

impl Mode {
    pub fn channels(self) -> u16 {
        match self {
            Mode::Mono => 1,
            _ => 2,
        }
    }
}

/// Parsed MPEG-1 Layer II frame header.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    /// `true` when the stream carries a 16-bit CRC immediately after the header.
    pub protection: bool,
    /// Audio bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Sampling frequency in Hz (32000, 44100 or 48000).
    pub sample_rate: u32,
    /// Padding slot present (1 additional byte for Layer II).
    pub padding: bool,
    pub mode: Mode,
    /// Index of the first intensity-stereo subband (joint stereo only).
    pub bound: u32,
}

impl Header {
    /// Total frame length in bytes, including the header.
    pub fn frame_length(&self) -> usize {
        // Layer II: frame_length = 144 * bitrate / sample_rate + padding
        let base = 144 * self.bitrate_kbps as usize * 1000 / self.sample_rate as usize;
        base + self.padding as usize
    }

    pub fn channels(&self) -> u16 {
        self.mode.channels()
    }

    /// Number of subbands that carry stereo data. Subbands from `bound` to 32
    /// are intensity-stereo coded (joint stereo only).
    pub fn sblimit(&self, allocation_table: &crate::tables::AllocTable) -> usize {
        allocation_table.sblimit
    }
}

/// MPEG-1 Layer II bitrate table (kbps). Index 0 = "free", index 15 = reserved.
const BITRATE_LAYER2_KBPS: [u32; 15] = [
    0, // free format
    32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
];

const SAMPLE_RATE_HZ: [u32; 3] = [44100, 48000, 32000];

/// Parse a 4-byte MPEG-1 Layer II frame header starting at `buf[0]`.
pub fn parse_header(buf: &[u8]) -> Result<Header> {
    if buf.len() < 4 {
        return Err(Error::invalid("mp2 header: need at least 4 bytes"));
    }
    let w = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);

    let sync = w >> 20;
    if sync != 0xFFF {
        return Err(Error::invalid("mp2 header: missing 0xFFF sync word"));
    }
    let id = (w >> 19) & 0x1; // 1 = MPEG-1
    let layer = (w >> 17) & 0x3; // 10 = Layer II
    let protection_bit = (w >> 16) & 0x1; // 0 = CRC present
    let bitrate_index = ((w >> 12) & 0xF) as usize;
    let sr_index = ((w >> 10) & 0x3) as usize;
    let padding = ((w >> 9) & 0x1) != 0;
    let mode_code = (w >> 6) & 0x3;
    let mode_ext = (w >> 4) & 0x3;

    if id != 1 {
        return Err(Error::unsupported(
            "mp2 header: MPEG-2/2.5 audio (low sample rate) not supported",
        ));
    }
    if layer != 0b10 {
        return Err(Error::unsupported(format!(
            "mp2 header: layer bits {layer:02b} — only Layer II is handled"
        )));
    }
    if bitrate_index == 0 {
        return Err(Error::unsupported("mp2 header: free-format not supported"));
    }
    if bitrate_index == 15 {
        return Err(Error::invalid("mp2 header: reserved bitrate index 15"));
    }
    if sr_index >= 3 {
        return Err(Error::invalid("mp2 header: reserved sampling index"));
    }

    let bitrate_kbps = BITRATE_LAYER2_KBPS[bitrate_index];
    let sample_rate = SAMPLE_RATE_HZ[sr_index];

    let mode = match mode_code {
        0 => Mode::Stereo,
        1 => Mode::JointStereo,
        2 => Mode::DualChannel,
        _ => Mode::Mono,
    };

    // Joint stereo: mode_extension selects the bound subband
    // (§2.4.2.3, Table 3-B.3): 00→4, 01→8, 10→12, 11→16.
    let bound = match mode {
        Mode::JointStereo => match mode_ext {
            0 => 4,
            1 => 8,
            2 => 12,
            _ => 16,
        },
        Mode::Mono => 32,
        _ => 32, // stereo / dual-channel: no intensity coding
    };

    // Validate Layer II bitrate × channel-mode combination (§2.4.2.3, Table 3-B.2).
    // Only a subset of (bitrate, mode) pairs is allowed:
    // - mono forbids bitrates ≥ 224 kbps
    // - stereo/JS/dual forbid bitrates ≤ 56 kbps (except 64 is also restricted
    //   to stereo modes — but the spec encodes this as "valid for stereo only")
    // We enforce the mono subset since it's a hard error.
    match mode {
        Mode::Mono => {
            if matches!(bitrate_kbps, 224 | 256 | 320 | 384) {
                return Err(Error::invalid(format!(
                    "mp2 header: bitrate {bitrate_kbps} kbps not permitted in single-channel mode"
                )));
            }
        }
        _ => {
            if matches!(bitrate_kbps, 32 | 48) {
                return Err(Error::invalid(format!(
                    "mp2 header: bitrate {bitrate_kbps} kbps not permitted in stereo modes"
                )));
            }
        }
    }

    let _ = protection_bit;
    Ok(Header {
        protection: protection_bit == 0,
        bitrate_kbps,
        sample_rate,
        padding,
        mode,
        bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stereo_192kbps_48k() {
        // sync=0xFFF, ID=1, layer=10, prot=1 (no CRC), bitrate=1010 (192),
        // sr=01 (48k), pad=0, priv=0, mode=00 (stereo), modeext=0, cp=0, orig=0, emph=00
        let w: u32 = 0xFFF_u32 << 20 | 1 << 19 | 0b10 << 17 | 1 << 16 | 0b1010 << 12 | 0b01 << 10;
        let bytes = w.to_be_bytes();
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.bitrate_kbps, 192);
        assert_eq!(h.sample_rate, 48000);
        assert_eq!(h.channels(), 2);
        assert_eq!(h.mode, Mode::Stereo);
        assert!(!h.protection);
        // Layer-II length at 192 kbps / 48 kHz = 144 * 192000 / 48000 = 576.
        assert_eq!(h.frame_length(), 576);
    }

    #[test]
    fn reject_layer3() {
        let w: u32 = 0xFFF_u32 << 20 | 1 << 19 | 0b01 << 17 | 1 << 16 | 0b1010 << 12 | 0b01 << 10;
        let bytes = w.to_be_bytes();
        assert!(parse_header(&bytes).is_err());
    }
}
