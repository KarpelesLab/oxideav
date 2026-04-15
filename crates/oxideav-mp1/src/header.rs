//! MPEG-1 Audio Layer I frame header parsing.
//!
//! Reference: ISO/IEC 11172-3:1993 §2.4.2.3.
//!
//! The 32-bit header layout (most-significant bit first):
//!
//! ```text
//!  Bits  Field
//!  12    sync word (0xFFF)
//!   1    ID  (1 = MPEG-1, 0 = reserved in strict MPEG-1; MPEG-2 extensions
//!            reuse this bit but Layer I on MPEG-2 is rare — we accept both
//!            but only compute frame sizes / tables for MPEG-1)
//!   2    layer (11 = Layer I, 10 = II, 01 = III)
//!   1    protection (0 = CRC present, 1 = absent)
//!   4    bitrate_index (0 = free, 15 = forbidden)
//!   2    sampling_frequency (00 = 44100, 01 = 48000, 10 = 32000, 11 = reserved)
//!   1    padding_bit
//!   1    private_bit
//!   2    mode (00 stereo, 01 joint_stereo, 10 dual_channel, 11 single_channel)
//!   2    mode_extension
//!   1    copyright
//!   1    original
//!   2    emphasis
//! ```

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// MPEG-1 Layer I bitrates (kbps), indexed by `bitrate_index`. Index 0 = free
/// format (variable), index 15 = forbidden.
pub const BITRATES_KBPS: [Option<u32>; 16] = [
    None,
    Some(32),
    Some(64),
    Some(96),
    Some(128),
    Some(160),
    Some(192),
    Some(224),
    Some(256),
    Some(288),
    Some(320),
    Some(352),
    Some(384),
    Some(416),
    Some(448),
    None,
];

/// MPEG-1 sampling frequencies (Hz), indexed by the `sampling_frequency` field.
pub const SAMPLE_RATES: [Option<u32>; 4] = [Some(44_100), Some(48_000), Some(32_000), None];

/// Layer identifier (bits per spec: 11=I, 10=II, 01=III, 00=reserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    LayerI,
    LayerII,
    LayerIII,
}

/// Channel mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMode {
    Stereo,
    JointStereo,
    DualChannel,
    SingleChannel,
}

impl ChannelMode {
    pub fn channel_count(self) -> u16 {
        match self {
            ChannelMode::SingleChannel => 1,
            _ => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emphasis {
    None,
    Ms5015,
    Reserved,
    CcitJ17,
}

#[derive(Clone, Debug)]
pub struct FrameHeader {
    /// True for MPEG-1, false for MPEG-2 LSF / MPEG-2.5 variants. This crate
    /// is MPEG-1 Layer I only; MPEG-2 Layer I is accepted for sync but the
    /// frame-size computation below assumes MPEG-1 sample-rate rules.
    pub mpeg1: bool,
    pub layer: Layer,
    pub protection: bool, // true = CRC present (protection bit == 0)
    pub bitrate_index: u8,
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    pub padding: bool,
    pub private: bool,
    pub mode: ChannelMode,
    pub mode_extension: u8,
    pub copyright: bool,
    pub original: bool,
    pub emphasis: Emphasis,
}

impl FrameHeader {
    /// Parse the 32-bit header from `bytes` (must be at least 4 bytes).
    pub fn parse(bytes: &[u8]) -> Result<FrameHeader> {
        if bytes.len() < 4 {
            return Err(Error::invalid("MP1 header: need 4 bytes"));
        }
        let mut br = BitReader::new(&bytes[..4]);
        let sync = br.read_u32(11)?;
        if sync != 0x7FF {
            return Err(Error::invalid(format!(
                "MP1 header: bad sync word {sync:#05x}"
            )));
        }
        // Next bit: version id (1 = MPEG-1, 0 = MPEG-2 LSF or reserved)
        let id_bit = br.read_u32(1)?;
        let mpeg1 = id_bit == 1;

        let layer_bits = br.read_u32(2)?;
        let layer = match layer_bits {
            0b11 => Layer::LayerI,
            0b10 => Layer::LayerII,
            0b01 => Layer::LayerIII,
            _ => return Err(Error::invalid("MP1 header: reserved layer bits")),
        };
        if layer != Layer::LayerI {
            return Err(Error::unsupported(format!(
                "MP1 header: layer {layer:?} not handled by this decoder"
            )));
        }
        let protection_bit = br.read_u32(1)?;
        // In the spec the protection_bit is 0 when CRC is present.
        let protection = protection_bit == 0;

        let bitrate_index = br.read_u32(4)? as u8;
        let bitrate_kbps = BITRATES_KBPS[bitrate_index as usize].ok_or_else(|| {
            Error::invalid(format!(
                "MP1 header: unsupported bitrate_index {bitrate_index}"
            ))
        })?;

        let sfreq_idx = br.read_u32(2)? as usize;
        let sample_rate = SAMPLE_RATES[sfreq_idx]
            .ok_or_else(|| Error::invalid("MP1 header: reserved sampling_frequency"))?;

        let padding = br.read_bit()?;
        let private = br.read_bit()?;
        let mode_bits = br.read_u32(2)?;
        let mode = match mode_bits {
            0b00 => ChannelMode::Stereo,
            0b01 => ChannelMode::JointStereo,
            0b10 => ChannelMode::DualChannel,
            0b11 => ChannelMode::SingleChannel,
            _ => unreachable!(),
        };
        let mode_extension = br.read_u32(2)? as u8;
        let copyright = br.read_bit()?;
        let original = br.read_bit()?;
        let emphasis_bits = br.read_u32(2)?;
        let emphasis = match emphasis_bits {
            0b00 => Emphasis::None,
            0b01 => Emphasis::Ms5015,
            0b10 => Emphasis::Reserved,
            0b11 => Emphasis::CcitJ17,
            _ => unreachable!(),
        };

        Ok(FrameHeader {
            mpeg1,
            layer,
            protection,
            bitrate_index,
            bitrate_kbps,
            sample_rate,
            padding,
            private,
            mode,
            mode_extension,
            copyright,
            original,
            emphasis,
        })
    }

    /// Layer I frame size in bytes, including header and optional CRC.
    ///
    /// Formula (spec §2.4.3.1): `frame_size = (12 * bitrate / sample_rate + pad) * 4`.
    pub fn frame_size(&self) -> usize {
        let bitrate_bps = self.bitrate_kbps * 1000;
        let pad = if self.padding { 1 } else { 0 };
        let slots = 12 * bitrate_bps / self.sample_rate + pad;
        (slots * 4) as usize
    }

    /// 384 samples per channel per Layer I frame.
    pub const fn samples_per_frame(&self) -> u32 {
        384
    }

    /// Bound-index into the Layer I allocation table. For non-joint-stereo
    /// this is 32 (all subbands use both channels). For joint stereo, the
    /// mode_extension field selects a lower bound where channels 0/1 share
    /// samples.
    pub fn bound(&self) -> u8 {
        if self.mode != ChannelMode::JointStereo {
            32
        } else {
            // §2.4.2.3 Table: mode_extension 0..=3 → bound 4, 8, 12, 16.
            match self.mode_extension {
                0 => 4,
                1 => 8,
                2 => 12,
                3 => 16,
                _ => 32,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_typical_header() {
        // Build a plausible Layer I header: sync, MPEG-1, Layer I, no CRC,
        // bitrate index 4 (128 kbps), sample rate 00 (44.1 kHz),
        // no pad, private 0, stereo 00, mode_ext 00, copyright 0, original 1,
        // emphasis 00.
        //
        // Bits:  1111 1111 1111 (sync 12) 1 (id) 11 (layer I) 1 (protection=no CRC)
        //        0100 (bitrate) 00 (fs=44.1) 0 (pad) 0 (private) 00 (stereo)
        //        00 (mode_ext) 0 (copyright) 1 (original) 00 (emphasis)
        //
        // Packing:
        // byte0 = 11111111
        // byte1 = 1111_1 111     ← sync + id = 13 bits of 1, then layer=11 → 1111_1111
        // Let me reconstruct carefully:
        //
        // bits 0..7:   11111111 (FF)
        // bits 8..15:  1111 1 11 1 = FF (sync 12 ones, id 1, layer 11, protection 1)
        // bits 16..23: 0100 00 0 0 = 0100_0000 = 0x40
        // bits 24..31: 00 00 0 1 00 = 0000_0100 = 0x04
        let bytes = [0xFF, 0xFF, 0x40, 0x04];
        let h = FrameHeader::parse(&bytes).expect("parse");
        assert_eq!(h.layer, Layer::LayerI);
        assert!(h.mpeg1);
        assert_eq!(h.bitrate_kbps, 128);
        assert_eq!(h.sample_rate, 44_100);
        assert_eq!(h.mode, ChannelMode::Stereo);
        assert_eq!(h.emphasis, Emphasis::None);
        assert!(!h.protection); // protection_bit=1 → no CRC
        assert!(h.original);
        // Frame size: (12*128000/44100 + 0) * 4 = 34 * 4 = 136 (floored div)
        assert_eq!(h.frame_size(), 136);
    }

    #[test]
    fn rejects_bad_sync() {
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert!(FrameHeader::parse(&bytes).is_err());
    }

    #[test]
    fn frame_size_padded() {
        // 32kbps @ 32kHz, pad=1 → (12*32000/32000 + 1) * 4 = 13 * 4 = 52
        // Build: sync, id=1, layer=11, protection=1, bitrate_idx=1, sfreq=10 (32k),
        //        pad=1, private=0, mode=00, mode_ext=00, cr=0, orig=0, emph=00.
        //
        // bits: 11111111 11111 1 11 1 0001 10 1 0 00 00 0 0 00
        //       = 11111111 11111111 00011010 00000000
        //       = 0xFF 0xFF 0x1A 0x00
        let bytes = [0xFF, 0xFF, 0x1A, 0x00];
        let h = FrameHeader::parse(&bytes).expect("parse");
        assert_eq!(h.sample_rate, 32_000);
        assert_eq!(h.bitrate_kbps, 32);
        assert!(h.padding);
        assert_eq!(h.frame_size(), 52);
    }

    #[test]
    fn joint_stereo_bound() {
        // Same as typical header but mode=01 (joint stereo), mode_ext=10 → bound 12.
        // Alter bits 24..31: mode=01, mode_ext=10 → 01 10 0 1 00 = 0110_0100 = 0x64
        let bytes = [0xFF, 0xFF, 0x40, 0x64];
        let h = FrameHeader::parse(&bytes).expect("parse");
        assert_eq!(h.mode, ChannelMode::JointStereo);
        assert_eq!(h.mode_extension, 2);
        assert_eq!(h.bound(), 12);
    }
}
