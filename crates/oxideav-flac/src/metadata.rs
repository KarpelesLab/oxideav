//! FLAC metadata block parsing (FLAC format spec, §METADATA_BLOCK).

use oxideav_core::{Error, Result};

pub const FLAC_MAGIC: [u8; 4] = *b"fLaC";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    StreamInfo = 0,
    Padding = 1,
    Application = 2,
    SeekTable = 3,
    VorbisComment = 4,
    CueSheet = 5,
    Picture = 6,
    Reserved(u8),
    Invalid,
}

impl BlockType {
    pub fn from_byte(b: u8) -> Self {
        match b & 0x7F {
            0 => Self::StreamInfo,
            1 => Self::Padding,
            2 => Self::Application,
            3 => Self::SeekTable,
            4 => Self::VorbisComment,
            5 => Self::CueSheet,
            6 => Self::Picture,
            127 => Self::Invalid,
            other => Self::Reserved(other),
        }
    }
}

/// Header of a single metadata block.
#[derive(Clone, Debug)]
pub struct BlockHeader {
    pub last: bool,
    pub block_type: BlockType,
    pub length: u32,
}

impl BlockHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(Error::NeedMore);
        }
        let b0 = bytes[0];
        let last = b0 & 0x80 != 0;
        let block_type = BlockType::from_byte(b0);
        let length = ((bytes[1] as u32) << 16) | ((bytes[2] as u32) << 8) | (bytes[3] as u32);
        Ok(Self {
            last,
            block_type,
            length,
        })
    }
}

/// Parsed STREAMINFO block contents.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

impl StreamInfo {
    /// Parse a STREAMINFO block. The block payload is exactly 34 bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 34 {
            return Err(Error::invalid("FLAC STREAMINFO block must be 34 bytes"));
        }
        let min_block_size = u16::from_be_bytes([bytes[0], bytes[1]]);
        let max_block_size = u16::from_be_bytes([bytes[2], bytes[3]]);
        let min_frame_size =
            ((bytes[4] as u32) << 16) | ((bytes[5] as u32) << 8) | (bytes[6] as u32);
        let max_frame_size =
            ((bytes[7] as u32) << 16) | ((bytes[8] as u32) << 8) | (bytes[9] as u32);
        // Bytes 10..18: 64 packed bits — sample_rate(20), channels-1(3), bps-1(5), total_samples(36).
        let packed = u64::from_be_bytes(bytes[10..18].try_into().expect("8 bytes"));
        let sample_rate = ((packed >> 44) & 0x000F_FFFF) as u32;
        let channels = (((packed >> 41) & 0x07) as u8) + 1;
        let bits_per_sample = (((packed >> 36) & 0x1F) as u8) + 1;
        let total_samples = packed & 0x0000_000F_FFFF_FFFF;
        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&bytes[18..34]);
        Ok(Self {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_streaminfo_round_trip() {
        // Build a STREAMINFO block by hand: sr=48000, ch=2, bps=16, total=96000.
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&4096u16.to_be_bytes()); // min block size
        block[2..4].copy_from_slice(&4096u16.to_be_bytes()); // max block size
        // min/max frame size: 0 (unknown) — leave zeros.
        // packed: sample_rate(20)=48000, channels-1(3)=1, bps-1(5)=15, total(36)=96000
        let packed: u64 = (48_000u64 << 44) | (1u64 << 41) | (15u64 << 36) | 96_000u64;
        block[10..18].copy_from_slice(&packed.to_be_bytes());
        // md5 left as zeros.
        let info = StreamInfo::parse(&block).unwrap();
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.total_samples, 96_000);
    }

    #[test]
    fn block_header_decodes_last_flag() {
        let h = BlockHeader::parse(&[0x80, 0x00, 0x00, 0x22]).unwrap();
        assert!(h.last);
        assert_eq!(h.block_type, BlockType::StreamInfo);
        assert_eq!(h.length, 34);
    }
}
