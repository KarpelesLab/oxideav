//! ADTS (Audio Data Transport Stream) frame header parser.
//!
//! ADTS is the typical streaming wrapper for AAC outside of MP4. Each frame
//! begins with a 7- or 9-byte header (ISO/IEC 13818-7 §6.2 and ISO/IEC
//! 14496-3 §1.A.2) followed by 1..=4 raw_data_blocks of payload.
//!
//! The header layout (no CRC, 7 bytes):
//! ```text
//! syncword              12 bits  (0xFFF)
//! id                     1 bit   (0=MPEG-4, 1=MPEG-2)
//! layer                  2 bits  (always 0)
//! protection_absent      1 bit   (1 = no CRC follows)
//! profile                2 bits  (0=Main, 1=LC, 2=SSR, 3=LTP)
//! sampling_freq_index    4 bits
//! private_bit            1 bit
//! channel_configuration  3 bits
//! original_copy          1 bit
//! home                   1 bit
//! copyright_id_bit       1 bit
//! copyright_id_start     1 bit
//! aac_frame_length      13 bits
//! adts_buffer_fullness  11 bits
//! number_of_raw_blocks   2 bits  (0 = single block per ADTS frame)
//! ```

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::syntax::{sample_rate, AOT_AAC_LC, AOT_AAC_LTP, AOT_AAC_MAIN, AOT_AAC_SSR};

/// Length of the ADTS fixed header in bytes when no CRC is present.
pub const ADTS_HEADER_NO_CRC: usize = 7;
/// Length of the ADTS fixed header in bytes when CRC is present.
pub const ADTS_HEADER_WITH_CRC: usize = 9;

/// Parsed ADTS fixed/variable header.
#[derive(Clone, Debug)]
pub struct AdtsHeader {
    pub mpeg_version: u8,
    pub protection_absent: bool,
    /// AAC object type / profile (1-based — `2` is AAC-LC).
    pub object_type: u8,
    pub sampling_freq_index: u8,
    pub channel_configuration: u8,
    /// Total ADTS frame size including the header itself.
    pub frame_length: usize,
    /// Number of raw_data_blocks in this ADTS frame minus one (i.e. 0..=3
    /// where 0 means the most common single-block layout).
    pub number_of_raw_blocks_minus_one: u8,
}

impl AdtsHeader {
    /// Sample rate in Hz; `None` if the index is reserved.
    pub fn sample_rate(&self) -> Option<u32> {
        sample_rate(self.sampling_freq_index)
    }

    /// Length of just the ADTS header bytes.
    pub fn header_length(&self) -> usize {
        if self.protection_absent {
            ADTS_HEADER_NO_CRC
        } else {
            ADTS_HEADER_WITH_CRC
        }
    }

    /// Length of the raw payload (frame_length minus the header).
    pub fn payload_length(&self) -> usize {
        self.frame_length.saturating_sub(self.header_length())
    }
}

/// Parse a single ADTS header from `data`.
///
/// Requires at least 7 bytes — returns `Error::NeedMore` otherwise.
pub fn parse_adts_header(data: &[u8]) -> Result<AdtsHeader> {
    if data.len() < ADTS_HEADER_NO_CRC {
        return Err(Error::NeedMore);
    }
    let mut br = BitReader::new(data);
    let sync = br.read_u32(12)?;
    if sync != 0xFFF {
        return Err(Error::invalid("ADTS: missing 0xFFF syncword"));
    }
    let id = br.read_u32(1)? as u8;
    let layer = br.read_u32(2)?;
    if layer != 0 {
        return Err(Error::invalid("ADTS: layer must be 0"));
    }
    let protection_absent = br.read_bit()?;
    let profile = br.read_u32(2)? as u8; // 0..=3
    let object_type = profile + 1; // ADTS profile is object_type-1
    let sampling_freq_index = br.read_u32(4)? as u8;
    let _private_bit = br.read_bit()?;
    let channel_configuration = br.read_u32(3)? as u8;
    let _original_copy = br.read_bit()?;
    let _home = br.read_bit()?;
    let _copyright_id_bit = br.read_bit()?;
    let _copyright_id_start = br.read_bit()?;
    let frame_length = br.read_u32(13)? as usize;
    let _adts_buffer_fullness = br.read_u32(11)?;
    let number_of_raw_blocks_minus_one = br.read_u32(2)? as u8;

    if frame_length < ADTS_HEADER_NO_CRC {
        return Err(Error::invalid("ADTS: frame_length shorter than header"));
    }

    // Validate object type
    if !matches!(
        object_type,
        AOT_AAC_MAIN | AOT_AAC_LC | AOT_AAC_SSR | AOT_AAC_LTP
    ) {
        return Err(Error::invalid("ADTS: invalid object_type"));
    }

    Ok(AdtsHeader {
        mpeg_version: id,
        protection_absent,
        object_type,
        sampling_freq_index,
        channel_configuration,
        frame_length,
        number_of_raw_blocks_minus_one,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_lc_mono() {
        // syncword=0xFFF, id=0, layer=0, protection_absent=1
        // profile=01 (LC), sf_idx=0100 (44100), priv=0, chan_cfg=001
        // pad bits, frame_length=0x100 (256), buf_fullness=0x7FF, blocks=0
        let mut br = [0u8; 7];
        // 0..1 byte 0: 0xFF
        br[0] = 0xFF;
        // bits 12: sync done; next bits: id(0), layer(00), prot(1) = 0xF1 then profile(01)..
        // Easiest: build manually with a writer.
        // 0xFF F1 -> sync(12)+id(0)+layer(00)+prot(1) — but high 12 of u16 = FFF1 >> 4 = 0xFFF, then 0001 next nibble.
        br[1] = 0xF1; // 1111 0001 — last 4 bits = 0001 = id(0)+layer(00)+prot(1)
                      // Next byte: profile(01)+sf_idx(0100)+priv(0)+chan_cfg high bits...
                      // bits: 01 0100 0 0 -> 0101 0000 = 0x50 — but channel_cfg has 3 bits so we span.
                      // Let's just use a precomputed real frame from our fixture.
        let bytes = std::fs::read("/tmp/aac_lc_mono.aac").expect("fixture missing");
        let hdr = parse_adts_header(&bytes).expect("parse hdr");
        assert_eq!(hdr.object_type, AOT_AAC_LC);
        assert_eq!(hdr.channel_configuration, 1);
        assert_eq!(hdr.sample_rate(), Some(44_100));
    }

    #[test]
    fn rejects_bad_sync() {
        let bad = [0u8; 7];
        assert!(parse_adts_header(&bad).is_err());
    }

    #[test]
    fn needs_more_when_short() {
        let short = [0xFFu8, 0xF1, 0x50];
        assert!(matches!(parse_adts_header(&short), Err(Error::NeedMore)));
    }
}
