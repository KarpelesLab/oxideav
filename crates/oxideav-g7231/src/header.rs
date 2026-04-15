//! G.723.1 frame header (Annex B packet framing).
//!
//! Every 30 ms frame begins with a 2-bit rate/type discriminator that
//! determines the total frame size:
//!
//! ```text
//!   00  high rate   — 6.3 kbit/s MP-MLQ       → 24 bytes (189 bits)
//!   01  low  rate   — 5.3 kbit/s ACELP        → 20 bytes (158 bits)
//!   10  SID frame   — silence insertion        →  4 bytes
//!   11  untransmitted / erasure / frame-loss   →  0 bytes (no payload)
//! ```
//!
//! Reference: ITU-T G.723.1 (May 2006) §5.4 / Annex B Table B.1.

use oxideav_core::{Error, Result};

/// Rate/type discriminator for a G.723.1 packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// 6.3 kbit/s MP-MLQ, 24-byte payload.
    HighRate,
    /// 5.3 kbit/s ACELP, 20-byte payload.
    LowRate,
    /// Silence-insertion descriptor, 4-byte payload.
    SidFrame,
    /// Untransmitted / erasure — no payload after the discriminator.
    Untransmitted,
}

impl FrameType {
    /// Decode the 2 LSBs of the first payload byte.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => FrameType::HighRate,
            0b01 => FrameType::LowRate,
            0b10 => FrameType::SidFrame,
            _ => FrameType::Untransmitted,
        }
    }

    /// Expected payload size in bytes, including the discriminator bits.
    pub fn frame_size(self) -> usize {
        match self {
            FrameType::HighRate => 24,
            FrameType::LowRate => 20,
            FrameType::SidFrame => 4,
            FrameType::Untransmitted => 1,
        }
    }

    /// Human-readable bit-rate label.
    pub fn bit_rate_label(self) -> &'static str {
        match self {
            FrameType::HighRate => "6.3 kbit/s",
            FrameType::LowRate => "5.3 kbit/s",
            FrameType::SidFrame => "SID (comfort-noise)",
            FrameType::Untransmitted => "untransmitted",
        }
    }
}

/// Inspect a packet's first byte and return the frame type + expected size.
pub fn parse_frame_type(data: &[u8]) -> Result<FrameType> {
    if data.is_empty() {
        return Err(Error::invalid("G.723.1: empty packet, no rate bits"));
    }
    Ok(FrameType::from_bits(data[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminator_matrix() {
        assert_eq!(FrameType::from_bits(0b00), FrameType::HighRate);
        assert_eq!(FrameType::from_bits(0b01), FrameType::LowRate);
        assert_eq!(FrameType::from_bits(0b10), FrameType::SidFrame);
        assert_eq!(FrameType::from_bits(0b11), FrameType::Untransmitted);
        // Upper bits must be ignored.
        assert_eq!(FrameType::from_bits(0xFC), FrameType::HighRate);
        assert_eq!(FrameType::from_bits(0x7D), FrameType::LowRate);
    }

    #[test]
    fn expected_sizes() {
        assert_eq!(FrameType::HighRate.frame_size(), 24);
        assert_eq!(FrameType::LowRate.frame_size(), 20);
        assert_eq!(FrameType::SidFrame.frame_size(), 4);
        assert_eq!(FrameType::Untransmitted.frame_size(), 1);
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse_frame_type(&[]).is_err());
    }

    #[test]
    fn parse_picks_low_rate() {
        // First byte 0x01 — LSBs = 01 → low rate.
        assert_eq!(parse_frame_type(&[0x01]).unwrap(), FrameType::LowRate);
    }
}
