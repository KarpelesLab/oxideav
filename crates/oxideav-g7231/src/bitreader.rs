//! Bit reader for ITU-T G.723.1 packet payloads.
//!
//! G.723.1 frames are packed little-endian: the bit-packing rules in Annex B
//! append bits into the least-significant position of the current byte first,
//! then fill upward. Fields that straddle a byte boundary have their low
//! bits in the earlier byte and their high bits in the later byte.
//!
//! This reader therefore consumes bits in LSB-first order within each byte,
//! byte-by-byte from the start of the payload.

use oxideav_core::{Error, Result};

/// LSB-first bit reader over a G.723.1 frame payload.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Number of bits already consumed from `data[byte_pos]` (0..8).
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Total bits already consumed.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 + self.bit_pos as u64
    }

    /// Total bits remaining in the payload.
    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.bit_position())
    }

    /// Read `n` bits (0..=32) as an unsigned integer, LSB-first packing.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(
            n <= 32,
            "G.723.1 BitReader::read_u32 supports up to 32 bits"
        );
        if n == 0 {
            return Ok(0);
        }
        if self.bits_remaining() < n as u64 {
            return Err(Error::invalid("G.723.1 BitReader: out of bits"));
        }
        let mut value: u32 = 0;
        let mut produced = 0;
        while produced < n {
            let byte = self.data[self.byte_pos] as u32;
            let take = (8 - self.bit_pos).min(n - produced);
            let chunk = (byte >> self.bit_pos) & ((1u32 << take) - 1);
            value |= chunk << produced;
            produced += take;
            self.bit_pos += take;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(value)
    }

    /// Read one bit as a bool.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u32(1)? != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_lsb_first() {
        // Byte 0x8D = 1000_1101 → LSB-first bits are 1,0,1,1,0,0,0,1.
        let mut br = BitReader::new(&[0x8D]);
        assert_eq!(br.read_u32(2).unwrap(), 0b01);
        assert_eq!(br.read_u32(3).unwrap(), 0b011);
        assert_eq!(br.read_u32(3).unwrap(), 0b100);
    }

    #[test]
    fn read_across_bytes() {
        // bytes: 0x34 = 0011_0100 (LSB→MSB: 0,0,1,0,1,1,0,0)
        //        0xAB = 1010_1011 (LSB→MSB: 1,1,0,1,0,1,0,1)
        // Read 6 bits then 10 bits.
        let mut br = BitReader::new(&[0x34, 0xAB]);
        let a = br.read_u32(6).unwrap();
        // low six bits of 0x34 = 110100 → binary (LSB-first collected) = 0b110100 = 0x34 & 0x3F = 0x34
        assert_eq!(a, 0x34 & 0x3F);
        let b = br.read_u32(10).unwrap();
        // high 2 bits of 0x34 (LSB-order, positions 6..7) plus all 8 bits of 0xAB.
        // top of byte 0 = (0x34 >> 6) & 0x3 = 0; then byte 1 = 0xAB → combined low=0, high=0xAB → 0xAB<<2 = 0x2AC.
        assert_eq!(b, 0xAB << 2);
    }

    #[test]
    fn out_of_bits() {
        let mut br = BitReader::new(&[0xFF]);
        assert!(br.read_u32(9).is_err());
    }
}
