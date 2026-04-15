//! MSB-first bit reader for MPEG audio bitstreams.
//!
//! MPEG-1 audio stores every multi-bit field with the most-significant bit
//! first within each byte. The reader keeps a 64-bit accumulator so callers
//! can request arbitrary widths up to 32 bits in a single call.

use oxideav_core::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to load into the accumulator.
    byte_pos: usize,
    /// Bits buffered from `data`, left-aligned in `acc` (high bits = next).
    acc: u64,
    /// Number of valid bits currently in `acc` (0..=64).
    bits_in_acc: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Total bits in the underlying buffer.
    pub fn total_bits(&self) -> u64 {
        self.data.len() as u64 * 8
    }

    /// Bits already consumed from the stream.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    /// Bits remaining in the stream (unread).
    pub fn bits_remaining(&self) -> u64 {
        self.total_bits() - self.bit_position()
    }

    /// Reload the accumulator from the underlying slice.
    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned integer.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "BitReader::read_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("mp2 BitReader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read a single bit as bool.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u32(1)? != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_small() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(4).unwrap(), 0x5);
        assert_eq!(br.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn remaining_bits() {
        let mut br = BitReader::new(&[0xFF, 0xFF]);
        assert_eq!(br.bits_remaining(), 16);
        let _ = br.read_u32(5).unwrap();
        assert_eq!(br.bits_remaining(), 11);
    }
}
