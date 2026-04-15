//! MSB-first bit reader for MPEG-1 Audio bitstreams.
//!
//! Every multi-bit field in the MPEG-1 audio spec is stored most-significant
//! bit first within each byte. The reader buffers bits into a 64-bit
//! accumulator so callers can request any width up to 32 bits in one go.

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

    /// Number of bits already consumed from the stream.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    /// True if the reader is positioned on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Skip remaining bits in the current byte, leaving the reader byte-aligned.
    pub fn align_to_byte(&mut self) {
        let drop = self.bits_in_acc % 8;
        self.acc <<= drop;
        self.bits_in_acc -= drop;
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
                return Err(Error::invalid("MP1 BitReader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read `n` bits as a signed integer, sign-extended from the high bit.
    pub fn read_i32(&mut self, n: u32) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        let raw = self.read_u32(n)? as i32;
        let shift = 32 - n;
        Ok((raw << shift) >> shift)
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
    fn read_u32_small() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(4).unwrap(), 0x5);
        assert_eq!(br.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn read_across_many_bytes() {
        // 0xFF 0xE0 0x00 0x00 — typical sync+header start, read 12-bit sync.
        let mut br = BitReader::new(&[0xFF, 0xE0, 0x00, 0x00]);
        assert_eq!(br.read_u32(12).unwrap(), 0xFFE);
        assert_eq!(br.read_u32(4).unwrap(), 0x0);
    }

    #[test]
    fn alignment_roundtrip() {
        let mut br = BitReader::new(&[0xFF, 0xFF]);
        br.read_u32(3).unwrap();
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.read_u32(8).unwrap(), 0xFF);
    }

    #[test]
    fn read_i32_negative() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_i32(4).unwrap(), -1);
    }
}
