//! MSB-first bit reader for MP3 bitstreams.
//!
//! MPEG-1/2 Layer III stores multi-bit fields with the most-significant bit
//! first within each byte. The reader keeps a 64-bit accumulator so callers
//! can request arbitrary widths up to 32 bits in one go. Shape mirrors
//! `oxideav_flac::bitreader::BitReader` — the two codecs use the same scheme.

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

    /// Number of bits already consumed from the underlying slice.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    /// Bytes already consumed (requires byte alignment).
    pub fn byte_position(&self) -> usize {
        debug_assert_eq!(
            self.bits_in_acc % 8,
            0,
            "byte_position requires byte alignment"
        );
        self.byte_pos - (self.bits_in_acc as usize) / 8
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Skip remaining bits in the current byte so the reader sits on a
    /// byte boundary.
    pub fn align_to_byte(&mut self) {
        let drop = self.bits_in_acc % 8;
        self.acc <<= drop;
        self.bits_in_acc -= drop;
    }

    /// Number of bits remaining in the stream.
    pub fn bits_remaining(&self) -> u64 {
        let total = self.data.len() as u64 * 8;
        total.saturating_sub(self.bit_position())
    }

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
                return Err(Error::invalid("BitReader: out of bits"));
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

    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u32(1)? != 0)
    }

    /// Peek `n` bits (1..=32) without advancing. Useful for Huffman decoding.
    pub fn peek_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "BitReader::peek_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("BitReader: out of bits"));
            }
        }
        Ok((self.acc >> (64 - n)) as u32)
    }

    /// Consume `n` already-peeked bits.
    pub fn consume(&mut self, n: u32) -> Result<()> {
        debug_assert!(n <= 32, "BitReader::consume up to 32 bits");
        if n == 0 {
            return Ok(());
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("BitReader: out of bits"));
            }
        }
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_basic() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(4).unwrap(), 0x5);
        assert_eq!(br.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn read_i32_sign_extends() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_i32(4).unwrap(), -1);
        assert_eq!(br.read_i32(4).unwrap(), -1);
    }

    #[test]
    fn peek_and_consume() {
        let mut br = BitReader::new(&[0xAB, 0xCD]);
        assert_eq!(br.peek_u32(4).unwrap(), 0xA);
        assert_eq!(br.peek_u32(8).unwrap(), 0xAB);
        br.consume(4).unwrap();
        assert_eq!(br.read_u32(4).unwrap(), 0xB);
    }

    #[test]
    fn aligns_cleanly() {
        let mut br = BitReader::new(&[0xFF, 0xFF]);
        let _ = br.read_u32(3).unwrap();
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.read_u32(8).unwrap(), 0xFF);
    }

    #[test]
    fn out_of_bits() {
        let mut br = BitReader::new(&[0x00]);
        let _ = br.read_u32(8).unwrap();
        assert!(br.read_u32(1).is_err());
    }

    #[test]
    fn bits_remaining_tracks() {
        let mut br = BitReader::new(&[0xFF, 0xFF]);
        assert_eq!(br.bits_remaining(), 16);
        br.read_u32(3).unwrap();
        assert_eq!(br.bits_remaining(), 13);
    }
}
