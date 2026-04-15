//! MSB-first bit reader for AAC bitstreams.
//!
//! AAC (ISO/IEC 14496-3) stores all multi-bit fields most-significant-bit first
//! within each byte. Huffman codebooks and noiseless-coding tables are also
//! decoded MSB-first. The reader keeps a 64-bit accumulator so callers can
//! request widths up to 32 bits at a time without a byte-boundary spill.

use oxideav_core::{Error, Result};

/// MSB-first bit reader over a borrowed byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to load into the accumulator.
    byte_pos: usize,
    /// Bits buffered from `data`, left-aligned (high bits = next to consume).
    acc: u64,
    /// Valid bits currently in `acc`, in the range `0..=64`.
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

    /// Bits already consumed from the logical stream.
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

    /// Total remaining bits (buffered + unread from slice).
    pub fn bits_remaining(&self) -> u64 {
        self.bits_in_acc as u64 + ((self.data.len() - self.byte_pos) as u64) * 8
    }

    /// Refill the 64-bit accumulator with whole bytes while there's room.
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
                return Err(Error::invalid("aac bitreader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read `n` bits (0..=64) as an unsigned integer.
    pub fn read_u64(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n <= 32 {
            return self.read_u32(n).map(|v| v as u64);
        }
        let hi = self.read_u32(n - 32)? as u64;
        let lo = self.read_u32(32)? as u64;
        Ok((hi << 32) | lo)
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

    /// Read a single bit as a bool.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u32(1)? != 0)
    }

    /// Peek `n` bits (0..=32) without consuming them.
    pub fn peek_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("aac bitreader: out of bits for peek"));
            }
        }
        Ok((self.acc >> (64 - n)) as u32)
    }

    /// Consume `n` previously-peeked bits.
    pub fn skip(&mut self, n: u32) -> Result<()> {
        let mut left = n;
        while left > 32 {
            self.read_u32(32)?;
            left -= 32;
        }
        self.read_u32(left)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_reads() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(4).unwrap(), 0x5);
        assert_eq!(br.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn peek_then_skip() {
        let mut br = BitReader::new(&[0xFF, 0x00]);
        assert_eq!(br.peek_u32(12).unwrap(), 0xFF0);
        br.skip(4).unwrap();
        assert_eq!(br.read_u32(8).unwrap(), 0xF0);
    }

    #[test]
    fn signed_extends() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_i32(4).unwrap(), -1);
        assert_eq!(br.read_i32(4).unwrap(), -1);
    }

    #[test]
    fn alignment() {
        let mut br = BitReader::new(&[0xFF, 0x55]);
        br.read_u32(3).unwrap();
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.read_u32(8).unwrap(), 0x55);
    }
}
