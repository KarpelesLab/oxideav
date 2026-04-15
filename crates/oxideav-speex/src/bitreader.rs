//! MSB-first bit reader for Speex bitstreams.
//!
//! Speex packs its codec parameters bit-by-bit, most-significant bit first
//! within each byte (see `libspeex/bits.c` in the xiph/speex reference).
//! The shape matches `oxideav_mp3::bitreader::BitReader`: a 64-bit
//! accumulator, left-aligned, so widths up to 32 bits can be extracted in
//! a single call.
//!
//! Speex frames generally advertise their length via a header/mode
//! selector, so this reader does not need the sophisticated refill logic
//! of arithmetic coders — it only needs enough bits for one field at a
//! time.

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

    /// Number of bits remaining in the stream.
    pub fn bits_remaining(&self) -> u64 {
        let total = self.data.len() as u64 * 8;
        total.saturating_sub(self.bit_position())
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Skip remaining bits in the current byte so the reader sits on a byte
    /// boundary.
    pub fn align_to_byte(&mut self) {
        let drop = self.bits_in_acc % 8;
        self.acc <<= drop;
        self.bits_in_acc -= drop;
    }

    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned integer, MSB first.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "BitReader::read_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("Speex bitreader: out of bits"));
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

    /// Peek `n` bits without advancing.
    pub fn peek_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "BitReader::peek_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("Speex bitreader: out of bits"));
            }
        }
        Ok((self.acc >> (64 - n)) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        // 0xA5 = 1010_0101
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
    fn tracks_remaining_bits() {
        let mut br = BitReader::new(&[0xFF, 0xFF]);
        assert_eq!(br.bits_remaining(), 16);
        br.read_u32(5).unwrap();
        assert_eq!(br.bits_remaining(), 11);
    }

    #[test]
    fn aligns_to_byte() {
        let mut br = BitReader::new(&[0xFF, 0x55]);
        br.read_u32(3).unwrap();
        assert!(!br.is_byte_aligned());
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.read_u32(8).unwrap(), 0x55);
    }

    #[test]
    fn peek_does_not_advance() {
        let mut br = BitReader::new(&[0xAB]);
        assert_eq!(br.peek_u32(4).unwrap(), 0xA);
        assert_eq!(br.peek_u32(8).unwrap(), 0xAB);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
    }

    #[test]
    fn out_of_bits_errors() {
        let mut br = BitReader::new(&[0x00]);
        br.read_u32(8).unwrap();
        assert!(br.read_u32(1).is_err());
    }
}
