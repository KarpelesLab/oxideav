//! MSB-first bit reader for Theora packets.
//!
//! Theora packs bits MSB-first within each byte (Theora spec §3.1). This is
//! the opposite of Vorbis (LSB-first) but matches most video formats.

use oxideav_core::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Byte offset of the next byte to fetch.
    byte_pos: usize,
    /// Buffered bits, high-aligned (next bit to emit is bit 63 of `acc`).
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

    /// Total bits consumed so far.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    /// Number of bits still available (unread), approximately.
    pub fn bits_remaining(&self) -> u64 {
        ((self.data.len() - self.byte_pos) as u64) * 8 + self.bits_in_acc as u64
    }

    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read up to 32 bits MSB-first.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::Eof);
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read up to 64 bits MSB-first.
    pub fn read_u64(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n == 0 {
            return Ok(0);
        }
        if n <= 32 {
            return Ok(self.read_u32(n)? as u64);
        }
        let hi = self.read_u32(n - 32)? as u64;
        let lo = self.read_u32(32)? as u64;
        Ok((hi << 32) | lo)
    }

    /// Read `n` bits as a signed integer (sign-extended from bit `n-1`).
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_byte() {
        // Byte 0xA5 = 0b10100101. MSB-first reading yields: 1,0,1,0,0,1,0,1.
        let mut br = BitReader::new(&[0xA5]);
        assert_eq!(br.read_u32(1).unwrap(), 1);
        assert_eq!(br.read_u32(1).unwrap(), 0);
        assert_eq!(br.read_u32(1).unwrap(), 1);
        assert_eq!(br.read_u32(1).unwrap(), 0);
        assert_eq!(br.read_u32(1).unwrap(), 0);
        assert_eq!(br.read_u32(1).unwrap(), 1);
        assert_eq!(br.read_u32(1).unwrap(), 0);
        assert_eq!(br.read_u32(1).unwrap(), 1);
    }

    #[test]
    fn multi_byte_msb() {
        // [0x12, 0x34] read as 16 bits MSB-first gives 0x1234.
        let mut br = BitReader::new(&[0x12, 0x34]);
        assert_eq!(br.read_u32(16).unwrap(), 0x1234);
    }

    #[test]
    fn big_field() {
        // 0x12345678 as 32 bits MSB-first
        let mut br = BitReader::new(&[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(br.read_u32(32).unwrap(), 0x12345678);
    }

    #[test]
    fn cross_byte() {
        // Reading 4 bits, then 8 bits, then 4 bits from 0xAB, 0xCD should give:
        // 0xA, 0xBC, 0xD
        let mut br = BitReader::new(&[0xAB, 0xCD]);
        assert_eq!(br.read_u32(4).unwrap(), 0xA);
        assert_eq!(br.read_u32(8).unwrap(), 0xBC);
        assert_eq!(br.read_u32(4).unwrap(), 0xD);
    }

    #[test]
    fn read_signed() {
        // 4 bits 0b1111 = -1 signed.
        let mut br = BitReader::new(&[0xF0]);
        assert_eq!(br.read_i32(4).unwrap(), -1);
    }
}
