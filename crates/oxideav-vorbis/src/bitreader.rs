//! LSB-first bit reader for Vorbis packets.
//!
//! Vorbis packs bits LSB-first within each byte (Vorbis I §2.1.4). For a
//! field of `N` bits read at a position where the byte's low `K` bits are
//! still available, the lowest `min(N, K)` bits of that byte are consumed
//! first, then the next byte continues. This is the *opposite* of FLAC's
//! big-endian/MSB-first packing.

use oxideav_core::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Byte offset of the next byte to fetch into the accumulator.
    byte_pos: usize,
    /// Buffered bits, low-aligned (next bit to emit is bit 0 of `acc`).
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

    /// True when the read position is on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << self.bits_in_acc;
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read up to 32 bits as an unsigned integer.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "Vorbis BitReader::read_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("Vorbis BitReader: out of bits"));
            }
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let v = (self.acc as u32) & mask;
        self.acc >>= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read up to 64 bits as an unsigned integer.
    pub fn read_u64(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n == 0 {
            return Ok(0);
        }
        if n <= 32 {
            return Ok(self.read_u32(n)? as u64);
        }
        let lo = self.read_u32(32)? as u64;
        let hi = self.read_u32(n - 32)? as u64;
        Ok(lo | (hi << 32))
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

    /// Read a 32-bit IEEE-like Vorbis float (Vorbis I §9.2.2). The encoded
    /// representation is mantissa (signed 21-bit) + sign + exponent (10-bit
    /// biased), unrelated to IEEE 754 layout.
    pub fn read_vorbis_float(&mut self) -> Result<f32> {
        let raw = self.read_u32(32)?;
        let mantissa_raw = raw & 0x001FFFFF;
        let sign = raw & 0x80000000;
        let exponent = ((raw & 0x7FE00000) >> 21) as i32;
        let mantissa = if sign != 0 {
            -(mantissa_raw as f64)
        } else {
            mantissa_raw as f64
        };
        Ok((mantissa * 2f64.powi(exponent - 788)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_first_byte() {
        // Byte 0xA5 = 0b10100101. LSB-first reading should yield bits in order:
        // 1, 0, 1, 0, 0, 1, 0, 1.
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
    fn multi_byte_lsb() {
        // [0x12, 0x34] read as 16 bits LSB-first should give 0x3412.
        let mut br = BitReader::new(&[0x12, 0x34]);
        assert_eq!(br.read_u32(16).unwrap(), 0x3412);
    }

    #[test]
    fn read_signed() {
        // 4 bits 0b1111 = -1 signed.
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_i32(4).unwrap(), -1);
    }
}
