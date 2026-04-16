//! MSB-first bit reader for GSM 06.10 frames.
//!
//! GSM frames pack their 260 bits most-significant-bit first into the 33
//! bytes of a standard-framing payload. All field widths are small
//! (1..=7 bits), so a simple byte-oriented accumulator is sufficient.

use oxideav_core::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit cursor measured from the start of `data`, MSB-first.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Total number of bits in the underlying buffer.
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Bit cursor (number of bits already consumed).
    pub fn bit_position(&self) -> usize {
        self.pos
    }

    /// Remaining bits in the buffer.
    pub fn remaining(&self) -> usize {
        self.total_bits().saturating_sub(self.pos)
    }

    /// Read `n` bits (0..=16) as an unsigned integer, MSB-first.
    pub fn read_u16(&mut self, n: u32) -> Result<u16> {
        debug_assert!(n <= 16, "GSM BitReader::read_u16 supports up to 16 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.total_bits() {
            return Err(Error::invalid("GSM BitReader: out of bits"));
        }
        let mut v: u16 = 0;
        for _ in 0..n {
            let byte = self.data[self.pos >> 3];
            let bit = (byte >> (7 - (self.pos & 7))) & 1;
            v = (v << 1) | bit as u16;
            self.pos += 1;
        }
        Ok(v)
    }

    /// Read one bit as bool.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_u16(1)? != 0)
    }
}

/// MSB-first bit writer companion.
///
/// Used by the encoder to pack the 260 bits of a GSM Full Rate frame into
/// 33 bytes in the same order the [`BitReader`] consumes them. Fields are
/// written big-endian within each byte.
pub struct BitWriter {
    pub data: Vec<u8>,
    pos: usize,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            pos: 0,
        }
    }

    /// Number of bits already written.
    pub fn bit_position(&self) -> usize {
        self.pos
    }

    pub fn write(&mut self, value: u16, n: u32) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_index = self.pos >> 3;
            if byte_index >= self.data.len() {
                self.data.push(0);
            }
            self.data[byte_index] |= bit << (7 - (self.pos & 7));
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_fields() {
        // 0xD0 = 1101 0000 — 4 bits = 0xD, then 4 bits = 0x0
        let mut br = BitReader::new(&[0xD0, 0x5A]);
        assert_eq!(br.read_u16(4).unwrap(), 0xD);
        assert_eq!(br.read_u16(4).unwrap(), 0x0);
        assert_eq!(br.read_u16(8).unwrap(), 0x5A);
    }

    #[test]
    fn read_across_bytes() {
        // bits: 1111 1111 | 1111 0000 — 12 bits = 0xFFF, 4 bits = 0x0
        let mut br = BitReader::new(&[0xFF, 0xF0]);
        assert_eq!(br.read_u16(12).unwrap(), 0xFFF);
        assert_eq!(br.read_u16(4).unwrap(), 0x0);
    }

    #[test]
    fn writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        w.write(0x6, 3); // 110
        w.write(0x1, 1); // 1
        w.write(0x5A, 8);
        w.write(0x0, 4);
        let mut r = BitReader::new(&w.data);
        assert_eq!(r.read_u16(3).unwrap(), 0x6);
        assert_eq!(r.read_u16(1).unwrap(), 0x1);
        assert_eq!(r.read_u16(8).unwrap(), 0x5A);
        assert_eq!(r.read_u16(4).unwrap(), 0x0);
    }

    #[test]
    fn exhaustion() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.read_u16(8).unwrap(), 0xFF);
        assert!(br.read_u16(1).is_err());
    }
}
