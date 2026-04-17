//! MSB-first bit writer for MPEG-1 Audio Layer I bitstreams. Counterpart
//! to [`crate::bitreader::BitReader`].
//!
//! Same wire convention as the reader: multi-bit fields are stored
//! most-significant bit first within each byte. Values up to 32 bits wide
//! can be written in a single call.

pub struct BitWriter {
    buf: Vec<u8>,
    /// Bits queued for output, left-aligned in `acc` (high bits first).
    acc: u64,
    /// Number of valid bits currently in `acc` (0..64).
    bits_in_acc: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Total number of bits written so far (including bits buffered in acc).
    pub fn bit_position(&self) -> u64 {
        self.buf.len() as u64 * 8 + self.bits_in_acc as u64
    }

    /// Append `n` bits (0..=32) of `value`. High bits of `value` beyond `n`
    /// are masked off.
    pub fn write_u32(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32, "BitWriter::write_u32 supports up to 32 bits");
        if n == 0 {
            return;
        }
        let mask: u64 = if n == 32 {
            0xFFFF_FFFF
        } else {
            (1u64 << n) - 1
        };
        let v = (value as u64) & mask;
        debug_assert!(self.bits_in_acc + n <= 64);
        self.acc |= v << (64 - n - self.bits_in_acc);
        self.bits_in_acc += n;
        while self.bits_in_acc >= 8 {
            let byte = (self.acc >> 56) as u8;
            self.buf.push(byte);
            self.acc <<= 8;
            self.bits_in_acc -= 8;
        }
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Pad current byte with zero bits so the writer is byte-aligned.
    pub fn align_to_byte(&mut self) {
        let pad = (8 - self.bits_in_acc % 8) % 8;
        if pad > 0 {
            self.write_u32(0, pad);
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn byte_len(&self) -> usize {
        self.buf.len()
    }

    /// Consume the writer, padding any partial byte with zeros, and return
    /// the final byte buffer.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn round_trip_u32() {
        let mut w = BitWriter::new();
        w.write_u32(0xA, 4);
        w.write_u32(0x5, 4);
        w.write_u32(0xC3, 8);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_u32(4).unwrap(), 0xA);
        assert_eq!(r.read_u32(4).unwrap(), 0x5);
        assert_eq!(r.read_u32(8).unwrap(), 0xC3);
    }

    #[test]
    fn align_pads_with_zeros() {
        let mut w = BitWriter::new();
        w.write_u32(0b101, 3);
        w.align_to_byte();
        assert_eq!(w.bytes(), &[0b10100000]);
    }
}
