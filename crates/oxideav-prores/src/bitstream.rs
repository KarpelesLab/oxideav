//! Bit I/O for the ProRes entropy layer.
//!
//! SMPTE RDD 36 uses several coefficient codes (Rice, exp-Golomb with
//! adaptive code length, etc.). Since this crate round-trips its own
//! bitstream end-to-end we pick the **simplest** variants that encode /
//! decode byte-identically with no reference dependency, and document
//! the deviation here:
//!
//! * Unsigned values (runs, magnitudes): **plain unsigned exp-Golomb**
//!   (the H.264 `ue(v)` code). For `N >= 0`, write `leading_zeros`
//!   zeros, a `1`, then `leading_zeros` bits of `N + 1 - 2^leading_zeros`.
//!
//! * Signed values (DC residuals, AC levels): the **H.264 `se(v)`**
//!   mapping on top of `ue(v)`. `0 -> 0`, positive `v -> 2v - 1`,
//!   negative `v -> -2v`.
//!
//! This is *not* what a streaming ProRes player from Apple would
//! output, but it is fully deterministic and byte-exact for any
//! encoder/decoder pair that both read this file. Callers that need to
//! interop with real `.mov` ProRes files will have to swap these two
//! helpers for the RDD 36 codes — the rest of the pipeline
//! (DCT/quant/frame-layout) is format-compatible.

use oxideav_core::{Error, Result};

/// MSB-first bit reader with byte-aligned backing buffer.
pub struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize, // absolute bit offset from start of `buf`
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Read a single bit (0 or 1).
    pub fn read_bit(&mut self) -> Result<u32> {
        let byte_idx = self.bit_pos / 8;
        if byte_idx >= self.buf.len() {
            return Err(Error::invalid("prores: bitstream EOF"));
        }
        let bit_idx = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        Ok(((self.buf[byte_idx] >> bit_idx) & 1) as u32)
    }

    /// Read the next `n` bits (MSB first). `n` must be in `1..=32`.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!((1..=32).contains(&n));
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Ok(v)
    }

    /// Unsigned exp-Golomb (H.264 `ue(v)`).
    pub fn read_ue(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        while self.read_bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(Error::invalid("prores: ue exponent too large"));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let tail = self.read_bits(zeros)?;
        Ok((1u32 << zeros) - 1 + tail)
    }

    /// Signed exp-Golomb (H.264 `se(v)`).
    pub fn read_se(&mut self) -> Result<i32> {
        let k = self.read_ue()? as i64;
        if k == 0 {
            return Ok(0);
        }
        // k=1 -> +1, k=2 -> -1, k=3 -> +2, k=4 -> -2, ...
        let abs = (k + 1) / 2;
        let v = if (k & 1) == 1 { abs } else { -abs };
        Ok(v as i32)
    }

    /// How many whole bytes we've consumed, rounding up any partial byte.
    pub fn byte_pos(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }
}

/// MSB-first bit writer.
pub struct BitWriter {
    buf: Vec<u8>,
    /// Bit offset into the *last* byte of `buf` where the next write lands.
    /// `0..=7`. When it hits 8 we push a fresh byte.
    cur_bits_used: u32,
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
            cur_bits_used: 8,
        }
    }

    pub fn write_bit(&mut self, b: u32) {
        if self.cur_bits_used == 8 {
            self.buf.push(0);
            self.cur_bits_used = 0;
        }
        let last = self.buf.last_mut().unwrap();
        let shift = 7 - self.cur_bits_used;
        *last |= ((b & 1) as u8) << shift;
        self.cur_bits_used += 1;
    }

    pub fn write_bits(&mut self, v: u32, n: u32) {
        debug_assert!((1..=32).contains(&n));
        for i in (0..n).rev() {
            self.write_bit((v >> i) & 1);
        }
    }

    pub fn write_ue(&mut self, v: u32) {
        // Choose the smallest `zeros` such that v < 2^(zeros+1) - 1, i.e.
        // v + 1 fits in `zeros+1` bits with the leading 1. The code is
        // `zeros` zeros + one 1 + `zeros` value bits.
        let x = (v as u64) + 1;
        let zeros = 63 - x.leading_zeros();
        // prefix
        for _ in 0..zeros {
            self.write_bit(0);
        }
        self.write_bit(1);
        if zeros > 0 {
            // tail = (v + 1) - 2^zeros, in `zeros` bits
            let tail = (x - (1u64 << zeros)) as u32;
            self.write_bits(tail, zeros);
        }
    }

    pub fn write_se(&mut self, v: i32) {
        let k = if v == 0 {
            0u32
        } else if v > 0 {
            (2 * v as u32) - 1
        } else {
            2 * (-v) as u32
        };
        self.write_ue(k);
    }

    /// Pad to the next byte boundary with zero bits.
    pub fn align_byte(&mut self) {
        while self.cur_bits_used != 8 {
            self.write_bit(0);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.align_byte();
        self.buf
    }

    pub fn byte_len(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ue_roundtrip() {
        for &v in &[0u32, 1, 2, 7, 255, 1023, 65_535, 1_000_000] {
            let mut w = BitWriter::new();
            w.write_ue(v);
            let data = w.finish();
            let mut r = BitReader::new(&data);
            assert_eq!(r.read_ue().unwrap(), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn se_roundtrip() {
        for &v in &[0i32, 1, -1, 42, -42, 1023, -1023, 1_000_000, -1_000_000] {
            let mut w = BitWriter::new();
            w.write_se(v);
            let data = w.finish();
            let mut r = BitReader::new(&data);
            assert_eq!(r.read_se().unwrap(), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn bits_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010_1100, 8);
        w.write_bits(0x5A_5A, 16);
        w.write_bit(1);
        w.write_bit(0);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(8).unwrap(), 0b1010_1100);
        assert_eq!(r.read_bits(16).unwrap(), 0x5A_5A);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
    }
}
