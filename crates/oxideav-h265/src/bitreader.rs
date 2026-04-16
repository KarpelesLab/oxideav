//! MSB-first bit reader plus Exp-Golomb helpers for HEVC RBSPs.
//!
//! HEVC syntax elements are coded MSB-first within each byte; the bit reader
//! mirrors the layout used by `oxideav-mpeg4video` so the two crates share a
//! mental model. After the RBSP has had emulation-prevention bytes stripped
//! (see `nal::extract_rbsp`), this reader can be applied directly.
//!
//! Exp-Golomb decoding follows ITU-T H.265 §9.2.

use oxideav_core::{Error, Result};

/// MSB-first bit reader over a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u64,
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

    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() as u64 - self.byte_pos as u64) * 8 + self.bits_in_acc as u64
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

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

    /// Read `n` bits (0..=32) as an unsigned integer.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("h265 bitreader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read a single bit (HEVC syntax: u(1) / flag).
    pub fn u1(&mut self) -> Result<u32> {
        self.u(1)
    }

    /// Read `n` bits (0..=64).
    pub fn u_long(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n <= 32 {
            return Ok(self.u(n)? as u64);
        }
        let hi = self.u(n - 32)? as u64;
        let lo = self.u(32)? as u64;
        Ok((hi << 32) | lo)
    }

    /// Skip `n` bits.
    pub fn skip(&mut self, mut n: u32) -> Result<()> {
        while n > 32 {
            self.u(32)?;
            n -= 32;
        }
        if n > 0 {
            self.u(n)?;
        }
        Ok(())
    }

    /// Unsigned Exp-Golomb code (ue(v)). §9.2.
    pub fn ue(&mut self) -> Result<u32> {
        let mut zeros: u32 = 0;
        while self.u1()? == 0 {
            zeros += 1;
            if zeros > 32 {
                return Err(Error::invalid("h265 ue(v): too many leading zeros"));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let suffix = self.u(zeros)?;
        Ok((1u32 << zeros) - 1 + suffix)
    }

    /// Signed Exp-Golomb code (se(v)). §9.2.2.
    pub fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        // mapping: 0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2, 4 -> -2, …
        let val = ((k + 1) >> 1) as i32;
        if k & 1 == 1 {
            Ok(val)
        } else {
            Ok(-val)
        }
    }

    /// Whether further RBSP data is present (§B.2 / §7.2 — heuristic that
    /// remaining bits aren't only the rbsp_trailing_bits stop bit).
    pub fn has_more_rbsp_data(&mut self) -> bool {
        // If only the stop bit + zero pad remain, no more RBSP data.
        let saved_byte = self.byte_pos;
        let saved_acc = self.acc;
        let saved_bits = self.bits_in_acc;
        let remaining = self.bits_remaining();
        if remaining == 0 {
            return false;
        }
        // Find last 1 bit in the unread tail.
        let mut found_one_after_stop = false;
        // Walk bit-by-bit: cheap because RBSP tails are tiny.
        let mut bits_left = remaining;
        while bits_left > 0 {
            let bit = self.u1().unwrap_or(0);
            bits_left -= 1;
            if bit == 1 {
                if bits_left == 0 {
                    // Last bit was 1 — that's the stop bit; nothing else.
                    found_one_after_stop = false;
                } else {
                    // We saw a one — but could be the stop bit. Continue and
                    // check if any further 1 follows.
                    while bits_left > 0 {
                        let b = self.u1().unwrap_or(0);
                        bits_left -= 1;
                        if b == 1 {
                            found_one_after_stop = true;
                            break;
                        }
                    }
                }
                break;
            }
        }
        // Restore state.
        self.byte_pos = saved_byte;
        self.acc = saved_acc;
        self.bits_in_acc = saved_bits;
        found_one_after_stop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_msb_first() {
        let data = [0b1011_0001u8, 0b0101_0101];
        let mut br = BitReader::new(&data);
        assert_eq!(br.u(1).unwrap(), 1);
        assert_eq!(br.u(2).unwrap(), 0b01);
        assert_eq!(br.u(5).unwrap(), 0b1_0001);
        assert_eq!(br.u(8).unwrap(), 0b0101_0101);
    }

    #[test]
    fn ue_and_se() {
        // Bits: 1 (=0), 010 (=1), 011 (=2), 00100 (=3)
        // Concatenated: 1 010 011 00100 = 1010_0110_0100
        let data = [0b1010_0110, 0b0100_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.ue().unwrap(), 0);
        assert_eq!(br.ue().unwrap(), 1);
        assert_eq!(br.ue().unwrap(), 2);
        assert_eq!(br.ue().unwrap(), 3);

        // se: codeNum 0 -> 0, 1 -> 1, 2 -> -1
        let data = [0b1010_0110, 0b0100_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.se().unwrap(), 0);
        assert_eq!(br.se().unwrap(), 1);
        assert_eq!(br.se().unwrap(), -1);
        assert_eq!(br.se().unwrap(), 2);
    }
}
