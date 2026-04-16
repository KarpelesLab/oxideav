//! MSB-first bit reader for MPEG-1 video bitstreams.
//!
//! MPEG-1 video stores fields MSB-first within each byte. Start codes are
//! byte-aligned and begin with `0x000001`, so the reader exposes helpers to
//! locate / skip to them.

use oxideav_core::{Error, Result};

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

    /// Construct a reader positioned at `byte_pos` within `data`.
    pub fn with_position(data: &'a [u8], byte_pos: usize) -> Self {
        Self {
            data,
            byte_pos,
            acc: 0,
            bits_in_acc: 0,
        }
    }

    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    pub fn byte_position(&self) -> usize {
        debug_assert_eq!(
            self.bits_in_acc % 8,
            0,
            "byte_position needs byte alignment"
        );
        self.byte_pos - (self.bits_in_acc as usize) / 8
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

    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("mpeg1video bitreader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read a signed value of `n` bits (two's-complement).
    pub fn read_i32(&mut self, n: u32) -> Result<i32> {
        let u = self.read_u32(n)?;
        if n == 0 {
            return Ok(0);
        }
        let sign = 1u32 << (n - 1);
        if u & sign != 0 {
            Ok(u as i32 - (1i64 << n) as i32)
        } else {
            Ok(u as i32)
        }
    }

    pub fn read_u1(&mut self) -> Result<u32> {
        self.read_u32(1)
    }

    /// Peek the next `n` bits without consuming them.
    pub fn peek_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("mpeg1video bitreader: peek past EOF"));
            }
        }
        Ok((self.acc >> (64 - n)) as u32)
    }

    pub fn consume(&mut self, n: u32) -> Result<()> {
        // Must already have bits available (used after peek).
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("mpeg1video bitreader: consume past EOF"));
            }
        }
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(())
    }

    /// Skip `n` bits (alias for read_u32 result-discarding).
    pub fn skip(&mut self, n: u32) -> Result<()> {
        let mut remaining = n;
        while remaining > 32 {
            self.read_u32(32)?;
            remaining -= 32;
        }
        self.read_u32(remaining)?;
        Ok(())
    }

    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() as u64 - self.byte_pos as u64) * 8 + self.bits_in_acc as u64
    }

    /// Consume zero-bits until byte alignment. In MPEG-1 the `next_start_code`
    /// procedure asserts that the stuffing consists of bits `0..` pattern —
    /// we don't actually verify that.
    pub fn byte_align(&mut self) {
        self.align_to_byte();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_msb_first() {
        // bytes 0b10110001, 0b01010101 → first bit = 1, next 2 bits = 01, next 5 = 10001
        let data = [0b1011_0001u8, 0b0101_0101];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_u32(1).unwrap(), 1);
        assert_eq!(br.read_u32(2).unwrap(), 0b01);
        assert_eq!(br.read_u32(5).unwrap(), 0b1_0001);
        assert_eq!(br.read_u32(8).unwrap(), 0b0101_0101);
    }

    #[test]
    fn peek_then_consume() {
        let data = [0xAB, 0xCD];
        let mut br = BitReader::new(&data);
        let p = br.peek_u32(8).unwrap();
        assert_eq!(p, 0xAB);
        br.consume(4).unwrap();
        assert_eq!(br.read_u32(4).unwrap(), 0xB);
        assert_eq!(br.read_u32(8).unwrap(), 0xCD);
    }
}
