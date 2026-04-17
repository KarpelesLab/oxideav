//! MSB-first bit writer for Speex bitstreams.
//!
//! Mirror of [`crate::bitreader::BitReader`]: bits are packed
//! most-significant-bit first within each byte, matching libspeex's
//! `speex_bits_pack`. Finishing the writer zero-pads the trailing
//! partial byte.

pub struct BitWriter {
    buf: Vec<u8>,
    /// Bit accumulator: bits live in the low `nbits` positions and are
    /// shifted out from the MSB into the next output byte.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
            acc: 0,
            nbits: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            acc: 0,
            nbits: 0,
        }
    }

    /// Write the low `len` bits of `value` MSB-first. `len` must be ≤ 32.
    pub fn write_bits(&mut self, value: u32, len: u32) {
        debug_assert!(len <= 32);
        if len == 0 {
            return;
        }
        let mask = if len == 32 {
            u32::MAX
        } else {
            (1u32 << len) - 1
        };
        let v = (value & mask) as u64;
        self.acc = (self.acc << len) | v;
        self.nbits += len;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let b = ((self.acc >> self.nbits) & 0xFF) as u8;
            self.buf.push(b);
        }
    }

    /// Current bit position.
    pub fn bit_position(&self) -> u64 {
        self.buf.len() as u64 * 8 + self.nbits as u64
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Pad to byte boundary with zero bits.
    pub fn align_to_byte(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
    }

    /// Flush any partial byte (zero-padded) and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.buf
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn round_trip_via_bitreader() {
        let mut bw = BitWriter::new();
        let payload = [(0xA, 4u32), (0x5, 4), (0xC3, 8), (0b101, 3), (0x1FFFF, 17)];
        for (v, n) in payload {
            bw.write_bits(v, n);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        for (v, n) in payload {
            let got = br.read_u32(n).unwrap();
            assert_eq!(got, v, "round-trip {n}-bit field");
        }
    }

    #[test]
    fn pads_trailing_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        let out = bw.finish();
        assert_eq!(out, vec![0b1010_0000]);
    }
}
