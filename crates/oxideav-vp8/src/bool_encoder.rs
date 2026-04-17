//! Boolean (arithmetic) encoder — write-side companion to [`crate::bool_decoder`].
//!
//! Ported from libvpx's `vp8_encode_bool` (the loop-free, count-based
//! variant mirrored in RFC 6386 §20.2). For every `write_bool(prob, bit)`
//! the decoder's `read_bool(prob)` will produce `bit`.
//!
//! Call pattern:
//! ```ignore
//! let mut enc = BoolEncoder::new();
//! enc.write_bool(128, true);
//! enc.write_literal(5, 10);     // 5-bit value
//! let bytes = enc.finish();
//! ```

/// Write-side boolean coder. Emits bytes into an internal buffer.
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    lowvalue: u32,
    /// -24 = empty, 0 = ready to emit one byte.
    count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            range: 255,
            lowvalue: 0,
            count: -24,
        }
    }

    fn add_one_to_output(buf: &mut Vec<u8>) {
        let mut x = buf.len() as isize - 1;
        while x >= 0 && buf[x as usize] == 0xff {
            buf[x as usize] = 0;
            x -= 1;
        }
        if x >= 0 {
            buf[x as usize] += 1;
        }
    }

    /// Encode a single boolean with probability `prob` (0..=255).
    pub fn write_bool(&mut self, prob: u32, bit: bool) {
        debug_assert!(prob <= 255);
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let (mut range, mut lowvalue) = if bit {
            (self.range - split, self.lowvalue.wrapping_add(split))
        } else {
            (split, self.lowvalue)
        };
        while range < 128 {
            range <<= 1;
            if (lowvalue & 0x80000000) != 0 {
                Self::add_one_to_output(&mut self.out);
            }
            lowvalue <<= 1;
            self.count += 1;
            if self.count == 0 {
                self.out.push(((lowvalue >> 24) & 0xff) as u8);
                lowvalue &= 0x00ffffff;
                self.count = -8;
            }
        }
        self.range = range;
        self.lowvalue = lowvalue;
    }

    /// Encode an `n`-bit unsigned literal, MSB-first, each bit uniform
    /// probability (prob=128). Mirrors `BoolDecoder::read_literal`.
    pub fn write_literal(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) != 0;
            self.write_bool(128, b);
        }
    }

    /// Encode a signed value: `n` magnitude bits followed by a sign bit.
    /// Mirrors `BoolDecoder::read_signed_literal`.
    pub fn write_signed_literal(&mut self, n: u32, value: i32) {
        let mag = value.unsigned_abs();
        self.write_literal(n, mag);
        self.write_bool(128, value < 0);
    }

    /// Write a single uniform-probability flag.
    pub fn write_flag(&mut self, b: bool) {
        self.write_bool(128, b);
    }

    /// Current compressed byte length (after accounting for everything
    /// already flushed to the buffer). Does NOT include the bytes that
    /// are still buffered inside `lowvalue`.
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }

    /// Finish encoding. Pads with 32 zero-probability-128 bits to flush
    /// the state register. Consumes `self`.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bool(128, false);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;

    #[test]
    fn roundtrip_literal_values() {
        let mut enc = BoolEncoder::new();
        let vals = [(8u32, 0xa5u32), (7, 0x55), (4, 9), (3, 5)];
        for &(n, v) in &vals {
            enc.write_literal(n, v);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &(n, v) in &vals {
            assert_eq!(dec.read_literal(n), v);
        }
    }

    #[test]
    fn roundtrip_signed_literals() {
        let mut enc = BoolEncoder::new();
        let vals = [(4i32, 7i32), (4, -3), (6, 30), (6, -30), (7, 0)];
        for &(n, v) in &vals {
            enc.write_signed_literal(n as u32, v);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        for &(n, v) in &vals {
            let got = dec.read_signed_literal(n as u32);
            if v == 0 {
                // -0 and +0 both decode to 0 in this scheme.
                assert_eq!(got.abs(), 0);
            } else {
                assert_eq!(got, v, "n={n} v={v}");
            }
        }
    }
}
