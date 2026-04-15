//! MSB-first bit reader for G.728 packed-index streams.
//!
//! The G.728 payload is just concatenated 10-bit codebook indices — one
//! per 5-sample analysis vector — packed MSB-first. A 20 ms packet
//! (32 vectors) is therefore 320 bits = 40 bytes, and there are no
//! framing headers to skip. The reader nevertheless supports arbitrary
//! widths up to 32 bits so helpers can peek at the shape / gain sub-
//! fields without extra state.

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

    /// Number of bits already consumed from the stream.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    /// True if the reader is positioned on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Number of bits left to read in the backing slice.
    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() as u64 - self.byte_pos as u64) * 8 + self.bits_in_acc as u64
    }

    /// Reload the accumulator from the underlying slice.
    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned integer, MSB-first.
    pub fn read_u32(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32, "G.728 BitReader::read_u32 supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("G.728 BitReader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read one 10-bit codebook index.
    pub fn read_index10(&mut self) -> Result<u16> {
        Ok(self.read_u32(super::INDEX_BITS)? as u16)
    }
}

/// Decompose a 10-bit codebook index into its (shape, sign, gain-magnitude)
/// components. The bit layout used in the ITU reference implementation is:
///
/// ```text
///   bit 9 .. 3 : 7-bit shape selector    (0..127)
///   bit 2      : excitation sign bit
///   bit 1 .. 0 : 2-bit gain magnitude    (0..3; with the sign gives 8 levels)
/// ```
///
/// The 3-bit "gain codebook index" that appears in the tables is therefore
/// `(sign << 2) | magnitude`; returning the pieces separately keeps the
/// lookup tables compact (8 positive magnitudes) without losing info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnpackedIndex {
    pub shape_index: u8,
    pub sign: u8,
    pub gain_mag: u8,
}

impl UnpackedIndex {
    pub fn from_raw(raw: u16) -> Self {
        Self {
            shape_index: ((raw >> 3) & 0x7F) as u8,
            sign: ((raw >> 2) & 0x01) as u8,
            gain_mag: (raw & 0x03) as u8,
        }
    }

    /// Combined 3-bit gain codebook index (sign | magnitude) for code paths
    /// that prefer to index the 8-entry "signed" table directly.
    pub fn gain_index(&self) -> u8 {
        (self.sign << 2) | self.gain_mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_three_indices_across_byte_boundaries() {
        // Three 10-bit codes packed MSB-first: 0x3FF, 0x155, 0x2AA.
        // Concatenated bits:
        //   1111111111 0101010101 1010101010
        // = 11111111 11010101 01011010 10101010   -> 4 bytes.
        let data = [0xFF, 0xD5, 0x5A, 0xAA];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_index10().unwrap(), 0x3FF);
        assert_eq!(br.read_index10().unwrap(), 0x155);
        assert_eq!(br.read_index10().unwrap(), 0x2AA);
    }

    #[test]
    fn exhausted_reader_errors() {
        let data = [0xFF];
        let mut br = BitReader::new(&data);
        // 10 bits requested but only 8 are available.
        assert!(br.read_index10().is_err());
    }

    #[test]
    fn unpack_splits_fields() {
        // bit layout: shape=0b1010101 (7b) | sign=1 (1b) | mag=0b10 (2b)
        //           = 10_1010_1110
        #[allow(clippy::unusual_byte_groupings)]
        let raw: u16 = 0b1010101_1_10;
        let u = UnpackedIndex::from_raw(raw);
        assert_eq!(u.shape_index, 0b1010101);
        assert_eq!(u.sign, 1);
        assert_eq!(u.gain_mag, 0b10);
        assert_eq!(u.gain_index(), 0b110);
    }

    #[test]
    fn bits_remaining_shrinks() {
        let data = [0xFF, 0xFF, 0xFF];
        let mut br = BitReader::new(&data);
        assert_eq!(br.bits_remaining(), 24);
        br.read_u32(10).unwrap();
        assert_eq!(br.bits_remaining(), 14);
    }
}
