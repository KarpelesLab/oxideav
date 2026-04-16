//! Tiny linear-scan VLC decoder.
//!
//! MPEG-1 VLC tables from Annex B of ISO/IEC 11172-2 are listed in the spec as
//! (codeword, symbol) pairs with codeword lengths up to ~17 bits. A linear
//! walk over the table per symbol is fast enough for textbook-scale decoders
//! and keeps the tables obvious to audit against the spec.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// One entry in a VLC table. `code` occupies the low `bits` bits (MSB-first).
#[derive(Clone, Copy, Debug)]
pub struct VlcEntry<T: Copy> {
    pub code: u32,
    pub bits: u8,
    pub value: T,
}

impl<T: Copy> VlcEntry<T> {
    pub const fn new(bits: u8, code: u32, value: T) -> Self {
        Self { code, bits, value }
    }
}

/// Decode one symbol from the table by peeking up to `max_bits` and picking the
/// entry whose code matches the current top bits after the appropriate shift.
pub fn decode<T: Copy>(br: &mut BitReader<'_>, table: &[VlcEntry<T>]) -> Result<T> {
    // Determine the longest code in the table so we know how many bits to peek.
    let max_bits = table.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    if max_bits == 0 {
        return Err(Error::invalid("vlc: empty table"));
    }
    // Make sure enough bits are available to peek max_bits; if not, peek
    // fewer — incomplete-at-EOF should fall through to "no match".
    let remaining = br.bits_remaining() as u32;
    let peek_bits = max_bits.min(remaining);
    if peek_bits == 0 {
        return Err(Error::invalid("vlc: no bits available"));
    }
    let peeked = br.peek_u32(peek_bits)?;
    // peeked occupies the top `peek_bits` bits of its logical field; to align
    // with `max_bits`, left-shift by the difference.
    let peeked_full = peeked << (max_bits - peek_bits);
    for e in table {
        if (e.bits as u32) > peek_bits {
            continue;
        }
        let shift = max_bits - e.bits as u32;
        let prefix = peeked_full >> shift;
        if prefix == e.code {
            br.consume(e.bits as u32)?;
            return Ok(e.value);
        }
    }
    Err(Error::invalid("vlc: no matching codeword"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_dc_luma_size() {
        // Table B-12 luma DC size codes; pick a couple and verify.
        // size=0: 100 (3 bits)
        // size=1: 00
        // size=2: 01
        // size=3: 101
        // size=6: 11110
        let table = crate::tables::dct_dc::luma();

        // Bitstream `100 00 01 101 11110 ...`
        // Pack into bytes MSB-first: 1_0000_0_01_1_01_1_11_10
        // Combined string: 100 00 01 101 11110 = "1000001101 11110" → 15 bits
        // padded to 16: "1000001101111100"
        let v: u16 = 0b1000_0011_0111_1100;
        let data = [(v >> 8) as u8, (v & 0xff) as u8];
        let mut br = BitReader::new(&data);

        assert_eq!(decode(&mut br, table).unwrap(), 0); // 100
        assert_eq!(decode(&mut br, table).unwrap(), 1); // 00
        assert_eq!(decode(&mut br, table).unwrap(), 2); // 01
        assert_eq!(decode(&mut br, table).unwrap(), 3); // 101
        assert_eq!(decode(&mut br, table).unwrap(), 6); // 11110
    }
}
