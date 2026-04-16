//! AAC Huffman codebooks — ISO/IEC 14496-3 §4.A.1.
//!
//! There are 11 spectral codebooks (1-11) plus 1 scalefactor codebook.
//! Each codebook is supplied as parallel `codes` + `bits` arrays. We
//! decode one symbol by reading bits MSB-first and linearly searching
//! all codes whose `bits` matches.
//!
//! For better performance with large books we group entries by length
//! at module init — the search becomes "advance to current bit length,
//! then look up in a sorted slice of (code, index) pairs". The biggest
//! book (book 11) has 289 entries, max 12 bits — this stays comfortably
//! cache-friendly.

use std::sync::OnceLock;

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::huffman_tables::{
    BOOK10_BITS, BOOK10_CODES, BOOK11_BITS, BOOK11_CODES, BOOK1_BITS, BOOK1_CODES, BOOK2_BITS,
    BOOK2_CODES, BOOK3_BITS, BOOK3_CODES, BOOK4_BITS, BOOK4_CODES, BOOK5_BITS, BOOK5_CODES,
    BOOK6_BITS, BOOK6_CODES, BOOK7_BITS, BOOK7_CODES, BOOK8_BITS, BOOK8_CODES, BOOK9_BITS,
    BOOK9_CODES, SCALEFACTOR_BITS, SCALEFACTOR_CODES,
};

/// One per code length: holds `(code, index)` pairs sorted by `code`.
#[derive(Clone, Debug, Default)]
struct LengthBucket {
    codes: Vec<u32>,
    indices: Vec<u16>,
}

#[derive(Clone, Debug, Default)]
struct LookupTable {
    /// `buckets[bits]` — bits are 1..=max_bits.
    buckets: Vec<LengthBucket>,
}

impl LookupTable {
    fn new(codes: &[u32], bits: &[u8]) -> Self {
        debug_assert_eq!(codes.len(), bits.len());
        let max_bits = bits.iter().copied().max().unwrap_or(0) as usize;
        let mut buckets: Vec<LengthBucket> = vec![LengthBucket::default(); max_bits + 1];
        for (idx, (&c, &b)) in codes.iter().zip(bits.iter()).enumerate() {
            if b == 0 {
                continue;
            }
            let buc = &mut buckets[b as usize];
            buc.codes.push(c);
            buc.indices.push(idx as u16);
        }
        // Sort each bucket so we can binary-search.
        for buc in buckets.iter_mut() {
            let mut paired: Vec<(u32, u16)> = buc
                .codes
                .iter()
                .copied()
                .zip(buc.indices.iter().copied())
                .collect();
            paired.sort_by_key(|&(c, _)| c);
            buc.codes = paired.iter().map(|&(c, _)| c).collect();
            buc.indices = paired.iter().map(|&(_, i)| i).collect();
        }
        Self { buckets }
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let mut acc: u32 = 0;
        for bits in 1..self.buckets.len() {
            acc = (acc << 1) | br.read_u32(1)?;
            let buc = &self.buckets[bits];
            if buc.codes.is_empty() {
                continue;
            }
            if let Ok(pos) = buc.codes.binary_search(&acc) {
                return Ok(buc.indices[pos]);
            }
        }
        Err(Error::invalid("aac huffman: code not found"))
    }
}

/// Construct `codes` from a u16 source slice.
fn lookup_from_u16(codes: &[u16], bits: &[u8]) -> LookupTable {
    let codes_u32: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
    LookupTable::new(&codes_u32, bits)
}

/// Spectral codebook descriptor.
pub struct SpectralBook {
    /// Dimensionality: 4 for books 1-4, 2 for books 5-11.
    pub dim: u8,
    /// Maximum absolute amplitude (LAV) of an unsigned codebook entry.
    pub lav: u8,
    /// True if the codebook is signed (each digit ranges in [-lav, lav]).
    pub signed: bool,
    /// True if amplitude can escape (only book 11, dim=2, lav=16).
    pub escape: bool,
    pub codes: &'static [u16],
    pub bits: &'static [u8],
    table: OnceLock<LookupTable>,
}

impl SpectralBook {
    pub const fn new(
        dim: u8,
        lav: u8,
        signed: bool,
        escape: bool,
        codes: &'static [u16],
        bits: &'static [u8],
    ) -> Self {
        Self {
            dim,
            lav,
            signed,
            escape,
            codes,
            bits,
            table: OnceLock::new(),
        }
    }

    fn table(&self) -> &LookupTable {
        self.table
            .get_or_init(|| lookup_from_u16(self.codes, self.bits))
    }
}

/// Decoded spectral coefficients (up to 4 entries).
pub type SpectralValues = [i16; 4];

/// Decode one Huffman codeword from `book`, then unpack it into 2 or 4
/// signed amplitudes per the codebook semantics. Returns `dim` valid
/// entries in the leading slots of the return value.
pub fn decode_spectral(
    br: &mut BitReader<'_>,
    book: &'static SpectralBook,
) -> Result<SpectralValues> {
    let idx = book.table().decode(br)?;
    let dim = book.dim as usize;
    let lav = book.lav as i16;
    let mut out = [0i16; 4];

    if book.signed {
        // Signed: split idx into `dim` mod-(2*lav+1) digits, each in [-lav, lav].
        let modulo = (lav as u32 * 2 + 1) as u16;
        let mut v = idx;
        for i in (0..dim).rev() {
            out[i] = (v % modulo) as i16 - lav;
            v /= modulo;
        }
    } else {
        // Unsigned: split idx into `dim` mod-(lav+1) digits in [0, lav].
        let modulo = (lav as u32 + 1) as u16;
        let mut v = idx;
        for i in (0..dim).rev() {
            out[i] = (v % modulo) as i16;
            v /= modulo;
        }
        // Then read sign bits for each non-zero coefficient.
        for i in 0..dim {
            if out[i] != 0 && br.read_bit()? {
                out[i] = -out[i];
            }
        }
    }

    if book.escape {
        // Book 11: ±16 means "escape — read amplitude from the bitstream".
        // The sign of the original ±16 is preserved.
        for i in 0..dim {
            if out[i].unsigned_abs() == 16 {
                let amp = read_escape(br)?;
                out[i] = if out[i] < 0 { -amp } else { amp };
            }
        }
    }
    Ok(out)
}

/// Read an escape-coded amplitude as used by codebook 11. The code is a
/// unary-prefix `1...1 0` of length `prefix` ones followed by `prefix+4`
/// raw bits — value is `(1 << (prefix+4)) | raw`.
pub fn read_escape(br: &mut BitReader<'_>) -> Result<i16> {
    let mut prefix: u32 = 0;
    while br.read_bit()? {
        prefix += 1;
        if prefix > 24 {
            return Err(Error::invalid("aac huffman: escape prefix too long"));
        }
    }
    let raw = br.read_u32(prefix + 4)?;
    Ok((((1u32) << (prefix + 4)) | raw) as i16)
}

pub static BOOK1: SpectralBook = SpectralBook::new(4, 1, true, false, BOOK1_CODES, BOOK1_BITS);
pub static BOOK2: SpectralBook = SpectralBook::new(4, 1, true, false, BOOK2_CODES, BOOK2_BITS);
pub static BOOK3: SpectralBook = SpectralBook::new(4, 2, false, false, BOOK3_CODES, BOOK3_BITS);
pub static BOOK4: SpectralBook = SpectralBook::new(4, 2, false, false, BOOK4_CODES, BOOK4_BITS);
pub static BOOK5: SpectralBook = SpectralBook::new(2, 4, true, false, BOOK5_CODES, BOOK5_BITS);
pub static BOOK6: SpectralBook = SpectralBook::new(2, 4, true, false, BOOK6_CODES, BOOK6_BITS);
pub static BOOK7: SpectralBook = SpectralBook::new(2, 7, false, false, BOOK7_CODES, BOOK7_BITS);
pub static BOOK8: SpectralBook = SpectralBook::new(2, 7, false, false, BOOK8_CODES, BOOK8_BITS);
pub static BOOK9: SpectralBook = SpectralBook::new(2, 12, false, false, BOOK9_CODES, BOOK9_BITS);
pub static BOOK10: SpectralBook = SpectralBook::new(2, 12, false, false, BOOK10_CODES, BOOK10_BITS);
pub static BOOK11: SpectralBook = SpectralBook::new(2, 16, false, true, BOOK11_CODES, BOOK11_BITS);

/// Resolve a spectral codebook index (1-based) into its descriptor.
pub fn spectral_book(idx: u8) -> Result<&'static SpectralBook> {
    Ok(match idx {
        1 => &BOOK1,
        2 => &BOOK2,
        3 => &BOOK3,
        4 => &BOOK4,
        5 => &BOOK5,
        6 => &BOOK6,
        7 => &BOOK7,
        8 => &BOOK8,
        9 => &BOOK9,
        10 => &BOOK10,
        11 => &BOOK11,
        _ => return Err(Error::invalid("aac huffman: invalid codebook index")),
    })
}

static SCALEFACTOR_TABLE: OnceLock<LookupTable> = OnceLock::new();

fn scalefactor_table() -> &'static LookupTable {
    SCALEFACTOR_TABLE.get_or_init(|| LookupTable::new(SCALEFACTOR_CODES, SCALEFACTOR_BITS))
}

/// Decode the next scalefactor delta code. Returns delta in [-60, 60].
pub fn decode_scalefactor_delta(br: &mut BitReader<'_>) -> Result<i32> {
    let idx = scalefactor_table().decode(br)?;
    Ok(idx as i32 - 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_table_has_entries() {
        for book in [
            &BOOK1, &BOOK2, &BOOK3, &BOOK4, &BOOK5, &BOOK6, &BOOK7, &BOOK8, &BOOK9, &BOOK10,
            &BOOK11,
        ] {
            // Ensure construction succeeds.
            let _ = book.table();
        }
        let _ = scalefactor_table();
    }

    #[test]
    fn book7_zero_pair_is_one_bit() {
        // Book 7 has a 1-bit code "0" for symbol 0 (the (0,0) pair).
        let buf = [0u8; 1];
        let mut br = BitReader::new(&buf);
        let v = decode_spectral(&mut br, &BOOK7).unwrap();
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 0);
    }

    #[test]
    fn scalefactor_zero_is_one_bit() {
        // Symbol 60 (delta 0) is the 1-bit code "0".
        let buf = [0u8; 1];
        let mut br = BitReader::new(&buf);
        let d = decode_scalefactor_delta(&mut br).unwrap();
        assert_eq!(d, 0);
    }
}
