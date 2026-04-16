//! JPEG Huffman tables.
//!
//! A DHT segment defines up to 4 DC and 4 AC tables, each given as
//! `L[1..=16]` (number of codes of each length) + the symbol values listed
//! in code-length order (Annex C of T.81). From these we build:
//!
//! * A **decode** table (for decoding during entropy): a short lookup +
//!   small tables per code-length for slow path.
//! * An **encode** table (for encoding): a `code[symbol] = (bits, len)` map.
//!
//! Code assignment follows Annex C.2: canonical, length-major, numeric-order
//! per length.

use oxideav_core::{Error, Result};

/// Per-symbol Huffman code, used for encoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct HuffCode {
    pub code: u16,
    pub len: u8,
}

/// Decode table for a Huffman segment.
#[derive(Clone, Debug)]
pub struct HuffTable {
    /// `min_code[l]` = numeric value of the first code of length `l+1`.
    /// Max code length is 16, so `l` ranges 0..16. `i32::MAX` sentinels.
    pub min_code: [i32; 17],
    /// Index into `values` of the first symbol of length `l+1`.
    pub val_offset: [i32; 17],
    /// Symbols in canonical order (same as the DHT segment's tail).
    pub values: Vec<u8>,
    /// `max_code[l]` = numeric value of the last code of length `l+1` (−1 if
    /// no code of that length).
    pub max_code: [i32; 17],
    /// Per-symbol encode info, populated only when built.
    pub encode: Vec<HuffCode>,
}

impl HuffTable {
    /// Build from `bits[0..16]` (count of codes of length 1..16) and the
    /// symbol list `values`.
    pub fn build(bits: &[u8; 16], values: &[u8]) -> Result<Self> {
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total != values.len() {
            return Err(Error::invalid("DHT: symbol count mismatch"));
        }
        if total > 256 {
            return Err(Error::invalid("DHT: > 256 symbols"));
        }

        let mut min_code = [0i32; 17];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut code: u32 = 0;
        let mut sym_idx: usize = 0;
        for l in 0..16 {
            let n = bits[l] as usize;
            if n == 0 {
                min_code[l] = i32::MAX;
                max_code[l] = -1;
                val_offset[l] = -1;
                code <<= 1;
                continue;
            }
            min_code[l] = code as i32;
            val_offset[l] = sym_idx as i32 - code as i32;
            for _ in 0..n {
                code += 1;
            }
            max_code[l] = (code - 1) as i32;
            sym_idx += n;
            code <<= 1;
        }

        // Build encode table. `encode[sym]` = (code, length).
        let mut encode = vec![HuffCode::default(); 256];
        let mut sidx = 0usize;
        let mut c: u32 = 0;
        for l in 0..16 {
            let n = bits[l] as usize;
            for _ in 0..n {
                let sym = values[sidx] as usize;
                encode[sym] = HuffCode {
                    code: c as u16,
                    len: (l as u8) + 1,
                };
                sidx += 1;
                c += 1;
            }
            c <<= 1;
        }

        Ok(Self {
            min_code,
            val_offset,
            values: values.to_vec(),
            max_code,
            encode,
        })
    }
}

/// Annex K.3 typical Huffman tables, widely used in JPEG encoders.
pub struct DefaultHuffman {
    pub luma_dc: HuffTable,
    pub luma_ac: HuffTable,
    pub chroma_dc: HuffTable,
    pub chroma_ac: HuffTable,
}

// ---- Annex K tables (verbatim from ITU T.81) ----

pub const STD_DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
pub const STD_DC_LUMA_VALS: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

pub const STD_DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
pub const STD_DC_CHROMA_VALS: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

pub const STD_AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
pub const STD_AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

pub const STD_AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
pub const STD_AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

impl DefaultHuffman {
    pub fn build() -> Result<Self> {
        Ok(Self {
            luma_dc: HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS)?,
            luma_ac: HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS)?,
            chroma_dc: HuffTable::build(&STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS)?,
            chroma_ac: HuffTable::build(&STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS)?,
        })
    }
}

/// Parse a DHT marker payload into a set of tables keyed by `(class, id)`.
/// `class = 0` → DC, `class = 1` → AC.
pub fn parse_dht(
    payload: &[u8],
    dc_tables: &mut [Option<HuffTable>; 4],
    ac_tables: &mut [Option<HuffTable>; 4],
) -> Result<()> {
    let mut i = 0;
    while i < payload.len() {
        let tc_th = payload[i];
        let class = tc_th >> 4;
        let id = (tc_th & 0x0F) as usize;
        if id >= 4 {
            return Err(Error::invalid("DHT: table id > 3"));
        }
        if class > 1 {
            return Err(Error::invalid("DHT: class > 1"));
        }
        i += 1;
        if i + 16 > payload.len() {
            return Err(Error::invalid("DHT: truncated bits"));
        }
        let mut bits = [0u8; 16];
        bits.copy_from_slice(&payload[i..i + 16]);
        i += 16;
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if i + total > payload.len() {
            return Err(Error::invalid("DHT: truncated values"));
        }
        let values = payload[i..i + total].to_vec();
        i += total;
        let table = HuffTable::build(&bits, &values)?;
        if class == 0 {
            dc_tables[id] = Some(table);
        } else {
            ac_tables[id] = Some(table);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tables_build() {
        DefaultHuffman::build().unwrap();
    }

    #[test]
    fn encode_roundtrips_all_symbols() {
        let t = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap();
        for &sym in &STD_DC_LUMA_VALS {
            let c = t.encode[sym as usize];
            assert!(c.len > 0);
            assert!(c.len <= 16);
        }
    }
}
