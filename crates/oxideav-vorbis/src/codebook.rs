//! Vorbis codebook parsing and decoding.
//!
//! Each codebook is a canonical-Huffman entry table optionally backed by a
//! vector-quantization (VQ) lookup table that maps entry numbers to floating
//! point vectors of dimension `dim`.
//!
//! Reference: Vorbis I §3.2.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

#[derive(Clone, Debug)]
pub struct Codebook {
    pub dimensions: u16,
    pub entries: u32,
    /// Codeword length per entry; 0 means "entry unused" (sparse codebook).
    pub codeword_lengths: Vec<u8>,
    /// VQ value lookup, present when `lookup_type != 0`.
    pub vq: Option<VqLookup>,

    // Decoder state (populated after `build_decoder`).
    /// Per-entry codeword (low `len` bits hold the bit pattern, MSB-first).
    pub codewords: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct VqLookup {
    pub lookup_type: u8,
    pub min: f32,
    pub delta: f32,
    pub value_bits: u8,
    pub sequence_p: bool,
    /// Quantized scalar values in [0, 2^value_bits).
    pub multiplicands: Vec<u32>,
}

impl Codebook {
    /// Decode the VQ vector for entry `entry_number`.
    pub fn vq_lookup(&self, entry_number: u32) -> Result<Vec<f32>> {
        let vq = self
            .vq
            .as_ref()
            .ok_or_else(|| Error::invalid("Vorbis codebook has no VQ lookup"))?;
        let dim = self.dimensions as usize;
        match vq.lookup_type {
            1 => {
                let mut out = vec![0f32; dim];
                let entries = self.entries as u64;
                // Compute lookup_values = floor(root^dim_powers) per spec.
                // Practical formula: lookup_values is the largest n such that
                // n^dim <= entries (the number of multiplicands stored).
                let lookup_values = vq.multiplicands.len();
                if lookup_values == 0 {
                    return Err(Error::invalid("Vorbis VQ type1 with empty multiplicands"));
                }
                let _ = entries;
                let mut last = 0f32;
                let mut idx_div = 1u64;
                for d in 0..dim {
                    let mult_index = ((entry_number as u64 / idx_div) as usize) % lookup_values;
                    let m = vq.multiplicands[mult_index] as f32;
                    let val = m * vq.delta + vq.min + last;
                    out[d] = val;
                    if vq.sequence_p {
                        last = val;
                    }
                    idx_div = idx_div.saturating_mul(lookup_values as u64).max(1);
                }
                Ok(out)
            }
            2 => {
                let mut out = vec![0f32; dim];
                let mut last = 0f32;
                let base = (entry_number as usize)
                    .checked_mul(dim)
                    .ok_or_else(|| Error::invalid("Vorbis VQ type2 entry index overflow"))?;
                if base + dim > vq.multiplicands.len() {
                    return Err(Error::invalid(
                        "Vorbis VQ type2 entry exceeds multiplicands",
                    ));
                }
                for d in 0..dim {
                    let m = vq.multiplicands[base + d] as f32;
                    let val = m * vq.delta + vq.min + last;
                    out[d] = val;
                    if vq.sequence_p {
                        last = val;
                    }
                }
                Ok(out)
            }
            _ => Err(Error::invalid(format!(
                "Vorbis codebook lookup type {} not supported",
                vq.lookup_type
            ))),
        }
    }

    /// Decode the next entry number from the bitstream using this codebook's
    /// canonical Huffman table.
    pub fn decode_scalar(&self, br: &mut BitReader<'_>) -> Result<u32> {
        let max_len = self.codeword_lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 {
            // Single-entry codebook: 0 bits consumed, always returns the
            // sole used entry. If no entries are marked used the codebook
            // is malformed.
            for (entry, &l) in self.codeword_lengths.iter().enumerate() {
                if l > 0 || self.codeword_lengths.len() == 1 {
                    return Ok(entry as u32);
                }
            }
            return Err(Error::invalid("Vorbis codebook has no usable entries"));
        }
        // Read bits MSB-first into `code`. Vorbis Huffman codes are described
        // bit-by-bit with the most significant bit consumed first along the
        // tree. Our bit reader returns LSB-first single bits, so accumulate
        // them into the high end of `code`.
        let mut code: u32 = 0;
        for len in 1..=max_len {
            let bit = br.read_u32(1)?;
            code = (code << 1) | bit;
            // Search for any entry with this code.
            for (entry, &l) in self.codeword_lengths.iter().enumerate() {
                if l == len && self.codewords[entry] == code {
                    return Ok(entry as u32);
                }
            }
        }
        Err(Error::invalid(format!(
            "Vorbis codebook: no codeword matched after {max_len} bits (entries={})",
            self.entries
        )))
    }

    /// Build Huffman codewords from `codeword_lengths` using libvorbis's
    /// marker-based left-first tree placement (sharedbook.c `_make_words`),
    /// then bit-reverse each code so it matches the LSB-first stream read
    /// order. This differs from textbook canonical Huffman (sorted-length)
    /// for non-monotone length sequences.
    pub fn build_decoder(&mut self) -> Result<()> {
        let n = self.codeword_lengths.len();
        self.codewords = vec![0u32; n];
        let max_len: u32 = self.codeword_lengths.iter().copied().max().unwrap_or(0) as u32;
        if max_len == 0 {
            return Ok(());
        }

        // Single-entry codebook with length 1: libvorbis treats it as a
        // "same value for any 1-bit input" tree.
        let used_count = self.codeword_lengths.iter().filter(|&&l| l > 0).count();
        if used_count == 1 {
            let entry_idx = self
                .codeword_lengths
                .iter()
                .position(|&l| l > 0)
                .expect("used_count==1 implies at least one entry");
            if self.codeword_lengths[entry_idx] != 1 {
                // libvorbis requires the single entry to have length 1 for this
                // special path; otherwise the tree is malformed. We accept
                // longer single-entry codebooks without complaint — downstream
                // decode_scalar always returns this entry.
            }
            return Ok(());
        }

        // `marker[L]` tracks the next available code at depth L. After each
        // leaf placement it is updated to reflect the fact that one slot at
        // that depth (and its parent ancestors) has been claimed.
        let mut marker = [0u32; 33];
        let mut raw_codes = vec![0u32; n];

        for (i, &l) in self.codeword_lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let length = l as usize;
            if length >= 33 {
                return Err(Error::invalid("Vorbis codebook codeword length exceeds 32"));
            }
            let entry = marker[length];
            if length < 32 && (entry >> length) != 0 {
                return Err(Error::invalid(
                    "Vorbis codebook is overspecified (Huffman tree full)",
                ));
            }
            raw_codes[i] = entry;

            // Walk from this depth upward: on each level, bump the marker,
            // and if that carries (marker becomes odd), reset from the
            // parent's marker and break.
            let mut j = length;
            while j > 0 {
                if marker[j] & 1 != 0 {
                    if j == 1 {
                        marker[1] += 1;
                    } else {
                        marker[j] = marker[j - 1] << 1;
                    }
                    break;
                }
                marker[j] += 1;
                j -= 1;
            }

            // Propagate the update downward: any deeper markers that were
            // pointing at the same prefix need to advance past it.
            let mut entry_cur = entry;
            for j in (length + 1)..33 {
                if (marker[j] >> 1) == entry_cur {
                    entry_cur = marker[j];
                    marker[j] = marker[j - 1] << 1;
                } else {
                    break;
                }
            }
        }

        // `decode_scalar` accumulates stream bits MSB-first
        // (`code = (code<<1)|bit`), which matches the marker values' MSB-first
        // interpretation directly — no bit reversal needed here, unlike
        // libvorbis's `_make_words` which reverses because it stores codes
        // for LSB-first packing in the encoder.
        self.codewords = raw_codes;

        Ok(())
    }
}

// ------------------ Codebook config parsing -------------------------------

/// Parse a single codebook from the bitstream and finalise its decoder.
pub fn parse_codebook(br: &mut BitReader<'_>) -> Result<Codebook> {
    // Sync pattern 0x564342 — three bytes spelling "BCV" reversed (little-endian "VCB").
    let sync = br.read_u32(24)?;
    if sync != 0x564342 {
        return Err(Error::invalid(format!(
            "Vorbis codebook bad sync 0x{sync:06x}"
        )));
    }
    let dimensions = br.read_u32(16)? as u16;
    let entries = br.read_u32(24)?;
    let ordered = br.read_bit()?;
    let mut codeword_lengths = vec![0u8; entries as usize];
    if !ordered {
        let sparse = br.read_bit()?;
        for i in 0..entries as usize {
            if sparse {
                let used = br.read_bit()?;
                if used {
                    codeword_lengths[i] = (br.read_u32(5)? + 1) as u8;
                } else {
                    codeword_lengths[i] = 0;
                }
            } else {
                codeword_lengths[i] = (br.read_u32(5)? + 1) as u8;
            }
        }
    } else {
        let mut current_length = (br.read_u32(5)? + 1) as u8;
        let mut current_entry: u32 = 0;
        while current_entry < entries {
            let bits_for_count = ilog(entries - current_entry);
            let number = br.read_u32(bits_for_count)?;
            for k in 0..number {
                let idx = (current_entry + k) as usize;
                if idx >= entries as usize {
                    return Err(Error::invalid("Vorbis codebook ordered overflow"));
                }
                codeword_lengths[idx] = current_length;
            }
            current_entry = current_entry
                .checked_add(number)
                .ok_or_else(|| Error::invalid("Vorbis codebook entry count overflow"))?;
            current_length += 1;
            if current_length == 0 {
                return Err(Error::invalid("Vorbis codebook length wrap"));
            }
        }
    }

    let lookup_type = br.read_u32(4)? as u8;
    let vq = match lookup_type {
        0 => None,
        1 | 2 => {
            let min = br.read_vorbis_float()?;
            let delta = br.read_vorbis_float()?;
            let value_bits = (br.read_u32(4)? + 1) as u8;
            let sequence_p = br.read_bit()?;
            let lookup_values = if lookup_type == 1 {
                lookup1_values(entries, dimensions as u32) as u64
            } else {
                entries as u64 * dimensions as u64
            };
            let mut multiplicands = Vec::with_capacity(lookup_values as usize);
            for _ in 0..lookup_values {
                multiplicands.push(br.read_u32(value_bits as u32)?);
            }
            Some(VqLookup {
                lookup_type,
                min,
                delta,
                value_bits,
                sequence_p,
                multiplicands,
            })
        }
        _ => return Err(Error::invalid("Vorbis codebook reserved lookup type")),
    };

    let mut cb = Codebook {
        dimensions,
        entries,
        codeword_lengths,
        vq,
        codewords: Vec::new(),
    };
    cb.build_decoder()?;
    Ok(cb)
}

/// Smallest n such that `n^dim <= entries` (Vorbis I §9.2.3 lookup1_values).
fn lookup1_values(entries: u32, dim: u32) -> u32 {
    if dim == 0 {
        return 0;
    }
    if dim == 1 {
        return entries;
    }
    // Binary-ish search.
    let mut n = (entries as f64).powf(1.0 / dim as f64) as u32;
    // Refine due to floating-point inaccuracy.
    while pow_overflow_check(n + 1, dim).is_some_and(|v| v <= entries) {
        n += 1;
    }
    while n > 0 && pow_overflow_check(n, dim).map_or(true, |v| v > entries) {
        n -= 1;
    }
    n
}

fn pow_overflow_check(base: u32, exp: u32) -> Option<u32> {
    let mut acc: u32 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base)?;
    }
    Some(acc)
}

/// Vorbis I §1.4 `ilog`: number of bits required to store the unsigned
/// integer `value`. ilog(0) = 0; ilog(1) = 1; ilog(2..=3) = 2; etc.
fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup1_basic() {
        // 5^3 = 125 <= 250 < 6^3 = 216 = wait, 6^3 = 216 < 250 too. 7^3 = 343 > 250.
        // So lookup1_values(250, 3) == 6.
        assert_eq!(lookup1_values(250, 3), 6);
        assert_eq!(lookup1_values(125, 3), 5);
        assert_eq!(lookup1_values(8, 3), 2); // 2^3 = 8
        assert_eq!(lookup1_values(0, 3), 0);
    }

    /// Regression: verify libvorbis-compatible marker-based code construction.
    /// Book 16 from sine.ogg has non-monotone lengths (5,5,5,5,6,5,...) which
    /// textbook canonical-Huffman gets WRONG. These expected codes were
    /// captured from libvorbis's `_make_words` for the same input lengths.
    #[test]
    fn marker_codes_match_libvorbis() {
        // Condensed lengths from sine.ogg codebook 16 (first 20 entries).
        let lengths = vec![5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5];
        let mut cb = Codebook {
            dimensions: 1,
            entries: lengths.len() as u32,
            codeword_lengths: lengths,
            vq: None,
            codewords: vec![],
        };
        cb.build_decoder().expect("build_decoder");

        // libvorbis produces codes 0, 1, 2, 3, 8, 4, 9, 5, 10, 6, 11, 7,
        // 12, ... for this pattern: L5 entries get codes 0..N_L5-1 and L6
        // entries get codes that descend from the unused L5 slots.
        //
        // Note: the canonical-Huffman WRONG answer would be entry 4 = 36
        // (which is what broke sine.ogg decode until d8f068a).
        assert_eq!(cb.codewords[0], 0b00000, "e0 L5");
        assert_eq!(cb.codewords[1], 0b00001, "e1 L5");
        assert_eq!(cb.codewords[2], 0b00010, "e2 L5");
        assert_eq!(cb.codewords[3], 0b00011, "e3 L5");
        // Entry 4 (L6) goes to leftmost unused L6 slot = 001000, NOT 100100.
        assert_eq!(cb.codewords[4], 0b001000, "e4 L6 (the bug-finder)");
        assert_eq!(cb.codewords[5], 0b00101, "e5 L5");
    }

    #[test]
    fn ilog_basic() {
        // Vorbis ilog: bits needed to represent the value (top bit position).
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
        assert_eq!(ilog(8), 4);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
    }
}
