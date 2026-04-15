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
                    let mult_index = ((entry_number as u64 / idx_div) as usize)
                        % lookup_values;
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
                let base = (entry_number as usize).checked_mul(dim).ok_or_else(|| {
                    Error::invalid("Vorbis VQ type2 entry index overflow")
                })?;
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
        // Linear scan with running bit accumulator. For correctness this is
        // straightforward — performance optimisation (lookup tables) can come
        // later.
        let max_len = self.codeword_lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 {
            return Err(Error::invalid("Vorbis codebook has no entries"));
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
        Err(Error::invalid("Vorbis codebook: no codeword matched"))
    }

    /// Build canonical-Huffman codewords from `codeword_lengths` per Vorbis I §3.2.1.
    pub fn build_decoder(&mut self) -> Result<()> {
        let n = self.codeword_lengths.len();
        self.codewords = vec![0u32; n];
        // Build canonical-Huffman codewords with a "next free codeword per
        // length" table. The simplest correct algorithm: track a `next_code`
        // counter at each length; assign codes in input order.
        // Each entry of length L claims `next_code[L]`, then the next
        // available code at L becomes prev+1; longer-length next codes are
        // updated by left-shifting.
        let max_len: u32 = self.codeword_lengths.iter().copied().max().unwrap_or(0) as u32;
        if max_len == 0 {
            return Ok(()); // No entries to encode; degenerate but not an error.
        }
        let mut next_code = vec![0u32; (max_len + 1) as usize];
        let mut count_per_len = vec![0u32; (max_len + 1) as usize];
        for &l in &self.codeword_lengths {
            if l > 0 {
                count_per_len[l as usize] += 1;
            }
        }
        // Compute first code at each length per the standard canonical Huffman recipe.
        let mut code: u32 = 0;
        for l in 1..=max_len as usize {
            code = (code + count_per_len[l - 1]) << 1;
            next_code[l] = code;
        }
        // Sanity: a complete tree at max length has exactly 2^max_len leaves.
        // Underspecified trees with one entry are allowed. We don't strictly
        // validate completeness here.
        for (i, &l) in self.codeword_lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            self.codewords[i] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
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
            let bits_for_count = bits_needed(entries - current_entry);
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
    while pow_overflow_check(n + 1, dim).map_or(false, |v| v <= entries) {
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

fn bits_needed(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        32 - (value - 1).leading_zeros()
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

    #[test]
    fn bits_needed_basic() {
        // bits_needed(N) = ceil(log2(N)) for N >= 1, 0 for N == 0 or 1.
        assert_eq!(bits_needed(0), 0);
        assert_eq!(bits_needed(1), 0);
        assert_eq!(bits_needed(2), 1);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 2);
        assert_eq!(bits_needed(5), 3);
        assert_eq!(bits_needed(255), 8);
        assert_eq!(bits_needed(256), 8);
        assert_eq!(bits_needed(257), 9);
    }
}
