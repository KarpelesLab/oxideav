//! Theora Setup header (packet 2). Spec §6.4.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::huffman::HuffmanTree;

/// Number of quantiser-base matrices stored.
const NBASE_MAX: usize = 384; // spec cap; usual streams store ≤ 16

/// Per-frame-type (intra, inter) by channel (luma, chroma).
#[derive(Clone, Debug)]
pub struct QuantRanges {
    /// For each of 6 (qti×ci×pli) combinations, the range boundaries.
    /// `nbms[i]` is the number of *base matrix indices* used.
    pub nbms: [u8; 6],
    /// For each combo, the size-1 array of range-widths (length nbms[i])
    /// plus the base-matrix indices (length nbms[i]+1).
    pub widths: [Vec<u8>; 6],
    pub bmis: [Vec<u8>; 6],
}

#[derive(Clone, Debug)]
pub struct Setup {
    /// Loop filter limit table; 64 entries.
    pub lflims: [u8; 64],
    /// AC scale table — 64 entries.
    pub ac_scale: [u16; 64],
    /// DC scale table — 64 entries.
    pub dc_scale: [u16; 64],
    /// Base quant matrices — up to NBASE_MAX of them.
    pub bms: Vec<[u8; 64]>,
    pub ranges: QuantRanges,
    /// 80 Huffman trees (16 HTI × 5 token categories). Indexed as
    /// `huffs[hti * 5 + hg]`.
    pub huffs: Vec<HuffmanTree>,
}

pub fn parse_setup_header(packet: &[u8]) -> Result<Setup> {
    if packet.len() < 7 {
        return Err(Error::invalid("Theora setup header too short"));
    }
    if packet[0] != 0x82 || &packet[1..7] != b"theora" {
        return Err(Error::invalid("not a Theora setup header"));
    }
    let mut br = BitReader::new(&packet[7..]);

    // Loop filter limit table: 64 × nbits entries.
    let nbits = br.read_u32(3)?;
    let mut lflims = [0u8; 64];
    for v in &mut lflims {
        *v = if nbits == 0 {
            0
        } else {
            br.read_u32(nbits)? as u8
        };
    }

    // Quant AC scale table: 64 × (ac_scale_nbits+1).
    let ac_scale_nbits = br.read_u32(4)? + 1;
    let mut ac_scale = [0u16; 64];
    for v in &mut ac_scale {
        *v = br.read_u32(ac_scale_nbits)? as u16;
    }
    // Quant DC scale table.
    let dc_scale_nbits = br.read_u32(4)? + 1;
    let mut dc_scale = [0u16; 64];
    for v in &mut dc_scale {
        *v = br.read_u32(dc_scale_nbits)? as u16;
    }

    // Base matrices: NBMS (nbms_nbits + 1) + 1.
    let nbms = br.read_u32(9)? as usize + 1;
    if nbms > NBASE_MAX {
        return Err(Error::invalid(format!(
            "Theora setup: too many base matrices: {nbms}"
        )));
    }
    let mut bms: Vec<[u8; 64]> = Vec::with_capacity(nbms);
    for _ in 0..nbms {
        let mut m = [0u8; 64];
        for v in &mut m {
            *v = br.read_u32(8)? as u8;
        }
        bms.push(m);
    }

    // Per-quant-range table per (qti×pli).
    let ranges = parse_quant_ranges(&mut br, nbms)?;

    // 80 Huffman trees: 16 HTI × 5 groups.
    let mut huffs = Vec::with_capacity(80);
    for _ in 0..80 {
        huffs.push(HuffmanTree::read_from(&mut br)?);
    }

    Ok(Setup {
        lflims,
        ac_scale,
        dc_scale,
        bms,
        ranges,
        huffs,
    })
}

fn parse_quant_ranges(br: &mut BitReader<'_>, nbms: usize) -> Result<QuantRanges> {
    // Spec §6.4.3 algorithm: for each of 6 tuples (qti, pli) with the intra/
    // chroma-luma-copy rules, if the "new table" flag is set (or qti==0 pli==0
    // always), parse the range widths and base-matrix indices; otherwise copy
    // from a previous slot.
    let mut nbms_arr = [0u8; 6];
    let mut widths: [Vec<u8>; 6] = Default::default();
    let mut bmis: [Vec<u8>; 6] = Default::default();
    let bmi_bits = bits_needed_for(nbms.saturating_sub(1) as u32);

    let mut qpi = 0usize;
    for qti in 0..2 {
        for pli in 0..3 {
            let mut newqr_flag = true;
            if qti > 0 || pli > 0 {
                newqr_flag = br.read_bit()?;
            }
            if !newqr_flag {
                // "rpqr" — copy from a previous slot.
                let rpqr = if qti > 0 { br.read_bit()? } else { false };
                let src = if rpqr { qpi - 3 } else { qpi - 1 };
                nbms_arr[qpi] = nbms_arr[src];
                widths[qpi] = widths[src].clone();
                bmis[qpi] = bmis[src].clone();
            } else {
                // Reproduce libtheora's loop: always read (width, next_bmi)
                // pairs until qi reaches 63. The last `next_bmi` is read but
                // has no further use; we still store it for completeness.
                let mut current_sz = Vec::new();
                let mut current_idx = Vec::new();
                let first_bmi = if bmi_bits == 0 {
                    0
                } else {
                    br.read_u32(bmi_bits)? as u8
                };
                current_idx.push(first_bmi);
                let mut qri = 0usize;
                let mut qi = 0u32;
                while qi < 63 {
                    let widths_bits = bits_needed_for(62 - qi);
                    let width_m1 = br.read_u32(widths_bits)?;
                    let width = width_m1 as u8 + 1;
                    current_sz.push(width);
                    qi += width as u32;
                    let next_bmi = if bmi_bits == 0 {
                        0
                    } else {
                        br.read_u32(bmi_bits)? as u8
                    };
                    current_idx.push(next_bmi);
                    qri += 1;
                    if qri > 63 {
                        return Err(Error::invalid("Theora quant-range overflow"));
                    }
                }
                if qi > 63 {
                    return Err(Error::invalid("Theora quant-range widths overflow past 63"));
                }
                nbms_arr[qpi] = qri as u8;
                widths[qpi] = current_sz;
                bmis[qpi] = current_idx;
            }
            qpi += 1;
        }
    }
    Ok(QuantRanges {
        nbms: nbms_arr,
        widths,
        bmis,
    })
}

/// Theora `ilog(v)` = position of the highest set bit + 1; `ilog(0)=0`.
/// Equivalent to `floor(log2(v)) + 1` for `v > 0`.
fn bits_needed_for(max: u32) -> u32 {
    if max == 0 {
        return 0;
    }
    32 - max.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short() {
        assert!(parse_setup_header(b"").is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let p = [0u8, b't', b'h', b'e', b'o', b'r', b'a'];
        assert!(parse_setup_header(&p).is_err());
    }

    #[test]
    fn ilog_works() {
        assert_eq!(bits_needed_for(0), 0);
        assert_eq!(bits_needed_for(1), 1);
        assert_eq!(bits_needed_for(2), 2);
        assert_eq!(bits_needed_for(7), 3);
        assert_eq!(bits_needed_for(8), 4);
    }
}
