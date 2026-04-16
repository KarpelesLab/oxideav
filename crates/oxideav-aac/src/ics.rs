//! Individual Channel Stream (ICS) — ISO/IEC 14496-3 §4.6.2.
//!
//! Decodes ics_info, scalefactor data, spectral data, and runs inverse
//! quantisation + scalefactor reconstruction. This is the bulk of an AAC
//! channel pair decode — both the SCE (mono) and CPE (stereo) elements
//! call into this module.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::huffman::{decode_scalefactor_delta, decode_spectral, spectral_book};
use crate::sfband::{num_swb_long, num_swb_short, SWB_LONG, SWB_SHORT};
use crate::syntax::{WindowSequence, WindowShape};

/// Maximum scalefactor bands per group (long-window worst case).
pub const MAX_SFB: usize = 51;
/// Number of spectral coefficients in a single AAC frame (long block).
pub const SPEC_LEN: usize = 1024;
/// Special codebook IDs.
pub const ZERO_HCB: u8 = 0;
pub const FIRST_PAIR_HCB: u8 = 5;
pub const ESC_HCB: u8 = 11;
pub const NOISE_HCB: u8 = 13;
pub const INTENSITY_HCB2: u8 = 14;
pub const INTENSITY_HCB: u8 = 15;

#[derive(Clone, Debug, Default)]
pub struct IcsInfo {
    pub window_sequence: WindowSequence,
    pub window_shape: WindowShape,
    pub max_sfb: u8,
    pub num_window_groups: u8,
    /// `window_group_length[g]` — number of windows in group `g` (sum=8 for short, =1 for long).
    pub window_group_length: [u8; 8],
    /// `scale_factor_grouping` raw bits (only meaningful for short-windows).
    pub scale_factor_grouping: u8,
    pub predictor_data_present: bool,
    pub sf_index: u8,
}

impl IcsInfo {
    pub fn num_swb(&self) -> usize {
        if self.window_sequence.is_eight_short() {
            num_swb_short(self.sf_index)
        } else {
            num_swb_long(self.sf_index)
        }
    }
}

pub fn parse_ics_info(br: &mut BitReader<'_>, sf_index: u8) -> Result<IcsInfo> {
    let _ics_reserved_bit = br.read_bit()?;
    let window_sequence = WindowSequence::from_u32(br.read_u32(2)?);
    let window_shape = WindowShape::from_bit(br.read_u32(1)?);
    let mut info = IcsInfo {
        window_sequence,
        window_shape,
        sf_index,
        ..Default::default()
    };

    if window_sequence.is_eight_short() {
        info.max_sfb = br.read_u32(4)? as u8;
        info.scale_factor_grouping = br.read_u32(7)? as u8;
        // Bit 6 of grouping = group with window 1, bit 5 = win 2, ..., bit 0 = win 7.
        // Window 0 always starts a new group.
        let mut groups = 1u8;
        let mut lengths = [1u8; 8];
        let mut cur_len = 1u8;
        for w in 1..8 {
            let bit = (info.scale_factor_grouping >> (6 - (w - 1))) & 1;
            if bit == 1 {
                cur_len += 1;
            } else {
                lengths[(groups - 1) as usize] = cur_len;
                groups += 1;
                cur_len = 1;
            }
        }
        lengths[(groups - 1) as usize] = cur_len;
        info.num_window_groups = groups;
        info.window_group_length = lengths;
    } else {
        info.max_sfb = br.read_u32(6)? as u8;
        info.predictor_data_present = br.read_bit()?;
        if info.predictor_data_present {
            // For AAC-LC, predictor_data_present must be 0; if it's set, refuse.
            return Err(Error::unsupported(
                "AAC: predictor_data_present in LC stream — Main/LTP not supported",
            ));
        }
        info.num_window_groups = 1;
        info.window_group_length[0] = 1;
    }

    Ok(info)
}

/// Section data: codebook + length-in-sfbs per (group, section).
#[derive(Clone, Debug, Default)]
pub struct SectionData {
    /// `sfb_cb[g * max_sfb + sfb]` — codebook index for that band/group.
    pub sfb_cb: Vec<u8>,
}

impl SectionData {
    pub fn empty() -> Self {
        Self::default()
    }
}

pub fn parse_section_data(br: &mut BitReader<'_>, info: &IcsInfo) -> Result<SectionData> {
    let max_sfb = info.max_sfb as usize;
    let groups = info.num_window_groups as usize;
    let sect_esc_val = if info.window_sequence.is_eight_short() {
        (1u32 << 3) - 1
    } else {
        (1u32 << 5) - 1
    };
    let sect_bits = if info.window_sequence.is_eight_short() {
        3
    } else {
        5
    };

    let mut sfb_cb = vec![0u8; groups * max_sfb];
    for g in 0..groups {
        let mut k = 0usize;
        while k < max_sfb {
            let cb = br.read_u32(4)? as u8;
            let mut len = 0u32;
            loop {
                let inc = br.read_u32(sect_bits)?;
                len += inc;
                if inc != sect_esc_val {
                    break;
                }
            }
            for _ in 0..len {
                if k >= max_sfb {
                    return Err(Error::invalid("AAC: section overruns max_sfb"));
                }
                sfb_cb[g * max_sfb + k] = cb;
                k += 1;
            }
        }
    }
    Ok(SectionData { sfb_cb })
}

/// Decode scalefactor data — fills `sf[g * max_sfb + sfb]` with quantised
/// scalefactors per ISO §4.6.2.3.
///
/// `global_gain` is the seed; deltas are decoded from the SF Huffman code.
/// IS bands and noise (PNS) bands use a separate accumulator/seed each.
pub fn parse_scalefactors(
    br: &mut BitReader<'_>,
    info: &IcsInfo,
    sec: &SectionData,
    global_gain: u8,
) -> Result<Vec<i32>> {
    let max_sfb = info.max_sfb as usize;
    let groups = info.num_window_groups as usize;
    let mut sf = vec![0i32; groups * max_sfb];

    let mut g_gain: i32 = global_gain as i32;
    let mut g_noise: i32 = global_gain as i32 - 90; // PNS seed (spec §4.6.2.3)
    let mut g_is: i32 = 0; // Intensity seed

    let mut noise_seed_set = false;

    for g in 0..groups {
        for sfb in 0..max_sfb {
            let cb = sec.sfb_cb[g * max_sfb + sfb];
            match cb {
                ZERO_HCB => sf[g * max_sfb + sfb] = 0,
                INTENSITY_HCB | INTENSITY_HCB2 => {
                    let d = decode_scalefactor_delta(br)?;
                    g_is += d;
                    sf[g * max_sfb + sfb] = g_is;
                }
                NOISE_HCB => {
                    if !noise_seed_set {
                        // First PNS band — read 9-bit dpcm_noise_nrg seed.
                        let raw = br.read_u32(9)? as i32;
                        g_noise += raw - 256;
                        noise_seed_set = true;
                    } else {
                        let d = decode_scalefactor_delta(br)?;
                        g_noise += d;
                    }
                    sf[g * max_sfb + sfb] = g_noise;
                }
                _ => {
                    let d = decode_scalefactor_delta(br)?;
                    g_gain += d;
                    sf[g * max_sfb + sfb] = g_gain;
                }
            }
        }
    }
    Ok(sf)
}

/// Group/window helpers.
pub fn group_starts(info: &IcsInfo) -> Vec<usize> {
    // Cumulative window starts for each group. Long block: [0, 1].
    let mut starts = Vec::with_capacity(info.num_window_groups as usize + 1);
    starts.push(0);
    let mut cum = 0usize;
    for g in 0..info.num_window_groups as usize {
        cum += info.window_group_length[g] as usize;
        starts.push(cum);
    }
    starts
}

/// Decode spectral data into `coef[1024]`. Implements §4.6.4 +§4.6.6 + §4.6.7
/// (Huffman → quantised spectrum → inverse-quant cubic root → scalefactor).
pub fn decode_spectrum_long(
    br: &mut BitReader<'_>,
    info: &IcsInfo,
    sec: &SectionData,
    sf: &[i32],
    coef: &mut [f32; SPEC_LEN],
) -> Result<()> {
    let swb_offsets = SWB_LONG[info.sf_index as usize];
    let max_sfb = info.max_sfb as usize;
    for sfb in 0..max_sfb {
        let cb = sec.sfb_cb[sfb];
        let start = swb_offsets[sfb] as usize;
        let end = swb_offsets[sfb + 1] as usize;
        if cb == ZERO_HCB {
            for s in start..end {
                coef[s] = 0.0;
            }
            continue;
        }
        if cb == NOISE_HCB || cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
            // PNS / IS — handled separately. Leave zeros in spectrum for now.
            for s in start..end {
                coef[s] = 0.0;
            }
            continue;
        }
        let book = spectral_book(cb)?;
        let dim = book.dim as usize;
        let scale = sf_to_gain(sf[sfb]);
        let mut s = start;
        while s < end {
            let vals = decode_spectral(br, book)?;
            for k in 0..dim {
                if s + k >= end {
                    break;
                }
                let q = vals[k] as f32;
                coef[s + k] = inv_quant(q) * scale;
            }
            s += dim;
        }
    }
    // Fill any unused trailing bands with zero.
    let last = swb_offsets[max_sfb] as usize;
    for s in last..SPEC_LEN {
        coef[s] = 0.0;
    }
    Ok(())
}

/// Decode short-window grouped spectral data into `coef[1024]`. The grouped
/// layout is: for each group (g) of length L_g, the L_g sub-windows of 128
/// coefs are interleaved per-sfb.
pub fn decode_spectrum_short(
    br: &mut BitReader<'_>,
    info: &IcsInfo,
    sec: &SectionData,
    sf: &[i32],
    coef: &mut [f32; SPEC_LEN],
) -> Result<()> {
    let swb_offsets = SWB_SHORT[info.sf_index as usize];
    let max_sfb = info.max_sfb as usize;
    let groups = info.num_window_groups as usize;
    let starts = group_starts(info);

    // Zero the entire spectrum first; bands not coded stay zero.
    for c in coef.iter_mut() {
        *c = 0.0;
    }

    for g in 0..groups {
        let group_len = info.window_group_length[g] as usize;
        let win_start_offset = starts[g] * 128;
        for sfb in 0..max_sfb {
            let cb = sec.sfb_cb[g * max_sfb + sfb];
            let band_start = swb_offsets[sfb] as usize;
            let band_end = swb_offsets[sfb + 1] as usize;
            let band_len = band_end - band_start;
            // grouped layout: L_g sub-windows interleaved by sfb.
            // Order: w=0, sfb's range; w=1, sfb's range; ...
            // Memory: write into coef positions `(win_start_offset + w*128 + band_start) ..`
            if cb == ZERO_HCB || cb == NOISE_HCB || cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
                // All zero (we'll fill IS/PNS later).
                continue;
            }
            let book = spectral_book(cb)?;
            let dim = book.dim as usize;
            let scale = sf_to_gain(sf[g * max_sfb + sfb]);
            // Decode group_len * band_len coefficients sequentially.
            let mut local = vec![0f32; group_len * band_len];
            let mut s = 0;
            while s < group_len * band_len {
                let vals = decode_spectral(br, book)?;
                for k in 0..dim {
                    if s + k < local.len() {
                        local[s + k] = inv_quant(vals[k] as f32) * scale;
                    }
                }
                s += dim;
            }
            // De-interleave into coef.
            for w in 0..group_len {
                for j in 0..band_len {
                    let dst = win_start_offset + w * 128 + band_start + j;
                    let src = w * band_len + j;
                    coef[dst] = local[src];
                }
            }
        }
    }
    Ok(())
}

/// Inverse quantisation: x = sign(q) * |q|^(4/3).
#[inline]
pub fn inv_quant(q: f32) -> f32 {
    if q == 0.0 {
        0.0
    } else {
        let a = q.abs();
        let p = a.powf(4.0 / 3.0);
        if q < 0.0 {
            -p
        } else {
            p
        }
    }
}

/// Convert a scalefactor (-100..150 typical) to a linear gain.
/// gain = 2^((sf - 100)/4) per ISO §4.6.2.3.
#[inline]
pub fn sf_to_gain(sf: i32) -> f32 {
    let exp = (sf - 100) as f32 / 4.0;
    2.0f32.powf(exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_quant_basic() {
        assert_eq!(inv_quant(0.0), 0.0);
        let v = inv_quant(8.0);
        // 8^(4/3) = 16
        assert!((v - 16.0).abs() < 1e-3);
        let v = inv_quant(-1.0);
        assert!((v + 1.0).abs() < 1e-6);
    }

    #[test]
    fn sf_to_gain_unity() {
        // sf=100 -> gain=1.
        assert!((sf_to_gain(100) - 1.0).abs() < 1e-6);
        // sf=104 -> gain=2.
        assert!((sf_to_gain(104) - 2.0).abs() < 1e-3);
    }
}
