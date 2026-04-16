//! MPEG-1 Layer III requantisation, reorder, antialias.
//!
//! Spec: ISO/IEC 11172-3 §2.4.3.4.
//!
//! The requantisation formula for a long block is:
//!
//!   xr[i] = sign(is[i]) * |is[i]|^(4/3)
//!         * 2^( (global_gain - 210) / 4.0 )
//!         * 2^( -scalefac_l[sfb] * (1 + scalefac_scale) * shift )
//!
//! where `shift` is 0.5 normally or `preflag`-adjusted. The short-block
//! form adds a subblock_gain term per-window.
//!
//! After requantisation short-block coefficients are reordered from
//! subband/scalefactor-band layout into window layout.
//!
//! Antialias: 8-tap butterfly across subband boundaries, applied only to
//! long blocks or the long portion of mixed blocks.

use crate::scalefactor::ScaleFactors;
use crate::sfband::{sfband_long, sfband_short};
use crate::sideinfo::GranuleChannel;

/// Preflag pretab - additional scalefactor bias applied when preflag=1.
/// ISO Table 3-B.6.
const PRETAB: [u8; 22] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0,
];

/// Per-sample requantisation. `is[i]` are the 576 integer coefficients
/// decoded from Huffman / count1 regions. `xr[i]` are the dequantised
/// float coefficients ready for stereo processing, antialias, IMDCT.
pub fn requantize_granule(
    is_: &[i32; 576],
    xr: &mut [f32; 576],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate: u32,
) {
    let global_gain = gc.global_gain as i32;
    let base_scale = f32_pow2((global_gain - 210) as f32 / 4.0);
    let scale_shift = if gc.scalefac_scale { 1.0 } else { 0.5 };

    // Branch on block type for sfb layout.
    if gc.window_switching_flag && gc.block_type == 2 {
        // Short or mixed block.
        let long_sfb_count = if gc.mixed_block_flag { 8 } else { 0 };
        let long_bounds = sfband_long(sample_rate);

        // Long portion (first `long_sfb_count` sfbs) — identical to long
        // handling.
        let long_end = long_bounds[long_sfb_count] as usize;
        for sfb in 0..long_sfb_count {
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            let pre = if gc.preflag { PRETAB[sfb] as i32 } else { 0 };
            let sf_exp = -scale_shift * (sf.l[sfb] as f32 + pre as f32);
            let s = base_scale * f32_pow2(sf_exp);
            for i in lo..hi {
                xr[i] = requant_sample(is_[i], s);
            }
        }

        // Short portion. Per ISO 11172-3 §2.4.3.4, short-block gain is:
        //   xr = sign(is) * |is|^(4/3)
        //        * 2^(0.25 * (global_gain - 210 - 8 * subblock_gain[w]))
        //        * 2^(-(1+scalefac_scale)/2 * scalefac_s[sfb][w])
        // The subblock_gain term applies inside the global-gain exponent
        // (factor 0.25 * 8 = 2.0 per unit), NOT also outside it.
        let short_bounds = sfband_short(sample_rate);
        let start_sfb = if gc.mixed_block_flag { 3 } else { 0 };
        let mut pos = long_end;
        for sfb in start_sfb..13 {
            let sfb_width = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
            for win in 0..3 {
                let sbgain = gc.subblock_gain[win] as i32;
                let sf_exp = -scale_shift * sf.s[sfb][win] as f32 - 2.0 * sbgain as f32;
                let s = base_scale * f32_pow2(sf_exp);
                for _i in 0..sfb_width {
                    if pos >= 576 {
                        return;
                    }
                    xr[pos] = requant_sample(is_[pos], s);
                    pos += 1;
                }
            }
        }
    } else {
        // Pure long block.
        let bounds = sfband_long(sample_rate);
        for sfb in 0..21 {
            let lo = bounds[sfb] as usize;
            let hi = bounds[sfb + 1] as usize;
            let pre = if gc.preflag { PRETAB[sfb] as i32 } else { 0 };
            let sf_exp = -scale_shift * (sf.l[sfb] as f32 + pre as f32);
            let s = base_scale * f32_pow2(sf_exp);
            for i in lo..hi.min(576) {
                xr[i] = requant_sample(is_[i], s);
            }
        }
    }
}

fn requant_sample(v: i32, scale: f32) -> f32 {
    if v == 0 {
        return 0.0;
    }
    let mag = (v.unsigned_abs() as f32).powf(4.0 / 3.0);
    let val = mag * scale;
    if v < 0 {
        -val
    } else {
        val
    }
}

#[inline]
fn f32_pow2(e: f32) -> f32 {
    (e * std::f32::consts::LN_2).exp()
}

// --------------- Reorder for short blocks ---------------

/// Reorder short-block coefficients from window-major (sfb→win→freq, the
/// layout produced by `requantize_granule`) to interleaved-by-window
/// (sfb→freq→win) layout that the IMDCT short path expects. ISO 11172-3
/// §2.4.3.4.10.5.
///
/// For pure short blocks, sfb 0..13 are reordered. For mixed blocks, only
/// sfb 3..13 of the short region are reordered (sfb 0..7 are long and stay
/// in subbands 0..1).
pub fn reorder_short(xr: &mut [f32; 576], gc: &GranuleChannel, sample_rate: u32) {
    if !(gc.window_switching_flag && gc.block_type == 2) {
        return;
    }
    let short_bounds = sfband_short(sample_rate);
    let (start_sfb, region_start) = if gc.mixed_block_flag {
        // Mixed: long part covers sfb 0..7 (long sfbs), so the short
        // region starts at long-sfb-index 8 in the granule layout.
        // For 44.1k that's offset 36.
        (3, sfband_long(sample_rate)[8] as usize)
    } else {
        (0, 0)
    };
    let mut buf = [0.0f32; 576];
    let mut pos = region_start;
    let mut sfb = start_sfb;
    while pos < 576 && sfb < 13 {
        let width = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
        // Source: window-major. xr[pos..pos + 3*width] holds win 0 (width),
        // win 1 (width), win 2 (width).
        // Destination: interleaved. For each freq j in 0..width and window
        // w in 0..3, dest is at offset 3*j + w within this sfb's region.
        for w in 0..3 {
            for j in 0..width {
                let src = pos + w * width + j;
                let dst = pos + 3 * j + w;
                if src < 576 && dst < 576 {
                    buf[dst] = xr[src];
                }
            }
        }
        pos += 3 * width;
        sfb += 1;
    }
    // Copy back the reordered region.
    xr[region_start..576].copy_from_slice(&buf[region_start..576]);
}

// --------------- Antialias ---------------

/// Antialias butterfly coefficients (ISO Table 3-B.9). c_s = cos, c_a = -sin.
#[rustfmt::skip]
const CS: [f32; 8] = [
    0.857_492_92, 0.881_742_0,  0.949_628_64, 0.983_314_6,
    0.995_517_8,  0.999_160_8,  0.999_899_2,  0.999_993_04,
];
#[rustfmt::skip]
const CA: [f32; 8] = [
   -0.514_495_76, -0.471_731_97, -0.313_377_46, -0.181_913_2,
   -0.094_574_19, -0.040_965_58, -0.014_197_132,-0.003_699_975,
];

/// Apply the 8-tap antialias butterfly across the 18-sample boundaries
/// of the first 18 subbands (except when block_type == 2, in which case
/// antialias is applied only to the long subbands of a mixed block).
pub fn antialias(xr: &mut [f32; 576], gc: &GranuleChannel) {
    let max_subband = if gc.window_switching_flag && gc.block_type == 2 {
        if gc.mixed_block_flag {
            2 // only long part of mixed block: subbands 0 and 1
        } else {
            0 // pure short block: no antialias
        }
    } else {
        32
    };
    for sb in 1..max_subband.min(32) {
        let base = 18 * sb;
        for i in 0..8 {
            let up = base - 1 - i;
            let dn = base + i;
            let a = xr[up];
            let b = xr[dn];
            xr[up] = a * CS[i] - b * CA[i];
            xr[dn] = b * CS[i] + a * CA[i];
        }
    }
}

// --------------- MS / Intensity Stereo ---------------

/// Apply MS stereo (rotate by 1/sqrt(2)) to a stereo pair of coefficients.
/// Condition: mode_extension bit 0x2 is set.
pub fn ms_stereo(xr_l: &mut [f32; 576], xr_r: &mut [f32; 576]) {
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    for i in 0..576 {
        let m = xr_l[i];
        let s = xr_r[i];
        xr_l[i] = (m + s) * inv_sqrt2;
        xr_r[i] = (m - s) * inv_sqrt2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requant_zero_stays_zero() {
        let is_ = [0i32; 576];
        let mut xr = [0.0f32; 576];
        let gc = GranuleChannel::default();
        let sf = ScaleFactors::default();
        requantize_granule(&is_, &mut xr, &gc, &sf, 44100);
        assert!(xr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ms_stereo_roundtrip() {
        let mut l = [0.0f32; 576];
        let mut r = [0.0f32; 576];
        l[0] = 0.7;
        r[0] = 0.3;
        // MS from L/R produces (L+R)/√2, (L-R)/√2.
        ms_stereo(&mut l, &mut r);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!((l[0] - (0.7 + 0.3) * inv_sqrt2).abs() < 1e-5);
        assert!((r[0] - (0.7 - 0.3) * inv_sqrt2).abs() < 1e-5);
    }

    #[test]
    fn antialias_identity_on_zero_input() {
        let mut xr = [0.0f32; 576];
        let gc = GranuleChannel::default();
        antialias(&mut xr, &gc);
        assert!(xr.iter().all(|&v| v == 0.0));
    }
}
