//! CELT bit allocator (RFC 6716 §4.3.3, libopus `rate.c`).
//!
//! Computes per-band PVQ pulse counts (`pulses`), fine-energy bit counts
//! (`fine_quant`), and the `fine_priority` ordering used by the §4.3.2.3
//! finalise pass — given a base "quality" lookup table, dynalloc boosts,
//! intensity / dual-stereo flags, and the running range-coder budget.

use crate::range_decoder::RangeDecoder;
use crate::tables::{
    ALLOC_STEPS, BAND_ALLOCATION, CACHE_BITS50, CACHE_INDEX50, EBAND_5MS, FINE_OFFSET,
    LOG2_FRAC_TABLE, LOGN400, MAX_FINE_BITS, NB_EBANDS,
};

const BITRES: i32 = 3;
const LOG_MAX_PSEUDO: i32 = 6;
const NB_ALLOC_VECTORS: usize = 11;

/// `bits2pulses` (libopus rate.h) — binary-search the PVQ cache to find the
/// largest pseudo-pulse-count `q` whose bit cost is ≤ `bits`.
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let lm = (lm + 1) as usize;
    let cache_off = CACHE_INDEX50[lm * NB_EBANDS + band] as usize;
    let cache = &CACHE_BITS50[cache_off..];
    let mut lo: i32 = 0;
    let mut hi: i32 = cache[0] as i32;
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_diff = bits
        - (if lo == 0 {
            -1
        } else {
            cache[lo as usize] as i32
        });
    let hi_diff = cache[hi as usize] as i32 - bits;
    if lo_diff <= hi_diff {
        lo
    } else {
        hi
    }
}

/// `pulses2bits` (libopus rate.h) — bit cost of `pulses` (a pseudo q value).
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    let lm = (lm + 1) as usize;
    let cache_off = CACHE_INDEX50[lm * NB_EBANDS + band] as usize;
    let cache = &CACHE_BITS50[cache_off..];
    if pulses == 0 {
        0
    } else {
        cache[pulses as usize] as i32 + 1
    }
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    balance: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeDecoder<'_>,
) -> usize {
    let alloc_floor = c << BITRES;
    let stereo = if c > 1 { 1 } else { 0 };
    let log_m = lm << BITRES;
    let mut lo: i32 = 0;
    let mut hi: i32 = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum: i32 = 0;
        let mut done = false;
        let mut j = end;
        while j > start {
            j -= 1;
            let tmp = bits1[j] + (mid * bits2[j] >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum: i32 = 0;
    let mut done = false;
    let mut j = end;
    while j > start {
        j -= 1;
        let mut tmp = bits1[j] + (lo * bits2[j] >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working backwards.
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let left = total - psum;
        let band_span = EBAND_5MS[coded_bands] as i32 - EBAND_5MS[start] as i32;
        let percoeff = if band_span > 0 { left / band_span } else { 0 };
        let left_after = left - band_span * percoeff;
        let band_w = EBAND_5MS[j] as i32 - EBAND_5MS[start] as i32;
        let rem = (left_after - band_w).max(0);
        let band_width = (EBAND_5MS[coded_bands] - EBAND_5MS[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if rc.decode_bit_logp(1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // Reclaim bits originally allocated to this band.
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    debug_assert!(coded_bands > start);

    if intensity_rsv > 0 {
        *intensity = start as i32 + rc.decode_uint((coded_bands + 1 - start) as u32) as i32;
    } else {
        *intensity = 0;
    }
    if *intensity <= start as i32 {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        *dual_stereo = if rc.decode_bit_logp(1) { 1 } else { 0 };
    } else {
        *dual_stereo = 0;
    }

    // Allocate the remaining bits.
    let mut left = total - psum;
    let band_span = EBAND_5MS[coded_bands] as i32 - EBAND_5MS[start] as i32;
    let percoeff = if band_span > 0 { left / band_span } else { 0 };
    left -= band_span * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
    }
    for j in start..coded_bands {
        let tmp = left.min((EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32);
        bits[j] += tmp;
        left -= tmp;
    }

    let mut bal: i32 = 0;
    let mut j = start;
    while j < coded_bands {
        let n0 = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let n = n0 << lm;
        let bit = bits[j] + bal;
        let mut excess: i32;
        if n > 1 {
            excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;
            let den = c * n
                + if c == 2 && n > 2 && *dual_stereo == 0 && (j as i32) < *intensity {
                    1
                } else {
                    0
                };
            let nclogn = den * (LOGN400[j] as i32 + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < den * 2 << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < den * 3 << BITRES {
                offset += nclogn >> 3;
            }
            let mut e = (bits[j] + offset + (den << (BITRES - 1))).max(0);
            e = (e / den) >> BITRES;
            if c * e > (bits[j] >> BITRES) {
                e = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = e.min(MAX_FINE_BITS);
            fine_priority[j] = if ebits[j] * (den << BITRES) >= bits[j] + offset {
                1
            } else {
                0
            };
            bits[j] -= c * ebits[j] << BITRES;
        } else {
            excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine_priority[j] = if extra_bits >= excess - bal { 1 } else { 0 };
            excess -= extra_bits;
        }
        bal = excess;
        j += 1;
    }
    *balance = bal;

    // Skipped bands use all their bits for fine energy.
    while j < end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = if ebits[j] < 1 { 1 } else { 0 };
        j += 1;
    }

    coded_bands
}

/// `clt_compute_allocation` (libopus rate.c).
///
/// Returns `coded_bands`. Fills `pulses`, `ebits`, `fine_priority` for
/// `[start, end)` and updates `*balance`, `*intensity`, `*dual_stereo`.
#[allow(clippy::too_many_arguments)]
pub fn clt_compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeDecoder<'_>,
) -> usize {
    total = total.max(0);
    let mut skip_start = start;
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    let mut intensity_rsv: i32 = 0;
    let mut dual_stereo_rsv: i32 = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut bits1 = vec![0i32; NB_EBANDS];
    let mut bits2 = vec![0i32; NB_EBANDS];
    let mut thresh = vec![0i32; NB_EBANDS];
    let mut trim_offset = vec![0i32; NB_EBANDS];

    for j in start..end {
        let bw = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        thresh[j] = (c << BITRES).max((3 * bw << lm << BITRES) >> 4);
        trim_offset[j] =
            c * bw * (alloc_trim - 5 - lm) * (end as i32 - j as i32 - 1) * (1 << (lm + BITRES))
                >> 6;
        if (bw << lm) == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }
    let mut lo: i32 = 1;
    let mut hi: i32 = NB_ALLOC_VECTORS as i32 - 1;
    loop {
        let mut done = false;
        let mut psum: i32 = 0;
        let mid = (lo + hi) >> 1;
        let mut j = end;
        while j > start {
            j -= 1;
            let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
            let mut bitsj =
                c * n * (BAND_ALLOCATION[(mid as usize) * NB_EBANDS + j] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;
    for j in start..end {
        let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let mut bits1j = c * n * (BAND_ALLOCATION[(lo as usize) * NB_EBANDS + j] as i32) << lm >> 2;
        let mut bits2j = if hi as usize >= NB_ALLOC_VECTORS {
            cap[j]
        } else {
            c * n * (BAND_ALLOCATION[(hi as usize) * NB_EBANDS + j] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }
    interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        rc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits2pulses_returns_q0_for_low_budget() {
        // band 8 LM=3 (5ms band index 8) — cache start at index (3+1)*21+8=92.
        let q = bits2pulses(8, 3, 0);
        assert_eq!(q, 0);
    }
}
