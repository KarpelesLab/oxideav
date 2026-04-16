//! CELT band decoding (RFC 6716 §4.3.4 — §4.3.6) — port of libopus `bands.c`.
//!
//! Decodes the per-band normalised PVQ shapes, applies stereo
//! reconstruction (intensity / dual / inverse), denormalises with the
//! decoded band energies, and runs the anti-collapse fix-up.

use crate::cwrs::decode_pulses;
use crate::range_decoder::{RangeDecoder, BITRES};
use crate::tables::{
    bitexact_cos, bitexact_log2tan, get_pulses, CACHE_BITS50, CACHE_INDEX50, EBAND_5MS, E_MEANS,
    LOGN400, NB_EBANDS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, SPREAD_AGGRESSIVE, SPREAD_NONE,
};

const NORM_SCALING: f32 = 1.0;
const Q15_ONE: f32 = 1.0;
const EPSILON: f32 = 1e-15;

/// Linear-congruential generator from libopus `celt_lcg_rand`.
#[inline]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn frac_mul16(a: i32, b: i32) -> i32 {
    (16_384 + (a * b)) >> 15
}

#[inline]
fn safe_shl(x: u32, s: u32) -> u32 {
    if s >= 32 {
        0
    } else {
        x << s
    }
}

#[inline]
fn safe_shl_i(x: i32, s: u32) -> i32 {
    if s >= 32 {
        0
    } else {
        x << s
    }
}

#[inline]
fn safe_shr_i(x: i32, s: u32) -> i32 {
    if s >= 32 {
        x >> 31
    } else {
        x >> s
    }
}

#[inline]
fn mask_for(b: i32) -> u32 {
    if b <= 0 {
        0
    } else if b >= 32 {
        u32::MAX
    } else {
        (1u32 << b as u32) - 1
    }
}

fn isqrt32(n: u32) -> u32 {
    (n as f64).sqrt() as u32
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

pub fn haar1(x: &mut [f32], n0: i32, stride: i32) {
    let n0 = n0 / 2;
    let f = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..stride as usize {
        for j in 0..n0 as usize {
            let s = stride as usize;
            let t1 = f * x[s * 2 * j + i];
            let t2 = f * x[s * (2 * j + 1) + i];
            x[s * 2 * j + i] = t1 + t2;
            x[s * (2 * j + 1) + i] = t1 - t2;
        }
    }
}

const ORDERY_TABLE: &[i32] = &[
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn ordery_offset(stride: i32) -> usize {
    (stride - 2) as usize
}

pub fn deinterleave_hadamard(x: &mut [f32], n0: i32, stride: i32, hadamard: bool) {
    let n = (n0 * stride) as usize;
    let mut tmp = vec![0f32; n];
    if hadamard && (stride == 2 || stride == 4 || stride == 8 || stride == 16) {
        let off = ordery_offset(stride);
        for i in 0..stride as usize {
            for j in 0..n0 as usize {
                tmp[ORDERY_TABLE[off + i] as usize * n0 as usize + j] = x[j * stride as usize + i];
            }
        }
    } else {
        for i in 0..stride as usize {
            for j in 0..n0 as usize {
                tmp[i * n0 as usize + j] = x[j * stride as usize + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

pub fn interleave_hadamard(x: &mut [f32], n0: i32, stride: i32, hadamard: bool) {
    let n = (n0 * stride) as usize;
    let mut tmp = vec![0f32; n];
    if hadamard && (stride == 2 || stride == 4 || stride == 8 || stride == 16) {
        let off = ordery_offset(stride);
        for i in 0..stride as usize {
            for j in 0..n0 as usize {
                tmp[j * stride as usize + i] = x[ORDERY_TABLE[off + i] as usize * n0 as usize + j];
            }
        }
    } else {
        for i in 0..stride as usize {
            for j in 0..n0 as usize {
                tmp[j * stride as usize + i] = x[i * n0 as usize + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

fn exp_rotation1(x: &mut [f32], len: i32, stride: i32, c: f32, s: f32) {
    let ms = -s;
    let len = len as usize;
    let stride = stride as usize;
    let mut i = 0;
    while i + stride < len {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
        i += 1;
    }
    if len < 2 * stride + 1 {
        return;
    }
    let mut i: isize = (len - 2 * stride - 1) as isize;
    while i >= 0 {
        let p = i as usize;
        let x1 = x[p];
        let x2 = x[p + stride];
        x[p + stride] = c * x2 + s * x1;
        x[p] = c * x1 + ms * x2;
        i -= 1;
    }
}

pub fn exp_rotation(x: &mut [f32], mut len: i32, dir: i32, stride: i32, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize] as f32;
    let gain = len as f32 / (len as f32 + factor * k as f32);
    let theta = 0.5 * gain * gain;
    let c = (theta * std::f32::consts::PI * 0.5).cos();
    let s = (theta * std::f32::consts::PI * 0.5).sin();
    let mut stride2 = 0i32;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    len /= stride;
    for i in 0..stride as usize {
        let off = i * len as usize;
        let len_a = len;
        let slice = &mut x[off..off + len_a as usize];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(slice, len_a, stride2, s, c);
            }
            exp_rotation1(slice, len_a, 1, c, s);
        } else {
            exp_rotation1(slice, len_a, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(slice, len_a, stride2, s, -c);
            }
        }
    }
}

fn extract_collapse_mask(iy: &[i32], n: i32, b: i32) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = (n / b) as usize;
    let mut mask: u32 = 0;
    for i in 0..b as usize {
        let mut tmp = 0i32;
        for j in 0..n0 {
            tmp |= iy[i * n0 + j];
        }
        if tmp != 0 {
            mask |= 1 << i;
        }
    }
    mask
}

fn normalise_residual(iy: &[i32], x: &mut [f32], n: usize, ryy: f32, gain: f32) {
    let g = gain / ryy.max(EPSILON).sqrt();
    for i in 0..n {
        x[i] = g * iy[i] as f32;
    }
}

fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = EPSILON;
    for &v in x.iter().take(n) {
        e += v * v;
    }
    let g = gain / e.sqrt();
    for v in x.iter_mut().take(n) {
        *v *= g;
    }
}

#[allow(clippy::too_many_arguments)]
fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: u32,
    spread: i32,
    b: i32,
    rc: &mut RangeDecoder<'_>,
    gain: f32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n >= 2);
    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k, rc) as f32;
    normalise_residual(&iy, x, n, ryy, gain);
    exp_rotation(x, n as i32, -1, b, k as i32, spread);
    extract_collapse_mask(&iy, n as i32, b)
}

#[derive(Default, Clone, Copy)]
struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

#[derive(Clone, Copy)]
struct BandCtx {
    spread: i32,
    tf_change: i32,
    remaining_bits: i32,
    seed: u32,
    band_index: usize,
    intensity: i32,
    disable_inv: bool,
    avoid_split_noise: bool,
}

#[allow(clippy::too_many_arguments)]
fn compute_theta(
    ctx: &mut BandCtx,
    rc: &mut RangeDecoder<'_>,
    sctx: &mut SplitCtx,
    n: i32,
    b: &mut i32,
    big_b: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
) {
    let pulse_cap = LOGN400[ctx.band_index] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && (ctx.band_index as i32) >= ctx.intensity {
        qn = 1;
    }
    let tell = rc.tell_frac() as i32;
    let mut itheta;
    let mut inv = false;
    if qn != 1 {
        if stereo && n > 2 {
            let p0 = 3i32;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            let fs = rc.decode(ft as u32) as i32;
            let x = if fs < (x0 + 1) * p0 {
                fs / p0
            } else {
                x0 + 1 + (fs - (x0 + 1) * p0)
            };
            let (fl, fh) = if x <= x0 {
                (p0 * x, p0 * (x + 1))
            } else {
                ((x - 1 - x0) + (x0 + 1) * p0, (x - x0) + (x0 + 1) * p0)
            };
            rc.dec_update(fl as u32, fh as u32, ft as u32);
            itheta = x;
        } else if b0 > 1 || stereo {
            itheta = rc.decode_uint((qn + 1) as u32) as i32;
        } else {
            // Triangular pdf.
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fm = rc.decode(ft as u32) as i32;
            let (fs, fl, theta);
            if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                theta = (isqrt32(8 * fm as u32 + 1) as i32 - 1) >> 1;
                fs = theta + 1;
                fl = theta * (theta + 1) >> 1;
            } else {
                theta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
                fs = qn + 1 - theta;
                fl = ft - ((qn + 1 - theta) * (qn + 2 - theta) >> 1);
            }
            rc.dec_update(fl as u32, (fl + fs) as u32, ft as u32);
            itheta = theta;
        }
        debug_assert!(itheta >= 0);
        itheta = (itheta * 16384) / qn;
    } else if stereo {
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = rc.decode_bit_logp(2);
        }
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    } else {
        itheta = 0;
    }
    let qalloc = rc.tell_frac() as i32 - tell;
    *b -= qalloc;
    let imid;
    let iside;
    let delta;
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= mask_for(big_b) as i32;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= safe_shl(mask_for(big_b), big_b as u32) as i32;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
    }
    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.delta = delta;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
    let _ = ctx.avoid_split_noise; // not used at decode time
}

fn quant_band_n1(
    ctx: &mut BandCtx,
    rc: &mut RangeDecoder<'_>,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut f32>,
) -> u32 {
    if ctx.remaining_bits >= 1 << BITRES {
        let sign = rc.decode_bits(1);
        ctx.remaining_bits -= 1 << BITRES;
        x[0] = if sign != 0 {
            -NORM_SCALING
        } else {
            NORM_SCALING
        };
    } else {
        x[0] = NORM_SCALING;
    }
    if let Some(yref) = y {
        if ctx.remaining_bits >= 1 << BITRES {
            let sign = rc.decode_bits(1);
            ctx.remaining_bits -= 1 << BITRES;
            yref[0] = if sign != 0 {
                -NORM_SCALING
            } else {
                NORM_SCALING
            };
        } else {
            yref[0] = NORM_SCALING;
        }
    }
    if let Some(out) = lowband_out {
        *out = x[0] * 0.0625;
    }
    1
}

const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

#[allow(clippy::too_many_arguments)]
fn quant_partition(
    ctx: &mut BandCtx,
    rc: &mut RangeDecoder<'_>,
    x: &mut [f32],
    n: i32,
    mut b: i32,
    big_b: i32,
    lowband: Option<&[f32]>,
    lm: i32,
    gain: f32,
    mut fill: i32,
) -> u32 {
    let i = ctx.band_index;
    let cache_off = CACHE_INDEX50[((lm + 1) as usize) * NB_EBANDS + i] as usize;
    let cache = &CACHE_BITS50[cache_off..];
    let csize = cache[0] as usize;
    let cmax = cache[csize] as i32;
    let big_b_cur;

    if lm != -1 && b > cmax + 12 && n > 2 {
        let half = (n / 2) as usize;
        if big_b == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        big_b_cur = (big_b + 1) >> 1;
        let mut sctx = SplitCtx::default();
        compute_theta(
            ctx,
            rc,
            &mut sctx,
            half as i32,
            &mut b,
            big_b_cur,
            big_b,
            lm - 1,
            false,
            &mut fill,
        );
        let imid = sctx.imid;
        let iside = sctx.iside;
        let mut delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;
        let mid = imid as f32 / 32768.0;
        let side = iside as f32 / 32768.0;
        if big_b > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = (delta + (n << BITRES >> (5 - lm))).min(0);
            }
        }
        let mbits = ((b - delta) / 2).max(0).min(b);
        let sbits = b - mbits;
        ctx.remaining_bits -= qalloc;
        let rebalance = ctx.remaining_bits;
        let (x_lo, x_hi) = x.split_at_mut(half);
        let lowband_lo = lowband.map(|lb| &lb[..half]);
        let lowband_hi = lowband.map(|lb| &lb[half..2 * half]);
        let cm: u32;
        if mbits >= sbits {
            ctx.band_index = i;
            let cm1 = quant_partition(
                ctx,
                rc,
                x_lo,
                half as i32,
                mbits,
                big_b_cur,
                lowband_lo,
                lm - 1,
                gain * mid,
                fill,
            );
            let mut sb = sbits;
            let reb = mbits - (rebalance - ctx.remaining_bits);
            if reb > 3 << BITRES && itheta != 0 {
                sb += reb - (3 << BITRES);
            }
            ctx.band_index = i;
            let cm2 = quant_partition(
                ctx,
                rc,
                x_hi,
                half as i32,
                sb,
                big_b_cur,
                lowband_hi,
                lm - 1,
                gain * side,
                safe_shr_i(fill, big_b_cur as u32),
            );
            cm = cm1 | safe_shl(cm2, (big_b >> 1) as u32);
        } else {
            ctx.band_index = i;
            let cm2 = quant_partition(
                ctx,
                rc,
                x_hi,
                half as i32,
                sbits,
                big_b_cur,
                lowband_hi,
                lm - 1,
                gain * side,
                safe_shr_i(fill, big_b_cur as u32),
            );
            let mut mb = mbits;
            let reb = sbits - (rebalance - ctx.remaining_bits);
            if reb > 3 << BITRES && itheta != 16384 {
                mb += reb - (3 << BITRES);
            }
            ctx.band_index = i;
            let cm1 = quant_partition(
                ctx,
                rc,
                x_lo,
                half as i32,
                mb,
                big_b_cur,
                lowband_lo,
                lm - 1,
                gain * mid,
                fill,
            );
            cm = cm1 | safe_shl(cm2, (big_b >> 1) as u32);
        }
        cm & mask_for(big_b)
    } else {
        let q = crate::rate::bits2pulses(i, lm, b);
        let mut curr_bits = crate::rate::pulses2bits(i, lm, q);
        ctx.remaining_bits -= curr_bits;
        let mut q_used = q;
        while ctx.remaining_bits < 0 && q_used > 0 {
            ctx.remaining_bits += curr_bits;
            q_used -= 1;
            curr_bits = crate::rate::pulses2bits(i, lm, q_used);
            ctx.remaining_bits -= curr_bits;
        }
        if q_used != 0 {
            let k = get_pulses(q_used) as u32;
            alg_unquant(x, n as usize, k, ctx.spread, big_b, rc, gain)
        } else {
            let cm_mask = mask_for(big_b);
            fill &= cm_mask as i32;
            if fill == 0 {
                for v in x.iter_mut().take(n as usize) {
                    *v = 0.0;
                }
                0
            } else if let Some(lb) = lowband {
                for j in 0..n as usize {
                    ctx.seed = celt_lcg_rand(ctx.seed);
                    let tmp = if ctx.seed & 0x8000 != 0 {
                        1.0 / 256.0
                    } else {
                        -1.0 / 256.0
                    };
                    x[j] = lb[j] + tmp;
                }
                renormalise_vector(x, n as usize, gain);
                fill as u32
            } else {
                for j in 0..n as usize {
                    ctx.seed = celt_lcg_rand(ctx.seed);
                    x[j] = ((ctx.seed as i32) >> 20) as f32;
                }
                renormalise_vector(x, n as usize, gain);
                cm_mask
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut BandCtx,
    rc: &mut RangeDecoder<'_>,
    x: &mut [f32],
    n: i32,
    b: i32,
    big_b: i32,
    lowband: Option<&[f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let b0 = big_b;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    n_b /= big_b;
    if n == 1 {
        let lb_out = lowband_out.map(|s| &mut s[0]);
        return quant_band_n1(ctx, rc, x, None, lb_out);
    }
    let tf_change = ctx.tf_change;
    if tf_change > 0 {
        recombine = tf_change;
    }
    let mut lowband_buf: Option<Vec<f32>> = None;
    let need_copy =
        lowband.is_some() && (recombine > 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1);
    if need_copy {
        let lb = lowband.unwrap();
        lowband_buf = Some(lb[..n as usize].to_vec());
    }
    let mut fill_v = fill;
    let mut big_b_cur = big_b;
    for k in 0..recombine {
        haar1(x, n >> k, 1 << k); // libopus: encoder also haar's X. Decoder doesn't, but this is the resynth path.
                                  // Actually for decode, only resynth applies haar; but we'll do it after partition.
        let _ = k;
        let lo = (fill_v & 0xF) as usize;
        let hi = ((fill_v >> 4) & 0xF) as usize;
        fill_v = BIT_INTERLEAVE_TABLE[lo] as i32 | ((BIT_INTERLEAVE_TABLE[hi] as i32) << 2);
        if let Some(buf) = lowband_buf.as_mut() {
            haar1(buf, n >> k, 1 << k);
        }
    }
    big_b_cur >>= recombine;
    n_b <<= recombine;

    while (n_b & 1) == 0 && tf_change < 0 {
        if let Some(buf) = lowband_buf.as_mut() {
            haar1(buf, n_b, big_b_cur);
        }
        fill_v |= safe_shl_i(fill_v, big_b_cur as u32);
        big_b_cur <<= 1;
        n_b >>= 1;
        time_divide += 1;
        if time_divide > 4 {
            break;
        }
    }
    let b0_new = big_b_cur;
    let n_b0 = n_b;
    if b0_new > 1 {
        if let Some(buf) = lowband_buf.as_mut() {
            deinterleave_hadamard(buf, n_b >> recombine, b0_new << recombine, long_blocks);
        }
    }
    let lowband_ref = lowband_buf.as_deref().or(lowband);
    let mut cm = quant_partition(ctx, rc, x, n, b, big_b_cur, lowband_ref, lm, gain, fill_v);

    // Resynthesis
    if b0_new > 1 {
        interleave_hadamard(x, n_b >> recombine, b0_new << recombine, long_blocks);
    }
    let mut n_b_local = n_b0;
    let mut big_b_cur2 = b0_new;
    for _ in 0..time_divide {
        big_b_cur2 >>= 1;
        n_b_local <<= 1;
        if (big_b_cur2 as u32) < 32 {
            cm |= cm >> big_b_cur2;
        }
        haar1(x, n_b_local, big_b_cur2);
    }
    for k in 0..recombine {
        let idx = (cm & 0xFF) as usize;
        cm = BIT_DEINTERLEAVE_TABLE[idx] as u32;
        haar1(x, n0 >> k, 1 << k);
    }
    big_b_cur2 <<= recombine;

    if let Some(out) = lowband_out {
        let nf = (n0 as f32 * 4_194_304.0).sqrt();
        for j in 0..n0 as usize {
            out[j] = nf * x[j];
        }
    }
    cm & mask_for(big_b_cur2)
}

#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandCtx,
    rc: &mut RangeDecoder<'_>,
    x: &mut [f32],
    y: &mut [f32],
    n: i32,
    b: i32,
    big_b: i32,
    lowband: Option<&[f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    fill: i32,
) -> u32 {
    if n == 1 {
        let lb_out = lowband_out.map(|s| &mut s[0]);
        return quant_band_n1(ctx, rc, x, Some(y), lb_out);
    }
    let orig_fill = fill;
    let mut fill_v = fill;
    let mut sctx = SplitCtx::default();
    let mut b_var = b;
    compute_theta(
        ctx,
        rc,
        &mut sctx,
        n,
        &mut b_var,
        big_b,
        big_b,
        lm,
        true,
        &mut fill_v,
    );
    let inv = sctx.inv;
    let imid = sctx.imid;
    let iside = sctx.iside;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;
    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;
    let mut cm: u32;
    if n == 2 {
        let mut mbits = b_var;
        let mut sbits = 0;
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;
        let mut sign = 0;
        if sbits != 0 {
            sign = rc.decode_bits(1) as i32;
        }
        let sign_v = 1 - 2 * sign;
        cm = if c {
            quant_band(
                ctx,
                rc,
                y,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q15_ONE,
                orig_fill,
            )
        } else {
            quant_band(
                ctx,
                rc,
                x,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q15_ONE,
                orig_fill,
            )
        };
        if c {
            x[0] = -(sign_v as f32) * y[1];
            x[1] = (sign_v as f32) * y[0];
        } else {
            y[0] = -(sign_v as f32) * x[1];
            y[1] = (sign_v as f32) * x[0];
        }
        let tx0 = x[0] * mid;
        let tx1 = x[1] * mid;
        let ty0 = y[0] * side;
        let ty1 = y[1] * side;
        x[0] = tx0 - ty0;
        y[0] = tx0 + ty0;
        x[1] = tx1 - ty1;
        y[1] = tx1 + ty1;
    } else {
        let mbits = ((b_var - delta) / 2).max(0).min(b_var);
        let sbits = b_var - mbits;
        ctx.remaining_bits -= qalloc;
        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_band(
                ctx,
                rc,
                x,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q15_ONE,
                fill_v,
            );
            let mut sb = sbits;
            let reb = mbits - (rebalance - ctx.remaining_bits);
            if reb > 3 << BITRES && itheta != 0 {
                sb += reb - (3 << BITRES);
            }
            cm |= quant_band(
                ctx,
                rc,
                y,
                n,
                sb,
                big_b,
                None,
                lm,
                None,
                side,
                safe_shr_i(fill_v, big_b as u32),
            );
        } else {
            cm = quant_band(
                ctx,
                rc,
                y,
                n,
                sbits,
                big_b,
                None,
                lm,
                None,
                side,
                safe_shr_i(fill_v, big_b as u32),
            );
            let mut mb = mbits;
            let reb = sbits - (rebalance - ctx.remaining_bits);
            if reb > 3 << BITRES && itheta != 16384 {
                mb += reb - (3 << BITRES);
            }
            cm |= quant_band(
                ctx,
                rc,
                x,
                n,
                mb,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q15_ONE,
                fill_v,
            );
        }
        if n != 2 {
            stereo_merge(x, y, mid, n as usize);
        }
        if inv {
            for v in y.iter_mut().take(n as usize) {
                *v = -*v;
            }
        }
    }
    cm
}

fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0f32;
    let mut side = 0f32;
    for j in 0..n {
        xp += x[j] * y[j];
        side += y[j] * y[j];
    }
    xp *= mid;
    let mid2 = mid * 0.5;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let lgain = 1.0 / el.max(EPSILON).sqrt();
    let rgain = 1.0 / er.max(EPSILON).sqrt();
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

/// Top-level: decode all bands. `x` and optional `y` are length `n_total`
/// per channel (not interleaved-by-channel).
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands(
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    mut dual_stereo: i32,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    rc: &mut RangeDecoder<'_>,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
    disable_inv: bool,
) {
    let m = 1i32 << lm;
    let big_b = if short_blocks { m } else { 1 };
    let nb_ebands = NB_EBANDS;
    let stereo = y.is_some();
    let c_count = if stereo { 2 } else { 1 };
    let norm_offset = (m * EBAND_5MS[start] as i32) as usize;
    let norm_len = (m as usize * EBAND_5MS[nb_ebands - 1] as usize - norm_offset).max(1);
    // Two norm channels: 0=mid (X side), 1=side (Y side)
    let mut norm = vec![0f32; 2 * norm_len];
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut y_local = y;
    let _ = band_e;

    for i in start..end {
        let n = ((EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32) * m;
        let tell = rc.tell_frac() as i32;
        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        let b = if i <= coded_bands - 1 {
            let denom = 3.min(coded_bands as i32 - i as i32).max(1);
            let curr_balance = balance / denom;
            (remaining_bits + 1)
                .min(pulses[i] + curr_balance)
                .clamp(0, 16383)
        } else {
            0
        };
        let tf_change = tf_res[i];
        if (m * EBAND_5MS[i] as i32 - n >= m * EBAND_5MS[start] as i32 || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        let effective_lowband = if lowband_offset != 0
            && (spread != SPREAD_AGGRESSIVE || big_b > 1 || tf_change < 0)
        {
            Some(((m * EBAND_5MS[lowband_offset] as i32 - norm_offset as i32 - n).max(0)) as usize)
        } else {
            None
        };
        let band_off = (m * EBAND_5MS[i] as i32) as usize;
        let band_len = n as usize;
        let lowband_x: Option<Vec<f32>> = effective_lowband.map(|lb_start| {
            let mut v = vec![0f32; band_len];
            let avail = norm_len.saturating_sub(lb_start);
            let take = band_len.min(avail);
            v[..take].copy_from_slice(&norm[lb_start..lb_start + take]);
            v
        });
        let lowband_y: Option<Vec<f32>> = if stereo {
            effective_lowband.map(|lb_start| {
                let mut v = vec![0f32; band_len];
                let avail = norm_len.saturating_sub(lb_start);
                let take = band_len.min(avail);
                v[..take].copy_from_slice(&norm[norm_len + lb_start..norm_len + lb_start + take]);
                v
            })
        } else {
            None
        };

        let mut ctx = BandCtx {
            spread,
            tf_change,
            remaining_bits,
            seed: *seed,
            band_index: i,
            intensity,
            disable_inv,
            avoid_split_noise: big_b > 1,
        };
        let mut x_buf = x[band_off..band_off + band_len].to_vec();
        let cm: u32;
        if let Some(yref) = y_local.as_mut() {
            let mut y_buf = yref[band_off..band_off + band_len].to_vec();
            if dual_stereo != 0 && i as i32 == intensity {
                dual_stereo = 0;
                for j in 0..norm_len {
                    norm[j] = 0.5 * (norm[j] + norm[norm_len + j]);
                }
            }
            cm = if dual_stereo != 0 {
                let cm_x = quant_band(
                    &mut ctx,
                    rc,
                    &mut x_buf,
                    n,
                    b / 2,
                    big_b,
                    lowband_x.as_deref(),
                    lm,
                    None,
                    Q15_ONE,
                    mask_for(big_b) as i32,
                );
                let cm_y = quant_band(
                    &mut ctx,
                    rc,
                    &mut y_buf,
                    n,
                    b / 2,
                    big_b,
                    lowband_y.as_deref(),
                    lm,
                    None,
                    Q15_ONE,
                    mask_for(big_b) as i32,
                );
                yref[band_off..band_off + band_len].copy_from_slice(&y_buf);
                cm_x | cm_y
            } else {
                let cm_v = quant_band_stereo(
                    &mut ctx,
                    rc,
                    &mut x_buf,
                    &mut y_buf,
                    n,
                    b,
                    big_b,
                    lowband_x.as_deref(),
                    lm,
                    None,
                    mask_for(big_b) as i32,
                );
                yref[band_off..band_off + band_len].copy_from_slice(&y_buf);
                cm_v
            };
        } else {
            cm = quant_band(
                &mut ctx,
                rc,
                &mut x_buf,
                n,
                b,
                big_b,
                lowband_x.as_deref(),
                lm,
                None,
                Q15_ONE,
                mask_for(big_b) as i32,
            );
        }
        x[band_off..band_off + band_len].copy_from_slice(&x_buf);
        // Update norm slot for folding (use current band's shape).
        let nstart = band_off - norm_offset;
        if nstart + band_len <= norm_len {
            norm[nstart..nstart + band_len].copy_from_slice(&x_buf);
            if stereo {
                if let Some(yref) = y_local.as_ref() {
                    norm[norm_len + nstart..norm_len + nstart + band_len]
                        .copy_from_slice(&yref[band_off..band_off + band_len]);
                }
            }
        }
        *seed = ctx.seed;
        collapse_masks[i * c_count] = cm as u8;
        if c_count > 1 {
            collapse_masks[i * c_count + 1] = cm as u8;
        }
        balance += pulses[i] + tell;
        update_lowband = b > (n << BITRES);
    }
}

/// Anti-collapse processing (RFC 6716 §4.3.5). Replaces zero-energy MDCT
/// short blocks with shaped noise.
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: i32,
    c: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1log_e: &[f32],
    prev2log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32;
        let depth = ((1 + pulses[i]) / n0) >> lm;
        let thresh = 0.5 * (-0.125 * depth as f32).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();
        for chan in 0..c {
            let mut prev1 = prev1log_e[chan * NB_EBANDS + i];
            let mut prev2 = prev2log_e[chan * NB_EBANDS + i];
            if c == 1 {
                prev1 = prev1.max(prev1log_e[NB_EBANDS + i]);
                prev2 = prev2.max(prev2log_e[NB_EBANDS + i]);
            }
            let ediff = (log_e[chan * NB_EBANDS + i] - prev1.min(prev2)).max(0.0);
            let mut r = 2.0 * (-ediff).exp2();
            if lm == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            r = r.min(thresh);
            r *= sqrt_1;
            let off = chan * size + (EBAND_5MS[i] as usize) * (1 << lm);
            let band_len = (n0 << lm) as usize;
            let mut renormalize = false;
            for k in 0..(1u32 << lm as u32) {
                if collapse_masks[i * c + chan] as u32 & (1 << k) == 0 {
                    for j in 0..n0 as usize {
                        seed = celt_lcg_rand(seed);
                        let val = if seed & 0x8000 != 0 { r } else { -r };
                        x[off + (j << lm) + k as usize] = val;
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x[off..off + band_len], band_len, Q15_ONE);
            }
        }
    }
}

/// Denormalise band shapes: multiply each band by `2^(bandLogE + eMeans)`.
pub fn denormalise_bands(
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: usize,
    end: usize,
    m: usize,
    silence: bool,
) {
    let n = m * EBAND_5MS[NB_EBANDS] as usize;
    let bound = if silence {
        0
    } else {
        m * EBAND_5MS[end] as usize
    };
    let (start, end) = if silence { (0, 0) } else { (start, end) };
    let zero_until = m * EBAND_5MS[start] as usize;
    for v in freq.iter_mut().take(zero_until) {
        *v = 0.0;
    }
    for i in start..end {
        let lg = band_log_e[i] + E_MEANS[i];
        let g = lg.min(32.0).exp2();
        let band_start = m * EBAND_5MS[i] as usize;
        let band_end = m * EBAND_5MS[i + 1] as usize;
        for j in band_start..band_end {
            freq[j] = g * x[j];
        }
    }
    if bound < n {
        for v in freq.iter_mut().take(n).skip(bound) {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn celt_lcg_rand_matches_libopus() {
        let r = celt_lcg_rand(0);
        assert_eq!(r, 1_013_904_223);
    }

    #[test]
    fn denormalise_silence_zeros_output() {
        let x = vec![1.0; 400];
        let mut freq = vec![1.0; 400];
        let bandlog = vec![0.0; NB_EBANDS];
        denormalise_bands(&x, &mut freq, &bandlog, 0, 21, 4, true);
        assert!(freq.iter().all(|&v| v == 0.0));
    }
}
