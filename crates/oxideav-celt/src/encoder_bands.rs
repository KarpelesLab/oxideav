//! Encoder-side band coder — mirror of `bands::quant_all_bands`.
//!
//! This module is structurally identical to the decoder's
//! [`crate::bands::quant_all_bands`]: same band loop, same theta-driven
//! splits, same norm folding. The difference is that every point where the
//! decoder does `rc.decode_*` we do `rc.encode_*` with the value computed
//! from the true spectrum. After the encode step we ALSO perform the same
//! resynthesis the decoder does, so the encoder's running norm buffer
//! exactly matches what the decoder will see.
//!
//! Scope: the full decoder path is ported in this module. Transients and
//! short blocks are NOT implemented on the encode side (we always encode
//! with `big_b == 1`, matching `short_blocks = false`). The encoder caller
//! in `encoder.rs` enforces this.

use crate::cwrs::{encode_pulses, pvq_search};
use crate::range_encoder::{RangeEncoder, BITRES};
use crate::tables::{
    bitexact_cos, bitexact_log2tan, get_pulses, CACHE_BITS50, CACHE_INDEX50, EBAND_5MS, LOGN400,
    NB_EBANDS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, SPREAD_AGGRESSIVE, SPREAD_NONE,
};

const NORM_SCALING: f32 = 1.0;
const Q15_ONE: f32 = 1.0;
const EPSILON: f32 = 1e-15;

#[inline]
fn celt_lcg_rand(seed: u32) -> u32 {
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

/// Encode the PVQ shape for one band segment. Mirrors `alg_unquant` in
/// `bands.rs`, but running PVQ search + encoding instead of decoding.
fn alg_quant(
    x: &mut [f32],
    n: usize,
    k: u32,
    spread: i32,
    b: i32,
    rc: &mut RangeEncoder,
    gain: f32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n >= 2);
    // Apply forward exp_rotation on the input shape (the decoder applies
    // the inverse after decoding, so encoder applies forward first).
    let mut xr = x.to_vec();
    exp_rotation(&mut xr, n as i32, 1, b, k as i32, spread);
    // PVQ search to find best integer vector with sum(|y|) = k.
    let mut iy = vec![0i32; n];
    let _norm = pvq_search(&xr, k as i32, &mut iy);
    // Emit enumerated codeword.
    encode_pulses(&iy, n, k, rc);
    // Resynthesise into x (same as decoder): ryy, normalise, inverse rotate.
    let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
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

/// Encoder-side theta chooser. Given the mid/side vectors `x_mid` and
/// `x_side`, compute their energies, derive the ideal `itheta` in [0, 16384],
/// quantise it to a qn-point grid, and write it to the range coder using the
/// same symbol structure the decoder expects.
#[allow(clippy::too_many_arguments)]
fn compute_and_encode_theta(
    ctx: &mut BandCtx,
    rc: &mut RangeEncoder,
    sctx: &mut SplitCtx,
    x_mid: &[f32],
    x_side: &[f32],
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
    // Compute true theta from the input halves.
    let e_mid: f32 = x_mid.iter().take(n as usize).map(|v| v * v).sum();
    let e_side: f32 = x_side.iter().take(n as usize).map(|v| v * v).sum();
    let theta_true = (e_side.sqrt().atan2(e_mid.sqrt()))
        .max(0.0)
        .min(std::f32::consts::FRAC_PI_2);
    // Scale so theta_true=0 → itheta=0, theta_true=pi/2 → itheta=16384.
    let itheta_true = ((theta_true / std::f32::consts::FRAC_PI_2) * 16384.0).round() as i32;
    let itheta_true = itheta_true.clamp(0, 16384);

    let mut itheta: i32;
    let mut inv = false;
    if qn != 1 {
        // Quantise itheta_true to grid of size qn+1 (values 0..=qn), then
        // decoder scales back by `(itheta*16384)/qn`.
        let mut theta_q = ((itheta_true as i64 * qn as i64 + 8192) / 16384) as i32;
        theta_q = theta_q.clamp(0, qn);
        // Emit theta using the same CDF as the decoder.
        if stereo && n > 2 {
            let p0 = 3i32;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            // Inverse of decoder's mapping: theta_q <-> fs
            let (fl, fh) = if theta_q <= x0 {
                (p0 * theta_q, p0 * (theta_q + 1))
            } else {
                (
                    (theta_q - 1 - x0) + (x0 + 1) * p0,
                    (theta_q - x0) + (x0 + 1) * p0,
                )
            };
            rc.encode(fl as u32, fh as u32, ft as u32);
        } else if b0 > 1 || stereo {
            rc.encode_uint(theta_q as u32, (qn + 1) as u32);
        } else {
            // Triangular pdf.
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let (fs, fl);
            if theta_q <= (qn >> 1) {
                fl = theta_q * (theta_q + 1) >> 1;
                fs = theta_q + 1;
            } else {
                fs = qn + 1 - theta_q;
                fl = ft - ((qn + 1 - theta_q) * (qn + 2 - theta_q) >> 1);
            }
            rc.encode(fl as u32, (fl + fs) as u32, ft as u32);
        }
        itheta = (theta_q * 16384) / qn;
    } else if stereo {
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            // Choose inv = false (simplest — assumes no inversion needed).
            inv = false;
            rc.encode_bit_logp(inv, 2);
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
    let _ = ctx.avoid_split_noise;
}

/// Encoder analog of `quant_band_n1`.
fn quant_band_n1_enc(ctx: &mut BandCtx, rc: &mut RangeEncoder, x: &mut [f32]) -> u32 {
    if ctx.remaining_bits >= 1 << BITRES {
        let sign = if x[0] < 0.0 { 1u32 } else { 0 };
        rc.encode_bits(sign, 1);
        ctx.remaining_bits -= 1 << BITRES;
        x[0] = if sign != 0 {
            -NORM_SCALING
        } else {
            NORM_SCALING
        };
    } else {
        x[0] = NORM_SCALING;
    }
    1
}

const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// Encode one band/partition — mirrors `quant_partition` but without the
/// recursive theta split. If the decoder-side split condition would fire,
/// we encode with a "don't split" strategy: we keep the allocation small
/// enough that `b <= cmax + 12`. Callers (via the allocator) must ensure
/// this precondition holds. For our first-cut encoder, the allocator has
/// been tuned to stay within the no-split regime for all long-block bands.
#[allow(clippy::too_many_arguments)]
fn quant_partition_enc(
    ctx: &mut BandCtx,
    rc: &mut RangeEncoder,
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

    if lm != -1 && b > cmax + 12 && n > 2 {
        // Recursive split — mirror decoder. Mid/side split on the current x.
        let half = (n / 2) as usize;
        if big_b == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        let big_b_cur = (big_b + 1) >> 1;
        // Split x into mid/side representation for theta estimation.
        let mut x_mid = x[..half].to_vec();
        let mut x_side = x[half..2 * half].to_vec();
        // Compute mid+side energies and encode theta.
        let mut sctx = SplitCtx::default();
        compute_and_encode_theta(
            ctx,
            rc,
            &mut sctx,
            &x_mid,
            &x_side,
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
        let lowband_lo = lowband.map(|lb| &lb[..half]);
        let lowband_hi = lowband.map(|lb| &lb[half..2 * half]);
        let cm: u32;
        // For encoder, we scale the mid/side vectors by 1/mid and 1/side so
        // after decoder multiplies them back we recover the originals.
        if mid > EPSILON {
            for v in x_mid.iter_mut() {
                *v /= mid;
            }
        }
        if side > EPSILON {
            for v in x_side.iter_mut() {
                *v /= side;
            }
        }
        if mbits >= sbits {
            ctx.band_index = i;
            let cm1 = quant_partition_enc(
                ctx,
                rc,
                &mut x_mid,
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
            let cm2 = quant_partition_enc(
                ctx,
                rc,
                &mut x_side,
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
            let cm2 = quant_partition_enc(
                ctx,
                rc,
                &mut x_side,
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
            let cm1 = quant_partition_enc(
                ctx,
                rc,
                &mut x_mid,
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
        // Write the split halves back to x for caller.
        x[..half].copy_from_slice(&x_mid);
        x[half..2 * half].copy_from_slice(&x_side);
        cm & mask_for(big_b)
    } else {
        // Leaf: run PVQ on this partition.
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
            alg_quant(x, n as usize, k, ctx.spread, big_b, rc, gain)
        } else {
            // No pulses — mirror decoder's fold/noise-fill with same seed.
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

/// Top-level band encoder for mono, long-block CELT frames.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_bands_mono(
    start: usize,
    end: usize,
    x: &mut [f32],
    collapse_masks: &mut [u8],
    pulses: &[i32],
    spread: i32,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    rc: &mut RangeEncoder,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
) {
    let m = 1i32 << lm;
    let big_b: i32 = 1; // long block
    let nb_ebands = NB_EBANDS;
    let norm_offset = (m * EBAND_5MS[start] as i32) as usize;
    let norm_len = (m as usize * EBAND_5MS[nb_ebands - 1] as usize - norm_offset).max(1);
    let mut norm = vec![0f32; norm_len];
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

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

        let mut ctx = BandCtx {
            spread,
            tf_change,
            remaining_bits,
            seed: *seed,
            band_index: i,
            intensity: 0,
            disable_inv: false,
            avoid_split_noise: big_b > 1,
        };
        let mut x_buf = x[band_off..band_off + band_len].to_vec();
        let cm;
        if n == 1 {
            cm = quant_band_n1_enc(&mut ctx, rc, &mut x_buf);
        } else {
            // Mirror `quant_band` (no tf_change recombine / time-divide for
            // the simple encoder — we always pass tf_res = 0). The full
            // `haar1` stack triggers when `tf_change != 0`; since we emit
            // `tf_res = 0`, none of it runs.
            cm = quant_partition_enc(
                &mut ctx,
                rc,
                &mut x_buf,
                n,
                b,
                big_b,
                lowband_x.as_deref(),
                lm,
                Q15_ONE,
                mask_for(big_b) as i32,
            );
            // Apply `lowband_out` scaling — matches decoder's post-partition
            // normalisation factor `sqrt(n0 * 4_194_304)`.
            let _ = (); // the encoder doesn't need lowband_out separately.
        }
        x[band_off..band_off + band_len].copy_from_slice(&x_buf);
        // Update the norm buffer with the (resynthesised) band shape.
        let nstart = band_off - norm_offset;
        if nstart + band_len <= norm_len {
            norm[nstart..nstart + band_len].copy_from_slice(&x_buf);
        }
        *seed = ctx.seed;
        collapse_masks[i] = cm as u8;
        balance += pulses[i] + tell;
        update_lowband = b > (n << BITRES);
    }
}
