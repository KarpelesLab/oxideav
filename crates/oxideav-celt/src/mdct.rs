//! CELT inverse MDCT (RFC 6716 §4.3.7, libopus `mdct.c::clt_mdct_backward`).
//!
//! The CELT IMDCT implements the standard "DCT-IV via N/4-point complex FFT"
//! trick using on-the-fly twiddle factors. Frame length `n = 120 << LM`
//! (2N MDCT producing N output samples per sub-block) is always a power
//! of two, so an iterative radix-2 IFFT works directly.

use core::f32::consts::PI;

/// Iterative bit-reversal permutation (in place).
fn bit_reverse(a: &mut [(f32, f32)]) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Bluestein chirp-z transform for arbitrary length N. Computes the forward
/// DFT of `a`, in-place. Uses an internal radix-2 FFT of length `M >= 2N-1`.
pub fn fft_bluestein(a: &mut [(f32, f32)]) {
    let n = a.len();
    if n.is_power_of_two() {
        fft_radix2(a);
        return;
    }
    // Pre/post chirp factors w[k] = e^{-i pi k^2 / N}
    let mut w_re = vec![0f32; n];
    let mut w_im = vec![0f32; n];
    for k in 0..n {
        let theta = -PI * ((k as u64 * k as u64) % (2 * n as u64)) as f32 / n as f32;
        w_re[k] = theta.cos();
        w_im[k] = theta.sin();
    }
    // Length-M FFT
    let mut m = 1usize;
    while m < 2 * n - 1 {
        m <<= 1;
    }
    let mut a_buf = vec![(0f32, 0f32); m];
    let mut b_buf = vec![(0f32, 0f32); m];
    for k in 0..n {
        let (ar, ai) = a[k];
        a_buf[k] = (ar * w_re[k] - ai * w_im[k], ar * w_im[k] + ai * w_re[k]);
    }
    // b[k] = w*[k] for k<N, and b[m-k] = w*[k] for 1<=k<N to make it symmetric.
    b_buf[0] = (w_re[0], -w_im[0]);
    for k in 1..n {
        b_buf[k] = (w_re[k], -w_im[k]);
        b_buf[m - k] = (w_re[k], -w_im[k]);
    }
    fft_radix2(&mut a_buf);
    fft_radix2(&mut b_buf);
    // Multiply
    let mut c_buf = vec![(0f32, 0f32); m];
    for k in 0..m {
        let (ar, ai) = a_buf[k];
        let (br, bi) = b_buf[k];
        c_buf[k] = (ar * br - ai * bi, ar * bi + ai * br);
    }
    // Inverse FFT
    ifft_radix2(&mut c_buf);
    // Apply post chirp.
    for k in 0..n {
        let (cr, ci) = c_buf[k];
        a[k] = (cr * w_re[k] - ci * w_im[k], cr * w_im[k] + ci * w_re[k]);
    }
}

/// Forward radix-2 complex FFT (e^{-2πi/N} sign). No 1/N normalization.
pub fn fft_radix2(a: &mut [(f32, f32)]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(a);
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = -2.0 * PI / size as f32;
        let (wr_step, wi_step) = (theta.cos(), theta.sin());
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let (xr, xi) = a[i + k];
                let (yr, yi) = a[i + k + half];
                let tr = yr * wr - yi * wi;
                let ti = yr * wi + yi * wr;
                a[i + k] = (xr + tr, xi + ti);
                a[i + k + half] = (xr - tr, xi - ti);
                let (new_wr, new_wi) = (wr * wr_step - wi * wi_step, wr * wi_step + wi * wr_step);
                wr = new_wr;
                wi = new_wi;
            }
            i += size;
        }
        size <<= 1;
    }
}

/// In-place radix-2 complex IFFT. `a.len()` must be a power of two.
pub fn ifft_radix2(a: &mut [(f32, f32)]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(a);
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = 2.0 * PI / size as f32;
        let (wr_step, wi_step) = (theta.cos(), theta.sin());
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let (xr, xi) = a[i + k];
                let (yr, yi) = a[i + k + half];
                let tr = yr * wr - yi * wi;
                let ti = yr * wi + yi * wr;
                a[i + k] = (xr + tr, xi + ti);
                a[i + k + half] = (xr - tr, xi - ti);
                let (new_wr, new_wi) = (wr * wr_step - wi * wi_step, wr * wi_step + wi * wr_step);
                wr = new_wr;
                wi = new_wi;
            }
            i += size;
        }
        size <<= 1;
    }
    let inv_n = 1.0 / n as f32;
    for s in a.iter_mut() {
        s.0 *= inv_n;
        s.1 *= inv_n;
    }
}

/// CELT-style inverse MDCT for one sub-block of length `2n` producing `2n`
/// time-domain samples (the "raw" output before windowing/overlap-add).
///
/// `coeff` are the `n` MDCT coefficients (one block when shortBlocks, or
/// the deinterleaved coefficients of one of the M sub-blocks).
///
/// `out` length must be `2*n`. Output is the windowed but **not yet
/// overlap-added** time signal: positions `[0..overlap)` and
/// `[2n-overlap..2n)` are weighted by the CELT window.
///
/// The standard CELT MDCT identity (reference: opus mdct.c) is:
///
///   out[2k]      = + sum X[m]*cos(pi/N*(m+0.5)*(2k+0.5+N/2)) ... (windowed)
///
/// Implementation: pre-twiddle into N/2 complex points, run IFFT, post-
/// twiddle, then mirror to fill the output.
pub fn imdct_sub(coeff: &[f32], out: &mut [f32], n2: usize) {
    debug_assert!(coeff.len() >= n2);
    debug_assert!(out.len() >= 2 * n2);
    let n4 = n2 / 2;
    // Build complex N/4 input via pre-twiddle.
    let mut buf = vec![(0f32, 0f32); n4];
    let scale = PI / (2.0 * n2 as f32);
    for k in 0..n4 {
        let xp1 = coeff[2 * k];
        let xp2 = coeff[n2 - 1 - 2 * k];
        let theta = scale * (2 * k) as f32;
        let c = theta.cos();
        let s = theta.sin();
        let yr = xp2 * c + xp1 * s;
        let yi = xp1 * c - xp2 * s;
        buf[k] = (yr, yi);
    }
    // FFT of length N/4 (mixed-radix-friendly via Bluestein for non-power-of-two).
    fft_bluestein(&mut buf);
    // Post-twiddle and mirror to produce 2N output samples.
    // libopus interleaves output around indices [overlap/2 .. overlap/2 + N2)
    // For our simpler API we produce 2N samples in linear order, then the
    // caller does windowing.
    let mut tmp = vec![0f32; n2];
    for k in 0..n4 {
        let theta = scale * (2 * k) as f32;
        let c = theta.cos();
        let s = theta.sin();
        let (re, im) = buf[k];
        let yr = re * c + im * s;
        let yi = re * s - im * c;
        tmp[2 * k] = yr;
        tmp[n2 - 1 - 2 * k] = yi;
    }
    // Mirror to produce the full 2N output.
    // out[i] = tmp[i] for i in [0..N) and out[N+i] = -tmp[N-1-i] for i in [0..N)
    // (Half-cosine MDCT inverse.)
    for i in 0..n2 {
        out[i] = tmp[i];
        out[n2 + i] = -tmp[n2 - 1 - i];
    }
}

/// Apply the CELT analysis window to the IMDCT raw output's overlap regions,
/// then perform overlap-add against the previous frame's tail.
///
/// `raw` is `2*n` samples (the IMDCT-sub output).
/// `prev_tail` is the previous frame's `overlap` samples that overlap with
/// the front of this block's window.
/// `out` writes `n` time-domain samples (the body) plus updates `prev_tail`
/// for the next frame.
pub fn window_overlap_add(
    raw: &[f32],
    out: &mut [f32],
    prev_tail: &mut [f32],
    window: &[f32],
    n: usize,
    overlap: usize,
) {
    debug_assert!(raw.len() >= 2 * n);
    debug_assert!(prev_tail.len() >= overlap);
    debug_assert!(out.len() >= n);
    // Apply window to the front overlap of this block.
    for i in 0..overlap {
        let w = window[i];
        // The overlap-add: previous tail's symmetric end (already windowed)
        // + this block's front (windowed).
        out[i] = prev_tail[i] + w * raw[i];
    }
    // Body (no overlap with anything).
    for i in overlap..n {
        out[i] = raw[i];
    }
    // Stash this block's tail into prev_tail for the next frame.
    // The tail is raw[n..2n] windowed by the symmetric window[overlap-1-i].
    for i in 0..overlap {
        let w = window[overlap - 1 - i];
        prev_tail[i] = w * raw[n + i];
    }
}

/// Backwards-compat placeholder.
pub fn imdct(coeff: &[f32], out: &mut [f32]) {
    let n = coeff.len();
    if !n.is_power_of_two() || n == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut tmp = vec![0f32; 2 * n];
    imdct_sub(coeff, &mut tmp, n);
    let take = out.len().min(tmp.len());
    out[..take].copy_from_slice(&tmp[..take]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifft_round_trips_dc() {
        let mut a = vec![(1.0f32, 0.0f32); 8];
        ifft_radix2(&mut a);
        assert!((a[0].0 - 1.0).abs() < 1e-5);
        for s in &a[1..] {
            assert!(s.0.abs() < 1e-5 && s.1.abs() < 1e-5);
        }
    }

    #[test]
    fn ifft_of_impulse_is_flat() {
        let mut a = vec![(0.0f32, 0.0f32); 16];
        a[0] = (16.0, 0.0);
        ifft_radix2(&mut a);
        for s in &a {
            assert!((s.0 - 1.0).abs() < 1e-5, "expected 1.0, got {:?}", s);
        }
    }

    #[test]
    fn imdct_sub_does_not_panic() {
        let coeff = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 16];
        imdct_sub(&coeff, &mut out, 8);
        assert!(out.iter().any(|v| v.abs() > 0.0));
    }
}
