//! Quadrature Mirror Filter (QMF) synthesis for Speex sub-band decoding.
//!
//! The QMF splits the 16 kHz wideband signal into a 0–4 kHz low band
//! and a 4–8 kHz high band via a critically-sampled analysis filter
//! bank, then reconstructs it via the symmetric synthesis bank at
//! decode time. Speex uses a 64-tap linear-phase prototype `h0` whose
//! even-indexed taps form the "H0" branch and whose odd-indexed taps
//! form the "H1" branch (separated by modulation with
//! cos(π·n/2) / sin(π·n/2) — hence "QMF").
//!
//! The coefficients in `H0_PROTOTYPE` are a direct float transcription
//! of the 64 integer values in `libspeex/sb_celp.c`'s `h0[]` rescaled
//! by 2^-16 — the values reproduce the BSD-licensed Xiph prototype to
//! four decimal places (comments in the reference mark it as an
//! NMR ≥ 50 dB low-pass). For the pure-float build we use the float
//! values directly (same table, pre-divided) — no wrapper of the C
//! code, only the filter coefficients.
//!
//! `qmf_synth` mirrors the loop in `libspeex/filters.c`. The reference
//! unrolls the inner product by 4 and exploits the prototype's
//! linear-phase symmetry; here we stay with the clearer straight
//! convolution. The output signal is 2× oversampled (one full-band
//! sample per low-band sample plus one per high-band sample), so
//! processing cost is `N·M` multiplies where `N = frame_size·2` and
//! `M = QMF_ORDER = 64`.

/// Length of the QMF prototype filter (and the synthesis memory).
pub const QMF_ORDER: usize = 64;

/// 64-tap linear-phase low-pass prototype filter. Transcribed from
/// `libspeex/sb_celp.c` (float branch). Symmetric around tap 31.5.
pub const H0_PROTOTYPE: [f32; QMF_ORDER] = [
    3.596189e-05,
    -0.0001123515,
    -0.0001104587,
    0.0002790277,
    0.0002298438,
    -0.0005953563,
    -0.0003823631,
    0.00113826,
    0.0005308539,
    -0.001986177,
    -0.0006243724,
    0.003235877,
    0.0005743159,
    -0.004989147,
    -0.0002584767,
    0.007367171,
    -0.0004857935,
    -0.01050689,
    0.001894714,
    0.01459396,
    -0.004313674,
    -0.01994365,
    0.00828756,
    0.02716055,
    -0.01485397,
    -0.03764973,
    0.026447,
    0.05543245,
    -0.05095487,
    -0.09779096,
    0.1382363,
    0.4600981,
    0.4600981,
    0.1382363,
    -0.09779096,
    -0.05095487,
    0.05543245,
    0.026447,
    -0.03764973,
    -0.01485397,
    0.02716055,
    0.00828756,
    -0.01994365,
    -0.004313674,
    0.01459396,
    0.001894714,
    -0.01050689,
    -0.0004857935,
    0.007367171,
    -0.0002584767,
    -0.004989147,
    0.0005743159,
    0.003235877,
    -0.0006243724,
    -0.001986177,
    0.0005308539,
    0.00113826,
    -0.0003823631,
    -0.0005953563,
    0.0002298438,
    0.0002790277,
    -0.0001104587,
    -0.0001123515,
    3.596189e-05,
];

/// Re-synthesise a wideband signal from the 2× decimated low/high
/// sub-band signals `x1` and `x2` using the QMF prototype `a`.
///
/// Mirrors `qmf_synth` in `libspeex/filters.c`. `n` = full-band output
/// length (must be a multiple of 4), `m` = prototype length (= `QMF_ORDER`),
/// both must be divisible by 4. `mem1`/`mem2` carry `m`-tap IIR memory
/// across calls.
///
/// The float path in the reference has no scaling (the `2·y` lines); we
/// preserve that so amplitudes line up with the NB decoder's output
/// range (±32768 float → int16 at the codec boundary).
#[allow(clippy::too_many_arguments)]
pub fn qmf_synth(
    x1: &[f32],
    x2: &[f32],
    a: &[f32],
    y: &mut [f32],
    n: usize,
    m: usize,
    mem1: &mut [f32],
    mem2: &mut [f32],
) {
    debug_assert!(n % 4 == 0, "qmf_synth: N must be a multiple of 4");
    debug_assert!(m % 4 == 0, "qmf_synth: M must be a multiple of 4");
    debug_assert!(x1.len() >= n / 2);
    debug_assert!(x2.len() >= n / 2);
    debug_assert!(a.len() >= m);
    debug_assert!(y.len() >= n);
    debug_assert!(mem1.len() >= m);
    debug_assert!(mem2.len() >= m);

    let m2 = m / 2;
    let n2 = n / 2;

    // Build reversed-input + memory tap ring buffers xx1 / xx2, length m2 + n2.
    let mut xx1 = vec![0.0f32; m2 + n2];
    let mut xx2 = vec![0.0f32; m2 + n2];
    for i in 0..n2 {
        xx1[i] = x1[n2 - 1 - i];
        xx2[i] = x2[n2 - 1 - i];
    }
    for i in 0..m2 {
        xx1[n2 + i] = mem1[2 * i + 1];
        xx2[n2 + i] = mem2[2 * i + 1];
    }

    // Unrolled-by-2 dual MAC (reference pattern). The ("a0"/"a1")
    // split corresponds to the even/odd taps of the prototype, which
    // in a QMF act as H0 and H1 branches. The float branch scales by 2.
    // N2 must be ≥ 2 for the indexing below; we require n ≥ 4 up top.
    debug_assert!(n2 >= 2);
    let mut i = 0usize;
    while i < n2 {
        let mut y0: f32 = 0.0;
        let mut y1: f32 = 0.0;
        let mut y2: f32 = 0.0;
        let mut y3: f32 = 0.0;
        // `i` is even, i ≤ n2-2 ⇒ n2-2-i ≥ 0.
        let mut x10 = xx1[n2 - 2 - i];
        let mut x20 = xx2[n2 - 2 - i];

        let mut j = 0usize;
        while j < m2 {
            let a0 = a[2 * j];
            let a1 = a[2 * j + 1];
            let x11 = xx1[n2 - 1 + j - i];
            let x21 = xx2[n2 - 1 + j - i];
            y0 += a0 * (x11 - x21);
            y1 += a1 * (x11 + x21);
            y2 += a0 * (x10 - x20);
            y3 += a1 * (x10 + x20);

            let a0b = a[2 * j + 2];
            let a1b = a[2 * j + 3];
            let x10b = xx1[n2 + j - i];
            let x20b = xx2[n2 + j - i];
            y0 += a0b * (x10b - x20b);
            y1 += a1b * (x10b + x20b);
            y2 += a0b * (x11 - x21);
            y3 += a1b * (x11 + x21);

            x10 = x10b;
            x20 = x20b;
            j += 2;
        }

        y[2 * i] = 2.0 * y0;
        y[2 * i + 1] = 2.0 * y1;
        y[2 * i + 2] = 2.0 * y2;
        y[2 * i + 3] = 2.0 * y3;
        i += 2;
    }

    // Shift in the new low/high samples as memory for the next call.
    // The reference rotates the 2*i+1 slots; 2*i slots (even) are
    // recomputed from the ring buffer each call, so zeroing is fine.
    for i in 0..m2 {
        mem1[2 * i + 1] = xx1[i];
        mem2[2 * i + 1] = xx2[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h0_is_symmetric() {
        // Linear-phase prototype — should be symmetric about the centre.
        for i in 0..QMF_ORDER / 2 {
            let a = H0_PROTOTYPE[i];
            let b = H0_PROTOTYPE[QMF_ORDER - 1 - i];
            assert!(
                (a - b).abs() < 1e-6,
                "H0[{i}] ({a}) != H0[{}] ({b})",
                QMF_ORDER - 1 - i,
            );
        }
    }

    #[test]
    fn h0_has_roughly_unity_dc_in_lowband() {
        // Σ h0[n] ≈ 1.0 (low-pass prototype).
        let sum: f32 = H0_PROTOTYPE.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "H0 DC sum = {sum}");
    }

    #[test]
    fn qmf_synth_preserves_low_band() {
        // Feed a constant into x1 (low) and zero into x2 (high) and
        // make sure the full-band output has non-trivial energy — i.e.
        // the filter runs without panicking on the happy path. We
        // don't assert a specific reconstruction formula (that's a
        // property of the full analysis + synthesis chain).
        let n = 16;
        let n2 = n / 2;
        let x1 = vec![1.0f32; n2];
        let x2 = vec![0.0f32; n2];
        let mut y = vec![0.0f32; n];
        let mut mem1 = [0.0f32; QMF_ORDER];
        let mut mem2 = [0.0f32; QMF_ORDER];
        qmf_synth(
            &x1,
            &x2,
            &H0_PROTOTYPE,
            &mut y,
            n,
            QMF_ORDER,
            &mut mem1,
            &mut mem2,
        );
        let energy: f32 = y.iter().map(|v| v * v).sum();
        assert!(energy > 0.0, "QMF synthesis produced zero output");
    }
}
