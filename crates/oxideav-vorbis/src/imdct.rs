//! Inverse Modified Discrete Cosine Transform + sin/sin window for Vorbis.
//!
//! The IMDCT here is the textbook O(N²) form. Vorbis blocksizes top out at
//! 8192, so a single transform is ~33M multiplies — slow but correct.
//! Optimisation (split-radix FFT, pre-twiddled butterflies) is left for
//! later passes.
//!
//! Vorbis I §1.3.4 (windowing) and §1.3.5 (IMDCT).

use std::f64::consts::PI;

/// Sin window value for position `i` in a window of length `n`.
///
/// Vorbis I §1.3.4: `W[i] = sin(0.5π · sin²((i + 0.5)/n · π))`.
/// Symmetric about `n/2` with W[0] and W[n-1] both near 0.
pub fn sin_window_sample(i: usize, n: usize) -> f32 {
    let inner = ((i as f64 + 0.5) / n as f64) * PI;
    let s = inner.sin();
    let outer = 0.5 * PI * s * s;
    outer.sin() as f32
}

/// Build the asymmetric Vorbis window for a block of length `n` given the
/// neighbouring window flags. The returned vector has length `n`. The four
/// transition cases come from §1.3.4 and depend on whether the previous /
/// next packet is a long block when this packet is also a long block.
///
/// For short blocks (always symmetric), `prev_long` and `next_long` are
/// ignored and a full sin window of length `n` is returned.
pub fn build_window(n: usize, blockflag: bool, prev_long: bool, next_long: bool) -> Vec<f32> {
    let mut w = vec![0f32; n];
    if !blockflag {
        // Short: symmetric sin window of length n.
        for i in 0..n {
            w[i] = sin_window_sample(i, n);
        }
        return w;
    }
    // Long: split into 4 quarters. Each "side" can be short or long depending
    // on the neighbour flag.
    let n2 = n / 2;
    let n4 = n / 4;
    let n8 = n / 8;
    let prev_n = if prev_long { n2 } else { n8 * 2 };
    let next_n = if next_long { n2 } else { n8 * 2 };
    let prev_start = n4 - prev_n / 2;
    let prev_end = prev_start + prev_n;
    let next_start = 3 * n4 - next_n / 2;
    let next_end = next_start + next_n;
    for i in 0..n {
        if i < prev_start {
            w[i] = 0.0;
        } else if i < prev_end {
            // Rising edge of a sin window of length prev_n.
            w[i] = sin_window_sample(i - prev_start, prev_n);
        } else if i < next_start {
            w[i] = 1.0;
        } else if i < next_end {
            // Falling edge of a sin window of length next_n.
            w[i] = sin_window_sample(next_n - 1 - (i - next_start), next_n);
        } else {
            w[i] = 0.0;
        }
    }
    w
}

/// Naive O(N²) IMDCT. Input has length N/2 (frequency-domain coefficients),
/// output has length N (time-domain samples).
///
/// Standard MDCT inverse:
///   x[n] = sum_{k=0}^{N/2 - 1} X[k] * cos(π/(2N) * (2n + 1 + N/2) * (2k + 1))
///
/// The windowing/normalization factor is left to the caller (multiply by
/// the window after this returns).
pub fn imdct_naive(spectrum: &[f32], output: &mut [f32]) {
    let half = spectrum.len();
    let n = half * 2;
    debug_assert_eq!(output.len(), n);
    // Vorbis I §1.3.5: X_n = (2/N) Σ Y_k cos(π/N * (n + 0.5 + N/4) * (2k+1))
    let scale = PI / (2.0 * n as f64);
    let nh = n as f64 / 2.0;
    let norm = 2.0 / n as f64;
    for i in 0..n {
        let base = (2.0 * i as f64 + 1.0 + nh) * scale;
        let mut acc = 0f64;
        for k in 0..half {
            let phase = base * (2.0 * k as f64 + 1.0);
            acc += spectrum[k] as f64 * phase.cos();
        }
        output[i] = (acc * norm) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_endpoints_short_block() {
        // Symmetric sin window: w[0] is small, w[n/2-1] is near 1 / 1, w[n-1] is small.
        let w = build_window(64, false, false, false);
        assert!(w[0] < 0.05);
        assert!(w[63] < 0.05);
        // Window squared should sum to ~n/2 (orthogonality with sin overlap).
        let sumsq: f32 = w.iter().map(|x| x * x).sum();
        assert!((sumsq - 32.0).abs() < 0.5, "sumsq = {sumsq}");
    }

    #[test]
    fn imdct_constant_input() {
        // A constant in frequency domain produces a (mostly) cosine in time.
        let spec = vec![1.0f32; 8];
        let mut out = vec![0f32; 16];
        imdct_naive(&spec, &mut out);
        // Output should be finite.
        for v in &out {
            assert!(v.is_finite());
        }
    }
}
