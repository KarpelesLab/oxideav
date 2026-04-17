//! 24-tap QMF (Quadrature Mirror Filter) analysis + synthesis for G.722.
//!
//! Implementation follows the ITU-T G.722 / SpanDSP polyphase structure
//! (Steve Underwood, public domain). The filter uses a symmetric 24-tap
//! response stored as 12 unique coefficients; analysis and synthesis both
//! index even and odd positions of a 24-sample shift register, one in
//! ascending and the other in descending coefficient order.
//!
//! - Analysis: two input PCM samples per call → one low-band + one
//!   high-band sample (both at 8 kHz).
//! - Synthesis: one low/high band pair per call → two output PCM samples
//!   (at 16 kHz).

/// 12 unique QMF coefficients (the 24-tap symmetric response is reconstructed
/// by the ascending / descending indexing in [`QmfAnalysis::process`] and
/// [`QmfSynthesis::process`]). Values match the ITU-T reference table used
/// by SpanDSP's public-domain `libg722`.
pub(crate) const QMF_COEFFS: [i32; 12] =
    [3, -11, 12, 32, -210, 951, 3876, -805, 362, -156, 53, -11];

/// Analysis-filter state: 24-sample shift register. `x[22]`/`x[23]` are the
/// newest samples; `x[0]`/`x[1]` the oldest.
#[derive(Clone, Debug)]
pub struct QmfAnalysis {
    x: [i32; 24],
}

impl Default for QmfAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfAnalysis {
    pub fn new() -> Self {
        Self { x: [0; 24] }
    }

    /// Consume two PCM samples at 16 kHz (`s0` is the first, `s1` the
    /// second) and return the `(low_band, high_band)` pair at 8 kHz.
    pub fn process(&mut self, s0: i16, s1: i16) -> (i16, i16) {
        // Shift the history down by two positions (older samples move left).
        for i in 0..22 {
            self.x[i] = self.x[i + 2];
        }
        self.x[22] = s0 as i32;
        self.x[23] = s1 as i32;

        let mut sumeven: i32 = 0;
        let mut sumodd: i32 = 0;
        for i in 0..12 {
            sumodd = sumodd.saturating_add(self.x[2 * i].saturating_mul(QMF_COEFFS[i]));
            sumeven = sumeven.saturating_add(self.x[2 * i + 1].saturating_mul(QMF_COEFFS[11 - i]));
        }
        // The spandsp reference uses >> 14; the resulting magnitude fits in
        // the low-band's 15-bit range.
        let xlow = clip15((sumeven + sumodd) >> 14);
        let xhigh = clip15((sumeven - sumodd) >> 14);
        (xlow, xhigh)
    }
}

/// Synthesis-filter state: 24-sample shift register (same structure as
/// analysis, but populated from `rlow ± rhigh`).
#[derive(Clone, Debug)]
pub struct QmfSynthesis {
    x: [i32; 24],
}

impl Default for QmfSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfSynthesis {
    pub fn new() -> Self {
        Self { x: [0; 24] }
    }

    /// Consume one reconstructed low-band / high-band pair and produce two
    /// 16-bit output samples at 16 kHz.
    pub fn process(&mut self, rlow: i16, rhigh: i16) -> (i16, i16) {
        for i in 0..22 {
            self.x[i] = self.x[i + 2];
        }
        self.x[22] = rlow as i32 + rhigh as i32;
        self.x[23] = rlow as i32 - rhigh as i32;

        let mut xout1: i32 = 0;
        let mut xout2: i32 = 0;
        for i in 0..12 {
            xout2 = xout2.saturating_add(self.x[2 * i].saturating_mul(QMF_COEFFS[i]));
            xout1 = xout1.saturating_add(self.x[2 * i + 1].saturating_mul(QMF_COEFFS[11 - i]));
        }
        (clip16(xout1 >> 11), clip16(xout2 >> 11))
    }
}

fn clip15(v: i32) -> i16 {
    v.clamp(-16_384, 16_383) as i16
}

fn clip16(v: i32) -> i16 {
    v.clamp(-32_768, 32_767) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_silence_stays_silent() {
        let mut qmf = QmfAnalysis::new();
        for _ in 0..50 {
            let (lo, hi) = qmf.process(0, 0);
            assert_eq!(lo, 0);
            assert_eq!(hi, 0);
        }
    }

    #[test]
    fn synthesis_silence_stays_silent() {
        let mut qmf = QmfSynthesis::new();
        for _ in 0..50 {
            let (a, b) = qmf.process(0, 0);
            assert_eq!(a, 0);
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn qmf_roundtrip_invertible() {
        // Feed a 1 kHz sine at 16 kHz through analysis → synthesis with no
        // ADPCM in between and verify PSNR > 30 dB.
        let mut ana = QmfAnalysis::new();
        let mut syn = QmfSynthesis::new();
        let n = 800;
        let mut inputs: Vec<i16> = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 16_000.0;
            inputs.push(((2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 10_000.0) as i16);
        }
        let mut outputs: Vec<i16> = Vec::with_capacity(n);
        for ch in inputs.chunks_exact(2) {
            let (lo, hi) = ana.process(ch[0], ch[1]);
            let (a, b) = syn.process(lo, hi);
            outputs.push(a);
            outputs.push(b);
        }
        let mut best = f64::NEG_INFINITY;
        let mut best_d = 0usize;
        for delay in 0..64 {
            let mut err = 0.0f64;
            let mut sig = 0.0f64;
            for i in delay..n {
                let x = inputs[i - delay] as f64;
                let y = outputs[i] as f64;
                err += (x - y).powi(2);
                sig += x * x;
            }
            let psnr = if err > 0.0 {
                10.0 * (sig / err).log10()
            } else {
                200.0
            };
            if psnr > best {
                best = psnr;
                best_d = delay;
            }
        }
        assert!(
            best > 30.0,
            "QMF not sufficiently invertible: best PSNR {best:.2} dB at delay {best_d}"
        );
    }
}
