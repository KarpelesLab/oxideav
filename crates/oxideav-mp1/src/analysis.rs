//! MPEG-1 Audio Layer I forward polyphase analysis filter bank.
//!
//! Forward counterpart of [`crate::synthesis`]. 32 PCM input samples per
//! step produce 32 subband samples, identical in structure to the Layer
//! II / III analysis step (ISO/IEC 11172-3 §C.1.3).
//!
//! The analysis window `C[i]` is the synthesis window `D[i]` scaled by
//! `1/32` (Annex C). We reuse [`crate::window::SYNTHESIS_WINDOW`]
//! directly and multiply by `1/32` in the windowing step.
//!
//! The analysis matrix is `M_a[i][k] = cos((2i + 1)(k - 16) π / 64)`
//! for `i = 0..32`, `k = 0..64`.

use crate::window::SYNTHESIS_WINDOW;
use std::sync::OnceLock;

/// Number of PCM samples per Layer I frame per channel (32 subbands × 12
/// samples).
pub const SAMPLES_PER_FRAME: usize = 384;

const ANALYSIS_NORM: f32 = 1.0 / 32.0;

/// Per-channel analysis state: a 512-sample input FIFO.
pub struct AnalysisState {
    x: Box<[f32; 512]>,
}

impl Default for AnalysisState {
    fn default() -> Self {
        Self {
            x: Box::new([0.0; 512]),
        }
    }
}

impl AnalysisState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed 32 PCM samples (chronological order, oldest first), produce
    /// 32 subband samples.
    pub fn analyze(&mut self, pcm: &[f32; 32], out: &mut [f32; 32]) {
        // 1. Shift FIFO by 32 (older samples slide forward).
        for i in (32..512).rev() {
            self.x[i] = self.x[i - 32];
        }
        // 2. Insert new samples in REVERSE (newest at index 0).
        for i in 0..32 {
            self.x[i] = pcm[31 - i];
        }

        // 3. Window with C[i] = D[i] / 32.
        let mut z = [0.0f32; 512];
        for i in 0..512 {
            z[i] = SYNTHESIS_WINDOW[i] * ANALYSIS_NORM * self.x[i];
        }

        // 4. Partial sum into 64 entries.
        let mut y = [0.0f32; 64];
        for i in 0..64 {
            let mut acc = 0.0f32;
            for j in 0..8 {
                acc += z[i + 64 * j];
            }
            y[i] = acc;
        }

        // 5. Matrix multiply.
        let m = matrix();
        for i in 0..32 {
            let mut acc = 0.0f32;
            for k in 0..64 {
                acc += m[i][k] * y[k];
            }
            out[i] = acc;
        }
    }
}

/// Run 12 analysis steps over 384 PCM samples for one Layer I frame,
/// producing a 32 × 12 subband buffer (subband-major).
pub fn analyze_frame(
    state: &mut AnalysisState,
    pcm: &[f32; SAMPLES_PER_FRAME],
    subbands_out: &mut [[f32; 12]; 32],
) {
    for step in 0..12 {
        let mut pcm32 = [0.0f32; 32];
        pcm32.copy_from_slice(&pcm[step * 32..step * 32 + 32]);
        let mut sub = [0.0f32; 32];
        state.analyze(&pcm32, &mut sub);
        for sb in 0..32 {
            subbands_out[sb][step] = sub[sb];
        }
    }
}

// 32×64 analysis matrix.
static MATRIX_STORAGE: OnceLock<[[f32; 64]; 32]> = OnceLock::new();

fn matrix() -> &'static [[f32; 64]; 32] {
    MATRIX_STORAGE.get_or_init(|| {
        let mut m = [[0.0f32; 64]; 32];
        let pi = std::f64::consts::PI;
        for i in 0..32 {
            for k in 0..64 {
                let angle = ((2 * i + 1) as f64) * ((k as f64) - 16.0) * pi / 64.0;
                m[i][k] = angle.cos() as f32;
            }
        }
        m
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_zero_is_zero() {
        let mut s = AnalysisState::new();
        let pcm = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        s.analyze(&pcm, &mut out);
        for v in out.iter() {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn analyze_dc_finite() {
        let mut s = AnalysisState::new();
        let pcm = [0.5f32; 32];
        let mut out = [0.0f32; 32];
        for _ in 0..32 {
            s.analyze(&pcm, &mut out);
        }
        for v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn analyze_frame_shape() {
        let mut s = AnalysisState::new();
        let pcm = [0.1f32; 384];
        let mut sub = [[0.0f32; 12]; 32];
        analyze_frame(&mut s, &pcm, &mut sub);
        for sb in 0..32 {
            for t in 0..12 {
                assert!(sub[sb][t].is_finite());
            }
        }
    }
}
