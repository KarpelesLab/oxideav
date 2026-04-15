//! MPEG-1 Audio subband synthesis filter bank.
//!
//! Implements the 32-band polyphase filter described in ISO/IEC 11172-3
//! Annex B / D. For each Layer I granule (1 frame holds 12 granules of 32
//! subband samples per channel) the filter produces 32 PCM output samples
//! per channel.
//!
//! Algorithm (per-channel state carries a 1024-sample FIFO `v`):
//!   1. For i = 1023..=64: `v[i] = v[i - 64]`.
//!   2. Matrix: `v[i] = sum_{k=0..31} N[i][k] * s[k]` for i = 0..63,
//!      where `N[i][k] = cos((16 + i) * (2k + 1) * pi / 64)`.
//!   3. Build 512 "u" taps: for i = 0..7, j = 0..31:
//!       u[64*i + j]      = v[128*i + j]
//!       u[64*i + 32 + j] = v[128*i + 96 + j]
//!   4. Window: `w[i] = u[i] * D[i]` for i = 0..511.
//!   5. Output: `pcm[j] = sum_{i=0..15} w[32*i + j]` for j = 0..31.

use crate::window::SYNTHESIS_WINDOW;

/// Per-channel synthesis state. The 1024-sample FIFO `v` accumulates matrixed
/// subband samples; it must survive between granules and frames.
pub struct SynthesisState {
    v: Box<[f32; 1024]>,
}

impl Default for SynthesisState {
    fn default() -> Self {
        Self {
            v: Box::new([0.0; 1024]),
        }
    }
}

impl SynthesisState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run one synthesis step with 32 subband samples, producing 32 PCM samples
    /// (as f32, in roughly the [-1.0, 1.0] range).
    pub fn synthesize(&mut self, subbands: &[f32; 32], out: &mut [f32; 32]) {
        // 1. Shift v by 64.
        for i in (64..1024).rev() {
            self.v[i] = self.v[i - 64];
        }

        // 2. Matrix step — compute v[0..64] from the 32 subband samples.
        // N[i][k] = cos(((2k + 1) * (16 + i)) * PI / 64). We compute using the
        // on-the-fly cosine (no extra tables needed for a first pass — this
        // can be replaced with a precomputed matrix or fast-DCT for speed).
        for i in 0..64 {
            let mut acc = 0.0f32;
            for k in 0..32 {
                acc += MATRIX[i][k] * subbands[k];
            }
            self.v[i] = acc;
        }

        // 3. Build the 512-sample u[] from v[].
        let mut u = [0.0f32; 512];
        for i in 0..8 {
            for j in 0..32 {
                u[64 * i + j] = self.v[128 * i + j];
                u[64 * i + 32 + j] = self.v[128 * i + 96 + j];
            }
        }

        // 4. Window with D[].
        for i in 0..512 {
            u[i] *= SYNTHESIS_WINDOW[i];
        }

        // 5. Output 32 PCM samples.
        for j in 0..32 {
            let mut acc = 0.0f32;
            for i in 0..16 {
                acc += u[32 * i + j];
            }
            out[j] = acc;
        }
    }
}

// Lazily build the 64x32 matrix at first use — avoids a 8 KiB literal and lets
// the compiler bake it into a `.rodata` section via `LazyLock` once Rust 1.80
// is the MSRV (it is). For simplicity we just use a `const fn` builder via
// `OnceLock`.
use std::sync::OnceLock;

static MATRIX_STORAGE: OnceLock<[[f32; 32]; 64]> = OnceLock::new();

#[allow(non_upper_case_globals)]
fn matrix() -> &'static [[f32; 32]; 64] {
    MATRIX_STORAGE.get_or_init(build_matrix)
}

// Access helper — `MATRIX[i][k]` reads lazily.
struct MatrixProxy;
static MATRIX: MatrixProxy = MatrixProxy;

impl std::ops::Index<usize> for MatrixProxy {
    type Output = [f32; 32];
    fn index(&self, i: usize) -> &Self::Output {
        &matrix()[i]
    }
}

fn build_matrix() -> [[f32; 32]; 64] {
    let mut m = [[0.0f32; 32]; 64];
    let pi = std::f64::consts::PI;
    for i in 0..64 {
        for k in 0..32 {
            let angle = ((2 * k + 1) as f64) * ((16 + i) as f64) * pi / 64.0;
            m[i][k] = angle.cos() as f32;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_produces_silence_for_zero_subbands() {
        let mut state = SynthesisState::new();
        let sub = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        // Run a few iterations so the filter state fully settles.
        for _ in 0..16 {
            state.synthesize(&sub, &mut out);
        }
        for s in out.iter() {
            assert!(s.abs() < 1e-6, "silence expected, got {s}");
        }
    }

    #[test]
    fn synthesis_produces_finite_output_for_impulse() {
        let mut state = SynthesisState::new();
        let mut sub = [0.0f32; 32];
        sub[0] = 1.0;
        let mut out = [0.0f32; 32];
        state.synthesize(&sub, &mut out);
        for s in out.iter() {
            assert!(s.is_finite());
        }
    }
}
