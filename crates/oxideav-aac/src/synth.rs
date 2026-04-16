//! Per-channel post-quantisation processing — windowing, IMDCT, overlap-add.
//!
//! ISO/IEC 14496-3 §4.6.11 and §4.6.18.
//!
//! Each channel keeps a 1024-sample overlap history (`prev`). Each frame:
//! 1. Run IMDCT (long: 1024→2048; short: 8 × 128→256).
//! 2. Apply the appropriate window across the full 2N IMDCT output.
//! 3. PCM = `prev + windowed_first_half`; new `prev` = `windowed_second_half`.

use crate::imdct::{imdct_long, imdct_short, LONG_INPUT, SHORT_INPUT};
use crate::syntax::{WindowSequence, WindowShape};
use crate::window::{kbd_long, kbd_short, sine_long, sine_short, LONG_LEN, SHORT_LEN};

pub const FRAME_LEN: usize = 1024;

/// Per-channel running state for IMDCT overlap.
#[derive(Clone, Debug)]
pub struct ChannelState {
    /// Right (already-windowed) half of last frame's IMDCT output, ready to overlap.
    pub prev: Vec<f32>,
    /// Window shape (W bit) of the previous block — determines the previous
    /// block's right-side window for asymmetric long_start/long_stop blocks.
    pub prev_shape: WindowShape,
    /// Window sequence of the previous block.
    pub prev_seq: WindowSequence,
}

impl ChannelState {
    pub fn new() -> Self {
        Self {
            prev: vec![0.0; FRAME_LEN],
            prev_shape: WindowShape::Sine,
            prev_seq: WindowSequence::OnlyLong,
        }
    }
}

impl Default for ChannelState {
    fn default() -> Self {
        Self::new()
    }
}

fn long_window(shape: WindowShape) -> &'static [f32] {
    match shape {
        WindowShape::Sine => sine_long(),
        WindowShape::Kbd => kbd_long(),
    }
}

fn short_window(shape: WindowShape) -> &'static [f32] {
    match shape {
        WindowShape::Sine => sine_short(),
        WindowShape::Kbd => kbd_short(),
    }
}

/// Build the AAC long-window weight vector (length 2N) for `seq`/`shape`.
/// The "previous shape" parameter governs the left half (positions 0..N).
fn long_window_full(seq: WindowSequence, shape: WindowShape, prev_shape: WindowShape) -> Vec<f32> {
    let n = LONG_LEN;
    let mut w = vec![0.0f32; 2 * n];
    let prev_w = long_window(prev_shape);
    let cur_w = long_window(shape);

    match seq {
        WindowSequence::OnlyLong => {
            // Left half: rising slope from previous shape.
            // w[n] = prev_w[n] for n in 0..N (rising)
            // w[N+n] = cur_w[N-1-n] for n in 0..N (falling)
            for i in 0..n {
                w[i] = prev_w[i];
                w[n + i] = cur_w[n - 1 - i];
            }
        }
        WindowSequence::LongStart => {
            // Left half: rising long window (prev shape).
            // Right half: 448 ones, 128 short-window right slope (cur shape), 448 zeros.
            for i in 0..n {
                w[i] = prev_w[i];
            }
            let cur_short = short_window(shape);
            for i in 0..448 {
                w[n + i] = 1.0;
            }
            for i in 0..128 {
                w[n + 448 + i] = cur_short[127 - i];
            }
            // 576..1024 stays zero.
        }
        WindowSequence::LongStop => {
            // Left half: 448 zeros, 128 short-window left slope (prev shape), 448 ones.
            // Right half: long window falling slope (cur shape).
            let prev_short = short_window(prev_shape);
            // 0..448 = 0
            for i in 0..128 {
                w[448 + i] = prev_short[i];
            }
            for i in 576..n {
                w[i] = 1.0;
            }
            for i in 0..n {
                w[n + i] = cur_w[n - 1 - i];
            }
        }
        WindowSequence::EightShort => {
            // Eight-short does not use this path — handled separately.
        }
    }
    w
}

/// Run IMDCT on `spec` (1024 spectral coefs) according to `seq`/`shape`,
/// applying windowing and overlap-add against `state.prev`. Writes 1024
/// PCM samples into `pcm` and updates `state` for the next frame.
pub fn imdct_and_overlap(
    spec: &[f32; 1024],
    seq: WindowSequence,
    shape: WindowShape,
    state: &mut ChannelState,
    pcm: &mut [f32; FRAME_LEN],
) {
    let n = LONG_LEN;
    if !matches!(seq, WindowSequence::EightShort) {
        let mut tmp = vec![0.0f32; 2 * LONG_INPUT];
        imdct_long(spec, &mut tmp);
        let win = long_window_full(seq, shape, state.prev_shape);
        // Apply window.
        for i in 0..(2 * n) {
            tmp[i] *= win[i];
        }
        // PCM = prev + tmp[0..N]
        for i in 0..n {
            pcm[i] = state.prev[i] + tmp[i];
        }
        // New overlap = tmp[N..2N]
        for i in 0..n {
            state.prev[i] = tmp[n + i];
        }
    } else {
        // Eight-short sequence — 8 sub-windows of 128 spectral coefs each,
        // each IMDCT'd to 256 samples then windowed and overlapped per ISO Fig. 4.24.
        let mut shorts = [[0.0f32; 2 * SHORT_INPUT]; 8];
        for w in 0..8 {
            let s = w * SHORT_INPUT;
            let chunk: &[f32] = &spec[s..s + SHORT_INPUT];
            imdct_short(chunk, &mut shorts[w]);
        }
        let cur_short = short_window(shape);
        let prev_short = short_window(state.prev_shape);
        // Window each sub-window. Sub-window 0 uses `prev_short` for its
        // rising side (left), `cur_short` for its falling side (right).
        // Sub-windows 1..7 use `cur_short` on both sides.
        for n_ in 0..SHORT_LEN {
            shorts[0][n_] *= prev_short[n_];
            shorts[0][SHORT_LEN + n_] *= cur_short[SHORT_LEN - 1 - n_];
        }
        for w in 1..8 {
            for n_ in 0..SHORT_LEN {
                shorts[w][n_] *= cur_short[n_];
                shorts[w][SHORT_LEN + n_] *= cur_short[SHORT_LEN - 1 - n_];
            }
        }
        // PCM layout per ISO Fig. 4.24:
        //   pcm[0..448]    = state.prev[0..448]
        //   pcm[448..576]  = state.prev[448..576] + s[0][0..128]
        //   pcm[576..704]  = s[0][128..256]      + s[1][0..128]
        //   pcm[704..832]  = s[1][128..256]      + s[2][0..128]
        //   pcm[832..960]  = s[2][128..256]      + s[3][0..128]
        //   pcm[960..1024] = s[3][128..192]
        for i in 0..448 {
            pcm[i] = state.prev[i];
        }
        for i in 0..SHORT_LEN {
            pcm[448 + i] = state.prev[448 + i] + shorts[0][i];
            pcm[576 + i] = shorts[0][SHORT_LEN + i] + shorts[1][i];
            pcm[704 + i] = shorts[1][SHORT_LEN + i] + shorts[2][i];
            pcm[832 + i] = shorts[2][SHORT_LEN + i] + shorts[3][i];
        }
        for i in 0..64 {
            pcm[960 + i] = shorts[3][SHORT_LEN + i];
        }
        // Build new overlap (= s[3][192..256]+s[4][0..64], etc.):
        //   prev[0..64]    = s[3][192..256] + s[4][0..64]
        //   prev[64..192]  = s[4][64..192]  + s[5][0..128]
        //   prev[192..320] = s[5][128..256] + s[6][0..128]
        //   prev[320..448] = s[6][128..256] + s[7][0..128]
        //   prev[448..576] = s[7][128..256]
        //   prev[576..1024]= 0
        for i in 0..64 {
            state.prev[i] = shorts[3][SHORT_LEN + 64 + i] + shorts[4][i];
        }
        for i in 0..SHORT_LEN {
            state.prev[64 + i] = shorts[4][SHORT_LEN + i] + shorts[5][i];
            state.prev[192 + i] = shorts[5][SHORT_LEN + i] + shorts[6][i];
            state.prev[320 + i] = shorts[6][SHORT_LEN + i] + shorts[7][i];
            state.prev[448 + i] = shorts[7][SHORT_LEN + i];
        }
        for i in 576..LONG_LEN {
            state.prev[i] = 0.0;
        }
    }
    state.prev_shape = shape;
    state.prev_seq = seq;
}
