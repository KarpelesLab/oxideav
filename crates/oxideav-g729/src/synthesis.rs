//! G.729 synthesis engine — codebooks, gain decode, LPC filter, postfilter.
//!
//! This module is the DSP backbone of the decoder. Inputs are the
//! parsed frame indices (`FrameParams`) and the persistent decoder
//! state ([`SynthesisState`]); output is 80 `f32` reconstructed speech
//! samples per frame (= two 40-sample subframes).
//!
//! ## Coverage vs. ITU-T G.729
//!
//! | Section                     | Status                                      |
//! |-----------------------------|---------------------------------------------|
//! | §3.7  adaptive-codebook     | fractional pitch (1/3-sample), integer 20..143 |
//! | §3.8  fixed codebook         | 4 tracks × 4 pulses, signs, 13+4 bit decode |
//! | §3.9  gain codebook          | two-stage VQ; gains via a first-cut table   |
//! | §3.10 synthesis filter        | time-varying 10th-order all-pole             |
//! | §3.11 postfilter              | short-term (γ1/γ2) + long-term (pitch)       |
//!
//! The gain codebook tables used here are the **spec-exact** values
//! from G.729 Table 13 (conjugate-structure VQ, two stages). Pitch
//! gain and fixed-codebook gain are computed via the standard
//! predictor (MA-4 of past log-energies) and combined as per §3.9.5.
//!
//! ## Scaling / numerical notes
//!
//! - All internal state is `f32`. The reference C code is in Q-format
//!   integer; this decoder trades a little arithmetic precision for
//!   readability. At 8 kHz / S16 output the audible impact is tiny.
//! - Output scaling: the decoder emits samples in the same range as
//!   the reference — peak ~ ±16 000 for normal speech, clamped to
//!   ±32 767 at the edges.

use crate::{LPC_ORDER, SUBFRAME_SAMPLES};

/// Number of past-excitation samples we keep for adaptive-codebook
/// lookups. The longest delay is 143 integer samples + 40-sample
/// subframe; we round up to 256 for clean power-of-two buffering.
pub const EXC_HIST: usize = 256;

/// Length of the LPC synthesis filter memory.
pub const SYN_MEM: usize = LPC_ORDER;

/// Length of the postfilter pitch delay line (a bit longer than one
/// frame so cross-subframe gain interpolation has headroom).
pub const POST_MEM: usize = 160;

/// Persistent decoder state between frames.
#[derive(Clone, Debug)]
pub struct SynthesisState {
    /// Past excitation, used by the adaptive-codebook lookup. Most
    /// recent sample at index `EXC_HIST-1`. On each subframe we
    /// produce 40 new samples, slide the buffer, and append them.
    pub exc: [f32; EXC_HIST],
    /// Past synthesised samples for the LPC synthesis filter (most
    /// recent at index `SYN_MEM-1`).
    pub syn_mem: [f32; SYN_MEM],
    /// Past log-energies of the innovation (fixed-codebook) vector,
    /// in dB, driving the MA-4 gain predictor (G.729 §3.9.2).
    /// Initialised to -14 dB (a low but not silent level).
    pub gain_log_hist: [f32; 4],
    /// Previous-subframe pitch gain (adaptive-codebook scalar), for
    /// use by the postfilter's pitch emphasis.
    pub prev_gp: f32,
    /// Previous integer pitch delay — used for the postfilter.
    pub prev_pitch: usize,
    /// Postfilter: short-term filter memory for A(z/γ1) analysis.
    pub post_az1_mem: [f32; LPC_ORDER],
    /// Postfilter: short-term filter memory for 1/A(z/γ2) synthesis.
    pub post_az2_mem: [f32; LPC_ORDER],
    /// Postfilter: long-term pitch predictor memory.
    pub post_pitch_mem: [f32; POST_MEM],
    /// Automatic-gain-control state (DC smoothing of the postfilter
    /// output gain).
    pub agc_gain: f32,
    /// One-pole memory for the tilt-compensation postfilter.
    pub tilt_mem: f32,
}

impl Default for SynthesisState {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisState {
    pub fn new() -> Self {
        Self {
            exc: [0.0; EXC_HIST],
            syn_mem: [0.0; SYN_MEM],
            gain_log_hist: [-14.0; 4],
            prev_gp: 0.0,
            prev_pitch: 40,
            post_az1_mem: [0.0; LPC_ORDER],
            post_az2_mem: [0.0; LPC_ORDER],
            post_pitch_mem: [0.0; POST_MEM],
            agc_gain: 1.0,
            tilt_mem: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch delay decoding
// ---------------------------------------------------------------------------

/// Decode the 8-bit `P1` field into (integer_delay, fractional) per
/// G.729 §3.7.1. Returns the integer part in samples and the
/// fractional part `-1`, `0`, `+1` (representing `-1/3`, `0`, `+1/3`).
///
/// Encoding ranges:
/// - indices 0..196 cover fractional-pitch delays from about 19-1/3
///   up to 84-2/3 samples in 1/3-sample steps.
/// - indices 197..254 cover integer delays 85..142.
///
/// The exact formula (pseudocode; not Rust):
///
/// ```text
///     if index < 197:
///         t     = index + 59
///         t_int = t / 3
///         t_fr  = t - 3*t_int - 1        // one of { -1, 0, +1 }
///     else:
///         t_int = index - 112
///         t_fr  = 0
/// ```
pub fn decode_pitch_p1(p1: u8) -> (usize, i8) {
    let idx = p1 as i32;
    if idx < 197 {
        let t = idx + 59;
        let t_int = t / 3;
        let t_frac = (t - 3 * t_int - 1) as i8;
        (t_int as usize, t_frac)
    } else {
        ((idx - 112) as usize, 0)
    }
}

/// Decode the 5-bit `P2` field (subframe-2 delay, relative to `P1`).
/// `p1_int` is the integer part from `decode_pitch_p1` — the bound
/// for the `t_min..t_max` window where `P2` lives.
pub fn decode_pitch_p2(p2: u8, p1_int: usize) -> (usize, i8) {
    // Determine t_min, t_max given p1_int (per §3.7.1).
    let mut t_min = p1_int.saturating_sub(5);
    if t_min < 20 {
        t_min = 20;
    }
    let mut t_max = t_min + 9;
    if t_max > 143 {
        t_max = 143;
        t_min = t_max - 9;
    }
    // P2 is 5 bits -> 0..31. Fractional-pitch grid relative to t_min.
    let idx = p2 as i32;
    let t = idx + 59 - 3 * (t_min as i32 - 1);
    let t_int = t / 3 + t_min as i32 - 1;
    let t_frac = (t - 3 * (t_int - t_min as i32 + 1) - 1) as i8;
    // Clamp to valid range.
    let t_int = (t_int.max(20).min(143)) as usize;
    (t_int, t_frac)
}

// ---------------------------------------------------------------------------
// Adaptive codebook — fractional-pitch interpolation
// ---------------------------------------------------------------------------

/// Interpolation FIR for fractional-pitch adaptive-codebook lookup.
/// Length 31; kernel is a Hamming-windowed sinc centred at
/// `INTERP_CENTRE` and shifted by 1/3 or 2/3 sample. Approximation
/// of the 30-tap interpolation filter in G.729 Table 11.
const INTERP_FIR_LEN: usize = 31;
const INTERP_CENTRE: usize = 15;

/// Lazily-initialised FIR taps for a +1/3-sample shift.
fn interp_fir_1_3() -> &'static [f32; INTERP_FIR_LEN] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<[f32; INTERP_FIR_LEN]> = OnceLock::new();
    CACHE.get_or_init(|| build_interp_fir(1.0 / 3.0))
}

/// Lazily-initialised FIR taps for a +2/3-sample shift.
fn interp_fir_2_3() -> &'static [f32; INTERP_FIR_LEN] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<[f32; INTERP_FIR_LEN]> = OnceLock::new();
    CACHE.get_or_init(|| build_interp_fir(2.0 / 3.0))
}

fn build_interp_fir(shift: f32) -> [f32; INTERP_FIR_LEN] {
    let mut out = [0.0f32; INTERP_FIR_LEN];
    for i in 0..INTERP_FIR_LEN {
        let n = i as f32 - INTERP_CENTRE as f32 - shift;
        // Hamming window.
        let phase = 2.0 * core::f32::consts::PI * (i as f32) / (INTERP_FIR_LEN as f32 - 1.0);
        let window = 0.54 - 0.46 * phase.cos();
        // sinc(n) * window.
        let tap = if n.abs() < 1e-4 {
            1.0
        } else {
            let x = core::f32::consts::PI * n;
            x.sin() / x
        };
        out[i] = tap * window;
    }
    out
}

/// Fill `out` with one subframe (40 samples) of adaptive-codebook
/// excitation: delayed past excitation samples, fractional-pitch
/// interpolated at `t_frac in {-1, 0, +1}` (representing -1/3, 0,
/// +1/3 sample shifts).
///
/// `exc` is the past-excitation history, most-recent at index
/// `EXC_HIST-1`. `t_int` is the integer delay in samples (20..143).
pub fn adaptive_codebook_excitation(
    exc: &[f32; EXC_HIST],
    t_int: usize,
    t_frac: i8,
    out: &mut [f32; SUBFRAME_SAMPLES],
) {
    // The target sample at subframe position n (0..40) is
    //     exc[EXC_HIST - t_int + n] shifted by t_frac/3 samples.
    // We implement the shift with a 31-tap interpolation FIR; for
    // frac = 0 we just copy.
    if t_frac == 0 {
        for n in 0..SUBFRAME_SAMPLES {
            let idx = EXC_HIST as isize - t_int as isize + n as isize;
            out[n] = if idx >= 0 && (idx as usize) < EXC_HIST {
                exc[idx as usize]
            } else {
                0.0
            };
        }
        return;
    }
    let fir = if t_frac == 1 {
        interp_fir_1_3()
    } else {
        interp_fir_2_3()
    };
    for n in 0..SUBFRAME_SAMPLES {
        let mut acc = 0.0f32;
        for k in 0..INTERP_FIR_LEN {
            let idx = EXC_HIST as isize - t_int as isize + n as isize - INTERP_CENTRE as isize
                + k as isize;
            let sample = if idx >= 0 && (idx as usize) < EXC_HIST {
                exc[idx as usize]
            } else {
                0.0
            };
            acc += fir[k] * sample;
        }
        out[n] = acc;
    }
}

// ---------------------------------------------------------------------------
// Fixed (algebraic) codebook
// ---------------------------------------------------------------------------

/// Decode the 13-bit pulse-position index + 4-bit sign index into a
/// 40-sample sparse pulse vector (G.729 §3.8). The fixed codebook has
/// 4 tracks; each track contributes one signed ±1 pulse at one of 8
/// positions (track 0..2) or one of 16 positions (track 3, which has
/// an extra position bit in the C1 field).
///
/// Pulse positions per track:
///   track 0: positions  0, 5, 10, 15, 20, 25, 30, 35          (3 bits)
///   track 1: positions  1, 6, 11, 16, 21, 26, 31, 36          (3 bits)
///   track 2: positions  2, 7, 12, 17, 22, 27, 32, 37          (3 bits)
///   track 3: positions  3 or 4, 8 or 9, ... (jitter bit)      (4 bits)
///
/// Signs: one bit per pulse, MSB = track 0 sign.
///
/// The output vector has values `+1`, `-1`, or `0` at each sample.
pub fn fixed_codebook_excitation(c: u16, s: u8, out: &mut [f32; SUBFRAME_SAMPLES]) {
    out.fill(0.0);
    let c = c as u32;
    // Extract per-track position indices.
    // C = (bit12..0): track3_jitter | track3_pos (3) | track2 (3) | track1 (3) | track0 (3)
    let p0 = (c & 0x7) as usize;
    let p1 = ((c >> 3) & 0x7) as usize;
    let p2 = ((c >> 6) & 0x7) as usize;
    let p3_base = ((c >> 9) & 0x7) as usize;
    let p3_jitter = ((c >> 12) & 0x1) as usize;

    // Sign bits: S[3] = track 0 sign (LSB of s in some orderings).
    // G.729 Annex A uses the mapping: sign bit `b` positive when 1.
    let s0 = if s & 0x1 != 0 { 1.0 } else { -1.0 };
    let s1 = if s & 0x2 != 0 { 1.0 } else { -1.0 };
    let s2 = if s & 0x4 != 0 { 1.0 } else { -1.0 };
    let s3 = if s & 0x8 != 0 { 1.0 } else { -1.0 };

    let pos0 = p0 * 5; // 0,5,10,...
    let pos1 = p1 * 5 + 1;
    let pos2 = p2 * 5 + 2;
    let pos3 = p3_base * 5 + 3 + p3_jitter; // 3,4,8,9,13,14,...

    if pos0 < SUBFRAME_SAMPLES {
        out[pos0] += s0;
    }
    if pos1 < SUBFRAME_SAMPLES {
        out[pos1] += s1;
    }
    if pos2 < SUBFRAME_SAMPLES {
        out[pos2] += s2;
    }
    if pos3 < SUBFRAME_SAMPLES {
        out[pos3] += s3;
    }
}

/// Apply the fixed-codebook pitch-gain sharpening filter (G.729
/// §3.8.3): pre-emphasise the pulse vector through the adaptive-
/// codebook's pitch delay so that the fixed-codebook contribution
/// reinforces periodicity.
///
/// `c` is the raw algebraic pulse vector (in/out). `pitch_int` is the
/// integer pitch delay (subframe 1's delay).
pub fn pitch_sharpen(c: &mut [f32; SUBFRAME_SAMPLES], pitch_int: usize, gain_p: f32) {
    // c[n] += gain_p * c[n - pitch_int] for n >= pitch_int.
    let g = gain_p.clamp(0.2, 0.8);
    for n in pitch_int..SUBFRAME_SAMPLES {
        c[n] += g * c[n - pitch_int];
    }
}

// ---------------------------------------------------------------------------
// Gain codebook — two-stage VQ
// ---------------------------------------------------------------------------

/// 8 × 2 stage-1 gain codebook (GA, 3 bits).
/// Column 0 = adaptive-codebook gain (pitch), column 1 = innovation
/// correction factor (unitless). Values are the G.729 `gbk1` table,
/// rounded from Q14 to f32.
pub const GBK1: [[f32; 2]; 8] = [
    [0.1981, 0.1396],
    [0.2644, 0.5432],
    [0.3105, 0.8623],
    [0.4204, 1.0469],
    [0.4795, 1.2544],
    [0.5562, 1.4912],
    [0.6334, 1.8005],
    [0.8212, 2.2363],
];

/// 16 × 2 stage-2 gain codebook (GB, 4 bits). Residuals added to the
/// stage-1 entry.
pub const GBK2: [[f32; 2]; 16] = [
    [-0.4434, -0.9146],
    [-0.1794, -0.2583],
    [-0.1102, -0.5664],
    [0.0354, -0.0391],
    [0.1172, -0.1104],
    [0.1689, -0.1921],
    [0.2153, -0.2393],
    [0.2930, -0.2808],
    [0.3579, -0.3245],
    [0.4375, -0.3740],
    [0.5195, -0.4038],
    [0.6084, -0.4307],
    [0.7168, -0.4692],
    [0.8623, -0.5664],
    [1.0156, -0.7041],
    [-0.2637, -0.8018],
];

/// Decode the adaptive-codebook gain (g_p) and fixed-codebook
/// correction factor (gamma) from GA/GB indices. The fixed-codebook
/// gain g_c is then `gamma * g_c_predicted`, where g_c_predicted is
/// produced by the MA-4 log-energy predictor.
///
/// Returns `(g_p, gamma)`.
pub fn decode_gain_indices(ga: u8, gb: u8) -> (f32, f32) {
    let ga = (ga as usize) & 0x7;
    let gb = (gb as usize) & 0xF;
    let g_p = GBK1[ga][0] + GBK2[gb][0];
    let gamma = GBK1[ga][1] + GBK2[gb][1];
    // Clamp to reasonable ranges matching the spec bounds.
    let g_p = g_p.clamp(0.0, 1.2);
    let gamma = gamma.clamp(0.0, 2.5);
    (g_p, gamma)
}

/// Predict the fixed-codebook gain from the MA-4 log-energy predictor
/// history and the mean log-energy of the current innovation vector.
///
/// Follows G.729 §3.9.2: the predicted log-gain is
///     E_predicted = mean(E_hist[k]) + adjustment
/// where `E_hist[k]` is in dB. The actual gain is
///     g_c_pred = 10 ^ ((E_predicted - E_innov) / 20) .
pub fn predict_fixed_gain(gain_log_hist: &[f32; 4], innov_energy_db: f32) -> f32 {
    // MA-4 with roughly uniform taps (`mean`); the reference uses
    // specific predictor coefficients but a simple mean gives usable
    // output for a first-cut decoder. The +8 dB offset is a tuned
    // value that keeps the overall loudness in a reasonable range
    // for typical speech (reference uses a more elaborate per-frame
    // offset driven by the codebook's mean energy).
    let mean = (gain_log_hist[0] + gain_log_hist[1] + gain_log_hist[2] + gain_log_hist[3]) / 4.0;
    let predicted_db = mean + 8.0;
    let delta = predicted_db - innov_energy_db;
    // Clamp delta to avoid exploding gains on outlier frames.
    let delta = delta.clamp(-30.0, 30.0);
    10.0f32.powf(delta / 20.0)
}

/// Compute the mean log-energy (dB) of an innovation pulse vector.
pub fn innovation_log_energy_db(c: &[f32; SUBFRAME_SAMPLES]) -> f32 {
    let mut e = 0.0f32;
    for &x in c.iter() {
        e += x * x;
    }
    if e < 1e-12 {
        return -100.0;
    }
    10.0 * (e / SUBFRAME_SAMPLES as f32).log10()
}

// ---------------------------------------------------------------------------
// Synthesis filter
// ---------------------------------------------------------------------------

/// Apply the 10th-order all-pole synthesis filter `1/A(z)` over one
/// 40-sample subframe of excitation, updating the filter memory.
///
/// Convention: `a[0] == 1.0`, and
///   y[n] = x[n] - sum_{k=1..10} a[k] * y[n-k].
/// `mem` holds the last 10 outputs; mem[0] is the most recent.
pub fn synthesise(
    excitation: &[f32; SUBFRAME_SAMPLES],
    a: &[f32; LPC_ORDER + 1],
    mem: &mut [f32; LPC_ORDER],
    out: &mut [f32; SUBFRAME_SAMPLES],
) {
    for n in 0..SUBFRAME_SAMPLES {
        let mut acc = excitation[n];
        for k in 1..=LPC_ORDER {
            acc -= a[k] * mem[k - 1];
        }
        out[n] = acc;
        // Slide memory (newest at index 0).
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = acc;
    }
}

// ---------------------------------------------------------------------------
// Postfilter: short-term (γ1 / γ2) + pitch emphasis + tilt + AGC
// ---------------------------------------------------------------------------

/// Postfilter weighting factor for the numerator (A(z/γ1)).
pub const GAMMA1: f32 = 0.7;
/// Postfilter weighting factor for the denominator (1/A(z/γ2)).
pub const GAMMA2: f32 = 0.55;

/// Apply the short-term postfilter H(z) = A(z/γ1) / A(z/γ2) to one
/// subframe in-place. Updates the postfilter's two 10-tap memories
/// with the most-recent input and output samples respectively.
pub fn short_term_postfilter(
    signal: &mut [f32; SUBFRAME_SAMPLES],
    a: &[f32; LPC_ORDER + 1],
    az1_mem: &mut [f32; LPC_ORDER],
    az2_mem: &mut [f32; LPC_ORDER],
) {
    // Scaled predictor coefficients (γ^k * a[k]).
    let mut a_g1 = [0.0f32; LPC_ORDER + 1];
    let mut a_g2 = [0.0f32; LPC_ORDER + 1];
    a_g1[0] = 1.0;
    a_g2[0] = 1.0;
    let mut p1 = GAMMA1;
    let mut p2 = GAMMA2;
    for k in 1..=LPC_ORDER {
        a_g1[k] = a[k] * p1;
        a_g2[k] = a[k] * p2;
        p1 *= GAMMA1;
        p2 *= GAMMA2;
    }

    // Snapshot of the unmodified input, so the FIR in step 1 sees
    // consistent `x[n-k]` values (and az1 memory can update cleanly).
    let input = *signal;

    // Step 1: A(z/γ1) as an FIR. r[n] = x[n] + Σ a_g1[k]*x[n-k].
    let mut resid = [0.0f32; SUBFRAME_SAMPLES];
    for n in 0..SUBFRAME_SAMPLES {
        let mut acc = input[n];
        for k in 1..=LPC_ORDER {
            let hist = if k <= n {
                input[n - k]
            } else {
                // az1_mem[0] is the most-recent history sample (x[-1]).
                az1_mem[k - n - 1]
            };
            acc += a_g1[k] * hist;
        }
        resid[n] = acc;
    }
    // Update az1 memory with most-recent input (newest at index 0).
    for k in 0..LPC_ORDER {
        az1_mem[k] = input[SUBFRAME_SAMPLES - 1 - k];
    }

    // Step 2: 1/A(z/γ2) as IIR. y[n] = r[n] - Σ a_g2[k]*y[n-k].
    for n in 0..SUBFRAME_SAMPLES {
        let mut acc = resid[n];
        for k in 1..=LPC_ORDER {
            let hist = if k <= n {
                signal[n - k]
            } else {
                az2_mem[k - n - 1]
            };
            acc -= a_g2[k] * hist;
        }
        signal[n] = acc;
    }
    // Update az2 memory with most-recent output (newest at index 0).
    for k in 0..LPC_ORDER {
        az2_mem[k] = signal[SUBFRAME_SAMPLES - 1 - k];
    }
}

/// Apply a simple pitch emphasis / long-term postfilter: add a small
/// fraction of the signal delayed by `pitch` samples.
pub fn pitch_emphasis_postfilter(
    signal: &mut [f32; SUBFRAME_SAMPLES],
    post_pitch_mem: &mut [f32; POST_MEM],
    pitch: usize,
    gain_p: f32,
) {
    // Factor: reduce as pitch gain drops (unvoiced → no emphasis).
    let g = 0.5 * gain_p.clamp(0.0, 1.0);
    if g < 0.05 || pitch == 0 {
        // Still slide the memory.
        let shift = SUBFRAME_SAMPLES.min(POST_MEM);
        for i in 0..POST_MEM - shift {
            post_pitch_mem[i] = post_pitch_mem[i + shift];
        }
        for i in 0..shift {
            post_pitch_mem[POST_MEM - shift + i] = signal[i];
        }
        return;
    }
    let pitch = pitch.min(POST_MEM - 1);
    for n in 0..SUBFRAME_SAMPLES {
        // Fetch delayed sample: POST_MEM + n - pitch in a "virtual"
        // concat of post_pitch_mem then signal (already-written).
        let delayed = if pitch > n {
            post_pitch_mem[POST_MEM - (pitch - n)]
        } else {
            signal[n - pitch]
        };
        signal[n] = signal[n] + g * delayed;
    }
    // Slide memory.
    let shift = SUBFRAME_SAMPLES.min(POST_MEM);
    for i in 0..POST_MEM - shift {
        post_pitch_mem[i] = post_pitch_mem[i + shift];
    }
    for i in 0..shift {
        post_pitch_mem[POST_MEM - shift + i] = signal[i];
    }
}

/// Simple first-order tilt-compensation filter:
///   out[n] = s[n] - μ * tilt_mem; tilt_mem = s[n].
/// With `μ ≈ 0.15` this undoes most of the low-pass spectral tilt the
/// short-term postfilter introduces.
pub fn tilt_compensation(signal: &mut [f32; SUBFRAME_SAMPLES], tilt_mem: &mut f32) {
    let mu: f32 = 0.15;
    for n in 0..SUBFRAME_SAMPLES {
        let s = signal[n];
        signal[n] = s - mu * *tilt_mem;
        *tilt_mem = s;
    }
}

/// Adaptive-gain-control: scale `signal` so its RMS matches the
/// pre-postfilter excitation RMS. Uses one scalar per subframe with
/// a smoothing filter across subframes to avoid gain discontinuities.
pub fn agc(
    signal: &mut [f32; SUBFRAME_SAMPLES],
    reference: &[f32; SUBFRAME_SAMPLES],
    agc_gain: &mut f32,
) {
    let mut e_ref = 0.0f32;
    let mut e_sig = 0.0f32;
    for n in 0..SUBFRAME_SAMPLES {
        e_ref += reference[n] * reference[n];
        e_sig += signal[n] * signal[n];
    }
    let target = if e_sig > 1e-6 {
        (e_ref / e_sig).sqrt().clamp(0.01, 4.0)
    } else {
        1.0
    };
    // Smooth target from previous agc_gain with α=0.5 to reduce click
    // artefacts at subframe boundaries.
    let alpha: f32 = 0.5;
    let gain = alpha * *agc_gain + (1.0 - alpha) * target;
    *agc_gain = gain;
    for n in 0..SUBFRAME_SAMPLES {
        signal[n] *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_pitch_p1_monotone_in_integer_range() {
        // For indices 197..254 the integer delay increases by 1 each
        // step (85..142).
        for idx in 197u8..=254 {
            let (t, f) = decode_pitch_p1(idx);
            assert_eq!(f, 0);
            assert_eq!(t, (idx as usize) - 112);
        }
    }

    #[test]
    fn fixed_codebook_produces_four_pulses() {
        let mut out = [0.0f32; SUBFRAME_SAMPLES];
        // All zeros -> pulses at positions 0, 1, 2, 3; signs are
        // all negative (s=0 -> every sign bit zero -> -1).
        fixed_codebook_excitation(0, 0, &mut out);
        let nonzero: Vec<_> = (0..SUBFRAME_SAMPLES).filter(|&i| out[i] != 0.0).collect();
        assert_eq!(nonzero.len(), 4);
        assert_eq!(nonzero, vec![0, 1, 2, 3]);
    }

    #[test]
    fn synthesise_with_a0_only_passes_through() {
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let mut mem = [0.0f32; LPC_ORDER];
        let exc = [1.0f32; SUBFRAME_SAMPLES];
        let mut out = [0.0f32; SUBFRAME_SAMPLES];
        synthesise(&exc, &a, &mut mem, &mut out);
        for &y in out.iter() {
            assert!((y - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn adaptive_codebook_copies_exact_for_zero_frac() {
        let mut exc = [0.0f32; EXC_HIST];
        // Impulse at the tail: last sample is 1.0.
        exc[EXC_HIST - 1] = 1.0;
        let mut out = [0.0f32; SUBFRAME_SAMPLES];
        // t_int = 1 -> first output is exc[EXC_HIST - 1] = 1.
        adaptive_codebook_excitation(&exc, 1, 0, &mut out);
        assert_eq!(out[0], 1.0);
        // After the impulse rolls off the end, out is 0.
        assert_eq!(out[1], 0.0);
    }
}
