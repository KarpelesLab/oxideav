//! G.728 LD-CELP decoder core state + first-cut synthesis loop.
//!
//! This module implements a first-cut G.728 decoder:
//!
//! 1. Bit-unpack: 10-bit codebook indices from the packed stream.
//! 2. Excitation: 128-entry shape codebook × 8-entry gain codebook
//!    (sign comes from the extra bit in the index).
//! 3. Backward-adaptive LPC (50th order): autocorrelation +
//!    Levinson-Durbin over the recent synthesis history, refreshed every
//!    4 vectors (2.5 ms).
//! 4. Synthesis: all-pole IIR filter fed by the per-vector excitation.
//! 5. Backward-adaptive log-gain prediction (10th order): tracks the
//!    excitation energy trajectory in the log domain.
//!
//! Deliberate deviations from the ITU-T reference (called out so future
//! work can close the gap):
//!
//! - The shape / gain codebooks in [`crate::tables`] are deterministic
//!   unit-RMS placeholders rather than the exact Annex A `CODEBK` / `GB`
//!   tables. Structure is right; numbers differ.
//! - Autocorrelation uses a fixed 100-sample Hamming window instead of
//!   the spec's recursive Barnwell (logarithmic) window.
//! - Bandwidth expansion is applied post-recursion (γ = 0.96 for LPC,
//!   0.90 for gain predictor) to guarantee filter stability even when
//!   the autocorrelation estimate is rank-deficient.
//! - Postfilter (adaptive long-term pitch + short-term spectral tilt
//!   compensation, §5.5 of the 2012 edition) is not implemented.
//!
//! Consequence: output is structured (non-silent, bounded, stable) but
//! does **not** bit-match the ITU reference decoder. Treat this as a
//! functional scaffold rather than a spec-compliant decoder.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::{BitReader, UnpackedIndex};
use crate::predictor::{
    update_gain_predictor, update_lpc_from_history, GAIN_HISTORY_LEN, HISTORY_LEN,
};
use crate::tables::{GAIN_CB, SHAPE_CB};
use crate::{CODEC_ID_STR, GAIN_ORDER, INDEX_BITS, LPC_ORDER, SAMPLE_RATE, VECTOR_SIZE};

/// Number of vectors between backward-adaptation refreshes (G.728 §3.7:
/// LPC re-estimation every 4 vectors = 20 samples = 2.5 ms).
pub const VECTORS_PER_BLOCK: u32 = 4;

// ---------------------------------------------------------------------------
// Core decoder state
// ---------------------------------------------------------------------------

/// Backward-adaptive 50th-order LPC synthesis filter state.
///
/// `a[0] ≡ 1.0` and `a[1..=LPC_ORDER]` are the AR predictor taps. The
/// synthesis filter realises `1 / A(z)` as:
///
/// ```text
///   y[n] = x[n] - sum_{k=1..=50} a[k] * y[n-k]
/// ```
///
/// The tap vector is refreshed every `VECTORS_PER_BLOCK` vectors by
/// running `update_lpc_from_history` over `synth_history`.
pub struct LpcPredictor {
    /// Current LPC synthesis coefficients `a[1..=50]` (a[0] ≡ 1.0).
    pub a: [f32; LPC_ORDER + 1],
    /// Delay line of past synthesised samples (most recent at index 0).
    pub history: [f32; LPC_ORDER],
    /// Longer history used to re-estimate `a` via autocorrelation +
    /// Levinson-Durbin. Most recent sample at index 0.
    pub synth_history: [f32; HISTORY_LEN],
    /// Number of vectors processed since the last coefficient update.
    pub vectors_since_update: u32,
}

impl Default for LpcPredictor {
    fn default() -> Self {
        let mut a = [0.0_f32; LPC_ORDER + 1];
        a[0] = 1.0;
        Self {
            a,
            history: [0.0; LPC_ORDER],
            synth_history: [0.0; HISTORY_LEN],
            vectors_since_update: 0,
        }
    }
}

impl LpcPredictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the all-pole synthesis filter to one 5-sample excitation
    /// vector, producing 5 reconstructed speech samples. Both
    /// `history` (short delay line) and `synth_history` (the
    /// autocorrelation-analysis window) are advanced by the output.
    pub fn synthesise(&mut self, excitation: &[f32; VECTOR_SIZE], out: &mut [f32; VECTOR_SIZE]) {
        for n in 0..VECTOR_SIZE {
            let mut acc = excitation[n];
            for k in 1..=LPC_ORDER {
                acc -= self.a[k] * self.history[k - 1];
            }
            // Hard clip to prevent runaway if the filter is briefly
            // ill-conditioned (can only happen if update_lpc_from_history
            // produces a marginally-stable filter and is hit with large
            // excitation).
            let y = acc.clamp(-1.0e4, 1.0e4);
            out[n] = y;
            // Shift short history: newest sample at index 0.
            for k in (1..LPC_ORDER).rev() {
                self.history[k] = self.history[k - 1];
            }
            self.history[0] = y;
            // Shift long (autocorrelation) history.
            for k in (1..HISTORY_LEN).rev() {
                self.synth_history[k] = self.synth_history[k - 1];
            }
            self.synth_history[0] = y;
        }
        self.vectors_since_update = self.vectors_since_update.wrapping_add(1);
    }

    /// Re-estimate `a` from `synth_history` using the Levinson-Durbin
    /// recursion + bandwidth expansion. Returns `true` if the update
    /// succeeded; the filter is left unchanged on failure.
    pub fn refresh_coefficients(&mut self) -> bool {
        let ok = update_lpc_from_history(&mut self.a, &self.synth_history);
        self.vectors_since_update = 0;
        ok
    }
}

/// Backward-adaptive 10th-order log-gain predictor (§3.9 of G.728).
///
/// Predicts the log-domain excitation gain from the 10 most recent
/// log-gains. Like the LPC predictor it is updated every
/// `VECTORS_PER_BLOCK` vectors, but from the gain trajectory rather
/// than the synthesis signal.
pub struct GainPredictor {
    /// Prediction coefficients `b[1..=GAIN_ORDER]` (b[0] ≡ 1.0).
    pub b: [f32; GAIN_ORDER + 1],
    /// Short delay line for prediction (newest at index 0).
    pub history: [f32; GAIN_ORDER],
    /// Longer history window for the Levinson-Durbin update.
    pub analysis_history: [f32; GAIN_HISTORY_LEN],
    /// Most recently predicted log gain (linear dB-ish units).
    pub last_log_gain: f32,
    /// Vectors since the last coefficient update.
    pub vectors_since_update: u32,
}

impl Default for GainPredictor {
    fn default() -> Self {
        let mut b = [0.0_f32; GAIN_ORDER + 1];
        b[0] = 1.0;
        Self {
            b,
            history: [0.0; GAIN_ORDER],
            analysis_history: [0.0; GAIN_HISTORY_LEN],
            last_log_gain: 0.0,
            vectors_since_update: 0,
        }
    }
}

impl GainPredictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce the next predicted log-gain (log base e). The actual
    /// excitation gain is recovered as `exp(log_gain)`.
    ///
    /// The spec's predictor is an AR model of the *mean-removed* log
    /// gain — that way a slowly-varying signal (near-DC log-gain) is
    /// tracked by the mean term and the AR coefficients only need to
    /// model deviations. We mirror that here: subtract the running
    /// mean from the history, apply `b` to the residual, add the mean
    /// back. Without this split the `b` vector from Levinson-Durbin
    /// on a near-constant history degenerates (reflection coefficient
    /// hits ±1 and the recursion bails out, leaving `b[1..]` at zero),
    /// which would pin the predicted log gain to zero forever.
    pub fn predict(&mut self) -> f32 {
        let mean = {
            let mut s = 0.0_f32;
            for k in 0..GAIN_ORDER {
                s += self.history[k];
            }
            s / GAIN_ORDER as f32
        };
        let mut acc = 0.0_f32;
        for k in 1..=GAIN_ORDER {
            acc -= self.b[k] * (self.history[k - 1] - mean);
        }
        let predicted = mean + acc;
        // Clamp against wild predictions (e.g., ±20 ≈ 9 dec e-folds).
        self.last_log_gain = predicted.clamp(-6.0, 6.0);
        self.last_log_gain
    }

    /// Slide histories, inserting the latest observed log-gain.
    pub fn push(&mut self, log_gain: f32) {
        let g = log_gain.clamp(-6.0, 6.0);
        for k in (1..GAIN_ORDER).rev() {
            self.history[k] = self.history[k - 1];
        }
        self.history[0] = g;
        for k in (1..GAIN_HISTORY_LEN).rev() {
            self.analysis_history[k] = self.analysis_history[k - 1];
        }
        self.analysis_history[0] = g;
        self.vectors_since_update = self.vectors_since_update.wrapping_add(1);
    }

    /// Re-estimate `b` from `analysis_history`.
    pub fn refresh_coefficients(&mut self) -> bool {
        let ok = update_gain_predictor(&mut self.b, &self.analysis_history);
        self.vectors_since_update = 0;
        ok
    }
}

/// Aggregate decoder state — LPC + gain predictor + vector counters.
pub struct G728State {
    pub lpc: LpcPredictor,
    pub gain: GainPredictor,
    /// Counter of decoded vectors since the start of the stream.
    pub vector_count: u64,
}

impl Default for G728State {
    fn default() -> Self {
        Self {
            lpc: LpcPredictor::new(),
            gain: GainPredictor::new(),
            vector_count: 0,
        }
    }
}

impl G728State {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode one raw 10-bit index into an excitation vector.
    ///
    /// This looks up the shape codebook row indexed by the top 7 bits,
    /// applies the sign bit, and scales by the 3-bit gain codebook entry
    /// combined with the current backward-predicted log-gain. The
    /// returned vector is *pre*-synthesis; feed it through
    /// `LpcPredictor::synthesise` to obtain PCM.
    pub fn excitation_from_index(&self, raw: u16) -> [f32; VECTOR_SIZE] {
        let idx = UnpackedIndex::from_raw(raw);
        let shape = &SHAPE_CB[idx.shape_index as usize];
        let mag = GAIN_CB[idx.gain_mag as usize];
        let sign: f32 = if idx.sign != 0 { -1.0 } else { 1.0 };
        // Backward-adaptive gain: exp(last_log_gain) scales the codebook
        // magnitude. last_log_gain is clamped to ±6 nats ≈ ±52 dB.
        let adaptive = self.gain.last_log_gain.exp();
        let scale = sign * mag * adaptive;
        [
            shape[0] * scale,
            shape[1] * scale,
            shape[2] * scale,
            shape[3] * scale,
            shape[4] * scale,
        ]
    }

    /// Decode a single 10-bit index into 5 PCM f32 samples and advance
    /// all state (LPC history, gain predictor, adaptation counters).
    ///
    /// Per §3.9 of the spec, the predicted log gain is refreshed **once
    /// per vector** (using the 10th-order gain predictor's current `b`
    /// coefficients over the running log-gain history) *before* the
    /// excitation is scaled. The coefficients themselves are re-estimated
    /// every 4 vectors via Levinson-Durbin. This split — fast per-vector
    /// prediction + slow coefficient adaptation — is what lets the log-
    /// gain track the signal envelope in real time without transmitting
    /// any side information.
    pub fn decode_vector(&mut self, raw: u16, out: &mut [f32; VECTOR_SIZE]) {
        // Per-vector log-gain prediction. This updates `last_log_gain`,
        // which `excitation_from_index` reads for the adaptive scale.
        self.gain.predict();

        let excitation = self.excitation_from_index(raw);

        // Synthesise: run excitation through 1 / A(z).
        self.lpc.synthesise(&excitation, out);

        // Update log-gain history from the magnitude of this excitation.
        // We use the log of the excitation RMS so the predictor tracks
        // the envelope in the log domain.
        let mut ss = 0.0_f32;
        for n in 0..VECTOR_SIZE {
            ss += excitation[n] * excitation[n];
        }
        let rms = (ss / VECTOR_SIZE as f32).sqrt();
        // ln(max(rms, eps))
        let log_g = rms.max(1.0e-6).ln();
        self.gain.push(log_g);

        self.vector_count = self.vector_count.wrapping_add(1);

        // Backward-adaptive coefficient refresh every 4 vectors.
        if self.lpc.vectors_since_update >= VECTORS_PER_BLOCK {
            self.lpc.refresh_coefficients();
        }
        if self.gain.vectors_since_update >= VECTORS_PER_BLOCK {
            self.gain.refresh_coefficients();
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder trait wiring
// ---------------------------------------------------------------------------

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "G.728 decoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.728 decoder: only mono is supported (got {channels} channels)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "G.728 decoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }
    Ok(Box::new(G728Decoder::new()))
}

struct G728Decoder {
    codec_id: CodecId,
    state: G728State,
    pending: Option<Packet>,
    eof: bool,
    time_base: TimeBase,
}

impl G728Decoder {
    fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            state: G728State::new(),
            pending: None,
            eof: false,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        }
    }

    /// Decode a packet's worth of 10-bit indices into an f32 PCM buffer.
    fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        // Each index is 10 bits = 1.25 bytes. The packet length must be
        // enough to hold at least one index. We accept any byte length
        // that corresponds to a whole number of 5-sample vectors (i.e.
        // the bit count is a multiple of 10), or 10-bit-rounded-up-to-
        // bytes framing.
        let total_bits = (data.len() as u64) * 8;
        let vectors = total_bits / (INDEX_BITS as u64);
        if vectors == 0 {
            return Err(Error::invalid(format!(
                "G.728: packet too short ({} bytes; need at least 2 bytes for 1 index)",
                data.len()
            )));
        }
        let mut br = BitReader::new(data);
        let mut pcm = Vec::with_capacity((vectors as usize) * VECTOR_SIZE);
        let mut vec_out = [0.0_f32; VECTOR_SIZE];
        for _ in 0..vectors {
            let raw = br.read_index10()?;
            self.state.decode_vector(raw, &mut vec_out);
            pcm.extend_from_slice(&vec_out);
        }
        Ok(pcm)
    }
}

impl Decoder for G728Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "G.728 decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let samples = self.decode_packet(&pkt.data)?;
        // Convert f32 -> S16 LE.
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in &samples {
            let v = s.round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: samples.len() as u32,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![bytes],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lpc_synthesise_zero_excitation_stays_silent() {
        let mut lpc = LpcPredictor::new();
        let exc = [0.0_f32; VECTOR_SIZE];
        let mut out = [0.0_f32; VECTOR_SIZE];
        lpc.synthesise(&exc, &mut out);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn lpc_with_unit_a0_passes_impulse_through() {
        // With a[1..] == 0 the filter is just y[n] = x[n], so an impulse
        // excitation comes out unchanged.
        let mut lpc = LpcPredictor::new();
        let exc = [1.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0; 5];
        lpc.synthesise(&exc, &mut out);
        assert_eq!(out, exc);
        // History is newest-first; after emitting [1, 0, 0, 0, 0] the impulse
        // is 4 slots deep.
        assert_eq!(lpc.history[0], 0.0);
        assert_eq!(lpc.history[4], 1.0);
    }

    #[test]
    fn gain_predictor_defaults_to_zero() {
        let mut gp = GainPredictor::new();
        assert_eq!(gp.predict(), 0.0);
    }

    #[test]
    fn excitation_from_index_is_nonzero_for_live_tables() {
        // Any non-zero gain index should produce output.
        // Layout: shape(7) | sign(1) | mag(2). The 2-bit mag field selects
        // GAIN_CB[0..=3] directly; the sign bit flips polarity.
        let st = G728State::new();
        let idx: u16 = (5 << 3) | (0 << 2) | 2; // shape=5, sign=+, mag=2 → GAIN_CB[2]
        let v = st.excitation_from_index(idx);
        // Gain predictor starts at 0 ⇒ exp(0) = 1; GAIN_CB[2] ≠ 0.
        assert!(v.iter().any(|&s| s.abs() > 0.0));
    }

    #[test]
    fn state_defaults_are_neutral() {
        let st = G728State::new();
        assert_eq!(st.vector_count, 0);
        assert_eq!(st.lpc.a[0], 1.0);
        assert_eq!(st.gain.b[0], 1.0);
    }

    #[test]
    fn make_decoder_returns_working_decoder() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        assert!(make_decoder(&params).is_ok());
    }

    #[test]
    fn make_decoder_rejects_wrong_sample_rate() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(16_000);
        assert!(make_decoder(&params).is_err());
    }

    #[test]
    fn decode_vector_produces_bounded_output() {
        let mut st = G728State::new();
        let mut out = [0.0_f32; VECTOR_SIZE];
        // Drive a fixed non-trivial index through 64 vectors.
        let raw: u16 = 0b0011010_0_10; // shape=26, sign=+, mag=2
        for _ in 0..64 {
            st.decode_vector(raw, &mut out);
            for &s in &out {
                assert!(s.is_finite(), "synthesis went non-finite: {s}");
                // LpcPredictor clamps to ±1e4 internally; the sample must
                // sit within that range plus a small i16-scale margin.
                assert!(s.abs() <= 1.0e4 + 1.0, "synthesis exploded: {s}");
            }
        }
    }

    #[test]
    fn zero_excitation_keeps_filter_stable() {
        // Feeding all-zero indices for a while must not cause growth.
        let mut st = G728State::new();
        let mut out = [0.0_f32; VECTOR_SIZE];
        // gain_mag = 0 ⇒ 0.125 * sign; set sign=0 but also shape=0.
        // Shape[0] has non-zero content (random placeholder), so to get
        // truly silent input we just feed the same code and check the
        // filter doesn't diverge — the gain predictor will adapt toward
        // the excitation envelope.
        let raw: u16 = 0; // shape=0, sign=0, mag=0
        let mut max_abs = 0.0_f32;
        for _ in 0..200 {
            st.decode_vector(raw, &mut out);
            for &s in &out {
                assert!(s.is_finite());
                if s.abs() > max_abs {
                    max_abs = s.abs();
                }
            }
        }
        // Output should stay bounded — much less than i16 full-scale.
        assert!(
            max_abs < 1.0e4,
            "constant-excitation output grew to {max_abs}"
        );
    }
}
