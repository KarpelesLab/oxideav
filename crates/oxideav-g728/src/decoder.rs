//! G.728 LD-CELP decoder core state + scaffolded entry point.
//!
//! This module establishes the data structures the full decoder will
//! operate on — the 50th-order LPC synthesis filter state, the 10th-order
//! log-gain predictor, and the fixed shape + gain codebook tables from
//! the ITU-T G.728 spec (2012 edition, Annex A reference code). The
//! synthesis loop itself (shape × gain excitation → LPC filter →
//! postfilter) is a follow-up; `make_decoder` currently returns
//! `Error::Unsupported`.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::bitreader::{BitReader, UnpackedIndex};

// ---------------------------------------------------------------------------
// Core decoder state
// ---------------------------------------------------------------------------

/// Backward-adaptive 50th-order LPC synthesis filter state.
///
/// The LPC coefficients are re-estimated every 4 vectors (every 20 samples)
/// from the decoder's own reconstructed speech via a windowed auto-
/// correlation + Levinson-Durbin recursion (G.728 Annex A, §3.7). Between
/// updates the coefficients are held constant and applied to the excitation
/// as an all-pole IIR filter.
pub struct LpcPredictor {
    /// Current LPC synthesis coefficients `a[1..=50]` (a[0] ≡ 1.0).
    pub a: [f32; super::LPC_ORDER + 1],
    /// Delay line of past synthesised samples (most recent at index 0).
    pub history: [f32; super::LPC_ORDER],
    /// Number of vectors processed since the last coefficient update.
    pub vectors_since_update: u32,
}

impl Default for LpcPredictor {
    fn default() -> Self {
        let mut a = [0.0f32; super::LPC_ORDER + 1];
        a[0] = 1.0;
        Self {
            a,
            history: [0.0; super::LPC_ORDER],
            vectors_since_update: 0,
        }
    }
}

impl LpcPredictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the all-pole synthesis filter to one 5-sample excitation vector,
    /// producing 5 reconstructed speech samples. The history is advanced by
    /// the output (not the input) so the next call sees the correct state.
    pub fn synthesise(
        &mut self,
        excitation: &[f32; super::VECTOR_SIZE],
        out: &mut [f32; super::VECTOR_SIZE],
    ) {
        for n in 0..super::VECTOR_SIZE {
            let mut acc = excitation[n];
            for k in 1..=super::LPC_ORDER {
                acc -= self.a[k] * self.history[k - 1];
            }
            out[n] = acc;
            // Shift history: newest sample at index 0.
            for k in (1..super::LPC_ORDER).rev() {
                self.history[k] = self.history[k - 1];
            }
            self.history[0] = acc;
        }
        self.vectors_since_update = self.vectors_since_update.wrapping_add(1);
    }
}

/// Backward-adaptive 10th-order log-gain predictor (§3.9 of G.728).
///
/// Predicts the log-domain excitation gain from the 10 most recent log-gains
/// of the quantised excitation. As with the LPC predictor the coefficients
/// are re-estimated every 4 vectors, but from the gain trajectory alone.
pub struct GainPredictor {
    /// Prediction coefficients (backward-adapted).
    pub b: [f32; super::GAIN_ORDER + 1],
    /// History of log-domain excitation gains (dB scale).
    pub log_gain_history: [f32; super::GAIN_ORDER],
    /// Most recently predicted log gain (dB).
    pub last_log_gain: f32,
}

impl Default for GainPredictor {
    fn default() -> Self {
        let mut b = [0.0f32; super::GAIN_ORDER + 1];
        b[0] = 1.0;
        Self {
            b,
            log_gain_history: [0.0; super::GAIN_ORDER],
            last_log_gain: 0.0,
        }
    }
}

impl GainPredictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce the next predicted log-gain (dB). The actual excitation
    /// gain is recovered as `10^(log_gain / 20)` once the predictor is
    /// fully wired in.
    pub fn predict(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for k in 1..=super::GAIN_ORDER {
            acc -= self.b[k] * self.log_gain_history[k - 1];
        }
        self.last_log_gain = acc;
        acc
    }

    /// Slide the log-gain history, inserting the latest quantised value.
    pub fn push(&mut self, log_gain_db: f32) {
        for k in (1..super::GAIN_ORDER).rev() {
            self.log_gain_history[k] = self.log_gain_history[k - 1];
        }
        self.log_gain_history[0] = log_gain_db;
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

    /// Decode one raw 10-bit index into an excitation vector (without
    /// yet running the LPC filter). The returned vector is the shape
    /// entry scaled by the signed gain magnitude; full synthesis will
    /// feed this through `LpcPredictor::synthesise`.
    pub fn excitation_from_index(&self, raw: u16) -> [f32; super::VECTOR_SIZE] {
        let idx = UnpackedIndex::from_raw(raw);
        let shape = &SHAPE_CB[idx.shape_index as usize];
        let mag = GAIN_CB[idx.gain_mag as usize];
        let sign: f32 = if idx.sign != 0 { -1.0 } else { 1.0 };
        let scale = sign * mag;
        [
            shape[0] * scale,
            shape[1] * scale,
            shape[2] * scale,
            shape[3] * scale,
            shape[4] * scale,
        ]
    }
}

// ---------------------------------------------------------------------------
// Decoder trait wiring
// ---------------------------------------------------------------------------

pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "G.728 decoder is a scaffold — codebook tables + state wired; \
         LPC re-estimation, gain adaptation, and postfilter pending",
    ))
}

/// Placeholder decoder type kept for future use; unused until the full
/// synthesis path lands. Held in the module so the `Decoder` impl can
/// grow incrementally without restructuring the public surface.
struct G728Decoder {
    codec_id: CodecId,
    state: G728State,
    pending: Option<Packet>,
    eof: bool,
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
        // The full synthesis loop will read `self.pending`, decode every
        // 10-bit index through `BitReader::read_index10`, drive the
        // excitation and LPC paths, then emit an `AudioFrame` of S16 PCM.
        // For now the scaffold surfaces `NeedMore` to signal that no
        // output is ever produced.
        let _ = BitReader::new(&[]);
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Codebook constants
// ---------------------------------------------------------------------------
//
// The tables below are the *shape* of the G.728 codebooks, not the ITU
// reference values. Each shape vector is 5 samples long (`VECTOR_SIZE`)
// and the gain codebook has 8 positive magnitudes (`GAIN_CB_SIZE`);
// the sign comes from the extra bit in the 10-bit index. Values are
// placeholders intended only so the scaffold compiles, type-checks, and
// exercises the excitation-formation code path in unit tests. Swapping
// in the spec values is a one-table change once the synthesis loop
// lands; the dimensions here match the spec exactly.

/// 128-entry × 5-sample shape codebook placeholder. All zeros — when the
/// full decoder path lands this is replaced with the Annex A "CODEBK"
/// floating-point table (normalised to unit RMS).
pub const SHAPE_CB: [[f32; super::VECTOR_SIZE]; super::SHAPE_CB_SIZE] =
    [[0.0; super::VECTOR_SIZE]; super::SHAPE_CB_SIZE];

/// 8-entry gain magnitude codebook placeholder. Replaced by Annex A
/// "GB" / "GQ" values when the synthesis path lands. Values are in the
/// linear (not log) domain; the accompanying sign bit in the 10-bit
/// index flips the polarity.
pub const GAIN_CB: [f32; super::GAIN_CB_SIZE] = [0.0; super::GAIN_CB_SIZE];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lpc_synthesise_zero_excitation_stays_silent() {
        let mut lpc = LpcPredictor::new();
        let exc = [0.0f32; super::super::VECTOR_SIZE];
        let mut out = [0.0f32; super::super::VECTOR_SIZE];
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
    fn excitation_from_index_returns_zero_for_placeholder_tables() {
        // Placeholder tables are all-zero, so any index yields silence.
        let st = G728State::new();
        let v = st.excitation_from_index(0x3FF);
        assert!(v.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn state_defaults_are_neutral() {
        let st = G728State::new();
        assert_eq!(st.vector_count, 0);
        assert_eq!(st.lpc.a[0], 1.0);
        assert_eq!(st.gain.b[0], 1.0);
    }

    #[test]
    fn make_decoder_is_unsupported() {
        let params = CodecParameters::audio(CodecId::new(super::super::CODEC_ID_STR));
        assert!(make_decoder(&params).is_err());
    }
}
