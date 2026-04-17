//! ITU-T G.728 LD-CELP encoder — analysis-by-synthesis pipeline symmetric
//! with the in-tree decoder.
//!
//! # Pipeline
//!
//! For every 5-sample input vector (0.625 ms of 8 kHz S16 PCM):
//!
//! 1. Compute the "zero-input response" (ZIR) of the current synthesis
//!    filter — the 5-sample output the decoder would produce with zero
//!    excitation, i.e. the memory tail of `1 / A(z)`.
//! 2. For every shape codeword `s` in [`SHAPE_CB`], compute the "zero-
//!    state response" (ZSR) — the 5-sample output of `1 / A(z)` fed the
//!    unit-RMS shape `s` with zero filter memory.
//! 3. Because the synthesis filter is linear, any candidate reconstruction
//!    is `ZIR + scale * ZSR_s` where `scale = sign * GAIN_CB[mag] *
//!    exp(last_log_gain)`. We search all 128 × 8 (shape, sign·mag)
//!    combinations for the one that minimises the L2 distance from the
//!    target vector (the input PCM).
//! 4. Pack the winning indices into a 10-bit `shape(7) | sign(1) | mag(2)`
//!    field. Four consecutive vectors pack into a 40-bit / 5-byte packet.
//! 5. Update the LPC synthesis history, log-gain history, and backward-
//!    adaptation counters using the **chosen** excitation — identical to
//!    the decoder — so the encoder and decoder state stay bit-identical
//!    vector-for-vector.
//!
//! # Symmetry with the decoder
//!
//! The encoder does not own any new tables: every coefficient it uses
//! lives in [`crate::predictor`] or [`crate::tables`], shared verbatim
//! with the decoder. That includes the 50th-order backward-adaptive LPC
//! (autocorrelation + Levinson-Durbin with bandwidth expansion), the
//! 10th-order log-gain predictor, the 128-entry shape codebook, and the
//! 8-entry gain magnitude codebook. A bitstream produced here decodes
//! correctly through [`crate::decoder::G728Decoder`] by construction.
//!
//! # Caveats
//!
//! The shape codebook in [`crate::tables::SHAPE_CB`] is a deterministic
//! unit-RMS placeholder — it is *not* the ITU Annex A `CODEBK` table.
//! Analysis-by-synthesis will lock onto whichever codeword happens to
//! best match the target, which is enough to demonstrate the pipeline
//! but will not produce spec-grade reconstructions until the real tables
//! are swapped in. The loop itself does not change when that happens.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::decoder::{G728State, VECTORS_PER_BLOCK};
use crate::tables::{GAIN_CB, SHAPE_CB};
use crate::{CODEC_ID_STR, GAIN_CB_SIZE, LPC_ORDER, SAMPLE_RATE, SHAPE_CB_SIZE, VECTOR_SIZE};

/// Four 10-bit indices packed MSB-first into one packet.
pub const VECTORS_PER_PACKET: usize = 4;
/// Packet duration in samples: 4 × 5 = 20 samples = 2.5 ms.
pub const PACKET_SAMPLES: usize = VECTORS_PER_PACKET * VECTOR_SIZE;
/// Packet size in bytes: 4 × 10 bits = 40 bits = 5 bytes.
pub const PACKET_BYTES: usize = 5;

/// Build a G.728 encoder. Accepts 8 kHz mono S16 input.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "G.728 encoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.728 encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "G.728 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "G.728 encoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE);
    output.bit_rate = Some(16_000);

    Ok(Box::new(G728Encoder::new(output)))
}

struct G728Encoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    /// Shared decoder state: LPC synthesis filter + log-gain predictor.
    /// Kept in lockstep with the decoder so every pushed excitation
    /// mirrors what the decoder would insert into its own history.
    state: G728State,
    /// Buffered incoming PCM (in samples). Flushed into encode in
    /// 5-sample chunks.
    pcm_queue: Vec<i16>,
    /// 10-bit codeword indices awaiting packetisation. Each entry is one
    /// encoded vector; every 4 are bit-packed into a 5-byte packet.
    index_queue: Vec<u16>,
    pending: VecDeque<Packet>,
    /// Running sample position used to stamp PTS on emitted packets.
    sample_pos: i64,
    eof: bool,
}

impl G728Encoder {
    fn new(output_params: CodecParameters) -> Self {
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
            state: G728State::new(),
            pcm_queue: Vec::new(),
            index_queue: Vec::with_capacity(VECTORS_PER_PACKET),
            pending: VecDeque::new(),
            sample_pos: 0,
            eof: false,
        }
    }

    /// Encode as many whole 5-sample vectors from `pcm_queue` as possible,
    /// emitting whole 4-vector packets as they fill up.
    fn drain(&mut self, final_flush: bool) {
        // Process complete 5-sample vectors.
        while self.pcm_queue.len() >= VECTOR_SIZE {
            let mut target = [0.0_f32; VECTOR_SIZE];
            for (i, v) in self.pcm_queue[..VECTOR_SIZE].iter().enumerate() {
                target[i] = *v as f32;
            }
            self.pcm_queue.drain(..VECTOR_SIZE);
            let idx = encode_vector(&mut self.state, &target);
            self.index_queue.push(idx);
            // If we have a full 4-vector packet, ship it.
            if self.index_queue.len() >= VECTORS_PER_PACKET {
                let start_sample = self.sample_pos;
                let bytes = pack_four_indices(&self.index_queue[..VECTORS_PER_PACKET]);
                self.index_queue.drain(..VECTORS_PER_PACKET);
                self.sample_pos += PACKET_SAMPLES as i64;
                let mut pkt = Packet::new(0, self.time_base, bytes);
                pkt.pts = Some(start_sample);
                pkt.dts = pkt.pts;
                pkt.duration = Some(PACKET_SAMPLES as i64);
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
            }
        }
        // On final flush, zero-pad the trailing partial vector / partial
        // packet so we never lose data. G.728's packet framing is
        // 4-vector aligned; we pad with zero PCM (encoder sees silence),
        // which the backward-adaptive predictors tolerate gracefully.
        if final_flush {
            if !self.pcm_queue.is_empty() {
                while self.pcm_queue.len() < VECTOR_SIZE {
                    self.pcm_queue.push(0);
                }
                // Recurse one more time (single pass; no risk of loop).
                self.drain(false);
            }
            while !self.index_queue.is_empty() {
                // Pad to 4 vectors with encoded "silent" indices — but the
                // natural thing is to encode real silence vectors so the
                // predictor stays consistent. Push zero target vectors
                // until we hit the boundary.
                let mut target = [0.0_f32; VECTOR_SIZE];
                for v in target.iter_mut() {
                    *v = 0.0;
                }
                let idx = encode_vector(&mut self.state, &target);
                self.index_queue.push(idx);
                if self.index_queue.len() >= VECTORS_PER_PACKET {
                    let start_sample = self.sample_pos;
                    let bytes = pack_four_indices(&self.index_queue[..VECTORS_PER_PACKET]);
                    self.index_queue.drain(..VECTORS_PER_PACKET);
                    self.sample_pos += PACKET_SAMPLES as i64;
                    let mut pkt = Packet::new(0, self.time_base, bytes);
                    pkt.pts = Some(start_sample);
                    pkt.dts = pkt.pts;
                    pkt.duration = Some(PACKET_SAMPLES as i64);
                    pkt.flags.keyframe = true;
                    self.pending.push_back(pkt);
                }
            }
        }
    }
}

impl Encoder for G728Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let af = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::invalid("G.728 encoder: audio frames only")),
        };
        if af.channels != 1 || af.sample_rate != SAMPLE_RATE {
            return Err(Error::invalid("G.728 encoder: input must be mono, 8000 Hz"));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "G.728 encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("G.728 encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("G.728 encoder: odd byte count"));
        }
        for chunk in bytes.chunks_exact(2) {
            self.pcm_queue
                .push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        self.drain(false);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.drain(true);
        }
        Ok(())
    }
}

// =========================================================================
// Analysis-by-synthesis core
// =========================================================================

/// Encode a single 5-sample target vector into a 10-bit codebook index and
/// advance `state` identically to the decoder's `decode_vector`.
///
/// Returns the raw 10-bit index (`shape(7) | sign(1) | mag(2)`).
///
/// The search is exhaustive over all 128 × 8 = 1024 candidates. For each
/// candidate we compute:
///
/// ```text
///   candidate_out[n] = ZIR[n] + scale * ZSR_shape[n]
/// ```
///
/// where `ZIR` is the synthesis filter's response to zero excitation
/// (memory-only tail) and `ZSR_shape` is the filter's response to the
/// unit-RMS shape vector with zero memory. Both are 5-sample vectors we
/// compute on the fly; because `1/A(z)` is linear in the excitation we
/// get the candidate output without running the full IIR per combination.
pub(crate) fn encode_vector(state: &mut G728State, target: &[f32; VECTOR_SIZE]) -> u16 {
    // Per-vector log-gain prediction (same first step the decoder runs).
    // This updates `last_log_gain` using the current gain-predictor
    // coefficients over the running log-gain history.
    state.gain.predict();

    // Adaptive scale applied by the decoder — same formula as
    // G728State::excitation_from_index.
    let adaptive_gain = state.gain.last_log_gain.exp();

    // ZIR: output of 1/A(z) with zero excitation, given the current
    // synthesis history. We run 5 steps of the IIR using a *copy* of the
    // short delay line; the real state is advanced below once we know
    // the winning excitation.
    let zir = compute_zir(&state.lpc.a, &state.lpc.history);

    // Pre-compute ZSR for each shape codeword: output of 1/A(z) fed
    // shape[0..5] with zero memory.
    let mut zsr: [[f32; VECTOR_SIZE]; SHAPE_CB_SIZE] = [[0.0; VECTOR_SIZE]; SHAPE_CB_SIZE];
    for s in 0..SHAPE_CB_SIZE {
        zsr[s] = compute_zsr(&state.lpc.a, &SHAPE_CB[s]);
    }

    // Search all (shape, sign, mag) triples.
    let mut best_shape = 0u16;
    let mut best_sign = 0u16; // 0 = +, 1 = -
    let mut best_mag = 0u16;
    let mut best_err = f32::INFINITY;

    for shape_idx in 0..SHAPE_CB_SIZE {
        let zsr_s = &zsr[shape_idx];
        for mag_idx in 0..GAIN_CB_SIZE {
            let mag = GAIN_CB[mag_idx];
            // Each magnitude is tried with +1 and -1 signs.
            for sign_bit in 0..2u8 {
                let sign_f: f32 = if sign_bit != 0 { -1.0 } else { 1.0 };
                let scale = sign_f * mag * adaptive_gain;
                let mut err = 0.0_f32;
                for n in 0..VECTOR_SIZE {
                    let out_n = zir[n] + scale * zsr_s[n];
                    let d = out_n - target[n];
                    err += d * d;
                }
                if err < best_err {
                    best_err = err;
                    best_shape = shape_idx as u16;
                    best_sign = sign_bit as u16;
                    // `GAIN_CB_SIZE == 4`; the 2-bit mag field covers all
                    // four magnitudes directly, with the sign bit doubling
                    // the reachable set to 8 signed levels.
                    best_mag = (mag_idx as u16) & 0x03;
                }
            }
        }
    }

    // Pack: shape(7) | sign(1) | mag(2)
    let raw = (best_shape << 3) | (best_sign << 2) | best_mag;

    // Now advance the decoder state with the chosen excitation — use the
    // exact same code path the decoder uses so the histories stay in sync.
    let mut reconstructed = [0.0_f32; VECTOR_SIZE];
    advance_state(state, raw, &mut reconstructed);

    raw
}

/// Apply the all-pole synthesis filter to a zero excitation, returning the
/// 5 samples that would be emitted from the current filter memory alone.
/// The filter state is **not** modified.
fn compute_zir(a: &[f32; LPC_ORDER + 1], history: &[f32; LPC_ORDER]) -> [f32; VECTOR_SIZE] {
    let mut hist = *history;
    let mut out = [0.0_f32; VECTOR_SIZE];
    for n in 0..VECTOR_SIZE {
        let mut acc = 0.0_f32;
        for k in 1..=LPC_ORDER {
            acc -= a[k] * hist[k - 1];
        }
        let y = acc.clamp(-1.0e4, 1.0e4);
        out[n] = y;
        // Shift short history: newest sample at index 0.
        for k in (1..LPC_ORDER).rev() {
            hist[k] = hist[k - 1];
        }
        hist[0] = y;
    }
    out
}

/// Apply the all-pole synthesis filter to `excitation` with **zero** filter
/// memory, returning the 5-sample output. This is the "zero-state
/// response" used in the linear decomposition `y = ZIR + ZSR`.
fn compute_zsr(a: &[f32; LPC_ORDER + 1], excitation: &[f32; VECTOR_SIZE]) -> [f32; VECTOR_SIZE] {
    let mut hist = [0.0_f32; LPC_ORDER];
    let mut out = [0.0_f32; VECTOR_SIZE];
    for n in 0..VECTOR_SIZE {
        let mut acc = excitation[n];
        for k in 1..=LPC_ORDER {
            acc -= a[k] * hist[k - 1];
        }
        let y = acc.clamp(-1.0e4, 1.0e4);
        out[n] = y;
        for k in (1..LPC_ORDER).rev() {
            hist[k] = hist[k - 1];
        }
        hist[0] = y;
    }
    out
}

/// Advance decoder state identically to `G728State::decode_vector`, using
/// the chosen raw 10-bit index. Fills `out` with the reconstructed PCM
/// (matches what the decoder will emit for this vector).
///
/// **Important**: the encoder's `encode_vector` already called
/// `state.gain.predict()` once at the start of the search to obtain the
/// current adaptive scale. We do **not** call it again here — the
/// decoder's per-vector sequence is `predict → decode → push → refresh`,
/// and the encoder mirrors that split across `encode_vector`'s preamble
/// and this post-selection advance.
fn advance_state(state: &mut G728State, raw: u16, out: &mut [f32; VECTOR_SIZE]) {
    let excitation = state.excitation_from_index(raw);
    state.lpc.synthesise(&excitation, out);
    let mut ss = 0.0_f32;
    for n in 0..VECTOR_SIZE {
        ss += excitation[n] * excitation[n];
    }
    let rms = (ss / VECTOR_SIZE as f32).sqrt();
    let log_g = rms.max(1.0e-6).ln();
    state.gain.push(log_g);
    state.vector_count = state.vector_count.wrapping_add(1);
    if state.lpc.vectors_since_update >= VECTORS_PER_BLOCK {
        state.lpc.refresh_coefficients();
    }
    if state.gain.vectors_since_update >= VECTORS_PER_BLOCK {
        state.gain.refresh_coefficients();
    }
}

/// Pack four 10-bit indices MSB-first into a 5-byte packet. Layout: the
/// first index lands in bits 39..30 of the 40-bit word, the second in
/// 29..20, and so on. Within each byte the high bit is bit 7.
fn pack_four_indices(indices: &[u16]) -> Vec<u8> {
    debug_assert_eq!(indices.len(), VECTORS_PER_PACKET);
    let mut out = vec![0u8; PACKET_BYTES];
    let mut bit_pos: usize = 0;
    for &raw in indices {
        let v = raw & 0x03FF;
        for b in (0..10).rev() {
            let bit = ((v >> b) & 1) as u8;
            let byte_idx = bit_pos / 8;
            let shift = 7 - (bit_pos % 8);
            out[byte_idx] |= bit << shift;
            bit_pos += 1;
        }
    }
    out
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_four_indices_known_pattern() {
        // Four indices of 0x3FF (all-ones) = 40 ones = 0xFF × 5.
        let bytes = pack_four_indices(&[0x3FF, 0x3FF, 0x3FF, 0x3FF]);
        assert_eq!(bytes, vec![0xFF; PACKET_BYTES]);
        // All-zeros.
        let bytes = pack_four_indices(&[0, 0, 0, 0]);
        assert_eq!(bytes, vec![0; PACKET_BYTES]);
        // Alternating patterns: 0x2AA, 0x155, 0x2AA, 0x155.
        // 10 10 10 10 10  01 01 01 01 01  10 10 10 10 10  01 01 01 01 01
        // Grouped into 8-bit bytes:
        //   10101010  10010101  01011010 10101001  01010101
        // = 0xAA      0x95      0x5A     0xA9      0x55
        let bytes = pack_four_indices(&[0x2AA, 0x155, 0x2AA, 0x155]);
        assert_eq!(bytes, vec![0xAA, 0x95, 0x5A, 0xA9, 0x55]);
    }

    #[test]
    fn encode_vector_produces_valid_index() {
        let mut st = G728State::new();
        let target = [100.0_f32, -200.0, 150.0, 50.0, -30.0];
        let raw = encode_vector(&mut st, &target);
        assert!(raw < 1024, "raw index out of 10-bit range: {raw}");
    }

    #[test]
    fn encoder_silence_pipeline_is_stable() {
        // Encode 200 vectors of silence — state should stay bounded.
        let mut st = G728State::new();
        let target = [0.0_f32; VECTOR_SIZE];
        for _ in 0..200 {
            let raw = encode_vector(&mut st, &target);
            assert!(raw < 1024);
            for v in st.lpc.history.iter() {
                assert!(v.is_finite(), "synth history went non-finite: {v}");
                assert!(v.abs() < 1.0e4, "synth history blew up: {v}");
            }
        }
    }

    #[test]
    fn compute_zir_with_identity_filter_is_zero() {
        // a[0]=1, a[k>0]=0 ⇒ filter is pass-through, so ZIR = 0 regardless
        // of history (the feedback sum is zero).
        let mut a = [0.0_f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let mut hist = [0.0_f32; LPC_ORDER];
        hist[0] = 42.0;
        hist[3] = -7.5;
        let z = compute_zir(&a, &hist);
        assert!(z.iter().all(|&s| s.abs() < 1e-6));
    }

    #[test]
    fn compute_zsr_with_identity_filter_is_input() {
        // a[0]=1, a[k>0]=0 ⇒ ZSR(x) = x.
        let mut a = [0.0_f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let x = [1.0_f32, -0.5, 0.25, 2.0, -1.0];
        let z = compute_zsr(&a, &x);
        for k in 0..VECTOR_SIZE {
            assert!((z[k] - x[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn make_encoder_rejects_stereo() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(2);
        assert!(make_encoder(&params).is_err());
    }

    #[test]
    fn make_encoder_accepts_valid_params() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        assert!(make_encoder(&params).is_ok());
    }
}
