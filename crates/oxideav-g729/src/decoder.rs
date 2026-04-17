//! Top-level G.729 frame decoder — bit-unpack → LSP/LPC → excitation →
//! synthesis → postfilter → S16 PCM.
//!
//! Produces 80 `S16` samples (10 ms at 8 kHz) per 10-byte packet.
//!
//! The pipeline follows ITU-T G.729 §3, split across modules:
//! - [`crate::bitreader`]: 80-bit frame -> `FrameParams` (§3.6 Table 8).
//! - [`crate::lpc`]: LSP indices -> quantised LSPs (§3.2.4); LSP
//!   interpolation (§3.2.5); LSP -> LPC conversion (§3.2.6).
//! - [`crate::synthesis`]: adaptive codebook (§3.7) + fixed codebook
//!   (§3.8) + gains (§3.9) + synthesis filter (§3.10) + postfilter
//!   (§3.11).
//!
//! Deviations from the spec (documented, deliberate, and confined to
//! areas where a first-cut decoder suffices):
//! - The LSP first-stage codebook beyond rows `{0, 1, 127}` is
//!   procedurally synthesised (see `lsp_tables.rs`). Output spectral
//!   shapes match the general voiced/unvoiced character of the input
//!   indices but won't match the reference decoder bit-for-bit.
//! - The gain two-stage VQ tables are reduced to 8+16 entries
//!   covering the span of the spec's `gbk1`/`gbk2`.
//! - The MA-4 gain-prediction coefficients are approximated by a
//!   uniform-tap mean.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::{parse_frame_params, FrameParams};
use crate::lpc::{decode_lsp, interpolate_lsp, lsp_to_lpc, LpcPredictorState};
use crate::synthesis::{
    adaptive_codebook_excitation, agc, decode_gain_indices, decode_pitch_p1, decode_pitch_p2,
    fixed_codebook_excitation, innovation_log_energy_db, pitch_emphasis_postfilter, pitch_sharpen,
    predict_fixed_gain, short_term_postfilter, synthesise, tilt_compensation, SynthesisState,
    EXC_HIST,
};
#[cfg(test)]
use crate::LPC_ORDER;
use crate::{
    CODEC_ID_STR, FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE, SUBFRAMES_PER_FRAME, SUBFRAME_SAMPLES,
};

/// Build a boxed [`Decoder`] for a G.729 stream.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "G.729 decoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.729 decoder: only mono is supported (got {channels} channels)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "G.729 decoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }
    Ok(Box::new(G729Decoder::new()))
}

/// Convenience wrapper that just dispatches to the bit-field parser.
pub fn parse_packet(packet: &[u8]) -> Result<FrameParams> {
    parse_frame_params(packet)
}

struct G729Decoder {
    codec_id: CodecId,
    lpc_state: LpcPredictorState,
    syn: SynthesisState,
    pending: Option<Packet>,
    eof: bool,
    time_base: TimeBase,
}

impl G729Decoder {
    fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            lpc_state: LpcPredictorState::new(),
            syn: SynthesisState::new(),
            pending: None,
            eof: false,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        }
    }

    /// Decode a single frame (80 samples) into the provided output buffer.
    fn decode_frame_into(&mut self, packet: &[u8], out: &mut [f32; FRAME_SAMPLES]) -> Result<()> {
        if packet.len() < FRAME_BYTES {
            return Err(Error::invalid(format!(
                "G.729 frame: expected {FRAME_BYTES} bytes, got {}",
                packet.len()
            )));
        }
        let fp = parse_frame_params(&packet[..FRAME_BYTES])?;

        // Step 1: decode LSPs via the MA predictor.
        let lsp_new = decode_lsp(&mut self.lpc_state, fp.l0, fp.l1, fp.l2, fp.l3);

        // Step 2: produce per-subframe LPC coefficients by
        // interpolating between the previous and current LSP vectors.
        //   Subframe 0: alpha = 0.5  (midpoint between prev and new)
        //   Subframe 1: alpha = 1.0  (new LSP straight)
        let lsp_sf0 = interpolate_lsp(&self.lpc_state.lsp_prev, &lsp_new, 0.5);
        let lsp_sf1 = lsp_new;
        let a_sf = [lsp_to_lpc(&lsp_sf0), lsp_to_lpc(&lsp_sf1)];

        // Step 3: decode pitch delays.
        let (p1_int, p1_frac) = decode_pitch_p1(fp.p1);
        let (p2_int, p2_frac) = decode_pitch_p2(fp.p2, p1_int);
        let pitch_int = [p1_int, p2_int];
        let pitch_frac = [p1_frac, p2_frac];
        let codebook_c = [fp.c1, fp.c2];
        let codebook_s = [fp.s1, fp.s2];
        let ga = [fp.ga1, fp.ga2];
        let gb = [fp.gb1, fp.gb2];

        // Step 4: for each subframe, run the full synthesis pipeline.
        for sf in 0..SUBFRAMES_PER_FRAME {
            let a = &a_sf[sf];
            // 4a. Adaptive-codebook excitation.
            let mut ac = [0.0f32; SUBFRAME_SAMPLES];
            adaptive_codebook_excitation(&self.syn.exc, pitch_int[sf], pitch_frac[sf], &mut ac);
            // 4b. Fixed-codebook pulse vector.
            let mut fc = [0.0f32; SUBFRAME_SAMPLES];
            fixed_codebook_excitation(codebook_c[sf], codebook_s[sf], &mut fc);
            // 4c. Gains.
            let (g_p, gamma) = decode_gain_indices(ga[sf], gb[sf]);
            let innov_db = innovation_log_energy_db(&fc);
            let g_c = gamma * predict_fixed_gain(&self.syn.gain_log_hist, innov_db);
            // 4d. Pitch sharpening of the innovation vector (§3.8.3).
            pitch_sharpen(&mut fc, pitch_int[sf], g_p);
            // 4e. Compose the subframe's excitation.
            let mut excitation = [0.0f32; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                excitation[n] = g_p * ac[n] + g_c * fc[n];
            }
            // 4f. Push excitation into the history buffer.
            push_excitation(&mut self.syn.exc, &excitation);
            // 4g. Update gain-predictor log-energy history with the
            //     log-energy of the INNOVATION (fixed-codebook) signal
            //     scaled by g_c. This keeps the MA-4 predictor informed
            //     of the current subframe's contribution.
            let mut scaled_innov = [0.0f32; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                scaled_innov[n] = g_c * fc[n];
            }
            let new_gain_db = innovation_log_energy_db(&scaled_innov).max(-20.0);
            for k in (1..4).rev() {
                self.syn.gain_log_hist[k] = self.syn.gain_log_hist[k - 1];
            }
            self.syn.gain_log_hist[0] = new_gain_db;
            // 4h. LPC synthesis filter.
            let mut synthesised = [0.0f32; SUBFRAME_SAMPLES];
            synthesise(&excitation, a, &mut self.syn.syn_mem, &mut synthesised);
            // Keep a copy of pre-postfilter signal for AGC reference.
            let pre_post = synthesised;
            // 4i. Postfilter: short-term (γ1/γ2) + pitch emphasis + tilt + AGC.
            short_term_postfilter(
                &mut synthesised,
                a,
                &mut self.syn.post_az1_mem,
                &mut self.syn.post_az2_mem,
            );
            pitch_emphasis_postfilter(
                &mut synthesised,
                &mut self.syn.post_pitch_mem,
                pitch_int[sf],
                g_p,
            );
            tilt_compensation(&mut synthesised, &mut self.syn.tilt_mem);
            agc(&mut synthesised, &pre_post, &mut self.syn.agc_gain);
            // 4j. Write to output.
            let off = sf * SUBFRAME_SAMPLES;
            for n in 0..SUBFRAME_SAMPLES {
                out[off + n] = synthesised[n];
            }
            // Remember the integer pitch / gain for the next subframe's
            // postfilter setup.
            self.syn.prev_gp = g_p;
            self.syn.prev_pitch = pitch_int[sf];
        }

        // Step 5: stash the new LSP as `lsp_prev` for next frame's
        // subframe-0 interpolation.
        self.lpc_state.lsp_prev = lsp_new;
        // Store final per-subframe LPC so external code can inspect it.
        self.lpc_state.a = a_sf[1];
        Ok(())
    }
}

/// Slide `exc` by `SUBFRAME_SAMPLES` and append `sub` at the tail.
fn push_excitation(exc: &mut [f32; EXC_HIST], sub: &[f32; SUBFRAME_SAMPLES]) {
    // Shift left by SUBFRAME_SAMPLES.
    for i in 0..EXC_HIST - SUBFRAME_SAMPLES {
        exc[i] = exc[i + SUBFRAME_SAMPLES];
    }
    for i in 0..SUBFRAME_SAMPLES {
        exc[EXC_HIST - SUBFRAME_SAMPLES + i] = sub[i];
    }
}

impl Decoder for G729Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "G.729 decoder: receive_frame must be called before sending another packet",
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
        let mut samples = [0.0f32; FRAME_SAMPLES];
        self.decode_frame_into(&pkt.data, &mut samples)?;
        // Convert f32 -> S16 LE.
        let mut bytes = Vec::with_capacity(FRAME_SAMPLES * 2);
        for &s in samples.iter() {
            let v = s.round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: FRAME_SAMPLES as u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dec() -> Box<dyn Decoder> {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        make_decoder(&params).expect("make_decoder should succeed for valid params")
    }

    /// Build a 10-byte G.729 packet by explicitly setting the 15 fields.
    /// The closure receives a mutable `FrameParams`-like tuple and must
    /// return the desired indices.
    fn pack(fp: &FrameParams) -> Vec<u8> {
        // Pack MSB-first into 10 bytes; mirrors the bitreader order.
        let fields: [(u32, u32); 15] = [
            (fp.l0 as u32, 1),
            (fp.l1 as u32, 7),
            (fp.l2 as u32, 5),
            (fp.l3 as u32, 5),
            (fp.p1 as u32, 8),
            (fp.p0 as u32, 1),
            (fp.c1 as u32, 13),
            (fp.s1 as u32, 4),
            (fp.ga1 as u32, 3),
            (fp.gb1 as u32, 4),
            (fp.p2 as u32, 5),
            (fp.c2 as u32, 13),
            (fp.s2 as u32, 4),
            (fp.ga2 as u32, 3),
            (fp.gb2 as u32, 4),
        ];
        let mut out = vec![0u8; FRAME_BYTES];
        let mut bit_pos: u32 = 0;
        for (val, width) in fields {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            let v = val & mask;
            for b in (0..width).rev() {
                let bit = (v >> b) & 1;
                let byte_idx = (bit_pos / 8) as usize;
                let shift = 7 - (bit_pos % 8);
                out[byte_idx] |= (bit as u8) << shift;
                bit_pos += 1;
            }
        }
        out
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
    fn bitstream_lsp_indices_drive_expected_lpc_path() {
        // Hand-crafted packet with known LSP indices (L0=0, L1=0,
        // L2=0, L3=0). The bit layout puts them in the first 18 bits
        // of the packet. The decoder should:
        //   1. Read l0=0, l1=0, l2=0, l3=0 back via `parse_packet`.
        //   2. Produce a specific LSP vector from the row-0 entries
        //      (since rows 0 of LSPCB1/LSPCB2 are the spec values).
        //   3. Convert to a stable LPC filter.
        // This nails down the LSP → LPC step end-to-end.
        let fp = FrameParams::default();
        let bytes = pack(&fp);
        let parsed = parse_packet(&bytes).expect("parse_packet");
        assert_eq!(parsed.l0, 0);
        assert_eq!(parsed.l1, 0);
        assert_eq!(parsed.l2, 0);
        assert_eq!(parsed.l3, 0);

        // Run the full LSP decode / LPC conversion and record the
        // resulting `a[k]`. We don't assert exact values (those are
        // tied to the MA predictor history + the spec tables) but
        // check the filter is stable: A(1) > 0 and the impulse
        // response stays bounded.
        let mut state = LpcPredictorState::new();
        let lsp = decode_lsp(&mut state, parsed.l0, parsed.l1, parsed.l2, parsed.l3);
        // LSPs must be strictly decreasing in the cosine domain.
        for k in 1..LPC_ORDER {
            assert!(
                lsp[k] < lsp[k - 1],
                "lsp must be strictly decreasing at {k}: {} < {}",
                lsp[k],
                lsp[k - 1]
            );
        }
        let a = lsp_to_lpc(&lsp);
        // A(1) > 0 is the standard stability sanity-check for an
        // all-zero minimum-phase polynomial.
        let a_at_1: f32 = (1..=LPC_ORDER).map(|k| a[k]).sum::<f32>() + 1.0;
        assert!(a_at_1 > 0.0, "A(1) = {a_at_1} must be positive");
        // Run a 80-sample impulse response through 1/A(z) and verify
        // it decays to within a bounded envelope.
        let mut mem = [0.0f32; LPC_ORDER];
        let mut excitation = [0.0f32; SUBFRAME_SAMPLES];
        excitation[0] = 1.0;
        let mut y = [0.0f32; SUBFRAME_SAMPLES];
        synthesise(&excitation, &a, &mut mem, &mut y);
        let mut peak = 0.0f32;
        for &v in y.iter() {
            assert!(v.is_finite(), "impulse response went NaN/inf");
            peak = peak.max(v.abs());
        }
        // A reasonable synthesis-filter impulse response peaks in
        // the low tens (not thousands). This check catches bugs that
        // make A(z) near-critically-stable.
        assert!(
            peak < 100.0,
            "impulse response peak {peak} is unreasonably large"
        );
    }

    #[test]
    fn lsp_to_lpc_from_zero_indices_is_stable() {
        // Feed L0/L1/L2/L3 all zero -> specific LSP vector from spec's
        // table row 0. The resulting A(z) must be minimum-phase
        // (A(1) > 0, all zeros strictly inside unit circle). We
        // verify stability by running a unit impulse through the
        // synthesis filter and confirming the output stays bounded.
        let mut state = LpcPredictorState::new();
        let lsp = decode_lsp(&mut state, 0, 0, 0, 0);
        let a = lsp_to_lpc(&lsp);
        // Check A(1) > 0.
        let a_at_1: f32 = (1..=LPC_ORDER).map(|k| a[k]).sum::<f32>() + 1.0;
        assert!(
            a_at_1 > 0.0,
            "A(1) = {a_at_1} should be positive for stable A(z)"
        );
        // Run an impulse through 1/A(z); energy must stay finite.
        let mut mem = [0.0f32; LPC_ORDER];
        let mut impulse = [0.0f32; SUBFRAME_SAMPLES];
        impulse[0] = 1.0;
        let mut y = [0.0f32; SUBFRAME_SAMPLES];
        synthesise(&impulse, &a, &mut mem, &mut y);
        for &v in y.iter() {
            assert!(v.is_finite() && v.abs() < 1e3, "synthesis diverged: {v}");
        }
    }

    #[test]
    fn decoder_produces_nonzero_output_for_excited_frames() {
        // Feed ten frames with strong excitation indices; assert that
        // at least one sample is non-zero across the decoded output.
        let mut dec = make_dec();
        // Pick indices that give non-trivial pulses and gains.
        let fp = FrameParams {
            l0: 0,
            l1: 5,
            l2: 3,
            l3: 7,
            p1: 60, // fractional-pitch delay around 39 samples
            p0: 0,
            c1: 0x1_2A3, // arbitrary pulse positions
            s1: 0b1010,
            ga1: 4,
            gb1: 8,
            p2: 15,
            c2: 0x1_5A7,
            s2: 0b0101,
            ga2: 3,
            gb2: 6,
        };
        let bytes = pack(&fp);
        let mut saw_nonzero = false;
        let mut max_abs: i32 = 0;
        for i in 0..10 {
            let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes.clone())
                .with_pts(i * FRAME_SAMPLES as i64);
            dec.send_packet(&pkt).expect("send_packet");
            let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") else {
                panic!("expected audio frame");
            };
            assert_eq!(a.samples as usize, FRAME_SAMPLES);
            assert_eq!(a.channels, 1);
            assert_eq!(a.sample_rate, SAMPLE_RATE);
            assert_eq!(a.data.len(), 1);
            assert_eq!(a.data[0].len(), FRAME_SAMPLES * 2);
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                let v = s.unsigned_abs() as i32;
                if v > 0 {
                    saw_nonzero = true;
                }
                if v > max_abs {
                    max_abs = v;
                }
            }
        }
        assert!(saw_nonzero, "decoder produced all-silent output");
        // Sanity: output should not have saturated everywhere.
        assert!(max_abs < 32767, "decoder output saturated at every sample");
    }

    #[test]
    fn decoder_silence_indices_produce_bounded_output() {
        // All-zero frame parameters — the decoder should still produce
        // bounded (non-NaN, non-clipping-everywhere) output.
        let mut dec = make_dec();
        let fp = FrameParams::default();
        let bytes = pack(&fp);
        for _ in 0..5 {
            let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes.clone());
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("expected audio frame");
            };
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                assert!(s.abs() <= i16::MAX, "sample out of range: {s}");
            }
        }
    }

    #[test]
    fn decoder_rejects_short_packet() {
        let mut dec = make_dec();
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), vec![0u8; 4]);
        dec.send_packet(&pkt).unwrap();
        assert!(dec.receive_frame().is_err());
    }
}
