//! CELT encoder — single-mode (mono, 48 kHz, 960 samples/frame, LM=3, FB).
//!
//! Scope (per RFC 6716 §4.3):
//!
//! * §4.3 frame header: silence=0, post_filter=None, transient=false, intra=true.
//! * §4.3.2.1 coarse energy: [`crate::quant_bands::quant_coarse_energy`] —
//!   Laplace-encoded, inter/intra prediction coefficients as per libopus.
//! * §4.3.2.2 fine energy: [`crate::quant_bands::quant_fine_energy`].
//! * §4.3.2.3 fine-energy finalise: [`crate::quant_bands::quant_energy_finalise`].
//! * §4.3.3 bit allocation: [`crate::encoder_rate::clt_compute_allocation_enc`] —
//!   matches the decoder's alloc table lookup, trim, and skip bits.
//! * §4.3.4 PVQ shape encoding: [`crate::encoder_bands::encode_all_bands_mono`]
//!   — per-band shape search, exp_rotation, canonical PVQ enumeration.
//! * §4.3.5 anti-collapse: not set (the encoder emits `transient=false` so
//!   no anti-collapse bit is reserved).
//! * §4.3.6 denormalisation: implicit in the decoder; encoder normalises
//!   the forward-MDCT coefficients before PVQ.
//! * §4.3.7 forward MDCT: [`crate::mdct::forward_mdct`] (direct definition).
//! * §4.3.8 comb post-filter: not applied.
//!
//! NOT implemented (fall back to long-block / no-boost path):
//!
//! * **Transient detection / short blocks** — we always emit `transient=false`
//!   and encode a single 960-sample block. Percussive content will suffer
//!   pre-echo artefacts.
//! * **Time-frequency change flags** — all `tf_res[i]` = 0.
//! * **Dynalloc band-energy boosts** — no per-band boost is emitted.
//! * **Stereo** — mono only. The encoder returns `Error::Unsupported` for
//!   multi-channel input.
//! * **Inter-frame energy prediction** — every frame is `intra=true`, which
//!   costs more bits on steady-state content but eliminates state drift.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::encoder_bands::encode_all_bands_mono;
use crate::encoder_rate::clt_compute_allocation_enc;
use crate::header::encode_header;
use crate::mdct::forward_mdct;
use crate::quant_bands::{quant_coarse_energy, quant_energy_finalise, quant_fine_energy};
use crate::range_encoder::{RangeEncoder, BITRES};
use crate::tables::{
    init_caps, lm_for_frame_samples, EBAND_5MS, E_MEANS, NB_EBANDS, SPREAD_ICDF, SPREAD_NORMAL,
    TRIM_ICDF,
};

/// True CELT frame length at LM=3: 960 samples = 20 ms at 48 kHz.
/// External callers feed and receive `FRAME_SAMPLES` samples per frame.
pub const FRAME_SAMPLES: usize = 960;
pub const SAMPLE_RATE: u32 = 48_000;
const OVERLAP: usize = 120;

/// Internal MDCT-coded length — `EBAND_5MS[21] * M = 800`. Bins 800..960
/// after forward MDCT are not transmitted; the decoder reconstructs them
/// as zero. This matches the decoder's `n = m * EBAND_5MS[NB_EBANDS]` = 800
/// (the output IMDCT time length is 2N = 1600, the remaining 320 samples
/// of the "true" 1920-sample long block are the zero-padded tail that the
/// next frame's overlap supplies).
pub const CODED_N: usize = 800;

/// Fixed target bitrate: 160 bytes/frame ≈ 64 kbit/s at 20 ms frames.
const DEFAULT_BYTES_PER_FRAME: usize = 160;

/// CELT window — same as libopus and the decoder's post-filter (120 taps).
#[rustfmt::skip]
const WINDOW_120: [f32; 120] = [
    6.7286966e-05, 0.00060551348, 0.0016815970, 0.0032947962, 0.0054439943,
    0.0081276923, 0.011344001, 0.015090633, 0.019364886, 0.024163635,
    0.029483315, 0.035319905, 0.041668911, 0.048525347, 0.055883718,
    0.063737999, 0.072081616, 0.080907428, 0.090207705, 0.099974111,
    0.11019769, 0.12086883, 0.13197729, 0.14351214, 0.15546177,
    0.16781389, 0.18055550, 0.19367290, 0.20715171, 0.22097682,
    0.23513243, 0.24960208, 0.26436860, 0.27941419, 0.29472040,
    0.31026818, 0.32603788, 0.34200931, 0.35816177, 0.37447407,
    0.39092462, 0.40749142, 0.42415215, 0.44088423, 0.45766484,
    0.47447104, 0.49127978, 0.50806798, 0.52481261, 0.54149077,
    0.55807973, 0.57455701, 0.59090049, 0.60708841, 0.62309951,
    0.63891306, 0.65450896, 0.66986776, 0.68497077, 0.69980010,
    0.71433873, 0.72857055, 0.74248043, 0.75605424, 0.76927895,
    0.78214257, 0.79463430, 0.80674445, 0.81846456, 0.82978733,
    0.84070669, 0.85121779, 0.86131698, 0.87100183, 0.88027111,
    0.88912479, 0.89756398, 0.90559094, 0.91320904, 0.92042270,
    0.92723738, 0.93365955, 0.93969656, 0.94535671, 0.95064907,
    0.95558353, 0.96017067, 0.96442171, 0.96834849, 0.97196334,
    0.97527906, 0.97830883, 0.98106616, 0.98356480, 0.98581869,
    0.98784191, 0.98964856, 0.99125274, 0.99266849, 0.99390969,
    0.99499004, 0.99592297, 0.99672162, 0.99739874, 0.99796667,
    0.99843728, 0.99882195, 0.99913147, 0.99937606, 0.99956527,
    0.99970802, 0.99981248, 0.99988613, 0.99993565, 0.99996697,
    0.99998518, 0.99999457, 0.99999859, 0.99999982, 1.0000000,
];

pub struct CeltEncoder {
    params: CodecParameters,
    pending: Vec<f32>,
    prev_tail: Vec<f32>,
    /// Previous frame's quantised band energies (per-channel × NB_EBANDS).
    old_band_e: Vec<f32>,
    output: VecDeque<Packet>,
    bytes_per_frame: usize,
    pts_counter: i64,
}

impl CeltEncoder {
    pub fn new(params: &CodecParameters) -> Result<Self> {
        let channels = params.channels.unwrap_or(1) as usize;
        if channels != 1 {
            return Err(Error::unsupported(
                "CELT encoder: only mono (single-channel) is supported in this build",
            ));
        }
        let sr = params.sample_rate.unwrap_or(SAMPLE_RATE);
        if sr != SAMPLE_RATE {
            return Err(Error::unsupported("CELT encoder: only 48 kHz is supported"));
        }
        let mut out_params = params.clone();
        out_params.channels = Some(1);
        out_params.sample_rate = Some(SAMPLE_RATE);
        Ok(Self {
            params: out_params,
            pending: Vec::new(),
            prev_tail: vec![0.0; OVERLAP],
            old_band_e: vec![0.0; NB_EBANDS * 2],
            output: VecDeque::new(),
            bytes_per_frame: DEFAULT_BYTES_PER_FRAME,
            pts_counter: 0,
        })
    }

    fn drain_frames(&mut self) -> Result<()> {
        while self.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.pending.drain(..FRAME_SAMPLES).collect();
            let pkt = self.encode_frame(&frame)?;
            self.output.push_back(pkt);
        }
        Ok(())
    }

    fn encode_frame(&mut self, pcm: &[f32]) -> Result<Packet> {
        debug_assert_eq!(pcm.len(), FRAME_SAMPLES);
        let lm = lm_for_frame_samples(FRAME_SAMPLES as u32) as i32;
        debug_assert_eq!(lm, 3);

        // Build the 2N-point MDCT input frame where N = CODED_N (the
        // coded-bin count). We resample the PCM into CODED_N samples by
        // simple truncation: keep the first CODED_N samples of the frame +
        // previous tail. This loses the top ~2 kHz of bandwidth per frame
        // but keeps the encoder's MDCT coefficient count consistent with
        // what the decoder's `pcm_per_ch = vec![0f32; 800]` expects.
        let n = CODED_N;
        let mut raw = vec![0f32; 2 * n];
        raw[..OVERLAP].copy_from_slice(&self.prev_tail);
        // Place up to CODED_N PCM samples starting after the overlap.
        let take = n.min(pcm.len());
        raw[OVERLAP..OVERLAP + take].copy_from_slice(&pcm[..take]);
        // Stash the tail of THIS frame's PCM (last OVERLAP samples) for the
        // next frame — this uses the raw frame (FRAME_SAMPLES=960) tail, not
        // the coded n=800 tail, so the OLA at the decoder side lines up.
        self.prev_tail
            .copy_from_slice(&pcm[FRAME_SAMPLES - OVERLAP..]);

        // Apply CELT window (only the overlap regions).
        crate::mdct::window_forward(&mut raw, &WINDOW_120, n, OVERLAP);

        // 2) Forward MDCT → N coefficients.
        let mut coeffs = vec![0f32; n];
        forward_mdct(&raw, &mut coeffs);

        // 3) Compute per-band log-energy and normalised shape.
        let m = 1i32 << lm;
        let mut band_log_e = vec![0f32; NB_EBANDS];
        let mut shape = vec![0f32; n];
        for i in 0..NB_EBANDS {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            let mut e: f32 = 0.0;
            for &c in &coeffs[lo..hi] {
                e += c * c;
            }
            let e = e.max(1e-30).sqrt();
            band_log_e[i] = e.log2() - E_MEANS[i];
            for c in &mut shape[lo..hi] {
                *c /= e;
            }
        }

        // 4) Range-code the frame.
        let bytes = self.bytes_per_frame;
        let mut rc = RangeEncoder::new(bytes as u32);
        // Header: silence=0, no post-filter, transient=0, intra=1.
        encode_header(&mut rc, false, None, false, true);

        // Coarse energy.
        let mut new_log_e = vec![0f32; NB_EBANDS * 2];
        new_log_e[..NB_EBANDS].copy_from_slice(&band_log_e);
        let old_before = self.old_band_e.clone();
        let mut old_e_bands = old_before.clone();
        quant_coarse_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            true,
            1,
            lm as usize,
        );

        // tf_decode: emit all zeros (no transient).
        // Decoder loop: `if tell + logp <= budget_after: decode_bit_logp(logp)`,
        // initial logp=4 (non-transient), then 5 for each. To keep tf_res=0
        // we emit `false` for every one of these bits.
        let budget = (bytes * 8) as u32;
        let mut tell_u = rc.tell() as u32;
        let mut logp = 4u32;
        let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        let mut tf_res = vec![0i32; NB_EBANDS];
        for _i in 0..NB_EBANDS {
            if tell_u + logp <= budget_after {
                rc.encode_bit_logp(false, logp);
                tell_u = rc.tell() as u32;
            }
            logp = 5;
        }
        // tf_select is only emitted if the two table rows differ.
        // TF_SELECT_TABLE[3][0] vs [3][2]: both are 0 (all non-transient entries
        // start with 0). So we don't write tf_select.
        let _ = tf_select_rsv;

        // Spread decision.
        let mut tell = rc.tell();
        let total_bits_check = (bytes * 8) as i32;
        if tell + 4 <= total_bits_check {
            rc.encode_icdf(SPREAD_NORMAL as usize, &SPREAD_ICDF, 5);
        }

        // dynalloc offsets: emit ALL zeros. Dec loop emits `decode_bit_logp(dynalloc_logp)`
        // until it gets back false. So we emit ONE false per band (no boosts).
        let cap = init_caps(lm as usize, 1);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = (bytes as i32) * 8 << BITRES;
        tell = rc.tell_frac() as i32;
        for i in 0..NB_EBANDS {
            let width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                // Emit `false` = no boost.
                rc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                tell = rc.tell_frac() as i32;
                break; // decoder breaks out on `!flag`.
            }
            offsets[i] = boost;
            // dynalloc_logp stays at 6 since we added no boost.
        }
        let _ = total_bits_frac;

        // Allocation trim — emit default (5).
        if tell + (6 << BITRES) <= total_bits_frac {
            rc.encode_icdf(5, &TRIM_ICDF, 7);
        }

        let mut bits = ((bytes as i32) * 8 << BITRES) - rc.tell_frac() as i32 - 1;
        // No anti-collapse rsv since transient=false.

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut balance = 0i32;
        let coded_bands = clt_compute_allocation_enc(
            0,
            NB_EBANDS,
            &offsets,
            &cap,
            5,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            1,
            lm,
            &mut rc,
        );

        // Fine energy.
        quant_fine_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            1,
        );

        // PVQ shape.
        let total_pvq_bits = (bytes as i32) * (8 << BITRES);
        let mut collapse_masks = vec![0u8; NB_EBANDS];
        let mut rng_local = 0u32;
        encode_all_bands_mono(
            0,
            NB_EBANDS,
            &mut shape,
            &mut collapse_masks,
            &pulses,
            SPREAD_NORMAL,
            &tf_res,
            total_pvq_bits,
            balance,
            &mut rc,
            lm,
            coded_bands,
            &mut rng_local,
        );

        // Final fine-energy pass (bits left from total - rc.tell()).
        let bits_left = (bytes as i32) * 8 - rc.tell();
        quant_energy_finalise(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            &fine_priority,
            bits_left,
            1,
        );

        // Commit energy state for next frame.
        self.old_band_e = old_e_bands;

        let buf = rc.done()?;
        let tb = TimeBase::new(1, SAMPLE_RATE as i64);
        let pts = self.pts_counter;
        self.pts_counter += FRAME_SAMPLES as i64;
        Ok(Packet::new(0, tb, buf)
            .with_pts(pts)
            .with_duration(FRAME_SAMPLES as i64))
    }
}

impl Encoder for CeltEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let audio = match frame {
            Frame::Audio(a) => a,
            _ => {
                return Err(Error::invalid(
                    "CELT encoder: expected audio frame, got video",
                ))
            }
        };
        if audio.channels != 1 {
            return Err(Error::unsupported(
                "CELT encoder: only mono input supported in this build",
            ));
        }
        let samples = extract_mono_f32(audio)?;
        self.pending.extend(samples.into_iter());
        self.drain_frames()
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output.pop_front() {
            Ok(p)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Pad with zeros to a frame boundary, then drain.
        if !self.pending.is_empty() {
            let rem = FRAME_SAMPLES - self.pending.len();
            self.pending.extend(std::iter::repeat(0.0f32).take(rem));
            self.drain_frames()?;
        }
        Ok(())
    }
}

/// Convert the `AudioFrame`'s samples to `Vec<f32>` (mono). Supports the
/// sample formats we actually hit in practice: F32, F32P, S16, S16P.
fn extract_mono_f32(audio: &AudioFrame) -> Result<Vec<f32>> {
    let n = audio.samples as usize;
    let mut out = vec![0f32; n];
    match audio.format {
        SampleFormat::F32 => {
            let bytes = &audio.data[0];
            if bytes.len() < n * 4 {
                return Err(Error::invalid("CELT encoder: F32 input too short"));
            }
            for i in 0..n {
                let b = &bytes[i * 4..i * 4 + 4];
                out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        SampleFormat::F32P => {
            let bytes = &audio.data[0];
            if bytes.len() < n * 4 {
                return Err(Error::invalid("CELT encoder: F32P input too short"));
            }
            for i in 0..n {
                let b = &bytes[i * 4..i * 4 + 4];
                out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        SampleFormat::S16 | SampleFormat::S16P => {
            let bytes = &audio.data[0];
            if bytes.len() < n * 2 {
                return Err(Error::invalid("CELT encoder: S16 input too short"));
            }
            for i in 0..n {
                let s = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = s as f32 / 32768.0;
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "CELT encoder: sample format {:?} not supported",
                other
            )));
        }
    }
    let _ = MediaType::Audio;
    Ok(out)
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(CeltEncoder::new(params)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CELT encoder and drive it with one frame of silence — the
    /// encoded packet should parse back via the decoder without error.
    #[test]
    fn silence_frame_produces_valid_packet() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        let pcm = vec![0.0f32; FRAME_SAMPLES];
        let pkt = enc.encode_frame(&pcm).unwrap();
        assert!(!pkt.data.is_empty());
        // Smoke-check via the range decoder: the header should parse.
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        let h = crate::header::decode_header(&mut rd);
        assert!(h.is_some(), "header should parse");
        let h = h.unwrap();
        assert!(!h.silence);
        assert!(!h.transient);
        assert!(h.intra);
    }
}
