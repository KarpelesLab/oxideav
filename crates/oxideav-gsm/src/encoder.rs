//! `oxideav_codec::Encoder` implementation for GSM 06.10 Full Rate.
//!
//! Mirror of [`crate::decoder`]: for every `dequantise` / `synthesis filter`
//! there is a matching `quantise` / `analysis filter` here. Pipeline:
//!
//! ```text
//! PCM s16 (160) → preprocess → LPC analysis → short-term analysis filter
//!                                    → 4 × (LTP analysis + RPE encoding)
//!                                    → bit-pack 33 bytes
//! ```
//!
//! The reference for numerical correctness is ETSI EN 300 961 §5.2 (and
//! the public-domain libgsm implementation by Jutta Degener). Code below
//! is re-derived, not copy-pasted.

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};
use std::collections::VecDeque;

use crate::bitreader::BitWriter;
use crate::frame::{FRAME_SIZE, MS_FRAME_SIZE};
use crate::math::{
    abs16, add, div16, l_add, mult, mult_r, norm32, sasr_i16, sasr_i32, saturate_i32_to_i16, shl,
    sub,
};
use crate::tables::{A, B, B_TIMES_TWO, DLB, FAC, H, INVA, MAC, MIC, NRFAC, QLB};

/// Build an encoder for the standard `gsm` or Microsoft `gsm_ms` framings.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(8_000);
    if sample_rate != 8_000 {
        return Err(Error::unsupported(format!(
            "GSM 06.10 encoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "GSM 06.10 encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "GSM 06.10 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    let variant = match params.codec_id.as_str() {
        crate::decoder::CODEC_ID_STANDARD => Variant::Standard,
        crate::decoder::CODEC_ID_MS => Variant::Microsoft,
        other => {
            return Err(Error::unsupported(format!(
                "GSM encoder: unknown codec id {other:?}"
            )))
        }
    };

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(8_000);
    output.codec_id = params.codec_id.clone();

    Ok(Box::new(GsmEncoder {
        output_params: output,
        variant,
        time_base: TimeBase::new(1, 8_000),
        analysis: AnalysisState::new(),
        pcm_queue: Vec::new(),
        pending_packets: VecDeque::new(),
        pending_first_half: None,
        first_half_pts: None,
        first_half_index: 0,
        frame_index: 0,
        eof: false,
    }))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Variant {
    Standard,
    Microsoft,
}

struct GsmEncoder {
    output_params: CodecParameters,
    variant: Variant,
    time_base: TimeBase,
    analysis: AnalysisState,
    /// Buffered PCM input (mono i16 samples).
    pcm_queue: Vec<i16>,
    pending_packets: VecDeque<Packet>,
    /// For MS framing: the first standard frame's 260-bit payload, waiting
    /// for its partner before the 65-byte combined packet can be emitted.
    pending_first_half: Option<[u8; 33]>,
    first_half_pts: Option<i64>,
    first_half_index: u64,
    frame_index: u64,
    eof: bool,
}

impl Encoder for GsmEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(a) => self.ingest(a),
            _ => Err(Error::invalid("GSM encoder: audio frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending_packets.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.drain_full_frames(true)?;
        }
        Ok(())
    }
}

impl GsmEncoder {
    fn ingest(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != 1 || frame.sample_rate != 8_000 {
            return Err(Error::invalid(
                "GSM encoder: input must be mono, 8000 Hz S16",
            ));
        }
        if frame.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "GSM encoder: input sample format must be S16",
            ));
        }
        let bytes = frame
            .data
            .first()
            .ok_or_else(|| Error::invalid("GSM encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("GSM encoder: odd byte count"));
        }
        for chunk in bytes.chunks_exact(2) {
            self.pcm_queue
                .push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        self.drain_full_frames(false)
    }

    fn drain_full_frames(&mut self, drain: bool) -> Result<()> {
        while self.pcm_queue.len() >= 160 {
            let mut pcm = [0i16; 160];
            pcm.copy_from_slice(&self.pcm_queue[..160]);
            self.pcm_queue.drain(..160);
            self.emit_standard_frame(&pcm)?;
        }
        if drain && !self.pcm_queue.is_empty() {
            let mut pcm = [0i16; 160];
            let n = self.pcm_queue.len();
            pcm[..n].copy_from_slice(&self.pcm_queue);
            self.pcm_queue.clear();
            self.emit_standard_frame(&pcm)?;
        }
        // For MS framing, if flushing leaves a lone first-half buffered,
        // pair it with a zero-filled second half so no frame is lost.
        if drain && self.variant == Variant::Microsoft {
            if let Some(first) = self.pending_first_half.take() {
                let second = encode_silent_half(&mut self.analysis);
                let combined = pack_ms_pair(&first, &second);
                let pts = self.first_half_pts;
                let idx = self.first_half_index;
                self.push_packet(combined, pts, idx);
            }
        }
        Ok(())
    }

    fn emit_standard_frame(&mut self, pcm: &[i16; 160]) -> Result<()> {
        let frame_idx = self.frame_index;
        self.frame_index += 1;
        let ff = self.analysis.analyse(pcm);
        let packed = pack_standard_frame(&ff);

        match self.variant {
            Variant::Standard => {
                let pts = Some(frame_idx as i64 * 160);
                self.push_packet(packed.to_vec(), pts, frame_idx);
            }
            Variant::Microsoft => {
                if let Some(first) = self.pending_first_half.take() {
                    let combined = pack_ms_pair(&first, &packed);
                    let pts = self.first_half_pts;
                    let idx = self.first_half_index;
                    self.push_packet(combined, pts, idx);
                } else {
                    self.pending_first_half = Some(packed);
                    self.first_half_pts = Some(frame_idx as i64 * 160);
                    self.first_half_index = frame_idx;
                }
            }
        }
        Ok(())
    }

    fn push_packet(&mut self, data: Vec<u8>, pts: Option<i64>, frame_idx: u64) {
        let duration = match self.variant {
            Variant::Standard => 160,
            Variant::Microsoft => 320,
        };
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.duration = Some(duration);
        pkt.flags.keyframe = true;
        let _ = frame_idx;
        self.pending_packets.push_back(pkt);
    }
}

/// All per-frame analysis fields ready for bit-packing.
#[derive(Clone, Debug)]
struct FrameFields {
    larc: [u8; 8],
    nc: [u8; 4],
    bc: [u8; 4],
    mc: [u8; 4],
    xmaxc: [u8; 4],
    xmc: [[u8; 13]; 4],
}

/// Per-sub-frame analysis result.
#[derive(Clone, Copy, Debug)]
struct SubFrameFields {
    nc: u8,
    bc: u8,
    mc: u8,
    xmaxc: u8,
    xmc: [u8; 13],
}

/// Persistent analysis state. Mirrors [`crate::synthesis::SynthesisState`].
#[derive(Clone, Debug)]
struct AnalysisState {
    /// Preprocessor: high-pass offset compensation state.
    pp_z1: i16,
    pp_l_z2: i32,
    /// Preprocessor: preemphasis previous output.
    pp_mp: i16,
    /// Short-term analysis filter state `u[0..8]` — paired with the
    /// synthesis `v[0..=8]` state on the decoder side.
    u: [i16; 8],
    /// Previous frame's decoded LARpp — for 4-subframe interpolation,
    /// exactly mirroring [`SynthesisState::larpp_prev`].
    larpp_prev: [i16; 8],
    /// LTP analysis history `dp[-120..-1]`: the previous reconstructed
    /// short-term residual. Stored as a sliding 120-sample buffer.
    dp_history: [i16; 120],
}

impl AnalysisState {
    fn new() -> Self {
        Self {
            pp_z1: 0,
            pp_l_z2: 0,
            pp_mp: 0,
            u: [0; 8],
            larpp_prev: [0; 8],
            dp_history: [0; 120],
        }
    }

    /// Encode one 160-sample frame, returning the frame fields bundled
    /// together via [`FrameFields`].
    fn analyse(&mut self, pcm: &[i16; 160]) -> FrameFields {
        // 1. Preprocess: downscale, offset-compensate, preemphasise.
        let mut so = [0i16; 160];
        self.preprocess(pcm, &mut so);

        // 2. LPC analysis: produce LARc and keep the current LARpp for
        //    short-term filtering + stash-for-next-frame.
        let larc = self.lpc_analysis(&so);
        let larpp_cur = decode_larc(&larc);

        // 3. Short-term analysis filter over the whole 160-sample window,
        //    using 4 interpolated sub-frames to mirror the synthesis path.
        let mut d = [0i16; 160];
        for k in 0..4 {
            let larp = interpolate_larpp(k, &self.larpp_prev, &larpp_cur);
            let rp = lar_to_rp(&larp);
            short_term_analysis(
                &rp,
                &mut self.u,
                &so[k * 40..k * 40 + 40],
                &mut d[k * 40..k * 40 + 40],
            );
        }
        self.larpp_prev = larpp_cur;

        // 4. Four sub-frame LTP + RPE passes. Each updates `dp_history`
        //    (the reconstructed residual).
        let mut nc = [0u8; 4];
        let mut bc = [0u8; 4];
        let mut mc = [0u8; 4];
        let mut xmaxc = [0u8; 4];
        let mut xmc = [[0u8; 13]; 4];
        for k in 0..4 {
            let d_sub = &d[k * 40..k * 40 + 40];
            let sf = self.encode_subframe(d_sub);
            nc[k] = sf.nc;
            bc[k] = sf.bc;
            mc[k] = sf.mc;
            xmaxc[k] = sf.xmaxc;
            xmc[k] = sf.xmc;
        }
        FrameFields {
            larc,
            nc,
            bc,
            mc,
            xmaxc,
            xmc,
        }
    }

    /// Per libgsm/preprocess.c.
    fn preprocess(&mut self, input: &[i16; 160], so: &mut [i16; 160]) {
        let mut z1 = self.pp_z1;
        let mut l_z2 = self.pp_l_z2;
        let mut mp = self.pp_mp;

        for k in 0..160 {
            // Downscale: (SASR(s, 3) << 2)  — keep 13 MSBs, room for math.
            let s = sasr_i16(input[k], 3);
            let so_k = (s as i32) << 2;
            let so_k = so_k as i16;

            // Offset compensation: s1 = SO - z1; z1 = SO
            let s1 = sub(so_k, z1);
            z1 = so_k;
            // L_s2 = s1 << 15
            let mut l_s2: i32 = (s1 as i32) << 15;
            // 31×16 multiply using msp/lsp split.
            let msp = sasr_i32(l_z2, 15) as i16;
            let lsp = l_z2 - ((msp as i32) << 15);
            l_s2 = l_s2.wrapping_add(mult_r(lsp as i16, 32_735) as i32);
            let l_temp = (msp as i32).wrapping_mul(32_735);
            l_z2 = l_add(l_temp, l_s2);
            // Round by adding 16384.
            let l_rounded = l_add(l_z2, 16_384);

            // Preemphasis: msp = MULT_R(mp, -28180); mp = SASR(L_temp, 15);
            //              so[k] = ADD(mp, msp)
            let msp_p = mult_r(mp, -28_180);
            mp = sasr_i32(l_rounded, 15) as i16;
            so[k] = add(mp, msp_p);
        }
        self.pp_z1 = z1;
        self.pp_l_z2 = l_z2;
        self.pp_mp = mp;
    }

    /// LPC analysis: autocorrelation → Schur → LAR → quantise.
    /// Per libgsm/lpc.c (§5.2.4–5.2.7).
    fn lpc_analysis(&mut self, so: &[i16; 160]) -> [u8; 8] {
        let acf = autocorrelation(so);
        let r = reflection_coefficients(&acf);
        let lar = transformation_to_log_area_ratios(&r);
        quantization_and_coding(&lar)
    }

    /// Encode one 40-sample sub-segment. Returns (Nc, bc, Mc, xmaxc, xMc[13]).
    ///
    /// Inputs the short-term residual `d[0..40]`. Produces and commits the
    /// corresponding reconstructed residual back into `dp_history` so the
    /// LTP loop sees the same past that the decoder does.
    fn encode_subframe(&mut self, d: &[i16]) -> SubFrameFields {
        debug_assert_eq!(d.len(), 40);
        // --- LTP parameters (Nc, bc) ---
        let (nc, bc) = calculation_of_the_ltp_parameters(d, &self.dp_history);

        // --- Long-term analysis filtering: dpp = MULT_R(QLB[bc], dp[k-Nc]);
        //     e = d - dpp. ---
        let bp = QLB[bc as usize];
        let mut dpp = [0i16; 40];
        let mut e_block = [0i16; 40];
        for k in 0..40 {
            // dp_history[i] corresponds to dp[-120 + i], so dp[k - Nc] is
            // dp_history[120 - Nc + k] when accessing indices -120..-1.
            // We also need dp[0..39] *during* this sub-frame: those entries
            // live at the end of dp_history once we've committed them. Since
            // Nc ∈ 40..=120, the access k - Nc is always in [-120..-1] so we
            // never reach "inside" this sub-frame.
            debug_assert!((40..=120).contains(&nc));
            let idx = 120 - nc as usize + k;
            dpp[k] = mult_r(bp, self.dp_history[idx]);
            e_block[k] = sub(d[k], dpp[k]);
        }

        // --- RPE encoding on e[0..40]. ---
        let x = weighting_filter(&e_block);
        let (mc, xm) = rpe_grid_selection(&x);
        let (xmc, mant, exp, xmaxc) = apcm_quantization(&xm);
        let xmp = apcm_inverse_quantization(&xmc, mant, exp);

        // --- Grid positioning: ep[0..40]. ---
        let mut ep = [0i16; 40];
        rpe_grid_positioning(mc, &xmp, &mut ep);

        // --- Update reconstructed short-term residual dp[0..39] = ep + dpp,
        //     and slide the history. ---
        let mut new_dp = [0i16; 40];
        for k in 0..40 {
            new_dp[k] = add(ep[k], dpp[k]);
        }
        // Slide dp_history left by 40, push new_dp onto the tail.
        for i in 0..80 {
            self.dp_history[i] = self.dp_history[i + 40];
        }
        self.dp_history[80..120].copy_from_slice(&new_dp);

        SubFrameFields {
            nc,
            bc,
            mc,
            xmaxc,
            xmc,
        }
    }
}

/// Zero-excitation MS "second half" used when flush pads an odd frame:
/// produce a fully-valid 260-bit frame using current LAR prev state but
/// silent input. Keeps the packet geometry consistent.
fn encode_silent_half(state: &mut AnalysisState) -> [u8; 33] {
    let pcm = [0i16; 160];
    let ff = state.analyse(&pcm);
    pack_standard_frame(&ff)
}

// -------- LPC analysis (Autocorrelation, Schur, LAR, quantization) --------

/// §5.2.4: autocorrelation with dynamic scaling to 9 lags, output scaled
/// by factor of 2 (matches libgsm's `L_ACF <<= 1` normalization).
fn autocorrelation(s_in: &[i16; 160]) -> [i32; 9] {
    // Find absolute max for dynamic scaling.
    let mut smax = 0i16;
    for &v in s_in.iter() {
        let t = abs16(v);
        if t > smax {
            smax = t;
        }
    }
    let scalauto = if smax == 0 {
        0
    } else {
        // 4 - norm( (longword)smax << 16 )
        let n = norm32((smax as i32) << 16);
        (4i32 - n as i32).clamp(0, 4) as u32
    };

    // Build a locally-scaled copy.
    let mut s = [0i16; 160];
    if scalauto > 0 {
        let shift = 14 - (scalauto as i32 - 1); // 16384 >> (n-1)
        let factor = (1i32 << shift) as i16;
        for i in 0..160 {
            s[i] = mult_r(s_in[i], factor);
        }
    } else {
        s[..160].copy_from_slice(s_in);
    }

    let mut acf = [0i32; 9];
    for k in 0..9 {
        let mut acc: i32 = 0;
        for i in k..160 {
            acc = acc.wrapping_add((s[i] as i32) * (s[i - k] as i32));
        }
        acf[k] = acc;
    }
    // <<1 to match L_MULT scaling.
    for k in 0..9 {
        acf[k] = acf[k].wrapping_shl(1);
    }
    acf
}

/// §5.2.5: Schur recursion → 8 reflection coefficients r[0..7], Q15.
fn reflection_coefficients(l_acf: &[i32; 9]) -> [i16; 8] {
    let mut r = [0i16; 8];
    if l_acf[0] == 0 {
        return r;
    }
    let temp = norm32(l_acf[0]);
    let mut acf = [0i16; 9];
    for i in 0..9 {
        acf[i] = sasr_i32(l_acf[i] << temp, 16) as i16;
    }
    let mut p = [0i16; 9];
    let mut k_arr = [0i16; 9];
    k_arr[1..=7].copy_from_slice(&acf[1..=7]);
    p[..=8].copy_from_slice(&acf[..=8]);
    for n in 1..=8 {
        let t = abs16(p[1]);
        if p[0] < t {
            // Zero out remaining reflection coefficients.
            for i in n..=8 {
                if i - 1 < 8 {
                    r[i - 1] = 0;
                }
            }
            return r;
        }
        let mut rn = div16(t, p[0]);
        if p[1] > 0 {
            rn = -rn;
        }
        r[n - 1] = rn;
        if n == 8 {
            return r;
        }
        // Schur update.
        let tmp = mult_r(p[1], rn);
        p[0] = add(p[0], tmp);
        for m in 1..=(8 - n) {
            let t1 = mult_r(k_arr[m], rn);
            let new_pm = add(p[m + 1], t1);
            let t2 = mult_r(p[m + 1], rn);
            let new_km = add(k_arr[m], t2);
            p[m] = new_pm;
            k_arr[m] = new_km;
        }
    }
    r
}

/// §5.2.6: reflection-coefficient → log-area-ratio transformation,
/// in-place on `r[]`.
fn transformation_to_log_area_ratios(r: &[i16; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        let t = abs16(r[i]);
        let mag = if t < 22_118 {
            t >> 1
        } else if t < 31_130 {
            t - 11_059
        } else {
            // (t - 26112) << 2 — saturated.
            let v = (t as i32 - 26_112) << 2;
            saturate_i32_to_i16(v)
        };
        out[i] = if r[i] < 0 { -mag } else { mag };
    }
    out
}

/// §5.2.7: quantise 8 LARs into `LARc[0..8]` bit-fields (unsigned with
/// bias `-MIC[i]`). Mirrors the decoder's `decode_larc` (which is the
/// inverse with bias re-added).
fn quantization_and_coding(lar: &[i16; 8]) -> [u8; 8] {
    let mut larc = [0u8; 8];
    for i in 0..8 {
        let a = A[i] as i32;
        let b = B_TIMES_TWO[i] as i32;
        let lar_i = lar[i] as i32;
        // temp = GSM_MULT(A, LAR) → (A*LAR) >> 15, saturated
        let mut temp = saturate_i32_to_i16((a * lar_i) >> 15) as i32;
        temp = l_add(temp, b);
        temp = l_add(temp, 256);
        temp = sasr_i32(temp, 9);
        let mac = MAC[i] as i32;
        let mic = MIC[i] as i32;
        let clipped = if temp > mac {
            mac - mic
        } else if temp < mic {
            0
        } else {
            temp - mic
        };
        larc[i] = (clipped & 0xFF) as u8;
    }
    larc
}

/// Mirror of [`crate::synthesis::decode_larc`] — kept local here so the
/// encoder can walk the same numerical path the decoder does.
fn decode_larc(larc: &[u8; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        let signed = larc[i] as i16 + MIC[i];
        let mut temp1 = signed - B[i];
        temp1 = shl(temp1, 10);
        out[i] = mult_r(temp1, INVA[i]);
    }
    out
}

/// Mirror of [`crate::synthesis::interpolate_larpp`].
fn interpolate_larpp(k: usize, prev: &[i16; 8], cur: &[i16; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        let p = prev[i] as i32;
        let c = cur[i] as i32;
        let v = match k {
            0 => (3 * p + c) >> 2,
            1 => (p + c) >> 1,
            2 => (p + 3 * c) >> 2,
            3 => c,
            _ => unreachable!(),
        };
        out[i] = saturate_i32_to_i16(v);
    }
    out
}

/// Mirror of [`crate::synthesis::lar_to_rp`].
fn lar_to_rp(larp: &[i16; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        let t = larp[i];
        let sign = t < 0;
        let mag = abs16(t);
        let rp = if mag < 11_059 {
            mag
        } else if mag < 20_070 {
            (mag >> 1) + 5_530
        } else {
            let m32 = mag as i32;
            ((m32 >> 2) + 11_059) as i16
        };
        let mut res = shl(rp, 4);
        if sign {
            res = sub(0, res);
        }
        out[i] = res;
    }
    out
}

// -------- Short-term analysis filter --------

/// §5.2.10. Inverse of `short_term_synthesis` in synthesis.rs — see libgsm
/// `short_term.c` for the reference formulation.
fn short_term_analysis(rp: &[i16; 8], u: &mut [i16; 8], s_in: &[i16], d_out: &mut [i16]) {
    debug_assert_eq!(s_in.len(), d_out.len());
    for k in 0..s_in.len() {
        let mut di = s_in[k];
        let mut sav = di;
        for i in 0..8 {
            let ui = u[i];
            let rpi = rp[i];
            u[i] = sav;
            let zzz = mult_r(rpi, di);
            let new_sav = add(ui, zzz);
            let zzz2 = mult_r(rpi, ui);
            di = add(di, zzz2);
            sav = new_sav;
        }
        d_out[k] = di;
    }
}

// -------- Long-term predictor analysis --------

/// §5.2.11: Calculation of LTP lag Nc and coded gain bc.
///
/// Inputs: 40-sample residual `d[]` and 120-sample history `dp[]` mapped
/// so that `dp[-120..-1]` is `dp[0..120]` (our slot 0 is libgsm's -120).
fn calculation_of_the_ltp_parameters(d: &[i16], dp: &[i16; 120]) -> (u8, u8) {
    debug_assert_eq!(d.len(), 40);
    // Find max |d[k]| for scaling.
    let mut dmax = 0i16;
    for &v in d {
        let t = abs16(v);
        if t > dmax {
            dmax = t;
        }
    }
    let scal: u32 = if dmax == 0 {
        0
    } else {
        let n = norm32((dmax as i32) << 16);
        6u32.saturating_sub(n)
    };

    // Working array wt[0..39] = SASR(d, scal).
    let mut wt = [0i16; 40];
    for k in 0..40 {
        wt[k] = sasr_i16(d[k], scal);
    }

    // Cross-correlation search across lambda = 40..=120.
    let mut l_max: i32 = 0;
    let mut nc: u8 = 40;
    for lambda in 40..=120u32 {
        let mut result: i32 = 0;
        // dp index for lag: libgsm's dp[k - lambda] where dp[-120..-1] maps
        // to our dp[0..120]. So dp[k - lambda] = our dp[120 - lambda + k]
        // since k-lambda is in [-120, -1] for this range.
        let base = 120 - lambda as usize; // dp[base + k]
        for k in 0..40 {
            result = result.wrapping_add((wt[k] as i32).wrapping_mul(dp[base + k] as i32));
        }
        if result > l_max {
            l_max = result;
            nc = lambda as u8;
        }
    }
    // L_max <<= 1 (restore L_MULT scaling).
    l_max = l_max.wrapping_shl(1);
    // Rescale: L_max = L_max >> (6 - scal)
    let shift = 6i32 - scal as i32;
    l_max = if shift >= 0 {
        l_max >> shift
    } else {
        l_max.wrapping_shl((-shift) as u32)
    };

    // Power of dp[k - Nc] residual.
    let mut l_power: i32 = 0;
    let base = 120 - nc as usize;
    for k in 0..40 {
        let t = sasr_i16(dp[base + k], 3) as i32;
        l_power = l_power.wrapping_add(t * t);
    }
    l_power = l_power.wrapping_shl(1);

    // Coding of LTP gain bc.
    if l_max <= 0 {
        return (nc, 0);
    }
    if l_max >= l_power {
        return (nc, 3);
    }
    let temp = norm32(l_power);
    let r = sasr_i32(l_max << temp, 16) as i16;
    let s = sasr_i32(l_power << temp, 16) as i16;

    let mut bc: u8 = 3;
    for b in 0..=2 {
        if r <= mult(s, DLB[b]) {
            bc = b as u8;
            break;
        }
    }
    (nc, bc)
}

// -------- RPE encoding --------

/// §5.2.13: block filtering / weighting.
fn weighting_filter(e_in: &[i16; 40]) -> [i16; 40] {
    // wt[k] for k in 0..50 comes from e[k - 5], with 0s outside 5..45.
    let mut x = [0i16; 40];
    for k in 0..40 {
        // Initial L_result = 8192 >> 1 = 4096 (the rounding constant).
        let mut acc: i32 = 4096;
        for i in 0..=10 {
            let h = H[i] as i32;
            if h == 0 {
                continue;
            }
            let idx = (k + i) as i32 - 5;
            let wt_val = if (0..40).contains(&idx) {
                e_in[idx as usize] as i32
            } else {
                0
            };
            acc = acc.wrapping_add(wt_val * h);
        }
        // SASR by 13 (= 14 - 1 compensation for the lost <<1).
        let shifted = sasr_i32(acc, 13);
        x[k] = saturate_i32_to_i16(shifted);
    }
    x
}

/// §5.2.14: pick best RPE grid (m ∈ 0..=3) by max sum-of-squares,
/// downsampled by factor 3. Returns (Mc, xM[13]).
fn rpe_grid_selection(x: &[i16; 40]) -> (u8, [i16; 13]) {
    let mut em: i32 = 0;
    let mut mc: u8 = 0;
    for m in 0..=3u8 {
        let mut acc: i32 = 0;
        for i in 0..=12 {
            let idx = m as usize + 3 * i;
            let t = sasr_i16(x[idx], 2) as i32;
            acc = acc.wrapping_add(t * t);
        }
        let acc = acc.wrapping_shl(1);
        if m == 0 {
            em = acc;
            mc = 0;
        } else if acc > em {
            em = acc;
            mc = m;
        }
    }
    let mut xm = [0i16; 13];
    for i in 0..=12 {
        xm[i] = x[mc as usize + 3 * i];
    }
    (mc, xm)
}

/// §5.2.14 companion: compute exp/mant from xmaxc. Mirrors libgsm's
/// `APCM_quantization_xmaxc_to_exp_mant` used in the inverse path.
fn xmaxc_to_exp_mant(xmaxc: u8) -> (i16, i16) {
    let mut exp: i16 = 0;
    if xmaxc > 15 {
        exp = sasr_i16(xmaxc as i16, 3) - 1;
    }
    let mut mant = xmaxc as i16 - (exp << 3);
    if mant == 0 {
        exp = -4;
        mant = 7;
    } else {
        while mant <= 7 {
            mant = (mant << 1) | 1;
            exp -= 1;
        }
        mant -= 8;
    }
    (exp, mant)
}

/// §5.2.15: APCM quantisation of the 13 RPE samples.
fn apcm_quantization(xm: &[i16; 13]) -> ([u8; 13], i16, i16, u8) {
    // Find xmax = max |xm[i]|.
    let mut xmax = 0i16;
    for &v in xm {
        let t = abs16(v);
        if t > xmax {
            xmax = t;
        }
    }
    // Exponent search: exp = 0; temp = SASR(xmax, 9);
    //   for i in 0..=5: itest |= (temp <= 0); temp = SASR(temp, 1);
    //                   if !itest: exp++
    let mut exp: i16 = 0;
    let mut temp = sasr_i16(xmax, 9);
    let mut itest = false;
    for _ in 0..=5 {
        if temp <= 0 {
            itest = true;
        }
        temp = sasr_i16(temp, 1);
        if !itest {
            exp += 1;
        }
    }
    let sh = (exp + 5).clamp(0, 15) as u32;
    let shifted = sasr_i16(xmax, sh);
    let xmaxc = add(shifted, exp << 3) as u8 & 0x3F;

    // Recompute exp/mant from xmaxc (matches decoder).
    let (exp2, mant2) = xmaxc_to_exp_mant(xmaxc);
    let temp1 = 6 - exp2;
    let temp2 = NRFAC[mant2 as usize];

    let mut xmc = [0u8; 13];
    for i in 0..13 {
        let shl_v = if (0..16).contains(&temp1) {
            (xm[i] as i32) << temp1
        } else {
            0
        };
        let shl_i16 = saturate_i32_to_i16(shl_v);
        let m = mult(shl_i16, temp2);
        let mut t = sasr_i16(m, 12);
        t += 4; // bias to positive 0..=7
                // Clamp to 3-bit range (spec guarantees this, but defensive).
        xmc[i] = (t.clamp(0, 7)) as u8;
    }
    (xmc, mant2, exp2, xmaxc)
}

/// §5.2.16: inverse APCM quantisation producing xMp[0..12]. Identical to
/// the decoder's dequantisation path — kept here so the analysis loop
/// sees exactly the quantised pulse values the decoder will see.
fn apcm_inverse_quantization(xmc: &[u8; 13], mant: i16, exp: i16) -> [i16; 13] {
    let temp1 = FAC[mant as usize];
    let temp2 = 6 - exp;
    let temp3 = if (1..16).contains(&temp2) {
        1i16 << (temp2 - 1)
    } else {
        0
    };
    let mut xmp = [0i16; 13];
    for i in 0..13 {
        let s = (xmc[i] as i16) << 1; // 0..=14
        let signed = s - 7; // -7..=7
        let shifted = (signed as i32) << 12; // 16-bit signed
        let shifted = saturate_i32_to_i16(shifted);
        let m = mult_r(temp1, shifted);
        let summed = add(m, temp3);
        // gsm_asr by temp2.
        xmp[i] = if temp2 >= 0 {
            sasr_i16(summed, temp2 as u32)
        } else if temp2 > -16 {
            shl(summed, (-temp2) as u32)
        } else {
            0
        };
    }
    xmp
}

/// §5.2.17: place xMp on grid Mc in a 40-sample zero-filled block ep[].
fn rpe_grid_positioning(mc: u8, xmp: &[i16; 13], ep: &mut [i16; 40]) {
    ep.fill(0);
    let m = (mc & 3) as usize;
    for i in 0..=12 {
        let pos = m + 3 * i;
        if pos < 40 {
            ep[pos] = xmp[i];
        }
    }
}

// -------- Frame packing --------

/// Pack the analysis fields into the canonical 33-byte `0xD`-prefixed
/// GSM Full Rate payload. Inverse of [`crate::frame::parse_frame`].
fn pack_standard_frame(ff: &FrameFields) -> [u8; 33] {
    let mut w = BitWriter::new();
    w.write(0xD, 4);
    pack_260_bit_payload(&mut w, ff);
    while w.data.len() < FRAME_SIZE {
        w.data.push(0);
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(&w.data[..FRAME_SIZE]);
    out
}

/// Pack two standard 260-bit payloads back-to-back into the 65-byte
/// Microsoft WAV-49 framing. No magic; 520 bits + 8 padding.
fn pack_ms_pair(first: &[u8; 33], second: &[u8; 33]) -> Vec<u8> {
    // Re-unpack each frame's fields (cheap — 260 bits) and repack without
    // the 4-bit magic, back-to-back. Since `pack_standard_frame` already
    // does a full MSB-first write, the simplest path is to parse-through.
    let f0 = crate::frame::parse_frame(first).expect("self-produced frame parses");
    let f1 = crate::frame::parse_frame(second).expect("self-produced frame parses");
    let mut w = BitWriter::new();
    pack_260_bit_payload(&mut w, &gsm_frame_to_fields(&f0));
    pack_260_bit_payload(&mut w, &gsm_frame_to_fields(&f1));
    while w.data.len() < MS_FRAME_SIZE {
        w.data.push(0);
    }
    w.data.truncate(MS_FRAME_SIZE);
    w.data
}

fn gsm_frame_to_fields(f: &crate::frame::GsmFrame) -> FrameFields {
    FrameFields {
        larc: f.larc,
        nc: [f.sub[0].nc, f.sub[1].nc, f.sub[2].nc, f.sub[3].nc],
        bc: [f.sub[0].bc, f.sub[1].bc, f.sub[2].bc, f.sub[3].bc],
        mc: [f.sub[0].mc, f.sub[1].mc, f.sub[2].mc, f.sub[3].mc],
        xmaxc: [
            f.sub[0].xmaxc,
            f.sub[1].xmaxc,
            f.sub[2].xmaxc,
            f.sub[3].xmaxc,
        ],
        xmc: [f.sub[0].xmc, f.sub[1].xmc, f.sub[2].xmc, f.sub[3].xmc],
    }
}

fn pack_260_bit_payload(w: &mut BitWriter, ff: &FrameFields) {
    const LAR_BITS: [u32; 8] = [6, 6, 5, 5, 4, 4, 3, 3];
    for i in 0..8 {
        w.write(ff.larc[i] as u16, LAR_BITS[i]);
    }
    for s in 0..4 {
        w.write(ff.nc[s] as u16, 7);
        w.write(ff.bc[s] as u16, 2);
        w.write(ff.mc[s] as u16, 2);
        w.write(ff.xmaxc[s] as u16, 6);
        for i in 0..13 {
            w.write(ff.xmc[s][i] as u16, 3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::parse_frame;

    fn make_params(id: &str) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(id));
        p.sample_rate = Some(8_000);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p
    }

    #[test]
    fn encoder_rejects_wrong_sample_rate() {
        let mut p = make_params(crate::decoder::CODEC_ID_STANDARD);
        p.sample_rate = Some(16_000);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn encoder_rejects_stereo() {
        let mut p = make_params(crate::decoder::CODEC_ID_STANDARD);
        p.channels = Some(2);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn silence_encodes_to_33_bytes_with_magic() {
        let mut enc = make_encoder(&make_params(crate::decoder::CODEC_ID_STANDARD)).unwrap();
        let pcm = vec![0u8; 160 * 2];
        let f = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 8_000,
            samples: 160,
            pts: Some(0),
            time_base: TimeBase::new(1, 8_000),
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(f)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.data.len(), FRAME_SIZE);
        assert_eq!((pkt.data[0] >> 4) & 0x0F, 0xD);
        // And the produced frame parses cleanly.
        let _parsed = parse_frame(&pkt.data).unwrap();
    }

    #[test]
    fn sine_wave_produces_nontrivial_ltp_lags() {
        // 200 Hz sine (period = 40 samples) should pick a consistent LTP
        // lag once past the first frame's warmup. We're not asserting a
        // specific value — just that the encoder exercises the full path.
        let mut enc = make_encoder(&make_params(crate::decoder::CODEC_ID_STANDARD)).unwrap();
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let mut bytes = Vec::with_capacity(160 * 2 * 4);
        for n in 0..(160 * 4) {
            let t = n as f32 / 8_000.0;
            let v = (two_pi * 200.0 * t).sin() * 20_000.0;
            let s = v as i16;
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let f = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 8_000,
            samples: (160 * 4) as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 8_000),
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(f)).unwrap();
        enc.flush().unwrap();
        let mut saw_ltp = false;
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), FRAME_SIZE);
            let parsed = parse_frame(&pkt.data).unwrap();
            for s in &parsed.sub {
                if s.xmaxc > 0 {
                    saw_ltp = true;
                }
                assert!(s.nc <= 127);
                assert!(s.bc <= 3);
                assert!(s.mc <= 3);
                for p in s.xmc {
                    assert!(p <= 7);
                }
            }
        }
        assert!(saw_ltp, "encoded silent output for a sine wave");
    }

    #[test]
    fn ms_variant_emits_65_byte_packets() {
        let mut enc = make_encoder(&make_params(crate::decoder::CODEC_ID_MS)).unwrap();
        let mut bytes = Vec::with_capacity(160 * 2 * 2);
        for n in 0..320 {
            let v = (n as i16) * 100;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 8_000,
            samples: 320,
            pts: Some(0),
            time_base: TimeBase::new(1, 8_000),
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(f)).unwrap();
        enc.flush().unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.data.len(), MS_FRAME_SIZE);
    }

    #[test]
    fn xmaxc_to_exp_mant_matches_decoder_table() {
        // Spot-check a few values the decoder's rpe_decode handles.
        let (exp, mant) = xmaxc_to_exp_mant(0);
        assert_eq!(exp, -4);
        assert_eq!(mant, 7);
        let (exp, mant) = xmaxc_to_exp_mant(16);
        // xmaxc=16 -> exp=1, mant = 16 - 8 = 8 → normalization keeps mant
        // in 0..7 by stripping the implicit leading bit (mant -= 8).
        assert_eq!(exp, 1);
        assert_eq!(mant, 0);
    }
}
