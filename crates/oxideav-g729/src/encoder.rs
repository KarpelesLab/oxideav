//! ITU-T G.729 (CS-ACELP, 8 kbit/s) encoder — analysis pipeline symmetric
//! to the in-tree decoder.
//!
//! # Pipeline
//!
//! For every 10 ms frame (80 S16 mono samples at 8 kHz):
//!
//! ```text
//!   PCM s16 → windowed LPC analysis (Levinson-Durbin, order 10)
//!           → LPC → LSP → split-VQ quantisation via LSPCB1 / LSPCB2
//!                              (the same tables the decoder reads back)
//!           → subframe loop (2 × 5 ms subframes, 40 samples each)
//!               - target residual = filter-weighted error signal
//!               - open-loop pitch search on weighted signal
//!               - closed-loop refinement with fractional 1/3 lag
//!               - ACELP 4-pulse fixed-codebook search (focused per track)
//!               - gain analysis + two-stage VQ via GBK1 / GBK2
//!           → bit-pack into 80-bit / 10-byte packet, ITU-T Table 8 order
//! ```
//!
//! # Symmetry with the decoder
//!
//! The encoder reuses every static table the decoder exposes:
//!   - [`LSPCB1_Q13`] / [`LSPCB2_Q13`] — LSP split VQ codebooks
//!   - [`FG_Q15`] / [`FG_SUM_Q15`] — MA-4 predictor coefficients
//!   - [`GBK1`] / [`GBK2`] — two-stage gain codebook
//!
//! The decoder module notes that the procedural rows of `LSPCB1_Q13` and
//! the small first-cut gain tables are **not** spec-exact. The encoder
//! inherits that caveat by design: the goal is that a bitstream produced
//! by this encoder round-trips cleanly through the in-tree decoder, not
//! that it is playable by an external G.729 implementation. Swapping the
//! tables for the ITU verbatim values later is a drop-in replacement
//! with no encoder logic changes.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::pitch_parity;
use crate::lpc::{interpolate_lsp, lsp_to_lpc, LpcPredictorState, MA_HISTORY};
use crate::lsp_tables::{FG_Q15, FG_SUM_Q15, LSPCB1_Q13, LSPCB2_Q13, MA_NP, M_HALF, NC0, NC1};
use crate::synthesis::{
    adaptive_codebook_excitation, fixed_codebook_excitation, SynthesisState, EXC_HIST, GBK1, GBK2,
};
use crate::{
    CODEC_ID_STR, FRAME_BYTES, FRAME_SAMPLES, LPC_ORDER, SAMPLE_RATE, SUBFRAMES_PER_FRAME,
    SUBFRAME_SAMPLES,
};

/// Build a G.729 encoder. Accepts 8 kHz mono S16 input.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "G.729 encoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.729 encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "G.729 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "G.729 encoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE);
    output.bit_rate = Some(8_000);

    Ok(Box::new(G729Encoder::new(output)))
}

struct G729Encoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    state: EncoderState,
    pcm_queue: Vec<i16>,
    pending: VecDeque<Packet>,
    frame_index: u64,
    eof: bool,
}

impl G729Encoder {
    fn new(output_params: CodecParameters) -> Self {
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
            state: EncoderState::new(),
            pcm_queue: Vec::new(),
            pending: VecDeque::new(),
            frame_index: 0,
            eof: false,
        }
    }

    fn drain(&mut self, final_flush: bool) {
        while self.pcm_queue.len() >= FRAME_SAMPLES {
            let mut pcm = [0i16; FRAME_SAMPLES];
            pcm.copy_from_slice(&self.pcm_queue[..FRAME_SAMPLES]);
            self.pcm_queue.drain(..FRAME_SAMPLES);
            self.emit_frame(&pcm);
        }
        if final_flush && !self.pcm_queue.is_empty() {
            let mut pcm = [0i16; FRAME_SAMPLES];
            for (i, &s) in self.pcm_queue.iter().enumerate() {
                pcm[i] = s;
            }
            self.pcm_queue.clear();
            self.emit_frame(&pcm);
        }
    }

    fn emit_frame(&mut self, pcm: &[i16; FRAME_SAMPLES]) {
        let idx = self.frame_index;
        self.frame_index += 1;
        let params = self.state.analyse(pcm);
        let bytes = pack_frame(&params);
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = Some(idx as i64 * FRAME_SAMPLES as i64);
        pkt.dts = pkt.pts;
        pkt.duration = Some(FRAME_SAMPLES as i64);
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
    }
}

impl Encoder for G729Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let af = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::invalid("G.729 encoder: audio frames only")),
        };
        if af.channels != 1 || af.sample_rate != SAMPLE_RATE {
            return Err(Error::invalid("G.729 encoder: input must be mono, 8000 Hz"));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "G.729 encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("G.729 encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("G.729 encoder: odd byte count"));
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
// Frame-level parameters (mirror `FrameParams` from the decoder bitreader).
// =========================================================================

/// Per-frame field values produced by the analyser; fed into `pack_frame`.
#[derive(Clone, Copy, Debug, Default)]
struct EncodedFrame {
    l0: u8,
    l1: u8,
    l2: u8,
    l3: u8,
    p1: u8,
    p0: u8,
    c1: u16,
    s1: u8,
    ga1: u8,
    gb1: u8,
    p2: u8,
    c2: u16,
    s2: u8,
    ga2: u8,
    gb2: u8,
}

/// Pack an [`EncodedFrame`] into the 80-bit / 10-byte ITU-T Table 8 layout
/// (MSB-first inside each byte, fields transmitted L0..GB2).
fn pack_frame(fp: &EncodedFrame) -> Vec<u8> {
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

// =========================================================================
// Analysis state
// =========================================================================

struct EncoderState {
    /// LSP predictor state (same type as the decoder's — so we quantise the
    /// LSP residual identically and can feed the same table back).
    lpc: LpcPredictorState,
    /// Synthesis state used for analysis-by-synthesis: the excitation
    /// history drives the adaptive codebook and the pitch sharpening
    /// reconstruction, so we mimic the decoder exactly during encode.
    syn: SynthesisState,
    /// Pre-emphasis filter memory (simple one-pole HPF).
    preemph_prev: f32,
    /// Previous-frame unquantised LSP, for open-loop pitch smoothing.
    #[allow(dead_code)]
    prev_lsp_raw: [f32; LPC_ORDER],
}

impl EncoderState {
    fn new() -> Self {
        let lpc = LpcPredictorState::new();
        let prev_lsp_raw = lpc.lsp_prev;
        Self {
            lpc,
            syn: SynthesisState::new(),
            preemph_prev: 0.0,
            prev_lsp_raw,
        }
    }

    /// Analyse a single 80-sample frame and produce the 15 bit-fields.
    fn analyse(&mut self, pcm: &[i16; FRAME_SAMPLES]) -> EncodedFrame {
        // -------- 1. Pre-process (HPF + normalise) --------
        let mut sig = [0.0f32; FRAME_SAMPLES];
        let mut prev = self.preemph_prev;
        for i in 0..FRAME_SAMPLES {
            let x = pcm[i] as f32;
            let y = x - 0.46 * prev; // mild HPF; G.729 uses a 140 Hz HPF
            prev = x;
            sig[i] = y;
        }
        self.preemph_prev = prev;

        // -------- 2. LPC analysis on a 240-sample window --------
        // The spec uses a 240-sample asymmetric window; we approximate
        // with a Hamming window spanning the current frame plus 80 lookback
        // and 80 look-ahead (we only have the current frame, so we mirror
        // the ends — good enough for a first-cut encoder).
        let a = lpc_analysis(&sig);

        // -------- 3. LPC → LSP (cosine domain) --------
        let lsp_unq = lpc_to_lsp(&a);

        // -------- 4. Quantise LSPs using the decoder's codebooks --------
        let (l0, l1, l2, l3, lsp_q) = quantise_lsp_with_predictor(&mut self.lpc, &lsp_unq);

        // -------- 5. Per-subframe LPC via LSP interpolation --------
        let lsp_sf0 = interpolate_lsp(&self.lpc.lsp_prev, &lsp_q, 0.5);
        let lsp_sf1 = lsp_q;
        let a_sf = [lsp_to_lpc(&lsp_sf0), lsp_to_lpc(&lsp_sf1)];

        // -------- 6. Subframe analysis --------
        let mut out = EncodedFrame::default();
        out.l0 = l0;
        out.l1 = l1;
        out.l2 = l2;
        out.l3 = l3;

        // Target signal: residual after LPC inverse filtering.
        //   r[n] = sum_{k=0..10} a[k] * s[n-k]
        // Using the quantised per-subframe A(z).
        let residual = lpc_residual(&sig, &a_sf);

        let mut first_p1_int: usize = 40;
        for sf in 0..SUBFRAMES_PER_FRAME {
            let off = sf * SUBFRAME_SAMPLES;
            let mut target = [0.0f32; SUBFRAME_SAMPLES];
            target.copy_from_slice(&residual[off..off + SUBFRAME_SAMPLES]);

            // ---- 6a. Pitch search ----
            let (t_int, t_frac) = pitch_search(
                &self.syn.exc,
                &target,
                if sf == 0 { None } else { Some(first_p1_int) },
            );
            if sf == 0 {
                first_p1_int = t_int;
                let p1 = encode_pitch_p1(t_int, t_frac);
                out.p1 = p1;
                out.p0 = pitch_parity(p1);
            } else {
                out.p2 = encode_pitch_p2(t_int, t_frac, first_p1_int);
            }

            // Adaptive-codebook excitation vector (unity-gain).
            let mut ac = [0.0f32; SUBFRAME_SAMPLES];
            adaptive_codebook_excitation(&self.syn.exc, t_int, t_frac, &mut ac);

            // ---- 6b. Adaptive-codebook gain (unconstrained LS) ----
            let g_p = gain_ls(&ac, &target).clamp(0.0, 1.2);

            // Compute residual after ACB contribution.
            let mut target2 = [0.0f32; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                target2[n] = target[n] - g_p * ac[n];
            }

            // ---- 6c. Fixed-codebook (ACELP 4-pulse) search ----
            let (c_idx, s_idx) = fixed_codebook_search(&target2);
            let mut fc = [0.0f32; SUBFRAME_SAMPLES];
            fixed_codebook_excitation(c_idx, s_idx, &mut fc);

            // ---- 6d. Fixed-codebook gain ----
            let g_c_raw = gain_ls(&fc, &target2).clamp(0.0, 32.0);

            // ---- 6e. Two-stage gain VQ over GBK1 / GBK2 ----
            // Search for the index pair that best matches (g_p, g_c_raw).
            let (ga, gb) = quantise_gain(g_p, g_c_raw);

            if sf == 0 {
                out.c1 = c_idx;
                out.s1 = s_idx;
                out.ga1 = ga;
                out.gb1 = gb;
            } else {
                out.c2 = c_idx;
                out.s2 = s_idx;
                out.ga2 = ga;
                out.gb2 = gb;
            }

            // ---- 6f. Update excitation history with the QUANTISED excitation
            //        so the next subframe's adaptive-codebook search sees the
            //        same history the decoder will. ----
            let (g_p_q, gamma_q) = dequantise_gain(ga, gb);
            // Use the raw fixed-codebook gain * gamma_q as the effective g_c;
            // matches the decoder's analysis-by-synthesis view closely enough
            // for our purposes.
            let g_c_q = gamma_q * g_c_raw.max(0.25);
            let mut excitation = [0.0f32; SUBFRAME_SAMPLES];
            for n in 0..SUBFRAME_SAMPLES {
                excitation[n] = g_p_q * ac[n] + g_c_q * fc[n];
            }
            push_excitation(&mut self.syn.exc, &excitation);
        }

        // -------- 7. Roll LSP predictor state --------
        self.lpc.lsp_prev = lsp_q;
        self.lpc.a = a_sf[1];
        out
    }
}

/// Slide `exc` left by `SUBFRAME_SAMPLES` and append `sub` at the tail.
fn push_excitation(exc: &mut [f32; EXC_HIST], sub: &[f32; SUBFRAME_SAMPLES]) {
    for i in 0..EXC_HIST - SUBFRAME_SAMPLES {
        exc[i] = exc[i + SUBFRAME_SAMPLES];
    }
    for i in 0..SUBFRAME_SAMPLES {
        exc[EXC_HIST - SUBFRAME_SAMPLES + i] = sub[i];
    }
}

// =========================================================================
// LPC analysis
// =========================================================================

/// Windowed autocorrelation + Levinson-Durbin recursion → LPC[0..=10].
fn lpc_analysis(sig: &[f32; FRAME_SAMPLES]) -> [f32; LPC_ORDER + 1] {
    // Hamming window over 80 samples (cheap replacement for the spec's
    // 240-sample asymmetric window).
    let mut w = [0.0f32; FRAME_SAMPLES];
    let n = FRAME_SAMPLES as f32;
    for i in 0..FRAME_SAMPLES {
        let phase = 2.0 * core::f32::consts::PI * (i as f32) / (n - 1.0);
        w[i] = sig[i] * (0.54 - 0.46 * phase.cos());
    }
    // Autocorrelation r[0..=10].
    let mut r = [0.0f64; LPC_ORDER + 1];
    for k in 0..=LPC_ORDER {
        let mut acc = 0.0f64;
        for i in k..FRAME_SAMPLES {
            acc += (w[i] as f64) * (w[i - k] as f64);
        }
        r[k] = acc;
    }
    // Small white-noise correction + 60 Hz bandwidth lag window.
    r[0] *= 1.0001;
    for k in 1..=LPC_ORDER {
        let f = 2.0 * core::f64::consts::PI * 60.0 * (k as f64) / (SAMPLE_RATE as f64);
        let lag = (-0.5 * f * f).exp();
        r[k] *= lag;
    }
    if r[0] <= 0.0 {
        return default_a();
    }
    // Levinson-Durbin.
    let mut a = [0.0f64; LPC_ORDER + 1];
    let mut a_prev = [0.0f64; LPC_ORDER + 1];
    a[0] = 1.0;
    a_prev[0] = 1.0;
    let mut e = r[0];
    for i in 1..=LPC_ORDER {
        let mut acc = r[i];
        for j in 1..i {
            acc += a_prev[j] * r[i - j];
        }
        let k_refl = -acc / e;
        a[i] = k_refl;
        for j in 1..i {
            a[j] = a_prev[j] + k_refl * a_prev[i - j];
        }
        e *= 1.0 - k_refl * k_refl;
        if e <= 1e-18 {
            return default_a();
        }
        a_prev.copy_from_slice(&a);
    }
    let mut out = [0.0f32; LPC_ORDER + 1];
    for i in 0..=LPC_ORDER {
        out[i] = a[i] as f32;
    }
    out[0] = 1.0;
    out
}

fn default_a() -> [f32; LPC_ORDER + 1] {
    let mut a = [0.0f32; LPC_ORDER + 1];
    a[0] = 1.0;
    a
}

/// LPC direct-form → LSP (cosine domain) via Chebyshev root-finding on
/// F1(z) = A(z) + z^-(p+1) * A(z^-1) and F2(z) = A(z) - z^-(p+1) * A(z^-1).
fn lpc_to_lsp(a: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER] {
    let p = LPC_ORDER;
    // Build f1, f2 of degree p/2 after factoring out (1 ± z^-1).
    let mut f1 = [0.0f32; LPC_ORDER / 2 + 1];
    let mut f2 = [0.0f32; LPC_ORDER / 2 + 1];
    f1[0] = 1.0;
    f2[0] = 1.0;
    let mut prev_f1 = 0.0f32;
    let mut prev_f2 = 0.0f32;
    for i in 1..=p / 2 {
        let ai = a[i];
        let api = a[p + 1 - i];
        f1[i] = ai + api - prev_f1;
        f2[i] = ai - api + prev_f2;
        prev_f1 = f1[i];
        prev_f2 = f2[i];
    }
    let r1 = cheby_roots(&f1);
    let r2 = cheby_roots(&f2);
    let mut lsp = [0.0f32; LPC_ORDER];
    // Interleave roots; fall back to uniform spread if the search ran short.
    let uni = |k: usize| -> f32 {
        let step = core::f32::consts::PI / (LPC_ORDER as f32 + 1.0);
        (step * (k as f32 + 1.0)).cos()
    };
    for k in 0..LPC_ORDER {
        if k % 2 == 0 {
            lsp[k] = r1.get(k / 2).copied().unwrap_or_else(|| uni(k));
        } else {
            lsp[k] = r2.get(k / 2).copied().unwrap_or_else(|| uni(k));
        }
    }
    // Enforce strictly-decreasing cos domain.
    for k in 1..LPC_ORDER {
        if lsp[k] >= lsp[k - 1] - 1e-4 {
            lsp[k] = lsp[k - 1] - 1e-3;
        }
    }
    // Clamp to (-1, 1) with small margin.
    for k in 0..LPC_ORDER {
        if lsp[k] > 0.9995 {
            lsp[k] = 0.9995;
        }
        if lsp[k] < -0.9995 {
            lsp[k] = -0.9995;
        }
    }
    lsp
}

/// Find real roots on `[-1, 1]` of a Chebyshev-expanded polynomial via
/// grid-bracket + bisection. Returns roots in decreasing x-order (i.e.
/// increasing omega).
fn cheby_roots(coeffs: &[f32]) -> Vec<f32> {
    let deg = coeffs.len().saturating_sub(1);
    let eval = |x: f32| -> f32 {
        // Clenshaw recurrence for Chebyshev series of the first kind.
        let mut b2 = 0.0f32;
        let mut b1 = 0.0f32;
        for k in (1..=deg).rev() {
            let b0 = 2.0 * x * b1 - b2 + coeffs[k];
            b2 = b1;
            b1 = b0;
        }
        x * b1 - b2 + coeffs[0]
    };
    const GRID: usize = 200;
    let mut roots = Vec::with_capacity(deg);
    let mut prev_x = 1.0f32;
    let mut prev_y = eval(prev_x);
    for i in 1..=GRID {
        let x = 1.0 - 2.0 * (i as f32 / GRID as f32);
        let y = eval(x);
        if prev_y * y < 0.0 {
            // Bisect on [x, prev_x].
            let mut lo = x;
            let mut hi = prev_x;
            let mut flo = y;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let fm = eval(mid);
                if fm * flo < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push(0.5 * (lo + hi));
            if roots.len() == deg {
                break;
            }
        }
        prev_x = x;
        prev_y = y;
    }
    roots
}

// =========================================================================
// LSP quantisation (symmetric with the decoder's MA-4 + LSPCB1/LSPCB2)
// =========================================================================

/// Quantise `lsp_unq` (cosine-domain LSPs) using the decoder's
/// `LSPCB1_Q13` (L1, 128 entries) + `LSPCB2_Q13` (L2/L3, 32 entries).
/// Also chooses the MA-predictor switch `L0 ∈ {0, 1}`.
///
/// Returns `(L0, L1, L2, L3, quantised LSP in cosine domain)`.
///
/// Implementation:
/// - Convert `lsp_unq` to LSF (radians), then to Q13.
/// - Subtract the MA-predictor expectation to get the *residual* that
///   `LSPCB1 + LSPCB2` needs to represent.
/// - Search `LSPCB1_Q13[l1] + LSPCB2_Q13[l2 (low)] + LSPCB2_Q13[l3 (high)]`
///   for the minimum L2 distance to the target residual. The low and
///   high halves are searched independently (split VQ).
/// - Do the search twice (predictor 0 and predictor 1) and keep the
///   smaller reconstruction error.
fn quantise_lsp_with_predictor(
    state: &mut LpcPredictorState,
    lsp_unq: &[f32; LPC_ORDER],
) -> (u8, u8, u8, u8, [f32; LPC_ORDER]) {
    // Target LSF in radians, then in Q13 (f32 for arithmetic).
    let mut lsf_target = [0.0f32; LPC_ORDER];
    for k in 0..LPC_ORDER {
        lsf_target[k] = lsp_unq[k].clamp(-1.0, 1.0).acos();
    }
    // Enforce strictly-increasing LSF + minimum spacing.
    for k in 1..LPC_ORDER {
        if lsf_target[k] <= lsf_target[k - 1] + 1e-3 {
            lsf_target[k] = lsf_target[k - 1] + 1e-3;
        }
    }
    let mut target_q13 = [0.0f32; LPC_ORDER];
    for k in 0..LPC_ORDER {
        target_q13[k] = lsf_target[k] * 8192.0;
    }

    // Try predictor 0 and predictor 1.
    let mut best_l0 = 0u8;
    let mut best_l1 = 0u8;
    let mut best_l2 = 0u8;
    let mut best_l3 = 0u8;
    let mut best_err = f32::INFINITY;
    let mut best_resid = [0i16; LPC_ORDER];

    for predictor in 0..2usize {
        // Expected predictor contribution (same formula as decoder).
        let fg = &FG_Q15[predictor];
        let fg_sum = &FG_SUM_Q15[predictor];
        // predicted_lsf_q13[j] = (sum_k fg[k][j] * prev_res_q13[k][j]) / 2^15
        let mut predicted = [0.0f32; LPC_ORDER];
        for j in 0..LPC_ORDER {
            let mut acc = 0.0f32;
            for k in 0..MA_NP {
                acc += (fg[k][j] as f32) * (state.freq_res_q13[k][j] as f32);
            }
            predicted[j] = acc / 32768.0;
        }
        // Target residual in Q13: fg_sum[j] * residual[j] = target_q13[j] - predicted[j]
        let mut resid_target = [0.0f32; LPC_ORDER];
        for j in 0..LPC_ORDER {
            let denom = (fg_sum[j] as f32) / 32768.0;
            if denom.abs() < 1e-6 {
                resid_target[j] = 0.0;
            } else {
                resid_target[j] = (target_q13[j] - predicted[j]) / denom;
            }
        }

        // Split-VQ search: the residual is LPCB1 row + concatenation of two
        // LPCB2 rows (low half + high half).
        // Best L1: search 128 entries minimising distance to low+high halves.
        let mut best_l1_for = 0u8;
        let mut best_l2_for = 0u8;
        let mut best_l3_for = 0u8;
        let mut best_err_for = f32::INFINITY;
        let mut best_resid_for = [0i16; LPC_ORDER];

        for l1 in 0..NC0 {
            let cb1 = &LSPCB1_Q13[l1];
            // Best L2 (low half, j=0..5).
            let mut b2 = 0usize;
            let mut b2_err = f32::INFINITY;
            for l2 in 0..NC1 {
                let cb2 = &LSPCB2_Q13[l2];
                let mut err = 0.0f32;
                for j in 0..M_HALF {
                    let recon = (cb1[j] as f32) + (cb2[j] as f32);
                    let d = recon - resid_target[j];
                    err += d * d;
                }
                if err < b2_err {
                    b2_err = err;
                    b2 = l2;
                }
            }
            // Best L3 (high half, j=5..10).
            let mut b3 = 0usize;
            let mut b3_err = f32::INFINITY;
            for l3 in 0..NC1 {
                let cb2 = &LSPCB2_Q13[l3];
                let mut err = 0.0f32;
                for j in 0..M_HALF {
                    let recon = (cb1[j + M_HALF] as f32) + (cb2[j] as f32);
                    let d = recon - resid_target[j + M_HALF];
                    err += d * d;
                }
                if err < b3_err {
                    b3_err = err;
                    b3 = l3;
                }
            }
            let err = b2_err + b3_err;
            if err < best_err_for {
                best_err_for = err;
                best_l1_for = l1 as u8;
                best_l2_for = b2 as u8;
                best_l3_for = b3 as u8;
                // Assemble the residual in Q13 using the selected entries.
                let cb2_lo = &LSPCB2_Q13[b2];
                let cb2_hi = &LSPCB2_Q13[b3];
                for j in 0..M_HALF {
                    best_resid_for[j] = cb1[j].saturating_add(cb2_lo[j]);
                    best_resid_for[j + M_HALF] = cb1[j + M_HALF].saturating_add(cb2_hi[j]);
                }
            }
        }

        if best_err_for < best_err {
            best_err = best_err_for;
            best_l0 = predictor as u8;
            best_l1 = best_l1_for;
            best_l2 = best_l2_for;
            best_l3 = best_l3_for;
            best_resid = best_resid_for;
        }
    }

    // Reconstruct the quantised LSF vector using the chosen predictor and
    // entries — mirror of the decoder's `decode_lsp` logic.
    let predictor = best_l0 as usize;
    let fg = &FG_Q15[predictor];
    let fg_sum = &FG_SUM_Q15[predictor];
    let mut lsf_q13_f = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        let mut acc: f32 = (fg_sum[j] as f32) * (best_resid[j] as f32);
        for k in 0..MA_HISTORY {
            acc += (fg[k][j] as f32) * (state.freq_res_q13[k][j] as f32);
        }
        lsf_q13_f[j] = acc / 32768.0;
    }
    // Push the chosen residual onto the predictor history (same as decoder).
    state.push_residual(best_resid);

    // Convert Q13 LSF → radians → cosine domain with spacing safeguards.
    let pi = core::f32::consts::PI;
    let eps = 0.0012f32;
    let mut lsf = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        lsf[j] = lsf_q13_f[j] / 8192.0;
    }
    if lsf[0] < eps {
        lsf[0] = eps;
    }
    for j in 1..LPC_ORDER {
        if lsf[j] < lsf[j - 1] + eps {
            lsf[j] = lsf[j - 1] + eps;
        }
    }
    if lsf[LPC_ORDER - 1] > pi - eps {
        lsf[LPC_ORDER - 1] = pi - eps;
        for j in (0..LPC_ORDER - 1).rev() {
            if lsf[j] > lsf[j + 1] - eps {
                lsf[j] = lsf[j + 1] - eps;
            }
        }
    }
    let mut lsp_q = [0.0f32; LPC_ORDER];
    for j in 0..LPC_ORDER {
        lsp_q[j] = lsf[j].cos();
    }

    (best_l0, best_l1, best_l2, best_l3, lsp_q)
}

// =========================================================================
// LPC residual
// =========================================================================

/// Compute per-frame LPC residual using per-subframe A(z) coefficients.
/// `r[n] = s[n] + sum_{k=1..10} a[k] * s[n-k]`.
fn lpc_residual(
    sig: &[f32; FRAME_SAMPLES],
    a_sf: &[[f32; LPC_ORDER + 1]; 2],
) -> [f32; FRAME_SAMPLES] {
    let mut mem = [0.0f32; LPC_ORDER];
    let mut out = [0.0f32; FRAME_SAMPLES];
    for sf in 0..SUBFRAMES_PER_FRAME {
        let a = &a_sf[sf];
        let base = sf * SUBFRAME_SAMPLES;
        for i in 0..SUBFRAME_SAMPLES {
            let x = sig[base + i];
            let mut acc = x;
            for k in 1..=LPC_ORDER {
                acc += a[k] * mem[k - 1];
            }
            out[base + i] = acc;
            for k in (1..LPC_ORDER).rev() {
                mem[k] = mem[k - 1];
            }
            mem[0] = x;
        }
    }
    out
}

// =========================================================================
// Pitch search (open-loop + closed-loop with 1/3 fractional lag)
// =========================================================================

/// Choose the best integer pitch + 1/3 fractional shift in the allowed
/// range 20..=143 (or a narrower ±5 window around `anchor` for subframe 2).
fn pitch_search(
    exc: &[f32; EXC_HIST],
    target: &[f32; SUBFRAME_SAMPLES],
    anchor: Option<usize>,
) -> (usize, i8) {
    let (lag_lo, lag_hi) = if let Some(a) = anchor {
        let lo = a.saturating_sub(5).max(20);
        let hi = (a + 5).min(143);
        (lo.max(20), hi)
    } else {
        (20usize, 143usize)
    };

    // Integer search: pick the integer lag maximising normalised
    // correlation (num^2 / den).
    let mut best_int = lag_lo;
    let mut best_score = -f32::INFINITY;
    for lag in lag_lo..=lag_hi {
        let mut cand = [0.0f32; SUBFRAME_SAMPLES];
        adaptive_codebook_excitation(exc, lag, 0, &mut cand);
        let (num, den) = xc_norm(&cand, target);
        if den < 1e-6 {
            continue;
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_int = lag;
        }
    }

    // Fractional refinement: compare frac ∈ {-1, 0, +1} on the neighbours.
    // `t_frac` uses the decoder's convention (-1, 0, +1 == -1/3, 0, +1/3).
    let mut best_frac: i8 = 0;
    for frac in [-1i8, 0, 1] {
        // Skip frac that would push us outside the bracket; the decoder's
        // adaptive_codebook_excitation handles out-of-range implicitly by
        // returning zeros, which yields a low score.
        let mut cand = [0.0f32; SUBFRAME_SAMPLES];
        adaptive_codebook_excitation(exc, best_int, frac, &mut cand);
        let (num, den) = xc_norm(&cand, target);
        if den < 1e-6 {
            continue;
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_frac = frac;
        }
    }
    (best_int, best_frac)
}

fn xc_norm(cand: &[f32; SUBFRAME_SAMPLES], target: &[f32; SUBFRAME_SAMPLES]) -> (f32, f32) {
    let mut num = 0.0f32;
    let mut den = 1e-9f32;
    for n in 0..SUBFRAME_SAMPLES {
        num += cand[n] * target[n];
        den += cand[n] * cand[n];
    }
    (num, den)
}

// =========================================================================
// Pitch-index encoding — inverse of `decode_pitch_p1` / `decode_pitch_p2`.
// =========================================================================

/// Encode (integer, frac) → 8-bit P1 index. Inverse of [`decode_pitch_p1`].
///
/// - Fractional range: integer 20..=84 + frac ∈ {-1, 0, +1} maps to 0..196.
/// - Integer-only range: 85..=142 maps to 197..254.
fn encode_pitch_p1(t_int: usize, t_frac: i8) -> u8 {
    let t_int = t_int.clamp(20, 143);
    if t_int <= 84 || (t_int == 85 && t_frac < 0) {
        // Fractional grid.
        let frac = t_frac.clamp(-1, 1) as i32;
        // decode formula: t = idx + 59; t_int = t / 3; t_frac = t - 3*t_int - 1
        //           idx = 3*t_int + t_frac + 1 - 59
        let idx = 3 * (t_int as i32) + frac + 1 - 59;
        idx.clamp(0, 196) as u8
    } else {
        // Integer grid.
        let idx = (t_int as i32 + 112).clamp(197, 254);
        idx as u8
    }
}

/// Encode (integer, frac) → 5-bit P2 index, given the anchor `p1_int`.
/// Inverse of [`decode_pitch_p2`].
fn encode_pitch_p2(t_int: usize, t_frac: i8, p1_int: usize) -> u8 {
    let mut t_min = p1_int.saturating_sub(5);
    if t_min < 20 {
        t_min = 20;
    }
    let mut t_max = t_min + 9;
    if t_max > 143 {
        t_max = 143;
        t_min = t_max - 9;
    }
    let t_int = t_int.clamp(t_min, t_max) as i32;
    let t_frac = t_frac.clamp(-1, 1) as i32;
    // Inverse of:
    //   t = idx + 59 - 3*(t_min - 1);
    //   t_int = t/3 + t_min - 1; t_frac = t - 3*(t_int - t_min + 1) - 1
    //   => t = 3 * (t_int - t_min + 1) + t_frac + 1
    //   => idx = t - 59 + 3*(t_min - 1)
    let tmin_i = t_min as i32;
    let t = 3 * (t_int - tmin_i + 1) + t_frac + 1;
    let idx = t - 59 + 3 * (tmin_i - 1);
    idx.clamp(0, 31) as u8
}

// =========================================================================
// Fixed-codebook search (ACELP 4-pulse, focused by track)
// =========================================================================

/// Depth-first 4-pulse ACELP search symmetric with the decoder's
/// `fixed_codebook_excitation`. For each track we pick the position (and
/// sign) that maximises the normalised correlation with the remaining
/// target signal. Returns `(c13, s4)` as used in the bit layout.
///
/// Track positions (§3.8, decoder side):
///   - track 0: 3-bit index, pos = 5*k       (0, 5, ...,35)
///   - track 1: 3-bit index, pos = 5*k + 1
///   - track 2: 3-bit index, pos = 5*k + 2
///   - track 3: 3-bit index + 1-bit jitter, pos = 5*k + 3 + jitter
fn fixed_codebook_search(target: &[f32; SUBFRAME_SAMPLES]) -> (u16, u8) {
    let mut residual = *target;
    let mut c: u32 = 0;
    let mut s: u32 = 0;

    // For each track, pick the best signed unit pulse from the residual.
    // Track order — 0, 1, 2, 3.
    // Focused depth-first search: each track picks greedily. This is
    // far from the spec's full depth-first search, but sidesteps the
    // 2^17 exhaustive search and still produces useful excitation.
    let tracks: [&[usize]; 4] = [
        &[0, 5, 10, 15, 20, 25, 30, 35],
        &[1, 6, 11, 16, 21, 26, 31, 36],
        &[2, 7, 12, 17, 22, 27, 32, 37],
        &[3, 4, 8, 9, 13, 14, 18, 19, 23, 24, 28, 29, 33, 34, 38, 39],
    ];
    for track_idx in 0..4 {
        let positions = tracks[track_idx];
        let mut best_pos_k: usize = 0;
        let mut best_sign_pos = false; // true ≡ +1
        let mut best_abs = -1.0f32;
        for (k, &pos) in positions.iter().enumerate() {
            let v = residual[pos];
            let av = v.abs();
            if av > best_abs {
                best_abs = av;
                best_pos_k = k;
                best_sign_pos = v >= 0.0;
            }
        }
        // Pack into C: track 0..2 use 3 bits; track 3 uses 3+1 (jitter in bit 12).
        if track_idx < 3 {
            c |= ((best_pos_k & 0x7) as u32) << (3 * track_idx);
        } else {
            // Track 3: 4-bit index in the positions[] list => 3-bit base + jitter.
            // `positions` is [3,4,8,9,13,14,...]. Base = k>>1; jitter = k & 1.
            let base = (best_pos_k >> 1) & 0x7;
            let jitter = best_pos_k & 0x1;
            c |= (base as u32) << 9;
            c |= (jitter as u32) << 12;
        }
        // Sign bit — decoder reads `s & 0x1` as track-0 sign.
        if best_sign_pos {
            s |= 1 << track_idx;
        }
        // Subtract the chosen pulse from the residual so later tracks
        // aren't all drawn to the same dominant peak.
        let chosen_pos = positions[best_pos_k];
        let amp = if best_sign_pos { 1.0 } else { -1.0 };
        let alpha = residual[chosen_pos]; // orthogonal projection of the unit pulse.
        let _ = amp;
        residual[chosen_pos] -= alpha;
    }

    ((c & 0x1FFF) as u16, (s & 0xF) as u8)
}

// =========================================================================
// Gain analysis + 2-stage VQ
// =========================================================================

/// Unconstrained LS gain for predictor `pred` against `target`:
/// g = <pred, target> / <pred, pred>.
fn gain_ls(pred: &[f32; SUBFRAME_SAMPLES], target: &[f32; SUBFRAME_SAMPLES]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 1e-9f32;
    for n in 0..SUBFRAME_SAMPLES {
        num += pred[n] * target[n];
        den += pred[n] * pred[n];
    }
    num / den
}

/// Find (GA, GB) indices whose GBK1[ga] + GBK2[gb] reconstruction is
/// closest to `(g_p, gamma_target)`. We treat `gamma_target` as the ratio
/// `g_c / <reference gain>`; the decoder applies its own MA-4 predictor
/// so we only need a plausible `gamma` in the same range the tables span.
fn quantise_gain(g_p: f32, g_c_raw: f32) -> (u8, u8) {
    // The decoder's gamma target is a unitless correction factor. We
    // derive an effective target gamma from g_c_raw by dividing out a
    // reference of 1.0 — i.e. we search for (GA,GB) minimising
    //   (g_p_target - (GBK1[ga][0] + GBK2[gb][0]))^2
    // + lambda * (gamma_target - (GBK1[ga][1] + GBK2[gb][1]))^2
    // with gamma_target = clamp(g_c_raw_normalised, 0.15..2.3).
    let gamma_target = g_c_raw.clamp(0.0, 4.0).sqrt().clamp(0.15, 2.3);
    let lambda = 0.3f32;
    let mut best = (0u8, 0u8);
    let mut best_err = f32::INFINITY;
    for ga in 0..GBK1.len() {
        for gb in 0..GBK2.len() {
            let gp = GBK1[ga][0] + GBK2[gb][0];
            let gm = GBK1[ga][1] + GBK2[gb][1];
            // Skip combinations that would fall outside the decoder's clamp.
            if !(0.0..=1.3).contains(&gp) {
                continue;
            }
            if !(0.0..=2.6).contains(&gm) {
                continue;
            }
            let d0 = g_p - gp;
            let d1 = gamma_target - gm;
            let err = d0 * d0 + lambda * d1 * d1;
            if err < best_err {
                best_err = err;
                best = (ga as u8, gb as u8);
            }
        }
    }
    best
}

fn dequantise_gain(ga: u8, gb: u8) -> (f32, f32) {
    // Same formulation as the decoder — kept local to avoid pulling the
    // decoder's `decode_gain_indices` out of its module (it also clamps
    // the inputs in decoder-specific ways that are close enough to our
    // reconstruction needs).
    let ga = (ga as usize) & 0x7;
    let gb = (gb as usize) & 0xF;
    let g_p = (GBK1[ga][0] + GBK2[gb][0]).clamp(0.0, 1.2);
    let gamma = (GBK1[ga][1] + GBK2[gb][1]).clamp(0.0, 2.5);
    (g_p, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_frame_known_pattern() {
        // Every field set to all-ones should produce the 0xFF...FF packet.
        let fp = EncodedFrame {
            l0: 1,
            l1: 0x7F,
            l2: 0x1F,
            l3: 0x1F,
            p1: 0xFF,
            p0: 1,
            c1: 0x1FFF,
            s1: 0xF,
            ga1: 0x7,
            gb1: 0xF,
            p2: 0x1F,
            c2: 0x1FFF,
            s2: 0xF,
            ga2: 0x7,
            gb2: 0xF,
        };
        let bytes = pack_frame(&fp);
        assert_eq!(bytes.len(), FRAME_BYTES);
        assert!(bytes.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn encode_decode_pitch_p1_round_trips_on_integer_grid() {
        use crate::synthesis::decode_pitch_p1;
        for t in 85..=142 {
            let idx = encode_pitch_p1(t, 0);
            let (dt, df) = decode_pitch_p1(idx);
            assert_eq!((dt, df), (t, 0));
        }
    }

    #[test]
    fn encode_decode_pitch_p1_round_trips_on_fractional_grid() {
        use crate::synthesis::decode_pitch_p1;
        for t in 20..=84 {
            for f in [-1i8, 0, 1] {
                let idx = encode_pitch_p1(t, f);
                let (dt, df) = decode_pitch_p1(idx);
                // Round-trip must hit the same lattice point.
                assert_eq!(dt, t, "int mismatch for (t={t}, f={f}) idx={idx}");
                assert_eq!(df, f, "frac mismatch for (t={t}, f={f}) idx={idx}");
            }
        }
    }

    #[test]
    fn quantise_gain_returns_valid_indices() {
        let (ga, gb) = quantise_gain(0.5, 1.0);
        assert!(ga < GBK1.len() as u8);
        assert!(gb < GBK2.len() as u8);
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
