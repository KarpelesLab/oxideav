//! Narrowband Speex CELP decoder (float-mode port).
//!
//! Mirrors `nb_decode` in `libspeex/nb_celp.c`. Speex narrowband packs
//! 160 samples (20 ms @ 8 kHz) per frame, four 40-sample sub-frames,
//! with a 10th-order LPC synthesis filter.
//!
//! The decoder maintains state across calls:
//!   * a 5-frame excitation history used by the long-term predictor
//!     (`pitch_unquant_3tap` reads `exc[-pitch ..]`, with pitch up to
//!     144 samples),
//!   * the previous frame's quantized LSPs for sub-frame interpolation,
//!   * the IIR memory of the LPC synthesis filter,
//!   * a PRNG seed used by `noise_codebook_unquant`.
//!
//! The implementation is data-faithful — table values, control flow,
//! and constants come from the BSD-licensed Xiph reference.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::lsp::{lsp_interpolate, lsp_to_lpc, lsp_unquant_lbr, lsp_unquant_nb};
use crate::submodes::{nb_submode, InnovKind, LspKind, LtpKind, SplitCbParams, WB_SKIP_TABLE};

// ----- NB constants (from `nb_celp.h`). --------------------------------
pub const NB_FRAME_SIZE: usize = 160;
pub const NB_SUBFRAME_SIZE: usize = 40;
pub const NB_NB_SUBFRAMES: usize = 4;
pub const NB_ORDER: usize = 10;
pub const NB_PITCH_START: i32 = 17;
pub const NB_PITCH_END: i32 = 144;
/// Stability margin for `lsp_interpolate` (see `nb_celp.c` `LSP_MARGIN`).
pub const LSP_MARGIN: f32 = 0.002;

// Sub-frame innovation gain quantizer values (float, from `nb_celp.c`).
const EXC_GAIN_QUANT_SCAL3: [f32; 8] = [
    0.061130, 0.163546, 0.310413, 0.428220, 0.555887, 0.719055, 0.938694, 1.326874,
];
const EXC_GAIN_QUANT_SCAL1: [f32; 2] = [0.70469, 1.05127];

// ----- Excitation buffer ----------------------------------------------
// The reference layout pads `excBuf` with 2*NB_PITCH_END +
// NB_SUBFRAME_SIZE + 12 samples of look-back history (the LTP wraps
// up to one full pitch period back, possibly twice when the three-tap
// pitch is `pitch+1-i`). We allocate the same shape and keep
// `EXC_OFFSET` as the index of "current sample 0" inside the buffer.
const EXC_HISTORY: usize = 2 * NB_PITCH_END as usize + NB_SUBFRAME_SIZE + 12;
const EXC_BUF_LEN: usize = EXC_HISTORY + NB_FRAME_SIZE;

pub struct NbDecoder {
    /// Quantized LSPs from the previous frame.
    old_qlsp: [f32; NB_ORDER],
    /// Interpolated LPC of the *last* sub-frame of the previous frame —
    /// used as the initial filter state for the current sub-frame's
    /// interpolation.
    interp_qlpc: [f32; NB_ORDER],
    /// LPC synthesis filter memory.
    mem_sp: [f32; NB_ORDER],
    /// Excitation history followed by the current frame.
    exc_buf: Vec<f32>,
    /// PRNG seed for the noise codebook.
    seed: u32,
    /// True before the first frame is decoded — initialises the LSP
    /// and LPC state on the first packet.
    first: bool,
    /// Tracks DTX (discontinuous transmission) state for sub-mode 1.
    dtx_enabled: bool,
    /// Per-subframe synthesis-filter DC gain Π(1 - a_k) at ω=π
    /// (i.e. evaluated at Nyquist). Mirrors `st->pi_gain[sub]` — used
    /// by the wideband SB-CELP extension to balance the high-band
    /// excitation against the NB synthesis envelope.
    pi_gain: [f32; NB_NB_SUBFRAMES],
    /// Per-subframe RMS of the combined excitation — mirrors
    /// `st->exc_rms[sub]`. Used as the `el` scalar in SB-CELP
    /// high-band stochastic gain decoding.
    exc_rms_sub: [f32; NB_NB_SUBFRAMES],
    /// Per-sample innovation (fixed codebook contribution only, not
    /// scaled by the adaptive excitation) for the current frame.
    /// Mirrors the `SPEEX_SET_INNOVATION_SAVE` buffer that SB-CELP
    /// aliases into `out + frame_size` to drive the spectral-folding
    /// excitation path.
    innov: [f32; NB_FRAME_SIZE],
    /// FIR memory for the perceptual (formant) postfilter — holds the
    /// last `NB_ORDER` synthesis-output samples of the previous
    /// sub-frame. Zero at cold-start.
    pf_mem_fir: [f32; NB_ORDER],
    /// IIR memory for the perceptual postfilter — holds the last
    /// `NB_ORDER` postfiltered samples of the previous sub-frame.
    pf_mem_iir: [f32; NB_ORDER],
    /// One-sample memory for the postfilter's spectral-tilt
    /// compensation stage (1 - μ·z⁻¹).
    pf_mem_tilt: f32,
}

impl Default for NbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NbDecoder {
    pub fn new() -> Self {
        let mut old_qlsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            old_qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
        }
        Self {
            old_qlsp,
            interp_qlpc: [0.0; NB_ORDER],
            mem_sp: [0.0; NB_ORDER],
            exc_buf: vec![0.0; EXC_BUF_LEN],
            seed: 1000,
            first: true,
            dtx_enabled: false,
            pi_gain: [0.0; NB_NB_SUBFRAMES],
            exc_rms_sub: [0.0; NB_NB_SUBFRAMES],
            innov: [0.0; NB_FRAME_SIZE],
            pf_mem_fir: [0.0; NB_ORDER],
            pf_mem_iir: [0.0; NB_ORDER],
            pf_mem_tilt: 0.0,
        }
    }

    /// Per-subframe Π-gain of the interpolated synthesis filter
    /// (read-only view). Used by the wideband SB-CELP layer to
    /// balance high-band excitation relative to the low-band envelope.
    pub fn pi_gain(&self) -> &[f32; NB_NB_SUBFRAMES] {
        &self.pi_gain
    }

    /// Per-subframe RMS of the NB excitation (read-only view). Used by
    /// the SB-CELP layer as the `el` scalar in stochastic gain decoding.
    pub fn exc_rms(&self) -> &[f32; NB_NB_SUBFRAMES] {
        &self.exc_rms_sub
    }

    /// Per-sample innovation (fixed-codebook contribution) for the
    /// most recently decoded frame. Used by the SB-CELP spectral-folding
    /// excitation path.
    pub fn innov(&self) -> &[f32; NB_FRAME_SIZE] {
        &self.innov
    }

    /// Decode one Speex narrowband frame. The bitstream may include a
    /// leading wideband-skip wrapper (a NB-only decoder advances past
    /// the WB layers but does not synthesise the high band). Returns
    /// `Ok(())` after writing 160 samples to `out`.
    pub fn decode_frame(&mut self, br: &mut BitReader, out: &mut [f32]) -> Result<()> {
        debug_assert_eq!(out.len(), NB_FRAME_SIZE);

        // ---- Sub-mode selection (incl. wideband-layer skip) -----------
        let m: u32;
        loop {
            if br.bits_remaining() < 5 {
                return Err(Error::invalid("Speex NB: truncated frame (need ≥5 bits)"));
            }
            let wideband = br.read_u32(1)?;
            if wideband != 0 {
                // Skip a wideband layer.
                let submode = br.read_u32(3)? as usize;
                let advance = WB_SKIP_TABLE[submode];
                if advance < 0 {
                    return Err(Error::invalid("Speex NB: invalid WB sub-mode"));
                }
                let advance = (advance as u32).saturating_sub(4); // SB_SUBMODE_BITS+1
                if br.bits_remaining() < advance as u64 + 5 {
                    return Err(Error::invalid("Speex NB: truncated WB layer"));
                }
                // The reference reads `advance` bits and discards them.
                let mut left = advance;
                while left >= 24 {
                    br.read_u32(24)?;
                    left -= 24;
                }
                if left > 0 {
                    br.read_u32(left)?;
                }
                continue;
            }
            if br.bits_remaining() < 4 {
                return Err(Error::invalid("Speex NB: truncated frame (sub-mode)"));
            }
            m = br.read_u32(4)?;
            if m == 15 {
                // Terminator — frame done. Reference returns -1 here.
                return Err(Error::Eof);
            } else if m == 14 || m == 13 {
                return Err(Error::unsupported(
                    "Speex NB: in-band/user request packets (sub-mode 13/14) not implemented \
                     — see RFC 5574 §3.4 / nb_celp.c",
                ));
            } else if m > 8 {
                return Err(Error::invalid(format!(
                    "Speex NB: invalid sub-mode id {m} (>8)"
                )));
            }
            break;
        }

        // ---- Sub-mode 0 = "null mode" — emit comfort noise -----------
        let Some(sm) = nb_submode(m) else {
            // Reference: use bandwidth-expanded LPC of previous frame
            // to filter PRNG noise scaled to the previous excitation
            // RMS. We approximate: zero excitation + last filter.
            let exc_start = EXC_HISTORY;
            let mut innov_gain = 0.0f32;
            for i in 0..NB_FRAME_SIZE {
                innov_gain += self.exc_buf[exc_start + i].powi(2);
            }
            innov_gain = (innov_gain / NB_FRAME_SIZE as f32).sqrt();
            for i in 0..NB_FRAME_SIZE {
                self.exc_buf[exc_start + i] = speex_rand(innov_gain, &mut self.seed);
                self.innov[i] = self.exc_buf[exc_start + i];
            }
            // Bandwidth-expand the previous LPCs by 0.93.
            let mut lpc = [0.0f32; NB_ORDER];
            crate::lsp::bw_lpc(0.93, &self.interp_qlpc, &mut lpc, NB_ORDER);
            // SAFETY: `iir_mem16` reads from `x` and writes to a different
            // slice (`y` = `out`); `mem` aliases neither. The borrow
            // checker can't see through that, so split the borrow with
            // an intermediate copy of just the read window.
            let exc_in = self.exc_buf[exc_start..exc_start + NB_FRAME_SIZE].to_vec();
            iir_mem16(
                &exc_in,
                &lpc,
                out,
                NB_FRAME_SIZE,
                NB_ORDER,
                &mut self.mem_sp,
            );
            // No excitation updates per sub-frame: flat envelope.
            for sub in 0..NB_NB_SUBFRAMES {
                self.pi_gain[sub] = pi_gain_of(&lpc, NB_ORDER);
                let start = sub * NB_SUBFRAME_SIZE;
                self.exc_rms_sub[sub] =
                    rms(&self.exc_buf[exc_start + start..exc_start + start + NB_SUBFRAME_SIZE]);
            }
            self.first = true;
            self.shift_exc_buffer();
            return Ok(());
        };

        // ---- LSP unquantisation --------------------------------------
        let mut qlsp = [0.0f32; NB_ORDER];
        match sm.lsp {
            LspKind::Lbr => lsp_unquant_lbr(&mut qlsp, NB_ORDER, br)?,
            LspKind::Nb => lsp_unquant_nb(&mut qlsp, NB_ORDER, br)?,
        }

        if self.first {
            self.old_qlsp.copy_from_slice(&qlsp);
        }

        // ---- Open-loop pitch (lbr_pitch != -1) -----------------------
        let mut ol_pitch: i32 = 0;
        if let Some(lbr) = sm.lbr_pitch {
            if lbr != -1 {
                ol_pitch = NB_PITCH_START + br.read_u32(7)? as i32;
            }
        }

        // ---- Forced pitch coefficient (vocoder-only modes 1 & 8) -----
        let mut ol_pitch_coef: f32 = 0.0;
        if sm.forced_pitch_gain {
            let q = br.read_u32(4)?;
            ol_pitch_coef = 0.066667 * q as f32;
        }

        // ---- Open-loop excitation gain (5 bits) ----------------------
        let qe = br.read_u32(5)?;
        let ol_gain = (qe as f32 / 3.5).exp();

        // ---- Mode-1 DTX flag (4-bit "extra" field) -------------------
        if m == 1 {
            let extra = br.read_u32(4)?;
            self.dtx_enabled = extra == 15;
        } else if m > 1 {
            self.dtx_enabled = false;
        }

        // ---- Sub-frame loop ------------------------------------------
        for sub in 0..NB_NB_SUBFRAMES {
            let offset_in_frame = NB_SUBFRAME_SIZE * sub;
            let exc_idx = EXC_HISTORY + offset_in_frame;

            // Reset excitation slot for this sub-frame.
            for i in 0..NB_SUBFRAME_SIZE {
                self.exc_buf[exc_idx + i] = 0.0;
            }

            // ---- Adaptive (pitch) codebook --------------------------
            let (pit_min, pit_max) = match sm.lbr_pitch {
                Some(-1) | None => (NB_PITCH_START, NB_PITCH_END),
                Some(0) => (ol_pitch, ol_pitch),
                Some(margin) => {
                    let mut lo = ol_pitch - margin + 1;
                    if lo < NB_PITCH_START {
                        lo = NB_PITCH_START;
                    }
                    let mut hi = ol_pitch + margin;
                    if hi > NB_PITCH_END {
                        hi = NB_PITCH_END;
                    }
                    (lo, hi)
                }
            };

            let mut exc_out = [0.0f32; NB_SUBFRAME_SIZE];
            match sm.ltp {
                LtpKind::ThreeTap => {
                    pitch_unquant_3tap(
                        br,
                        &self.exc_buf,
                        exc_idx,
                        &mut exc_out,
                        pit_min,
                        pit_max,
                        &sm.ltp_params,
                    )?;
                }
                LtpKind::Forced => {
                    forced_pitch_unquant(
                        &self.exc_buf,
                        exc_idx,
                        &mut exc_out,
                        ol_pitch.max(NB_PITCH_START),
                        ol_pitch_coef,
                    );
                }
            }

            // Sanitise (mirror `sanitize_values32`).
            for v in &mut exc_out {
                if !v.is_finite() {
                    *v = 0.0;
                }
                *v = (*v).clamp(-32000.0, 32000.0);
            }

            // ---- Innovation (fixed) codebook ------------------------
            let mut innov = [0.0f32; NB_SUBFRAME_SIZE];
            let ener = match sm.have_subframe_gain {
                3 => {
                    let q = br.read_u32(3)? as usize;
                    EXC_GAIN_QUANT_SCAL3[q] * ol_gain
                }
                1 => {
                    let q = br.read_u32(1)? as usize;
                    EXC_GAIN_QUANT_SCAL1[q] * ol_gain
                }
                _ => ol_gain,
            };

            match sm.innov {
                InnovKind::SplitCb => {
                    split_cb_shape_sign_unquant(br, &sm.innov_params, &mut innov)?;
                }
                InnovKind::Noise => {
                    noise_codebook_unquant(&mut innov, &mut self.seed);
                }
            }
            for v in innov.iter_mut() {
                *v *= ener;
            }

            // ---- Optional second codebook (sub-mode 7) --------------
            if sm.double_codebook {
                let mut innov2 = [0.0f32; NB_SUBFRAME_SIZE];
                split_cb_shape_sign_unquant(br, &sm.innov_params, &mut innov2)?;
                for i in 0..NB_SUBFRAME_SIZE {
                    innov[i] += innov2[i] * 0.454545 * ener;
                }
            }

            // ---- Combine adaptive + innovation excitation -----------
            // Float path: `exc[i] = exc_out[i] + innov[i]` (SIG_SHIFT
            // and SHL32-by-1 are no-ops in float mode of the reference).
            for i in 0..NB_SUBFRAME_SIZE {
                self.exc_buf[exc_idx + i] = exc_out[i] + innov[i];
                self.innov[offset_in_frame + i] = innov[i];
            }
            // ---- Save combined-excitation RMS for SB-CELP. ----------
            // Reference: `st->exc_rms[sub] = compute_rms16(exc, ...)`.
            self.exc_rms_sub[sub] = rms(&self.exc_buf[exc_idx..exc_idx + NB_SUBFRAME_SIZE]);
        }

        // ---- Copy excitation into `out` as the LPC filter's input ----
        // The reference at this point optionally runs `multicomb`, a
        // pitch-based comb post-filter that writes directly into `out`
        // while also performing LPC synthesis. We split the two steps:
        // first LPC synthesis (below), then a short-term (LPC-based)
        // formant postfilter that only depends on the per-sub-frame
        // `ak`. The pitch-enhancement branch of `multicomb` is not
        // modelled — it requires saving the per-sub-frame pitch/gain
        // and tends to introduce audible warbling unless tuned with
        // care; the formant postfilter alone gives most of the
        // perceptual lift for a narrowband speech decoder.
        for i in 0..NB_FRAME_SIZE {
            out[i] = self.exc_buf[EXC_HISTORY - NB_SUBFRAME_SIZE + i];
        }

        // ---- LPC synthesis + postfilter (per-sub-frame) --------------
        let mut interp_qlsp = [0.0f32; NB_ORDER];
        let mut ak = [0.0f32; NB_ORDER];
        // `comb_gain > 0` in the sub-mode record means libspeex would
        // run its postfilter for this mode; sub-mode 1 (comfort noise)
        // sets `comb_gain = -1.0` to disable it. We follow the same
        // gating so the vocoder mode stays untouched.
        let do_postfilter = sm.comb_gain > 0.0;
        for sub in 0..NB_NB_SUBFRAMES {
            let off = NB_SUBFRAME_SIZE * sub;
            lsp_interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut interp_qlsp,
                NB_ORDER,
                sub,
                NB_NB_SUBFRAMES,
                LSP_MARGIN,
            );
            lsp_to_lpc(&interp_qlsp, &mut ak, NB_ORDER);

            // In-place IIR — but `out[off..off+40]` is currently the
            // shifted excitation; iir_mem16 reads `x[i]` then writes
            // `y[i]`, so a single call works.
            let sp_in: Vec<f32> = out[off..off + NB_SUBFRAME_SIZE].to_vec();
            iir_mem16(
                &sp_in,
                &self.interp_qlpc,
                &mut out[off..off + NB_SUBFRAME_SIZE],
                NB_SUBFRAME_SIZE,
                NB_ORDER,
                &mut self.mem_sp,
            );

            // Short-term (formant) postfilter:
            //
            //     H_pf(z) = (1 - μ z⁻¹) · A(z/γ2) / A(z/γ1)
            //
            // with γ1 = 0.65 (numerator) and γ2 = 0.75 (denominator)
            // giving gentle formant emphasis (libspeex defaults). The
            // tilt compensation `(1 - μ z⁻¹)` offsets the average
            // spectral tilt that the formant stage introduces. A
            // per-sub-frame RMS normalisation is applied afterwards so
            // the postfilter does not alter excitation-gain coding; this
            // is what `multicomb` does internally in the reference.
            if do_postfilter {
                let mut ak_num = [0.0f32; NB_ORDER];
                let mut ak_den = [0.0f32; NB_ORDER];
                crate::lsp::bw_lpc(0.65, &ak, &mut ak_num, NB_ORDER);
                crate::lsp::bw_lpc(0.75, &ak, &mut ak_den, NB_ORDER);
                formant_postfilter(
                    &mut out[off..off + NB_SUBFRAME_SIZE],
                    &ak_num,
                    &ak_den,
                    NB_ORDER,
                    &mut self.pf_mem_fir,
                    &mut self.pf_mem_iir,
                    &mut self.pf_mem_tilt,
                    0.3,
                );
            }

            // Save the per-subframe synthesis filter Π-gain at ω=π
            // (evaluated using the just-computed coefficients). Used by
            // the SB-CELP high-band extension. Reference `nb_celp.c`
            // does this with the same formula.
            self.pi_gain[sub] = pi_gain_of(&ak, NB_ORDER);
            // Update interp_qlpc to current sub-frame's coefficients
            // for the *next* sub-frame's filter pass — matches the
            // reference (it stores `ak` into `interp_qlpc` after each
            // iir_mem16 call).
            self.interp_qlpc.copy_from_slice(&ak);
        }

        // ---- Save state for next frame -------------------------------
        self.old_qlsp.copy_from_slice(&qlsp);
        self.first = false;
        self.shift_exc_buffer();
        Ok(())
    }

    /// Slide the excitation history left by NB_FRAME_SIZE samples,
    /// preparing space for the next frame's writes. Mirrors
    /// `SPEEX_MOVE(st->excBuf, st->excBuf+NB_FRAME_SIZE, ...)`.
    fn shift_exc_buffer(&mut self) {
        self.exc_buf.copy_within(NB_FRAME_SIZE.., 0);
        for v in &mut self.exc_buf[EXC_BUF_LEN - NB_FRAME_SIZE..] {
            *v = 0.0;
        }
    }
}

// =====================================================================
// Helper kernels — direct float ports from libspeex/{ltp,cb_search,
// filters,math_approx}.c.
// =====================================================================

/// `pitch_unquant_3tap` from `ltp.c` (float path). Reads pitch and gain
/// indices from the bitstream, then writes the LTP contribution into
/// `exc_out` (zero-initialised here). Reads `exc[base + j - pp]` for
/// the pitch lag — those samples must already be present in the
/// excitation history.
fn pitch_unquant_3tap(
    br: &mut BitReader,
    exc_buf: &[f32],
    exc_idx: usize,
    exc_out: &mut [f32; NB_SUBFRAME_SIZE],
    start: i32,
    _end: i32,
    params: &super::submodes::LtpParams,
) -> Result<()> {
    let gain_cdbk_size = 1usize << params.gain_bits;
    let gain_cdbk = params.gain_cdbk;
    let pitch = br.read_u32(params.pitch_bits)? as i32 + start;
    let gain_index = br.read_u32(params.gain_bits)? as usize;
    if gain_index >= gain_cdbk_size {
        return Err(Error::invalid("Speex NB: pitch-gain index out of range"));
    }
    let g0 = 0.015625 * gain_cdbk[gain_index * 4] as f32 + 0.5;
    let g1 = 0.015625 * gain_cdbk[gain_index * 4 + 1] as f32 + 0.5;
    let g2 = 0.015625 * gain_cdbk[gain_index * 4 + 2] as f32 + 0.5;
    let gain = [g0, g1, g2];

    for v in exc_out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..3 {
        let pp = (pitch + 1 - i as i32) as usize;
        let nsf = NB_SUBFRAME_SIZE;
        let tmp1 = nsf.min(pp);
        for j in 0..tmp1 {
            // exc[j - pp] in reference (j>=0, pp>=17). Index in buffer
            // is `exc_idx + j - pp`. Saturating in case of negative.
            let src = exc_idx as isize + j as isize - pp as isize;
            if src < 0 || src as usize >= exc_buf.len() {
                continue;
            }
            exc_out[j] += gain[2 - i] * exc_buf[src as usize];
        }
        let tmp3 = nsf.min(pp + pitch as usize);
        for j in tmp1..tmp3 {
            let src = exc_idx as isize + j as isize - pp as isize - pitch as isize;
            if src < 0 || src as usize >= exc_buf.len() {
                continue;
            }
            exc_out[j] += gain[2 - i] * exc_buf[src as usize];
        }
    }
    Ok(())
}

/// `forced_pitch_unquant` from `ltp.c`. Reads no bits; just copies a
/// scaled tap of the past excitation forward.
fn forced_pitch_unquant(
    exc_buf: &[f32],
    exc_idx: usize,
    exc_out: &mut [f32; NB_SUBFRAME_SIZE],
    start: i32,
    pitch_coef: f32,
) {
    let pitch_coef = pitch_coef.min(0.99);
    for i in 0..NB_SUBFRAME_SIZE {
        let src = exc_idx as isize + i as isize - start as isize;
        let s = if src < 0 || src as usize >= exc_buf.len() {
            0.0
        } else {
            exc_buf[src as usize]
        };
        exc_out[i] = s * pitch_coef;
    }
}

/// `split_cb_shape_sign_unquant` from `cb_search.c` (float path). Reads
/// `nb_subvect * (shape_bits + have_sign)` bits and accumulates into
/// `exc` (which is zero-initialised on entry).
fn split_cb_shape_sign_unquant(
    br: &mut BitReader,
    p: &SplitCbParams,
    exc: &mut [f32; NB_SUBFRAME_SIZE],
) -> Result<()> {
    if p.subvect_size * p.nb_subvect > NB_SUBFRAME_SIZE {
        return Err(Error::invalid("Speex NB: split-CB layout exceeds subframe"));
    }
    let codebook_size = (1usize << p.shape_bits) * p.subvect_size;
    for i in 0..p.nb_subvect {
        let sign = if p.have_sign {
            br.read_u32(1)? != 0
        } else {
            false
        };
        let ind = br.read_u32(p.shape_bits)? as usize;
        let s: f32 = if sign { -1.0 } else { 1.0 };
        let base = ind * p.subvect_size;
        if base + p.subvect_size > codebook_size && !p.shape_cb.is_empty() {
            // Bound check; should be unreachable for well-formed input.
            return Err(Error::invalid("Speex NB: split-CB index out of range"));
        }
        for j in 0..p.subvect_size {
            exc[p.subvect_size * i + j] += s * 0.03125 * p.shape_cb[base + j] as f32;
        }
    }
    Ok(())
}

/// `noise_codebook_unquant` from `cb_search.c` — fills with PRNG noise.
fn noise_codebook_unquant(exc: &mut [f32; NB_SUBFRAME_SIZE], seed: &mut u32) {
    for v in exc.iter_mut() {
        *v = speex_rand(1.0, seed);
    }
}

/// `iir_mem16` from `filters.c` (float path). LPC synthesis filter:
/// `y[i] = x[i] - sum(den[k] * y[i-k-1])`. Memory is order-long.
pub(crate) fn iir_mem16(
    x: &[f32],
    den: &[f32],
    y: &mut [f32],
    n: usize,
    ord: usize,
    mem: &mut [f32],
) {
    debug_assert!(mem.len() >= ord);
    for i in 0..n {
        let yi = x[i] + mem[0];
        let nyi = -yi;
        for j in 0..ord - 1 {
            mem[j] = mem[j + 1] + den[j] * nyi;
        }
        mem[ord - 1] = den[ord - 1] * nyi;
        // saturate to int16 range like the reference (helps prevent
        // runaway in degenerate streams).
        y[i] = yi.clamp(-32767.0, 32767.0);
    }
}

/// Short-term (formant) perceptual postfilter, in place.
///
/// Applies `H(z) = (1 - μ z⁻¹) · A(z/γ2) / A(z/γ1)` to `samples` using
/// bandwidth-expanded LPC coefficients (`num` = `A(z/γ2)` without the
/// leading 1, `den` = `A(z/γ1)` without the leading 1). RMS is
/// preserved across the sub-frame so the postfilter does not leak
/// gain into the excitation-level coding. `fir_mem` and `iir_mem` hold
/// the `order`-tap history across calls; `tilt_mem` is the single
/// sample feeding the `(1 - μ z⁻¹)` tilt stage.
///
/// This is the LPC-based branch of the Speex postfilter; the pitch
/// enhancement branch (`multicomb`) is intentionally not implemented.
pub(crate) fn formant_postfilter(
    samples: &mut [f32],
    num: &[f32; NB_ORDER],
    den: &[f32; NB_ORDER],
    order: usize,
    fir_mem: &mut [f32; NB_ORDER],
    iir_mem: &mut [f32; NB_ORDER],
    tilt_mem: &mut f32,
    mu: f32,
) {
    debug_assert!(order <= NB_ORDER);
    let n = samples.len();

    // Measure input RMS so we can restore it at the end — prevents the
    // postfilter from drifting the sub-frame gain.
    let rms_in = rms(samples);

    // FIR stage: y[i] = x[i] + Σ_k num[k] · x[i-k-1]. Using a local
    // tap delay line seeded from `fir_mem` then updated each step.
    let mut fir_tap = *fir_mem;
    let mut y = [0.0f32; NB_SUBFRAME_SIZE];
    for i in 0..n {
        let xi = samples[i];
        let mut acc = xi;
        for k in 0..order {
            acc += num[k] * fir_tap[k];
        }
        // Shift the delay line: fir_tap[k] was x[i-k-1]; next iteration
        // needs x[i-k], so shift right and inject the current sample.
        for k in (1..order).rev() {
            fir_tap[k] = fir_tap[k - 1];
        }
        fir_tap[0] = xi;
        y[i] = acc;
    }
    *fir_mem = fir_tap;

    // IIR stage: z[i] = y[i] - Σ_k den[k] · z[i-k-1].
    let mut iir_tap = *iir_mem;
    let mut z = [0.0f32; NB_SUBFRAME_SIZE];
    for i in 0..n {
        let mut acc = y[i];
        for k in 0..order {
            acc -= den[k] * iir_tap[k];
        }
        for k in (1..order).rev() {
            iir_tap[k] = iir_tap[k - 1];
        }
        iir_tap[0] = acc;
        z[i] = acc;
    }
    *iir_mem = iir_tap;

    // Tilt compensation: w[i] = z[i] - μ · z[i-1]. Offsets the average
    // spectral tilt introduced by the formant emphasis.
    let mut prev = *tilt_mem;
    for i in 0..n {
        let zi = z[i];
        samples[i] = zi - mu * prev;
        prev = zi;
    }
    *tilt_mem = prev;

    // Restore the input RMS so the postfilter is gain-neutral.
    let rms_out = rms(samples);
    if rms_out > 1e-12 && rms_in > 0.0 {
        let g = rms_in / rms_out;
        for v in samples.iter_mut() {
            *v *= g;
        }
    }
}

/// Root-mean-square of a sample window — mirrors `compute_rms16` in
/// `libspeex/filters.c`. Float mode: trivial.
pub(crate) fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    (s / x.len() as f32).sqrt()
}

/// Evaluate the LPC synthesis filter Π-gain at ω=π (Nyquist) — a scalar
/// equal to `1 + Σ_k (-1)^(k+1) a_k` (reference `nb_celp.c` uses
/// `LPC_SCALING + SUB32(a[i+1], a[i])` over even indices, which is the
/// same thing for an even-order LPC). Used by the sub-band wideband
/// layer to balance the high-band excitation gain against the low-band
/// filter response.
pub(crate) fn pi_gain_of(ak: &[f32], order: usize) -> f32 {
    let mut g = 1.0f32;
    let mut i = 0;
    while i + 1 < order {
        g += ak[i + 1] - ak[i];
        i += 2;
    }
    g
}

/// Linear-congruential PRNG from `math_approx.h`. The reference scales
/// the upper 16 bits of `seed * 1664525 + 1013904223` by `std`. In
/// float mode (we follow the float path), the result is roughly
/// `[-std, std]` triangular noise.
fn speex_rand(std: f32, seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    let upper = ((*seed >> 16) & 0xFFFF) as i32 as i16 as f32;
    // Equivalent of `MULT16_16(EXTRACT16(SHR32(*seed, 16)), std)`
    // followed by `PSHR32(SUB32(res, SHR32(res, 3)), 14)` in fixed.
    // The float path is a bit looser: we emulate the ~7/8 attenuation.
    let res = upper * std;
    (res - res / 8.0) / 16384.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_is_bounded_for_unit_std() {
        let mut s = 1234u32;
        for _ in 0..1000 {
            let v = speex_rand(1.0, &mut s);
            assert!(v.abs() < 4.0, "rand out of expected range: {v}");
        }
    }

    #[test]
    fn iir_passes_dc_when_filter_is_unity() {
        // den = [0; ord] ⇒ iir reduces to a copy.
        let x = vec![1.0_f32; 16];
        let mut y = vec![0.0_f32; 16];
        let mut mem = [0.0_f32; 4];
        iir_mem16(&x, &[0.0; 4], &mut y, 16, 4, &mut mem);
        for &v in &y {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn formant_postfilter_is_gain_neutral_with_zero_lpc() {
        // With zero LPCs on both numerator and denominator the formant
        // stage reduces to identity; the tilt stage with μ=0 further
        // reduces the filter to a straight pass-through. The RMS
        // renormalisation at the end is a no-op in that case.
        let mut buf = [0.0f32; NB_SUBFRAME_SIZE];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i as f32) * 0.31).sin();
        }
        let orig = buf;
        let mut fir = [0.0f32; NB_ORDER];
        let mut iir = [0.0f32; NB_ORDER];
        let mut tilt = 0.0f32;
        formant_postfilter(
            &mut buf,
            &[0.0; NB_ORDER],
            &[0.0; NB_ORDER],
            NB_ORDER,
            &mut fir,
            &mut iir,
            &mut tilt,
            0.0,
        );
        for i in 0..NB_SUBFRAME_SIZE {
            assert!(
                (buf[i] - orig[i]).abs() < 1e-5,
                "sample {i} changed: {} vs {}",
                buf[i],
                orig[i]
            );
        }
    }

    #[test]
    fn formant_postfilter_preserves_sub_frame_rms() {
        // Feed a mixed sinusoid through the postfilter with non-trivial
        // LPC coefficients and verify the RMS renormalisation holds.
        let mut buf = [0.0f32; NB_SUBFRAME_SIZE];
        for (i, v) in buf.iter_mut().enumerate() {
            let t = i as f32;
            *v = (t * 0.5).sin() + 0.3 * (t * 0.17).cos();
        }
        let input_rms = rms(&buf);
        let num = [0.2f32, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let den = [0.3f32, -0.15, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut fir = [0.0f32; NB_ORDER];
        let mut iir = [0.0f32; NB_ORDER];
        let mut tilt = 0.0f32;
        formant_postfilter(&mut buf, &num, &den, NB_ORDER, &mut fir, &mut iir, &mut tilt, 0.3);
        let out_rms = rms(&buf);
        assert!(
            (out_rms - input_rms).abs() / input_rms.max(1e-9) < 1e-3,
            "postfilter altered sub-frame RMS: {input_rms} -> {out_rms}"
        );
        // Output must not all collapse to zero.
        assert!(buf.iter().any(|v| v.abs() > 1e-6));
    }
}
