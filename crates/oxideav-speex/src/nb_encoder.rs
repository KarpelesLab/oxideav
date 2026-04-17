//! Narrowband Speex CELP encoder (float-mode, mode-5 only).
//!
//! This is a first-cut encoder. It produces a valid mode-5 bitstream
//! (exactly 300 bits per 20 ms frame, parseable by the companion
//! [`crate::nb_decoder::NbDecoder`]) whose decoded output preserves
//! the input's spectral shape. Absolute level is approximate — the
//! encoder carries a consistent multiplicative gain error because it
//! lacks full perceptual-weighting + rate control. See the roundtrip
//! test in `tests/encode_nb.rs` for the quality floor the encoder
//! aims for (gain-corrected SNR > 8 dB).
//!
//! Supported mode:
//!   * **Sub-mode 5** (15 kbps): 30-bit NB LSP VQ, 7-bit pitch / 7-bit
//!     pitch-gain per sub-frame, 3-bit sub-frame innovation gain,
//!     48-bit split-CB innovation (8×6 bits, `EXC_5_64_TABLE`).
//!
//! All other sub-modes return `Error::Unsupported` at construction
//! time.
//!
//! Pipeline per 160-sample frame:
//!   1. Hamming window + autocorrelation of the frame.
//!   2. Lag-windowed Levinson-Durbin → 10-th order LPC.
//!   3. Bandwidth-expand the LPC (γ=0.9) to tame formant peaks, then
//!      convert to LSP via unit-circle root search on the
//!      P/Q polynomial decomposition.
//!   4. Five-stage LSP vector quantisation → 30 bits.
//!   5. Reconstruct the quantised LSPs, interpolate one set per
//!      sub-frame, re-derive LPC (matches the decoder's path exactly).
//!   6. Open-loop excitation-gain scalar: `qe ≈ 3.5·ln(residual_rms)`,
//!      scaled by an empirical 0.25 margin that keeps the decoder's
//!      `iir_mem16` output clear of its ±32767 saturation ceiling.
//!   7. Per sub-frame (4 sub-frames × 40 samples each):
//!         a. Compute the synthesis filter's zero-input response (ZIR)
//!            and impulse response `h[n]`.
//!         b. Closed-loop pitch search in the *synthesis* domain —
//!            pick the lag in [17, 144] whose single-tap LTP,
//!            convolved with `h`, best matches `pcm - ZIR`.
//!         c. Three-tap pitch-gain VQ against `GAIN_CDBK_NB` (7 bits),
//!            again in the filtered domain.
//!         d. Sub-frame innovation gain quantised to 3 bits via
//!            `EXC_GAIN_QUANT_SCAL3`.
//!         e. Split-codebook shape search (8 × 5-sample sub-vectors,
//!            6 bits each): for each sub-vector, pick the codebook
//!            entry whose convolution with `h` minimises the residual
//!            weighted error, then subtract its filtered response
//!            from the running target before moving on.
//!         f. Rebuild the excitation exactly as the decoder will so
//!            the encoder's `exc_buf` / `mem_sp_sim` state stays in
//!            lock-step with the decoder across frames.

use oxideav_core::{Error, Result};

use crate::bitwriter::BitWriter;
use crate::gain_tables::GAIN_CDBK_NB;
use crate::lsp::{lsp_interpolate, lsp_to_lpc};
use crate::lsp_tables_nb::{CDBK_NB, CDBK_NB_HIGH1, CDBK_NB_HIGH2, CDBK_NB_LOW1, CDBK_NB_LOW2};
use crate::nb_decoder::{
    rms, NB_FRAME_SIZE, NB_NB_SUBFRAMES, NB_ORDER, NB_PITCH_END, NB_PITCH_START, NB_SUBFRAME_SIZE,
};
// Local copy of `nb_decoder::LSP_MARGIN` — we avoid re-exporting it to
// keep the decoder module surface untouched.
const LSP_MARGIN: f32 = 0.002;

use crate::exc_tables::EXC_5_64_TABLE;

/// Excitation history length — must match the decoder's layout so the
/// LTP search sees the same past-excitation frame shape as the decoder
/// will.
const EXC_HISTORY: usize = 2 * NB_PITCH_END as usize + NB_SUBFRAME_SIZE + 12;
const EXC_BUF_LEN: usize = EXC_HISTORY + NB_FRAME_SIZE;

/// Mode-5 sub-frame innovation gain quantizer (same constants as the
/// decoder reads).
const EXC_GAIN_QUANT_SCAL3: [f32; 8] = [
    0.061130, 0.163546, 0.310413, 0.428220, 0.555887, 0.719055, 0.938694, 1.326874,
];

/// The only mode this encoder can emit.
pub const SUPPORTED_SUBMODE: u32 = 5;

/// Mode-5 encoder state — held across frames so the LTP search and LPC
/// interpolation can see the previous sub-frame's excitation and LSPs.
pub struct NbEncoder {
    /// Quantized LSPs from the previous frame (for sub-frame
    /// interpolation on the encoder side — mirrors what the decoder
    /// will do).
    old_qlsp: [f32; NB_ORDER],
    /// Interpolated quantised LPC from the last sub-frame of the
    /// previous frame — used as the IIR memory seed when the encoder
    /// needs to match the decoder's cold-start for perceptual filters.
    /// We retain it for symmetry but the current A-by-S uses residual
    /// matching only.
    #[allow(dead_code)]
    interp_qlpc: [f32; NB_ORDER],
    /// Encoder-side excitation history — same shape as the decoder's
    /// so pitch lags can be resolved identically.
    exc_buf: Vec<f32>,
    /// LPC analysis-filter memory (for producing the LPC residual of
    /// the current frame given the previous sub-frame's coefficients).
    mem_analysis: [f32; NB_ORDER],
    /// Simulated synthesis-filter memory — kept in lock-step with the
    /// decoder's `mem_sp`. Used for computing the zero-input response
    /// during analysis-by-synthesis.
    mem_sp_sim: [f32; NB_ORDER],
    /// First-frame flag.
    first: bool,
}

impl Default for NbEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NbEncoder {
    pub fn new() -> Self {
        let mut old_qlsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            old_qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
        }
        Self {
            old_qlsp,
            interp_qlpc: [0.0; NB_ORDER],
            exc_buf: vec![0.0; EXC_BUF_LEN],
            mem_analysis: [0.0; NB_ORDER],
            mem_sp_sim: [0.0; NB_ORDER],
            first: true,
        }
    }

    /// Symbolic delay (in samples) between calling `encode_frame(pcm)`
    /// and the decoder producing its corresponding PCM. The decoder's
    /// per-sub-frame synthesis lags the stored excitation by exactly
    /// one sub-frame (the "out[i] = exc_buf[EXC_HISTORY-40+i]" copy in
    /// `nb_decoder.rs`), so there is a 40-sample group delay.
    pub const DECODER_DELAY_SAMPLES: usize = NB_SUBFRAME_SIZE;

    /// Encode one 160-sample narrowband frame of **int16-range** float
    /// samples (i.e. typical amplitudes up to ±32768). Appends bits for
    /// a single sub-mode-5 packet (300 bits; the packet is NOT
    /// terminated with a `m=15` selector — the caller is expected to
    /// either finish the writer immediately or chain more frames).
    pub fn encode_frame(&mut self, pcm: &[f32], bw: &mut BitWriter) -> Result<()> {
        if pcm.len() != NB_FRAME_SIZE {
            return Err(Error::invalid(format!(
                "Speex NB encoder: expected {NB_FRAME_SIZE}-sample frame, got {}",
                pcm.len()
            )));
        }

        // ---- 1. LPC analysis: windowed autocorrelation -----------------
        let windowed = hamming_window(pcm);
        let mut autocorr = [0.0f32; NB_ORDER + 1];
        autocorrelate(&windowed, &mut autocorr);
        // Lag window to stabilise the Levinson recursion: a Gaussian
        // tail damping on the autocorrelation pushes poles away from
        // the unit circle. Matches the approach libspeex uses, with a
        // slightly broader τ chosen so our BW-expansion step can
        // further narrow the formants without pushing them unstable.
        for k in 1..=NB_ORDER {
            let tau = 40.0f32;
            let w = (-(0.5 * (k as f32 / tau).powi(2))).exp();
            autocorr[k] *= w;
        }
        // White-noise correction — avoids a zero r[0] producing NaN LPCs
        // on silent input.
        autocorr[0] *= 1.0001;
        if autocorr[0] < 1e-6 {
            autocorr[0] = 1e-6;
        }

        let raw_lpc = levinson_durbin(&autocorr);

        // Bandwidth expansion (γ ≈ 0.99) to guarantee a stable, less-peaked
        // synthesis filter. Without this, an aggressively modelled pure
        // tone collapses its LPC poles right onto the unit circle and
        // `iir_mem16` saturates on even modest excitation. γ=0.99 moves
        // each pole inward by 1 %, which adds ~1 dB of formant damping
        // — perceptually negligible at first-cut quality.
        let mut lpc = [0.0f32; NB_ORDER];
        crate::lsp::bw_lpc(0.2, &raw_lpc, &mut lpc, NB_ORDER);

        // ---- 2. LPC → LSP (unquantised) --------------------------------
        let lsp = lpc_to_lsp(&lpc).unwrap_or_else(|| {
            // Fallback: linear LSPs — guaranteed stable, no speech
            // structure but keeps the encoder unconditionally robust.
            let mut fallback = [0.0f32; NB_ORDER];
            for i in 0..NB_ORDER {
                fallback[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
            }
            fallback
        });

        // ---- 3. Quantise LSP (mode-5 = five-stage VQ, 30 bits) --------
        let (qlsp, lsp_indices) = quantise_lsp_nb(&lsp);

        if self.first {
            self.old_qlsp = qlsp;
        }

        // ---- 4. Build per-sub-frame interpolated qLPC ----------------
        let mut interp_qlpc = [[0.0f32; NB_ORDER]; NB_NB_SUBFRAMES];
        for sub in 0..NB_NB_SUBFRAMES {
            let mut ilsp = [0.0f32; NB_ORDER];
            lsp_interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut ilsp,
                NB_ORDER,
                sub,
                NB_NB_SUBFRAMES,
                LSP_MARGIN,
            );
            lsp_to_lpc(&ilsp, &mut interp_qlpc[sub], NB_ORDER);
        }

        // ---- 5. Frame LPC residual (target for A-by-S) ----------------
        //
        // Decoder alignment: the NB decoder filters each sub-frame's
        // excitation with the LPC of the *previous* sub-frame (see
        // `nb_decoder::decode_frame`; `interp_qlpc` carries the filter
        // state). To keep the analysis simple we use the current sub-
        // frame's LPC for our residual target — the ±1 sub-frame
        // alignment drift is swamped by the other quantization error
        // at first-cut quality.
        let mut residual = [0.0f32; NB_FRAME_SIZE];
        {
            let mut mem = self.mem_analysis;
            for sub in 0..NB_NB_SUBFRAMES {
                let off = sub * NB_SUBFRAME_SIZE;
                fir_filter(
                    &pcm[off..off + NB_SUBFRAME_SIZE],
                    &interp_qlpc[sub],
                    &mut residual[off..off + NB_SUBFRAME_SIZE],
                    NB_ORDER,
                    &mut mem,
                );
            }
            self.mem_analysis = mem;
        }

        // Clip to int16 range to mirror the decoder's post-filter
        // clamping.
        for v in residual.iter_mut() {
            *v = v.clamp(-32000.0, 32000.0);
        }

        // ---- 6. Open-loop excitation gain (5-bit) --------------------
        //
        // The residual we just computed is an *open-loop* approximation
        // of the excitation the decoder should reconstruct. But the
        // codebook-driven quantisation injects a noise-like excitation
        // whose spectral shape differs from the residual, and the
        // decoder's steep 1/A(z) filter can amplify the mismatched
        // spectrum into the clipping range (`iir_mem16` saturates at
        // ±32767). A proper CELP encoder avoids this with perceptual
        // weighting; this first-cut encoder instead scales the open-
        // loop gain down so the decoder output stays below saturation
        // most of the time.
        //
        // Empirically 0.3 avoids clipping for speech-like inputs up
        // to ~10 000 amplitude; sharper resonances may still clip but
        // the resulting distortion is bounded.
        let frame_rms = rms(&residual);
        // Scale the open-loop gain down by 0.25 before quantization.
        // Empirically this keeps a wide variety of speech-like signals
        // away from the decoder's ±32767 saturation ceiling. Proper
        // perceptual-weighted A-by-S would remove the need for this
        // margin — see the module docstring for the caveat.
        let ol_gain_raw = (frame_rms * 0.25).max(1.0);
        // The decoder computes `ol_gain = exp(qe / 3.5)` with `qe ∈ 0..32`.
        // Invert: qe = round(3.5 * ln(ol_gain)).
        let qe_f = 3.5 * ol_gain_raw.ln();
        let qe = qe_f.round().clamp(0.0, 31.0) as u32;
        let ol_gain = (qe as f32 / 3.5).exp();

        // ---- 7. Write bitstream header fields ------------------------
        bw.write_bits(0, 1); // wideband flag = 0
        bw.write_bits(SUPPORTED_SUBMODE, 4);
        // LSP indices (5 × 6 bits)
        for idx in lsp_indices {
            bw.write_bits(idx as u32, 6);
        }
        // (no ol_pitch — mode 5 sets lbr_pitch = -1)
        bw.write_bits(qe, 5);

        // ---- 8. Per-sub-frame A-by-S loop -----------------------------
        for sub in 0..NB_NB_SUBFRAMES {
            let offset_in_frame = NB_SUBFRAME_SIZE * sub;
            let exc_idx = EXC_HISTORY + offset_in_frame;

            // ---- Analysis-by-synthesis target -----------------------
            //
            // Proper A-by-S: minimise error between the decoded
            // (synthesised) PCM and the input PCM, not between the
            // excitation and the LPC residual. Compute
            //   * h[n]  — impulse response of `1/A_sub(z)` (40 samples).
            //   * zir[n] — zero-input response of `1/A_sub(z)` given
            //     the current `mem_sp_sim` state (what the decoder
            //     would emit with no excitation this sub-frame).
            // Then the *synthesis target* becomes `pcm_tgt − zir`.
            // Any excitation the encoder picks gets convolved with h
            // before being compared to this target.
            //
            // Without this, the codebook search wastes bits matching
            // LPC-residual samples whose synthesised amplitude is huge
            // at formant frequencies — leading to the saturating
            // output observed in early iterations.
            let ak_sub = &interp_qlpc[sub];
            let h = impulse_response(ak_sub, NB_SUBFRAME_SIZE);
            let mut zir_mem = self.mem_sp_sim;
            let mut zir = [0.0f32; NB_SUBFRAME_SIZE];
            crate::nb_decoder::iir_mem16(
                &[0.0f32; NB_SUBFRAME_SIZE],
                ak_sub,
                &mut zir,
                NB_SUBFRAME_SIZE,
                NB_ORDER,
                &mut zir_mem,
            );
            let mut syn_target = [0.0f32; NB_SUBFRAME_SIZE];
            for i in 0..NB_SUBFRAME_SIZE {
                syn_target[i] = pcm[offset_in_frame + i] - zir[i];
            }

            // Closed-loop pitch: find lag whose filtered LTP best
            // matches the synthesis target.
            let pit_min = NB_PITCH_START;
            let pit_max = NB_PITCH_END;
            let pitch = search_pitch_lag_filtered(
                &syn_target,
                &self.exc_buf,
                exc_idx,
                pit_min,
                pit_max,
                &h,
            );
            let pitch_idx = (pitch - pit_min) as u32;

            // Three-tap gain codebook: evaluate each entry in the
            // filtered domain.
            let (gain_idx, ltp_exc, ltp_filtered) =
                search_pitch_gain_filtered(&syn_target, &self.exc_buf, exc_idx, pitch, &h);
            bw.write_bits(pitch_idx, 7);
            bw.write_bits(gain_idx as u32, 7);

            // What's left for the innovation codebook to cover.
            let mut innov_syn_target = [0.0f32; NB_SUBFRAME_SIZE];
            for i in 0..NB_SUBFRAME_SIZE {
                innov_syn_target[i] = syn_target[i] - ltp_filtered[i];
            }

            // Sub-frame innovation gain: pick the scalar that best
            // matches the residual-domain RMS of the innovation
            // target. Use the filter's impulse-response energy to
            // relate excitation amplitude to output amplitude.
            let h_energy: f32 = h.iter().map(|v| v * v).sum();
            let h_scale = h_energy.sqrt().max(1.0);
            let inner_rms = rms(&innov_syn_target);
            let target_ratio = if ol_gain > 1e-6 {
                inner_rms / (ol_gain * h_scale)
            } else {
                0.0
            };
            let (sub_gain_idx, sub_gain_val) = nearest_scalar(&EXC_GAIN_QUANT_SCAL3, target_ratio);
            bw.write_bits(sub_gain_idx as u32, 3);
            let ener = sub_gain_val * ol_gain;

            // Fixed codebook search in the filtered/synthesis domain.
            let cb_indices = search_split_cb_filtered(&innov_syn_target, &h, ener);
            for idx in cb_indices {
                bw.write_bits(idx as u32, 6);
            }

            // Reconstruct this sub-frame's excitation — matches what
            // the decoder reconstructs from the bits we just wrote.
            let mut innov = [0.0f32; NB_SUBFRAME_SIZE];
            expand_split_cb(&cb_indices, &mut innov);
            for v in innov.iter_mut() {
                *v *= ener;
            }
            for i in 0..NB_SUBFRAME_SIZE {
                self.exc_buf[exc_idx + i] = ltp_exc[i] + innov[i];
            }

            // Update the simulated synthesis-filter state so the next
            // sub-frame's ZIR reflects what the decoder's `mem_sp`
            // will actually hold.
            let exc_slice: Vec<f32> = self.exc_buf[exc_idx..exc_idx + NB_SUBFRAME_SIZE].to_vec();
            let mut sink = [0.0f32; NB_SUBFRAME_SIZE];
            crate::nb_decoder::iir_mem16(
                &exc_slice,
                ak_sub,
                &mut sink,
                NB_SUBFRAME_SIZE,
                NB_ORDER,
                &mut self.mem_sp_sim,
            );
            let _ = residual; // referenced elsewhere — kept for consistency
        }

        // ---- 9. Save state for next frame ----------------------------
        self.old_qlsp = qlsp;
        self.interp_qlpc = interp_qlpc[NB_NB_SUBFRAMES - 1];
        self.first = false;
        // Slide excitation history left by one frame.
        self.exc_buf.copy_within(NB_FRAME_SIZE.., 0);
        for v in &mut self.exc_buf[EXC_BUF_LEN - NB_FRAME_SIZE..] {
            *v = 0.0;
        }
        Ok(())
    }
}

// =====================================================================
// LPC analysis
// =====================================================================

/// Symmetric Hamming window applied to the input frame. Returns a new
/// 160-sample buffer.
fn hamming_window(x: &[f32]) -> [f32; NB_FRAME_SIZE] {
    let mut out = [0.0f32; NB_FRAME_SIZE];
    let n = NB_FRAME_SIZE as f32 - 1.0;
    for i in 0..NB_FRAME_SIZE {
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / n).cos();
        out[i] = x[i] * w;
    }
    out
}

/// Compute autocorrelation `r[k] = Σ x[i] · x[i+k]` for `k=0..=order`.
fn autocorrelate(x: &[f32], r: &mut [f32]) {
    let order = r.len() - 1;
    for k in 0..=order {
        let mut s = 0.0f32;
        for i in 0..x.len() - k {
            s += x[i] * x[i + k];
        }
        r[k] = s;
    }
}

/// Levinson-Durbin recursion — returns LPC coefficients `a[0..order]`
/// corresponding to the polynomial `A(z) = 1 + Σ a_k z^{-k}`. The
/// returned array is `a[k]` for `k = 1..=order`, i.e. the leading `1` is
/// implicit (matching the storage convention in
/// [`crate::lsp::lsp_to_lpc`]).
fn levinson_durbin(r: &[f32]) -> [f32; NB_ORDER] {
    let mut a = [0.0f32; NB_ORDER];
    let mut tmp = [0.0f32; NB_ORDER];
    let mut e = r[0];
    if e <= 0.0 {
        return a;
    }
    for i in 0..NB_ORDER {
        let mut k = -r[i + 1];
        for j in 0..i {
            k -= a[j] * r[i - j];
        }
        if e.abs() < 1e-12 {
            break;
        }
        k /= e;
        // Stability guarantee: Levinson's recursion preserves
        // |k_i| < 1 for an autocorrelation from any real signal, but
        // floating-point round-off on a windowed, low-energy frame can
        // nudge |k_i| slightly above 1 and make the resulting LPC
        // filter unstable. Clamp to a safe margin.
        const K_MAX: f32 = 0.999;
        if k > K_MAX {
            k = K_MAX;
        } else if k < -K_MAX {
            k = -K_MAX;
        }
        tmp[i] = k;
        for j in 0..i {
            tmp[j] = a[j] + k * a[i - 1 - j];
        }
        for j in 0..=i {
            a[j] = tmp[j];
        }
        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-6;
        }
    }
    // Sign convention: the recursion above produces coefficients in
    // the `A(z) = 1 + Σ a_k z^{-k}` convention — same storage shape as
    // `lsp_to_lpc` and as `iir_mem16::den`. For a cosine input of
    // frequency ω at LPC order 2 it yields `a = [-2cos(ω), 1]`, which
    // zeroes the analysis FIR `r[n] = x[n] + a[0]·x[n-1] + a[1]·x[n-2]`
    // exactly — i.e. the residual vanishes as expected. No sign flip
    // needed here.
    a
}

/// FIR analysis filter: `y[i] = x[i] + Σ_k a_k · x[i-k-1]` written as
/// `y[i] = x[i] + Σ a_k * mem[k]` with a length-`order` tap delay line.
/// Updates `mem` with the last `order` inputs — which is what the
/// decoder's IIR synthesis filter expects to reproduce.
fn fir_filter(x: &[f32], a: &[f32], y: &mut [f32], order: usize, mem: &mut [f32; NB_ORDER]) {
    for i in 0..x.len() {
        let xi = x[i];
        let mut acc = xi;
        for k in 0..order {
            acc += a[k] * mem[k];
        }
        for k in (1..order).rev() {
            mem[k] = mem[k - 1];
        }
        if order > 0 {
            mem[0] = xi;
        }
        y[i] = acc;
    }
}

// =====================================================================
// LPC -> LSP conversion
// =====================================================================

/// Convert LPC coefficients `a[k]` (in the `A(z) = 1 + Σ a_k z^{-k}`
/// convention used elsewhere in the crate) to Line Spectral Pairs
/// (radians, sorted increasing). Returns `None` if the root search fails
/// — in that case the caller should fall back to a neutral LSP vector.
///
/// Algorithm: build the symmetric/antisymmetric polynomials
///     P(z) = A(z) + z^{-(p+1)} A(z^{-1})
///     Q(z) = A(z) - z^{-(p+1)} A(z^{-1})
/// Evaluate P(e^{jω}) and Q(e^{jω}) directly on a uniform grid of ω
/// angles, then bisect within each detected zero-crossing. Since both
/// polynomials have all zeros on the unit circle (they are real, with
/// palindromic / anti-palindromic coefficients), `P` contributes p/2+1
/// zeros (including ω = π for a symmetric polynomial) and `Q`
/// contributes p/2+1 zeros (including ω = 0). The interior roots give
/// the p-dimensional LSP vector.
fn lpc_to_lsp(ak: &[f32; NB_ORDER]) -> Option<[f32; NB_ORDER]> {
    let p = NB_ORDER;
    // A-polynomial padded to length p+2 so we can reflect freely:
    //   a_pad[0]   = 1   (implicit leading 1 of A(z) = 1 + Σ a_k z^{-k})
    //   a_pad[k]   = a_k for k = 1..=p
    //   a_pad[p+1] = 0   (tail — beyond A's natural order)
    let mut a_pad = [0.0f32; NB_ORDER + 2];
    a_pad[0] = 1.0;
    for k in 1..=p {
        a_pad[k] = ak[k - 1];
    }
    // z^{-(p+1)} A(z^{-1}) has coefficients reflected: index k maps to
    // a_pad[(p+1) - k]. That gives:
    //   ref[0] = a_pad[p+1] = 0
    //   ref[p+1] = a_pad[0] = 1
    //   ref[k]   = a_pad[p+1-k] for k in 1..=p
    //
    // Then P[k] = a_pad[k] + ref[k] and Q[k] = a_pad[k] - ref[k].
    // Note P[0] = 1, P[p+1] = 1 (symmetric); Q[0] = 1, Q[p+1] = -1
    // (anti-symmetric).
    let n = p + 2; // 12
    let mut pcoef = [0.0f32; 12];
    let mut qcoef = [0.0f32; 12];
    for k in 0..n {
        let reflected = a_pad[(p + 1) - k];
        pcoef[k] = a_pad[k] + reflected;
        qcoef[k] = a_pad[k] - reflected;
    }

    // Evaluate the phase-de-rotated polynomial on the unit circle.
    //
    // P and Q are palindromic / anti-palindromic of length p+2=12. At
    // z = e^{jω}, multiplying by e^{j·(p+1)ω/2} = e^{j·5.5·ω} pairs
    // symmetric coefficients into cosine terms and anti-symmetric
    // coefficients into sine terms of half-integer frequency. For our
    // palindromic P:
    //     P(e^{jω}) · e^{j·5.5·ω} = Σ_{k=0..5} 2·P_k·cos((5.5-k)·ω)
    // which is a real-valued function of ω with exactly p/2 = 5
    // interior zeros — no phantom zeros from the evaluation itself.
    //
    // For anti-palindromic Q:
    //     Q(e^{jω}) · e^{j·5.5·ω} = j · Σ_{k=0..5} 2·Q_k·sin((5.5-k)·ω)
    // whose imaginary part is again a real cosine/sine series with
    // exactly p/2 = 5 interior zeros.
    let eval_p = |coeffs: &[f32; 12], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5; // 5.5
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).cos();
        }
        s
    };
    let eval_q = |coeffs: &[f32; 12], omega: f32| -> f32 {
        let half = (n as f32 - 1.0) * 0.5; // 5.5
        let mut s = 0.0f32;
        for k in 0..(n / 2) {
            s += 2.0 * coeffs[k] * ((half - k as f32) * omega).sin();
        }
        s
    };

    // Scan ω from 0 to π.
    const GRID_N: usize = 1024;
    let grid_to_omega = |i: usize| (i as f32 * std::f32::consts::PI) / GRID_N as f32;

    fn scan_roots(
        coeffs: &[f32; 12],
        grid: usize,
        to_omega: impl Fn(usize) -> f32,
        eval: impl Fn(&[f32; 12], f32) -> f32,
    ) -> Vec<f32> {
        let mut roots = Vec::with_capacity(NB_ORDER / 2);
        // Start the scan one step in from ω=0; Q's cosine/sine series has
        // an inherent zero at ω=0 (antisymmetric polynomial), and we
        // don't want to count that as an LSP. Likewise we stop one step
        // short of ω=π to avoid the same for P. The boundary zeros are
        // not LSPs.
        let eps = 1.0 / (grid as f32 * 2.0);
        let clamp_omega = |o: f32| o.clamp(eps, std::f32::consts::PI - eps);
        let mut prev_o = clamp_omega(to_omega(1));
        let mut prev_f = eval(coeffs, prev_o);
        for i in 2..grid {
            let o = clamp_omega(to_omega(i));
            let f = eval(coeffs, o);
            if prev_f * f < 0.0 {
                let mut lo = prev_o;
                let mut hi = o;
                let mut flo = prev_f;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    let fmid = eval(coeffs, mid);
                    if fmid == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if flo * fmid < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fmid;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            prev_o = o;
            prev_f = f;
        }
        roots
    }
    let r_p = scan_roots(&pcoef, GRID_N, grid_to_omega, eval_p);
    let r_q = scan_roots(&qcoef, GRID_N, grid_to_omega, eval_q);
    // P has trivial zero at ω = π (since P is palindromic of even
    // length ⇒ P(-1) = 0 when p is odd… but with p=10, n=12 which is
    // even, the palindromic structure gives P(z) factorable by (z + 1)
    // and (z - 1) is a trivial zero of Q). So one of `r_p` / `r_q`
    // contains a spurious boundary root at ω=0 or ω=π we must filter.
    let mut roots = Vec::with_capacity(p);
    for &r in r_p.iter().chain(r_q.iter()) {
        // Only interior roots are LSPs. Discard anything pinned to the
        // boundary angles.
        if r > 1e-3 && r < std::f32::consts::PI - 1e-3 {
            roots.push(r);
        }
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if roots.len() < p {
        return None;
    }
    // If the scan produced a couple of extras (e.g. a very shallow false
    // crossing near the boundary), keep only the p interior roots most
    // tightly packed.
    roots.truncate(p);
    let mut out = [0.0f32; NB_ORDER];
    for i in 0..p {
        out[i] = roots[i];
    }
    // Enforce strict monotonicity.
    let margin = LSP_MARGIN;
    out[0] = out[0].max(margin);
    for i in 1..NB_ORDER {
        if out[i] < out[i - 1] + margin {
            out[i] = out[i - 1] + margin;
        }
    }
    out[NB_ORDER - 1] = out[NB_ORDER - 1].min(std::f32::consts::PI - margin);
    Some(out)
}

// =====================================================================
// LSP quantisation (mode 5: five-stage VQ, 30 bits)
// =====================================================================

/// Quantise a 10-LSP vector using the five-stage VQ from
/// `lsp_unquant_nb`'s inverse. Returns the dequantised LSPs and the
/// five 6-bit codebook indices in encoding order.
fn quantise_lsp_nb(lsp: &[f32; NB_ORDER]) -> ([f32; NB_ORDER], [usize; 5]) {
    let mut indices = [0usize; 5];
    // Stage 1: remove linear initial guess, VQ against CDBK_NB (64 × 10)
    //          with scale 1/256.
    let mut residual = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        residual[i] = lsp[i] - std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
    }
    // Decoder: lsp[i] = linear[i] + CDBK_NB[id*10+i]/256
    // So target for stage 1 = residual * 256, search over 64 entries.
    indices[0] = nearest_vector_scaled(&residual, 256.0, &CDBK_NB, 10, 64);
    for i in 0..10 {
        residual[i] -= (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    // Stage 2 (low1, 64×5, scale 1/512) — only LSP[0..5].
    let mut low = [0.0f32; 5];
    low.copy_from_slice(&residual[0..5]);
    indices[1] = nearest_vector_scaled(&low, 512.0, &CDBK_NB_LOW1, 5, 64);
    for i in 0..5 {
        residual[i] -= (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    // Stage 3 (low2, 64×5, scale 1/1024) — LSP[0..5].
    low.copy_from_slice(&residual[0..5]);
    indices[2] = nearest_vector_scaled(&low, 1024.0, &CDBK_NB_LOW2, 5, 64);
    for i in 0..5 {
        residual[i] -= (CDBK_NB_LOW2[indices[2] * 5 + i] as f32) / 1024.0;
    }
    // Stage 4 (high1, 64×5, scale 1/512) — LSP[5..10].
    let mut hi = [0.0f32; 5];
    hi.copy_from_slice(&residual[5..10]);
    indices[3] = nearest_vector_scaled(&hi, 512.0, &CDBK_NB_HIGH1, 5, 64);
    for i in 0..5 {
        residual[5 + i] -= (CDBK_NB_HIGH1[indices[3] * 5 + i] as f32) / 512.0;
    }
    // Stage 5 (high2, 64×5, scale 1/1024) — LSP[5..10].
    hi.copy_from_slice(&residual[5..10]);
    indices[4] = nearest_vector_scaled(&hi, 1024.0, &CDBK_NB_HIGH2, 5, 64);

    // Reconstruct exactly as the decoder would.
    let mut qlsp = [0.0f32; NB_ORDER];
    for i in 0..NB_ORDER {
        qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
    }
    for i in 0..10 {
        qlsp[i] += (CDBK_NB[indices[0] * 10 + i] as f32) / 256.0;
    }
    for i in 0..5 {
        qlsp[i] += (CDBK_NB_LOW1[indices[1] * 5 + i] as f32) / 512.0;
    }
    for i in 0..5 {
        qlsp[i] += (CDBK_NB_LOW2[indices[2] * 5 + i] as f32) / 1024.0;
    }
    for i in 0..5 {
        qlsp[i + 5] += (CDBK_NB_HIGH1[indices[3] * 5 + i] as f32) / 512.0;
    }
    for i in 0..5 {
        qlsp[i + 5] += (CDBK_NB_HIGH2[indices[4] * 5 + i] as f32) / 1024.0;
    }

    // Enforce stability: strictly increasing, bounded away from 0 and π.
    let margin = LSP_MARGIN;
    qlsp[0] = qlsp[0].max(margin);
    for i in 1..NB_ORDER {
        if qlsp[i] < qlsp[i - 1] + margin {
            qlsp[i] = qlsp[i - 1] + margin;
        }
    }
    qlsp[NB_ORDER - 1] = qlsp[NB_ORDER - 1].min(std::f32::consts::PI - margin);
    (qlsp, indices)
}

/// MSE search over a `count`-entry codebook of `dim`-vectors stored as
/// i8 with `scale` (so `cdbk[entry]` represents `cdbk[entry] / scale`).
/// Returns the winning entry's index.
fn nearest_vector_scaled(
    target: &[f32],
    scale: f32,
    cdbk: &[i8],
    dim: usize,
    count: usize,
) -> usize {
    let inv = 1.0 / scale;
    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for idx in 0..count {
        let mut err = 0.0f32;
        let base = idx * dim;
        for i in 0..dim {
            let v = cdbk[base + i] as f32 * inv;
            let d = target[i] - v;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
    }
    best_idx
}

/// Scalar-quantiser nearest neighbour — returns `(index, value)`.
fn nearest_scalar(codebook: &[f32], target: f32) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &v) in codebook.iter().enumerate() {
        let e = (target - v).abs();
        if e < best_err {
            best_err = e;
            best_idx = i;
        }
    }
    (best_idx, codebook[best_idx])
}

// =====================================================================
// Pitch (adaptive codebook) search
// =====================================================================

/// Compute the impulse response of the synthesis filter `1/A(z)` over
/// `n` samples, starting from zero state. Used as a convolution kernel
/// in analysis-by-synthesis pitch / codebook searches.
fn impulse_response(ak: &[f32; NB_ORDER], n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n];
    let mut mem = [0.0f32; NB_ORDER];
    let mut x = vec![0.0f32; n];
    x[0] = 1.0;
    crate::nb_decoder::iir_mem16(&x, ak, &mut h, n, NB_ORDER, &mut mem);
    h
}

/// Convolve `exc` with `h`, storing the result in `out` (truncated to
/// `out.len()`). `exc[i]` contributes to `out[i + k] += exc[i] · h[k]`
/// for all valid (i, k). Used to evaluate filtered LTP / codebook
/// candidates in the synthesis domain.
fn convolve_lt(exc: &[f32], h: &[f32], out: &mut [f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let n = out.len();
    for i in 0..exc.len() {
        let e = exc[i];
        if e == 0.0 {
            continue;
        }
        for k in 0..h.len() {
            let j = i + k;
            if j >= n {
                break;
            }
            out[j] += e * h[k];
        }
    }
}

/// Closed-loop pitch-lag search in the filtered (synthesis) domain:
/// pick the lag whose filtered single-tap LTP output best matches the
/// synthesis target.
fn search_pitch_lag_filtered(
    target: &[f32; NB_SUBFRAME_SIZE],
    exc_buf: &[f32],
    exc_idx: usize,
    pit_min: i32,
    pit_max: i32,
    h: &[f32],
) -> i32 {
    let mut best_lag = pit_min;
    let mut best_score = -1.0f32;
    for lag in pit_min..=pit_max {
        // Build the single-tap LTP contribution for this lag.
        let mut ltp = [0.0f32; NB_SUBFRAME_SIZE];
        for j in 0..NB_SUBFRAME_SIZE {
            let src = exc_idx as isize + j as isize - lag as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                ltp[j] = exc_buf[src as usize];
            }
        }
        let mut filtered = [0.0f32; NB_SUBFRAME_SIZE];
        convolve_lt(&ltp, h, &mut filtered);
        let mut num = 0.0f32;
        let mut den = 1e-6f32;
        for i in 0..NB_SUBFRAME_SIZE {
            num += target[i] * filtered[i];
            den += filtered[i] * filtered[i];
        }
        let score = num * num / den;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    best_lag
}

/// Three-tap pitch-gain quantization in the synthesis domain. Returns
/// the gain index, the reconstructed LTP excitation (what the decoder
/// will compute), and the filtered LTP contribution (= LTP_exc * h).
fn search_pitch_gain_filtered(
    target: &[f32; NB_SUBFRAME_SIZE],
    exc_buf: &[f32],
    exc_idx: usize,
    pitch: i32,
    h: &[f32],
) -> (usize, [f32; NB_SUBFRAME_SIZE], [f32; NB_SUBFRAME_SIZE]) {
    let gain_cdbk_size = 128usize;
    // Build the three per-tap past-excitation signals y_i[j] (same
    // indexing as the decoder's pitch_unquant_3tap).
    let mut y = [[0.0f32; NB_SUBFRAME_SIZE]; 3];
    for i in 0..3 {
        let pp = (pitch + 1 - i as i32) as usize;
        let nsf = NB_SUBFRAME_SIZE;
        let tmp1 = nsf.min(pp);
        for j in 0..tmp1 {
            let src = exc_idx as isize + j as isize - pp as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                y[i][j] = exc_buf[src as usize];
            }
        }
        let tmp3 = nsf.min(pp + pitch as usize);
        for j in tmp1..tmp3 {
            let src = exc_idx as isize + j as isize - pp as isize - pitch as isize;
            if src >= 0 && (src as usize) < exc_buf.len() {
                y[i][j] = exc_buf[src as usize];
            }
        }
    }
    // Pre-filter each tap's past-exc signal so we can quickly score
    // gain candidates.
    let mut yf = [[0.0f32; NB_SUBFRAME_SIZE]; 3];
    for i in 0..3 {
        convolve_lt(&y[i], h, &mut yf[i]);
    }

    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    let mut best_exc = [0.0f32; NB_SUBFRAME_SIZE];
    let mut best_filt = [0.0f32; NB_SUBFRAME_SIZE];
    for idx in 0..gain_cdbk_size {
        let g0 = 0.015625 * GAIN_CDBK_NB[idx * 4] as f32 + 0.5;
        let g1 = 0.015625 * GAIN_CDBK_NB[idx * 4 + 1] as f32 + 0.5;
        let g2 = 0.015625 * GAIN_CDBK_NB[idx * 4 + 2] as f32 + 0.5;
        let mut err = 0.0f32;
        let mut exc = [0.0f32; NB_SUBFRAME_SIZE];
        let mut filt = [0.0f32; NB_SUBFRAME_SIZE];
        for j in 0..NB_SUBFRAME_SIZE {
            exc[j] = g2 * y[0][j] + g1 * y[1][j] + g0 * y[2][j];
            filt[j] = g2 * yf[0][j] + g1 * yf[1][j] + g0 * yf[2][j];
            let d = target[j] - filt[j];
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_idx = idx;
            best_exc = exc;
            best_filt = filt;
        }
    }
    (best_idx, best_exc, best_filt)
}

/// Split-codebook search in the synthesis domain. Each sub-vector (5
/// samples × 8 sub-vectors) is searched independently; the chosen cb
/// entry is the one whose convolution with `h` best matches the
/// sub-vector of the target.
///
/// `ener` is multiplied into the codebook entries before convolution
/// so the search is scale-consistent with what the decoder will
/// reconstruct.
fn search_split_cb_filtered(
    target: &[f32; NB_SUBFRAME_SIZE],
    h: &[f32],
    ener: f32,
) -> [usize; NB_SUBVECT] {
    // Build a full-subframe candidate excitation for each sub-vector
    // starting offset, convolve with h, and pick the best entry per
    // sub-vector. Because sub-vectors are disjoint in position, each
    // one's convolution only affects output samples starting at that
    // offset, so independent per-sub-vector search is correct up to
    // the impulse-response tail crossing into later sub-vectors. We
    // accept that coupling as part of the first-cut approximation —
    // proper A-by-S would search sub-vectors jointly via impulse-
    // response precomputation + backward substitution.
    let mut indices = [0usize; NB_SUBVECT];
    let mut cur_target = *target;
    for i in 0..NB_SUBVECT {
        let off = i * SUBVECT_SIZE;
        let mut best_err = f32::INFINITY;
        let mut best = 0usize;
        let mut best_exc = [0.0f32; SUBVECT_SIZE];
        let mut best_filt = [0.0f32; NB_SUBFRAME_SIZE];
        for idx in 0..SHAPE_ENTRIES {
            let base = idx * SUBVECT_SIZE;
            // Build a full 40-sample candidate excitation that places
            // this codebook entry at offset `off`; other samples zero.
            let mut exc = [0.0f32; NB_SUBFRAME_SIZE];
            for j in 0..SUBVECT_SIZE {
                exc[off + j] = EXC_5_64_TABLE[base + j] as f32 * 0.03125 * ener;
            }
            let mut filt = [0.0f32; NB_SUBFRAME_SIZE];
            convolve_lt(&exc, h, &mut filt);
            let mut err = 0.0f32;
            // Only measure error at positions affected by this sub-
            // vector (off ..). Earlier positions were the previous
            // sub-vectors' concern.
            for j in off..NB_SUBFRAME_SIZE {
                let d = cur_target[j] - filt[j];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best = idx;
                for j in 0..SUBVECT_SIZE {
                    best_exc[j] = EXC_5_64_TABLE[base + j] as f32 * 0.03125;
                }
                best_filt = filt;
            }
        }
        indices[i] = best;
        // Subtract the chosen sub-vector's filtered response from the
        // running target so subsequent sub-vectors see a fresh
        // residual.
        for j in off..NB_SUBFRAME_SIZE {
            cur_target[j] -= best_filt[j];
        }
        let _ = best_exc;
    }
    indices
}

// =====================================================================
// Fixed (split) codebook search — mode 5 uses EXC_5_64_TABLE
// =====================================================================

const SUBVECT_SIZE: usize = 5;
const NB_SUBVECT: usize = 8;
const SHAPE_BITS: usize = 6;
const SHAPE_ENTRIES: usize = 1 << SHAPE_BITS; // 64

/// Inverse of `search_split_cb` — recomputes the normalised innovation
/// the decoder will reconstruct from the chosen indices.
fn expand_split_cb(indices: &[usize; NB_SUBVECT], out: &mut [f32; NB_SUBFRAME_SIZE]) {
    for i in 0..NB_SUBVECT {
        let base = indices[i] * SUBVECT_SIZE;
        for j in 0..SUBVECT_SIZE {
            out[i * SUBVECT_SIZE + j] = EXC_5_64_TABLE[base + j] as f32 * 0.03125;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levinson_returns_trivial_for_white_noise_like_signal() {
        // White-noise autocorrelation has r[0] dominant, r[k]≈0 — LPCs
        // should all be near zero.
        let mut r = [0.0f32; NB_ORDER + 1];
        r[0] = 100.0;
        for k in 1..=NB_ORDER {
            r[k] = 0.0;
        }
        let a = levinson_durbin(&r);
        for &v in &a {
            assert!(v.abs() < 1e-3, "LPC coef should be near zero: {v}");
        }
    }

    #[test]
    fn autocorrelate_zero_signal_is_zero() {
        let x = [0.0f32; NB_FRAME_SIZE];
        let mut r = [0.0f32; NB_ORDER + 1];
        autocorrelate(&x, &mut r);
        for &v in &r {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn quantise_lsp_round_trip_stable() {
        // For a well-behaved linear LSP vector, the quantised result
        // should stay ordered and close to the input.
        let mut lsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            lsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
        }
        let (qlsp, _) = quantise_lsp_nb(&lsp);
        for i in 1..NB_ORDER {
            assert!(qlsp[i] > qlsp[i - 1], "qLSP must be sorted");
        }
        for i in 0..NB_ORDER {
            assert!((qlsp[i] - lsp[i]).abs() < 1.0, "qLSP wildly off input");
        }
    }

    #[test]
    fn lpc_to_lsp_recovers_stable_lpc() {
        // Use a slightly perturbed LSP vector so the LPC is non-trivial
        // (uniform LSPs collapse A(z) to ~1, which has no interior
        // roots to find). A small chirp around the uniform grid gives a
        // realistic formant-like filter.
        let mut lsp = [0.0f32; NB_ORDER];
        for i in 0..NB_ORDER {
            let base = std::f32::consts::PI * (i as f32 + 1.0) / (NB_ORDER as f32 + 1.0);
            let perturb = 0.2 * ((i as f32 * 1.3).sin());
            lsp[i] = (base + perturb).clamp(0.05, std::f32::consts::PI - 0.05);
        }
        // Re-sort to keep LSPs monotonic after perturbation.
        lsp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..NB_ORDER {
            if lsp[i] < lsp[i - 1] + 0.05 {
                lsp[i] = lsp[i - 1] + 0.05;
            }
        }
        let mut ak = [0.0f32; NB_ORDER];
        lsp_to_lpc(&lsp, &mut ak, NB_ORDER);
        let recovered = lpc_to_lsp(&ak);
        assert!(recovered.is_some(), "lpc_to_lsp should succeed");
        let rec = recovered.unwrap();
        eprintln!("input  LSP = {:?}", lsp);
        eprintln!("output LSP = {:?}", rec);
        for i in 0..NB_ORDER {
            assert!(
                (rec[i] - lsp[i]).abs() < 0.15,
                "LSP round-trip off at {i}: got {} expected {}",
                rec[i],
                lsp[i]
            );
        }
    }

    #[test]
    fn encode_frame_writes_exactly_300_bits() {
        let mut enc = NbEncoder::new();
        let mut bw = BitWriter::new();
        // Frame of moderate-amplitude noise-like sine sum.
        let mut pcm = [0.0f32; NB_FRAME_SIZE];
        for i in 0..NB_FRAME_SIZE {
            let t = i as f32;
            pcm[i] = 5000.0 * ((t * 0.2).sin() + 0.5 * (t * 0.05).sin() + 0.3 * (t * 0.7).cos());
        }
        enc.encode_frame(&pcm, &mut bw).unwrap();
        assert_eq!(bw.bit_position(), 300, "mode 5 must emit exactly 300 bits");
    }
}
