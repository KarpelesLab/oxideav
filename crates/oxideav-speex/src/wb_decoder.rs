//! Speex wideband (sub-band CELP) decoder — float path.
//!
//! Mirrors `sb_decode` in `libspeex/sb_celp.c`. A wideband frame is a
//! narrowband frame (the "low band", 0–4 kHz, 8 kHz sampled) immediately
//! followed by an optional extension that carries the 4–8 kHz "high
//! band". The output is reconstructed at 16 kHz by running both bands
//! through a 64-tap QMF synthesis bank.
//!
//! Bitstream layout (as produced by the reference encoder):
//!
//! ```text
//!   +---- NB frame (m bits, sub-mode dependent) ----+
//!   | 1 bit  wideband-bit                           |
//!   | 3 bit  SB sub-mode id  (== wb_submodeN)       |
//!   | rest   SB-CELP parameters (LSP + per-subframe |
//!   |        gain + optional innovation)            |
//!   +-----------------------------------------------+
//! ```
//!
//! If the wideband bit is 0, the packet is narrowband-only and the
//! decoder emits 160 low-band samples upsampled to 320 via a zero high
//! band through QMF — matching the reference `submodeID == 0` path.
//!
//! The high-band synthesis state carried across frames:
//!   * 8th-order quantized LSPs from the previous frame,
//!   * 8th-order interpolated LPC of the last sub-frame,
//!   * 8-tap LPC synthesis filter memory,
//!   * 40-sample previous-subframe excitation (for the 1-subframe
//!     delay baked into the reference),
//!   * 64-tap QMF synthesis memory for each of the low and high sides.
//!
//! We don't implement DTX / packet loss concealment (the reference
//! `sb_decode_lost` path) — a lost wideband frame falls back to low-band
//! decode, which is what the NB layer already handles.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::lsp::{lsp_interpolate, lsp_to_lpc, lsp_unquant_high};
use crate::nb_decoder::{iir_mem16, NbDecoder, NB_NB_SUBFRAMES};
use crate::qmf::{qmf_synth, H0_PROTOTYPE, QMF_ORDER};
use crate::submodes::SplitCbParams;
use crate::wb_submodes::{wb_submode, WbInnov};

/// Order of the high-band LPC synthesis filter. Mirrors
/// `sb_wb_mode.lpcSize = 8`.
pub const WB_LPC_ORDER: usize = 8;
/// Number of SB-CELP sub-frames per wideband frame (`frameSize /
/// subframeSize = 160 / 40 = 4`).
pub const WB_NB_SUBFRAMES: usize = 4;
/// SB-CELP sub-frame size (high-band samples per sub-frame at 8 kHz
/// decimated rate).
pub const WB_SUBFRAME_SIZE: usize = 40;
/// Low-band frame size, in 8 kHz samples.
pub const WB_FRAME_SIZE: usize = 160;
/// Full-band (wideband) frame size, in 16 kHz samples.
pub const WB_FULL_FRAME_SIZE: usize = 320;
/// LSP margin for high-band LSP interpolation (reference sets this to
/// 0.05 radians — five times the NB margin since the high-band filter
/// is more prone to instability).
pub const LSP_MARGIN_HIGH: f32 = 0.05;
/// Folding gain for WB sub-mode 1 (`sb_wb_mode.folding_gain = 0.9`).
pub const FOLDING_GAIN: f32 = 0.9;

/// Stochastic gain quantization boundaries for SB-CELP (float path) —
/// mirrors `gc_quant_bound[16]` in `sb_celp.c`.
const GC_QUANT_BOUND: [f32; 16] = [
    0.97979, 1.28384, 1.68223, 2.20426, 2.88829, 3.78458, 4.95900, 6.49787, 8.51428, 11.15642,
    14.61846, 19.15484, 25.09895, 32.88761, 43.09325, 56.46588,
];

pub struct WbDecoder {
    nb: NbDecoder,
    old_qlsp: [f32; WB_LPC_ORDER],
    interp_qlpc: [f32; WB_LPC_ORDER],
    mem_sp: [f32; WB_LPC_ORDER],
    /// Previous sub-frame's excitation — fed into the synthesis filter
    /// on the next sub-frame (see the 1-sub-frame delay in `sb_celp.c`
    /// around `iir_mem16(st->excBuf, ...)`).
    prev_exc: [f32; WB_SUBFRAME_SIZE],
    /// QMF synthesis memory for the low-band branch.
    g0_mem: [f32; QMF_ORDER],
    /// QMF synthesis memory for the high-band branch.
    g1_mem: [f32; QMF_ORDER],
    first: bool,
}

impl Default for WbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WbDecoder {
    pub fn new() -> Self {
        // Reference `sb_decoder_init` starts old_qlsp at
        // `i * π / (order+1)` (same linear spread as NB).
        let mut old_qlsp = [0.0f32; WB_LPC_ORDER];
        for i in 0..WB_LPC_ORDER {
            old_qlsp[i] = std::f32::consts::PI * (i as f32 + 1.0) / (WB_LPC_ORDER as f32 + 1.0);
        }
        Self {
            nb: NbDecoder::new(),
            old_qlsp,
            interp_qlpc: [0.0; WB_LPC_ORDER],
            mem_sp: [0.0; WB_LPC_ORDER],
            prev_exc: [0.0; WB_SUBFRAME_SIZE],
            g0_mem: [0.0; QMF_ORDER],
            g1_mem: [0.0; QMF_ORDER],
            first: true,
        }
    }

    /// Decode one wideband Speex frame. Writes `WB_FULL_FRAME_SIZE`
    /// floats to `out`. On a truncated packet, returns `Error::Eof`
    /// after decoding whatever prefix was complete (the NB decoder
    /// does the same on the `m == 15` terminator — see `nb_decoder.rs`).
    pub fn decode_frame(&mut self, br: &mut BitReader, out: &mut [f32]) -> Result<()> {
        debug_assert_eq!(out.len(), WB_FULL_FRAME_SIZE);

        // ---- Low-band (narrowband) decode ---------------------------
        // The NB output goes into the first half of `out`. We pass the
        // NB decoder a local buffer rather than aliasing the slice so
        // the borrow checker stays happy.
        let mut low = [0.0f32; WB_FRAME_SIZE];
        self.nb.decode_frame(br, &mut low)?;
        out[..WB_FRAME_SIZE].copy_from_slice(&low);

        // Zero the high-band slot; it's written below if the wb layer
        // is present, otherwise QMF reads the zeros.
        for v in &mut out[WB_FRAME_SIZE..WB_FULL_FRAME_SIZE] {
            *v = 0.0;
        }

        // ---- Wideband-bit + sub-mode selector -----------------------
        let mut submode_id = 0u32;
        if br.bits_remaining() >= 1 {
            let wb_bit = br.read_u32(1)?;
            if wb_bit == 1 {
                if br.bits_remaining() < 3 {
                    return Err(Error::invalid("Speex WB: truncated sub-mode selector"));
                }
                submode_id = br.read_u32(3)?;
            }
        }

        let sm = if submode_id == 0 {
            None
        } else {
            match wb_submode(submode_id) {
                Some(sm) => Some(sm),
                None => {
                    return Err(Error::invalid(format!(
                        "Speex WB: invalid sub-mode id {submode_id}"
                    )));
                }
            }
        };

        // ---- Null / pass-through submode: QMF from zero high band --
        let Some(sm) = sm else {
            // Reference: run the previous-frame synthesis filter over
            // zero excitation once, then QMF. We skip that since a
            // silent high band is already a valid PLC output.
            let (low_in, high_in) = out.split_at_mut(WB_FRAME_SIZE);
            let mut full = vec![0.0f32; WB_FULL_FRAME_SIZE];
            qmf_synth(
                low_in,
                high_in,
                &H0_PROTOTYPE,
                &mut full,
                WB_FULL_FRAME_SIZE,
                QMF_ORDER,
                &mut self.g0_mem,
                &mut self.g1_mem,
            );
            out.copy_from_slice(&full);
            self.first = true;
            return Ok(());
        };

        // ---- High-band LSPs ----------------------------------------
        let mut qlsp = [0.0f32; WB_LPC_ORDER];
        lsp_unquant_high(&mut qlsp, WB_LPC_ORDER, br)?;
        if self.first {
            self.old_qlsp.copy_from_slice(&qlsp);
        }

        // Collect per-subframe context from the NB decoder — needed
        // for the folding excitation path and the filter-gain ratio.
        let low_pi_gain = *self.nb.pi_gain();
        let low_exc_rms = *self.nb.exc_rms();
        let low_innov = *self.nb.innov();

        // ---- Sub-frame loop ----------------------------------------
        let mut interp_qlsp = [0.0f32; WB_LPC_ORDER];
        let mut ak = [0.0f32; WB_LPC_ORDER];
        for sub in 0..WB_NB_SUBFRAMES {
            let offset = WB_SUBFRAME_SIZE * sub;

            // Interpolate LSPs, convert to LPC.
            lsp_interpolate(
                &self.old_qlsp,
                &qlsp,
                &mut interp_qlsp,
                WB_LPC_ORDER,
                sub,
                WB_NB_SUBFRAMES,
                LSP_MARGIN_HIGH,
            );
            lsp_to_lpc(&interp_qlsp, &mut ak, WB_LPC_ORDER);

            // Compute the response ratio between the low and high
            // filters in the middle of the band (4 kHz) — this is the
            // gain scaling that keeps the high band commensurate with
            // the low band's envelope.
            //
            //   rh = 1 + Σ_{even k} (ak[k+1] - ak[k])   (ω = π of the
            //                                           high filter)
            //   rl = low_pi_gain[sub]                   (ω = π of the
            //                                           low filter)
            //   filter_ratio = (rl + .01) / (rh + .01)
            let mut rh = 1.0f32;
            let mut k = 0;
            while k + 1 < WB_LPC_ORDER {
                rh += ak[k + 1] - ak[k];
                k += 2;
            }
            let rl = low_pi_gain[sub.min(NB_NB_SUBFRAMES - 1)];
            let filter_ratio = (rl + 0.01) / (rh + 0.01);

            // ---- Excitation recovery -------------------------------
            let mut exc = [0.0f32; WB_SUBFRAME_SIZE];
            match &sm.innov {
                WbInnov::Folding => {
                    // Reference: 5-bit `quant`, g = exp(.125 * (q-10)).
                    let quant = br.read_u32(5)? as i32;
                    let g = (0.125f32 * (quant - 10) as f32).exp();
                    let eps = 1e-6f32;
                    let g = g / filter_ratio.max(eps);

                    // Fold the NB innovation into the high band by
                    // alternating-sign scaling. `low_innov[offset+i]`
                    // is 8 kHz sampled; wideband is 16 kHz so the
                    // effective operation inverts the spectrum (0–4 kHz
                    // → 4–8 kHz is the mirror image under QMF).
                    let mut i = 0;
                    while i + 1 < WB_SUBFRAME_SIZE {
                        let s0 = low_innov[offset + i];
                        let s1 = low_innov[offset + i + 1];
                        exc[i] = FOLDING_GAIN * s0 * g;
                        exc[i + 1] = -FOLDING_GAIN * s1 * g;
                        i += 2;
                    }
                }
                WbInnov::SplitCb(p) => {
                    // 4-bit gain index + split-VQ shape/sign.
                    let qgc = br.read_u32(4)? as usize;
                    let el = low_exc_rms[sub.min(NB_NB_SUBFRAMES - 1)];
                    let gc = 0.87360 * GC_QUANT_BOUND[qgc];
                    let eps = 1e-6f32;
                    let scale = gc * el / filter_ratio.max(eps);

                    split_cb_shape_sign_unquant(br, p, &mut exc)?;
                    for v in &mut exc {
                        *v *= scale;
                    }

                    if sm.double_codebook {
                        let mut innov2 = [0.0f32; WB_SUBFRAME_SIZE];
                        split_cb_shape_sign_unquant(br, p, &mut innov2)?;
                        let weight = 0.4 * scale;
                        for i in 0..WB_SUBFRAME_SIZE {
                            exc[i] += innov2[i] * weight;
                        }
                    }
                }
            }

            // Sanitise.
            for v in &mut exc {
                if !v.is_finite() {
                    *v = 0.0;
                }
                *v = v.clamp(-32000.0, 32000.0);
            }

            // ---- Synthesis filter (1-subframe delay). --------------
            // Reference pipelines: this call filters the *previous*
            // sub-frame's excitation through the *previous* LPC — so
            // the first sub-frame of a freshly-initialised decoder
            // emits near-zero output (the filter memory is zero and
            // the previous excitation is zero).
            let sp_slice =
                &mut out[WB_FRAME_SIZE + offset..WB_FRAME_SIZE + offset + WB_SUBFRAME_SIZE];
            iir_mem16(
                &self.prev_exc,
                &self.interp_qlpc,
                sp_slice,
                WB_SUBFRAME_SIZE,
                WB_LPC_ORDER,
                &mut self.mem_sp,
            );

            // Store current excitation / LPC for the next sub-frame.
            self.prev_exc.copy_from_slice(&exc);
            self.interp_qlpc.copy_from_slice(&ak);
        }

        // ---- QMF synthesis: 2× upsample + combine -------------------
        let mut full = vec![0.0f32; WB_FULL_FRAME_SIZE];
        let (low_in, high_in) = out.split_at_mut(WB_FRAME_SIZE);
        qmf_synth(
            low_in,
            high_in,
            &H0_PROTOTYPE,
            &mut full,
            WB_FULL_FRAME_SIZE,
            QMF_ORDER,
            &mut self.g0_mem,
            &mut self.g1_mem,
        );
        out.copy_from_slice(&full);

        // Save state.
        self.old_qlsp.copy_from_slice(&qlsp);
        self.first = false;
        Ok(())
    }
}

/// Split-VQ shape+sign unquant for the wideband high-band. Identical
/// to the NB version but accepts our local `[f32; WB_SUBFRAME_SIZE]`
/// buffer and the `have_sign = true` path that NB never exercises.
fn split_cb_shape_sign_unquant(
    br: &mut BitReader,
    p: &SplitCbParams,
    exc: &mut [f32; WB_SUBFRAME_SIZE],
) -> Result<()> {
    if p.subvect_size * p.nb_subvect > WB_SUBFRAME_SIZE {
        return Err(Error::invalid("Speex WB: split-CB layout exceeds subframe"));
    }
    for i in 0..p.nb_subvect {
        let sign = if p.have_sign {
            br.read_u32(1)? != 0
        } else {
            false
        };
        let ind = br.read_u32(p.shape_bits)? as usize;
        let s: f32 = if sign { -1.0 } else { 1.0 };
        let base = ind * p.subvect_size;
        if base + p.subvect_size > p.shape_cb.len() {
            return Err(Error::invalid("Speex WB: split-CB index out of range"));
        }
        for j in 0..p.subvect_size {
            exc[p.subvect_size * i + j] += s * 0.03125 * p.shape_cb[base + j] as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_initialises_state_defaults() {
        let d = WbDecoder::new();
        assert!(d.first);
        // old_qlsp spread from ~π/9 to ~8π/9.
        assert!(d.old_qlsp[0] > 0.0);
        assert!(d.old_qlsp[WB_LPC_ORDER - 1] < std::f32::consts::PI);
    }
}
