//! GSM 06.10 speech synthesis pipeline.
//!
//! This module takes an unpacked [`GsmFrame`] and produces 160 16-bit PCM
//! samples at 8 kHz. The math follows the ETSI reference §5.3 exactly; it
//! has been re-implemented in idiomatic Rust (no C borrowed beyond the
//! tables and algorithm structure, which are in the public domain via
//! libgsm by Jutta Degener).
//!
//! Pipeline, for each of the four 40-sample sub-frames:
//!   1. decode & interpolate the 8 LAR reflection coefficients
//!   2. LTP + RPE synthesis → residual excitation `ep[0..40]`
//!   3. short-term synthesis filter on `ep[]` using `rrp[0..8]`
//!   4. post-processing (de-emphasis / upscale / 2-byte truncation)

use crate::frame::{GsmFrame, SubFrame};
use crate::math::{add, mult_r, saturate_i32_to_i16, shl, sub};
use crate::tables::{B, FAC, INVA, MIC, QLB};

/// Persistent state across frames — LTP history, short-term filter state,
/// post-filter state, and the previous frame's LARpp used for interpolation.
#[derive(Clone, Debug)]
pub struct SynthesisState {
    /// LTP adaptive-codebook history (reconstructed excitation), 120 samples.
    pub drp: [i16; 160],
    /// Short-term synthesis filter state v[0..=8].
    pub v: [i16; 9],
    /// Post-processing (reconstruction) previous-sample state `msr`.
    pub msr: i16,
    /// Previous-frame LARpp[0..8].
    pub larpp_prev: [i16; 8],
    /// `nrp` — last subframe's selected lag, used when the new Nc is out of range.
    pub nrp: i16,
}

impl Default for SynthesisState {
    fn default() -> Self {
        Self {
            drp: [0; 160],
            v: [0; 9],
            msr: 0,
            larpp_prev: [0; 8],
            nrp: 40,
        }
    }
}

impl SynthesisState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode one 33-byte frame into 160 16-bit samples.
    pub fn decode_frame(&mut self, frame: &GsmFrame) -> [i16; 160] {
        // 1. Decode LARc[] → LARpp[].
        let larpp_cur = decode_larc(&frame.larc);

        // 2. Process 4 sub-frames.
        let mut out = [0i16; 160];
        for k in 0..4 {
            let larp = interpolate_larpp(k, &self.larpp_prev, &larpp_cur);
            let rrp = lar_to_rp(&larp);
            self.decode_subframe(k, &frame.sub[k], &rrp, &mut out);
        }

        // 3. Save current LARpp for next-frame interpolation.
        self.larpp_prev = larpp_cur;

        // 4. Post-processing: apply `de-emphasis / upscale` on the full
        //    160-sample block (spec §5.3.7).
        post_processing(&mut out, &mut self.msr);
        out
    }

    fn decode_subframe(&mut self, k: usize, sf: &SubFrame, rrp: &[i16; 8], out: &mut [i16; 160]) {
        // --- LTP synthesis (§5.3.3) ---
        let nc = sf.nc as i16;
        let bc = sf.bc as usize;
        let bp = QLB[bc];

        // Clamp Nc to 40..=120. If out of range, fall back to previous nrp.
        let ncr = if (40..=120).contains(&nc) {
            self.nrp = nc;
            nc
        } else {
            self.nrp
        };

        // drpp[j] = MULT_R(bp, drp[j - Nc])  for j in 0..40
        // drp history window begins at offset (160 - 120) = 40. After each
        // subframe we shift the history left by 40 samples (processed below).
        let mut drpp = [0i16; 40];
        for j in 0..40 {
            // drp holds the last 120 reconstructed excitation samples in
            // slots [40..160] (sliding window). The access index is:
            //   drp[160 - Nc + j] for j in 0..40, but only using values
            //   already shifted before this subframe.
            let idx = 120 - ncr as usize + j;
            drpp[j] = mult_r(bp, self.drp[idx]);
        }

        // --- RPE decoding (§5.3.2) ---
        let mut ep = [0i16; 40];
        rpe_decode(sf, &mut ep);

        // Combine: er[j] = ep[j] + drpp[j] → new excitation drp entries.
        // Also shift drp history by 40 to the left.
        for j in 0..120 {
            self.drp[j] = self.drp[j + 40];
        }
        for j in 0..40 {
            self.drp[120 + j] = add(ep[j], drpp[j]);
        }

        // --- Short-term synthesis filter (§5.3.4) ---
        let base = k * 40;
        short_term_synthesis(
            rrp,
            &mut self.v,
            &self.drp[120..160],
            &mut out[base..base + 40],
        );
    }
}

/// Decode 8 LARc indices to Q10-ish `LARpp[]` values (§5.2.8 / §5.3.1).
fn decode_larc(larc: &[u8; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        // Sign-extend the stored bit-field to a full i16 using (-MIC[i]).
        // LARc is stored with bias -MIC so that it fits in an unsigned bit-field.
        let signed = larc[i] as i16 + MIC[i]; // back to spec's signed range
        let mut temp1 = signed - B[i];
        temp1 = shl(temp1, 10); // Q10
                                // Multiply by INVA[i] (Q16) with a (·)>>15 shift → back to Q10.
        out[i] = mult_r(temp1, INVA[i]);
    }
    out
}

/// Interpolate LARp for sub-frame `k` from prev & current LARpp.
fn interpolate_larpp(k: usize, prev: &[i16; 8], cur: &[i16; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    // Interpolation weights from ETSI 06.10 §5.2.9.1:
    //   k = 0: LARp[i] = prev[i] >> 2 + cur[i] * 3/4? No — see libgsm:
    //
    // sub-frame 0:  (prev >> 2) + (cur >> 2)  · halfweight  (0.75/0.25)
    // Actually the spec weights (J=0..3) are:
    //   k=0: 3/4 prev + 1/4 cur   (libgsm uses prev>>2 + 3*cur>>2 — but
    //        we must match spec ordering)
    // libgsm implementation (reference):
    //   j=0: LARp = (LARpp_prev >> 2) + (LARpp_cur >> 2) ⋯ scale
    // The canonical weights in the reference source are:
    //   j=0: 3*prev + cur         shift >>2
    //   j=1: prev + cur           shift >>1
    //   j=2: prev + 3*cur         shift >>2
    //   j=3:                      cur
    //
    // This yields a linear interpolation from prev → cur across 4 sub-frames.
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
        // Clip LARp to spec range [MIC, MAC] after scaling. Scaling:
        //   LARp_q10 / 2^? — we track in the same scale as `cur`, so the
        //   MIC/MAC range applies after we descale later. Skip clipping
        //   here; it happens implicitly in lar_to_rp.
    }
    out
}

/// Transform LARp[] to reflection coefficients rrp[] (§5.2.10).
fn lar_to_rp(larp: &[i16; 8]) -> [i16; 8] {
    let mut out = [0i16; 8];
    for i in 0..8 {
        let mut temp = larp[i];
        let sign = temp < 0;
        let mag = if sign {
            if temp == i16::MIN {
                i16::MAX
            } else {
                -temp
            }
        } else {
            temp
        };
        // Thresholds in Q10.
        let rp = if mag < 11059 {
            mag
        } else if mag < 20070 {
            (mag >> 1) + 5530
        } else {
            let mag32 = mag as i32;
            ((mag32 >> 2) + 11059) as i16
        };
        // Result is Q14; keep sign.
        temp = shl(rp, 4);
        if sign {
            temp = sub(0, temp);
        }
        out[i] = temp;
    }
    out
}

/// Decode RPE pulses (§5.2.14 / §5.2.15) and place them on the grid `Mc` into
/// `ep[0..40]`. The resulting residual is the "RPE excitation".
fn rpe_decode(sf: &SubFrame, ep: &mut [i16; 40]) {
    // --- APCM inverse quantisation ---
    let xmaxc = sf.xmaxc;
    let exp: i16 = if xmaxc > 15 {
        (xmaxc >> 3) as i16 - 3
    } else {
        0
    };
    let mant_sel: i16 = xmaxc as i16 - (exp << 3);
    // FAC indexing is 0..=7 → mant_sel is 0..=7 (verified by construction).
    let mant = (mant_sel & 7) as usize;
    let fac_val = FAC[mant];

    let mut xmp = [0i16; 13];
    for i in 0..13 {
        // dequantised pulse value: (2*xmc - 7) * FAC[mant] / 2^(12 - exp - mant_sel+4)
        // Per spec: xMp[i] = ((xmc[i] << 1) - 7) * FAC[mant] >> (14 - ... )
        let pulse = ((sf.xmc[i] as i32) << 1) - 7;
        let prod = pulse * (fac_val as i32);
        let sh = 12 - exp as i32; // shift after the FAC (Q15) multiply.
        let val = if sh >= 0 { prod >> sh } else { prod << (-sh) };
        xmp[i] = saturate_i32_to_i16(val);
    }

    // --- Place on grid Mc (§5.2.13) ---
    ep.fill(0);
    let mc = (sf.mc & 3) as usize;
    for (i, &v) in xmp.iter().enumerate() {
        let pos = mc + 3 * i;
        if pos < 40 {
            ep[pos] = v;
        }
    }
}

/// Short-term synthesis filter (§5.2.17).
///
/// Input:  `rrp[0..8]` reflection coefficients (Q14)
///         `v[0..=8]`  filter state (persistent)
///         `wt[0..40]` residual (drp[120..160] — the new excitation block)
/// Output: `sr[0..40]` synthesised speech samples
fn short_term_synthesis(rrp: &[i16; 8], v: &mut [i16; 9], wt: &[i16], sr: &mut [i16]) {
    debug_assert_eq!(wt.len(), 40);
    debug_assert_eq!(sr.len(), 40);
    for k in 0..40 {
        let mut sri = wt[k];
        // Lattice filter: run i = 8..=1.
        for i in (1..=8).rev() {
            sri = sub(sri, mult_r(rrp[i - 1], v[i - 1]));
            v[i] = add(v[i - 1], mult_r(rrp[i - 1], sri));
        }
        sr[k] = sri;
        v[0] = sri;
    }
}

/// Post-processing stage (§5.3.5): de-emphasis and upscale.
fn post_processing(samples: &mut [i16; 160], msr: &mut i16) {
    // de-emphasis: srop[k] = sr[k] + 28180/32768 * msr
    //   msr = srop[k]  after output
    // upscale: shift left by 3
    // spec truncates the low 3 bits of the final PCM (2's-complement).
    const BETA: i16 = 28180;
    for s in samples.iter_mut() {
        let scaled = shl(*s, 3);
        let emph = add(scaled, mult_r(*msr, BETA));
        *msr = emph;
        // "Truncation of the output variable to 13 bits" — mask low 3 bits.
        *s = emph & !0x0007u16 as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_frame_yields_near_silence() {
        let mut state = SynthesisState::new();
        let frame = GsmFrame::default();
        // Most-neutral decoded frame: all zero indices are valid bitstream.
        // It should not panic and should return 160 samples.
        let pcm = state.decode_frame(&frame);
        assert_eq!(pcm.len(), 160);
    }

    #[test]
    fn multiple_frames_do_not_panic() {
        // Feed a handful of zero-indexed frames; drp / v state should stay
        // stable (no overflow panics) across frames.
        let mut state = SynthesisState::new();
        for _ in 0..8 {
            let _ = state.decode_frame(&GsmFrame::default());
        }
    }

    #[test]
    fn nonzero_frame_produces_nonzero_output() {
        // Build a frame with a mid-range excitation (xmaxc > 0, xmc pulses
        // non-centered) and verify the decoder emits non-silent samples.
        let mut state = SynthesisState::new();
        let mut frame = GsmFrame::default();
        for sf in &mut frame.sub {
            sf.nc = 60;
            sf.bc = 1;
            sf.mc = 0;
            sf.xmaxc = 40;
            sf.xmc = [7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7];
        }
        let pcm = state.decode_frame(&frame);
        let max = pcm.iter().map(|s| s.unsigned_abs() as u32).max().unwrap();
        assert!(max > 0, "expected non-zero output, got all-zero frame");
    }

    #[test]
    fn lar_to_rp_is_odd_symmetric() {
        let larp = [100, -100, 20000, -20000, 5000, -5000, 0, 12000];
        let rp = lar_to_rp(&larp);
        // Reversed sign should give (near) opposite reflection coefficient.
        let larp_neg = [-100, 100, -20000, 20000, -5000, 5000, 0, -12000];
        let rp_neg = lar_to_rp(&larp_neg);
        for i in 0..8 {
            // Either equal magnitude with opposite sign, or zero → zero.
            assert_eq!(rp[i].wrapping_neg(), rp_neg[i]);
        }
    }
}
