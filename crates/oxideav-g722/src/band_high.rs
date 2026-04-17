//! High-band ADPCM — 2-bit quantiser with backward-adaptive predictor.
//!
//! Same caveat as [`crate::band_low`]: this implementation is
//! self-consistent (decoder is the exact inverse of encoder given identical
//! inputs) rather than bit-exact with ITU-T G.722 tables. The pipeline
//! shape (quantise → update predictor + scale → form next prediction) is
//! unchanged; only the tables differ.

#[derive(Clone, Debug)]
pub struct HighBand {
    s: i32,
    det: i32,
    r1: i32,
    r2: i32,
    d_hist: [i32; 2],
    a1: i32,
    a2: i32,
    b0: i32,
}

impl Default for HighBand {
    fn default() -> Self {
        Self::new()
    }
}

impl HighBand {
    pub fn new() -> Self {
        Self {
            s: 0,
            det: 64,
            r1: 0,
            r2: 0,
            d_hist: [0; 2],
            a1: 0,
            a2: 0,
            b0: 0,
        }
    }

    /// Encode a 15-bit-range high-band sample to a 2-bit code (0..=3).
    pub fn encode(&mut self, xh: i32) -> u8 {
        let eh = xh - self.s;
        let code = quantise(eh, self.det);
        self.update(code);
        code
    }

    /// Decode a 2-bit code to a reconstructed high-band sample.
    pub fn decode(&mut self, code: u8) -> i32 {
        let code = code & 0x3;
        let dq = dequantise(code, self.det);
        let r = sat15(self.s + dq);
        self.update_with(code, dq, r);
        r
    }

    fn update(&mut self, code: u8) {
        let code = code & 0x3;
        let dq = dequantise(code, self.det);
        let r = sat15(self.s + dq);
        self.update_with(code, dq, r);
    }

    fn update_with(&mut self, code: u8, dq: i32, r: i32) {
        // Log-domain scale-factor adapter.
        const WH: [i32; 4] = [-214, -214, 798, 798];
        let w = WH[(code & 0x3) as usize];
        self.det = adapt_det(self.det, w);

        // Single-zero predictor coefficient update.
        let decayed = self.b0 - (self.b0 >> 8);
        let delta = if dq == 0 || self.d_hist[0] == 0 {
            0
        } else if (dq > 0) == (self.d_hist[0] > 0) {
            128
        } else {
            -128
        };
        self.b0 = (decayed + delta).clamp(-32_768, 32_767);

        // 2-pole predictor coefficients.
        let wd_pol2 = if (r > 0) == (self.r2 > 0) && (r != 0 && self.r2 != 0) {
            64
        } else {
            -64
        };
        let a2_new = ((self.a2 as i64 * 32_640) / 32_768) as i32 + wd_pol2;
        let a2_new = a2_new.clamp(-12_288, 12_288);
        let wd_pol1 = if (r > 0) == (self.r1 > 0) && (r != 0 && self.r1 != 0) {
            96
        } else {
            -96
        };
        let a1_new = ((self.a1 as i64 * 32_640) / 32_768) as i32 + wd_pol1;
        let a1_limit = 15_360 - a2_new.max(0);
        let a1_new = a1_new.clamp(-a1_limit.max(1), a1_limit.max(1));
        self.a1 = a1_new;
        self.a2 = a2_new;

        // Shift histories.
        self.d_hist[1] = self.d_hist[0];
        self.d_hist[0] = dq;
        self.r2 = self.r1;
        self.r1 = r;

        // Next predictor estimate.
        let sp = ((self.a1 * self.r1) >> 15) + ((self.a2 * self.r2) >> 15);
        let sz = (self.b0 * self.d_hist[0]) >> 15;
        self.s = sat15(sp + sz);
    }
}

/// 2-bit quantiser. Code 0 is silence (magnitude 0, positive-sign), 1 is
/// small negative, 2 is small positive, 3 is large. Zero input → code 0.
fn quantise(eh: i32, det: i32) -> u8 {
    if eh == 0 {
        return 0;
    }
    let sign = eh < 0;
    let mag = eh.unsigned_abs() as i32;
    let step = det.max(1);
    // Two magnitude tiers: small (mag < 2*step) → tier 0, else tier 1.
    let tier = if mag < 2 * step { 0 } else { 1 };
    // Bit 0 = sign, bit 1 = tier.
    (sign as u8) | (tier << 1)
}

fn dequantise(code: u8, det: i32) -> i32 {
    if code == 0 {
        return 0;
    }
    let sign = code & 1;
    let tier = (code >> 1) & 1;
    let mag = if tier == 0 { det } else { det * 3 };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

fn adapt_det(det: i32, w: i32) -> i32 {
    let scaled = (det as i64 * (4096 + w as i64)) / 4096;
    (scaled as i32).clamp(8, 32_767)
}

fn sat15(v: i32) -> i32 {
    v.clamp(-16_384, 16_383)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_stays_silent() {
        let mut enc = HighBand::new();
        let mut dec = HighBand::new();
        for _ in 0..400 {
            let code = enc.encode(0);
            assert_eq!(code, 0);
            let r = dec.decode(code);
            assert_eq!(r, 0);
        }
    }

    #[test]
    fn encoder_decoder_in_sync() {
        let mut enc = HighBand::new();
        let mut dec = HighBand::new();
        for n in 0..400 {
            let x = ((n as f32 * 0.25).cos() * 3_000.0) as i32;
            let code = enc.encode(x);
            let _ = dec.decode(code);
            assert_eq!(enc.s, dec.s);
            assert_eq!(enc.det, dec.det);
        }
    }
}
