//! Inverse quantisation for MPEG-4 Part 2 (§7.4.4).
//!
//! Two modes, selected by the VOL's `mpeg_quant` flag:
//! * **H.263 quantisation** (the default XVID/DivX mode) — very simple: each
//!   AC coefficient `l != 0` dequantises to
//!   `(2 * Q * |l| + Q) * sign(l)` if `Q` is odd,
//!   `(2 * Q * |l| + Q - 1) * sign(l)` if `Q` is even.
//! * **MPEG-4 quantisation** — uses an 8x8 quant matrix similar to MPEG-1/2,
//!   with mismatch control.
//!
//! Only the H.263 path is filled in for this session scaffold; the MPEG-4
//! matrix path stubs out clearly.

use oxideav_core::{Error, Result};

use crate::headers::vol::VideoObjectLayer;

/// Dequantise one intra block's AC coefficients in-place (index 0 is the DC
/// coefficient and is left untouched — DC is handled separately by the
/// caller, with prediction). `coeffs[i]` is the raw decoded level; on return
/// `coeffs[i]` holds the reconstructed coefficient.
///
/// `quant` is the current `vop_quant` (1..=31 for quant_precision=5).
pub fn dequantise_intra_h263(coeffs: &mut [i32; 64], quant: u32) -> Result<()> {
    if quant == 0 {
        return Err(Error::invalid("mpeg4 iq: quant = 0"));
    }
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    for i in 1..64 {
        let l = coeffs[i];
        if l == 0 {
            continue;
        }
        let abs = l.abs();
        let mut val = 2 * q * abs + q_plus;
        if l < 0 {
            val = -val;
        }
        coeffs[i] = val.clamp(-2048, 2047);
    }
    Ok(())
}

/// MPEG-4 (matrix) quantisation — not implemented in this session. Returns
/// `Unsupported` with a clear follow-up message.
pub fn dequantise_intra_mpeg4(
    _coeffs: &mut [i32; 64],
    _quant: u32,
    _vol: &VideoObjectLayer,
) -> Result<()> {
    Err(Error::unsupported(
        "mpeg4 iq: MPEG-4 matrix quant path: follow-up",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h263_intra_even_quant() {
        // Q=4 (even). Level 3 -> 2*4*3 + (4-1) = 24 + 3 = 27.
        let mut c = [0i32; 64];
        c[1] = 3;
        dequantise_intra_h263(&mut c, 4).unwrap();
        assert_eq!(c[1], 27);
    }

    #[test]
    fn h263_intra_odd_quant() {
        // Q=5. Level -2 -> -(2*5*2 + 5) = -25.
        let mut c = [0i32; 64];
        c[2] = -2;
        dequantise_intra_h263(&mut c, 5).unwrap();
        assert_eq!(c[2], -25);
    }
}
