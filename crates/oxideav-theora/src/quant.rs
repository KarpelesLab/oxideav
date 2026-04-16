//! Quantization matrix computation (spec §6.4.3).
//!
//! Builds a 64-entry quantization matrix in natural order, given the setup
//! header's base matrices, quant ranges and scale tables plus (qti, pli, qi).

use crate::headers::Setup;

/// Minimum quantization value per coefficient, per table 6.18.
/// Returns `QMIN` for coefficient index `ci` and quant-type-index `qti`.
fn qmin(qti: usize, ci: usize) -> i32 {
    match (qti, ci) {
        (0, 0) => 16, // intra DC
        (0, _) => 8,  // intra AC
        (1, 0) => 32, // inter DC
        (1, _) => 16, // inter AC
        _ => 8,
    }
}

/// Compute one quantization matrix (natural order) per spec §6.4.3.
///
/// * `setup` — parsed setup header.
/// * `qti` — 0 for intra, 1 for inter.
/// * `pli` — 0 (Y), 1 (Cb), 2 (Cr).
/// * `qi` — quantization index, 0..63.
pub fn build_qmat(setup: &Setup, qti: usize, pli: usize, qi: u8) -> [i32; 64] {
    // Index into QuantRanges arrays: 3*qti + pli.
    let qpi = qti * 3 + pli;
    let widths = &setup.ranges.widths[qpi];
    let bmis = &setup.ranges.bmis[qpi];

    // Locate the quant range such that qistart <= qi <= qiend.
    let mut qistart: i32 = 0;
    let mut qiend: i32 = 0;
    let mut qri: usize = 0;
    let qi_i = qi as i32;
    if widths.is_empty() {
        // Degenerate: a single base matrix covers all of 0..=63. Treat as a
        // range of width 63 mapping bmi[0] → bmi[0].
        qistart = 0;
        qiend = 63;
        qri = 0;
    } else {
        let mut acc: i32 = 0;
        for (i, &w) in widths.iter().enumerate() {
            let next = acc + w as i32;
            if qi_i <= next {
                qistart = acc;
                qiend = next;
                qri = i;
                break;
            }
            acc = next;
            if i == widths.len() - 1 {
                qistart = acc - w as i32;
                qiend = acc;
                qri = i;
            }
        }
    }
    let bmi_idx = bmis.first().copied().unwrap_or(0) as usize;
    let bmj_idx = if bmis.len() > qri + 1 {
        bmis[qri + 1] as usize
    } else {
        bmi_idx
    };
    let bmi_idx = if !bmis.is_empty() {
        bmis[qri] as usize
    } else {
        0
    };

    let bmi = &setup.bms[bmi_idx];
    let bmj = &setup.bms[bmj_idx];

    let width = if widths.is_empty() {
        1i32
    } else {
        widths[qri] as i32
    };

    let mut qmat = [0i32; 64];
    for ci in 0..64 {
        // BM[ci] interpolation between bmi and bmj weighted by qi position.
        let a = 2 * (qiend - qi_i) * bmi[ci] as i32;
        let b = 2 * (qi_i - qistart) * bmj[ci] as i32;
        let num = a + b + width;
        let bm_ci = num.div_euclid(2 * width);
        let qscale = if ci == 0 {
            setup.dc_scale[qi as usize] as i32
        } else {
            setup.ac_scale[qi as usize] as i32
        };
        let qm_min = qmin(qti, ci);
        let v = (qscale * bm_ci).div_euclid(100) * 4;
        qmat[ci] = qm_min.max(v.min(4096));
    }
    qmat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_setup() -> Setup {
        use crate::headers::setup::QuantRanges;
        Setup {
            lflims: [5u8; 64],
            ac_scale: [100u16; 64],
            dc_scale: [100u16; 64],
            bms: vec![[10u8; 64], [10u8; 64]],
            ranges: QuantRanges {
                nbms: [1u8; 6],
                widths: [vec![63], vec![63], vec![63], vec![63], vec![63], vec![63]],
                bmis: [
                    vec![0, 1],
                    vec![0, 1],
                    vec![0, 1],
                    vec![0, 1],
                    vec![0, 1],
                    vec![0, 1],
                ],
            },
            huffs: Vec::new(),
        }
    }

    #[test]
    fn basic_qmat_respects_qmin() {
        let setup = fake_setup();
        let qmat = build_qmat(&setup, 0, 0, 30);
        // DC: scale*bm/100*4 = 100*10/100*4 = 40, but QMIN for intra-DC = 16,
        // 40 > 16, so 40.
        assert_eq!(qmat[0], 40);
        // AC: scale*bm/100*4 = 40, QMIN intra-AC = 8, so 40.
        assert_eq!(qmat[1], 40);
    }
}
