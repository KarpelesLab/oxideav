//! VP9 intra prediction primitives — §8.5.1.
//!
//! VP9 defines 10 intra luma modes (`DC_PRED`, `V_PRED`, `H_PRED`, six
//! directional `D*_PRED` variants, and `TM_PRED`). This module implements
//! the simplest three — `DC_PRED`, `V_PRED`, `H_PRED` — which are enough to
//! reconstruct "flat" or "edge-gradient" blocks in a minimal I-frame. The
//! remaining seven modes return `Error::Unsupported` with a precise §ref so
//! higher layers can report exactly where the decoder gave up.
//!
//! All predictors operate on `u8` samples (8-bit decode only). Higher
//! bit-depths (VP9 profiles 2 & 3) are out of scope for this first
//! pixel-output milestone.
//!
//! Parallel to the AV1 `intra` module — both crates expose the same
//! `predict(mode, neighbours, w, h, dst, stride)` surface so downstream
//! pipeline code can be shared as the decoders mature.

use oxideav_core::{Error, Result};

/// VP9 intra prediction modes — `intra_mode` in the spec (§7.4.5 Table
/// 7-5 / §8.5.1). Values match the spec's numbering so they can be
/// decoded from tree-coded tables without translation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraMode {
    /// `DC_PRED` — mean of the available neighbour rows.
    Dc = 0,
    /// `V_PRED` — copy the above row down (vertical).
    V = 1,
    /// `H_PRED` — copy the left column across (horizontal).
    H = 2,
    /// `D45_PRED` — 45-degree directional.
    D45 = 3,
    /// `D135_PRED` — 135-degree directional.
    D135 = 4,
    /// `D117_PRED` — 117-degree directional.
    D117 = 5,
    /// `D153_PRED` — 153-degree directional.
    D153 = 6,
    /// `D207_PRED` — 207-degree directional.
    D207 = 7,
    /// `D63_PRED` — 63-degree directional.
    D63 = 8,
    /// `TM_PRED` — "true motion" (a.k.a. Paeth in WebP).
    Tm = 9,
}

impl IntraMode {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::Dc,
            1 => Self::V,
            2 => Self::H,
            3 => Self::D45,
            4 => Self::D135,
            5 => Self::D117,
            6 => Self::D153,
            7 => Self::D207,
            8 => Self::D63,
            9 => Self::Tm,
            _ => return Err(Error::invalid(format!("vp9 intra: invalid mode {v}"))),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Dc => "DC_PRED",
            Self::V => "V_PRED",
            Self::H => "H_PRED",
            Self::D45 => "D45_PRED",
            Self::D135 => "D135_PRED",
            Self::D117 => "D117_PRED",
            Self::D153 => "D153_PRED",
            Self::D207 => "D207_PRED",
            Self::D63 => "D63_PRED",
            Self::Tm => "TM_PRED",
        }
    }
}

/// Neighbour pixel availability for the block being predicted. Mirrors the
/// `AboveRow` / `LeftCol` arrays the VP9 spec builds in §8.5.1.
#[derive(Clone, Copy, Debug)]
pub struct Neighbours<'a> {
    /// `above[0..w]` — row of samples immediately above the block.
    pub above: Option<&'a [u8]>,
    /// `left[0..h]` — column of samples immediately left of the block,
    /// ordered top-to-bottom.
    pub left: Option<&'a [u8]>,
}

/// Run `mode` over a `w × h` block. The predictor writes row-major into
/// `dst` with stride `dst_stride`. Returns `Ok(())` on success.
pub fn predict(
    mode: IntraMode,
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    debug_assert!(w > 0 && h > 0);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);
    match mode {
        IntraMode::Dc => dc_pred(n, w, h, dst, dst_stride),
        IntraMode::V => v_pred(n, w, h, dst, dst_stride),
        IntraMode::H => h_pred(n, w, h, dst, dst_stride),
        _ => Err(Error::unsupported(format!(
            "vp9 intra {}: §8.5.1 {} not implemented in parse-only crate",
            mode.name(),
            mode_section_id(mode),
        ))),
    }
}

/// Section identifier within §8.5.1 for error reporting — maps each mode to
/// its sub-clause description.
fn mode_section_id(mode: IntraMode) -> &'static str {
    match mode {
        IntraMode::Dc | IntraMode::V | IntraMode::H => "(non-directional)",
        IntraMode::D45
        | IntraMode::D135
        | IntraMode::D117
        | IntraMode::D153
        | IntraMode::D207
        | IntraMode::D63 => "(directional)",
        IntraMode::Tm => "(TM / true-motion)",
    }
}

/// `DC_PRED` — mean of the available neighbour rows. Pads to the block
/// size. If neither row is available the mid-grey value `1 << (bitdepth-1)`
/// is used (bitdepth=8 → 128), matching VP9 §8.5.1.
fn dc_pred(
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    let dc = match (n.above, n.left) {
        (Some(a), Some(l)) => {
            let sum_a: u32 = a.iter().take(w).map(|&v| v as u32).sum();
            let sum_l: u32 = l.iter().take(h).map(|&v| v as u32).sum();
            let total = sum_a + sum_l;
            let denom = (w + h) as u32;
            ((total + denom / 2) / denom) as u8
        }
        (Some(a), None) => {
            let s: u32 = a.iter().take(w).map(|&v| v as u32).sum();
            ((s + (w as u32) / 2) / (w as u32)) as u8
        }
        (None, Some(l)) => {
            let s: u32 = l.iter().take(h).map(|&v| v as u32).sum();
            ((s + (h as u32) / 2) / (h as u32)) as u8
        }
        (None, None) => 128,
    };
    fill(dst, dst_stride, w, h, dc);
    Ok(())
}

/// `V_PRED` — copy the above row down.
fn v_pred(
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    let above = n.above.ok_or_else(|| {
        Error::invalid("vp9 V_PRED: above-row unavailable (§8.5.1)")
    })?;
    if above.len() < w {
        return Err(Error::invalid(
            "vp9 V_PRED: above-row shorter than block width",
        ));
    }
    for row in 0..h {
        let base = row * dst_stride;
        dst[base..base + w].copy_from_slice(&above[..w]);
    }
    Ok(())
}

/// `H_PRED` — copy each left-column sample across its row.
fn h_pred(
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    let left = n.left.ok_or_else(|| {
        Error::invalid("vp9 H_PRED: left-column unavailable (§8.5.1)")
    })?;
    if left.len() < h {
        return Err(Error::invalid(
            "vp9 H_PRED: left-column shorter than block height",
        ));
    }
    for row in 0..h {
        let v = left[row];
        let base = row * dst_stride;
        for c in 0..w {
            dst[base + c] = v;
        }
    }
    Ok(())
}

fn fill(dst: &mut [u8], dst_stride: usize, w: usize, h: usize, v: u8) {
    for row in 0..h {
        let base = row * dst_stride;
        for c in 0..w {
            dst[base + c] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_pred_average_of_neighbours() {
        let above = [100u8; 4];
        let left = [120u8; 4];
        let n = Neighbours {
            above: Some(&above),
            left: Some(&left),
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::Dc, n, 4, 4, &mut dst, 4).unwrap();
        // (4 * 100 + 4 * 120) / 8 = 110
        for &v in &dst {
            assert_eq!(v, 110);
        }
    }

    #[test]
    fn dc_pred_midgrey_without_neighbours() {
        let n = Neighbours {
            above: None,
            left: None,
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::Dc, n, 4, 4, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn v_pred_copies_above_row() {
        let above = [10u8, 20, 30, 40];
        let n = Neighbours {
            above: Some(&above),
            left: None,
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::V, n, 4, 4, &mut dst, 4).unwrap();
        for row in 0..4 {
            assert_eq!(&dst[row * 4..row * 4 + 4], &above[..]);
        }
    }

    #[test]
    fn h_pred_copies_left_column() {
        let left = [10u8, 20, 30, 40];
        let n = Neighbours {
            above: None,
            left: Some(&left),
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::H, n, 4, 4, &mut dst, 4).unwrap();
        for row in 0..4 {
            assert_eq!(dst[row * 4], left[row]);
            assert_eq!(dst[row * 4 + 3], left[row]);
        }
    }

    #[test]
    fn unsupported_modes_return_clear_error() {
        let n = Neighbours {
            above: None,
            left: None,
        };
        let mut dst = [0u8; 16];
        let err = predict(IntraMode::Tm, n, 4, 4, &mut dst, 4).unwrap_err();
        match err {
            Error::Unsupported(s) => {
                assert!(s.contains("TM_PRED"), "msg: {s}");
                assert!(s.contains("§8.5.1"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
