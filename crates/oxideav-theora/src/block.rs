//! Intra-frame block decode.
//!
//! Implements:
//!   * DCT coefficient token decode (spec §7.7)
//!   * DC prediction reversal (spec §7.8)
//!   * Dequantisation (spec §7.9.2) and integer inverse DCT (spec §7.9.3)
//!   * Pixel reconstruction (spec §7.9.4) for INTRA blocks
//!   * Loop filter (spec §7.10)
//!
//! Inter-frame support is out of scope: the decoder rejects inter packets
//! with Error::Unsupported.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::coded_order::FrameLayout;
use crate::dct::{idct2d, INV_ZIGZAG};
use crate::headers::{Headers, PixelFormat, Setup};
use crate::quant::build_qmat;

/// Per-frame decode context.
pub struct IntraFrameDecoder<'a> {
    pub headers: &'a Headers,
    pub layout: FrameLayout,
    /// QIS[0..NQIS], frame-level qi values.
    pub qis: [u8; 3],
    pub nqis: usize,
    /// Per-block qi-index. For intra it's always 0 (single qi).
    /// Index 0..NBS.
    pub qiis: Vec<u8>,
    /// Per-block BCODED flag. For intra, all 1.
    pub bcoded: Vec<bool>,
    /// Per-block coefficient count (tracks 64 at EOB, ≤n for partial zeros).
    pub ncoeffs: Vec<u8>,
    /// Quantised coefficients, NBS*64 laid out in zig-zag order (coeffs[bi*64 + ti]).
    pub coeffs: Vec<i32>,
    /// Luma Huffman-table index for DC+AC groups (chosen per ti).
    pub hti_l: [u8; 5],
    pub hti_c: [u8; 5],
}

impl<'a> IntraFrameDecoder<'a> {
    pub fn new(headers: &'a Headers) -> Self {
        let id = &headers.identification;
        let layout = FrameLayout::new(id.fmbw as u32, id.fmbh as u32, id.pf);
        let nbs = layout.nbs as usize;
        Self {
            headers,
            layout,
            qis: [0u8; 3],
            nqis: 0,
            qiis: vec![0u8; nbs],
            bcoded: vec![true; nbs],
            ncoeffs: vec![0u8; nbs],
            coeffs: vec![0i32; nbs * 64],
            hti_l: [0u8; 5],
            hti_c: [0u8; 5],
        }
    }

    /// §7.1 frame header decode. Caller must have already consumed the leading
    /// "data-packet" bit and frame-type bit before invoking this.
    ///
    /// This function continues from the frame-header position: reads QIS[0]
    /// (6 bits), MOREQIS, optional QIS[1..2], and for intra the 3 reserved
    /// bits.
    pub fn read_intra_frame_header(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        self.qis[0] = br.read_u32(6)? as u8;
        let mut nqis = 1usize;
        if br.read_bit()? {
            self.qis[1] = br.read_u32(6)? as u8;
            nqis = 2;
            if br.read_bit()? {
                self.qis[2] = br.read_u32(6)? as u8;
                nqis = 3;
            }
        }
        self.nqis = nqis;
        // Intra: 3 reserved bits that MUST be zero.
        let reserved = br.read_u32(3)?;
        if reserved != 0 {
            return Err(Error::invalid(
                "Theora intra frame: non-zero reserved header bits",
            ));
        }
        Ok(())
    }

    /// §7.3 coded-block flags. For intra, all blocks are coded.
    pub fn fill_bcoded_intra(&mut self) {
        // Already initialised to `true`; nothing to do.
        self.bcoded.fill(true);
    }

    /// §7.6 Block-level qi. For intra with NQIS==1 this is a no-op (QIIS stays
    /// zero); higher NQIS can still occur for intra frames per spec — follow
    /// the procedure.
    pub fn read_qiis(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        // `qiis` is already zero-initialised.
        for qii in 0..(self.nqis.saturating_sub(1)) {
            // NBITS = number of coded blocks with QIIS[bi] == qii.
            let nbits = self
                .bcoded
                .iter()
                .zip(self.qiis.iter())
                .filter(|(&c, &q)| c && q as usize == qii)
                .count();
            if nbits == 0 {
                continue;
            }
            let bits = read_long_run_bitstring(br, nbits)?;
            let mut it = bits.into_iter();
            for bi in 0..self.bcoded.len() {
                if self.bcoded[bi] && self.qiis[bi] as usize == qii {
                    let b = it
                        .next()
                        .ok_or_else(|| Error::invalid("Theora: truncated QIIS bit string"))?;
                    self.qiis[bi] += b as u8;
                }
            }
        }
        Ok(())
    }

    /// §7.7.3 DCT coefficient decode. Fills `coeffs` and `ncoeffs`.
    pub fn decode_coefficients(&mut self, br: &mut BitReader<'_>) -> Result<()> {
        let nbs = self.layout.nbs as usize;
        let nlbs = (self.layout.planes[0].nbw * self.layout.planes[0].nbh) as usize;
        let mut tis = vec![0u8; nbs]; // current token index per block
        let mut eobs: u32 = 0;
        for ti in 0..64u8 {
            if ti == 0 {
                // DC: read 4-bit HTI for luma and chroma.
                self.hti_l[0] = br.read_u32(4)? as u8;
                self.hti_c[0] = br.read_u32(4)? as u8;
            } else if ti == 1 {
                // AC: read HTI for luma and chroma (groups 1..4 share the
                // same htiL/htiC).
                let l = br.read_u32(4)? as u8;
                let c = br.read_u32(4)? as u8;
                for g in 1..5 {
                    self.hti_l[g] = l;
                    self.hti_c[g] = c;
                }
            }
            for bi in 0..nbs {
                if !self.bcoded[bi] || tis[bi] != ti {
                    continue;
                }
                self.ncoeffs[bi] = ti;
                if eobs > 0 {
                    // Consume one step of the EOB run.
                    for tj in (ti as usize)..64 {
                        self.coeffs[bi * 64 + tj] = 0;
                    }
                    tis[bi] = 64;
                    eobs -= 1;
                } else {
                    // Pick the Huffman table group based on ti (Table 7.42).
                    let hg = huffman_group_for_ti(ti) as usize;
                    let hti = if bi < nlbs {
                        16 * hg + self.hti_l[hg] as usize
                    } else {
                        16 * hg + self.hti_c[hg] as usize
                    };
                    let tree = self
                        .headers
                        .setup
                        .huffs
                        .get(hti)
                        .ok_or_else(|| Error::invalid("Theora: bad Huffman table index"))?;
                    let token = tree.decode(br)?;
                    if token < 7 {
                        // EOB token.
                        eobs = decode_eob_token(br, token, self.n_remaining(&tis))?;
                        // Apply to this block.
                        for tj in (ti as usize)..64 {
                            self.coeffs[bi * 64 + tj] = 0;
                        }
                        tis[bi] = 64;
                        eobs = eobs.saturating_sub(1);
                    } else {
                        decode_coef_token(
                            br,
                            token,
                            bi,
                            ti,
                            &mut self.coeffs,
                            &mut tis,
                            &mut self.ncoeffs,
                        )?;
                    }
                }
            }
        }
        if eobs != 0 {
            return Err(Error::invalid(
                "Theora: non-zero EOB run left at end of coefficient decode",
            ));
        }
        Ok(())
    }

    /// Count remaining coded blocks whose token index is < 64 (used for the
    /// special "zero-length" EOB-6 token).
    fn n_remaining(&self, tis: &[u8]) -> u32 {
        self.bcoded
            .iter()
            .zip(tis.iter())
            .filter(|(&c, &t)| c && t < 64)
            .count() as u32
    }

    /// §7.8.2 Inverting DC prediction. Runs in raster order per plane.
    pub fn undo_dc_prediction(&mut self) {
        for pli in 0..3 {
            let mut last_dc = [0i32; 3];
            let plane = &self.layout.planes[pli];
            let nbw = plane.nbw;
            let nbh = plane.nbh;
            // Iterate in raster order (row 0 = bottom, but order doesn't matter
            // for left/below-left/below/below-right neighbours as long as it's
            // bottom-up row-by-row — which matches "raster order" in this
            // bottom-left coordinate system).
            for by in 0..nbh {
                for bx in 0..nbw {
                    let bi = self.layout.global_coded(pli, bx, by) as usize;
                    if !self.bcoded[bi] {
                        continue;
                    }
                    // Intra: rfi = 0. All neighbour filtering reduces to
                    // "block exists and is coded".
                    let has_left = bx > 0;
                    let has_down = by > 0;
                    let has_down_left = bx > 0 && by > 0;
                    let has_down_right = bx + 1 < nbw && by > 0;
                    let neighbour = |x: u32, y: u32| -> i32 {
                        let j = self.layout.global_coded(pli, x, y) as usize;
                        self.coeffs[j * 64]
                    };
                    let mut p = [false; 4];
                    let mut pv = [0i32; 4];
                    if has_left {
                        p[0] = true;
                        pv[0] = neighbour(bx - 1, by);
                    }
                    if has_down_left {
                        p[1] = true;
                        pv[1] = neighbour(bx - 1, by - 1);
                    }
                    if has_down {
                        p[2] = true;
                        pv[2] = neighbour(bx, by - 1);
                    }
                    if has_down_right {
                        p[3] = true;
                        pv[3] = neighbour(bx + 1, by - 1);
                    }
                    let dcpred = if !p.iter().any(|&b| b) {
                        last_dc[0]
                    } else {
                        let (weights, pdiv) = weights_for(p);
                        let mut sum: i32 = 0;
                        for i in 0..4 {
                            if p[i] {
                                sum += weights[i] * pv[i];
                            }
                        }
                        let mut pred = if sum < 0 {
                            // Division truncation towards zero (like C).
                            -((-sum) / pdiv)
                        } else {
                            sum / pdiv
                        };
                        if p[0] && p[1] && p[2] {
                            if (pred - pv[2]).abs() > 128 {
                                pred = pv[2];
                            } else if (pred - pv[0]).abs() > 128 {
                                pred = pv[0];
                            } else if (pred - pv[1]).abs() > 128 {
                                pred = pv[1];
                            }
                        }
                        pred
                    };
                    let dc = self.coeffs[bi * 64] + dcpred;
                    // Truncate to 16-bit signed.
                    let dc = (dc as i16) as i32;
                    self.coeffs[bi * 64] = dc;
                    last_dc[0] = dc; // intra → rfi = 0
                }
            }
        }
    }

    /// §7.9.4 complete reconstruction (intra-only subset).
    /// Writes to per-plane buffers. Each buffer is `nbw*8 × nbh*8` laid out
    /// top-down (row 0 is the top). Allocates and returns the three planes.
    pub fn reconstruct(&self) -> [Vec<u8>; 3] {
        let id = &self.headers.identification;
        let qi0 = self.qis[0];

        // For intra frames every block is INTRA, so qti=0 always.
        // Precompute quant matrices for DC + AC for each (pli, qi).
        // There are at most NQIS different AC qi values + 1 DC qi value.
        let setup: &Setup = &self.headers.setup;
        let mut ac_qmats: [Vec<Option<[i32; 64]>>; 3] =
            [vec![None; 64], vec![None; 64], vec![None; 64]];
        let mut dc_qmats: [[i32; 64]; 3] = [[0i32; 64]; 3];
        for pli in 0..3 {
            dc_qmats[pli] = build_qmat(setup, 0, pli, qi0);
        }
        let cache_ac = |pli: usize, qi: u8, cache: &mut [Vec<Option<[i32; 64]>>; 3]| {
            if cache[pli][qi as usize].is_none() {
                cache[pli][qi as usize] = Some(build_qmat(setup, 0, pli, qi));
            }
        };

        let mut planes_out: [Vec<u8>; 3] = Default::default();
        for pli in 0..3 {
            let (pw, ph) =
                crate::coded_order::plane_pixel_dims(id.fmbw as u32, id.fmbh as u32, id.pf, pli);
            planes_out[pli] = vec![0u8; (pw * ph) as usize];
        }

        for bi in 0..(self.layout.nbs as usize) {
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            // Block pixel corner: lower-left origin. We'll convert to top-down
            // pixel coords when writing.
            let plane = &self.layout.planes[pli];
            let pw = plane.nbw * 8;
            let ph = plane.nbh * 8;
            if !self.bcoded[bi] {
                continue;
            }

            // Dequant & IDCT.
            let mut res = [0i32; 64];
            if self.ncoeffs[bi] < 2 {
                // DC-only shortcut.
                let qmat = &dc_qmats[pli];
                let c = ((self.coeffs[bi * 64] * qmat[0]) + 15) >> 5;
                let c16 = (c as i16) as i32;
                res.fill(c16);
            } else {
                let qi = self.qis[self.qiis[bi] as usize];
                cache_ac(pli, qi, &mut ac_qmats);
                let ac = ac_qmats[pli][qi as usize].as_ref().unwrap();
                let dc_q = &dc_qmats[pli];
                let mut dqc = [0i32; 64];
                // ci=0: DC uses DC qmat.
                dqc[0] = ((self.coeffs[bi * 64] * dc_q[0]) as i16) as i32;
                for ci in 1..64 {
                    let zzi = INV_ZIGZAG[ci];
                    let c = self.coeffs[bi * 64 + zzi] * ac[ci];
                    dqc[ci] = (c as i16) as i32;
                }
                res = idct2d(&dqc);
            }
            // Add the INTRA predictor (constant 128) and clamp.
            // Write into plane, translating from bottom-left to top-down.
            let bx_px = bx * 8;
            let by_px_bottom = by * 8; // distance from bottom
            let out = &mut planes_out[pli];
            for ry in 0..8u32 {
                // ry=0 is the bottom row of the 8x8 block per spec §7.9.3.2.
                // In top-down output, this maps to pixel row (ph - 1 - (by_px_bottom + ry)).
                let py = ph - 1 - (by_px_bottom + ry);
                for rx in 0..8u32 {
                    let px = bx_px + rx;
                    let p = res[(ry as usize) * 8 + rx as usize] + 128;
                    let clamped = p.clamp(0, 255) as u8;
                    out[(py * pw + px) as usize] = clamped;
                }
            }
        }

        planes_out
    }

    /// §7.10 Complete loop filter. Operates in-place on top-down plane buffers.
    pub fn loop_filter(&self, planes: &mut [Vec<u8>; 3]) {
        let l = self.headers.setup.lflims[self.qis[0] as usize] as i32;
        if l == 0 {
            return;
        }
        let id = &self.headers.identification;
        for bi in 0..(self.layout.nbs as usize) {
            if !self.bcoded[bi] {
                continue;
            }
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            let plane = &self.layout.planes[pli];
            let pw = (plane.nbw * 8) as i32;
            let ph = (plane.nbh * 8) as i32;
            let bx_px = (bx * 8) as i32;
            // by is from the bottom; convert block's lower-left y to top-down.
            // The "lower-left" in the spec's bottom-left coord system is the
            // row with minimum y. In a top-down buffer, that row is
            // `ph - 1 - (by*8)`. But the filter indexes from FY..=FY+3 into
            // RECP — those are the four rows above the edge in bottom-left
            // terms. We flip when we actually access the buffer.
            let by_px_bottom = (by * 8) as i32;
            let buf = &mut planes[pli];
            // Left edge of block bi.
            if bx_px > 0 {
                let fx = bx_px - 2;
                let fy = by_px_bottom;
                horizontal_filter(buf, pw, ph, fx, fy, l);
            }
            // Bottom edge of block bi: only if by > 0.
            if by_px_bottom > 0 {
                let fx = bx_px;
                let fy = by_px_bottom - 2;
                vertical_filter(buf, pw, ph, fx, fy, l);
            }
            // Right edge of bi: only if the right neighbour is *not* coded.
            // For intra all blocks are coded, so the right neighbour (if it
            // exists) is always coded — the spec says "if BCODED[bj] is zero"
            // so we'd skip unless at frame edge. For intra we still filter
            // frame edges? No — the right-edge check is only triggered when
            // (BX+8) < RPW and the right neighbour is uncoded. For intra, we
            // skip entirely. Same for top edge.
            // (In inter this is what picks up the discontinuity across an
            // uncoded neighbour.)
            // Nothing to do here for intra.
            let _ = id;
        }
    }
}

fn horizontal_filter(buf: &mut [u8], pw: i32, ph: i32, fx: i32, fy: i32, l: i32) {
    // `fy` is the bottom pixel of the 8-tall edge column in bottom-left coords.
    // The filter modifies columns FX+1 and FX+2 for 8 consecutive rows (by in
    // 0..8 meaning vertical). Convert to top-down row for each by.
    for by in 0..8 {
        let row_bottom = fy + by;
        if row_bottom < 0 || row_bottom >= ph {
            continue;
        }
        if fx < 0 || fx + 3 >= pw {
            continue;
        }
        let top_y = ph - 1 - row_bottom;
        let row = (top_y * pw) as usize;
        let a = buf[row + fx as usize] as i32;
        let b = buf[row + (fx + 1) as usize] as i32;
        let c = buf[row + (fx + 2) as usize] as i32;
        let d = buf[row + (fx + 3) as usize] as i32;
        let r = (a - 3 * b + 3 * c - d + 4) >> 3;
        let r_lim = lflim(r, l);
        let nb = (b + r_lim).clamp(0, 255) as u8;
        let nc = (c - r_lim).clamp(0, 255) as u8;
        buf[row + (fx + 1) as usize] = nb;
        buf[row + (fx + 2) as usize] = nc;
    }
}

fn vertical_filter(buf: &mut [u8], pw: i32, ph: i32, fx: i32, fy: i32, l: i32) {
    // Filters rows FY+1 and FY+2 (in bottom-left coords) across 8 columns.
    for bx in 0..8 {
        let col = fx + bx;
        if col < 0 || col >= pw {
            continue;
        }
        if fy < 0 || fy + 3 >= ph {
            continue;
        }
        // Map bottom-left rows FY..FY+3 to top-down rows.
        // Spec says RECP[FY+0] is the lower row, RECP[FY+3] the upper row.
        // So the 4-tap goes from below edge up past edge.
        let rowy = |yb: i32| -> usize { ((ph - 1 - yb) * pw + col) as usize };
        let a = buf[rowy(fy)] as i32;
        let b = buf[rowy(fy + 1)] as i32;
        let c = buf[rowy(fy + 2)] as i32;
        let d = buf[rowy(fy + 3)] as i32;
        let r = (a - 3 * b + 3 * c - d + 4) >> 3;
        let r_lim = lflim(r, l);
        let nb = (b + r_lim).clamp(0, 255) as u8;
        let nc = (c - r_lim).clamp(0, 255) as u8;
        buf[rowy(fy + 1)] = nb;
        buf[rowy(fy + 2)] = nc;
    }
}

fn lflim(r: i32, l: i32) -> i32 {
    if r <= -2 * l {
        0
    } else if r <= -l {
        -r - 2 * l
    } else if r < l {
        r
    } else if r < 2 * l {
        -r + 2 * l
    } else {
        0
    }
}

fn huffman_group_for_ti(ti: u8) -> u8 {
    match ti {
        0 => 0,
        1..=5 => 1,
        6..=14 => 2,
        15..=27 => 3,
        _ => 4,
    }
}

fn decode_eob_token(br: &mut BitReader<'_>, token: u8, remaining_blocks: u32) -> Result<u32> {
    Ok(match token {
        0 => 1,
        1 => 2,
        2 => 3,
        3 => br.read_u32(2)? + 4,
        4 => br.read_u32(3)? + 8,
        5 => br.read_u32(4)? + 16,
        6 => {
            let v = br.read_u32(12)?;
            if v == 0 {
                remaining_blocks
            } else {
                v
            }
        }
        _ => return Err(Error::invalid("Theora: invalid EOB token")),
    })
}

fn decode_coef_token(
    br: &mut BitReader<'_>,
    token: u8,
    bi: usize,
    ti: u8,
    coeffs: &mut [i32],
    tis: &mut [u8],
    ncoeffs: &mut [u8],
) -> Result<()> {
    let block = bi * 64;
    let zz_put = |coeffs: &mut [i32], ti: usize, v: i32| {
        if ti < 64 {
            coeffs[block + ti] = v;
        }
    };
    let ti_u = ti as usize;
    match token {
        7 => {
            let rlen = (br.read_u32(3)? + 1) as usize;
            for tj in ti_u..(ti_u + rlen).min(64) {
                coeffs[block + tj] = 0;
            }
            tis[bi] = (ti_u + rlen).min(64) as u8;
        }
        8 => {
            let rlen = (br.read_u32(6)? + 1) as usize;
            for tj in ti_u..(ti_u + rlen).min(64) {
                coeffs[block + tj] = 0;
            }
            tis[bi] = (ti_u + rlen).min(64) as u8;
        }
        9 => {
            zz_put(coeffs, ti_u, 1);
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        10 => {
            zz_put(coeffs, ti_u, -1);
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        11 => {
            zz_put(coeffs, ti_u, 2);
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        12 => {
            zz_put(coeffs, ti_u, -2);
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        13..=16 => {
            let base = (token - 10) as i32; // 3,4,5,6
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u, if sign { -base } else { base });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        17 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(1)? as i32 + 7;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        18 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(2)? as i32 + 9;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        19 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(3)? as i32 + 13;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        20 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(4)? as i32 + 21;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        21 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(5)? as i32 + 37;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        22 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(9)? as i32 + 69;
            zz_put(coeffs, ti_u, if sign { -mag } else { mag });
            tis[bi] = ti + 1;
            ncoeffs[bi] = tis[bi];
        }
        23 => {
            // One zero + ±1.
            zz_put(coeffs, ti_u, 0);
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u + 1, if sign { -1 } else { 1 });
            tis[bi] = ti + 2;
            ncoeffs[bi] = tis[bi];
        }
        24 => {
            for tj in ti_u..(ti_u + 2).min(64) {
                coeffs[block + tj] = 0;
            }
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u + 2, if sign { -1 } else { 1 });
            tis[bi] = ti + 3;
            ncoeffs[bi] = tis[bi];
        }
        25 => {
            for tj in ti_u..(ti_u + 3).min(64) {
                coeffs[block + tj] = 0;
            }
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u + 3, if sign { -1 } else { 1 });
            tis[bi] = ti + 4;
            ncoeffs[bi] = tis[bi];
        }
        26 => {
            for tj in ti_u..(ti_u + 4).min(64) {
                coeffs[block + tj] = 0;
            }
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u + 4, if sign { -1 } else { 1 });
            tis[bi] = ti + 5;
            ncoeffs[bi] = tis[bi];
        }
        27 => {
            for tj in ti_u..(ti_u + 5).min(64) {
                coeffs[block + tj] = 0;
            }
            let sign = br.read_bit()?;
            zz_put(coeffs, ti_u + 5, if sign { -1 } else { 1 });
            tis[bi] = ti + 6;
            ncoeffs[bi] = tis[bi];
        }
        28 => {
            let sign = br.read_bit()?;
            let rlen = (br.read_u32(2)? + 6) as usize;
            for tj in ti_u..(ti_u + rlen).min(64) {
                coeffs[block + tj] = 0;
            }
            zz_put(coeffs, ti_u + rlen, if sign { -1 } else { 1 });
            tis[bi] = (ti_u + rlen + 1).min(64) as u8;
            ncoeffs[bi] = tis[bi];
        }
        29 => {
            let sign = br.read_bit()?;
            let rlen = (br.read_u32(3)? + 10) as usize;
            for tj in ti_u..(ti_u + rlen).min(64) {
                coeffs[block + tj] = 0;
            }
            zz_put(coeffs, ti_u + rlen, if sign { -1 } else { 1 });
            tis[bi] = (ti_u + rlen + 1).min(64) as u8;
            ncoeffs[bi] = tis[bi];
        }
        30 => {
            zz_put(coeffs, ti_u, 0);
            let sign = br.read_bit()?;
            let mag = br.read_u32(1)? as i32 + 2;
            zz_put(coeffs, ti_u + 1, if sign { -mag } else { mag });
            tis[bi] = ti + 2;
            ncoeffs[bi] = tis[bi];
        }
        31 => {
            let sign = br.read_bit()?;
            let mag = br.read_u32(1)? as i32 + 2;
            let rlen = (br.read_u32(1)? + 2) as usize;
            for tj in ti_u..(ti_u + rlen).min(64) {
                coeffs[block + tj] = 0;
            }
            zz_put(coeffs, ti_u + rlen, if sign { -mag } else { mag });
            tis[bi] = (ti_u + rlen + 1).min(64) as u8;
            ncoeffs[bi] = tis[bi];
        }
        _ => {
            return Err(Error::invalid(format!(
                "Theora: invalid coef token {token}"
            )))
        }
    }
    Ok(())
}

/// Long-run bit string decode (spec §7.2.1).
pub fn read_long_run_bitstring(br: &mut BitReader<'_>, nbits: usize) -> Result<Vec<bool>> {
    let mut out = Vec::with_capacity(nbits);
    let mut bit = false;
    let mut is_first = true;
    while out.len() < nbits {
        if is_first {
            bit = br.read_bit()?;
            is_first = false;
        }
        let (rstart, rbits) = long_run_huffman(br)?;
        let roffs = br.read_u32(rbits)?;
        let rlen = rstart + roffs;
        for _ in 0..rlen {
            if out.len() >= nbits {
                break;
            }
            out.push(bit);
        }
        if out.len() == nbits {
            break;
        }
        if rlen == 4129 {
            bit = br.read_bit()?;
        } else {
            bit = !bit;
        }
    }
    Ok(out)
}

fn long_run_huffman(br: &mut BitReader<'_>) -> Result<(u32, u32)> {
    // Table 7.7: 0→(1,0); 10→(2,1); 110→(4,1); 1110→(6,2); 11110→(10,3);
    // 111110→(18,4); 111111→(34,12).
    if !br.read_bit()? {
        return Ok((1, 0));
    }
    if !br.read_bit()? {
        return Ok((2, 1));
    }
    if !br.read_bit()? {
        return Ok((4, 1));
    }
    if !br.read_bit()? {
        return Ok((6, 2));
    }
    if !br.read_bit()? {
        return Ok((10, 3));
    }
    if !br.read_bit()? {
        return Ok((18, 4));
    }
    Ok((34, 12))
}

/// Weights and divisor for DC prediction, from spec Table 7.47. `p` is the
/// 4-entry availability vector [L, DL, D, DR]. Returns `(W, PDIV)`.
fn weights_for(p: [bool; 4]) -> ([i32; 4], i32) {
    let key = (p[0] as u8) | ((p[1] as u8) << 1) | ((p[2] as u8) << 2) | ((p[3] as u8) << 3);
    match key {
        0b0000 => ([0, 0, 0, 0], 1),
        0b0001 => ([1, 0, 0, 0], 1),      // L
        0b0010 => ([0, 1, 0, 0], 1),      // DL
        0b0011 => ([1, 0, 0, 0], 1),      // L+DL → use L
        0b0100 => ([0, 0, 1, 0], 1),      // D
        0b0101 => ([1, 0, 1, 0], 2),      // L+D
        0b0110 => ([0, 0, 1, 0], 1),      // DL+D → use D
        0b0111 => ([29, -26, 29, 0], 32), // L+DL+D
        0b1000 => ([0, 0, 0, 1], 1),      // DR
        0b1001 => ([75, 0, 0, 53], 128),  // L+DR
        0b1010 => ([0, 1, 0, 1], 2),      // DL+DR
        0b1011 => ([75, 0, 0, 53], 128),  // L+DL+DR
        0b1100 => ([0, 0, 1, 0], 1),      // D+DR
        0b1101 => ([75, 0, 0, 53], 128),  // L+D+DR
        0b1110 => ([0, 3, 10, 3], 16),    // DL+D+DR
        0b1111 => ([29, -26, 29, 0], 32), // all four
        _ => ([0, 0, 0, 0], 1),
    }
}

/// Unused helper kept for type inference: ensure FrameLayout & PixelFormat still
/// wire up after feature churn.
#[allow(dead_code)]
fn _assert_layout_compiles(pf: PixelFormat) -> FrameLayout {
    FrameLayout::new(4, 3, pf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lflim_behaviour() {
        assert_eq!(lflim(0, 5), 0);
        assert_eq!(lflim(3, 5), 3);
        assert_eq!(lflim(-3, 5), -3);
        assert_eq!(lflim(6, 5), 4); // -6+10 = 4
        assert_eq!(lflim(-6, 5), -4);
        assert_eq!(lflim(20, 5), 0);
        assert_eq!(lflim(-20, 5), 0);
    }

    #[test]
    fn huffman_groups() {
        assert_eq!(huffman_group_for_ti(0), 0);
        assert_eq!(huffman_group_for_ti(1), 1);
        assert_eq!(huffman_group_for_ti(5), 1);
        assert_eq!(huffman_group_for_ti(6), 2);
        assert_eq!(huffman_group_for_ti(14), 2);
        assert_eq!(huffman_group_for_ti(15), 3);
        assert_eq!(huffman_group_for_ti(27), 3);
        assert_eq!(huffman_group_for_ti(28), 4);
        assert_eq!(huffman_group_for_ti(63), 4);
    }
}
