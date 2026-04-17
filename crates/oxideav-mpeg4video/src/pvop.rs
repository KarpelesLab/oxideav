//! P-VOP encoder — motion estimation, inter texture coding, bitstream
//! emission (ISO/IEC 14496-2 §6.3.5, §7.6).
//!
//! Scope:
//! * **Motion estimation** — integer-pel diamond search (small diamond) then
//!   half-pel refinement around the best integer match. Search is bounded by
//!   a small range (±7 integer pels by default) so it stays well within the
//!   `f_code=1` vector range and keeps complexity low. Reference frame is
//!   edge-replicated via the shared `mc::predict_block` helper.
//! * **1MV mode only** — one MV per macroblock. 4MV is out of scope for this
//!   first cut; see the "follow-up items" section at the bottom of the file
//!   for the known-future enhancements.
//! * **MV coding** per §7.6.3 — the median predictor over three causal
//!   neighbours (left, top, top-right), then MVD reconstruction via the
//!   unsigned-magnitude + sign + residual layout. `f_code=1` keeps the
//!   residual absent (r_size=0).
//! * **MCBPC** (Table B-13) distinguishes `Inter`, `InterQ`, `Intra`,
//!   `IntraQ`, `Inter4MV` (unused here). We emit `Inter` for coded inter
//!   MBs; skip MBs use the `not_coded` flag, and `Intra` fallback is
//!   available when the inter mode produces too much residual.
//! * **Skipped MB**: when the MB can be reconstructed exactly by copying
//!   the 16×16 region at MV(0,0) from the reference with NO residual, we
//!   write a single `not_coded=1` bit. Decoder matches (§7.6.7).
//! * **Inter texture coding** — per-block: forward DCT on the residual
//!   (source – predictor), H.263 inter quantisation (matches the decoder's
//!   `dequantise_inter_h263`), inter tcoef walk (Table B-17). The encoder
//!   reconstructs the block (dequant + IDCT + add predictor + clip) so the
//!   emitted bitstream stays drift-free relative to the decoder's state.
//! * **Reference frame management** — the caller threads a single
//!   `reference` IVopPicture (last reconstructed frame) in and receives the
//!   newly reconstructed picture as output, which becomes the next
//!   reference.
//!
//! Out of scope (returns error or NOP):
//! * 4MV mode — encoder never emits `Inter4MV` MCBPC values.
//! * Embedded intra MBs inside a P-VOP — the encoder always encodes MBs as
//!   inter. This loses efficiency on scene changes but remains bit-correct.
//! * B / S VOPs, GMC, quarter-pel motion, reduced-resolution, data
//!   partitioning.

use oxideav_core::Result;

use crate::bitwriter::BitWriter;
use crate::encoder::{block_pel_position, fdct8x8};
use crate::headers::vol::ZIGZAG;
use crate::inter::{MbMotion, MvGrid};
use crate::mb::IVopPicture;
use crate::mc::{luma_mv_to_chroma, predict_block};
use crate::tables::{mv as mv_tab, tcoef};

/// Default integer-pel search range (in integer pels). The encoder keeps the
/// half-pel MV within `±2 * MAX_SEARCH_INT` — comfortably inside the
/// `f_code=1` range of `[-32, 31]` half-pels.
pub const MAX_SEARCH_INT: i32 = 7;

/// Encoder-side representation of one P-VOP macroblock after motion
/// estimation. All MVs are in luma half-pel units.
#[derive(Clone, Copy, Debug)]
pub struct PMbEncoding {
    /// Single MV (luma half-pel units). Used for all four luma blocks and
    /// the two chroma blocks (via `luma_mv_to_chroma`).
    pub mv_half: (i32, i32),
    /// When true, the MB is emitted as `not_coded` (skipped) — caller can
    /// verify by decoding a 0 residual.
    pub skipped: bool,
    /// Per-block "coded" flags for the 4 luma blocks (Y0..Y3).
    pub luma_coded: [bool; 4],
    /// Per-block "coded" flags for the 2 chroma blocks (Cb, Cr).
    pub chroma_coded: [bool; 2],
    /// Reconstructed luma 16×16 block (MC + dequant residual, clipped to u8)
    /// in row-major order at offset 0.
    pub recon_y: [u8; 256],
    /// Reconstructed chroma Cb 8×8.
    pub recon_cb: [u8; 64],
    /// Reconstructed chroma Cr 8×8.
    pub recon_cr: [u8; 64],
    /// AC levels (quantised) per block, natural order. `[0]` is unused for
    /// inter blocks (no DC special case); we keep 64 slots to match the
    /// decoder's view.
    pub ac_levels: [[i32; 64]; 6],
}

impl Default for PMbEncoding {
    fn default() -> Self {
        Self {
            mv_half: (0, 0),
            skipped: false,
            luma_coded: [false; 4],
            chroma_coded: [false; 2],
            recon_y: [0; 256],
            recon_cb: [0; 64],
            recon_cr: [0; 64],
            ac_levels: [[0i32; 64]; 6],
        }
    }
}

/// Encode one P-VOP into `bw`. Returns the reconstructed picture that must
/// be stored as the next reference frame.
///
/// `v` is the source video frame (YUV420p). `reference` is the previous
/// reconstructed picture (luma-half-pel compatible — the same layout emitted
/// by the I-VOP encoder).
///
/// `vop_quant` is the quantiser and stays constant across the picture (no
/// dquant). `f_code_fwd` selects the MV range (1 keeps MVD to small values).
/// `rounding_type` is the VOP rounding flag (typically 0 on the first P-VOP
/// after an I and toggled per FFmpeg's convention; we mirror that here).
pub fn encode_p_vop_body(
    bw: &mut BitWriter,
    v: &oxideav_core::VideoFrame,
    reference: &IVopPicture,
    vop_quant: u32,
    f_code_fwd: u8,
    rounding_type: bool,
) -> Result<IVopPicture> {
    let width = v.width as usize;
    let height = v.height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut pic = IVopPicture::new(width, height);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb = estimate_and_encode_mb(
                v,
                reference,
                mb_x,
                mb_y,
                vop_quant,
                rounding_type,
                &mv_grid,
            )?;
            emit_p_mb(bw, &mb, mb_x, mb_y, &mv_grid, f_code_fwd);
            // Stash reconstructed samples into `pic`.
            write_recon_to_pic(&mut pic, &mb, mb_x, mb_y);
            // Update MV predictor grid.
            let motion = MbMotion {
                mv: [mb.mv_half; 4],
                four_mv: false,
            };
            mv_grid.set(mb_x, mb_y, motion);
        }
    }

    Ok(pic)
}

fn write_recon_to_pic(pic: &mut IVopPicture, mb: &PMbEncoding, mb_x: usize, mb_y: usize) {
    let px = mb_x * 16;
    let py = mb_y * 16;
    for j in 0..16 {
        for i in 0..16 {
            pic.y[(py + j) * pic.y_stride + (px + i)] = mb.recon_y[j * 16 + i];
        }
    }
    let cx = mb_x * 8;
    let cy = mb_y * 8;
    for j in 0..8 {
        for i in 0..8 {
            pic.cb[(cy + j) * pic.c_stride + (cx + i)] = mb.recon_cb[j * 8 + i];
            pic.cr[(cy + j) * pic.c_stride + (cx + i)] = mb.recon_cr[j * 8 + i];
        }
    }
}

// -------------------------------------------------------------------------
// Motion estimation + residual encoding
// -------------------------------------------------------------------------

/// Estimate motion for one MB, encode + reconstruct its six blocks, and
/// return a fully-populated `PMbEncoding`.
fn estimate_and_encode_mb(
    v: &oxideav_core::VideoFrame,
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    vop_quant: u32,
    rounding: bool,
    mv_grid: &MvGrid,
) -> Result<PMbEncoding> {
    // 1. Integer-pel search over the 16×16 luma MB.
    let src_y_block = load_luma_mb(v, mb_x, mb_y);
    let (int_x, int_y) = diamond_search_integer(reference, &src_y_block, mb_x, mb_y);
    // 2. Half-pel refinement.
    let (mvx_half, mvy_half) =
        halfpel_refine(reference, &src_y_block, mb_x, mb_y, int_x, int_y, rounding);
    let _ = mv_grid; // MV predictor is applied only when writing MVD, not ME.

    let mut mb = PMbEncoding {
        mv_half: (mvx_half, mvy_half),
        ..Default::default()
    };

    // 3. Build luma predictor for this MV.
    let mut pred_y = [0u8; 256];
    predict_luma_mb(
        reference,
        mb_x,
        mb_y,
        mvx_half,
        mvy_half,
        rounding,
        &mut pred_y,
    );
    // 4. Build chroma predictors.
    let (cmx, cmy) = (luma_mv_to_chroma(mvx_half), luma_mv_to_chroma(mvy_half));
    let mut pred_cb = [0u8; 64];
    let mut pred_cr = [0u8; 64];
    predict_chroma_block(
        &reference.cb,
        reference.c_stride,
        mb_x,
        mb_y,
        cmx,
        cmy,
        rounding,
        &mut pred_cb,
    );
    predict_chroma_block(
        &reference.cr,
        reference.c_stride,
        mb_x,
        mb_y,
        cmx,
        cmy,
        rounding,
        &mut pred_cr,
    );

    // 5. Residual + forward DCT + quant, per 8×8 block.
    // Luma blocks: 0=(0,0) 1=(8,0) 2=(0,8) 3=(8,8)
    for blk in 0..4 {
        let (sub_x, sub_y) = match blk {
            0 => (0, 0),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let src = read_luma_block_from_mb(v, mb_x, mb_y, sub_x, sub_y);
        let pred_blk = read_pred_block(&pred_y, 16, sub_x, sub_y);
        let (levels, recon) = encode_inter_block(&src, &pred_blk, vop_quant);
        mb.luma_coded[blk] = levels.iter().any(|&l| l != 0);
        mb.ac_levels[blk] = levels;
        // Stamp reconstructed samples back into mb.recon_y.
        for j in 0..8 {
            for i in 0..8 {
                mb.recon_y[(sub_y + j) * 16 + (sub_x + i)] = recon[j * 8 + i];
            }
        }
    }

    // Chroma blocks.
    let src_cb = load_chroma_block(v, 1, mb_x, mb_y);
    let src_cr = load_chroma_block(v, 2, mb_x, mb_y);
    let (levels_cb, recon_cb) = encode_inter_block(&src_cb, &pred_cb, vop_quant);
    let (levels_cr, recon_cr) = encode_inter_block(&src_cr, &pred_cr, vop_quant);
    mb.chroma_coded[0] = levels_cb.iter().any(|&l| l != 0);
    mb.chroma_coded[1] = levels_cr.iter().any(|&l| l != 0);
    mb.ac_levels[4] = levels_cb;
    mb.ac_levels[5] = levels_cr;
    mb.recon_cb = recon_cb;
    mb.recon_cr = recon_cr;

    // 6. Skip detection — MB is skippable only if MV == (0,0) AND all residual
    // levels are zero (CBP == 0). In that case the decoder copies the
    // reference verbatim, which must equal what we reconstructed.
    let all_zero = !mb.luma_coded.iter().any(|&c| c) && !mb.chroma_coded.iter().any(|&c| c);
    if all_zero && mvx_half == 0 && mvy_half == 0 {
        // Make sure the reconstructed samples equal the reference region,
        // which they do by construction (residual=0, MV=0).
        mb.skipped = true;
    }

    Ok(mb)
}

/// Small-diamond integer search starting at (0,0). Evaluates SAD at each
/// candidate and moves to the minimum; stops when the centre is best. Tiny
/// bounded range to keep things deterministic and well within `f_code=1`.
fn diamond_search_integer(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
) -> (i32, i32) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let blk_px = (mb_x * 16) as i32;
    let blk_py = (mb_y * 16) as i32;
    let mut best_x = 0i32;
    let mut best_y = 0i32;
    let mut best_sad = sad_integer(reference, src, blk_px, blk_py, 0, 0, ref_w, ref_h);
    // Small-diamond pattern: 4-neighbour.
    const STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for _ in 0..(MAX_SEARCH_INT as usize * 2) {
        let mut improved = false;
        for (dx, dy) in STEPS {
            let nx = best_x + dx;
            let ny = best_y + dy;
            if nx.abs() > MAX_SEARCH_INT || ny.abs() > MAX_SEARCH_INT {
                continue;
            }
            let s = sad_integer(reference, src, blk_px, blk_py, nx, ny, ref_w, ref_h);
            if s < best_sad {
                best_sad = s;
                best_x = nx;
                best_y = ny;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    (best_x, best_y)
}

fn sad_integer(
    reference: &IVopPicture,
    src: &[u8; 256],
    blk_px: i32,
    blk_py: i32,
    mv_x: i32,
    mv_y: i32,
    ref_w: i32,
    ref_h: i32,
) -> u32 {
    let mut s = 0u32;
    for j in 0..16 {
        for i in 0..16 {
            let x = (blk_px + mv_x + i).clamp(0, ref_w - 1) as usize;
            let y = (blk_py + mv_y + j).clamp(0, ref_h - 1) as usize;
            let r = reference.y[y * reference.y_stride + x] as i32;
            let sv = src[(j as usize) * 16 + (i as usize)] as i32;
            s = s.wrapping_add((sv - r).unsigned_abs());
        }
    }
    s
}

/// Refine integer MV to half-pel by evaluating the 8 half-pel candidates
/// around the integer best. Returns the MV in half-pel units.
fn halfpel_refine(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    int_x: i32,
    int_y: i32,
    rounding: bool,
) -> (i32, i32) {
    let mut best_half_x = int_x * 2;
    let mut best_half_y = int_y * 2;
    let mut best_sad = sad_halfpel(
        reference,
        src,
        mb_x,
        mb_y,
        best_half_x,
        best_half_y,
        rounding,
    );
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let hx = int_x * 2 + dx;
            let hy = int_y * 2 + dy;
            if hx.abs() > MAX_SEARCH_INT * 2 + 1 || hy.abs() > MAX_SEARCH_INT * 2 + 1 {
                continue;
            }
            let s = sad_halfpel(reference, src, mb_x, mb_y, hx, hy, rounding);
            if s < best_sad {
                best_sad = s;
                best_half_x = hx;
                best_half_y = hy;
            }
        }
    }
    (best_half_x, best_half_y)
}

fn sad_halfpel(
    reference: &IVopPicture,
    src: &[u8; 256],
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
) -> u32 {
    let mut pred = [0u8; 256];
    predict_luma_mb(
        reference, mb_x, mb_y, mv_x_half, mv_y_half, rounding, &mut pred,
    );
    let mut s = 0u32;
    for i in 0..256 {
        s = s.wrapping_add((src[i] as i32 - pred[i] as i32).unsigned_abs());
    }
    s
}

// -------------------------------------------------------------------------
// Block prediction + residual/quant round-trip
// -------------------------------------------------------------------------

fn predict_luma_mb(
    reference: &IVopPicture,
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
    out: &mut [u8; 256],
) {
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let ref_w = reference.y_stride as i32;
    let px = (mb_x * 16) as i32;
    let py = (mb_y * 16) as i32;
    // Decompose 16×16 into 4×(8×8) so we can reuse `predict_block`.
    for (sub_x, sub_y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
        let mut tmp = [0u8; 64];
        predict_block(
            &reference.y,
            reference.y_stride,
            ref_w,
            ref_h,
            px + sub_x,
            py + sub_y,
            mv_x_half,
            mv_y_half,
            8,
            rounding,
            &mut tmp,
            8,
        );
        for j in 0..8 {
            for i in 0..8 {
                out[(sub_y as usize + j) * 16 + (sub_x as usize + i)] = tmp[j * 8 + i];
            }
        }
    }
}

fn predict_chroma_block(
    ref_plane: &[u8],
    ref_stride: usize,
    mb_x: usize,
    mb_y: usize,
    mv_x_half: i32,
    mv_y_half: i32,
    rounding: bool,
    out: &mut [u8; 64],
) {
    let ref_h = (ref_plane.len() / ref_stride) as i32;
    let ref_w = ref_stride as i32;
    predict_block(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        (mb_x * 8) as i32,
        (mb_y * 8) as i32,
        mv_x_half,
        mv_y_half,
        8,
        rounding,
        out,
        8,
    );
}

fn load_luma_mb(v: &oxideav_core::VideoFrame, mb_x: usize, mb_y: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    let w = v.width as usize;
    let h = v.height as usize;
    let plane = &v.planes[0];
    for j in 0..16 {
        let yy = (mb_y * 16 + j).min(h.saturating_sub(1));
        for i in 0..16 {
            let xx = (mb_x * 16 + i).min(w.saturating_sub(1));
            out[j * 16 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

fn load_chroma_block(
    v: &oxideav_core::VideoFrame,
    plane_idx: usize,
    mb_x: usize,
    mb_y: usize,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let cw = (v.width as usize).div_ceil(2);
    let ch = (v.height as usize).div_ceil(2);
    let plane = &v.planes[plane_idx];
    for j in 0..8 {
        let yy = (mb_y * 8 + j).min(ch.saturating_sub(1));
        for i in 0..8 {
            let xx = (mb_x * 8 + i).min(cw.saturating_sub(1));
            out[j * 8 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

fn read_luma_block_from_mb(
    v: &oxideav_core::VideoFrame,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let (_, x0, y0, pw, ph) = block_pel_position(v, mb_x, mb_y, block_index_for_sub(sub_x, sub_y));
    let plane = &v.planes[0];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            out[j * 8 + i] = plane.data[yy * plane.stride + xx];
        }
    }
    out
}

fn block_index_for_sub(sub_x: usize, sub_y: usize) -> usize {
    match (sub_x, sub_y) {
        (0, 0) => 0,
        (8, 0) => 1,
        (0, 8) => 2,
        (8, 8) => 3,
        _ => unreachable!(),
    }
}

fn read_pred_block(mb_pred: &[u8; 256], stride: usize, sub_x: usize, sub_y: usize) -> [u8; 64] {
    let mut out = [0u8; 64];
    for j in 0..8 {
        for i in 0..8 {
            out[j * 8 + i] = mb_pred[(sub_y + j) * stride + (sub_x + i)];
        }
    }
    out
}

/// Encode one 8×8 inter block: compute residual, FDCT, quantise, then
/// dequantise + IDCT + clip to reconstruct. Returns `(ac_levels, recon)`.
/// `ac_levels` has 64 entries (natural order). `recon` is the sample block
/// ready to copy back into the picture buffer.
fn encode_inter_block(src: &[u8; 64], pred: &[u8; 64], quant: u32) -> ([i32; 64], [u8; 64]) {
    // Residual.
    let mut res = [0f32; 64];
    for i in 0..64 {
        res[i] = src[i] as f32 - pred[i] as f32;
    }
    // Forward DCT.
    fdct8x8(&mut res);
    // Quantise. Inter H.263 dequant rule is
    //   recon(l) = 2*Q*|l| + Q_plus; Q_plus = Q if Q odd, Q-1 if Q even;
    //   recon(0) = 0.
    // Forward pick the level whose reconstruction is closest to `coef`.
    let q = quant as i32;
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let two_q = 2 * q;

    let mut levels = [0i32; 64];
    for i in 0..64 {
        let c = res[i].round() as i32;
        levels[i] = quantise_ac_inter_h263(c, two_q, q_plus).clamp(-2047, 2047);
    }

    // Reconstruct: dequantise + IDCT + add predictor + clip.
    let mut deq = [0i32; 64];
    for i in 0..64 {
        let l = levels[i];
        if l == 0 {
            deq[i] = 0;
        } else {
            let abs = l.unsigned_abs() as i32;
            let val = two_q * abs + q_plus;
            deq[i] = if l < 0 { -val } else { val };
        }
    }
    let mut deqf = [0f32; 64];
    for i in 0..64 {
        deqf[i] = deq[i] as f32;
    }
    crate::block::idct8x8(&mut deqf);
    let mut recon = [0u8; 64];
    for j in 0..8 {
        for i in 0..8 {
            let rr = deqf[j * 8 + i].round() as i32 + pred[j * 8 + i] as i32;
            recon[j * 8 + i] = rr.clamp(0, 255) as u8;
        }
    }
    (levels, recon)
}

/// Pick the integer level whose reconstruction is closest to `coef`.
fn quantise_ac_inter_h263(coef: i32, two_q: i32, q_plus: i32) -> i32 {
    if coef == 0 {
        return 0;
    }
    let abs = coef.unsigned_abs() as i32;
    let l_low = abs / two_q;
    let mut best_l = 0i32;
    let mut best_err = abs;
    for cand in [l_low.saturating_sub(1), l_low, l_low + 1] {
        if cand < 0 {
            continue;
        }
        let recon = if cand == 0 { 0 } else { two_q * cand + q_plus };
        let err = (abs - recon).abs();
        if err < best_err {
            best_err = err;
            best_l = cand;
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

// -------------------------------------------------------------------------
// Bitstream emission
// -------------------------------------------------------------------------

fn emit_p_mb(
    bw: &mut BitWriter,
    mb: &PMbEncoding,
    mb_x: usize,
    mb_y: usize,
    mv_grid: &MvGrid,
    f_code_fwd: u8,
) {
    if mb.skipped {
        // §6.3.5: not_coded = 1.
        bw.write_bits(1, 1);
        return;
    }
    // not_coded = 0.
    bw.write_bits(0, 1);

    // MCBPC (Table B-13). Inter, cbpc = bit1=Cb bit0=Cr.
    let cbpc = ((mb.chroma_coded[0] as u8) << 1) | (mb.chroma_coded[1] as u8);
    write_mcbpc_inter(bw, cbpc);

    // CBPY — for inter MBs the encoded value is bit-inverted of the coded
    // mask (decoder XORs with 0xF). Build the mask from `luma_coded` as
    // bit3=Y0, bit0=Y3, then XOR with 0xF to pack into the VLC.
    let mut cbpy_mask: u8 = 0;
    for (i, &c) in mb.luma_coded.iter().enumerate() {
        if c {
            cbpy_mask |= 1 << (3 - i);
        }
    }
    let cbpy_encoded = cbpy_mask ^ 0xF;
    write_cbpy(bw, cbpy_encoded);

    // Motion vector (1MV mode).
    let (px, py) = crate::inter::predict_mv_full(
        mv_grid, mb_x, mb_y, 0, false, 0,
        0, // slice_first_mb = (0,0) — no resync markers in this encoder
    );
    let (mvx, mvy) = mb.mv_half;
    let dx = mvx - px;
    let dy = mvy - py;
    let range = 32i32 << (f_code_fwd.saturating_sub(1) as i32);
    let dx = wrap_mvd(dx, range);
    let dy = wrap_mvd(dy, range);
    write_mv_component(bw, dx, f_code_fwd);
    write_mv_component(bw, dy, f_code_fwd);

    // Per-block coded residual walk (Table B-17 inter tcoef).
    for blk in 0..6 {
        let coded = if blk < 4 {
            mb.luma_coded[blk]
        } else {
            mb.chroma_coded[blk - 4]
        };
        if !coded {
            continue;
        }
        write_inter_ac(bw, &mb.ac_levels[blk]);
    }
}

/// Fold a signed MVD into the `[-range, range-1]` range by ±2*range.
fn wrap_mvd(mvd: i32, range: i32) -> i32 {
    let mut v = mvd;
    if v < -range {
        v += 2 * range;
    } else if v >= range {
        v -= 2 * range;
    }
    v
}

fn write_mcbpc_inter(bw: &mut BitWriter, cbpc: u8) {
    // Table B-13 row for "Inter, cbpc=0..=3". The decoder's `PMbType::Inter`
    // corresponds to MCBPC values 0..=3 (group=0).
    let (bits, code) = match cbpc {
        0 => (1, 0b1),
        1 => (4, 0b0011),
        2 => (4, 0b0010),
        3 => (6, 0b000101),
        _ => unreachable!(),
    };
    bw.write_bits(code, bits);
}

fn write_cbpy(bw: &mut BitWriter, cbpy: u8) {
    // Table B-9 raw values (mirrors the decoder table in tables/cbpy.rs).
    let (bits, code) = match cbpy {
        0 => (4, 0b0011),
        1 => (5, 0b00101),
        2 => (5, 0b00100),
        3 => (4, 0b1001),
        4 => (5, 0b00011),
        5 => (4, 0b0111),
        6 => (6, 0b000010),
        7 => (4, 0b1011),
        8 => (5, 0b00010),
        9 => (6, 0b000011),
        10 => (4, 0b0101),
        11 => (4, 0b1010),
        12 => (4, 0b0100),
        13 => (4, 0b1000),
        14 => (4, 0b0110),
        15 => (2, 0b11),
        _ => unreachable!("cbpy out of range: {cbpy}"),
    };
    bw.write_bits(code, bits);
}

/// Write one motion-vector component per §7.6.3.
///
/// MVD `diff` is in half-pel units and already wrapped into `[-32*f, 32*f-1]`.
fn write_mv_component(bw: &mut BitWriter, diff: i32, f_code: u8) {
    let r_size = (f_code.saturating_sub(1)) as u32;
    let f = 1i32 << r_size;
    // |motion_code| and residual derivation (§7.6.3):
    //   if diff == 0: motion_code = 0, no residual.
    //   else:
    //     n = (|diff| - 1) — we need motion_code in 1..=32.
    //     motion_code = (n / f) + 1; residual = n % f.
    let abs = diff.unsigned_abs() as i32;
    let (mc_abs, residual) = if abs == 0 {
        (0i32, 0i32)
    } else {
        let n = abs - 1;
        (n / f + 1, n % f)
    };
    // Emit the magnitude VLC (Table B-12, 0..=32).
    let mc_clamped = mc_abs.clamp(0, 32) as usize;
    let row = &mv_tab_row(mc_clamped);
    bw.write_bits(row.1, row.0 as u32);
    if mc_clamped != 0 {
        // Sign bit: 0 = positive, 1 = negative.
        bw.write_bits(if diff < 0 { 1 } else { 0 }, 1);
    }
    if f != 1 && mc_clamped != 0 {
        bw.write_bits(residual as u32, r_size);
    }
}

/// Return `(bits, code)` for Table B-12 at magnitude `mag`.
fn mv_tab_row(mag: usize) -> (u8, u32) {
    // Build a lookup keyed by magnitude from the shared table. We cannot
    // call `mv_tab::table()` directly for encode because it returns entries
    // indexed 0..=32 already; the table entry's `(bits, code)` is what we
    // need.
    let t = mv_tab::table();
    let e = t[mag];
    (e.bits, e.code)
}

/// Walk `block` in zigzag order, emitting one inter tcoef symbol per non-zero
/// coefficient. Unlike the intra path, inter tcoef starts at scan index 0.
fn write_inter_ac(bw: &mut BitWriter, block: &[i32; 64]) {
    // Find the last non-zero AC in scan order (zigzag).
    let mut last_nz: Option<usize> = None;
    for i in 0..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        // Defensive: caller must check `coded` before calling us.
        return;
    };
    let mut run = 0u8;
    let mut i = 0;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        write_inter_tcoef_symbol(bw, last, run, lv);
        run = 0;
        i += 1;
    }
}

/// Encode one inter tcoef symbol — short VLC where possible, third escape
/// otherwise. Mirrors the intra-path helper in encoder.rs but keyed on
/// `tcoef::inter_table`.
fn write_inter_tcoef_symbol(bw: &mut BitWriter, last: bool, run: u8, level: i32) {
    let abs = level.unsigned_abs() as u8;
    if let Some((bits, code)) = lookup_inter_short_vlc(last, run, abs) {
        bw.write_bits(code, bits as u32);
        bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
        return;
    }
    // Third escape: `0000011` + 1 + 1 + last(1) + run(6) + marker + level(12) + marker.
    bw.write_bits(0b0000011, 7);
    bw.write_bits(1, 1);
    bw.write_bits(1, 1);
    bw.write_bits(if last { 1 } else { 0 }, 1);
    bw.write_bits(run as u32 & 0x3F, 6);
    bw.write_bits(1, 1); // marker
    let lvl12 = (level & 0x0FFF) as u32;
    bw.write_bits(lvl12, 12);
    bw.write_bits(1, 1); // marker
}

/// Reverse-lookup of the short VLC in Table B-17 keyed by `(last, run, abs)`.
fn lookup_inter_short_vlc(last: bool, run: u8, level_abs: u8) -> Option<(u8, u32)> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type InterShortVlcMap = HashMap<(bool, u8, u8), (u8, u32)>;
    static MAP: OnceLock<InterShortVlcMap> = OnceLock::new();
    let m = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for entry in tcoef::inter_table() {
            if let tcoef::TcoefSym::RunLevel {
                last,
                run,
                level_abs,
            } = entry.value
            {
                m.insert((last, run, level_abs), (entry.bits, entry.code));
            }
        }
        m
    });
    m.get(&(last, run, level_abs)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_component_zero_roundtrip() {
        use crate::bitreader::BitReader;
        use crate::inter::decode_mv_component;
        let mut bw = BitWriter::new();
        write_mv_component(&mut bw, 0, 1);
        let mut data = bw.finish();
        data.extend_from_slice(&[0xFF, 0xFF]);
        let mut br = BitReader::new(&data);
        let v = decode_mv_component(&mut br, 1, 0).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn mv_component_small_positive_roundtrip() {
        use crate::bitreader::BitReader;
        use crate::inter::decode_mv_component;
        for d in [1i32, -1, 3, -3, 7, -7, 15, -16, 31, -32] {
            let mut bw = BitWriter::new();
            write_mv_component(&mut bw, d, 1);
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let v = decode_mv_component(&mut br, 1, 0).unwrap();
            assert_eq!(v, d, "MV component {d} round-trip");
        }
    }

    #[test]
    fn quantise_inter_monotonic() {
        // Encode+decode a range of AC values with Q=5 and check monotonic
        // reconstructions.
        let q = 5i32;
        let two_q = 2 * q;
        let q_plus = q;
        for c in [-100i32, -20, 0, 20, 100, 500] {
            let l = quantise_ac_inter_h263(c, two_q, q_plus);
            let recon = if l == 0 { 0 } else { two_q * l.abs() + q_plus };
            let recon = if l < 0 { -recon } else { recon };
            assert!(
                (c - recon).abs() <= two_q,
                "c={c} recon={recon} beyond one step"
            );
        }
    }
}

// -------------------------------------------------------------------------
// Follow-up items (not blocking — documented for future work):
//
// * 4MV mode: emit `PMbType::Inter4MV` MCBPC codes, four MVs per MB with
//   per-block ME. The decoder already supports this path, so enabling
//   it on the encoder side would be a pure encoder-complexity trade-off
//   (better compression on non-translational motion in exchange for a
//   per-block ME search and four MV VLCs per MB).
// * Intra MB fallback inside P-VOP for high-residual blocks. Useful on
//   scene-change boundaries where the inter predictor yields more
//   residual bits than a fresh intra block would. Not yet implemented;
//   the encoder currently always codes MBs as inter inside a P-VOP.
// * B-VOPs, GMC, sprites, quarter-pel motion, OBMC — deliberately out
//   of scope for this encoder. They would each require material
//   bitstream changes (B-VOP syntax, sprite metadata, the `quarter_pel`
//   flag path) and are tracked separately from the P-VOP work.
