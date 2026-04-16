//! Macroblock / slice-level decoding.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::block::{copy_prediction, decode_intra_block, decode_non_intra_block};
use crate::headers::{PictureHeader, PictureType, SequenceHeader};
use crate::motion::{self, MvPredictor};
use crate::picture::PictureBuffer;
use crate::tables::{cbp, mb_type, mba};
use crate::vlc;

/// Running slice-decode state carried between macroblocks.
pub struct SliceState {
    pub dc_pred: [i32; 3],
    pub fwd: MvPredictor,
    pub bwd: MvPredictor,
    /// True if the previous macroblock in this slice had a forward MV.
    /// Required for B-frame "skipped MB inherits previous MB type".
    pub last_had_forward: bool,
    /// True if the previous macroblock in this slice had a backward MV.
    pub last_had_backward: bool,
}

impl SliceState {
    pub fn new() -> Self {
        Self {
            dc_pred: [1024, 1024, 1024],
            fwd: MvPredictor::default(),
            bwd: MvPredictor::default(),
            last_had_forward: false,
            last_had_backward: false,
        }
    }
}

impl Default for SliceState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode one slice.
#[allow(clippy::too_many_arguments)]
pub fn decode_slice(
    br: &mut BitReader<'_>,
    slice_start_code: u8,
    seq: &SequenceHeader,
    pic_hdr: &PictureHeader,
    pic: &mut PictureBuffer,
    fwd_ref: Option<&PictureBuffer>,
    bwd_ref: Option<&PictureBuffer>,
) -> Result<()> {
    let picture_type = pic_hdr.picture_type;
    // Quantiser_scale (5 bits).
    let mut quant_scale = br.read_u32(5)? as u8;
    if quant_scale == 0 {
        quant_scale = 1;
    }

    // extra_bit_slice.
    while br.read_u32(1)? == 1 {
        br.read_u32(8)?;
    }

    let mut state = SliceState::new();

    let mb_row = slice_start_code as i32 - 1;
    if mb_row < 0 || (mb_row as usize) >= pic.mb_height {
        return Err(Error::invalid("slice: MB row out of range"));
    }
    let mb_width = pic.mb_width as i32;
    let mut mb_addr: i32 = mb_row * mb_width - 1;
    let mut first_mb = true;

    loop {
        // Read MB address increment — may be a sequence of stuffing codes
        // followed by an escape + actual increment.
        let mut incr: u32 = 0;
        loop {
            let sym = vlc::decode(br, mba::table())?;
            if sym == mba::STUFFING {
                continue;
            }
            if sym == mba::ESCAPE {
                incr += 33;
                continue;
            }
            incr += sym as u32;
            break;
        }
        let prev_mb_addr = mb_addr;
        mb_addr += incr as i32;
        if mb_addr >= (mb_row + 1) * mb_width {
            return Err(Error::invalid("slice: MB address past end of row"));
        }

        // Handle skipped MBs (those between prev+1..mb_addr exclusive).
        // For I-pictures: spec forbids skipped MBs (every MB must be
        // intra-coded). For P-pictures: skipped = no MC (MV=0) + forward
        // prediction from the reference, no residual. For B-pictures:
        // skipped = same MV type and MV values as previous MB, no
        // residual.
        if !first_mb && incr > 1 {
            match picture_type {
                PictureType::I => {
                    return Err(Error::invalid("I-picture: skipped MB not allowed"));
                }
                PictureType::P => {
                    // Per §2.4.4.2 / §2.4.3.4: P-skip MBs have MV=(0,0)
                    // and reset MV predictors.
                    state.fwd.reset();
                    state.bwd.reset();
                    state.last_had_forward = true;
                    state.last_had_backward = false;
                    for skip_addr in (prev_mb_addr + 1)..mb_addr {
                        let sx = (skip_addr % mb_width) as usize;
                        let sy = (skip_addr / mb_width) as usize;
                        fill_forward_predict(pic, fwd_ref, sx, sy, 0, 0)?;
                    }
                }
                PictureType::B => {
                    // Skipped B MB inherits same MV direction + values as
                    // previous MB.
                    for skip_addr in (prev_mb_addr + 1)..mb_addr {
                        let sx = (skip_addr % mb_width) as usize;
                        let sy = (skip_addr / mb_width) as usize;
                        let fwd_mv = if state.last_had_forward {
                            Some((state.fwd.x, state.fwd.y))
                        } else {
                            None
                        };
                        let bwd_mv = if state.last_had_backward {
                            Some((state.bwd.x, state.bwd.y))
                        } else {
                            None
                        };
                        fill_bidir_predict(pic, fwd_ref, bwd_ref, sx, sy, fwd_mv, bwd_mv)?;
                    }
                }
                PictureType::D => {
                    return Err(Error::unsupported("D-picture not supported"));
                }
            }
        }

        first_mb = false;

        let mb_x = (mb_addr % mb_width) as usize;
        let mb_y = (mb_addr / mb_width) as usize;

        // macroblock_type per picture.
        let mb_type_flags = match picture_type {
            PictureType::I => vlc::decode(br, mb_type::I_TABLE)?,
            PictureType::P => vlc::decode(br, mb_type::P_TABLE)?,
            PictureType::B => vlc::decode(br, mb_type::B_TABLE)?,
            PictureType::D => {
                return Err(Error::unsupported("D-picture not supported"));
            }
        };

        if mb_type_flags.quant {
            let qs = br.read_u32(5)? as u8;
            if qs != 0 {
                quant_scale = qs;
            }
        }

        // Parse MV(s).
        let mut fwd_mv: Option<(i32, i32)> = None;
        let mut bwd_mv: Option<(i32, i32)> = None;
        if mb_type_flags.motion_forward {
            let mx = motion::decode_motion_component(
                br,
                pic_hdr.forward_f_code,
                pic_hdr.full_pel_forward_vector,
                &mut state.fwd.x,
            )?;
            let my = motion::decode_motion_component(
                br,
                pic_hdr.forward_f_code,
                pic_hdr.full_pel_forward_vector,
                &mut state.fwd.y,
            )?;
            fwd_mv = Some((mx, my));
        } else if matches!(picture_type, PictureType::P) && !mb_type_flags.intra {
            // No-MC P MB: vector is (0,0) and predictor resets.
            state.fwd.reset();
            fwd_mv = Some((0, 0));
        }
        if mb_type_flags.motion_backward {
            let mx = motion::decode_motion_component(
                br,
                pic_hdr.backward_f_code,
                pic_hdr.full_pel_backward_vector,
                &mut state.bwd.x,
            )?;
            let my = motion::decode_motion_component(
                br,
                pic_hdr.backward_f_code,
                pic_hdr.full_pel_backward_vector,
                &mut state.bwd.y,
            )?;
            bwd_mv = Some((mx, my));
        }

        // Track MV availability for B-skip inheritance.
        if matches!(picture_type, PictureType::B) && !mb_type_flags.intra {
            state.last_had_forward = fwd_mv.is_some();
            state.last_had_backward = bwd_mv.is_some();
        }

        // Per §2.4.4.1: DC predictors reset whenever a non-intra MB is
        // seen (so the next intra MB starts from 1024 again).
        if !mb_type_flags.intra {
            state.dc_pred = [1024, 1024, 1024];
        }

        // Parse coded_block_pattern if this MB has pattern bit set.
        let cbp_bits: u8 = if mb_type_flags.pattern {
            vlc::decode(br, cbp::table())?
        } else if mb_type_flags.intra {
            0b111111
        } else {
            0
        };

        if mb_type_flags.intra {
            // Pure intra MB.
            decode_mb_intra(br, seq, &mut state, pic, mb_x, mb_y, quant_scale)?;
        } else {
            // Non-intra MB: apply motion compensation and optional
            // residual add.
            decode_mb_inter(
                br,
                seq,
                &mut state,
                pic,
                fwd_ref,
                bwd_ref,
                mb_x,
                mb_y,
                quant_scale,
                fwd_mv,
                bwd_mv,
                cbp_bits,
            )?;
        }

        // If the intra flag wasn't set, we already reset dc_pred earlier.
        // But the spec further says: at the end of an intra MB, predictors
        // stay in force (already updated by decode_intra_block).

        if !matches!(picture_type, PictureType::B) {
            // For I/P we track MV history so subsequent skip handling
            // within the slice is consistent.
            state.last_had_forward = mb_type_flags.motion_forward
                || (!mb_type_flags.intra && matches!(picture_type, PictureType::P));
            state.last_had_backward = mb_type_flags.motion_backward;
        }

        // Slice termination: loop until the next start code. The next
        // start code is `0x000001XX` (preceded optionally by stuffing zero
        // bytes), so we peek up to the next 23 bits (or fewer near the
        // tail) and break only if all the bits we can see are zero.
        //
        // Also break when we've reached the right edge of the row — the
        // next MB would be in the next slice/row.
        if mb_addr + 1 >= (mb_row + 1) * mb_width {
            break;
        }
        let avail = br.bits_remaining().min(23) as u32;
        if avail == 0 {
            break;
        }
        let peek = br.peek_u32(avail)?;
        if peek == 0 {
            // All trailing bits are zero — at the start of a start code.
            break;
        }
    }

    Ok(())
}

fn decode_mb_intra(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    state: &mut SliceState,
    pic: &mut PictureBuffer,
    mb_x: usize,
    mb_y: usize,
    quant_scale: u8,
) -> Result<()> {
    for b in 0..6usize {
        let (is_chroma, comp_idx, dst_x0, dst_y0, stride_ptr) = block_layout(b, mb_x, mb_y, pic);
        let buf: &mut [u8] = match comp_idx {
            0 => &mut pic.y[..],
            1 => &mut pic.cb[..],
            _ => &mut pic.cr[..],
        };
        let sub = &mut buf[dst_y0 * stride_ptr + dst_x0..];
        decode_intra_block(
            br,
            is_chroma,
            &mut state.dc_pred[comp_idx],
            quant_scale,
            &seq.intra_quantiser,
            sub,
            stride_ptr,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mb_inter(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    _state: &mut SliceState,
    pic: &mut PictureBuffer,
    fwd_ref: Option<&PictureBuffer>,
    bwd_ref: Option<&PictureBuffer>,
    mb_x: usize,
    mb_y: usize,
    quant_scale: u8,
    fwd_mv: Option<(i32, i32)>,
    bwd_mv: Option<(i32, i32)>,
    cbp_bits: u8,
) -> Result<()> {
    // Build 16x16 luma + 8x8 Cb + 8x8 Cr prediction.
    let mut pred_y = [0u8; 16 * 16];
    let mut pred_cb = [0u8; 8 * 8];
    let mut pred_cr = [0u8; 8 * 8];

    build_prediction(
        fwd_ref,
        bwd_ref,
        mb_x,
        mb_y,
        fwd_mv,
        bwd_mv,
        &mut pred_y,
        &mut pred_cb,
        &mut pred_cr,
    )?;

    // For each block, either decode non-intra residual + add, or just copy
    // prediction.
    for b in 0..6usize {
        let (is_chroma, comp_idx, dst_x0, dst_y0, stride_ptr) = block_layout(b, mb_x, mb_y, pic);

        // CBP order per §2.4.3.6: bits 5..0 map to blocks (Y0,Y1,Y2,Y3,Cb,Cr).
        // Bit 5 (0x20) = block 0; bit 0 (0x01) = block 5.
        let coded = (cbp_bits & (1 << (5 - b))) != 0;

        let (pred_slice, pred_stride, blk_size): (&[u8], usize, usize) = match b {
            0 => (&pred_y[0..], 16, 8),
            1 => (&pred_y[8..], 16, 8),
            2 => (&pred_y[16 * 8..], 16, 8),
            3 => (&pred_y[16 * 8 + 8..], 16, 8),
            4 => (&pred_cb[..], 8, 8),
            5 => (&pred_cr[..], 8, 8),
            _ => unreachable!(),
        };
        let _ = blk_size;

        let buf: &mut [u8] = match comp_idx {
            0 => &mut pic.y[..],
            1 => &mut pic.cb[..],
            _ => &mut pic.cr[..],
        };
        let sub = &mut buf[dst_y0 * stride_ptr + dst_x0..];
        if coded {
            decode_non_intra_block(
                br,
                quant_scale,
                &seq.non_intra_quantiser,
                pred_slice,
                pred_stride,
                sub,
                stride_ptr,
            )?;
        } else {
            let _ = is_chroma;
            copy_prediction(pred_slice, pred_stride, 8, sub, stride_ptr);
        }
    }
    Ok(())
}

/// Returns (is_chroma, component_index, dst_x0, dst_y0, plane_stride).
fn block_layout(
    b: usize,
    mb_x: usize,
    mb_y: usize,
    pic: &PictureBuffer,
) -> (bool, usize, usize, usize, usize) {
    match b {
        0 => (false, 0, mb_x * 16, mb_y * 16, pic.y_stride),
        1 => (false, 0, mb_x * 16 + 8, mb_y * 16, pic.y_stride),
        2 => (false, 0, mb_x * 16, mb_y * 16 + 8, pic.y_stride),
        3 => (false, 0, mb_x * 16 + 8, mb_y * 16 + 8, pic.y_stride),
        4 => (true, 1, mb_x * 8, mb_y * 8, pic.c_stride),
        5 => (true, 2, mb_x * 8, mb_y * 8, pic.c_stride),
        _ => unreachable!(),
    }
}

/// Build 16x16 luma + 8x8 chroma prediction buffers. Handles forward-only,
/// backward-only, or interpolated (average) prediction.
#[allow(clippy::too_many_arguments)]
fn build_prediction(
    fwd_ref: Option<&PictureBuffer>,
    bwd_ref: Option<&PictureBuffer>,
    mb_x: usize,
    mb_y: usize,
    fwd_mv: Option<(i32, i32)>,
    bwd_mv: Option<(i32, i32)>,
    pred_y: &mut [u8; 16 * 16],
    pred_cb: &mut [u8; 8 * 8],
    pred_cr: &mut [u8; 8 * 8],
) -> Result<()> {
    // Collect the 1 or 2 motion-compensated reference patches.
    let mut have_fwd = false;
    let mut have_bwd = false;
    let mut fwd_y = [0u8; 16 * 16];
    let mut fwd_cb = [0u8; 8 * 8];
    let mut fwd_cr = [0u8; 8 * 8];
    let mut bwd_y = [0u8; 16 * 16];
    let mut bwd_cb = [0u8; 8 * 8];
    let mut bwd_cr = [0u8; 8 * 8];

    if let Some((mx, my)) = fwd_mv {
        let Some(refp) = fwd_ref else {
            return Err(Error::invalid("forward MV without forward reference"));
        };
        mc_mb(
            refp,
            mb_x,
            mb_y,
            mx,
            my,
            &mut fwd_y,
            &mut fwd_cb,
            &mut fwd_cr,
        );
        have_fwd = true;
    }
    if let Some((mx, my)) = bwd_mv {
        let Some(refp) = bwd_ref else {
            return Err(Error::invalid("backward MV without backward reference"));
        };
        mc_mb(
            refp,
            mb_x,
            mb_y,
            mx,
            my,
            &mut bwd_y,
            &mut bwd_cb,
            &mut bwd_cr,
        );
        have_bwd = true;
    }

    match (have_fwd, have_bwd) {
        (true, false) => {
            pred_y.copy_from_slice(&fwd_y);
            pred_cb.copy_from_slice(&fwd_cb);
            pred_cr.copy_from_slice(&fwd_cr);
        }
        (false, true) => {
            pred_y.copy_from_slice(&bwd_y);
            pred_cb.copy_from_slice(&bwd_cb);
            pred_cr.copy_from_slice(&bwd_cr);
        }
        (true, true) => {
            for i in 0..16 * 16 {
                pred_y[i] = ((fwd_y[i] as u32 + bwd_y[i] as u32 + 1) >> 1) as u8;
            }
            for i in 0..8 * 8 {
                pred_cb[i] = ((fwd_cb[i] as u32 + bwd_cb[i] as u32 + 1) >> 1) as u8;
            }
            for i in 0..8 * 8 {
                pred_cr[i] = ((fwd_cr[i] as u32 + bwd_cr[i] as u32 + 1) >> 1) as u8;
            }
        }
        (false, false) => {
            return Err(Error::invalid(
                "inter MB with neither forward nor backward MV",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mc_mb(
    refp: &PictureBuffer,
    mb_x: usize,
    mb_y: usize,
    mv_x: i32,
    mv_y: i32,
    dst_y: &mut [u8; 16 * 16],
    dst_cb: &mut [u8; 8 * 8],
    dst_cr: &mut [u8; 8 * 8],
) {
    let mb_px = (mb_x * 16) as i32;
    let mb_py = (mb_y * 16) as i32;
    motion::predict_block(
        &refp.y,
        refp.y_stride,
        refp.y_stride as i32,
        (refp.y.len() / refp.y_stride) as i32,
        mb_px,
        mb_py,
        mv_x,
        mv_y,
        16,
        dst_y,
        16,
    );
    let c_px = (mb_x * 8) as i32;
    let c_py = (mb_y * 8) as i32;
    let mv_cx = motion::scale_mv_to_chroma(mv_x);
    let mv_cy = motion::scale_mv_to_chroma(mv_y);
    motion::predict_block(
        &refp.cb,
        refp.c_stride,
        refp.c_stride as i32,
        (refp.cb.len() / refp.c_stride) as i32,
        c_px,
        c_py,
        mv_cx,
        mv_cy,
        8,
        dst_cb,
        8,
    );
    motion::predict_block(
        &refp.cr,
        refp.c_stride,
        refp.c_stride as i32,
        (refp.cr.len() / refp.c_stride) as i32,
        c_px,
        c_py,
        mv_cx,
        mv_cy,
        8,
        dst_cr,
        8,
    );
}

fn fill_forward_predict(
    pic: &mut PictureBuffer,
    fwd_ref: Option<&PictureBuffer>,
    mb_x: usize,
    mb_y: usize,
    mv_x: i32,
    mv_y: i32,
) -> Result<()> {
    let refp = fwd_ref.ok_or_else(|| Error::invalid("skip MB without forward ref"))?;
    let mut y = [0u8; 16 * 16];
    let mut cb = [0u8; 8 * 8];
    let mut cr = [0u8; 8 * 8];
    mc_mb(refp, mb_x, mb_y, mv_x, mv_y, &mut y, &mut cb, &mut cr);
    write_mb(pic, mb_x, mb_y, &y, &cb, &cr);
    Ok(())
}

fn fill_bidir_predict(
    pic: &mut PictureBuffer,
    fwd_ref: Option<&PictureBuffer>,
    bwd_ref: Option<&PictureBuffer>,
    mb_x: usize,
    mb_y: usize,
    fwd_mv: Option<(i32, i32)>,
    bwd_mv: Option<(i32, i32)>,
) -> Result<()> {
    let mut y = [0u8; 16 * 16];
    let mut cb = [0u8; 8 * 8];
    let mut cr = [0u8; 8 * 8];
    build_prediction(
        fwd_ref, bwd_ref, mb_x, mb_y, fwd_mv, bwd_mv, &mut y, &mut cb, &mut cr,
    )?;
    write_mb(pic, mb_x, mb_y, &y, &cb, &cr);
    Ok(())
}

fn write_mb(
    pic: &mut PictureBuffer,
    mb_x: usize,
    mb_y: usize,
    y: &[u8; 16 * 16],
    cb: &[u8; 8 * 8],
    cr: &[u8; 8 * 8],
) {
    let ys = pic.y_stride;
    let cs = pic.c_stride;
    for j in 0..16 {
        let off = (mb_y * 16 + j) * ys + mb_x * 16;
        pic.y[off..off + 16].copy_from_slice(&y[j * 16..j * 16 + 16]);
    }
    for j in 0..8 {
        let off = (mb_y * 8 + j) * cs + mb_x * 8;
        pic.cb[off..off + 8].copy_from_slice(&cb[j * 8..j * 8 + 8]);
        pic.cr[off..off + 8].copy_from_slice(&cr[j * 8..j * 8 + 8]);
    }
}
