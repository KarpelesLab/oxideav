//! VP8 I-frame (keyframe) encoder — RFC 6386.
//!
//! Scope for v1:
//! * Keyframes only (no P-frames).
//! * DC_PRED for every luma 16×16 MB and every chroma 8×8 MB (the most
//!   uniformly-applicable mode — no context-sensitive choice required).
//! * Fixed quantiser (default `qindex = 50`, mid-quality).
//! * Loop filter disabled (`filter_level = 0`).
//! * Single token partition.
//! * Accepted pixel format: `PixelFormat::Yuv420P`.
//!
//! The encoder mirrors the decoder's pipeline closely so that the two
//! agree on MB state:
//!   1. Forward 4×4 DCT on each Y/U/V 4×4 residual block.
//!   2. Forward WHT on the 16 DC coefficients (Y2 path) for non-B_PRED MBs.
//!   3. Quantise by the per-block stepsize.
//!   4. Immediately reconstruct the MB (dequantise + inverse transform
//!      + prediction) so the next MB's DC_PRED neighbours are bit-exact.
//!   5. Encode the quantised coefficients with the default token
//!      probabilities using the write-side boolean coder.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::bool_encoder::BoolEncoder;
use crate::fdct::{fdct4x4, fwht4x4};
use crate::frame_tag::KEYFRAME_SYNC_CODE;
use crate::intra::{predict_16x16, predict_8x8};
use crate::tables::coeff_probs::{CoeffProbs, DEFAULT_COEF_PROBS};
use crate::tables::quant::{
    clamp_qindex, uv_ac_step, uv_dc_step, y2_ac_step, y2_dc_step, y_ac_step, y_dc_step,
};
use crate::tables::token_tree::{COEF_BANDS, ZIGZAG};
use crate::tables::trees::DC_PRED;
use crate::transform::{idct4x4, iwht4x4};

/// Default qindex. 50 ≈ mid-quality; the codec accepts 0..=127.
pub const DEFAULT_QINDEX: u8 = 50;

/// Encoder factory used by [`crate::register_codecs`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("vp8 encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(Error::invalid(format!(
            "vp8 encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        qindex: DEFAULT_QINDEX,
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Build an encoder with an explicit qindex. Useful for tests and for
/// callers that want finer control than the default quality.
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,
) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("vp8 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "vp8 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }
    let frame_rate = params.frame_rate.unwrap_or(Rational::new(30, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);
    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);
    Ok(Box::new(Vp8Encoder {
        output_params,
        width,
        height,
        qindex: qindex.min(127),
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Vp8Encoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    qindex: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Vp8Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("vp8 encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(format!(
                "vp8 encoder: frame dims {}x{} do not match encoder {}x{}",
                v.width, v.height, self.width, self.height
            )));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::invalid("vp8 encoder: only Yuv420P input frames"));
        }
        if v.planes.len() < 3 {
            return Err(Error::invalid("vp8 encoder: expected 3 planes"));
        }

        // Every frame is an I-frame in this version.
        let data = encode_keyframe(self.width, self.height, self.qindex, v)?;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

/// Encode one keyframe. Returns the raw VP8 bitstream for the frame.
pub fn encode_keyframe(
    width: u32,
    height: u32,
    qindex: u8,
    frame: &VideoFrame,
) -> Result<Vec<u8>> {
    let mb_w = ((width + 15) / 16) as usize;
    let mb_h = ((height + 15) / 16) as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;

    // Copy (and MB-pad) the source into our own buffers.
    let (src_y, src_u, src_v) = extract_mb_padded(frame, mb_w, mb_h)?;

    // Allocate reconstruction buffers (they track, pixel-for-pixel, what
    // the decoder will produce — needed for intra prediction context in
    // subsequent MBs).
    let mut rec_y = vec![0u8; y_stride * y_buf_h];
    let mut rec_u = vec![0u8; uv_stride * uv_buf_h];
    let mut rec_v = vec![0u8; uv_stride * uv_buf_h];

    // Pre-compute quant steps.
    let qi = clamp_qindex(qindex as i32);
    let q = QuantCtx {
        y_dc: y_dc_step(qi as i32),
        y_ac: y_ac_step(qi as i32),
        y2_dc: y2_dc_step(qi as i32),
        y2_ac: y2_ac_step(qi as i32),
        uv_dc: uv_dc_step(qi as i32),
        uv_ac: uv_ac_step(qi as i32),
    };

    // --- Compressed header ---
    let mut hdr_enc = BoolEncoder::new();
    // color_space + clamping_type (1 bit each)
    hdr_enc.write_literal(1, 0);
    hdr_enc.write_literal(1, 0);
    // segmentation enabled = 0
    hdr_enc.write_bool(128, false);
    // loop filter: filter_type=0, level=0 (disables LF), sharpness=0,
    //              mode_ref_delta_enabled=0.
    hdr_enc.write_literal(1, 0);
    hdr_enc.write_literal(6, 0);
    hdr_enc.write_literal(3, 0);
    hdr_enc.write_bool(128, false);
    // log2_nb_partitions = 0 (1 partition).
    hdr_enc.write_literal(2, 0);
    // Quant: y_ac_qi + zero deltas (each delta is a 1-bit "present" flag = 0).
    hdr_enc.write_literal(7, qi as u32);
    for _ in 0..5 {
        hdr_enc.write_bool(128, false);
    }
    // refresh_entropy_probs = 0 (we keep defaults).
    hdr_enc.write_bool(128, false);
    // Skip per-prob coefficient probability updates — send "no update" for all.
    emit_no_coef_prob_updates(&mut hdr_enc);
    // mb_skip_enabled = 0.
    hdr_enc.write_bool(128, false);

    // --- MB mode info (still boolean-coded into the same first partition) ---
    // All MBs: segment id (not written, seg disabled); skip (not written,
    // skip disabled); y_mode = DC_PRED (KF_YMODE_TREE leaf code 1, probs 145);
    // uv_mode = DC_PRED (KF_UV_MODE_TREE leaf code 0, prob 142).
    // KF_YMODE_TREE = [-B_PRED, 2, 4, 6, -DC, -V, -H, -TM], probs = [145, 156, 163, 128].
    //  DC_PRED path: bit0=1 (prob 145 → not B_PRED), goto idx 2, bit1=0 (prob 156 → not V/H/TM yet), goto idx 4, bit2=0 (prob 163 → DC_PRED leaf).
    // KF_UV_MODE_TREE = [-DC, 2, -V, 4, -H, -TM], probs = [142, 114, 183].
    //  DC_PRED path: bit0=0 (prob 142).
    //
    // We first walk every MB to write its mode info AND simultaneously
    // compute + reconstruct its blocks (so DC_PRED neighbours are
    // bit-exact for the NEXT MB). We stash the quantised coefficients
    // per MB and emit them as a separate token partition after the
    // first partition (matching the decoder which uses a fresh
    // BoolDecoder for the token stream).
    let mut mb_encoded: Vec<MbEncoded> = Vec::with_capacity(mb_w * mb_h);
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // Y mode (DC_PRED).
            hdr_enc.write_bool(145, true);
            hdr_enc.write_bool(156, false);
            hdr_enc.write_bool(163, false);
            // UV mode (DC_PRED).
            hdr_enc.write_bool(142, false);

            let mb_rec = encode_intra_mb_dc(
                &src_y, &src_u, &src_v, &mut rec_y, &mut rec_u, &mut rec_v,
                y_stride, uv_stride, y_buf_h, uv_buf_h,
                mb_x, mb_y, mb_w, mb_h, &q,
            );
            mb_encoded.push(mb_rec);
        }
    }

    let first_partition = hdr_enc.finish();

    // --- Token partition (separate BoolEncoder) ---
    let mut tok_enc = BoolEncoder::new();
    let mut nz_y_above = vec![[0u8; 4]; mb_w];
    let mut nz_uv_above = vec![[0u8; 2]; mb_w];
    let mut nz_v_above = vec![[0u8; 2]; mb_w];
    let mut nz_y2_above = vec![0u8; mb_w];
    let coef_probs: &CoeffProbs = &DEFAULT_COEF_PROBS;

    for mb_y in 0..mb_h {
        let mut nz_y_left = [0u8; 4];
        let mut nz_u_left = [0u8; 2];
        let mut nz_v_left = [0u8; 2];
        let mut nz_y2_left = 0u8;
        for mb_x in 0..mb_w {
            let mb_rec = &mb_encoded[mb_y * mb_w + mb_x];
            // Y2 DC block.
            let nctx = nz_y2_above[mb_x] + nz_y2_left;
            let nz = encode_block(
                &mut tok_enc,
                coef_probs,
                /*plane=*/ 1,
                nctx as usize,
                &mb_rec.y2_coeffs,
                0,
            );
            let nzf = if nz > 0 { 1 } else { 0 };
            nz_y2_above[mb_x] = nzf;
            nz_y2_left = nzf;

            for by in 0..4 {
                for bx in 0..4 {
                    let idx = by * 4 + bx;
                    let nctx = nz_y_above[mb_x][bx] + nz_y_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        0,
                        nctx as usize,
                        &mb_rec.y_coeffs[idx],
                        1,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_y_above[mb_x][bx] = nzf;
                    nz_y_left[by] = nzf;
                }
            }
            // U and V are interleaved per (by, bx) position — the decoder
            // reads U then V for each sub-block, not all U then all V.
            for by in 0..2 {
                for bx in 0..2 {
                    let idx = by * 2 + bx;
                    let nctx = nz_uv_above[mb_x][bx] + nz_u_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.u_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_uv_above[mb_x][bx] = nzf;
                    nz_u_left[by] = nzf;
                    let nctx = nz_v_above[mb_x][bx] + nz_v_left[by];
                    let nz = encode_block(
                        &mut tok_enc,
                        coef_probs,
                        2,
                        nctx as usize,
                        &mb_rec.v_coeffs[idx],
                        0,
                    );
                    let nzf = if nz > 0 { 1 } else { 0 };
                    nz_v_above[mb_x][bx] = nzf;
                    nz_v_left[by] = nzf;
                }
            }
        }
    }
    let token_partition = tok_enc.finish();

    // --- 3-byte frame tag ---
    // frame_type=0 (I), version=0, show_frame=1, first_partition_size.
    let part_size = first_partition.len() as u32;
    if part_size >= (1 << 19) {
        return Err(Error::invalid(format!(
            "vp8 encoder: first partition too large ({} bytes)",
            part_size
        )));
    }
    // frame_type=0 (bit 0), version=0 (bits 1..3 all zero), show_frame=1 (bit 4),
    // first_partition_size in bits 5..23.
    let tag_word: u32 = (1u32 << 4) | (part_size << 5);
    let mut out = Vec::with_capacity(10 + first_partition.len() + token_partition.len());
    out.push((tag_word & 0xff) as u8);
    out.push(((tag_word >> 8) & 0xff) as u8);
    out.push(((tag_word >> 16) & 0xff) as u8);

    // --- Keyframe 7-byte header: sync + w/h words ---
    out.extend_from_slice(&KEYFRAME_SYNC_CODE);
    let w = width as u16 & 0x3fff;
    let h = height as u16 & 0x3fff;
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());

    // First partition (header + mode info) then the single token partition.
    // Since log2_nb_partitions=0 → nb_parts=1, no partition size table.
    out.extend_from_slice(&first_partition);
    out.extend_from_slice(&token_partition);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Macroblock encode (DC_PRED only)
// ---------------------------------------------------------------------------

struct QuantCtx {
    /// Not used by the encoder directly (the decoder ignores Y-block DC
    /// for non-B_PRED MBs and uses the Y2-derived DC instead) — kept for
    /// documentation / future use by a B_PRED path.
    #[allow(dead_code)]
    y_dc: i32,
    y_ac: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

/// Output of per-MB encode: quantised coefficients for each block and the
/// Y2 block (the 16 DC coefficients passed through forward WHT).
struct MbEncoded {
    y2_coeffs: [i16; 16],
    y_coeffs: [[i16; 16]; 16],
    u_coeffs: [[i16; 16]; 4],
    v_coeffs: [[i16; 16]; 4],
}

#[allow(clippy::too_many_arguments)]
fn encode_intra_mb_dc(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    rec_y: &mut [u8],
    rec_u: &mut [u8],
    rec_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    _y_buf_h: usize,
    _uv_buf_h: usize,
    mb_x: usize,
    mb_y: usize,
    _mb_w: usize,
    _mb_h: usize,
    q: &QuantCtx,
) -> MbEncoded {
    let mb_xp = mb_x * 16;
    let mb_yp = mb_y * 16;

    // Gather DC_PRED neighbours for the 16x16 luma prediction.
    let mut above_arr = [0u8; 16];
    let mut left_arr = [0u8; 16];
    let above_avail = mb_yp > 0;
    let left_avail = mb_xp > 0;
    if above_avail {
        for i in 0..16 {
            above_arr[i] = rec_y[(mb_yp - 1) * y_stride + mb_xp + i];
        }
    }
    if left_avail {
        for j in 0..16 {
            left_arr[j] = rec_y[(mb_yp + j) * y_stride + mb_xp - 1];
        }
    }
    let tl = if above_avail && left_avail {
        Some(rec_y[(mb_yp - 1) * y_stride + mb_xp - 1])
    } else if above_avail {
        Some(127)
    } else if left_avail {
        Some(129)
    } else {
        None
    };
    let mut pred = vec![0u8; 16 * 16];
    predict_16x16(
        DC_PRED,
        if above_avail { Some(&above_arr) } else { None },
        if left_avail { Some(&left_arr) } else { None },
        tl,
        &mut pred,
        16,
    );

    // Compute 4×4 residuals and apply fdct.
    // We'll produce:
    //   raw_dc[16] = DC coefficient (position 0) of each 4x4 block, pre-quant
    //   raw_ac[16][15] = AC coefficients, pre-quant
    // Then forward-WHT over the 16 DCs to produce the Y2 block, quantise
    // that, inverse-WHT to get reconstructed DCs, combine with quantised AC
    // for the final per-sub-block dequantised transform + residual apply.

    let mut raw_dc_y = [0i32; 16];
    let mut raw_ac_y = [[0i32; 16]; 16]; // index [block][pos] — pos 0 is the DC slot (unused: Y2 carries DC).
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut blk = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let src = src_y[(mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c] as i32;
                let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                blk[r * 4 + c] = src - p;
            }
        }
        let coeffs = fdct4x4(&blk);
        raw_dc_y[bi] = coeffs[0];
        raw_ac_y[bi] = coeffs;
    }

    // Forward WHT on the 16 DC values.
    let y2_raw = fwht4x4(&raw_dc_y);
    // Quantise Y2 (DC step = y2_dc, AC step = y2_ac).
    let mut y2_q = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_q[i] = quant(y2_raw[i], step);
    }
    // Dequantise + inverse WHT → reconstructed DCs.
    let mut y2_deq = [0i16; 16];
    for i in 0..16 {
        let step = if i == 0 { q.y2_dc } else { q.y2_ac };
        y2_deq[i] = (y2_q[i] as i32 * step) as i16;
    }
    let rec_dc = iwht4x4(&y2_deq);

    // Quantise AC of every 4x4 block (position >= 1 with y_ac step).
    let mut y_q = [[0i16; 16]; 16];
    for bi in 0..16 {
        // AC positions only (1..16).
        for k in 1..16 {
            y_q[bi][k] = quant(raw_ac_y[bi][k], q.y_ac);
        }
        // DC position 0 is ignored in the token stream (skipped via start=1).
        y_q[bi][0] = 0;
    }

    // Reconstruct each 4x4: deq = [rec_dc[bi], y_q[bi][1]*y_ac, ...]; res = idct(deq); out = pred + res.
    for bi in 0..16 {
        let by = bi / 4;
        let bx = bi % 4;
        let mut deq = [0i16; 16];
        deq[0] = rec_dc[bi];
        for k in 1..16 {
            deq[k] = (y_q[bi][k] as i32 * q.y_ac) as i16;
        }
        let res = idct4x4(&deq);
        for r in 0..4 {
            for c in 0..4 {
                let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                let rr = res[r * 4 + c] as i32;
                let dst_y_idx = (mb_yp + by * 4 + r) * y_stride + mb_xp + bx * 4 + c;
                rec_y[dst_y_idx] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }

    // --- Chroma (8x8 DC_PRED) ---
    let mut u_q = [[0i16; 16]; 4];
    let mut v_q = [[0i16; 16]; 4];
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for plane_sel in 0..2 {
        let (src, rec, q_coeffs) = match plane_sel {
            0 => (src_u, &mut *rec_u, &mut u_q),
            _ => (src_v, &mut *rec_v, &mut v_q),
        };
        let above_avail_c = mb_yc > 0;
        let left_avail_c = mb_xc > 0;
        let mut above = [0u8; 8];
        let mut left = [0u8; 8];
        if above_avail_c {
            for i in 0..8 {
                above[i] = rec[(mb_yc - 1) * uv_stride + mb_xc + i];
            }
        }
        if left_avail_c {
            for j in 0..8 {
                left[j] = rec[(mb_yc + j) * uv_stride + mb_xc - 1];
            }
        }
        let tl = if above_avail_c && left_avail_c {
            Some(rec[(mb_yc - 1) * uv_stride + mb_xc - 1])
        } else if above_avail_c {
            Some(127)
        } else if left_avail_c {
            Some(129)
        } else {
            None
        };
        let mut pred_uv = vec![0u8; 8 * 8];
        predict_8x8(
            DC_PRED,
            if above_avail_c { Some(&above) } else { None },
            if left_avail_c { Some(&left) } else { None },
            tl,
            &mut pred_uv,
            8,
        );
        for bi in 0..4 {
            let by = bi / 2;
            let bx = bi % 2;
            let mut blk = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    let sidx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    let s = src[sidx] as i32;
                    let p = pred_uv[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    blk[r * 4 + c] = s - p;
                }
            }
            let coeffs = fdct4x4(&blk);
            let mut blk_q = [0i16; 16];
            blk_q[0] = quant(coeffs[0], q.uv_dc);
            for k in 1..16 {
                blk_q[k] = quant(coeffs[k], q.uv_ac);
            }
            q_coeffs[bi] = blk_q;
            // Reconstruct.
            let mut deq = [0i16; 16];
            deq[0] = (blk_q[0] as i32 * q.uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (blk_q[k] as i32 * q.uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let pidx = (by * 4 + r) * 8 + bx * 4 + c;
                    let p = pred_uv[pidx] as i32;
                    let rr = res[r * 4 + c] as i32;
                    let didx = (mb_yc + by * 4 + r) * uv_stride + mb_xc + bx * 4 + c;
                    rec[didx] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    MbEncoded {
        y2_coeffs: y2_q,
        y_coeffs: y_q,
        u_coeffs: u_q,
        v_coeffs: v_q,
    }
}

/// Quantise a single coefficient using `step`. Uses symmetric rounding
/// towards zero with `step/2` bias — close to what libvpx's reference
/// encoder does for intra blocks and adequate for our decoder's
/// multiply-by-step dequantiser.
#[inline]
fn quant(v: i32, step: i32) -> i16 {
    if step <= 0 {
        return 0;
    }
    let half = step / 2;
    let q = if v >= 0 {
        (v + half) / step
    } else {
        -((-v + half) / step)
    };
    q.clamp(-2048, 2047) as i16
}

// ---------------------------------------------------------------------------
// Token (coefficient) entropy encode
// ---------------------------------------------------------------------------

/// Encode one transform block's 16 coefficients into the boolean coder.
/// Returns the number of coefficients encoded (last-non-zero + 1), or 0
/// if the entire block is zero starting from `start`.
///
/// Mirrors [`crate::tokens::decode_block`] bit-for-bit — the decoder's
/// flat `p[0..10]` look-up table is the authoritative tree walk, so the
/// write side uses the exact same branch structure.
fn encode_block(
    enc: &mut BoolEncoder,
    probs: &CoeffProbs,
    plane: usize,
    nctx: usize,
    coeffs: &[i16; 16],
    start: usize,
) -> u8 {
    let plane_probs = &probs[plane];
    // Reorder coefficients into zigzag order to match what the decoder
    // stores at `coeffs[ZIGZAG[n]]` — but we're working with unzigzagged
    // block-local coeffs here, so when encoding we iterate in zigzag.
    // Find the last non-zero in zigzag order, starting at `start`.
    let mut last_nz = None::<usize>;
    for n in start..16 {
        let c = coeffs[ZIGZAG[n]];
        if c != 0 {
            last_nz = Some(n);
        }
    }
    let last = match last_nz {
        Some(n) => n,
        None => {
            // EOB at the very start: emit p[0]=0 (not coded) at (band, nctx=nctx).
            let p = &plane_probs[COEF_BANDS[start]][nctx];
            enc.write_bool(p[0] as u32, false);
            return 0;
        }
    };

    let mut n = start;
    let mut ctx = nctx;
    // First p-vector.
    let mut p = &plane_probs[COEF_BANDS[n]][ctx];
    // p[0] = EOB bit. We have coefficients to emit, so emit p[0]=1.
    enc.write_bool(p[0] as u32, true);

    loop {
        // Zero-run loop: emit p[1]=0 for zeros.
        while coeffs[ZIGZAG[n]] == 0 {
            enc.write_bool(p[1] as u32, false);
            n += 1;
            // Context after zero is 0.
            p = &plane_probs[COEF_BANDS[n]][0];
        }
        // Now we have a non-zero coefficient at zigzag position n.
        enc.write_bool(p[1] as u32, true);

        let raw = coeffs[ZIGZAG[n]] as i32;
        let v = raw.unsigned_abs() as i32;
        emit_magnitude(enc, p, v);
        ctx = if v == 1 { 1 } else { 2 };

        // Sign bit.
        enc.write_bool(128, raw < 0);

        n += 1;
        if n == 16 {
            return 16;
        }
        p = &plane_probs[COEF_BANDS[n]][ctx];
        if n > last {
            // Emit EOB.
            enc.write_bool(p[0] as u32, false);
            return (last + 1) as u8;
        }
        // Continue — emit p[0]=1 (not EOB).
        enc.write_bool(p[0] as u32, true);
    }
}

/// Write the magnitude of a non-zero coefficient following the coef-tree
/// branch structure in [`crate::tokens::decode_block`]. `p` is the
/// 11-element probability array for the current (band, ctx).
fn emit_magnitude(enc: &mut BoolEncoder, p: &[u8; 11], v: i32) {
    // Match the decoder's branch-by-branch ladder.
    if v == 1 {
        enc.write_bool(p[2] as u32, false);
        return;
    }
    enc.write_bool(p[2] as u32, true);
    // v >= 2
    if v <= 4 {
        enc.write_bool(p[3] as u32, false);
        if v == 2 {
            enc.write_bool(p[4] as u32, false);
        } else {
            // v == 3 or 4
            enc.write_bool(p[4] as u32, true);
            enc.write_bool(p[5] as u32, v == 4);
        }
        return;
    }
    enc.write_bool(p[3] as u32, true);
    // v >= 5
    if v <= 10 {
        enc.write_bool(p[6] as u32, false);
        if v <= 6 {
            enc.write_bool(p[7] as u32, false);
            enc.write_bool(159, v == 6);
        } else {
            enc.write_bool(p[7] as u32, true);
            // v in {7,8,9,10}. Encoding: read_bool(165) gives high bit (0 → 7/8, 1 → 9/10), then read_bool(145) picks within pair.
            let hi = if v >= 9 { 1 } else { 0 };
            enc.write_bool(165, hi == 1);
            let low = (v - 7 - 2 * hi) as u32; // 0 for 7/9, 1 for 8/10
            enc.write_bool(145, low == 1);
        }
        return;
    }
    enc.write_bool(p[6] as u32, true);
    // v >= 11 → one of CAT3..CAT6.
    // Categories: CAT3 base=11 (range 11..=18, 3 extra bits), CAT4 base=19 (4 bits),
    //             CAT5 base=35 (5 bits), CAT6 base=67 (11 bits).
    let (cat, base) = if v < 19 {
        (0, 11)
    } else if v < 35 {
        (1, 19)
    } else if v < 67 {
        (2, 35)
    } else {
        (3, 67)
    };
    // cat encoded via p[8] (bit1) and p[9 + bit1] (bit0), cat = 2*bit1 + bit0.
    let bit1 = (cat >> 1) & 1;
    let bit0 = cat & 1;
    enc.write_bool(p[8] as u32, bit1 == 1);
    enc.write_bool(p[9 + bit1] as u32, bit0 == 1);
    // Extra bits for the magnitude within this category.
    let extra_bits_tab: &[u8] = match cat {
        0 => &[173, 148, 140],
        1 => &[176, 155, 140, 135],
        2 => &[180, 157, 141, 134, 130],
        _ => &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
    };
    let extra = (v - base) as u32;
    let nbits = extra_bits_tab.len();
    for i in 0..nbits {
        let bit = ((extra >> (nbits - 1 - i)) & 1) as u8;
        enc.write_bool(extra_bits_tab[i] as u32, bit != 0);
    }
}

/// Emit "no probability updates" for the whole 4×8×3×11 coefficient prob
/// table. Mirrors the decoder's `update_coef_probs` but always sending
/// `read_bool(upd)=false`.
fn emit_no_coef_prob_updates(enc: &mut BoolEncoder) {
    use crate::tables::coeff_probs::COEF_UPDATE_PROBS;
    use crate::tables::token_tree::{NUM_BANDS, NUM_CTX, NUM_PROBS, NUM_TYPES};
    for i in 0..NUM_TYPES {
        for j in 0..NUM_BANDS {
            for k in 0..NUM_CTX {
                for l in 0..NUM_PROBS {
                    let upd = COEF_UPDATE_PROBS[i][j][k][l] as u32;
                    enc.write_bool(upd, false);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bool_decoder::BoolDecoder;
    use crate::tables::coeff_probs::DEFAULT_COEF_PROBS;
    use crate::tokens::{decode_block, BlockType};

    fn roundtrip_one_block(coeffs: &[i16; 16], plane: usize, nctx: u8, start: usize) {
        let mut enc = BoolEncoder::new();
        let _nz_enc = encode_block(&mut enc, &DEFAULT_COEF_PROBS, plane, nctx as usize, coeffs, start);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let bt = match plane {
            0 => BlockType::YAfterY2,
            1 => BlockType::Y2,
            2 => BlockType::UV,
            _ => BlockType::YNoY2,
        };
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, bt, nctx, &mut out, start);
        // Only compare positions from `start` onwards — positions below
        // `start` are reserved for (skipped) DC coefficients and the
        // decoder zeros them.
        for i in start..16 {
            let zz_idx = crate::tables::token_tree::ZIGZAG[i];
            assert_eq!(
                out[zz_idx], coeffs[zz_idx],
                "coeff at zigzag pos {i} (raw {zz_idx}) mismatch: in={} out={}",
                coeffs[zz_idx], out[zz_idx]
            );
        }
    }

    #[test]
    fn block_roundtrip_all_zero() {
        let coeffs = [0i16; 16];
        roundtrip_one_block(&coeffs, 0, 0, 1);
        roundtrip_one_block(&coeffs, 1, 0, 0);
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_dc_only() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        roundtrip_one_block(&coeffs, 1, 0, 0);
        coeffs[0] = -3;
        roundtrip_one_block(&coeffs, 2, 0, 0);
    }

    #[test]
    fn block_roundtrip_small_values() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;
        coeffs[1] = 2;
        coeffs[4] = 3;  // zigzag index 2 → (1,0); etc.
        coeffs[8] = -1;
        roundtrip_one_block(&coeffs, 0, 0, 1);
    }

    #[test]
    fn block_roundtrip_y2_negative_dc() {
        // Specific case observed from the single-MB uniform test.
        let mut coeffs = [0i16; 16];
        coeffs[0] = -19;
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }

    #[test]
    fn block_roundtrip_y2_sparse_ctx1() {
        // Case observed from multi-MB gradient test (mb 1,0 Y2).
        let coeffs: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        roundtrip_one_block(&coeffs, 1, 1, 0);
    }

    #[test]
    fn block_roundtrip_y_with_dc_zeroed() {
        // When encoding Y blocks (plane=0, start=1), the decoder zeros out
        // coeffs[0]. Double-check that our decoder interprets coefficients
        // correctly when many blocks are written back-to-back.
        let mut y_blks = [[0i16; 16]; 16];
        // Populate with small AC values.
        for bi in 0..16 {
            y_blks[bi][1] = -3;
            y_blks[bi][4] = -2;
        }
        let mut enc = BoolEncoder::new();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let nz = encode_block(&mut enc, &DEFAULT_COEF_PROBS, 0, nctx as usize, &y_blks[idx], 1);
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
            }
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut nz_left = [0u8; 4];
        let mut nz_above = [0u8; 4];
        for by in 0..4 {
            for bx in 0..4 {
                let idx = by * 4 + bx;
                let nctx = nz_above[bx] + nz_left[by];
                let mut out = [0i16; 16];
                let nz = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::YAfterY2, nctx, &mut out, 1);
                let nzf = if nz > 0 { 1 } else { 0 };
                nz_above[bx] = nzf;
                nz_left[by] = nzf;
                for k in 1..16 {
                    let zz = crate::tables::token_tree::ZIGZAG[k];
                    assert_eq!(out[zz], y_blks[idx][zz], "block {idx} zz pos {k}");
                }
            }
        }
    }

    #[test]
    fn block_roundtrip_two_y2_back_to_back() {
        // Mirror the multi-MB Y2 encode → decode chain.
        let a: [i16; 16] = [-79, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let b: [i16; 16] = [7, -3, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0];
        let mut enc = BoolEncoder::new();
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 0, &a, 0);
        // Both blocks nonzero → ctx for the second is 1.
        encode_block(&mut enc, &DEFAULT_COEF_PROBS, 1, 1, &b, 0);
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf).unwrap();
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 0, &mut out, 0);
        assert_eq!(out, a);
        let mut out = [0i16; 16];
        let _ = decode_block(&mut dec, &DEFAULT_COEF_PROBS, BlockType::Y2, 1, &mut out, 0);
        assert_eq!(out, b);
    }

    #[test]
    fn block_roundtrip_category_magnitudes() {
        let mut coeffs = [0i16; 16];
        // Put values of varying category in a variety of zigzag positions.
        coeffs[0] = 6;    // cat ~ in-range of decoder path >=5
        coeffs[1] = 9;    // category within the 7..=10 range
        coeffs[2] = 15;   // CAT3
        coeffs[3] = 25;   // CAT4
        coeffs[4] = -50;  // CAT5
        coeffs[5] = 100;  // CAT6
        roundtrip_one_block(&coeffs, 1, 0, 0);
    }
}

/// Copy the 3 planes of a video frame into MB-aligned (16/8 pixel) buffers.
/// Edge-replicate when frame dimensions are not multiples of 16.
fn extract_mb_padded(
    v: &VideoFrame,
    mb_w: usize,
    mb_h: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let width = v.width as usize;
    let height = v.height as usize;
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_h = mb_h * 16;
    let uv_h = mb_h * 8;

    let y_plane = &v.planes[0];
    let u_plane = &v.planes[1];
    let v_plane = &v.planes[2];

    let mut y_out = vec![0u8; y_stride * y_h];
    for j in 0..y_h {
        let src_row = j.min(height - 1);
        let src_start = src_row * y_plane.stride;
        for i in 0..y_stride {
            let src_col = i.min(width - 1);
            y_out[j * y_stride + i] = y_plane.data[src_start + src_col];
        }
    }
    let uv_w = (width + 1) / 2;
    let uv_src_h = (height + 1) / 2;
    let mut u_out = vec![0u8; uv_stride * uv_h];
    let mut v_out = vec![0u8; uv_stride * uv_h];
    for j in 0..uv_h {
        let src_row = j.min(uv_src_h - 1);
        let u_start = src_row * u_plane.stride;
        let v_start = src_row * v_plane.stride;
        for i in 0..uv_stride {
            let src_col = i.min(uv_w - 1);
            u_out[j * uv_stride + i] = u_plane.data[u_start + src_col];
            v_out[j * uv_stride + i] = v_plane.data[v_start + src_col];
        }
    }
    Ok((y_out, u_out, v_out))
}
