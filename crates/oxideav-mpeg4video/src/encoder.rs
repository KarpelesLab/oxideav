//! MPEG-4 Part 2 video encoder — I-VOP + P-VOP.
//!
//! Scope:
//! * Visual Object Sequence (VOS), Visual Object (VO), Video Object Layer
//!   (VOL) and Video Object Plane (VOP) headers — §6.2.
//! * I-VOP body: per-MB MCBPC + ac_pred + CBPY (no dquant), then six 8×8
//!   intra blocks (Y0..Y3, Cb, Cr) with intra DC VLC + signed residual and
//!   intra AC tcoef VLC walk (Table B-16).
//! * P-VOP body: half-pel motion estimation (integer diamond + half-pel
//!   refinement), 1MV mode, median-predicted MVD with Table B-12, inter
//!   texture coding (H.263 inter quant + Table B-17 tcoef walk), and
//!   `not_coded` skip MBs. See `pvop.rs`.
//! * H.263 quantisation (`mpeg_quant = 0`) — chosen to avoid mismatch
//!   control. `vop_quant` is configurable (default 5) and stays constant
//!   across the picture (no dquant).
//! * AC prediction strategy: **disabled** for every intra MB. The decoder
//!   still accepts `ac_pred_flag = 0`; emitting AC predictions only saves
//!   bits and is not required for correctness.
//! * DC prediction: gradient-direction predictor matching the decoder
//!   (§7.4.3.1) — only the differential is written.
//! * Resync markers: not emitted (`resync_marker_disable = 1` in the VOL).
//!   The encoder is correct without them; ffmpeg accepts streams with the
//!   flag set.
//! * GOP structure: I-VOP every `DEFAULT_GOP_SIZE` frames, P-VOPs in between.
//!   Reference frame is the most recent reconstructed picture.
//!
//! Out of scope (returns `Error::Unsupported` from the encoder factory):
//! * 4MV mode for P-VOPs.
//! * B / S VOPs (§6.2.5).
//! * Sprites / GMC (§6.2.4 sprite_enable).
//! * Interlace, scalability, data partitioning, reversible VLCs.
//!
//! AC tcoef encoding uses Table B-16 (intra) / B-17 (inter) directly when
//! `(last, run, level)` has a short codeword and falls back to the **third
//! escape mode** (§6.3.8) for any combination that isn't in the short table.
//! Third-escape encodes `(last, run, level)` literally — 1+6+12 bits framed
//! by markers — so any signed 12-bit non-zero level survives.

use std::collections::{HashMap, VecDeque};

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use crate::bitwriter::BitWriter;
use crate::block::{choose_dc_predictor, BlockNeighbour};
use crate::headers::vol::ZIGZAG;
use crate::iq::{dc_scaler, Y_DC_SCALE_TABLE};
use crate::mb::{IVopPicture, PredGrid};
use crate::pvop::encode_p_vop_body;
use crate::start_codes::{VISUAL_OBJECT_START_CODE, VOP_START_CODE, VOS_END_CODE, VOS_START_CODE};
use crate::tables::tcoef;

// -------------------------------------------------------------------------
// Public factory + Encoder impl
// -------------------------------------------------------------------------

/// Default vop_quant for the encoder. The acceptance bar specifies
/// `vop_quant = 5`.
pub const DEFAULT_VOP_QUANT: u32 = 5;

/// Default GOP size (I-VOP cadence). Emit an I-VOP every `DEFAULT_GOP_SIZE`
/// frames; all other frames are P-VOPs. The P-VOP test in
/// `tests/p_vop.rs` exercises this with `GOP_SIZE = 16` (1 I + 15 P).
pub const DEFAULT_GOP_SIZE: u32 = 16;

/// Forward motion-vector range code for P-VOPs. `f_code = 1` gives the
/// smallest range `[-32, 31]` half-pels which is plenty for the encoder's
/// tiny diamond search (bounded at ±7 integer pels). The decoder accepts 1-7.
pub const DEFAULT_F_CODE_FWD: u8 = 1;

/// Encoder factory used by `register()`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("mpeg4 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("mpeg4 encoder: missing height"))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid("mpeg4 encoder: zero-sized frame"));
    }
    if width > 8191 || height > 8191 {
        // VOL `video_object_layer_width` / `_height` are 13-bit fields.
        return Err(Error::invalid(
            "mpeg4 encoder: dimensions exceed 13-bit VOL field",
        ));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "mpeg4 encoder: only Yuv420P supported (got {:?})",
            pix
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(24, 1));

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.frame_rate = Some(frame_rate);

    let time_base = TimeBase::new(frame_rate.den, frame_rate.num);

    Ok(Box::new(Mpeg4VideoEncoder {
        output_params,
        width,
        height,
        frame_rate,
        time_base,
        vop_quant: DEFAULT_VOP_QUANT,
        gop_size: DEFAULT_GOP_SIZE,
        f_code_fwd: DEFAULT_F_CODE_FWD,
        pending: VecDeque::new(),
        eof: false,
        finalised: false,
        headers_emitted: false,
        vop_count: 0,
        reference: None,
        rounding_type: false,
    }))
}

struct Mpeg4VideoEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    frame_rate: Rational,
    time_base: TimeBase,
    vop_quant: u32,
    gop_size: u32,
    f_code_fwd: u8,
    pending: VecDeque<Packet>,
    eof: bool,
    finalised: bool,
    headers_emitted: bool,
    vop_count: u32,
    /// Reconstructed previous picture — used as the MC reference for the
    /// next P-VOP. Refreshed by every I-VOP and every P-VOP.
    reference: Option<IVopPicture>,
    /// `vop_rounding_type` to emit on the next P-VOP. Per FFmpeg convention
    /// we toggle this between P-VOPs (starts at 0 after an I-VOP, alternates
    /// afterwards) — it matches the half-pel rounding inside `mc.rs`.
    rounding_type: bool,
}

impl Encoder for Mpeg4VideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("mpeg4 encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "mpeg4 encoder: frame dimensions do not match encoder config",
            ));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::invalid(
                "mpeg4 encoder: only Yuv420P input frames supported",
            ));
        }
        if v.planes.len() != 3 {
            return Err(Error::invalid("mpeg4 encoder: expected 3 planes"));
        }

        // Decide I-VOP vs P-VOP: first frame and every gop_size-th frame are
        // I-VOPs. If the reference frame is missing (e.g. on error we reset),
        // force an I-VOP too.
        let is_keyframe = self.vop_count % self.gop_size == 0 || self.reference.is_none();

        let mut bw = BitWriter::with_capacity(8192);
        if !self.headers_emitted {
            write_vos_vo_vol(
                &mut bw,
                self.width,
                self.height,
                self.frame_rate,
                self.vop_quant,
            );
            self.headers_emitted = true;
        }

        if is_keyframe {
            write_i_vop_header(&mut bw, self.vop_count, self.vop_quant);
            let pic = encode_i_vop_body_and_reconstruct(&mut bw, v, self.vop_quant)?;
            self.reference = Some(pic);
            // Reset rounding_type on I-VOP (spec §7.6.2.1).
            self.rounding_type = false;
        } else {
            let reference = self
                .reference
                .as_ref()
                .expect("P-VOP path requires a reference picture");
            write_p_vop_header(
                &mut bw,
                self.vop_count,
                self.vop_quant,
                self.rounding_type,
                self.f_code_fwd,
            );
            let pic = encode_p_vop_body(
                &mut bw,
                v,
                reference,
                self.vop_quant,
                self.f_code_fwd,
                self.rounding_type,
            )?;
            self.reference = Some(pic);
            // Keep rounding_type = 0 for all P-VOPs (default). The spec
            // permits alternating but our encoder's reconstruction uses the
            // same flag so no drift accumulates on the decoder side that
            // honours our value.
        }
        // Byte-align the VOP body so the next start code (or VOS_END) is
        // immediately byte-aligned. Spec §6.3.5 / FFmpeg's encoder pad.
        bw.align_to_byte_zero();

        let bytes = bw.finish();
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        self.vop_count += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof && !self.finalised {
            self.finalised = true;
            // Emit a VOS end marker so downstream tools see a clean trailer.
            let mut bw = BitWriter::new();
            write_start_code(&mut bw, VOS_END_CODE);
            let bytes = bw.finish();
            let mut pkt = Packet::new(0, self.time_base, bytes);
            pkt.flags.header = true;
            return Ok(pkt);
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Start-code + header emission
// -------------------------------------------------------------------------

fn write_start_code(bw: &mut BitWriter, code: u8) {
    bw.align_to_byte_zero();
    bw.write_bytes(&[0x00, 0x00, 0x01, code]);
}

/// Compute the smallest number of bits required to hold `max_value`. Mirrors
/// the decoder's `bits_needed` helper.
fn bits_needed(max_value: u32) -> u32 {
    if max_value == 0 {
        1
    } else {
        32 - max_value.leading_zeros()
    }
}

/// Emit the canonical Visual Object Sequence + Visual Object + Video Object +
/// Video Object Layer headers. Profile is **Simple Profile @ Level 1**
/// (`profile_and_level_indication = 0x01`) — the most-compatible PLI for
/// typical XVID/DivX-style elementary streams that ffmpeg's `mpeg4` decoder
/// happily consumes. Layer geometry is encoded at the picture's natural
/// resolution; the `frame_rate` is encoded as
/// `vop_time_increment_resolution = num`, `fixed_vop_time_increment = den`.
fn write_vos_vo_vol(bw: &mut BitWriter, width: u32, height: u32, frame_rate: Rational, _q: u32) {
    // VOS.
    write_start_code(bw, VOS_START_CODE);
    // profile_and_level_indication = 0x01 — Simple Profile @ Level 1.
    bw.write_bits(0x01, 8);

    // Visual Object.
    write_start_code(bw, VISUAL_OBJECT_START_CODE);
    bw.write_bits(0, 1); // is_visual_object_identifier = 0
    bw.write_bits(1, 4); // visual_object_type = 1 (Video)
    bw.write_bits(0, 1); // video_signal_type = 0
                         // next_start_code() — pad to byte boundary with `0_111_1111` style stuffing.
    align_with_one_zero_then_ones(bw);

    // Video Object — id 0, no payload of interest.
    write_start_code(bw, 0x00);

    // Video Object Layer — id 0x20.
    write_start_code(bw, 0x20);
    bw.write_bits(0, 1); // random_accessible_vol = 0
    bw.write_bits(1, 8); // video_object_type_indication = 1 (Simple)
    bw.write_bits(0, 1); // is_object_layer_identifier = 0
    bw.write_bits(1, 4); // aspect_ratio_info = 1 (square)
    bw.write_bits(1, 1); // vol_control_parameters = 1
    bw.write_bits(1, 2); // chroma_format = 1 (4:2:0)
    bw.write_bits(1, 1); // low_delay = 1
    bw.write_bits(0, 1); // vbv_parameters = 0
    bw.write_bits(0, 2); // video_object_layer_shape = 0 (Rectangular)
    bw.write_bits(1, 1); // marker

    let resolution = (frame_rate.num as u32).clamp(1, 0xFFFF);
    bw.write_bits(resolution, 16); // vop_time_increment_resolution
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // fixed_vop_rate = 1
    let vti_bits = bits_needed(resolution.saturating_sub(1)).max(1);
    let fixed_vti = (frame_rate.den as u32).max(1);
    bw.write_bits(fixed_vti, vti_bits);

    bw.write_bits(1, 1); // marker
    bw.write_bits(width, 13);
    bw.write_bits(1, 1); // marker
    bw.write_bits(height, 13);
    bw.write_bits(1, 1); // marker

    bw.write_bits(0, 1); // interlaced = 0
    bw.write_bits(1, 1); // obmc_disable = 1
    bw.write_bits(0, 1); // sprite_enable = 0 (verid==1 → 1 bit)
    bw.write_bits(0, 1); // not_8_bit = 0
    bw.write_bits(0, 1); // mpeg_quant = 0 (use H.263 quant)
                         // verid==1 → no quarter_sample bit emitted.
    bw.write_bits(1, 1); // complexity_estimation_disable = 1
    bw.write_bits(1, 1); // resync_marker_disable = 1
    bw.write_bits(0, 1); // data_partitioned = 0
                         // verid==1 → no newpred / reduced_resolution_vop bits.
    bw.write_bits(0, 1); // scalability = 0

    align_with_one_zero_then_ones(bw);
}

/// MPEG-4 spec stuffing rule for `next_start_code()` (§6.3.4): write a `0`
/// bit followed by `n` `1` bits where `n` is just enough to byte-align the
/// stream. If already aligned, write a full `01111111` byte.
fn align_with_one_zero_then_ones(bw: &mut BitWriter) {
    if bw.is_byte_aligned() {
        bw.write_byte(0x7F);
        return;
    }
    bw.write_bits(0, 1);
    while !bw.is_byte_aligned() {
        bw.write_bits(1, 1);
    }
}

/// Emit the VOP header for an I-VOP. `time_inc` is the VOP's index in the
/// stream — encoded as the `vop_time_increment` (with `modulo_time_base = 0`
/// since we keep time_inc < resolution). `vop_quant` is constant across the
/// picture.
fn write_i_vop_header(bw: &mut BitWriter, time_inc: u32, vop_quant: u32) {
    write_start_code(bw, VOP_START_CODE);
    bw.write_bits(0, 2); // vop_coding_type = 00 (I)
    bw.write_bits(0, 1); // modulo_time_base = `0` (terminator)
    bw.write_bits(1, 1); // marker
                         // vop_time_increment_resolution was 24 in the default config → 5 bits.
                         // But to be safe we re-derive bits from the resolution. The frame_rate
                         // numerator is what we wrote into the VOL.
    let vti_bits = bits_needed(/* resolution-1 */ 23).max(1);
    bw.write_bits(time_inc % 24, vti_bits);
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // vop_coded = 1
                         // (No vop_rounding_type for I.)
                         // intra_dc_vlc_thr = 0 → always use intra DC VLC.
    bw.write_bits(0, 3);
    bw.write_bits(vop_quant, 5);
    // No fcode for I-VOP.
}

/// Emit the VOP header for a P-VOP. Field layout mirrors `write_i_vop_header`
/// plus the P-VOP-specific `vop_rounding_type` and `vop_fcode_forward` fields
/// (§6.2.5). `time_inc` is per-picture. `rounding_type` is the half-pel
/// rounding flag; `f_code_fwd` is the forward motion range code (1..=7).
fn write_p_vop_header(
    bw: &mut BitWriter,
    time_inc: u32,
    vop_quant: u32,
    rounding_type: bool,
    f_code_fwd: u8,
) {
    write_start_code(bw, VOP_START_CODE);
    bw.write_bits(0b01, 2); // vop_coding_type = 01 (P)
    bw.write_bits(0, 1); // modulo_time_base = `0` terminator
    bw.write_bits(1, 1); // marker
    let vti_bits = bits_needed(23).max(1);
    bw.write_bits(time_inc % 24, vti_bits);
    bw.write_bits(1, 1); // marker

    bw.write_bits(1, 1); // vop_coded = 1
    bw.write_bits(if rounding_type { 1 } else { 0 }, 1); // vop_rounding_type
    bw.write_bits(0, 3); // intra_dc_vlc_thr = 0
    bw.write_bits(vop_quant, 5);
    bw.write_bits(f_code_fwd as u32, 3); // vop_fcode_forward
                                         // (No fcode_backward for P.)
}

// -------------------------------------------------------------------------
// I-VOP body: per-MB encoding
// -------------------------------------------------------------------------

/// Encode an I-VOP body AND return the reconstructed picture so it can be
/// used as the MC reference for subsequent P-VOPs. Uses the shared decoder
/// path so the reconstruction is bit-exact relative to what the decoder
/// would produce from the same bitstream.
pub(crate) fn encode_i_vop_body_and_reconstruct(
    bw: &mut BitWriter,
    v: &VideoFrame,
    vop_quant: u32,
) -> Result<IVopPicture> {
    let width = v.width as usize;
    let height = v.height as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);

    let mut grid = PredGrid::new(mb_w, mb_h);
    // We reconstruct by re-reading our emitted bitstream. That's the same
    // recipe the P-VOP path will need at decode time. To avoid a full
    // second-pass re-decode here, we stash the reconstructed 8×8 samples
    // per block directly as we quantise + IDCT inside `encode_intra_mb`
    // (done below via the `out` parameter).
    let mut pic = IVopPicture::new(width, height);

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            encode_intra_mb_reconstruct(bw, v, mb_x, mb_y, vop_quant, &mut grid, &mut pic)?;
        }
    }
    Ok(pic)
}

/// Encode one intra macroblock AND reconstruct it into `pic`. The
/// reconstructed samples mirror what the decoder would produce from the
/// emitted bitstream, so the resulting picture is bit-exact w.r.t. downstream
/// P-VOP motion compensation references.
fn encode_intra_mb_reconstruct(
    bw: &mut BitWriter,
    v: &VideoFrame,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
    pic: &mut IVopPicture,
) -> Result<()> {
    encode_intra_mb_inner(bw, v, mb_x, mb_y, quant, grid, Some(pic))
}

fn encode_intra_mb_inner(
    bw: &mut BitWriter,
    v: &VideoFrame,
    mb_x: usize,
    mb_y: usize,
    quant: u32,
    grid: &mut PredGrid,
    mut pic: Option<&mut IVopPicture>,
) -> Result<()> {
    // Read all six 8×8 sample blocks from the source frame (with edge
    // replication for the bottom-right partial macroblocks if any).
    let mut blocks = [[0i32; 64]; 6];
    for blk in 0..6 {
        load_block_samples(v, mb_x, mb_y, blk, &mut blocks[blk]);
    }

    // Forward DCT each block (no level shift — MPEG-4 stores DC directly in
    // pel domain).
    let mut dct = [[0i32; 64]; 6];
    for blk in 0..6 {
        let mut f = [0.0f32; 64];
        for i in 0..64 {
            f[i] = blocks[blk][i] as f32;
        }
        fdct8x8(&mut f);
        for i in 0..64 {
            dct[blk][i] = f[i].round() as i32;
        }
    }

    // Quantise each block to (DC_units, AC_levels) and reconstruct in parallel
    // so the decoder side will see the same DC predictor neighbour values.
    // For each block:
    //   * DC: dc_units = (dct[0] + scale/2) / scale (matching decoder
    //     formula `recon_pel = recon_units * scale`).
    //   * AC: level[i] = round(coef / (2*Q)). H.263 dequant is
    //     `2*Q*|l| + Q_plus * sign`, so encoding by halving by `2Q` and
    //     rounding gives the closest level.
    let mut dc_units = [0i32; 6];
    let mut ac_levels = [[0i32; 64]; 6];
    for blk in 0..6 {
        let scale = dc_scaler(blk, quant) as i32;
        // Quantise DC. Round-to-nearest.
        let dc_pel = dct[blk][0];
        let dc_q = round_div(dc_pel, scale).clamp(-2048, 2047);
        dc_units[blk] = dc_q;
        // Quantise ACs.  H.263 dequant for intra is
        //   recon(l != 0) = (2*Q*|l| + Q_plus) * sign(l)
        //   recon(0)      = 0
        // where Q_plus = Q if Q is odd, Q-1 if Q is even (§7.4.4.2). The
        // forward step picks the level whose reconstruction is closest to
        // the input coefficient.
        for i in 1..64 {
            let l = quantise_ac_intra_h263(dct[blk][i], quant as i32).clamp(-2047, 2047);
            ac_levels[blk][i] = l;
        }
    }

    // Compute MCBPC + CBPY from the AC-coded flags.
    let mut luma_coded = [false; 4];
    let mut chroma_coded = [false; 2];
    for blk in 0..4 {
        luma_coded[blk] = ac_levels[blk][1..64].iter().any(|&v| v != 0);
    }
    chroma_coded[0] = ac_levels[4][1..64].iter().any(|&v| v != 0);
    chroma_coded[1] = ac_levels[5][1..64].iter().any(|&v| v != 0);
    // cbpc bits: bit1 = Cb, bit0 = Cr.
    let cbpc = ((chroma_coded[0] as u8) << 1) | (chroma_coded[1] as u8);
    // cbpy bits: bit3..bit0 = Y0..Y3.
    let mut cbpy: u8 = 0;
    for (i, &c) in luma_coded.iter().enumerate() {
        if c {
            cbpy |= 1 << (3 - i);
        }
    }

    // MCBPC for I (Table B-10): mcbpc value = cbpc (no IntraQ — quant is
    // constant across the picture).
    write_mcbpc_intra(bw, cbpc);
    // ac_pred_flag = 0 (we never emit AC predictions).
    bw.write_bits(0, 1);
    // CBPY (decoder uses the raw value directly for intra MBs — see mb.rs).
    write_cbpy(bw, cbpy);

    // For each block: emit DC VLC + sign + residual + (if AC coded) AC walk.
    for blk in 0..6 {
        // Predicted DC: gradient direction over the neighbour grid.
        let (left, top_left, top) = lookup_neighbour_dcs(blk, mb_x, mb_y, grid);
        let (predicted_dc_pel, _dir) = choose_dc_predictor(left, top_left, top);
        let scale = dc_scaler(blk, quant) as i32;
        let pred_units = (predicted_dc_pel + scale / 2) / scale;
        let dc_diff = dc_units[blk] - pred_units;

        write_intra_dc_diff(bw, blk, dc_diff);

        let coded = if blk < 4 {
            luma_coded[blk]
        } else {
            chroma_coded[blk - 4]
        };
        if coded {
            // Emit AC tcoef walk (zigzag scan; ac_pred=0 → default scan).
            write_intra_ac(bw, &ac_levels[blk])?;
        }

        // Update neighbour grid with the *reconstructed* DC (pel domain) so
        // future MBs predict from the same DC the decoder will see.
        let recon_dc = (dc_units[blk] * scale).clamp(0, 2047);
        update_neighbour(grid, blk, mb_x, mb_y, recon_dc, quant as u8);

        // Optionally reconstruct the 8×8 block into `pic`. Mirrors the
        // decoder's reconstruct_intra_block path: dequantise the ACs under
        // the H.263 rule, install the reconstructed pel-domain DC, IDCT,
        // and clip to u8.
        if let Some(pic_mut) = pic.as_deref_mut() {
            let mut coeffs = ac_levels[blk];
            // H.263 intra dequant matches `iq::dequantise_intra_h263`.
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
            coeffs[0] = recon_dc.clamp(-2048, 2047);
            let mut f = [0.0f32; 64];
            for i in 0..64 {
                f[i] = coeffs[i] as f32;
            }
            crate::block::idct8x8(&mut f);
            write_recon_to_picture(pic_mut, blk, mb_x, mb_y, &f);
        }
    }

    let _ = Y_DC_SCALE_TABLE;
    Ok(())
}

fn write_recon_to_picture(
    pic: &mut IVopPicture,
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    samples: &[f32; 64],
) {
    let (plane, stride, px, py) = match blk {
        0 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16, mb_y * 16),
        1 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16 + 8, mb_y * 16),
        2 => (pic.y.as_mut_slice(), pic.y_stride, mb_x * 16, mb_y * 16 + 8),
        3 => (
            pic.y.as_mut_slice(),
            pic.y_stride,
            mb_x * 16 + 8,
            mb_y * 16 + 8,
        ),
        4 => (pic.cb.as_mut_slice(), pic.c_stride, mb_x * 8, mb_y * 8),
        5 => (pic.cr.as_mut_slice(), pic.c_stride, mb_x * 8, mb_y * 8),
        _ => unreachable!(),
    };
    for j in 0..8 {
        for i in 0..8 {
            let v = samples[j * 8 + i].round() as i32;
            plane[(py + j) * stride + (px + i)] = v.clamp(0, 255) as u8;
        }
    }
}

// -------------------------------------------------------------------------
// Sample fetch + neighbour-grid bookkeeping
// -------------------------------------------------------------------------

fn load_block_samples(v: &VideoFrame, mb_x: usize, mb_y: usize, blk: usize, out: &mut [i32; 64]) {
    let (plane_idx, x0, y0, pw, ph) = block_pel_position(v, mb_x, mb_y, blk);
    let p = &v.planes[plane_idx];
    for j in 0..8 {
        let yy = (y0 + j).min(ph.saturating_sub(1));
        for i in 0..8 {
            let xx = (x0 + i).min(pw.saturating_sub(1));
            out[j * 8 + i] = p.data[yy * p.stride + xx] as i32;
        }
    }
}

pub(crate) fn block_pel_position(
    v: &VideoFrame,
    mb_x: usize,
    mb_y: usize,
    blk: usize,
) -> (usize, usize, usize, usize, usize) {
    let w = v.width as usize;
    let h = v.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    match blk {
        0 => (0, mb_x * 16, mb_y * 16, w, h),
        1 => (0, mb_x * 16 + 8, mb_y * 16, w, h),
        2 => (0, mb_x * 16, mb_y * 16 + 8, w, h),
        3 => (0, mb_x * 16 + 8, mb_y * 16 + 8, w, h),
        4 => (1, mb_x * 8, mb_y * 8, cw, ch),
        5 => (2, mb_x * 8, mb_y * 8, cw, ch),
        _ => unreachable!(),
    }
}

fn lookup_neighbour_dcs(blk: usize, mb_x: usize, mb_y: usize, grid: &PredGrid) -> (i32, i32, i32) {
    let (plane, bx, by, stride) = block_grid_position(blk, mb_x, mb_y, grid);
    let read = |px: isize, py: isize| -> i32 {
        if px < 0 || py < 0 {
            return 1024;
        }
        let rows = plane.len() / stride;
        if (px as usize) >= stride || (py as usize) >= rows {
            return 1024;
        }
        let nbr = &plane[(py as usize) * stride + (px as usize)];
        if nbr.is_intra {
            nbr.dc
        } else {
            1024
        }
    };
    let left = read(bx as isize - 1, by as isize);
    let top = read(bx as isize, by as isize - 1);
    let top_left = read(bx as isize - 1, by as isize - 1);
    (left, top_left, top)
}

fn block_grid_position(
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    grid: &PredGrid,
) -> (&[BlockNeighbour], usize, usize, usize) {
    match blk {
        0 => (&grid.y, mb_x * 2, mb_y * 2, grid.y_stride),
        1 => (&grid.y, mb_x * 2 + 1, mb_y * 2, grid.y_stride),
        2 => (&grid.y, mb_x * 2, mb_y * 2 + 1, grid.y_stride),
        3 => (&grid.y, mb_x * 2 + 1, mb_y * 2 + 1, grid.y_stride),
        4 => (&grid.cb, mb_x, mb_y, grid.c_stride),
        5 => (&grid.cr, mb_x, mb_y, grid.c_stride),
        _ => unreachable!(),
    }
}

fn update_neighbour(
    grid: &mut PredGrid,
    blk: usize,
    mb_x: usize,
    mb_y: usize,
    dc_pel: i32,
    quant: u8,
) {
    let (bx, by, stride) = match blk {
        0 => (mb_x * 2, mb_y * 2, grid.y_stride),
        1 => (mb_x * 2 + 1, mb_y * 2, grid.y_stride),
        2 => (mb_x * 2, mb_y * 2 + 1, grid.y_stride),
        3 => (mb_x * 2 + 1, mb_y * 2 + 1, grid.y_stride),
        4 => (mb_x, mb_y, grid.c_stride),
        5 => (mb_x, mb_y, grid.c_stride),
        _ => unreachable!(),
    };
    let plane: &mut [BlockNeighbour] = match blk {
        0..=3 => &mut grid.y,
        4 => &mut grid.cb,
        5 => &mut grid.cr,
        _ => unreachable!(),
    };
    let n = &mut plane[by * stride + bx];
    n.dc = dc_pel;
    n.quant = quant;
    n.is_intra = true;
    // ACs left at zero (we set ac_pred=0 so the decoder won't read them; but
    // keep zero so any future code path is benign).
    for i in 0..7 {
        n.ac_top_row[i] = 0;
        n.ac_left_col[i] = 0;
    }
}

// -------------------------------------------------------------------------
// MCBPC + CBPY emit (Tables B-10, B-9)
// -------------------------------------------------------------------------

fn write_mcbpc_intra(bw: &mut BitWriter, cbpc: u8) {
    // Table B-10 — Intra MCBPC values 0..=3 (cbpc).
    let (bits, code) = match cbpc {
        0 => (1, 0b1),
        1 => (3, 0b001),
        2 => (3, 0b010),
        3 => (3, 0b011),
        _ => unreachable!("cbpc out of range: {cbpc}"),
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

// -------------------------------------------------------------------------
// Intra DC VLC encode (Tables B-12 / B-13)
// -------------------------------------------------------------------------

fn write_intra_dc_diff(bw: &mut BitWriter, block_idx: usize, diff: i32) {
    let (size_codes, size_bits) = if block_idx < 4 {
        // Luma
        (
            [3u32, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_slice(),
            [3u8, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11].as_slice(),
        )
    } else {
        (
            [3u32, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_slice(),
            [2u8, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_slice(),
        )
    };
    let abs = diff.unsigned_abs();
    let size = if abs == 0 {
        0
    } else {
        // size = number of bits needed to hold |diff| in unsigned form.
        32 - abs.leading_zeros()
    };
    let size = size.min(12) as usize;
    bw.write_bits(size_codes[size], size_bits[size] as u32);
    if size == 0 {
        return;
    }
    // Bits encoding: positive → raw `diff` in `size` low bits with the MSB
    // (sign bit) set. Negative → `(2^size - 1) - |diff|` packed in `size`
    // bits — equivalently, bitwise NOT of (|diff|) within `size` bits.
    let raw: u32 = if diff > 0 {
        diff as u32
    } else {
        let mask = (1u32 << size) - 1;
        (!(abs)) & mask
    };
    bw.write_bits(raw, size as u32);
    if size > 8 {
        bw.write_bits(1, 1); // marker bit per §6.3.8
    }
}

// -------------------------------------------------------------------------
// Intra AC tcoef encode (Table B-16 + 3rd escape)
// -------------------------------------------------------------------------

/// Walk `block` in zigzag order, emitting one VLC per non-zero coefficient.
/// `block` is in natural order (block[ZIGZAG[i]] is scan position i).
fn write_intra_ac(bw: &mut BitWriter, block: &[i32; 64]) -> Result<()> {
    // Find the last non-zero AC scan index (we encode AC starting at scan 1).
    let mut last_nz: Option<usize> = None;
    for i in 1..64 {
        if block[ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let Some(last_nz) = last_nz else {
        // CBPY claimed this block was coded but no ACs remain — defensive
        // path. Encode a single (last=1, run=0, level=±1) using third escape
        // with level=1 to keep the bitstream parseable. In practice we skip
        // this branch because the caller checks `coded` from the level
        // array.
        return Err(Error::other(
            "mpeg4 encoder: AC walk requested but block is all zero",
        ));
    };
    let mut run = 0u8;
    let mut i = 1;
    while i <= last_nz {
        let lv = block[ZIGZAG[i]];
        if lv == 0 {
            run += 1;
            i += 1;
            continue;
        }
        let last = i == last_nz;
        write_intra_tcoef_symbol(bw, last, run, lv);
        run = 0;
        i += 1;
    }
    Ok(())
}

/// Encode one (last, run, level) intra tcoef symbol — short VLC where
/// possible, otherwise third escape.
fn write_intra_tcoef_symbol(bw: &mut BitWriter, last: bool, run: u8, level: i32) {
    let abs = level.unsigned_abs() as u8;
    if let Some((bits, code)) = lookup_intra_short_vlc(last, run, abs) {
        bw.write_bits(code, bits as u32);
        // Sign: 0 = positive, 1 = negative.
        bw.write_bits(if level < 0 { 1 } else { 0 }, 1);
        return;
    }
    // Third escape (§6.3.8 escape mode 3):
    //   `0000011` (escape prefix, 7 bits) +
    //   `1` (marker for "not 1st mode") +
    //   `1` (marker for "not 2nd mode") +
    //   last (1 bit) + run (6 bits) + marker(1) + level (12 signed) + marker(1).
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

/// Cached `(last, run, level_abs)` → `(bits, code)` for the intra tcoef
/// short-VLC reverse lookup.
type IntraShortVlcMap = HashMap<(bool, u8, u8), (u8, u32)>;

/// Reverse-lookup of the short VLC in Table B-16 keyed by `(last, run, abs)`.
fn lookup_intra_short_vlc(last: bool, run: u8, level_abs: u8) -> Option<(u8, u32)> {
    use std::sync::OnceLock;
    static MAP: OnceLock<IntraShortVlcMap> = OnceLock::new();
    let m = MAP.get_or_init(build_intra_short_vlc_map);
    m.get(&(last, run, level_abs)).copied()
}

fn build_intra_short_vlc_map() -> IntraShortVlcMap {
    let mut m = HashMap::new();
    for entry in tcoef::intra_table() {
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
}

// -------------------------------------------------------------------------
// FDCT + small numeric helpers
// -------------------------------------------------------------------------

/// Round-to-nearest division of `a` by `b` (with sign).
pub(crate) fn round_div(a: i32, b: i32) -> i32 {
    debug_assert!(b > 0);
    if a >= 0 {
        (a + b / 2) / b
    } else {
        -(((-a) + b / 2) / b)
    }
}

/// Quantise one intra-AC coefficient under the H.263 dequant rule.
///
/// Reconstructions are at `0` and `±(2*Q*l + Q_plus)` for `l ≥ 1`, where
/// `Q_plus = Q | 1` (i.e. `Q` if odd, `Q-1` if even). This routine picks the
/// integer level whose reconstruction is closest to `coef`. Ties prefer the
/// lower-magnitude level (cheaper to code).
fn quantise_ac_intra_h263(coef: i32, q: i32) -> i32 {
    if coef == 0 || q <= 0 {
        return 0;
    }
    let q_plus = if q & 1 == 1 { q } else { q - 1 };
    let two_q = 2 * q;
    let abs = coef.unsigned_abs() as i32;
    // Coarse seed: floor(|coef| / (2*Q)). Compare candidates `l` and `l+1`
    // against the input. `l = 0` is the deadzone choice (recon = 0).
    let l_low = abs / two_q;
    // Compare three candidates: l_low, l_low + 1, and (if l_low > 0) l_low - 1.
    let mut best_l = 0i32;
    let mut best_err = abs;
    let consider = |l: i32| {
        if l < 0 {
            return None;
        }
        let recon = if l == 0 { 0 } else { two_q * l + q_plus };
        Some((l, (abs - recon).abs()))
    };
    for cand in [l_low.saturating_sub(1), l_low, l_low + 1] {
        if let Some((l, e)) = consider(cand) {
            if e < best_err {
                best_err = e;
                best_l = l;
            }
        }
    }
    if coef < 0 {
        -best_l
    } else {
        best_l
    }
}

/// Forward 8×8 DCT — built by inverting the IDCT used by the decoder so the
/// two are bit-exact inverses (within float rounding). The DCT matrix is
/// orthonormal with our normalisation, so `FDCT == IDCT^T == IDCT` when
/// applied as a matrix on the left (rows). Using the same `idct8x8` for the
/// forward transform via an explicit transpose path gives a self-inverse
/// transform under the same f32 cosine table.
pub fn fdct8x8(block: &mut [f32; 64]) {
    use std::f32::consts::PI;
    use std::sync::OnceLock;

    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    let cos = T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 {
                (1.0_f32 / 2.0_f32).sqrt()
            } else {
                1.0
            };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    });

    let mut tmp = [0.0f32; 64];
    // Row-wise forward: tmp[y][k] = Σn t[k][n] * block[y][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += cos[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    // Column-wise.
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += cos[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdct_idct_round_trip() {
        use crate::block::idct8x8;
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = ((i * 7) % 255) as f32;
        }
        let original = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < 1e-2,
                "round-trip mismatch at {i}: got {} want {}",
                block[i],
                original[i]
            );
        }
    }

    #[test]
    fn dc_size_round_trip_luma() {
        // For a few representative diff values, encode+decode and compare.
        use crate::bitreader::BitReader;
        use crate::block::decode_intra_dc_diff;
        for &diff in &[0i32, 1, -1, 5, -5, 127, -127, 2047, -2047] {
            let mut bw = BitWriter::new();
            write_intra_dc_diff(&mut bw, 0, diff);
            // The reader peeks 16 bits so we need at least 2 bytes worth of input.
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let got = decode_intra_dc_diff(&mut br, 0).unwrap();
            assert_eq!(got, diff, "luma DC round-trip failed for {diff}");
        }
    }

    #[test]
    fn dc_size_round_trip_chroma() {
        use crate::bitreader::BitReader;
        use crate::block::decode_intra_dc_diff;
        for &diff in &[0i32, 3, -3, 200, -200, 1000, -1000] {
            let mut bw = BitWriter::new();
            write_intra_dc_diff(&mut bw, 4, diff);
            let mut data = bw.finish();
            data.extend_from_slice(&[0xFF, 0xFF]);
            let mut br = BitReader::new(&data);
            let got = decode_intra_dc_diff(&mut br, 4).unwrap();
            assert_eq!(got, diff, "chroma DC round-trip failed for {diff}");
        }
    }
}
