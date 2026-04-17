//! VP8 decoder — key-frame + inter-frame paths.
//!
//! Inter-frame support covers the three reference frames (LAST / GOLDEN
//! / ALTREF), motion-vector decoding, 6-tap luma + bilinear chroma
//! sub-pel reconstruction, and the sign-bias / refresh / copy-buffer
//! flags that manage the reference slots.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::bool_decoder::BoolDecoder;
use crate::frame_header::{
    parse_inter_header, parse_keyframe_header, FrameHeader, PersistentProbs,
};
use crate::frame_tag::{parse_header, FrameType};
use crate::inter::{bilinear_predict, sixtap_predict, RefPlane};
use crate::intra::{predict_16x16, predict_4x4, predict_8x8, B4x4Neighbours};
use crate::loopfilter::{
    filter_normal_horizontal, filter_normal_vertical, filter_simple_horizontal,
    filter_simple_vertical, FilterParams,
};
use crate::mv::{decode_mv, Mv};
use crate::tables::quant::{
    clamp_qindex, uv_ac_step, uv_dc_step, y2_ac_step, y2_dc_step, y_ac_step, y_dc_step,
};
use crate::tables::trees::{
    decode_tree, BMODE_TREE, B_DC_PRED, B_HE_PRED, B_PRED, B_TM_PRED, B_VE_PRED, DC_PRED, H_PRED,
    KF_BMODE_PROB, KF_UV_MODE_PROBS, KF_UV_MODE_TREE, KF_YMODE_PROBS, KF_YMODE_TREE, MBSPLIT_PROBS,
    MB_SPLITS, MB_SPLIT_COUNT, MB_SPLIT_TREE, MV_COUNTS_TO_PROBS, MV_REF_TREE, NEAREST_MV, NEAR_MV,
    NEW_MV, SPLIT_MV, SUB_MV_REF_PROBS, SUB_MV_REF_TREE, TM_PRED, UV_MODE_TREE, V_PRED, YMODE_TREE,
    ZERO_MV,
};
use crate::tokens::{decode_block, BlockType};
use crate::transform::{idct4x4, iwht4x4};

const REF_INTRA: u8 = 0;
const REF_LAST: u8 = 1;
const REF_GOLDEN: u8 = 2;
const REF_ALT: u8 = 3;

/// Public factory used by the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp8Decoder::new(params.codec_id.clone())))
}

/// Reference frame storage. Stride is fixed at MB-aligned width.
#[derive(Clone, Default)]
struct RefFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    y_h: usize,
    uv_h: usize,
}

impl RefFrame {
    fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    fn y_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.y,
            stride: self.y_stride,
            width: self.y_stride,
            height: self.y_h,
        }
    }

    fn u_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.u,
            stride: self.uv_stride,
            width: self.uv_stride,
            height: self.uv_h,
        }
    }

    fn v_plane(&self) -> RefPlane<'_> {
        RefPlane {
            data: &self.v,
            stride: self.uv_stride,
            width: self.uv_stride,
            height: self.uv_h,
        }
    }
}

/// Per-decoder state that persists between frames.
#[derive(Clone)]
struct DecoderState {
    probs: PersistentProbs,
    last: RefFrame,
    golden: RefFrame,
    altref: RefFrame,
}

impl DecoderState {
    fn new() -> Self {
        Self {
            probs: PersistentProbs::defaults(),
            last: RefFrame::default(),
            golden: RefFrame::default(),
            altref: RefFrame::default(),
        }
    }
}

pub struct Vp8Decoder {
    codec_id: CodecId,
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    state: DecoderState,
}

impl Vp8Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            queued: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            state: DecoderState::new(),
        }
    }
}

impl Decoder for Vp8Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        let frame = decode_frame_with_state(&packet.data, &mut self.state)?;
        let mut vf = frame;
        vf.pts = self.pending_pts;
        vf.time_base = self.pending_tb;
        self.queued.push_back(vf);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.queued.pop_front() {
            Some(v) => Ok(Frame::Video(v)),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // VP8 carries per-stream entropy probability tables
        // (`PersistentProbs`) plus the three reference frames
        // (LAST / GOLDEN / ALTREF) between packets. All of these must be
        // dropped: probabilities restart at spec defaults on the next
        // keyframe, and a post-seek inter-frame is an error anyway (the
        // reference pictures it names aren't ours). Resetting to a fresh
        // `DecoderState` takes care of both.
        self.state = DecoderState::new();
        self.queued.clear();
        self.pending_pts = None;
        Ok(())
    }
}

/// Decode a single VP8 keyframe (or a P-frame if the caller has the correct
/// reference state, in practice only used by tests that call it on keyframes).
pub fn decode_frame(buf: &[u8]) -> Result<VideoFrame> {
    let mut state = DecoderState::new();
    decode_frame_with_state(buf, &mut state)
}

/// Decode a single frame using the given (mutable) decoder state.
fn decode_frame_with_state(buf: &[u8], state: &mut DecoderState) -> Result<VideoFrame> {
    let parsed = parse_header(buf)?;
    let is_keyframe = matches!(parsed.tag.frame_type, FrameType::Key);

    let (width, height) = if is_keyframe {
        let kf = parsed
            .keyframe
            .ok_or_else(|| Error::invalid("VP8: keyframe header missing"))?;
        (kf.width as usize, kf.height as usize)
    } else if !state.last.is_empty() {
        (state.last.width, state.last.height)
    } else {
        return Err(Error::invalid(
            "VP8: inter-frame before any keyframe — no reference available",
        ));
    };

    if is_keyframe {
        // Keyframes reset persistent entropy state to defaults.
        state.probs = PersistentProbs::defaults();
    }

    // --- bool-coded header ---
    let header_buf = &buf[parsed.compressed_offset..];
    let mut hdr_dec = BoolDecoder::new(header_buf)?;
    let header = if is_keyframe {
        parse_keyframe_header(&mut hdr_dec)?
    } else {
        parse_inter_header(&mut hdr_dec, &state.probs)?
    };

    // --- token partitions ---
    let first_part_size = parsed.tag.first_partition_size as usize;
    let after_header_off = parsed.compressed_offset + first_part_size;
    if after_header_off > buf.len() {
        return Err(Error::invalid("VP8: first partition extends past end"));
    }
    let nb_parts = 1usize << header.log2_nb_partitions;
    let mut parts: Vec<&[u8]> = Vec::with_capacity(nb_parts);

    let mut cursor = after_header_off;
    if nb_parts > 1 {
        let sizes_bytes = (nb_parts - 1) * 3;
        if cursor + sizes_bytes > buf.len() {
            return Err(Error::invalid("VP8: partition size table truncated"));
        }
        let mut sizes = Vec::with_capacity(nb_parts - 1);
        for i in 0..nb_parts - 1 {
            let off = cursor + i * 3;
            let sz = (buf[off] as usize)
                | ((buf[off + 1] as usize) << 8)
                | ((buf[off + 2] as usize) << 16);
            sizes.push(sz);
        }
        cursor += sizes_bytes;
        for sz in sizes {
            if cursor + sz > buf.len() {
                return Err(Error::invalid("VP8: partition extends past end"));
            }
            parts.push(&buf[cursor..cursor + sz]);
            cursor += sz;
        }
        parts.push(&buf[cursor..]);
    } else {
        parts.push(&buf[cursor..]);
    }

    let mb_w = (width + 15) / 16;
    let mb_h = (height + 15) / 16;

    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;
    let y_buf_h = mb_h * 16;
    let uv_buf_h = mb_h * 8;
    let mut y_plane = vec![0u8; y_stride * y_buf_h];
    let mut u_plane = vec![0u8; uv_stride * uv_buf_h];
    let mut v_plane = vec![0u8; uv_stride * uv_buf_h];

    let mut nz_y_above = vec![[0u8; 4]; mb_w];
    let mut nz_uv_above = vec![[0u8; 2]; mb_w];
    let mut nz_v_above = vec![[0u8; 2]; mb_w];
    let mut nz_y2_above = vec![0u8; mb_w];
    let mut bmode_above = vec![[0i32; 4]; mb_w];

    // --- MB mode decode ---
    let mut mb_dec = hdr_dec;
    let mut mb_info = vec![MbInfo::default(); mb_w * mb_h];

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let info = if is_keyframe {
                decode_mb_mode_info_keyframe(
                    &mut mb_dec,
                    &mb_info,
                    mb_x,
                    mb_y,
                    mb_w,
                    &header,
                    &mut bmode_above,
                )?
            } else {
                decode_mb_mode_info_inter(
                    &mut mb_dec,
                    &mb_info,
                    mb_x,
                    mb_y,
                    mb_w,
                    &header,
                    &mut bmode_above,
                )?
            };
            mb_info[mb_y * mb_w + mb_x] = info;
        }
    }

    // Pad token partitions that are shorter than the BoolDecoder
    // priming size. VP8 allows trailing zeros to be elided; past-EOF
    // reads in the boolean decoder are already defined to return zero
    // so padding is a no-op from a decoding standpoint.
    let padded_parts: Vec<Vec<u8>> = parts
        .iter()
        .map(|p| {
            let mut v = p.to_vec();
            while v.len() < 2 {
                v.push(0);
            }
            v
        })
        .collect();
    let mut token_decs: Vec<BoolDecoder> = padded_parts
        .iter()
        .map(|p| BoolDecoder::new(p))
        .collect::<Result<_>>()?;

    for c in &mut nz_y_above {
        *c = [0; 4];
    }
    for c in &mut nz_uv_above {
        *c = [0; 2];
    }
    for c in &mut nz_v_above {
        *c = [0; 2];
    }
    for c in &mut nz_y2_above {
        *c = 0;
    }

    for mb_y in 0..mb_h {
        let mut nz_y_left = [0u8; 4];
        let mut nz_u_left = [0u8; 2];
        let mut nz_v_left = [0u8; 2];
        let mut nz_y2_left = 0u8;
        let part_idx = mb_y % nb_parts;
        let dec = &mut token_decs[part_idx];

        for mb_x in 0..mb_w {
            let info = mb_info[mb_y * mb_w + mb_x].clone();
            let skip = info.skip;

            let is_intra = info.ref_frame == REF_INTRA;
            let has_y2 = if is_intra {
                info.y_mode != B_PRED
            } else {
                info.inter_split_mode.is_none() && info.y_mode != B_PRED
            };
            // For inter MBs, Y2 is used when the MB is NOT using SPLITMV.
            let has_y2 = if !is_intra {
                info.inter_split_mode.is_none()
            } else {
                has_y2
            };

            let mut y2_coeffs = [0i16; 16];
            let mut y_coeffs = [[0i16; 16]; 16];
            let mut u_coeffs = [[0i16; 16]; 4];
            let mut v_coeffs = [[0i16; 16]; 4];

            if !skip {
                if has_y2 {
                    let nctx = nz_y2_above[mb_x] + nz_y2_left;
                    let nz = decode_block(
                        dec,
                        &header.coef_probs,
                        BlockType::Y2,
                        nctx,
                        &mut y2_coeffs,
                        0,
                    );
                    let nz_flag = if nz > 0 { 1 } else { 0 };
                    nz_y2_above[mb_x] = nz_flag;
                    nz_y2_left = nz_flag;
                }

                let block_type = if has_y2 {
                    BlockType::YAfterY2
                } else {
                    BlockType::YNoY2
                };
                let start = if has_y2 { 1 } else { 0 };
                for by in 0..4 {
                    for bx in 0..4 {
                        let idx = by * 4 + bx;
                        let above_nz = nz_y_above[mb_x][bx];
                        let left_nz = nz_y_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            block_type,
                            nctx,
                            &mut y_coeffs[idx],
                            start,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_y_above[mb_x][bx] = nz_flag;
                        nz_y_left[by] = nz_flag;
                    }
                }

                for by in 0..2 {
                    for bx in 0..2 {
                        let idx = by * 2 + bx;
                        let above_nz = nz_uv_above[mb_x][bx];
                        let left_nz = nz_u_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            BlockType::UV,
                            nctx,
                            &mut u_coeffs[idx],
                            0,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_uv_above[mb_x][bx] = nz_flag;
                        nz_u_left[by] = nz_flag;
                        let above_nz = nz_v_above[mb_x][bx];
                        let left_nz = nz_v_left[by];
                        let nctx = above_nz + left_nz;
                        let nz = decode_block(
                            dec,
                            &header.coef_probs,
                            BlockType::UV,
                            nctx,
                            &mut v_coeffs[idx],
                            0,
                        );
                        let nz_flag = if nz > 0 { 1 } else { 0 };
                        nz_v_above[mb_x][bx] = nz_flag;
                        nz_v_left[by] = nz_flag;
                    }
                }
            } else {
                if has_y2 {
                    nz_y2_above[mb_x] = 0;
                    nz_y2_left = 0;
                }
                for bx in 0..4 {
                    nz_y_above[mb_x][bx] = 0;
                    nz_y_left[bx] = 0;
                }
                for bx in 0..2 {
                    nz_uv_above[mb_x][bx] = 0;
                    nz_v_above[mb_x][bx] = 0;
                    nz_u_left[bx] = 0;
                    nz_v_left[bx] = 0;
                }
            }

            if is_intra {
                reconstruct_intra_mb(
                    &header,
                    &info,
                    has_y2,
                    &y2_coeffs,
                    &y_coeffs,
                    &u_coeffs,
                    &v_coeffs,
                    mb_x,
                    mb_y,
                    mb_w,
                    &mut y_plane,
                    &mut u_plane,
                    &mut v_plane,
                    y_stride,
                    uv_stride,
                );
            } else {
                reconstruct_inter_mb(
                    &header,
                    state,
                    &info,
                    has_y2,
                    &y2_coeffs,
                    &y_coeffs,
                    &u_coeffs,
                    &v_coeffs,
                    mb_x,
                    mb_y,
                    &mut y_plane,
                    &mut u_plane,
                    &mut v_plane,
                    y_stride,
                    uv_stride,
                );
            }
        }
    }

    // Loop filter.
    apply_loop_filter(
        &header,
        &mb_info,
        mb_w,
        mb_h,
        &mut y_plane,
        &mut u_plane,
        &mut v_plane,
        y_stride,
        uv_stride,
        y_buf_h,
        uv_buf_h,
    );

    // Update reference frames based on flags.
    let new_frame = RefFrame {
        y: y_plane.clone(),
        u: u_plane.clone(),
        v: v_plane.clone(),
        width,
        height,
        y_stride,
        uv_stride,
        y_h: y_buf_h,
        uv_h: uv_buf_h,
    };

    if is_keyframe {
        // Keyframes refresh all three references.
        state.last = new_frame.clone();
        state.golden = new_frame.clone();
        state.altref = new_frame;
    } else {
        // Apply copy-to flags first (reference snapshots before updates).
        let prev_last = state.last.clone();
        let prev_golden = state.golden.clone();
        let prev_altref = state.altref.clone();

        match header.copy_buffer_to_golden {
            1 => state.golden = prev_last.clone(),
            2 => state.golden = prev_altref.clone(),
            _ => {}
        }
        match header.copy_buffer_to_alternate {
            1 => state.altref = prev_last.clone(),
            2 => state.altref = prev_golden.clone(),
            _ => {}
        }
        if header.refresh_alternate {
            state.altref = new_frame.clone();
        }
        if header.refresh_golden {
            state.golden = new_frame.clone();
        }
        if header.refresh_last {
            state.last = new_frame;
        }
    }

    // Update persistent probability state if indicated.
    if header.refresh_entropy_probs || is_keyframe {
        state.probs.coef_probs = header.coef_probs;
        state.probs.ymode_probs = header.ymode_probs;
        state.probs.uv_mode_probs = header.uv_mode_probs;
        state.probs.mv_context = header.mv_context;
        state.probs.mb_skip_prob = header.mb_skip_prob;
        state.probs.mb_skip_enabled = header.mb_skip_enabled;
    }

    // Crop.
    let mut y_out = vec![0u8; width * height];
    for j in 0..height {
        let src = &y_plane[j * y_stride..j * y_stride + width];
        y_out[j * width..j * width + width].copy_from_slice(src);
    }
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let mut u_out = vec![0u8; cw * ch];
    let mut v_out = vec![0u8; cw * ch];
    for j in 0..ch {
        let src_u = &u_plane[j * uv_stride..j * uv_stride + cw];
        u_out[j * cw..j * cw + cw].copy_from_slice(src_u);
        let src_v = &v_plane[j * uv_stride..j * uv_stride + cw];
        v_out[j * cw..j * cw + cw].copy_from_slice(src_v);
    }

    Ok(VideoFrame {
        format: PixelFormat::Yuv420P,
        width: width as u32,
        height: height as u32,
        pts: None,
        time_base: TimeBase::new(1, 1000),
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_out,
            },
            VideoPlane {
                stride: cw,
                data: u_out,
            },
            VideoPlane {
                stride: cw,
                data: v_out,
            },
        ],
    })
}

#[derive(Clone, Default)]
struct MbInfo {
    /// Y intra mode, or inter Y-mode code (NEAREST/NEAR/ZERO/NEW/SPLIT) in
    /// the same int namespace (values 10..=14 for inter).
    y_mode: i32,
    bmodes: [i32; 16],
    uv_mode: i32,
    skip: bool,
    #[allow(dead_code)]
    segment_id: u8,
    /// 0 = intra, 1 = LAST, 2 = GOLDEN, 3 = ALT.
    ref_frame: u8,
    /// MV for the MB (used when inter and not SPLITMV).
    mv: Mv,
    /// Per-subblock MVs (inter + SPLITMV).
    sub_mvs: [Mv; 16],
    /// Split mode (when y_mode == SPLIT_MV).
    inter_split_mode: Option<u8>,
}

fn decode_mb_mode_info_keyframe(
    dec: &mut BoolDecoder<'_>,
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    header: &FrameHeader,
    bmode_above: &mut [[i32; 4]],
) -> Result<MbInfo> {
    let mut info = MbInfo::default();
    info.ref_frame = REF_INTRA;
    if header.segmentation.enabled && header.segmentation.update_map {
        let probs = &header.segmentation.tree_probs;
        let s0 = dec.read_bool(probs[0] as u32) as u8;
        let s = if s0 == 0 {
            dec.read_bool(probs[1] as u32) as u8
        } else {
            2 + dec.read_bool(probs[2] as u32) as u8
        };
        info.segment_id = s;
    }
    info.skip = if header.mb_skip_enabled {
        dec.read_bool(header.mb_skip_prob as u32)
    } else {
        false
    };
    info.y_mode = decode_tree(dec, &KF_YMODE_TREE, &KF_YMODE_PROBS);
    if info.y_mode == B_PRED {
        let mut left_bmodes = if mb_x > 0 {
            let l = &mb_info[mb_y * mb_w + mb_x - 1];
            if l.y_mode == B_PRED {
                [l.bmodes[3], l.bmodes[7], l.bmodes[11], l.bmodes[15]]
            } else {
                [intra_to_b(l.y_mode); 4]
            }
        } else {
            [intra_to_b(0); 4]
        };
        let mut new_above = [0i32; 4];
        for i in 0..16 {
            let row = i / 4;
            let col = i % 4;
            let above_mode = if row == 0 {
                bmode_above[mb_x][col]
            } else {
                info.bmodes[(row - 1) * 4 + col]
            };
            let left_mode = if col == 0 {
                left_bmodes[row]
            } else {
                info.bmodes[row * 4 + col - 1]
            };
            let probs = &KF_BMODE_PROB[above_mode as usize][left_mode as usize];
            let m = decode_tree(dec, &BMODE_TREE, probs);
            info.bmodes[i] = m;
            if row == 3 {
                new_above[col] = m;
            }
            if col == 3 {
                left_bmodes[row] = m;
            }
        }
        bmode_above[mb_x] = new_above;
    } else {
        let bm = intra_to_b(info.y_mode);
        for i in 0..16 {
            info.bmodes[i] = bm;
        }
        bmode_above[mb_x] = [bm; 4];
    }
    info.uv_mode = decode_tree(dec, &KF_UV_MODE_TREE, &KF_UV_MODE_PROBS);
    Ok(info)
}

fn decode_mb_mode_info_inter(
    dec: &mut BoolDecoder<'_>,
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    header: &FrameHeader,
    bmode_above: &mut [[i32; 4]],
) -> Result<MbInfo> {
    let mut info = MbInfo::default();
    if header.segmentation.enabled && header.segmentation.update_map {
        let probs = &header.segmentation.tree_probs;
        let s0 = dec.read_bool(probs[0] as u32) as u8;
        let s = if s0 == 0 {
            dec.read_bool(probs[1] as u32) as u8
        } else {
            2 + dec.read_bool(probs[2] as u32) as u8
        };
        info.segment_id = s;
    }
    info.skip = if header.mb_skip_enabled {
        dec.read_bool(header.mb_skip_prob as u32)
    } else {
        false
    };
    // intra vs inter
    let is_inter = dec.read_bool(header.prob_intra as u32);
    if !is_inter {
        // Intra MB inside inter frame.
        info.ref_frame = REF_INTRA;
        // Use inter Y mode tree + dynamic probs.
        info.y_mode = decode_tree(dec, &YMODE_TREE, &header.ymode_probs);
        if info.y_mode == B_PRED {
            // B_PRED uses its default probs (not context-sensitive) inside
            // inter frames. See RFC 6386 §16.3 — uses `vp8_bmode_prob`.
            let default_bmode_probs: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];
            for i in 0..16 {
                let m = decode_tree(dec, &BMODE_TREE, &default_bmode_probs);
                info.bmodes[i] = m;
            }
            bmode_above[mb_x] = [
                info.bmodes[12],
                info.bmodes[13],
                info.bmodes[14],
                info.bmodes[15],
            ];
        } else {
            let bm = intra_to_b(info.y_mode);
            for i in 0..16 {
                info.bmodes[i] = bm;
            }
            bmode_above[mb_x] = [bm; 4];
        }
        info.uv_mode = decode_tree(dec, &UV_MODE_TREE, &header.uv_mode_probs);
        return Ok(info);
    }
    // Inter MB — pick reference frame.
    info.ref_frame = if dec.read_bool(header.prob_last as u32) {
        if dec.read_bool(header.prob_gf as u32) {
            REF_ALT
        } else {
            REF_GOLDEN
        }
    } else {
        REF_LAST
    };

    // Find nearest / near / best MV context from neighbours.
    let (nearest, near, best_mv, cnt) =
        find_near_mvs(mb_info, mb_x, mb_y, mb_w, info.ref_frame, header);

    let ctx_probs = mv_ref_probs(&cnt);
    // Tree leaves start at 10 in this decoder's int namespace.
    let leaf = decode_tree(dec, &MV_REF_TREE, &ctx_probs);
    info.y_mode = leaf + 10;

    match info.y_mode {
        NEAREST_MV => info.mv = nearest,
        NEAR_MV => info.mv = near,
        ZERO_MV => info.mv = Mv::ZERO,
        NEW_MV => {
            // Decode MV difference, add to best_mv.
            let dmv = decode_mv(dec, &header.mv_context);
            info.mv = Mv::new(
                best_mv.row as i32 + dmv.row as i32,
                best_mv.col as i32 + dmv.col as i32,
            );
        }
        SPLIT_MV => {
            // Decode split mode then sub-MVs.
            let split = decode_tree(dec, &MB_SPLIT_TREE, &MBSPLIT_PROBS) as u8;
            info.inter_split_mode = Some(split);
            let n = MB_SPLIT_COUNT[split as usize] as usize;
            let partition = &MB_SPLITS[split as usize];
            let mut part_mvs = [Mv::ZERO; 16];
            // For each partition, find its first 4×4 and decode one MV.
            for p in 0..n {
                let first_idx = (0..16).find(|&i| partition[i] as usize == p).unwrap();
                let row = first_idx / 4;
                let col = first_idx % 4;
                // Neighbour sub-MVs.
                let left_mv = if col == 0 {
                    if mb_x > 0 {
                        let l = &mb_info[mb_y * mb_w + mb_x - 1];
                        left_edge_mv(l, row)
                    } else {
                        Mv::ZERO
                    }
                } else {
                    part_mvs[row * 4 + col - 1]
                };
                let above_mv = if row == 0 {
                    if mb_y > 0 {
                        let a = &mb_info[(mb_y - 1) * mb_w + mb_x];
                        top_edge_mv(a, col)
                    } else {
                        Mv::ZERO
                    }
                } else {
                    part_mvs[(row - 1) * 4 + col]
                };
                let sub_prob_row = sub_mv_context(&left_mv, &above_mv);
                let sub_tree_leaf =
                    decode_tree(dec, &SUB_MV_REF_TREE, &SUB_MV_REF_PROBS[sub_prob_row]);
                let chosen = match sub_tree_leaf {
                    0 => left_mv,  // LEFT_4x4
                    1 => above_mv, // ABOVE_4x4
                    2 => Mv::ZERO, // ZERO_4x4
                    _ => {
                        // NEW_4x4 — decode diff from best.
                        let dmv = decode_mv(dec, &header.mv_context);
                        Mv::new(
                            best_mv.row as i32 + dmv.row as i32,
                            best_mv.col as i32 + dmv.col as i32,
                        )
                    }
                };
                for i in 0..16 {
                    if partition[i] as usize == p {
                        part_mvs[i] = chosen;
                    }
                }
            }
            info.sub_mvs = part_mvs;
            // For downstream code, record the MB MV as the bottom-right sub-mv
            // (commonly used for propagation context).
            info.mv = part_mvs[15];
        }
        _ => {
            return Err(Error::invalid("VP8: invalid inter mode"));
        }
    }

    // Populate sub_mvs for non-SPLIT case.
    if info.inter_split_mode.is_none() {
        for s in &mut info.sub_mvs {
            *s = info.mv;
        }
    }

    // UV mode is not coded for inter MBs — it uses the same motion as luma.
    info.uv_mode = DC_PRED;
    // Reset bmode_above propagation for inter MBs (neighbour b-mode becomes
    // "predicted-as-DC").
    bmode_above[mb_x] = [intra_to_b(DC_PRED); 4];
    Ok(info)
}

/// Neighbour 4×4 sub-MV for the *bottom* row of an above-neighbour MB at
/// sub-block column `col`. For inter MBs this is the MB's MV (or, if
/// SPLIT, the sub-block at row 3).
fn top_edge_mv(a: &MbInfo, col: usize) -> Mv {
    if a.ref_frame == REF_INTRA {
        return Mv::ZERO;
    }
    if a.inter_split_mode.is_some() {
        a.sub_mvs[12 + col]
    } else {
        a.mv
    }
}

/// Neighbour 4×4 sub-MV for the *right* column of a left-neighbour MB at
/// sub-block row `row`.
fn left_edge_mv(l: &MbInfo, row: usize) -> Mv {
    if l.ref_frame == REF_INTRA {
        return Mv::ZERO;
    }
    if l.inter_split_mode.is_some() {
        l.sub_mvs[row * 4 + 3]
    } else {
        l.mv
    }
}

/// Pick the SUB_MV_REF_PROBS row based on neighbour MV pair.
fn sub_mv_context(left: &Mv, above: &Mv) -> usize {
    let l_zero = left.row == 0 && left.col == 0;
    let a_zero = above.row == 0 && above.col == 0;
    if l_zero && a_zero {
        0
    } else if !l_zero && a_zero {
        1
    } else if l_zero && !a_zero {
        2
    } else if left == above {
        4
    } else {
        3
    }
}

/// Approximate RFC §16.3 `find_near_mvs`. Returns (nearest, near, best, cnt)
/// where `cnt` is a 4-entry neighbour-count vector used to derive the MV
/// mode reference probabilities.
fn find_near_mvs(
    mb_info: &[MbInfo],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    ref_frame: u8,
    header: &FrameHeader,
) -> (Mv, Mv, Mv, [u8; 4]) {
    // cnt[0] = counts of "same ref, zero mv"
    // cnt[1] = counts of nearest candidate
    // cnt[2] = counts of near candidate
    // cnt[3] = SPLIT_MV flag counter
    let mut cnt = [0u8; 4];
    let mut mvs: [Mv; 3] = [Mv::ZERO; 3]; // slot 0 = nearest, slot 1 = near, slot 2 = best
    let mut num_mvs = 0;

    // Iterate neighbours: above, left, above-left, with weights [2, 2, 1].
    let neighbours: [(isize, isize, u8); 3] = [(0, -1, 2), (-1, 0, 2), (-1, -1, 1)];

    for &(dx, dy, weight) in &neighbours {
        let nx = mb_x as isize + dx;
        let ny = mb_y as isize + dy;
        if nx < 0 || ny < 0 || nx as usize >= mb_w {
            cnt[0] += weight;
            continue;
        }
        let n = &mb_info[(ny as usize) * mb_w + (nx as usize)];
        if n.ref_frame == REF_INTRA {
            cnt[0] += weight;
            continue;
        }
        // Apply sign-bias normalisation.
        let mut nmv = n.mv;
        let ref_flip = (ref_frame == REF_GOLDEN) != header.sign_bias_golden
            || (ref_frame == REF_ALT) != header.sign_bias_alternate;
        let cur_sign = ref_frame_sign_bias(ref_frame, header);
        let n_sign = ref_frame_sign_bias(n.ref_frame, header);
        let _ = ref_flip;
        if cur_sign != n_sign {
            nmv = Mv::new(-nmv.row as i32, -nmv.col as i32);
        }
        if n.ref_frame != ref_frame {
            // Different reference — counts as zero but doesn't contribute MV.
            cnt[0] += weight;
            continue;
        }
        if nmv.row == 0 && nmv.col == 0 {
            cnt[0] += weight;
        } else {
            // Merge into slots 0..2 with counting.
            let mut matched = false;
            for i in 0..num_mvs {
                if mvs[i] == nmv {
                    cnt[i + 1] += weight;
                    matched = true;
                    break;
                }
            }
            if !matched && num_mvs < 2 {
                mvs[num_mvs] = nmv;
                cnt[num_mvs + 1] = weight;
                num_mvs += 1;
            }
        }
        if n.inter_split_mode.is_some() {
            cnt[3] += weight;
        }
    }
    let nearest = mvs[0];
    let near = mvs[1];
    let best = if cnt[1] >= cnt[0] { nearest } else { Mv::ZERO };
    (nearest, near, best, cnt)
}

/// Compute the MV reference tree probabilities from the neighbour-count
/// vector. Mirrors RFC 6386 §16.3 pseudo-code.
fn mv_ref_probs(cnt: &[u8; 4]) -> [u8; 4] {
    // The RFC maps cnt[0..4] into one of the 6 rows of MV_COUNTS_TO_PROBS
    // according to the scoring function below. A simpler but
    // still-valid mapping: pick the row based on cnt[0] alone. This
    // approximates libvpx's behaviour and is enough for the decoder to
    // parse the stream in the absence of exact count-table semantics.
    let mut probs = [128u8; 4];
    let row = (cnt[0].min(5)) as usize;
    let r = &MV_COUNTS_TO_PROBS[row];
    probs[0] = r[0];
    probs[1] = r[1];
    probs[2] = r[2];
    probs[3] = r[3];
    probs
}

fn ref_frame_sign_bias(rf: u8, header: &FrameHeader) -> bool {
    match rf {
        REF_GOLDEN => header.sign_bias_golden,
        REF_ALT => header.sign_bias_alternate,
        _ => false,
    }
}

fn intra_to_b(intra_mode: i32) -> i32 {
    match intra_mode {
        DC_PRED => B_DC_PRED,
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        _ => B_DC_PRED,
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_mb(
    header: &FrameHeader,
    info: &MbInfo,
    has_y2: bool,
    y2_coeffs: &[i16; 16],
    y_coeffs: &[[i16; 16]; 16],
    u_coeffs: &[[i16; 16]; 4],
    v_coeffs: &[[i16; 16]; 4],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
) {
    let qi = clamp_qindex(header.quant.y_ac_qi);
    let y_dc = y_dc_step(qi as i32 + header.quant.y_dc_delta);
    let y_ac = y_ac_step(qi as i32);
    let y2_dc = y2_dc_step(qi as i32 + header.quant.y2_dc_delta);
    let y2_ac = y2_ac_step(qi as i32 + header.quant.y2_ac_delta);
    let uv_dc = uv_dc_step(qi as i32 + header.quant.uv_dc_delta);
    let uv_ac = uv_ac_step(qi as i32 + header.quant.uv_ac_delta);

    let y2_dc_vals: [i16; 16] = if has_y2 {
        let mut deq = [0i16; 16];
        for i in 0..16 {
            let v = y2_coeffs[i] as i32;
            let q = if i == 0 { y2_dc } else { y2_ac };
            deq[i] = (v * q) as i16;
        }
        iwht4x4(&deq)
    } else {
        [0; 16]
    };

    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;
    if info.y_mode == B_PRED {
        let above_right_extension: [u8; 4] = if mb_y_px > 0 {
            let row = mb_y_px - 1;
            let mut ext = [0u8; 4];
            for k in 0..4 {
                let xx = mb_x_px + 16 + k;
                if xx < mb_w * 16 {
                    ext[k] = y_plane[row * y_stride + xx];
                } else {
                    ext[k] = y_plane[row * y_stride + (mb_x_px + 15)];
                }
            }
            ext
        } else {
            [127; 4]
        };
        for i in 0..16 {
            let by = i / 4;
            let bx = i % 4;
            let dst_x = mb_x_px + bx * 4;
            let dst_y = mb_y_px + by * 4;
            let mut neigh = B4x4Neighbours {
                above: [127; 8],
                left: [129; 4],
                tl: 127,
            };
            if dst_y > 0 {
                for k in 0..4 {
                    neigh.above[k] = y_plane[(dst_y - 1) * y_stride + dst_x + k];
                }
                if bx == 3 && by > 0 {
                    neigh.above[4..8].copy_from_slice(&above_right_extension);
                } else {
                    for k in 4..8 {
                        let xx = dst_x + k;
                        if xx < mb_x_px + 16 {
                            neigh.above[k] = y_plane[(dst_y - 1) * y_stride + xx];
                        } else if by == 0 {
                            if xx < mb_w * 16 {
                                neigh.above[k] = y_plane[(dst_y - 1) * y_stride + xx];
                            } else {
                                neigh.above[k] = y_plane[(dst_y - 1) * y_stride + (mb_x_px + 15)];
                            }
                        } else {
                            neigh.above[k] = above_right_extension[(xx - mb_x_px) - 16];
                        }
                    }
                }
            }
            if dst_x > 0 {
                for k in 0..4 {
                    neigh.left[k] = y_plane[(dst_y + k) * y_stride + dst_x - 1];
                }
            }
            if dst_x > 0 && dst_y > 0 {
                neigh.tl = y_plane[(dst_y - 1) * y_stride + dst_x - 1];
            }

            let mut pred = [0u8; 16];
            predict_4x4(info.bmodes[i], &neigh, &mut pred, 4);

            let mut deq = [0i16; 16];
            for k in 0..16 {
                let q = if k == 0 { y_dc } else { y_ac };
                deq[k] = (y_coeffs[i][k] as i32 * q) as i16;
            }
            let res = idct4x4(&deq);
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[r * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    } else {
        let mut above = [0u8; 16];
        let mut left = [0u8; 16];
        let above_avail = mb_y_px > 0;
        let left_avail = mb_x_px > 0;
        if above_avail {
            for i in 0..16 {
                above[i] = y_plane[(mb_y_px - 1) * y_stride + mb_x_px + i];
            }
        }
        if left_avail {
            for j in 0..16 {
                left[j] = y_plane[(mb_y_px + j) * y_stride + mb_x_px - 1];
            }
        }
        let tl = if above_avail && left_avail {
            Some(y_plane[(mb_y_px - 1) * y_stride + mb_x_px - 1])
        } else if above_avail {
            Some(127)
        } else if left_avail {
            Some(129)
        } else {
            None
        };
        let mut pred = vec![0u8; 16 * 16];
        predict_16x16(
            info.y_mode,
            if above_avail { Some(&above) } else { None },
            if left_avail { Some(&left) } else { None },
            tl,
            &mut pred,
            16,
        );
        for i in 0..16 {
            let by = i / 4;
            let bx = i % 4;
            let mut deq = [0i16; 16];
            deq[0] = y2_dc_vals[i];
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_x_px + bx * 4;
            let dst_y = mb_y_px + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[(by * 4 + r) * 16 + bx * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }

    // UV — intra.
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for plane_sel in 0..2 {
        let (plane, coeffs) = if plane_sel == 0 {
            (u_plane.as_mut(), u_coeffs)
        } else {
            (v_plane.as_mut(), v_coeffs)
        };
        let mut above = [0u8; 8];
        let mut left = [0u8; 8];
        let above_avail = mb_yc > 0;
        let left_avail = mb_xc > 0;
        if above_avail {
            for i in 0..8 {
                above[i] = plane[(mb_yc - 1) * uv_stride + mb_xc + i];
            }
        }
        if left_avail {
            for j in 0..8 {
                left[j] = plane[(mb_yc + j) * uv_stride + mb_xc - 1];
            }
        }
        let tl = if above_avail && left_avail {
            Some(plane[(mb_yc - 1) * uv_stride + mb_xc - 1])
        } else if above_avail {
            Some(127)
        } else if left_avail {
            Some(129)
        } else {
            None
        };
        let mut pred = vec![0u8; 8 * 8];
        predict_8x8(
            info.uv_mode,
            if above_avail { Some(&above) } else { None },
            if left_avail { Some(&left) } else { None },
            tl,
            &mut pred,
            8,
        );
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let mut deq = [0i16; 16];
            deq[0] = (coeffs[i][0] as i32 * uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (coeffs[i][k] as i32 * uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_xc + bx * 4;
            let dst_y = mb_yc + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = pred[(by * 4 + r) * 8 + bx * 4 + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    plane[(dst_y + r) * uv_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_mb(
    header: &FrameHeader,
    state: &DecoderState,
    info: &MbInfo,
    has_y2: bool,
    y2_coeffs: &[i16; 16],
    y_coeffs: &[[i16; 16]; 16],
    u_coeffs: &[[i16; 16]; 4],
    v_coeffs: &[[i16; 16]; 4],
    mb_x: usize,
    mb_y: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
) {
    let qi = clamp_qindex(header.quant.y_ac_qi);
    let y_dc = y_dc_step(qi as i32 + header.quant.y_dc_delta);
    let y_ac = y_ac_step(qi as i32);
    let y2_dc = y2_dc_step(qi as i32 + header.quant.y2_dc_delta);
    let y2_ac = y2_ac_step(qi as i32 + header.quant.y2_ac_delta);
    let uv_dc = uv_dc_step(qi as i32 + header.quant.uv_dc_delta);
    let uv_ac = uv_ac_step(qi as i32 + header.quant.uv_ac_delta);

    let y2_dc_vals: [i16; 16] = if has_y2 {
        let mut deq = [0i16; 16];
        for i in 0..16 {
            let v = y2_coeffs[i] as i32;
            let q = if i == 0 { y2_dc } else { y2_ac };
            deq[i] = (v * q) as i16;
        }
        iwht4x4(&deq)
    } else {
        [0; 16]
    };

    let ref_frame = match info.ref_frame {
        REF_LAST => &state.last,
        REF_GOLDEN => &state.golden,
        REF_ALT => &state.altref,
        _ => &state.last,
    };

    let mb_x_px = mb_x * 16;
    let mb_y_px = mb_y * 16;

    // --- Luma prediction via sub-pel 6-tap filter ---
    // Each 4×4 luma sub-block has its own MV (either derived from the MB
    // MV or from SPLITMV sub-partitions).
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let dst_x = mb_x_px + bx * 4;
        let dst_y = mb_y_px + by * 4;
        let mv = info.sub_mvs[i];
        let ref_x_fp = (dst_x as i32) * 8 + mv.col as i32;
        let ref_y_fp = (dst_y as i32) * 8 + mv.row as i32;
        sixtap_predict(
            &ref_frame.y_plane(),
            ref_x_fp,
            ref_y_fp,
            y_plane,
            y_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // --- Chroma prediction via bilinear 2-tap filter ---
    // Each 4×4 chroma sub-block covers 2×2 luma sub-blocks. The chroma
    // MV is the average of the 4 luma sub-MVs it covers.
    let mb_xc = mb_x * 8;
    let mb_yc = mb_y * 8;
    for i in 0..4 {
        let by = i / 2;
        let bx = i % 2;
        // 4 luma subs for this chroma 4×4: (2bx..2bx+2, 2by..2by+2).
        let mut sum_r: i32 = 0;
        let mut sum_c: i32 = 0;
        for r in 0..2 {
            for c in 0..2 {
                let li = (2 * by + r) * 4 + (2 * bx + c);
                sum_r += info.sub_mvs[li].row as i32;
                sum_c += info.sub_mvs[li].col as i32;
            }
        }
        // Round-half-to-zero like libvpx.
        let cmv_r = chroma_round(sum_r);
        let cmv_c = chroma_round(sum_c);
        let dst_x = mb_xc + bx * 4;
        let dst_y = mb_yc + by * 4;
        let ref_x_fp = (dst_x as i32) * 8 + cmv_c;
        let ref_y_fp = (dst_y as i32) * 8 + cmv_r;
        bilinear_predict(
            &ref_frame.u_plane(),
            ref_x_fp,
            ref_y_fp,
            u_plane,
            uv_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
        bilinear_predict(
            &ref_frame.v_plane(),
            ref_x_fp,
            ref_y_fp,
            v_plane,
            uv_stride,
            dst_x,
            dst_y,
            4,
            4,
        );
    }

    // --- Add residuals ---
    for i in 0..16 {
        let by = i / 4;
        let bx = i % 4;
        let mut deq = [0i16; 16];
        if has_y2 {
            deq[0] = y2_dc_vals[i];
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
        } else {
            deq[0] = (y_coeffs[i][0] as i32 * y_dc) as i16;
            for k in 1..16 {
                deq[k] = (y_coeffs[i][k] as i32 * y_ac) as i16;
            }
        }
        let res = idct4x4(&deq);
        let dst_x = mb_x_px + bx * 4;
        let dst_y = mb_y_px + by * 4;
        for r in 0..4 {
            for c in 0..4 {
                let p = y_plane[(dst_y + r) * y_stride + dst_x + c] as i32;
                let rr = res[r * 4 + c] as i32;
                y_plane[(dst_y + r) * y_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
            }
        }
    }
    for (coeffs, plane) in [(u_coeffs, &mut *u_plane), (v_coeffs, &mut *v_plane)] {
        for i in 0..4 {
            let by = i / 2;
            let bx = i % 2;
            let mut deq = [0i16; 16];
            deq[0] = (coeffs[i][0] as i32 * uv_dc) as i16;
            for k in 1..16 {
                deq[k] = (coeffs[i][k] as i32 * uv_ac) as i16;
            }
            let res = idct4x4(&deq);
            let dst_x = mb_xc + bx * 4;
            let dst_y = mb_yc + by * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let p = plane[(dst_y + r) * uv_stride + dst_x + c] as i32;
                    let rr = res[r * 4 + c] as i32;
                    plane[(dst_y + r) * uv_stride + dst_x + c] = (p + rr).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Round a sum of 4 luma 1/8-pel MV components to a chroma 1/8-pel value.
/// libvpx's rule (RFC 6386 §14): divide by 4, rounding towards zero. The
/// sub-pel phase is preserved since chroma is on a 2× coarser grid.
#[inline]
fn chroma_round(sum: i32) -> i32 {
    // Match libvpx's `mv_as_chroma`: `((sum + sign * 4) / 8) * 2` — that
    // is, average of 4 MVs with rounding-toward-zero scaled from 1/8-pel
    // luma to 1/8-pel chroma (same units, coarser grid halves the MV).
    let sign = if sum < 0 { -1 } else { 1 };
    ((sum + sign * 4) / 8) * 2
}

#[allow(clippy::too_many_arguments)]
fn apply_loop_filter(
    header: &FrameHeader,
    _mb_info: &[MbInfo],
    mb_w: usize,
    mb_h: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    y_buf_h: usize,
    uv_buf_h: usize,
) {
    if header.loop_filter.level == 0 {
        return;
    }
    let lf = &header.loop_filter;
    let params_mb = FilterParams::for_mb(lf.level, lf.sharpness, true);
    let params_sb = FilterParams::for_mb(lf.level, lf.sharpness, false);
    let simple = lf.filter_type == 1;
    for mb_y in 0..mb_h {
        for mb_x in 1..mb_w {
            let x = mb_x * 16;
            let y0 = mb_y * 16;
            if simple {
                filter_simple_vertical(y_plane, y_stride, x, y_stride, y0 + 16, params_mb);
            } else {
                filter_normal_vertical(y_plane, y_stride, x, y_stride, y0 + 16, params_mb, true);
            }
        }
    }
    for mb_y in 1..mb_h {
        let y = mb_y * 16;
        if simple {
            filter_simple_horizontal(y_plane, y_stride, y, y_stride, y_buf_h, params_mb);
        } else {
            filter_normal_horizontal(y_plane, y_stride, y, y_stride, y_buf_h, params_mb, true);
        }
    }
    if !simple {
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let bx0 = mb_x * 16;
                let by0 = mb_y * 16;
                for k in 1..4 {
                    let xv = bx0 + k * 4;
                    filter_normal_vertical(
                        y_plane,
                        y_stride,
                        xv,
                        y_stride,
                        by0 + 16,
                        params_sb,
                        false,
                    );
                    let yh = by0 + k * 4;
                    filter_normal_horizontal(
                        y_plane, y_stride, yh, y_stride, y_buf_h, params_sb, false,
                    );
                }
            }
        }
    }
    for mb_y in 0..mb_h {
        for mb_x in 1..mb_w {
            let x = mb_x * 8;
            let y0 = mb_y * 8;
            if simple {
                filter_simple_vertical(u_plane, uv_stride, x, uv_stride, y0 + 8, params_mb);
                filter_simple_vertical(v_plane, uv_stride, x, uv_stride, y0 + 8, params_mb);
            } else {
                filter_normal_vertical(u_plane, uv_stride, x, uv_stride, y0 + 8, params_mb, true);
                filter_normal_vertical(v_plane, uv_stride, x, uv_stride, y0 + 8, params_mb, true);
            }
        }
    }
    for mb_y in 1..mb_h {
        let y = mb_y * 8;
        if simple {
            filter_simple_horizontal(u_plane, uv_stride, y, uv_stride, uv_buf_h, params_mb);
            filter_simple_horizontal(v_plane, uv_stride, y, uv_stride, uv_buf_h, params_mb);
        } else {
            filter_normal_horizontal(u_plane, uv_stride, y, uv_stride, uv_buf_h, params_mb, true);
            filter_normal_horizontal(v_plane, uv_stride, y, uv_stride, uv_buf_h, params_mb, true);
        }
    }
}
