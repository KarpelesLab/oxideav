//! MPEG-4 Part 2 video decoder.
//!
//! Scope:
//! * Parses Visual Object Sequence, Visual Object, Video Object Layer and
//!   Video Object Plane headers from a stream of annexed start codes.
//! * Populates `CodecParameters` from the VOL.
//! * **Decodes I-VOPs** — full intra path (DC+AC VLCs, AC/DC prediction,
//!   H.263 + MPEG-4 dequantisation, IDCT).
//! * **Decodes P-VOPs** — half-pel motion compensation, 1MV / 4MV modes,
//!   inter texture decode, MV-median prediction, and skipped MBs.
//! * Holds one reference picture (`prev_ref`) — refreshed by each I-VOP and
//!   each newly-reconstructed P-VOP.
//!
//! Out of scope (returns `Error::Unsupported`):
//! * B-VOPs, S-VOPs (sprites), GMC.
//! * Quarter-pel motion (`quarter_sample` rejected at VOL parse time).
//! * Interlaced field coding, scalability, data partitioning.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame,
};

use crate::bitreader::BitReader;
use crate::headers::vol::{parse_vol, VideoObjectLayer};
use crate::headers::vop::{parse_vop, VideoObjectPlane, VopCodingType};
use crate::headers::vos::{parse_visual_object, parse_vos, VisualObject, VisualObjectSequence};
use crate::inter::{decode_p_mb, MvGrid};
use crate::mb::{decode_intra_mb, IVopPicture, PredGrid};
use crate::resync::{try_consume_resync_marker_after, ResyncResult};
use crate::start_codes::{
    self, is_video_object, is_video_object_layer, GOV_START_CODE, USER_DATA_START_CODE,
    VIDEO_SESSION_ERROR_CODE, VISUAL_OBJECT_START_CODE, VOP_START_CODE, VOS_END_CODE,
    VOS_START_CODE,
};

/// Factory for the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mpeg4VideoDecoder::new(params.codec_id.clone())))
}

pub struct Mpeg4VideoDecoder {
    codec_id: CodecId,
    buffer: Vec<u8>,
    vos: Option<VisualObjectSequence>,
    vo: Option<VisualObject>,
    vol: Option<VideoObjectLayer>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
    /// Last decoded reference picture — used as `prev_ref` for the next
    /// P-VOP. Refreshed by each I-VOP and each P-VOP.
    prev_ref: Option<IVopPicture>,
}

impl Mpeg4VideoDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            vos: None,
            vo: None,
            vol: None,
            ready_frames: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 90_000),
            eof: false,
            prev_ref: None,
        }
    }

    pub fn vol(&self) -> Option<&VideoObjectLayer> {
        self.vol.as_ref()
    }

    /// Walk start codes in the buffer, updating header state and dispatching
    /// VOPs. I-VOPs and P-VOPs are decoded; B/S VOPs return `Unsupported`.
    fn process(&mut self) -> Result<()> {
        let data = std::mem::take(&mut self.buffer);
        let markers: Vec<(usize, u8)> = start_codes::iter_start_codes(&data).collect();
        for (idx, (pos, code)) in markers.iter().enumerate() {
            let payload_end = markers.get(idx + 1).map(|(p, _)| *p).unwrap_or(data.len());
            let payload_start = *pos + 4;
            if payload_start > data.len() {
                break;
            }
            let payload = &data[payload_start..payload_end];

            match *code {
                VOS_START_CODE => {
                    let mut br = BitReader::new(payload);
                    self.vos = Some(parse_vos(&mut br)?);
                }
                VISUAL_OBJECT_START_CODE => {
                    let mut br = BitReader::new(payload);
                    self.vo = Some(parse_visual_object(&mut br)?);
                }
                c if is_video_object(c) => {
                    // Video object start code — no payload of interest.
                }
                c if is_video_object_layer(c) => {
                    let mut br = BitReader::new(payload);
                    self.vol = Some(parse_vol(&mut br)?);
                }
                GOV_START_CODE | USER_DATA_START_CODE | VIDEO_SESSION_ERROR_CODE | VOS_END_CODE => {
                    // Not yet used by this decoder — skip.
                }
                VOP_START_CODE => {
                    let Some(vol) = self.vol.clone() else {
                        return Err(Error::invalid("mpeg4: VOP before VOL"));
                    };
                    let mut br = BitReader::new(payload);
                    let vop = parse_vop(&mut br, &vol)?;
                    self.handle_vop(&vol, &vop, &mut br)?;
                }
                _ => {
                    // Unknown marker — skip.
                }
            }
        }
        Ok(())
    }

    fn handle_vop(
        &mut self,
        vol: &VideoObjectLayer,
        vop: &VideoObjectPlane,
        br: &mut BitReader<'_>,
    ) -> Result<()> {
        if !vop.vop_coded {
            // "Not coded" VOP — repeat the previous frame as a placeholder.
            // For a simple decoder we just don't emit anything; downstream
            // can hold the previous frame.
            return Ok(());
        }
        match vop.vop_coding_type {
            VopCodingType::I => {
                let pic = decode_ivop_pic(vol, vop, br)?;
                let frame = pic_to_video_frame(vol, &pic, self.pending_pts, self.pending_tb);
                self.prev_ref = Some(pic);
                self.ready_frames.push_back(frame);
                Ok(())
            }
            VopCodingType::P => {
                let Some(reference) = self.prev_ref.as_ref() else {
                    return Err(Error::invalid("mpeg4 P-VOP: no reference frame yet"));
                };
                let pic = decode_pvop_pic(vol, vop, br, reference)?;
                let frame = pic_to_video_frame(vol, &pic, self.pending_pts, self.pending_tb);
                self.prev_ref = Some(pic);
                self.ready_frames.push_back(frame);
                Ok(())
            }
            VopCodingType::B => Err(Error::unsupported(
                "mpeg4 B frames: follow-up (bidirectional MC)",
            )),
            VopCodingType::S => Err(Error::unsupported("mpeg4 S-VOP (sprite): out of scope")),
        }
    }
}

/// Decode an I-VOP and return the reconstructed `IVopPicture`.
pub fn decode_ivop_pic(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
) -> Result<IVopPicture> {
    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut grid = PredGrid::new(mb_w, mb_h);

    let mb_total = (mb_w * mb_h) as u32;
    let mut quant = vop.vop_quant;
    let mut mb_idx: u32 = 0;
    while (mb_idx as usize) < mb_w * mb_h {
        let mb_x = (mb_idx as usize) % mb_w;
        let mb_y = (mb_idx as usize) / mb_w;
        quant =
            decode_intra_mb(br, mb_x, mb_y, quant, vol, vop, &mut pic, &mut grid).map_err(|e| {
                oxideav_core::Error::invalid(format!("mpeg4 I-VOP MB ({mb_x},{mb_y}): {e}"))
            })?;
        mb_idx += 1;
        if (mb_idx as usize) >= mb_w * mb_h {
            break;
        }
        match try_consume_resync_marker_after(br, vol, vop, mb_total, mb_idx)? {
            ResyncResult::None => {}
            ResyncResult::Resync { mb_num, new_quant } => {
                if mb_num < mb_idx || mb_num >= mb_total {
                    return Err(Error::invalid(format!(
                        "mpeg4 I-VOP: resync mb_num={mb_num} not at or after current={mb_idx}"
                    )));
                }
                grid = PredGrid::new(mb_w, mb_h);
                if new_quant != 0 {
                    quant = new_quant;
                }
                mb_idx = mb_num;
            }
        }
    }
    Ok(pic)
}

/// Decode a P-VOP relative to `reference` and return the reconstructed
/// `IVopPicture`.
pub fn decode_pvop_pic(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
    reference: &IVopPicture,
) -> Result<IVopPicture> {
    let mb_w = vol.mb_width() as usize;
    let mb_h = vol.mb_height() as usize;
    let mut pic = IVopPicture::new(vol.width as usize, vol.height as usize);
    let mut pred_grid = PredGrid::new(mb_w, mb_h);
    let mut mv_grid = MvGrid::new(mb_w, mb_h);

    let mb_total = (mb_w * mb_h) as u32;
    let mut quant = vop.vop_quant;
    let mut mb_idx: u32 = 0;
    let mut slice_first_mb = (0usize, 0usize);
    while (mb_idx as usize) < mb_w * mb_h {
        let mb_x = (mb_idx as usize) % mb_w;
        let mb_y = (mb_idx as usize) / mb_w;
        quant = decode_p_mb(
            br,
            mb_x,
            mb_y,
            quant,
            vol,
            vop,
            &mut pic,
            &mut pred_grid,
            &mut mv_grid,
            reference,
            slice_first_mb,
        )
        .map_err(|e| {
            oxideav_core::Error::invalid(format!("mpeg4 P-VOP MB ({mb_x},{mb_y}): {e}"))
        })?;
        mb_idx += 1;
        if (mb_idx as usize) >= mb_w * mb_h {
            break;
        }
        match try_consume_resync_marker_after(br, vol, vop, mb_total, mb_idx)? {
            ResyncResult::None => {}
            ResyncResult::Resync { mb_num, new_quant } => {
                if mb_num < mb_idx || mb_num >= mb_total {
                    return Err(Error::invalid(format!(
                        "mpeg4 P-VOP: resync mb_num={mb_num} not at or after current={mb_idx}"
                    )));
                }
                // Reset prediction state across packet boundaries.
                pred_grid = PredGrid::new(mb_w, mb_h);
                mv_grid = MvGrid::new(mb_w, mb_h);
                if new_quant != 0 {
                    quant = new_quant;
                }
                mb_idx = mb_num;
                slice_first_mb = ((mb_idx as usize) % mb_w, (mb_idx as usize) / mb_w);
            }
        }
    }
    Ok(pic)
}

/// Build a stride-packed YUV420P `VideoFrame` from an `IVopPicture`.
pub fn pic_to_video_frame(
    vol: &VideoObjectLayer,
    pic: &IVopPicture,
    pts: Option<i64>,
    tb: TimeBase,
) -> VideoFrame {
    let w = vol.width as usize;
    let h = vol.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        y[row * w..row * w + w].copy_from_slice(&pic.y[row * pic.y_stride..row * pic.y_stride + w]);
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        cb[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cb[row * pic.c_stride..row * pic.c_stride + cw]);
        cr[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cr[row * pic.c_stride..row * pic.c_stride + cw]);
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w as u32,
        height: h as u32,
        pts,
        time_base: tb,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

/// Decode a single I-VOP body and return a `VideoFrame`. Kept for backwards
/// compatibility with existing tests.
pub fn decode_ivop(
    vol: &VideoObjectLayer,
    vop: &VideoObjectPlane,
    br: &mut BitReader<'_>,
    pts: Option<i64>,
    tb: TimeBase,
) -> Result<VideoFrame> {
    let pic = decode_ivop_pic(vol, vop, br)?;
    Ok(pic_to_video_frame(vol, &pic, pts, tb))
}

impl Decoder for Mpeg4VideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.process()
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
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

    fn reset(&mut self) -> Result<()> {
        // Drop the NAL-accumulator buffer, the ready-frame queue, and the
        // stored reference picture (used as `prev_ref` for the next P-VOP).
        // VOS / VO / VOL are stream-level config extracted once from the
        // bitstream — kept so we can decode subsequent VOPs without
        // waiting for the headers to reappear.
        self.buffer.clear();
        self.ready_frames.clear();
        self.pending_pts = None;
        self.eof = false;
        self.prev_ref = None;
        Ok(())
    }
}

/// Build a `CodecParameters` from a VOL.
pub fn codec_parameters_from_vol(vol: &VideoObjectLayer) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(vol.width);
    params.height = Some(vol.height);
    let (num, den) = vol.frame_rate();
    if num > 0 && den > 0 {
        params.frame_rate = Some(Rational::new(num, den));
    }
    params
}
