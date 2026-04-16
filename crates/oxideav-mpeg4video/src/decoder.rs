//! MPEG-4 Part 2 video decoder.
//!
//! Scope this session:
//! * Parses Visual Object Sequence, Visual Object, Video Object Layer and
//!   Video Object Plane headers from a stream of annexed start codes.
//! * Populates `CodecParameters` from the VOL.
//! * Rejects inter / sprite / quarter-pel / interlaced / scalable streams up
//!   front with `Error::Unsupported`, so callers know the follow-up still
//!   owes them a proper decoder.
//! * Returns `Error::Unsupported("mpeg4 I-VOP MB decode: follow-up")` when a
//!   caller sends a VOP for decode — the full macroblock / AC-DC prediction
//!   / IDCT path is the planned follow-up.
//!
//! The decoder still records the parsed headers so `codec_parameters()` and
//! header-sniffing tests exercise the correct code path.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, Result, TimeBase, VideoFrame,
};

use crate::bitreader::BitReader;
use crate::headers::vol::{parse_vol, VideoObjectLayer};
use crate::headers::vop::{parse_vop, VideoObjectPlane, VopCodingType};
use crate::headers::vos::{parse_visual_object, parse_vos, VisualObject, VisualObjectSequence};
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
        }
    }

    pub fn vol(&self) -> Option<&VideoObjectLayer> {
        self.vol.as_ref()
    }

    /// Walk start codes in the buffer, updating header state and dispatching
    /// VOPs. Only runs header parsing for now — VOP bodies propagate as
    /// `Unsupported` to the caller.
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
                    let Some(vol) = self.vol.as_ref() else {
                        return Err(Error::invalid("mpeg4: VOP before VOL"));
                    };
                    let mut br = BitReader::new(payload);
                    let vop = parse_vop(&mut br, vol)?;
                    self.handle_vop(vop)?;
                }
                _ => {
                    // Unknown marker — skip.
                }
            }
        }
        Ok(())
    }

    fn handle_vop(&mut self, vop: VideoObjectPlane) -> Result<()> {
        if !vop.vop_coded {
            return Ok(());
        }
        match vop.vop_coding_type {
            VopCodingType::I => Err(Error::unsupported(
                "mpeg4 I-VOP MB decode: follow-up (headers parse OK; \
                 texture VLC + AC/DC prediction + IDCT still to land)",
            )),
            VopCodingType::P => Err(Error::unsupported(
                "mpeg4 P frames: follow-up (motion compensation + inter MBs)",
            )),
            VopCodingType::B => Err(Error::unsupported(
                "mpeg4 B frames: follow-up (bidirectional MC)",
            )),
            VopCodingType::S => Err(Error::unsupported("mpeg4 S-VOP (sprite): out of scope")),
        }
    }
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
}

/// Build a `CodecParameters` from a VOL. Useful for demuxer plumbing that
/// wants to expose picture dimensions + frame rate before handing the first
/// packet to a decoder instance.
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
