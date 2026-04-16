//! MPEG-1 video decoder driving the layered parse (sequence → GOP → picture
//! → slice → MB → block).
//!
//! Current milestone: decodes I-frames only. P and B pictures cause the
//! decoder to return an `Unsupported` error at the first non-intra macroblock.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, Result, TimeBase, VideoFrame,
};

use crate::bitreader::BitReader;
use crate::headers::{
    frame_rate_for_code, parse_gop_header, parse_picture_header, parse_sequence_header,
    PictureHeader, PictureType, SequenceHeader,
};
use crate::mb::decode_slice;
use crate::picture::PictureBuffer;
use crate::start_codes::{
    self, EXTENSION_START_CODE, GROUP_START_CODE, SEQUENCE_END_CODE, SEQUENCE_ERROR_CODE,
    SEQUENCE_HEADER_CODE, USER_DATA_START_CODE,
};

/// Factory for the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mpeg1VideoDecoder::new(params.codec_id.clone())))
}

pub struct Mpeg1VideoDecoder {
    codec_id: CodecId,
    // Persistent-across-packet buffer so callers can stream one packet at a
    // time; we keep consuming bytes as complete pictures are produced.
    buffer: Vec<u8>,
    seq_header: Option<SequenceHeader>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
}

impl Mpeg1VideoDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            seq_header: None,
            ready_frames: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 90_000),
            eof: false,
        }
    }

    /// Process as many complete pictures as possible from the buffered stream.
    fn try_decode(&mut self) -> Result<()> {
        loop {
            let Some(picture_end) = find_picture_end(&self.buffer) else {
                return Ok(());
            };
            // Decode everything from buffer[0..picture_end] (one picture).
            // Carve out the slice so we don't hold `&mut self` while reading.
            let (head, tail) = self.buffer.split_at(picture_end);
            let decoded = decode_one_picture(head, &mut self.seq_header)?;
            if let Some(mut frame) = decoded {
                frame.pts = self.pending_pts;
                frame.time_base = self.pending_tb;
                self.ready_frames.push_back(frame);
            }
            let remaining = tail.to_vec();
            self.buffer = remaining;
        }
    }
}

impl Decoder for Mpeg1VideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.try_decode()
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            // On EOF, try to decode whatever is left.
            if !self.buffer.is_empty() {
                // Append trailing start code to force the last picture out.
                let sentinel = [0u8, 0, 1, SEQUENCE_END_CODE];
                self.buffer.extend_from_slice(&sentinel);
                let _ = self.try_decode();
                if let Some(f) = self.ready_frames.pop_front() {
                    return Ok(Frame::Video(f));
                }
            }
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

/// Locate the end position of the next picture in `buf` — i.e. the offset of
/// the start code (picture/gop/sequence_end/sequence_header) that immediately
/// follows the body of a picture. Returns `None` if no complete picture is
/// present.
fn find_picture_end(buf: &[u8]) -> Option<usize> {
    // We need: first, a picture_start_code (0x00). Then the next start code
    // that is one of (picture, gop, sequence_header, sequence_end) marks
    // the end of this picture.
    let iter = start_codes::iter_start_codes(buf);
    let mut picture_seen = false;
    for (pos, code) in iter {
        if !picture_seen {
            if code == start_codes::PICTURE_START_CODE {
                picture_seen = true;
            }
            continue;
        }
        match code {
            start_codes::PICTURE_START_CODE
            | GROUP_START_CODE
            | SEQUENCE_HEADER_CODE
            | SEQUENCE_END_CODE => return Some(pos),
            _ => continue,
        }
    }
    None
}

/// Decode a single picture from `data`, which is expected to contain exactly
/// one picture (possibly preceded by sequence / GOP headers). `seq_header` is
/// updated in-place when a sequence header is encountered.
fn decode_one_picture(
    data: &[u8],
    seq_header: &mut Option<SequenceHeader>,
) -> Result<Option<VideoFrame>> {
    let mut pic_header: Option<PictureHeader> = None;
    let mut picture: Option<PictureBuffer> = None;
    let mut dc_pred = [128i32, 128, 128];
    let mut sequence_was_just_parsed = false;

    let markers: Vec<(usize, u8)> = start_codes::iter_start_codes(data).collect();
    for (i, (pos, code)) in markers.iter().enumerate() {
        // Determine where this marker's payload ends (start of next marker).
        let payload_end = markers.get(i + 1).map(|(p, _)| *p).unwrap_or(data.len());
        let payload_start = pos + 4; // skip 0x000001XX
        if payload_start > data.len() {
            break;
        }
        let payload = &data[payload_start..payload_end];

        match *code {
            SEQUENCE_HEADER_CODE => {
                let mut br = BitReader::new(payload);
                let sh = parse_sequence_header(&mut br)?;
                *seq_header = Some(sh);
                sequence_was_just_parsed = true;
            }
            EXTENSION_START_CODE | USER_DATA_START_CODE => {
                // Ignored.
            }
            GROUP_START_CODE => {
                let mut br = BitReader::new(payload);
                let _gop = parse_gop_header(&mut br)?;
            }
            start_codes::PICTURE_START_CODE => {
                let mut br = BitReader::new(payload);
                let ph = parse_picture_header(&mut br)?;
                pic_header = Some(ph.clone());
                let Some(seq) = seq_header.as_ref() else {
                    return Err(Error::invalid("picture before sequence header"));
                };
                picture = Some(PictureBuffer::new(
                    seq.horizontal_size as usize,
                    seq.vertical_size as usize,
                    ph.picture_type,
                    ph.temporal_reference,
                ));
            }
            SEQUENCE_END_CODE => break,
            SEQUENCE_ERROR_CODE => continue,
            c if start_codes::is_slice(c) => {
                let Some(seq) = seq_header.as_ref() else {
                    return Err(Error::invalid("slice before sequence header"));
                };
                let Some(ph) = pic_header.as_ref() else {
                    return Err(Error::invalid("slice before picture header"));
                };
                let Some(pic) = picture.as_mut() else {
                    return Err(Error::invalid("slice: no picture buffer"));
                };
                let mut br = BitReader::new(payload);
                decode_slice(&mut br, c, seq, ph.picture_type, pic, &mut dc_pred)?;
            }
            _ => {
                // Unknown marker — skip payload.
            }
        }
    }

    let _ = sequence_was_just_parsed;

    let Some(pic) = picture else {
        return Ok(None);
    };
    // For milestone #2 we only emit I-frames. Drop P/B quietly until the next
    // milestone lands motion compensation.
    if !matches!(pic.picture_type, PictureType::I) {
        return Ok(None);
    }

    Ok(Some(pic.to_video_frame(None, TimeBase::new(1, 90_000))))
}

/// Build a `CodecParameters` from a sequence header (used by demuxers).
pub fn codec_parameters_from_sequence_header(sh: &SequenceHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new("mpeg1video"));
    params.width = Some(sh.horizontal_size);
    params.height = Some(sh.vertical_size);
    if let Some((n, d)) = frame_rate_for_code(sh.frame_rate_code) {
        params.frame_rate = Some(Rational::new(n, d));
    }
    if sh.bit_rate != 0 && sh.bit_rate != 0x3FFFF {
        // `bit_rate` is in units of 400 bits/s.
        params.bit_rate = Some(sh.bit_rate as u64 * 400);
    }
    params
}
