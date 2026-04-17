//! MPEG-1 video decoder driving the layered parse (sequence → GOP → picture
//! → slice → MB → block).

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, Result, TimeBase, VideoFrame,
};

use crate::bitreader::BitReader;
use crate::headers::{
    frame_rate_for_code, parse_gop_header, parse_picture_header, parse_sequence_header, GopHeader,
    PictureHeader, PictureType, SequenceHeader,
};
use crate::mb::decode_slice;
use crate::picture::{PictureBuffer, ReferenceManager};
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
    buffer: Vec<u8>,
    seq_header: Option<SequenceHeader>,
    gop_header: Option<GopHeader>,
    /// GOP-start PTS anchor: temporal_reference 0 in the current GOP maps to
    /// this PTS.
    gop_anchor_pts: Option<i64>,
    /// Highest temporal_reference seen so far in the current GOP, used to
    /// roll over `gop_anchor_pts` at the next GOP boundary when timecode
    /// information is unavailable.
    gop_max_tr: u16,
    /// Frame duration in time_base units for temporal_reference-based PTS
    /// reconstruction.
    frame_duration: Option<i64>,
    refs: ReferenceManager,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    time_base: TimeBase,
    eof: bool,
}

impl Mpeg1VideoDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        let tb = TimeBase::new(1, 90_000);
        Self {
            codec_id,
            buffer: Vec::new(),
            seq_header: None,
            gop_header: None,
            gop_anchor_pts: None,
            gop_max_tr: 0,
            frame_duration: None,
            refs: ReferenceManager::new(),
            ready_frames: VecDeque::new(),
            pending_pts: None,
            pending_tb: tb,
            time_base: tb,
            eof: false,
        }
    }

    /// Process as many complete pictures as possible from the buffered stream.
    fn try_decode(&mut self) -> Result<()> {
        loop {
            let Some(picture_end) = find_picture_end(&self.buffer) else {
                return Ok(());
            };
            let (head, _tail) = self.buffer.split_at(picture_end);
            let head = head.to_vec();
            self.decode_one_picture(&head)?;
            self.buffer.drain(..picture_end);
        }
    }

    fn decode_one_picture(&mut self, data: &[u8]) -> Result<()> {
        let mut pic_header: Option<PictureHeader> = None;
        let mut picture: Option<PictureBuffer> = None;
        let mut sequence_was_just_parsed = false;

        let markers: Vec<(usize, u8)> = start_codes::iter_start_codes(data).collect();
        for (i, (pos, code)) in markers.iter().enumerate() {
            let payload_end = markers.get(i + 1).map(|(p, _)| *p).unwrap_or(data.len());
            let payload_start = pos + 4;
            if payload_start > data.len() {
                break;
            }
            let payload = &data[payload_start..payload_end];

            match *code {
                SEQUENCE_HEADER_CODE => {
                    let mut br = BitReader::new(payload);
                    let sh = parse_sequence_header(&mut br)?;
                    // Derive frame duration from the sequence frame rate.
                    if let Some((num, den)) = frame_rate_for_code(sh.frame_rate_code) {
                        // time_base = 1/90000 by default. duration_in_ticks
                        // = (den / num) seconds × (1 / time_base) ticks/sec.
                        let r = self.time_base.as_rational();
                        // Compute (den / num) / (r.num / r.den) = den*r.den /
                        // (num*r.num).
                        let ticks = (den as i128 * r.den as i128) / (num as i128 * r.num as i128);
                        self.frame_duration = Some(ticks as i64);
                    }
                    self.seq_header = Some(sh);
                    sequence_was_just_parsed = true;
                }
                EXTENSION_START_CODE | USER_DATA_START_CODE => {}
                GROUP_START_CODE => {
                    let mut br = BitReader::new(payload);
                    let gop = parse_gop_header(&mut br)?;
                    // Advance the GOP anchor: first GOP uses the demuxer-
                    // supplied PTS, subsequent GOPs bump by
                    // `(gop_max_tr + 1) * frame_duration`.
                    if let Some(anchor) = self.gop_anchor_pts {
                        if let Some(dur) = self.frame_duration {
                            self.gop_anchor_pts = Some(anchor + (self.gop_max_tr as i64 + 1) * dur);
                        }
                    } else {
                        self.gop_anchor_pts = self.pending_pts.or(Some(0));
                    }
                    self.gop_max_tr = 0;
                    self.gop_header = Some(gop);
                }
                start_codes::PICTURE_START_CODE => {
                    let mut br = BitReader::new(payload);
                    let ph = parse_picture_header(&mut br)?;
                    let Some(seq) = self.seq_header.as_ref() else {
                        return Err(Error::invalid("picture before sequence header"));
                    };
                    if ph.temporal_reference > self.gop_max_tr {
                        self.gop_max_tr = ph.temporal_reference;
                    }
                    pic_header = Some(ph.clone());
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
                    let Some(seq) = self.seq_header.as_ref() else {
                        return Err(Error::invalid("slice before sequence header"));
                    };
                    let Some(ph) = pic_header.as_ref() else {
                        return Err(Error::invalid("slice before picture header"));
                    };
                    let Some(pic) = picture.as_mut() else {
                        return Err(Error::invalid("slice: no picture buffer"));
                    };
                    let mut br = BitReader::new(payload);
                    // References:
                    //   P-frame forward ref   = most-recent I/P anchor   = next_ref
                    //   B-frame forward ref   = older I/P anchor         = prev_ref
                    //   B-frame backward ref  = most-recent I/P anchor   = next_ref
                    let (fwd_ref, bwd_ref) = match ph.picture_type {
                        PictureType::P => (self.refs.backward(), None),
                        PictureType::B => (self.refs.forward(), self.refs.backward()),
                        _ => (None, None),
                    };
                    decode_slice(&mut br, c, seq, ph, pic, fwd_ref, bwd_ref)?;
                }
                _ => {}
            }
        }

        let _ = sequence_was_just_parsed;

        let Some(mut pic) = picture else {
            return Ok(());
        };

        // Compute display PTS at decode time so it's insensitive to later
        // GOP anchor roll-overs.
        pic.display_pts = self.compute_display_pts(pic.temporal_reference);

        match pic.picture_type {
            PictureType::I | PictureType::P => {
                // Rotate reference pictures. `push_anchor` returns the
                // previous `next_ref` (now `prev_ref`) as "ready for
                // display" — all B-pictures that depend on it as backward
                // anchor have already been decoded & emitted.
                if let Some(displaced) = self.refs.push_anchor(pic) {
                    self.ready_frames
                        .push_back(displaced.to_video_frame(displaced.display_pts, self.time_base));
                }
            }
            PictureType::B => {
                // B-pictures are emitted directly in decode order — their
                // display PTS (computed above) captures the required
                // ordering.
                let pts = pic.display_pts;
                self.ready_frames
                    .push_back(pic.to_video_frame(pts, self.time_base));
            }
            PictureType::D => {
                return Err(Error::unsupported("D-picture not supported"));
            }
        }

        Ok(())
    }

    fn compute_display_pts(&self, temporal_reference: u16) -> Option<i64> {
        // For packet-attached PTS the simplest correct model is:
        //   display_pts = gop_anchor_pts + temporal_reference * frame_duration
        // where `gop_anchor_pts` is the PTS of tr=0 in the current GOP.
        match (self.gop_anchor_pts, self.frame_duration) {
            (Some(anchor), Some(dur)) => Some(anchor + temporal_reference as i64 * dur),
            _ => self.pending_pts,
        }
    }

    fn flush_remaining_refs(&mut self) {
        for pic in self.refs.drain() {
            self.ready_frames
                .push_back(pic.to_video_frame(pic.display_pts, self.time_base));
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
        self.time_base = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.try_decode()
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            if !self.buffer.is_empty() {
                let sentinel = [0u8, 0, 1, SEQUENCE_END_CODE];
                self.buffer.extend_from_slice(&sentinel);
                let _ = self.try_decode();
            }
            self.flush_remaining_refs();
            if let Some(f) = self.ready_frames.pop_front() {
                return Ok(Frame::Video(f));
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

    fn reset(&mut self) -> Result<()> {
        // Wipe all decode state that must not survive a seek:
        //   * the NAL-accumulator buffer,
        //   * GOP header + anchor PTS + highest temporal_reference,
        //   * the reference-picture manager (forward+backward I/P anchors),
        //   * the ready_frames queue.
        // `seq_header` is stream-level config — if the container has one
        // it won't resend it after a seek, and it may still be needed to
        // decode subsequent pictures. Frame duration is derived from the
        // sequence header so it stays too.
        self.buffer.clear();
        self.gop_header = None;
        self.gop_anchor_pts = None;
        self.gop_max_tr = 0;
        self.refs = ReferenceManager::new();
        self.ready_frames.clear();
        self.pending_pts = None;
        self.eof = false;
        Ok(())
    }
}

/// Locate the end position of the next picture in `buf`.
fn find_picture_end(buf: &[u8]) -> Option<usize> {
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

/// Build a `CodecParameters` from a sequence header (used by demuxers).
pub fn codec_parameters_from_sequence_header(sh: &SequenceHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new("mpeg1video"));
    params.width = Some(sh.horizontal_size);
    params.height = Some(sh.vertical_size);
    if let Some((n, d)) = frame_rate_for_code(sh.frame_rate_code) {
        params.frame_rate = Some(Rational::new(n, d));
    }
    if sh.bit_rate != 0 && sh.bit_rate != 0x3FFFF {
        params.bit_rate = Some(sh.bit_rate as u64 * 400);
    }
    params
}
