//! `Decoder` / `Encoder` implementations for SRT and WebVTT.
//!
//! The decoder consumes a single `Packet` carrying the cue text in its
//! format's natural form and emits a [`Frame::Subtitle`] with the fully
//! parsed cue. The encoder is the mirror. In both directions the
//! per-packet layout matches what the companion container produces /
//! consumes.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, TimeBase};

use crate::{srt, webvtt};

pub const SRT_CODEC_ID: &str = "subrip";
pub const WEBVTT_CODEC_ID: &str = "webvtt";

/// Build a subtitle decoder by dispatching on `params.codec_id`.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let kind = classify(&params.codec_id)?;
    Ok(Box::new(SubtitleDecoder {
        kind,
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Build a subtitle encoder by dispatching on `params.codec_id`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let kind = classify(&params.codec_id)?;
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(SubtitleEncoder {
        kind,
        params: p,
        pending: VecDeque::new(),
    }))
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Srt,
    WebVtt,
}

fn classify(id: &CodecId) -> Result<Kind> {
    match id.as_str() {
        SRT_CODEC_ID => Ok(Kind::Srt),
        WEBVTT_CODEC_ID => Ok(Kind::WebVtt),
        other => Err(Error::unsupported(format!(
            "not a subtitle codec id: {other}"
        ))),
    }
}

// ---- Decoder -------------------------------------------------------------

struct SubtitleDecoder {
    kind: Kind,
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for SubtitleDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let cue = match self.kind {
            Kind::Srt => srt::bytes_to_cue(&packet.data)?,
            Kind::WebVtt => webvtt::bytes_to_cue(&packet.data)?,
        };
        // If the packet carried an overriding pts/duration, honour it.
        let mut cue = cue;
        if let Some(pts) = packet.pts {
            // Rescale the pts from packet.time_base into microseconds.
            let us = rescale_to_us(pts, packet.time_base);
            let span = cue.end_us - cue.start_us;
            cue.start_us = us;
            cue.end_us = us + span;
        }
        self.pending.push_back(Frame::Subtitle(cue));
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(f);
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

// ---- Encoder -------------------------------------------------------------

struct SubtitleEncoder {
    kind: Kind,
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for SubtitleEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("subtitle encoder: expected Frame::Subtitle")),
        };
        let payload = match self.kind {
            Kind::Srt => srt::cue_to_bytes(cue),
            Kind::WebVtt => webvtt::cue_to_bytes(cue),
        };
        let tb = TimeBase::new(1, 1_000_000);
        let mut pkt = Packet::new(0, tb, payload);
        pkt.pts = Some(cue.start_us);
        pkt.dts = Some(cue.start_us);
        pkt.duration = Some((cue.end_us - cue.start_us).max(0));
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

fn rescale_to_us(pts: i64, tb: TimeBase) -> i64 {
    // Avoid 128-bit math helper dependency; use ratio-based conversion.
    // Using the existing tb.rescale.
    tb.rescale(pts, TimeBase::new(1, 1_000_000))
}
