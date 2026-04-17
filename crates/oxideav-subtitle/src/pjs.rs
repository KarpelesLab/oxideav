//! PJS (.pjs, Phoenix Japanese Subtitle) parser and writer.
//!
//! One cue per line: `start_frame,end_frame,"text"`. Frame-based;
//! uses 25 fps by default. Text is in quotes; `|` or `\r\n` inside the
//! quoted body becomes a hard line break.
//!
//! ```text
//! 25,75,"Hello world"
//! 100,150,"Line one|Line two"
//! ```

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::SubtitleTrack;

pub const CODEC_ID: &str = "pjs";
pub const DEFAULT_FPS: f64 = 25.0;

/// Parse a PJS payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    let mut cues: Vec<SubtitleCue> = Vec::new();
    let fps = DEFAULT_FPS;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let (start_frame, end_frame, body) = match parse_line(line) {
            Some(v) => v,
            None => continue,
        };
        cues.push(SubtitleCue {
            start_us: frame_to_us(start_frame, fps),
            end_us: frame_to_us(end_frame, fps),
            style_ref: None,
            positioning: None,
            segments: body_to_segments(body),
        });
    }
    Ok(SubtitleTrack {
        cues,
        ..SubtitleTrack::default()
    })
}

/// Re-emit a track as PJS bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let fps = DEFAULT_FPS;
    let mut out = String::new();
    for cue in &track.cues {
        let sf = us_to_frame(cue.start_us, fps);
        let ef = us_to_frame(cue.end_us, fps);
        out.push_str(&format!(
            "{},{},\"{}\"\n",
            sf,
            ef,
            escape_pjs(&render_body(&cue.segments))
        ));
    }
    Ok(out.into_bytes())
}

/// Quick probe: positive score if a couple of lines parse as
/// `digits,digits,"..."`.
pub fn probe(buf: &[u8]) -> u8 {
    let text = strip_bom(buf);
    let mut checked = 0;
    let mut hits = 0;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        checked += 1;
        if parse_line(line).is_some() {
            hits += 1;
        }
        if checked >= 5 {
            break;
        }
    }
    if checked == 0 {
        return 0;
    }
    if hits == checked {
        75
    } else if hits > 0 {
        40
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a pjs codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(PjsDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a pjs codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(PjsEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let fps = DEFAULT_FPS;
    let sf = us_to_frame(cue.start_us, fps);
    let ef = us_to_frame(cue.end_us, fps);
    format!(
        "{},{},\"{}\"",
        sf,
        ef,
        escape_pjs(&render_body(&cue.segments))
    )
    .into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let line = text
        .lines()
        .map(|l| l.trim_end_matches('\r').trim())
        .find(|l| !l.is_empty())
        .ok_or_else(|| Error::invalid("pjs: empty cue"))?;
    let (sf, ef, body) = parse_line(line).ok_or_else(|| Error::invalid("pjs: bad cue line"))?;
    Ok(SubtitleCue {
        start_us: frame_to_us(sf, DEFAULT_FPS),
        end_us: frame_to_us(ef, DEFAULT_FPS),
        style_ref: None,
        positioning: None,
        segments: body_to_segments(body),
    })
}

// ---------------------------------------------------------------------------

fn parse_line(line: &str) -> Option<(i64, i64, &str)> {
    // `digits , digits , "text"`
    let c1 = line.find(',')?;
    let sf: i64 = line[..c1].trim().parse().ok()?;
    let after1 = &line[c1 + 1..];
    let c2 = after1.find(',')?;
    let ef: i64 = after1[..c2].trim().parse().ok()?;
    let after2 = after1[c2 + 1..].trim();
    // Quoted text. Strip a single surrounding pair of `"` if present;
    // otherwise take the rest verbatim.
    let body = if let Some(s1) = after2.strip_prefix('"') {
        if let Some(stripped) = s1.strip_suffix('"') {
            stripped
        } else {
            s1
        }
    } else {
        after2
    };
    Some((sf, ef, body))
}

fn body_to_segments(body: &str) -> Vec<Segment> {
    // `|` or literal `\r\n` inside the quoted content → LineBreak.
    let mut pieces: Vec<String> = vec![String::new()];
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '|' => pieces.push(String::new()),
            '\\' => {
                // Handle `\n`, `\r\n`.
                match chars.peek() {
                    Some('n') => {
                        chars.next();
                        pieces.push(String::new());
                    }
                    Some('r') => {
                        chars.next();
                        if let Some('\\') = chars.peek() {
                            chars.next();
                            if let Some('n') = chars.peek() {
                                chars.next();
                            }
                        }
                        pieces.push(String::new());
                    }
                    _ => pieces.last_mut().unwrap().push('\\'),
                }
            }
            _ => pieces.last_mut().unwrap().push(c),
        }
    }
    let mut out: Vec<Segment> = Vec::new();
    for (idx, p) in pieces.iter().enumerate() {
        if idx > 0 {
            out.push(Segment::LineBreak);
        }
        if !p.is_empty() {
            out.push(Segment::Text(p.clone()));
        }
    }
    out
}

fn render_body(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_flat(segments, &mut out);
    out
}

fn append_flat(segs: &[Segment], out: &mut String) {
    for seg in segs {
        match seg {
            Segment::Text(s) => out.push_str(&s.replace('|', "/")),
            Segment::LineBreak => out.push('|'),
            Segment::Bold(c) | Segment::Italic(c) | Segment::Underline(c) | Segment::Strike(c) => {
                append_flat(c, out)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_flat(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(_) => {}
        }
    }
}

fn escape_pjs(body: &str) -> String {
    body.replace('"', "'")
}

fn frame_to_us(f: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((f as f64 / fps) * 1_000_000.0).round() as i64
}

fn us_to_frame(us: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((us as f64 / 1_000_000.0) * fps).round() as i64
}

fn strip_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

// ---------------------------------------------------------------------------

struct PjsDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for PjsDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let mut cue = bytes_to_cue(&packet.data)?;
        if let Some(pts) = packet.pts {
            let us = packet.time_base.rescale(pts, TimeBase::new(1, 1_000_000));
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

struct PjsEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for PjsEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("pjs encoder: expected Frame::Subtitle")),
        };
        let tb = TimeBase::new(1, 1_000_000);
        let payload = cue_to_bytes(cue);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let src = "25,75,\"Hello world\"\n100,150,\"Second\"\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, frame_to_us(25, DEFAULT_FPS));
    }

    #[test]
    fn parse_linebreak() {
        let src = "0,10,\"line1|line2\"\n";
        let t = parse(src.as_bytes()).unwrap();
        let breaks = t.cues[0]
            .segments
            .iter()
            .filter(|s| matches!(s, Segment::LineBreak))
            .count();
        assert_eq!(breaks, 1);
    }

    #[test]
    fn probe_positive() {
        assert!(probe(b"0,25,\"hi\"\n") > 0);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"not a pjs file\n"), 0);
    }
}
