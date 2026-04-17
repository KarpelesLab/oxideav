//! VPlayer (.txt, .vpl) parser and writer.
//!
//! One cue per line. Timing is a colon-separated `HH:MM:SS` prefix, with
//! the cue body following a final colon:
//!
//! ```text
//! 00:00:01:Hello world
//! 00:00:04:Second line
//! 00:00:07:Last line
//! ```
//!
//! The end time of each cue is the start of the following cue, or (for
//! the final cue) a 3-second fallback.
//!
//! There is no inline formatting defined; `|` is sometimes used for a
//! hard line break and we honour it.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::SubtitleTrack;

pub const CODEC_ID: &str = "vplayer";

/// Fallback duration for the final cue when the file offers no later
/// timestamp. Microseconds.
pub const TRAILING_CUE_US: i64 = 3_000_000;

/// Parse a VPlayer payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    // Pass 1: collect timing + body pairs.
    let mut raw_cues: Vec<(i64, String)> = Vec::new();
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if let Some((start_us, body)) = parse_line(line) {
            raw_cues.push((start_us, body.to_string()));
        }
    }

    // Pass 2: set end times from successor starts.
    let mut cues: Vec<SubtitleCue> = Vec::with_capacity(raw_cues.len());
    for i in 0..raw_cues.len() {
        let (start_us, body) = raw_cues[i].clone();
        let end_us = if i + 1 < raw_cues.len() {
            raw_cues[i + 1].0.max(start_us + 1)
        } else {
            start_us + TRAILING_CUE_US
        };
        cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments: body_to_segments(&body),
        });
    }

    Ok(SubtitleTrack {
        cues,
        ..SubtitleTrack::default()
    })
}

/// Re-emit a track as VPlayer bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let mut out = String::new();
    for cue in &track.cues {
        out.push_str(&format!("{}:", fmt_hms(cue.start_us)));
        out.push_str(&render_body(&cue.segments));
        out.push('\n');
    }
    Ok(out.into_bytes())
}

/// Quick probe — positive score if the first non-empty line parses as
/// `HH:MM:SS:text`.
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
        70
    } else if hits > 0 {
        35
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a vplayer codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(VPlayerDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a vplayer codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(VPlayerEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------
// Packet shape: `HH:MM:SS:text[|more text]`

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let mut s = format!("{}:", fmt_hms(cue.start_us));
    s.push_str(&render_body(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let line = text
        .lines()
        .map(|l| l.trim_end_matches('\r').trim_end())
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| Error::invalid("vplayer: empty cue"))?;
    let (start_us, body) =
        parse_line(line).ok_or_else(|| Error::invalid("vplayer: bad cue line"))?;
    Ok(SubtitleCue {
        start_us,
        end_us: start_us + TRAILING_CUE_US,
        style_ref: None,
        positioning: None,
        segments: body_to_segments(body),
    })
}

// ---------------------------------------------------------------------------

fn parse_line(line: &str) -> Option<(i64, &str)> {
    // `HH:MM:SS:body` — find the 3rd colon.
    let bytes = line.as_bytes();
    let mut colon_idx = [0usize; 3];
    let mut found = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b':' {
            if found < 3 {
                colon_idx[found] = i;
            }
            found += 1;
            if found == 3 {
                break;
            }
        }
    }
    if found < 3 {
        return None;
    }
    // HH, MM, SS components must be all digits and of sane magnitude.
    let hh = &line[..colon_idx[0]];
    let mm = &line[colon_idx[0] + 1..colon_idx[1]];
    let ss = &line[colon_idx[1] + 1..colon_idx[2]];
    let body = &line[colon_idx[2] + 1..];
    let h: u32 = hh.trim().parse().ok()?;
    let m: u32 = mm.trim().parse().ok()?;
    let s: u32 = ss.trim().parse().ok()?;
    if m >= 60 || s >= 60 {
        return None;
    }
    let us = (h as i64) * 3_600_000_000 + (m as i64) * 60_000_000 + (s as i64) * 1_000_000;
    Some((us, body))
}

fn body_to_segments(body: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (idx, piece) in body.split('|').enumerate() {
        if idx > 0 {
            out.push(Segment::LineBreak);
        }
        if !piece.is_empty() {
            out.push(Segment::Text(piece.to_string()));
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

fn fmt_hms(us: i64) -> String {
    let us = us.max(0);
    let total_s = us / 1_000_000;
    let h = total_s / 3600;
    let m = (total_s / 60) % 60;
    let s = total_s % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
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

struct VPlayerDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for VPlayerDecoder {
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
        // Prefer the packet duration when available — VPlayer's on-wire
        // shape has no end time.
        if let Some(dur) = packet.duration {
            let dur_us = packet.time_base.rescale(dur, TimeBase::new(1, 1_000_000));
            cue.end_us = cue.start_us + dur_us;
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

struct VPlayerEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for VPlayerEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("vplayer encoder: expected Frame::Subtitle")),
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
        let src = "00:00:01:Hello\n00:00:04:World\n00:00:07:Last\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 3);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 4_000_000);
        // Last cue uses trailing fallback.
        assert_eq!(t.cues[2].end_us, 7_000_000 + TRAILING_CUE_US);
    }

    #[test]
    fn parse_pipe_as_linebreak() {
        let src = "00:00:01:line1|line2\n";
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
        assert!(probe(b"00:00:01:hello\n") > 0);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"random text\nhello world\n"), 0);
    }
}
