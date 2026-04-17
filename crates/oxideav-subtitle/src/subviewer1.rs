//! SubViewer 1.0 (.sub) parser and writer.
//!
//! SubViewer 1 is a marker-framed format. Structure:
//!
//! ```text
//! **START SCRIPT** 00:00:01
//! 00:00:01,0
//! first subtitle|second line
//!
//! 00:00:04,0
//! next
//!
//! **END SCRIPT**
//! ```
//!
//! Each cue is an absolute start time with no end time; the end is
//! inferred from the next cue's start (or start + 3s for the trailing
//! cue). `|` is a hard line break. `[br]` is also accepted.
//!
//! Unknown inline tags survive as [`Segment::Raw`] text.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id for SubViewer 1.
pub const CODEC_ID: &str = "subviewer1";

/// Default duration for the final cue when no successor exists, in microseconds.
const DEFAULT_TRAIL_US: i64 = 3_000_000;

pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt),
        ..SubtitleTrack::default()
    };

    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut i = 0usize;

    // Capture the opening marker (if any) as extradata so the writer can
    // restore it.
    let mut extradata = String::new();
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            extradata.push('\n');
            i += 1;
            continue;
        }
        if trimmed.to_ascii_uppercase().starts_with("**START SCRIPT**") {
            extradata.push_str(lines[i]);
            extradata.push('\n');
            i += 1;
            break;
        }
        if parse_timestamp(trimmed).is_some() {
            break;
        }
        // Other pre-cue lines — keep in extradata.
        extradata.push_str(lines[i]);
        extradata.push('\n');
        i += 1;
    }
    track.extradata = extradata.into_bytes();

    // Walk cues.
    let mut cues: Vec<SubtitleCue> = Vec::new();
    while i < lines.len() {
        // Skip blanks + end-marker.
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        if trimmed.to_ascii_uppercase().starts_with("**END SCRIPT**") {
            break;
        }
        let start = match parse_timestamp(trimmed) {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };
        i += 1;

        // Collect body lines up to the next blank or end marker.
        let mut body_lines: Vec<&str> = Vec::new();
        while i < lines.len() {
            let tr = lines[i].trim();
            if tr.is_empty() {
                break;
            }
            if tr.to_ascii_uppercase().starts_with("**END SCRIPT**") {
                break;
            }
            // If the next non-empty line is itself a timestamp, stop (robust to
            // missing blank-line separators).
            if parse_timestamp(tr).is_some() && !body_lines.is_empty() {
                break;
            }
            body_lines.push(lines[i]);
            i += 1;
        }
        let body = body_lines.join("\n");
        let segments = parse_body(&body);
        cues.push(SubtitleCue {
            start_us: start,
            end_us: start, // provisional — fixed up after we know the successor.
            style_ref: None,
            positioning: None,
            segments,
        });
    }

    // Fix up end times.
    let n = cues.len();
    for idx in 0..n {
        let next = cues.get(idx + 1).map(|c| c.start_us);
        cues[idx].end_us = match next {
            Some(ns) => ns,
            None => cues[idx].start_us + DEFAULT_TRAIL_US,
        };
    }

    track.cues = cues;
    Ok(track)
}

pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let mut out = String::new();

    if !track.extradata.is_empty() {
        out.push_str(&String::from_utf8_lossy(&track.extradata));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    } else {
        // Synthesise a minimal header using the first cue's start.
        let start = track.cues.first().map(|c| c.start_us).unwrap_or(0);
        out.push_str(&format!("**START SCRIPT** {}\n", format_hhmmss(start)));
    }

    for cue in &track.cues {
        out.push_str(&format_timestamp(cue.start_us));
        out.push('\n');
        out.push_str(&render_body(&cue.segments));
        out.push('\n');
        out.push('\n');
    }

    out.push_str("**END SCRIPT**\n");
    Ok(out.into_bytes())
}

pub fn probe(buf: &[u8]) -> u8 {
    if looks_like_subviewer1(buf) {
        80
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a subviewer1 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(SvDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a subviewer1 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(SvEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

struct SvDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for SvDecoder {
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

struct SvEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for SvEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => {
                return Err(Error::invalid(
                    "subviewer1 encoder: expected Frame::Subtitle",
                ))
            }
        };
        let payload = cue_to_bytes(cue);
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

pub(crate) fn looks_like_subviewer1(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut saw_start = false;
    let mut saw_ts = false;
    for line in &lines {
        let tr = line.trim();
        if tr.is_empty() {
            continue;
        }
        if tr.to_ascii_uppercase().starts_with("**START SCRIPT**") {
            saw_start = true;
            continue;
        }
        if tr.to_ascii_uppercase().starts_with("**END SCRIPT**") {
            return saw_start || saw_ts;
        }
        if parse_timestamp(tr).is_some() {
            saw_ts = true;
            if saw_start {
                return true;
            }
        }
        if saw_start && saw_ts {
            return true;
        }
    }
    saw_start && saw_ts
}

// ---------------------------------------------------------------------------
// Timestamp + body parsing.

/// Parse `HH:MM:SS[,F]` or `HH:MM:SS.F` into microseconds. The fractional
/// part of SubViewer 1 is typically a single tenth-of-a-second digit.
fn parse_timestamp(s: &str) -> Option<i64> {
    let s = s.trim();
    // Must look like HH:MM:SS*
    if s.len() < 7 || s.as_bytes()[2] != b':' {
        return None;
    }
    let (hms, frac) = match s.find(|c: char| c == ',' || c == '.') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: i64 = parts[0].parse().ok()?;
    let m: i64 = parts[1].parse().ok()?;
    let sec: i64 = parts[2].parse().ok()?;
    let mut total = (h * 3600 + m * 60 + sec) * 1_000_000;
    if !frac.is_empty() {
        total += parse_fraction_us(frac);
    }
    Some(total)
}

fn parse_fraction_us(frac: &str) -> i64 {
    // SubViewer 1 writes a single tenth-of-a-second digit. We accept longer
    // runs too.
    let digits: String = frac.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return 0;
    }
    let f = format!("0.{}", digits);
    let v: f64 = f.parse().unwrap_or(0.0);
    (v * 1_000_000.0).round() as i64
}

fn format_timestamp(us: i64) -> String {
    let us = us.max(0);
    let whole = us / 1_000_000;
    let tenths = (us % 1_000_000) / 100_000; // one tenth-of-a-second digit
    let s = (whole % 60) as u32;
    let m = ((whole / 60) % 60) as u32;
    let h = (whole / 3600) as u32;
    format!("{:02}:{:02}:{:02},{}", h, m, s, tenths)
}

fn format_hhmmss(us: i64) -> String {
    let us = us.max(0);
    let whole = us / 1_000_000;
    let s = (whole % 60) as u32;
    let m = ((whole / 60) % 60) as u32;
    let h = (whole / 3600) as u32;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

fn parse_body(body: &str) -> Vec<Segment> {
    // Start from the raw line joined by '\n' (already present). SubViewer 1
    // puts in-line breaks as `|` or `[br]`. Convert those into
    // Segment::LineBreak. Join paragraph lines back with LineBreak too.
    let mut out: Vec<Segment> = Vec::new();
    let pieces: Vec<&str> = body.split('\n').collect();
    for (idx, piece) in pieces.iter().enumerate() {
        // Within a line, split on `|` and `[br]`.
        append_line(piece, &mut out);
        if idx + 1 < pieces.len() {
            out.push(Segment::LineBreak);
        }
    }
    out
}

fn append_line(line: &str, out: &mut Vec<Segment>) {
    // Replace `[br]` (case-insensitive) with `|`, then split.
    let normalised = replace_ignore_case(line, "[br]", "|");
    let parts: Vec<&str> = normalised.split('|').collect();
    for (j, part) in parts.iter().enumerate() {
        if !part.is_empty() {
            out.push(Segment::Text(part.to_string()));
        }
        if j + 1 < parts.len() {
            out.push(Segment::LineBreak);
        }
    }
}

fn replace_ignore_case(s: &str, needle: &str, replacement: &str) -> String {
    let needle_lc = needle.to_ascii_lowercase();
    let lc = s.to_ascii_lowercase();
    let mut result = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if lc[i..].starts_with(&needle_lc) {
            result.push_str(replacement);
            i += needle.len();
        } else {
            let ch = s[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    result
}

fn render_body(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_segments(segments, &mut out);
    out
}

fn append_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => out.push_str(s),
            Segment::LineBreak => out.push('|'),
            Segment::Bold(c) | Segment::Italic(c) | Segment::Underline(c) | Segment::Strike(c) => {
                append_segments(c, out)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_segments(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

fn decode_utf8_lossy_stripping_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

// ---------------------------------------------------------------------------
// Packet helpers.

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    // Packet payload: `START_TS\nBODY` — end time derivable from pkt.duration.
    let mut s = String::new();
    s.push_str(&format_timestamp(cue.start_us));
    s.push('\n');
    s.push_str(&render_body(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    while lines.first().map(|l| l.trim().is_empty()).unwrap_or(false) {
        lines.remove(0);
    }
    let ts = lines
        .first()
        .ok_or_else(|| Error::invalid("subviewer1: empty cue"))?
        .trim();
    let start = parse_timestamp(ts).ok_or_else(|| Error::invalid("subviewer1: bad timestamp"))?;
    let body = if lines.len() > 1 {
        lines[1..].join("\n")
    } else {
        String::new()
    };
    let segments = parse_body(&body);
    Ok(SubtitleCue {
        start_us: start,
        end_us: start + DEFAULT_TRAIL_US,
        style_ref: None,
        positioning: None,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "**START SCRIPT** 00:00:00
00:00:01,0
first|second line

00:00:04,0
hello world

00:00:07,5
third

00:00:10,0
fourth

00:00:12,0
fifth
**END SCRIPT**
";

    #[test]
    fn parses_five_cues() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 5);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        // End of cue 0 comes from start of cue 1.
        assert_eq!(t.cues[0].end_us, 4_000_000);
        assert_eq!(t.cues[2].start_us, 7_500_000);
        // Trailing cue gets +3s fallback.
        assert_eq!(t.cues[4].end_us, 12_000_000 + DEFAULT_TRAIL_US);
    }

    #[test]
    fn pipe_is_line_break() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let mut count = 0;
        for s in &t.cues[0].segments {
            if matches!(s, Segment::LineBreak) {
                count += 1;
            }
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn roundtrips() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let out = write(&t).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("**START SCRIPT**"));
        assert!(s.contains("**END SCRIPT**"));
        assert!(s.contains("00:00:01,0"));
        assert!(s.contains("first|second line"));
    }

    #[test]
    fn probe_yes() {
        assert!(probe(SAMPLE.as_bytes()) > 0);
    }

    #[test]
    fn probe_no() {
        assert_eq!(probe(b"WEBVTT\n"), 0);
    }
}

#[cfg(test)]
mod _probe {
    use super::*;
    fn _types(b: &[u8]) -> bool {
        probe(b) > 0 && parse(b).is_ok() && write(&SubtitleTrack::default()).is_ok()
    }
}
