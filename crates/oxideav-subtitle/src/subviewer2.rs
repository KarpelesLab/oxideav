//! SubViewer 2.0 (.sub) parser and writer.
//!
//! SubViewer 2 carries a small metadata block then cue lines:
//!
//! ```text
//! [INFORMATION]
//! [TITLE]
//! Example
//! [AUTHOR]
//! Me
//! [END INFORMATION]
//! [SUBTITLE]
//!
//! 00:00:01.00,00:00:03.50
//! Hello|World
//!
//! 00:00:04.00,00:00:06.00
//! Second cue
//! ```
//!
//! Timing lines are `HH:MM:SS.hh,HH:MM:SS.hh`. `|` is a hard line break;
//! `[br]` is also accepted. Metadata tags (`[TITLE]`, `[AUTHOR]`, etc.)
//! populate the track's `metadata` table; the raw `[INFORMATION]` block is
//! also kept as extradata so round-trip preserves ordering.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id for SubViewer 2.
pub const CODEC_ID: &str = "subviewer2";

pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt),
        ..SubtitleTrack::default()
    };

    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut i = 0usize;

    // Parse the INFORMATION block (if present).
    let mut extradata = String::new();
    let mut in_info = false;
    let mut subtitle_seen = false;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let upper = trimmed.to_ascii_uppercase();

        if upper == "[INFORMATION]" {
            in_info = true;
            extradata.push_str(line);
            extradata.push('\n');
            i += 1;
            continue;
        }
        if upper == "[END INFORMATION]" {
            in_info = false;
            extradata.push_str(line);
            extradata.push('\n');
            i += 1;
            continue;
        }
        if upper == "[SUBTITLE]" {
            extradata.push_str(line);
            extradata.push('\n');
            subtitle_seen = true;
            i += 1;
            break;
        }

        if in_info {
            extradata.push_str(line);
            extradata.push('\n');
            // Collect `[KEY]\nvalue` pairs.
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                let key = trimmed[1..trimmed.len() - 1].to_ascii_lowercase();
                // Value is the next non-empty line that isn't another bracket tag.
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                if j < lines.len() {
                    let vtr = lines[j].trim();
                    if !(vtr.starts_with('[') && vtr.ends_with(']')) {
                        track.metadata.push((key, vtr.to_string()));
                    }
                }
            }
        } else {
            // Pre-info preamble — keep in extradata.
            extradata.push_str(line);
            extradata.push('\n');
        }
        i += 1;
    }

    // If there was no `[SUBTITLE]` marker, just start scanning for timings.
    let _ = subtitle_seen;
    track.extradata = extradata.into_bytes();

    // Parse cues.
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() {
            i += 1;
            continue;
        }

        let (start_us, end_us) = match parse_timing_line(trimmed) {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };
        i += 1;

        // Body lines — stop at next blank / next timing.
        let mut body_lines: Vec<&str> = Vec::new();
        while i < lines.len() {
            let tr = lines[i].trim();
            if tr.is_empty() {
                break;
            }
            if parse_timing_line(tr).is_some() {
                break;
            }
            body_lines.push(lines[i]);
            i += 1;
        }
        let body = body_lines.join("\n");
        let segments = parse_body(&body);
        track.cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments,
        });
    }

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
        out.push_str("[INFORMATION]\n");
        for (k, v) in &track.metadata {
            out.push_str(&format!("[{}]\n{}\n", k.to_ascii_uppercase(), v));
        }
        out.push_str("[END INFORMATION]\n[SUBTITLE]\n");
    }

    for cue in &track.cues {
        out.push('\n');
        out.push_str(&format_timing_line(cue.start_us, cue.end_us));
        out.push('\n');
        out.push_str(&render_body(&cue.segments));
        out.push('\n');
    }

    Ok(out.into_bytes())
}

pub fn probe(buf: &[u8]) -> u8 {
    if looks_like_subviewer2(buf) {
        92
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a subviewer2 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(Sv2Decoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a subviewer2 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(Sv2Encoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

struct Sv2Decoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for Sv2Decoder {
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

struct Sv2Encoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for Sv2Encoder {
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
                    "subviewer2 encoder: expected Frame::Subtitle",
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

pub(crate) fn looks_like_subviewer2(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    let lc = text.to_ascii_lowercase();
    if lc.contains("[information]") && lc.contains("[subtitle]") {
        return true;
    }
    // Fallback: find a `HH:MM:SS.hh,HH:MM:SS.hh` line somewhere near the top.
    for line in text.lines().take(40) {
        if parse_timing_line(line.trim()).is_some() {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Timestamp + body parsing.

fn parse_timing_line(line: &str) -> Option<(i64, i64)> {
    let comma = line.find(',')?;
    let (l, r) = line.split_at(comma);
    let r = &r[1..];
    let s = parse_timestamp(l.trim())?;
    // Right side: only take up to first whitespace — tolerate trailing tokens.
    let r_ts = r.split_whitespace().next()?;
    let e = parse_timestamp(r_ts)?;
    Some((s, e))
}

fn parse_timestamp(s: &str) -> Option<i64> {
    let (hms, frac) = match s.find(|c: char| c == '.' || c == ',') {
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
    total += parse_fraction_us(frac);
    Some(total)
}

fn parse_fraction_us(frac: &str) -> i64 {
    let digits: String = frac.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return 0;
    }
    let f = format!("0.{}", digits);
    let v: f64 = f.parse().unwrap_or(0.0);
    (v * 1_000_000.0).round() as i64
}

fn format_timing_line(start_us: i64, end_us: i64) -> String {
    format!(
        "{},{}",
        format_timestamp(start_us),
        format_timestamp(end_us)
    )
}

fn format_timestamp(us: i64) -> String {
    let us = us.max(0);
    let whole = us / 1_000_000;
    let hundredths = (us % 1_000_000) / 10_000; // two fractional digits
    let s = (whole % 60) as u32;
    let m = ((whole / 60) % 60) as u32;
    let h = (whole / 3600) as u32;
    format!("{:02}:{:02}:{:02}.{:02}", h, m, s, hundredths)
}

fn parse_body(body: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let pieces: Vec<&str> = body.split('\n').collect();
    for (idx, piece) in pieces.iter().enumerate() {
        append_line(piece, &mut out);
        if idx + 1 < pieces.len() {
            out.push(Segment::LineBreak);
        }
    }
    out
}

fn append_line(line: &str, out: &mut Vec<Segment>) {
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
    let mut s = String::new();
    s.push_str(&format_timing_line(cue.start_us, cue.end_us));
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
    let ts_line = lines
        .first()
        .ok_or_else(|| Error::invalid("subviewer2: empty cue"))?
        .trim();
    let (start, end) =
        parse_timing_line(ts_line).ok_or_else(|| Error::invalid("subviewer2: bad timing"))?;
    let body = if lines.len() > 1 {
        lines[1..].join("\n")
    } else {
        String::new()
    };
    let segments = parse_body(&body);
    Ok(SubtitleCue {
        start_us: start,
        end_us: end,
        style_ref: None,
        positioning: None,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "[INFORMATION]
[TITLE]
Example
[AUTHOR]
Me
[END INFORMATION]
[SUBTITLE]

00:00:01.00,00:00:03.50
Hello|World

00:00:04.00,00:00:06.00
Second cue

00:00:07.25,00:00:08.00
Short one

00:00:09.00,00:00:11.00
multi
line

00:00:12.00,00:00:13.50
Last
";

    #[test]
    fn parses_five_cues() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 5);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_500_000);
        assert_eq!(t.cues[2].start_us, 7_250_000);
    }

    #[test]
    fn parses_metadata() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert!(t
            .metadata
            .iter()
            .any(|(k, v)| k == "title" && v == "Example"));
        assert!(t.metadata.iter().any(|(k, v)| k == "author" && v == "Me"));
    }

    #[test]
    fn pipe_is_line_break() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let mut lb = 0;
        for s in &t.cues[0].segments {
            if matches!(s, Segment::LineBreak) {
                lb += 1;
            }
        }
        assert_eq!(lb, 1);
    }

    #[test]
    fn roundtrips() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let out = write(&t).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("[INFORMATION]"));
        assert!(s.contains("[SUBTITLE]"));
        assert!(s.contains("00:00:01.00,00:00:03.50"));
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
