//! MPsub (.sub) parser and writer.
//!
//! MPsub uses *relative* timing — each cue's first number is the gap
//! from the previous cue's START, and the second number is this cue's
//! duration. Both are seconds (`FORMAT=TIME`) or frames (`FORMAT=FRAMES`).
//!
//! ```text
//! TITLE=Demo
//! AUTHOR=Anon
//! FORMAT=TIME
//!
//! 2.5 3.0
//! First cue
//! multi-line OK
//!
//! 1.0 4.0
//! Second cue
//! ```
//!
//! We track a running cursor: `start(n) = start(n-1) + rel_start`,
//! `end(n) = start(n) + duration`. Multi-line text is joined with
//! [`Segment::LineBreak`]. No inline formatting is defined in the spec;
//! we round-trip text verbatim.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::SubtitleTrack;

pub const CODEC_ID: &str = "mpsub";

/// Default FPS used when `FORMAT=FRAMES` is declared. Can be overridden
/// by a `FPS=` header, though that's rare.
pub const DEFAULT_FPS: f64 = 25.0;

/// Parse an MPsub payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();

    let mut mode = Mode::Time;
    let mut fps = DEFAULT_FPS;
    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut cursor_us: i64 = 0;

    // Walk blocks. A block is: timing line + one-or-more text lines,
    // separated by blank lines. Header lines (KEY=VALUE) appear before
    // any timing block and start with an uppercase ASCII word followed by `=`.
    let mut cues: Vec<SubtitleCue> = Vec::new();
    let mut i = 0;
    let mut header_done = false;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.is_empty() {
            i += 1;
            continue;
        }

        // Header line? (only while no cue has been parsed yet).
        if !header_done {
            if let Some(eq) = line.find('=') {
                let key = line[..eq].trim();
                let val = line[eq + 1..].trim();
                if is_header_key(key) {
                    let key_upper = key.to_ascii_uppercase();
                    match key_upper.as_str() {
                        "FORMAT" => {
                            mode = match val.to_ascii_uppercase().as_str() {
                                "FRAMES" => Mode::Frames,
                                _ => Mode::Time,
                            };
                        }
                        "FPS" => {
                            if let Ok(v) = val.parse::<f64>() {
                                if v > 0.0 {
                                    fps = v;
                                }
                            }
                        }
                        _ => {}
                    }
                    metadata.push((key.to_string(), val.to_string()));
                    i += 1;
                    continue;
                }
            }
        }

        // Try to parse as a timing line: two whitespace-separated numbers.
        let (rel_start, duration) = match parse_timing(line) {
            Some(v) => v,
            None => {
                // Not a timing line and not a header — skip.
                i += 1;
                continue;
            }
        };
        header_done = true;

        let (rel_start_us, duration_us) = match mode {
            Mode::Time => (seconds_to_us(rel_start), seconds_to_us(duration)),
            Mode::Frames => {
                let rs = frames_to_us(rel_start, fps);
                let d = frames_to_us(duration, fps);
                (rs, d)
            }
        };

        let start_us = cursor_us + rel_start_us;
        let end_us = start_us + duration_us;
        cursor_us = start_us;
        i += 1;

        // Collect text lines until the next blank line.
        let mut text_lines: Vec<&str> = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text_lines.push(lines[i]);
            i += 1;
        }

        let segments = build_segments(&text_lines);
        cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments,
        });
    }

    Ok(SubtitleTrack {
        cues,
        metadata,
        ..SubtitleTrack::default()
    })
}

/// Re-emit a track as MPsub bytes. Always writes `FORMAT=TIME`.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let mut out = String::new();
    // Preserve TITLE / AUTHOR / FILE if present, then force FORMAT=TIME.
    let mut wrote_format = false;
    for (k, v) in &track.metadata {
        let key_upper = k.to_ascii_uppercase();
        if key_upper == "FORMAT" {
            out.push_str("FORMAT=TIME\n");
            wrote_format = true;
        } else if key_upper == "FPS" {
            // Drop — we're writing FORMAT=TIME so FPS is meaningless.
            continue;
        } else if is_header_key(k) {
            out.push_str(&format!("{}={}\n", k, v));
        }
    }
    if !wrote_format {
        out.push_str("FORMAT=TIME\n");
    }
    out.push('\n');

    let mut cursor_us: i64 = 0;
    for cue in &track.cues {
        let rel = cue.start_us - cursor_us;
        let dur = (cue.end_us - cue.start_us).max(0);
        cursor_us = cue.start_us;
        out.push_str(&format!("{} {}\n", fmt_secs(rel), fmt_secs(dur)));
        let body = render_body(&cue.segments);
        if body.is_empty() {
            out.push('\n');
        } else {
            out.push_str(&body);
            if !body.ends_with('\n') {
                out.push('\n');
            }
        }
        out.push('\n');
    }

    Ok(out.into_bytes())
}

/// Quick probe — positive score if a `FORMAT=` header appears early.
pub fn probe(buf: &[u8]) -> u8 {
    let text = strip_bom(buf);
    let mut checked = 0;
    let mut saw_format = false;
    let mut saw_timing = false;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        checked += 1;
        let upper = line.to_ascii_uppercase();
        if upper.starts_with("FORMAT=TIME") || upper.starts_with("FORMAT=FRAMES") {
            saw_format = true;
        }
        if parse_timing(line).is_some() {
            saw_timing = true;
        }
        if checked >= 10 {
            break;
        }
    }
    match (saw_format, saw_timing) {
        (true, true) => 85,
        (true, false) => 60,
        _ => 0,
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an mpsub codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(MpsubDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an mpsub codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(MpsubEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------
// Packet shape — one cue carries absolute start+end so decode doesn't need
// the prior cue for context. Shape:
//
//   start_seconds duration_seconds
//   text line 1
//   text line 2

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let dur = (cue.end_us - cue.start_us).max(0);
    let mut s = format!("{} {}\n", fmt_secs(cue.start_us), fmt_secs(dur));
    s.push_str(&render_body(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut i = 0;
    while i < lines.len() && lines[i].trim().is_empty() {
        i += 1;
    }
    if i >= lines.len() {
        return Err(Error::invalid("mpsub: empty cue"));
    }
    let (start, dur) =
        parse_timing(lines[i].trim()).ok_or_else(|| Error::invalid("mpsub: bad timing"))?;
    let start_us = seconds_to_us(start);
    let end_us = start_us + seconds_to_us(dur);
    let body: Vec<&str> = lines[i + 1..]
        .iter()
        .take_while(|l| !l.trim().is_empty())
        .copied()
        .collect();
    Ok(SubtitleCue {
        start_us,
        end_us,
        style_ref: None,
        positioning: None,
        segments: build_segments(&body),
    })
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Mode {
    Time,
    Frames,
}

fn is_header_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn parse_timing(line: &str) -> Option<(f64, f64)> {
    let mut parts = line.split_whitespace();
    let a = parts.next()?.parse::<f64>().ok()?;
    let b = parts.next()?.parse::<f64>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    Some((a, b))
}

fn build_segments(lines: &[&str]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if idx > 0 {
            out.push(Segment::LineBreak);
        }
        if !line.is_empty() {
            out.push(Segment::Text(line.to_string()));
        }
    }
    out
}

fn render_body(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_plain(segments, &mut out);
    // Trim trailing newlines (we'll add our own blank-line separator).
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

fn append_plain(segs: &[Segment], out: &mut String) {
    for seg in segs {
        match seg {
            Segment::Text(s) => out.push_str(s),
            Segment::LineBreak => out.push('\n'),
            Segment::Bold(c) | Segment::Italic(c) | Segment::Underline(c) | Segment::Strike(c) => {
                append_plain(c, out)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_plain(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(_) => {}
        }
    }
}

fn seconds_to_us(s: f64) -> i64 {
    (s * 1_000_000.0).round() as i64
}

fn frames_to_us(f: f64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((f / fps) * 1_000_000.0).round() as i64
}

fn fmt_secs(us: i64) -> String {
    let neg = us < 0;
    let abs = us.unsigned_abs();
    let whole = abs / 1_000_000;
    let frac = abs % 1_000_000;
    // Preserve up to 3 decimals; strip trailing zeros.
    let mut frac_str = format!("{:06}", frac);
    frac_str.truncate(3);
    while frac_str.ends_with('0') {
        frac_str.pop();
    }
    let base = if frac_str.is_empty() {
        format!("{}", whole)
    } else {
        format!("{}.{}", whole, frac_str)
    };
    if neg {
        format!("-{}", base)
    } else {
        base
    }
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

struct MpsubDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for MpsubDecoder {
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

struct MpsubEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for MpsubEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("mpsub encoder: expected Frame::Subtitle")),
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
    fn parse_header_and_cues() {
        let src = "TITLE=demo\nFORMAT=TIME\n\n2.5 3.0\nHello\n\n1.0 4.0\nWorld\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 2_500_000);
        assert_eq!(t.cues[0].end_us, 5_500_000);
        // Second cue: rel 1.0s after first's start (2.5s) → 3.5s start
        assert_eq!(t.cues[1].start_us, 3_500_000);
        assert_eq!(t.cues[1].end_us, 7_500_000);
    }

    #[test]
    fn parse_multiline_body() {
        let src = "FORMAT=TIME\n\n0 2\nline1\nline2\n";
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
        assert!(probe(b"FORMAT=TIME\n\n1.0 2.0\nhi\n") > 0);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"random unrelated text\nwith nothing\n"), 0);
    }
}
