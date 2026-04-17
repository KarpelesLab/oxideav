//! MicroDVD (.sub, .txt) parser and writer.
//!
//! Line-based, frame-timed format. Each cue fits on a single line:
//!
//! ```text
//! {1}{1}23.976
//! {25}{75}Hello|world
//! {90}{120}{y:i}some italic text
//! {150}{180}{c:$00FFFF}yellow text|second line
//! ```
//!
//! * `{start_frame}{end_frame}text` — one cue per line.
//! * Frame-based timing; the optional *first* cue `{1}{1}<fps>` sets the
//!   frame rate (e.g. `23.976`, `25`, `29.97`). Default fps is 25.
//! * Inline `{y:b}`/`{y:i}`/`{y:u}` flip on bold/italic/underline for the
//!   rest of the line.
//! * `{c:$BBGGRR}` — BGR hex colour (note: B comes first, unlike SRT's RRGGBB).
//! * `|` inside text is a hard line break.
//!
//! Unknown `{...}` tags fall through as [`Segment::Raw`] so re-emit stays
//! faithful.

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};
use std::collections::VecDeque;

use crate::ir::SubtitleTrack;

/// Codec id used by this module.
pub const CODEC_ID: &str = "microdvd";

/// Default frame rate used when the file does not carry a `{1}{1}<fps>` hint.
pub const DEFAULT_FPS: f64 = 25.0;

/// Parse a MicroDVD payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    let mut cues: Vec<SubtitleCue> = Vec::new();
    let mut fps = DEFAULT_FPS;

    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let (start_frame, end_frame, rest) = match parse_frame_header(line) {
            Some(v) => v,
            None => continue,
        };

        // FPS-setter line: `{1}{1}<fps>` with a parseable float body and no
        // other formatting — applies retroactively only for subsequent cues
        // (there shouldn't be any prior cues anyway).
        if start_frame == 1 && end_frame == 1 {
            if let Ok(v) = rest.trim().parse::<f64>() {
                if v > 0.0 && v < 1_000.0 {
                    fps = v;
                    continue;
                }
            }
        }

        let start_us = frame_to_us(start_frame, fps);
        let end_us = frame_to_us(end_frame, fps);
        let segments = parse_inline(rest);
        cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments,
        });
    }

    let mut t = SubtitleTrack::default();
    // Preserve the fps in metadata so the writer can re-emit matching frame
    // numbers even if the track is round-tripped.
    t.metadata.push(("microdvd_fps".into(), format!("{}", fps)));
    t.cues = cues;
    Ok(t)
}

/// Re-emit a track as MicroDVD bytes. Uses the fps from metadata
/// (`microdvd_fps`) if present, otherwise [`DEFAULT_FPS`].
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let fps = track
        .metadata
        .iter()
        .find(|(k, _)| k == "microdvd_fps")
        .and_then(|(_, v)| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_FPS);
    let mut out = String::new();
    // Emit the fps line for non-default rates so re-readers keep in sync.
    if (fps - DEFAULT_FPS).abs() > 1e-6 {
        out.push_str(&format!("{{1}}{{1}}{}\n", fmt_fps(fps)));
    }
    for cue in &track.cues {
        let sf = us_to_frame(cue.start_us, fps);
        let ef = us_to_frame(cue.end_us, fps);
        out.push_str(&format!("{{{}}}{{{}}}", sf, ef));
        out.push_str(&render_inline(&cue.segments));
        out.push('\n');
    }
    Ok(out.into_bytes())
}

/// Quick probe — looks at the first non-empty lines for `{n}{n}...` shape.
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
        if parse_frame_header(line).is_some() {
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
        80
    } else if hits > 0 {
        50
    } else {
        0
    }
}

/// Build a MicroDVD decoder (packet = one cue in MicroDVD on-wire form).
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a microdvd codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(MicroDvdDecoder {
        codec_id: params.codec_id.clone(),
        fps: extract_fps(&params.extradata),
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Build a MicroDVD encoder.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a microdvd codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(MicroDvdEncoder {
        params: p,
        fps: extract_fps(&params.extradata),
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------

/// On-wire per-packet form: a single MicroDVD cue line (no trailing newline).
pub(crate) fn cue_to_bytes(cue: &SubtitleCue, fps: f64) -> Vec<u8> {
    let sf = us_to_frame(cue.start_us, fps);
    let ef = us_to_frame(cue.end_us, fps);
    let mut s = format!("{{{}}}{{{}}}", sf, ef);
    s.push_str(&render_inline(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8], fps: f64) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let line = text
        .lines()
        .map(|l| l.trim_end_matches('\r').trim())
        .find(|l| !l.is_empty())
        .ok_or_else(|| Error::invalid("microdvd: empty cue"))?;
    let (start_frame, end_frame, rest) =
        parse_frame_header(line).ok_or_else(|| Error::invalid("microdvd: bad cue header"))?;
    Ok(SubtitleCue {
        start_us: frame_to_us(start_frame, fps),
        end_us: frame_to_us(end_frame, fps),
        style_ref: None,
        positioning: None,
        segments: parse_inline(rest),
    })
}

// ---------------------------------------------------------------------------
// Header

fn parse_frame_header(line: &str) -> Option<(i64, i64, &str)> {
    let line = line.trim_start();
    let rest = line.strip_prefix('{')?;
    let end1 = rest.find('}')?;
    let f1: i64 = rest[..end1].trim().parse().ok()?;
    let after1 = &rest[end1 + 1..];
    let after1 = after1.strip_prefix('{')?;
    let end2 = after1.find('}')?;
    let f2: i64 = after1[..end2].trim().parse().ok()?;
    let body = &after1[end2 + 1..];
    Some((f1, f2, body))
}

// ---------------------------------------------------------------------------
// Inline tags

fn parse_inline(body: &str) -> Vec<Segment> {
    // Split by `|` first — each piece is an independent styled line, joined
    // with LineBreak.
    let mut out: Vec<Segment> = Vec::new();
    for (idx, piece) in body.split('|').enumerate() {
        if idx > 0 {
            out.push(Segment::LineBreak);
        }
        parse_line(piece, &mut out);
    }
    out
}

fn parse_line(line: &str, out: &mut Vec<Segment>) {
    // Scan for `{...}` tags. A tag like `{y:b}`, `{y:i}`, `{y:u}`,
    // `{Y:b}`, `{c:$BBGGRR}`, `{C:$BBGGRR}`, `{f:...}`, `{s:...}` — the
    // capital versions are "whole-line" and the lowercase are "rest-of-line"
    // in the MicroDVD spec, but practically both flip the remainder of the
    // line. We treat any opener by wrapping the remainder in a single
    // segment.
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut text_buf = String::new();
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(close) = memchr(b'}', &bytes[i + 1..]) {
                let tag = &line[i + 1..i + 1 + close];
                let rest_start = i + 1 + close + 1;
                let rest = &line[rest_start..];
                if !text_buf.is_empty() {
                    out.push(Segment::Text(std::mem::take(&mut text_buf)));
                }
                if let Some(seg) = classify_tag(tag, rest) {
                    out.push(seg);
                    return; // remaining text already consumed inside wrapper
                } else {
                    // Unknown tag — keep verbatim, continue after it.
                    out.push(Segment::Raw(format!("{{{}}}", tag)));
                    i = rest_start;
                    continue;
                }
            } else {
                // Unterminated `{` — treat as plain text.
                text_buf.push('{');
                i += 1;
            }
        } else {
            text_buf.push(bytes[i] as char);
            i += 1;
        }
    }
    if !text_buf.is_empty() {
        out.push(Segment::Text(text_buf));
    }
}

/// Returns `Some(segment)` if this is a recognised opener that consumes
/// the rest of the line, `None` otherwise.
fn classify_tag(tag: &str, rest: &str) -> Option<Segment> {
    let (name, value) = match tag.split_once(':') {
        Some(nv) => nv,
        None => return None,
    };
    let name_lc = name.trim().to_ascii_lowercase();
    match name_lc.as_str() {
        "y" | "s" => {
            // `y:b`, `y:i`, `y:u`, or combinations like `y:bi`.
            let mut children: Vec<Segment> = Vec::new();
            parse_line(rest, &mut children);
            let mut wrapped = children;
            for ch in value.chars() {
                wrapped = match ch.to_ascii_lowercase() {
                    'b' => vec![Segment::Bold(wrapped)],
                    'i' => vec![Segment::Italic(wrapped)],
                    'u' => vec![Segment::Underline(wrapped)],
                    's' => vec![Segment::Strike(wrapped)],
                    _ => wrapped,
                };
            }
            // If we built at least one wrapper, return the outermost.
            wrapped.into_iter().next()
        }
        "c" => {
            // `c:$BBGGRR` — BGR!
            let rgb = parse_bgr(value)?;
            let mut children: Vec<Segment> = Vec::new();
            parse_line(rest, &mut children);
            Some(Segment::Color { rgb, children })
        }
        "f" => {
            // Font family override — `f:Arial`.
            let family = value.trim().to_string();
            let mut children: Vec<Segment> = Vec::new();
            parse_line(rest, &mut children);
            Some(Segment::Font {
                family: if family.is_empty() {
                    None
                } else {
                    Some(family)
                },
                size: None,
                children,
            })
        }
        _ => None,
    }
}

fn parse_bgr(s: &str) -> Option<(u8, u8, u8)> {
    let hex = s.trim().trim_start_matches('$').trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let b = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let r = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}

// ---------------------------------------------------------------------------
// Inline renderer

fn render_inline(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_inline(segments, &mut out);
    out
}

fn append_inline(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => {
                // `|` in the text is our line-break sentinel — replace it
                // to avoid ambiguity.
                out.push_str(&s.replace('|', "/"));
            }
            Segment::LineBreak => out.push('|'),
            Segment::Bold(c) => {
                out.push_str("{y:b}");
                append_inline(c, out);
            }
            Segment::Italic(c) => {
                out.push_str("{y:i}");
                append_inline(c, out);
            }
            Segment::Underline(c) => {
                out.push_str("{y:u}");
                append_inline(c, out);
            }
            Segment::Strike(c) => {
                out.push_str("{y:s}");
                append_inline(c, out);
            }
            Segment::Color { rgb, children } => {
                let (r, g, b) = *rgb;
                out.push_str(&format!("{{c:${:02X}{:02X}{:02X}}}", b, g, r));
                append_inline(children, out);
            }
            Segment::Font {
                family, children, ..
            } => {
                if let Some(f) = family {
                    out.push_str(&format!("{{f:{}}}", f));
                }
                append_inline(children, out);
            }
            Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_inline(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

// ---------------------------------------------------------------------------
// Time

fn frame_to_us(frame: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((frame as f64 / fps) * 1_000_000.0).round() as i64
}

fn us_to_frame(us: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((us as f64 / 1_000_000.0) * fps).round() as i64
}

fn fmt_fps(fps: f64) -> String {
    // Keep three decimals, strip trailing zeros.
    let mut s = format!("{:.3}", fps);
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

// ---------------------------------------------------------------------------
// Misc

fn strip_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

fn extract_fps(extradata: &[u8]) -> f64 {
    // The packet-level codec carries fps via an ASCII extradata blob:
    // `fps=<value>` on a single line. Falls back to DEFAULT_FPS.
    if extradata.is_empty() {
        return DEFAULT_FPS;
    }
    let text = String::from_utf8_lossy(extradata);
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("fps=") {
            if let Ok(f) = v.trim().parse::<f64>() {
                if f > 0.0 {
                    return f;
                }
            }
        }
    }
    DEFAULT_FPS
}

// ---------------------------------------------------------------------------
// Codec wrappers

struct MicroDvdDecoder {
    codec_id: CodecId,
    fps: f64,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for MicroDvdDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let mut cue = bytes_to_cue(&packet.data, self.fps)?;
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

struct MicroDvdEncoder {
    params: CodecParameters,
    fps: f64,
    pending: VecDeque<Packet>,
}

impl Encoder for MicroDvdEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("microdvd encoder: expected Frame::Subtitle")),
        };
        let tb = TimeBase::new(1, 1_000_000);
        let payload = cue_to_bytes(cue, self.fps);
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
    fn parse_simple() {
        let src = "{25}{75}Hello world\n{100}{150}Second line\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, frame_to_us(25, DEFAULT_FPS));
        assert_eq!(t.cues[0].end_us, frame_to_us(75, DEFAULT_FPS));
    }

    #[test]
    fn parse_fps_header() {
        let src = "{1}{1}23.976\n{24}{48}Text\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 1);
        assert_eq!(t.cues[0].start_us, frame_to_us(24, 23.976));
    }

    #[test]
    fn parse_styling_and_linebreak() {
        let src = "{10}{20}{y:i}italic|plain\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 1);
        match &t.cues[0].segments[0] {
            Segment::Italic(_) => {}
            other => panic!("expected italic: {other:?}"),
        }
    }

    #[test]
    fn probe_positive() {
        let s = b"{1}{1}25\n{25}{50}hi\n";
        assert!(probe(s) > 0);
    }

    #[test]
    fn probe_negative() {
        let s = b"random text\nno braces here\n";
        assert_eq!(probe(s), 0);
    }

    #[test]
    fn roundtrip_bgr_color() {
        let src = "{10}{20}{c:$0000FF}red text\n";
        let t = parse(src.as_bytes()).unwrap();
        match &t.cues[0].segments[0] {
            Segment::Color { rgb, .. } => assert_eq!(*rgb, (255, 0, 0)),
            other => panic!("expected Color: {other:?}"),
        }
        let out = write(&t).unwrap();
        let out_s = String::from_utf8(out).unwrap();
        assert!(out_s.contains("{c:$0000FF}"));
    }
}
