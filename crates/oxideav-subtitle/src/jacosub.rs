//! JACOsub (.jss, .js) parser and writer.
//!
//! JACOsub is an old Japanese-Anime-Club subtitle format. Structure:
//!
//! ```text
//! # This is a comment line (header area).
//! #TITLE Some title
//! #AUTHOR Somebody
//! #TIMERES 100            ; fractional units per second (default 30)
//! #SHIFT 0                ; global offset in time-res units
//!
//! @0:00:01.00 0:00:03.00 D hello \Bworld\b
//! @0:00:04.00 0:00:06.00 D second cue\nwith a line break
//! ```
//!
//! Each cue line begins with `@` and carries `start end [directive] text`.
//! The fractional part of each timestamp is in units of `1 / TIMERES`
//! seconds (so with `TIMERES 100` the `.00` is centiseconds, with
//! `TIMERES 30` it's 1/30s frames).
//!
//! Supported inline tags:
//!
//!   `\B` / `\b` — bold on / off
//!   `\I` / `\i` — italic on / off
//!   `\U` / `\u` — underline on / off
//!   `\n`        — line break
//!
//! Anything we don't recognise (`\C` colour directives, `{y:...}` ASS-like
//! overrides, timing shift cues, etc.) survives as [`Segment::Raw`].

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id for the JACOsub text-subtitle codec.
pub const CODEC_ID: &str = "jacosub";

/// Default TIMERES when no header directive is present.
const DEFAULT_TIMERES: u32 = 30;

/// Parse a UTF-8 JACOsub payload into a track.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt), // No dedicated enum variant — closest stand-in.
        ..SubtitleTrack::default()
    };

    let mut timeres: u32 = DEFAULT_TIMERES;
    let mut shift_units: i64 = 0;

    // Collect extradata (the full header block, up to but not including the
    // first `@...` cue). This keeps remux faithful.
    let mut extradata = String::new();
    let mut in_body = false;

    for line_raw in text.split('\n') {
        let line = line_raw.trim_end_matches('\r');
        let trimmed = line.trim();

        // Cue lines start with `@`.
        if trimmed.starts_with('@') {
            in_body = true;
            if let Some(cue) = parse_cue_line(trimmed, timeres, shift_units) {
                track.cues.push(cue);
            }
            continue;
        }

        if !in_body {
            extradata.push_str(line);
            extradata.push('\n');
        }

        if trimmed.is_empty() {
            continue;
        }

        // Header directives (start with `#`).
        if let Some(rest) = trimmed.strip_prefix('#') {
            let rest = rest.trim_start();
            // `#TIMERES 100`, `#SHIFT 0`, `#TITLE ...`, etc.
            let (key, value) = split_first_word(rest);
            let key_lc = key.to_ascii_lowercase();
            match key_lc.as_str() {
                "timeres" => {
                    if let Ok(v) = value.trim().parse::<u32>() {
                        if v > 0 {
                            timeres = v;
                        }
                    }
                }
                "shift" => {
                    if let Ok(v) = value.trim().parse::<i64>() {
                        shift_units = v;
                    }
                }
                "title" | "author" | "comment" | "source" | "director" | "prg" | "qtitle" => {
                    if !value.is_empty() {
                        track.metadata.push((key_lc, value.trim().to_string()));
                    }
                }
                _ => {
                    // Ignore unknown directives — the extradata keeps the raw form.
                }
            }
            continue;
        }

        // Any other body-preamble line is ignored; already captured in extradata.
    }

    track.extradata = extradata.into_bytes();
    // Stash TIMERES so the writer can mirror it even when we didn't see one.
    track
        .metadata
        .push(("timeres".to_string(), timeres.to_string()));
    if shift_units != 0 {
        track
            .metadata
            .push(("shift".to_string(), shift_units.to_string()));
    }
    Ok(track)
}

/// Re-emit a track as JACOsub bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let mut out = String::new();

    // Prefer to replay the captured extradata verbatim if it was previously
    // parsed from JACOsub. It holds every `#` directive the source file had.
    if !track.extradata.is_empty() {
        out.push_str(&String::from_utf8_lossy(&track.extradata));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    } else {
        // Synthesise a minimal header.
        if let Some(title) = track.metadata.iter().find(|(k, _)| k == "title") {
            out.push_str(&format!("#TITLE {}\n", title.1));
        }
        if let Some(author) = track.metadata.iter().find(|(k, _)| k == "author") {
            out.push_str(&format!("#AUTHOR {}\n", author.1));
        }
        let timeres = track
            .metadata
            .iter()
            .find(|(k, _)| k == "timeres")
            .and_then(|(_, v)| v.parse::<u32>().ok())
            .unwrap_or(100);
        out.push_str(&format!("#TIMERES {}\n", timeres));
        out.push('\n');
    }

    // Use the same TIMERES the header declared (or 100 if synthesised).
    let timeres = track
        .metadata
        .iter()
        .find(|(k, _)| k == "timeres")
        .and_then(|(_, v)| v.parse::<u32>().ok())
        .unwrap_or(100);

    for cue in &track.cues {
        out.push('@');
        out.push_str(&format_ts(cue.start_us.max(0), timeres));
        out.push(' ');
        out.push_str(&format_ts(cue.end_us.max(0), timeres));
        out.push_str(" D ");
        out.push_str(&render_segments(&cue.segments));
        out.push('\n');
    }

    Ok(out.into_bytes())
}

/// Quick header check — did the buffer look like JACOsub?
pub fn probe(buf: &[u8]) -> u8 {
    if looks_like_jacosub(buf) {
        85
    } else {
        0
    }
}

/// Factory for a codec-registry decoder binding.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a jacosub codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(JacosubDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Factory for a codec-registry encoder binding.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a jacosub codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(JacosubEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

struct JacosubDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for JacosubDecoder {
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

struct JacosubEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for JacosubEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("jacosub encoder: expected Frame::Subtitle")),
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

pub(crate) fn looks_like_jacosub(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    let mut saw_header = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('@') {
            // Must look like `@HH:MM:SS.FF HH:MM:SS.FF ...`.
            let body = &trimmed[1..];
            let mut parts = body.split_whitespace();
            let a = match parts.next() {
                Some(v) => v,
                None => return false,
            };
            let b = match parts.next() {
                Some(v) => v,
                None => return false,
            };
            return parse_jss_timestamp(a, DEFAULT_TIMERES).is_some()
                && parse_jss_timestamp(b, DEFAULT_TIMERES).is_some();
        }
        if trimmed.starts_with('#') {
            let rest = trimmed[1..].trim_start().to_ascii_lowercase();
            if rest.starts_with("title ")
                || rest.starts_with("timeres ")
                || rest.starts_with("author ")
                || rest.starts_with("shift ")
                || rest.starts_with("prg ")
                || rest.starts_with("qtitle ")
            {
                saw_header = true;
                continue;
            }
            continue;
        }
        // Something else before a cue — non-JACOsub.
        if saw_header {
            continue;
        }
        return false;
    }
    saw_header
}

// ---------------------------------------------------------------------------
// Cue line parsing.

fn parse_cue_line(line: &str, timeres: u32, shift_units: i64) -> Option<SubtitleCue> {
    // Drop leading `@`.
    let body = line.strip_prefix('@')?.trim_start();
    // Whitespace-separated: start end [directive] text...
    let (start_s, rest) = split_first_word(body);
    if start_s.is_empty() {
        return None;
    }
    let (end_s, rest) = split_first_word(rest);
    if end_s.is_empty() {
        return None;
    }
    let start_units = parse_jss_timestamp(start_s, timeres)?;
    let end_units = parse_jss_timestamp(end_s, timeres)?;

    // Apply global SHIFT (in TIMERES units).
    let sec_per_unit_us = 1_000_000i64 / timeres as i64;
    let start_us = (start_units + shift_units) * sec_per_unit_us;
    let end_us = (end_units + shift_units) * sec_per_unit_us;

    // The remainder may start with a directive (D, I, T, ...). `D` = dialogue
    // (default); everything we don't explicitly recognise is also treated as
    // a dialogue body — the directive char is preserved into the text only
    // when it wasn't a known control flag.
    let (maybe_dir, after_dir) = split_first_word(rest);
    let (directive_consumed, text) = if is_known_directive(maybe_dir) {
        (true, after_dir.trim_start().to_string())
    } else {
        (false, rest.to_string())
    };
    let _ = directive_consumed;

    let segments = parse_inline_tags(&text);

    Some(SubtitleCue {
        start_us,
        end_us,
        style_ref: None,
        positioning: None,
        segments,
    })
}

fn is_known_directive(tok: &str) -> bool {
    if tok.len() != 1 {
        return false;
    }
    matches!(
        tok.chars().next().unwrap(),
        'D' | 'I' | 'T' | 'C' | 'd' | 'i' | 't' | 'c'
    )
}

/// Parse a timestamp of shape `HH:MM:SS.FF` — where `FF` is a fractional in
/// units of `1/timeres` seconds. Returns the total count of `1/timeres`-units
/// from zero. A `SS` with no fractional part is tolerated.
fn parse_jss_timestamp(s: &str, timeres: u32) -> Option<i64> {
    let (whole, frac) = match s.find('.') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let parts: Vec<&str> = whole.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (
            parts[0].parse::<i64>().ok()?,
            parts[1].parse::<i64>().ok()?,
            parts[2].parse::<i64>().ok()?,
        ),
        2 => (
            0i64,
            parts[0].parse::<i64>().ok()?,
            parts[1].parse::<i64>().ok()?,
        ),
        1 => (0i64, 0i64, parts[0].parse::<i64>().ok()?),
        _ => return None,
    };
    let frac_units: i64 = if frac.is_empty() {
        0
    } else {
        frac.parse::<i64>().ok()?
    };
    let total_seconds = h * 3600 + m * 60 + sec;
    Some(total_seconds * timeres as i64 + frac_units)
}

fn format_ts(us: i64, timeres: u32) -> String {
    let total_units = (us as i128 * timeres as i128) / 1_000_000;
    let total_units = total_units.max(0) as i64;
    let units_per_sec = timeres as i64;
    let frac = total_units % units_per_sec;
    let secs_total = total_units / units_per_sec;
    let s = (secs_total % 60) as u32;
    let m = ((secs_total / 60) % 60) as u32;
    let h = (secs_total / 3600) as u32;
    // Width of fraction is log10(timeres) ceiling (2 for 100, 2 for 30, etc.).
    let width = if timeres >= 10 { 2 } else { 1 };
    format!("{}:{:02}:{:02}.{:0width$}", h, m, s, frac, width = width)
}

// ---------------------------------------------------------------------------
// Inline tag parsing.

fn parse_inline_tags(body: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut stack: Vec<Wrap> = Vec::new();
    let mut buf = String::new();
    let mut chars = body.chars().peekable();

    fn flush(out: &mut Vec<Segment>, stack: &mut Vec<Wrap>, buf: &mut String) {
        if buf.is_empty() {
            return;
        }
        let text = std::mem::take(buf);
        push_into(out, stack, Segment::Text(text));
    }

    while let Some(c) = chars.next() {
        if c == '\\' {
            let next = match chars.next() {
                Some(v) => v,
                None => {
                    buf.push('\\');
                    break;
                }
            };
            match next {
                'n' | 'N' => {
                    flush(&mut out, &mut stack, &mut buf);
                    push_into(&mut out, &mut stack, Segment::LineBreak);
                }
                'B' => {
                    flush(&mut out, &mut stack, &mut buf);
                    stack.push(Wrap::Bold(Vec::new()));
                }
                'b' => {
                    flush(&mut out, &mut stack, &mut buf);
                    pop_wrap(&mut out, &mut stack, WrapKind::Bold);
                }
                'I' => {
                    flush(&mut out, &mut stack, &mut buf);
                    stack.push(Wrap::Italic(Vec::new()));
                }
                'i' => {
                    flush(&mut out, &mut stack, &mut buf);
                    pop_wrap(&mut out, &mut stack, WrapKind::Italic);
                }
                'U' => {
                    flush(&mut out, &mut stack, &mut buf);
                    stack.push(Wrap::Underline(Vec::new()));
                }
                'u' => {
                    flush(&mut out, &mut stack, &mut buf);
                    pop_wrap(&mut out, &mut stack, WrapKind::Underline);
                }
                other => {
                    // Unrecognised escape (e.g. `\C` colour, `\T` tag): preserve as raw
                    // so round-trip doesn't lose it.
                    flush(&mut out, &mut stack, &mut buf);
                    let raw = format!("\\{}", other);
                    push_into(&mut out, &mut stack, Segment::Raw(raw));
                }
            }
            continue;
        }
        buf.push(c);
    }
    flush(&mut out, &mut stack, &mut buf);

    // Any still-open wraps become flat wrappers around what they collected.
    while let Some(w) = stack.pop() {
        let seg = w.finish();
        if stack.is_empty() {
            out.push(seg);
        } else {
            match stack.last_mut().unwrap() {
                Wrap::Bold(v) | Wrap::Italic(v) | Wrap::Underline(v) => v.push(seg),
            }
        }
    }
    out
}

enum Wrap {
    Bold(Vec<Segment>),
    Italic(Vec<Segment>),
    Underline(Vec<Segment>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WrapKind {
    Bold,
    Italic,
    Underline,
}

impl Wrap {
    fn kind(&self) -> WrapKind {
        match self {
            Wrap::Bold(_) => WrapKind::Bold,
            Wrap::Italic(_) => WrapKind::Italic,
            Wrap::Underline(_) => WrapKind::Underline,
        }
    }
    fn finish(self) -> Segment {
        match self {
            Wrap::Bold(v) => Segment::Bold(v),
            Wrap::Italic(v) => Segment::Italic(v),
            Wrap::Underline(v) => Segment::Underline(v),
        }
    }
}

fn push_into(out: &mut Vec<Segment>, stack: &mut [Wrap], seg: Segment) {
    if let Some(top) = stack.last_mut() {
        match top {
            Wrap::Bold(v) | Wrap::Italic(v) | Wrap::Underline(v) => v.push(seg),
        }
    } else {
        out.push(seg);
    }
}

fn pop_wrap(out: &mut Vec<Segment>, stack: &mut Vec<Wrap>, kind: WrapKind) {
    // Pop the innermost wrap of matching kind. If nothing matches, silently
    // drop the close (malformed input).
    let Some(idx) = stack.iter().rposition(|w| w.kind() == kind) else {
        return;
    };
    // Close everything above `idx` in order.
    while stack.len() > idx + 1 {
        let inner = stack.pop().unwrap().finish();
        match stack.last_mut() {
            Some(Wrap::Bold(v)) | Some(Wrap::Italic(v)) | Some(Wrap::Underline(v)) => v.push(inner),
            None => out.push(inner),
        }
    }
    // Now pop the target.
    let target = stack.pop().unwrap().finish();
    if let Some(parent) = stack.last_mut() {
        match parent {
            Wrap::Bold(v) | Wrap::Italic(v) | Wrap::Underline(v) => v.push(target),
        }
    } else {
        out.push(target);
    }
}

fn render_segments(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_segments(segments, &mut out);
    out
}

fn append_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => out.push_str(s),
            Segment::LineBreak => out.push_str("\\n"),
            Segment::Bold(c) => {
                out.push_str("\\B");
                append_segments(c, out);
                out.push_str("\\b");
            }
            Segment::Italic(c) => {
                out.push_str("\\I");
                append_segments(c, out);
                out.push_str("\\i");
            }
            Segment::Underline(c) => {
                out.push_str("\\U");
                append_segments(c, out);
                out.push_str("\\u");
            }
            Segment::Strike(c)
            | Segment::Color { children: c, .. }
            | Segment::Font { children: c, .. }
            | Segment::Voice { children: c, .. }
            | Segment::Class { children: c, .. }
            | Segment::Karaoke { children: c, .. } => {
                // Not representable in JACOsub — flatten.
                append_segments(c, out);
            }
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

// ---------------------------------------------------------------------------
// Packet helpers (mirror srt::cue_to_bytes / bytes_to_cue shape).

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let mut s = String::new();
    // TIMERES 100 for packet form (gives us centiseconds).
    s.push('@');
    s.push_str(&format_ts(cue.start_us, 100));
    s.push(' ');
    s.push_str(&format_ts(cue.end_us, 100));
    s.push_str(" D ");
    s.push_str(&render_segments(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('@') {
            if let Some(c) = parse_cue_line(trimmed, 100, 0) {
                return Ok(c);
            }
        }
    }
    Err(Error::invalid("JACOsub: no cue in payload"))
}

// ---------------------------------------------------------------------------

fn decode_utf8_lossy_stripping_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

fn split_first_word(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], s[i..].trim_start()),
        None => (s, ""),
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "#TITLE Demo
#AUTHOR Me
#TIMERES 100
#SHIFT 0

@0:00:01.00 0:00:03.00 D hello
@0:00:04.00 0:00:06.00 D \\Bbold\\b world
";

    #[test]
    fn parses_simple() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_000_000);
    }

    #[test]
    fn roundtrip_contains_cue() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let out = write(&t).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("@0:00:01.00 0:00:03.00 D hello"), "got: {s}");
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
    // Type-check helper — keeps the file compiling standalone even before
    // lib.rs wires it into the main module tree.
    use super::*;
    fn _types(b: &[u8]) -> bool {
        probe(b) > 0 && parse(b).is_ok() && write(&SubtitleTrack::default()).is_ok()
    }
}
