//! RealText (.rt) parser and writer.
//!
//! RealText is an HTML-ish subtitle format used by RealMedia. Structure:
//!
//! ```text
//! <window type="generic" duration="30.0" width="320" height="64">
//! <time begin="0.0" end="3.0"/>
//! <font size="2" color="#FF0000">Hello</font><br/>
//! <time begin="3.0" end="5.0"/>
//! World
//! </window>
//! ```
//!
//! `<time begin end/>` elements act as cue boundaries. Content between
//! one `<time .../>` and the next becomes the text of the first cue.
//!
//! Supported inline elements:
//!
//!   `<b>…</b>`, `<i>…</i>`, `<u>…</u>`   — bold / italic / underline
//!   `<font color="#RRGGBB" size="N">…</font>`
//!   `<br/>` / `<br>`                       — line break
//!
//! `<window …>` is captured into extradata so round-trip keeps its
//! attributes. Unknown elements survive as [`Segment::Raw`].

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id for the RealText text-subtitle codec.
pub const CODEC_ID: &str = "realtext";

pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt), // No dedicated enum variant — stand-in.
        ..SubtitleTrack::default()
    };

    // Stash extradata = everything up to and including the `<window ...>`
    // opener so a round-trip keeps window attributes intact.
    let mut i = 0usize;
    let lc = text.to_ascii_lowercase();
    if let Some(win_start) = lc.find("<window") {
        if let Some(end_rel) = text[win_start..].find('>') {
            let header_end = win_start + end_rel + 1;
            track.extradata = text[..header_end].as_bytes().to_vec();
            // Append a trailing newline for neat formatting.
            if !track.extradata.ends_with(b"\n") {
                track.extradata.push(b'\n');
            }
            i = header_end;
        }
    }

    // Walk the rest, splitting into cues at each `<time .../>`.
    let mut pending_start: Option<i64> = None;
    let mut pending_end: Option<i64> = None;
    let mut pending_body = String::new();

    let bytes_s = text.as_bytes();
    while i < bytes_s.len() {
        if bytes_s[i] == b'<' {
            let end_rel = match text[i..].find('>') {
                Some(p) => p,
                None => {
                    pending_body.push_str(&text[i..]);
                    break;
                }
            };
            let tag = &text[i + 1..i + end_rel];
            i += end_rel + 1;

            let tag_name = tag
                .split(|c: char| c.is_whitespace() || c == '/')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();

            match tag_name.as_str() {
                "time" => {
                    // Flush pending, start a new cue.
                    if let (Some(s), Some(e)) = (pending_start, pending_end) {
                        let segments = parse_inline_html(pending_body.trim());
                        track.cues.push(SubtitleCue {
                            start_us: s,
                            end_us: e,
                            style_ref: None,
                            positioning: None,
                            segments,
                        });
                    }
                    pending_body.clear();
                    pending_start = attr_value(tag, "begin").and_then(|v| parse_seconds_ts(&v));
                    pending_end = attr_value(tag, "end").and_then(|v| parse_seconds_ts(&v));
                    // `<time begin dur/>` also supported.
                    if pending_end.is_none() {
                        if let (Some(s), Some(dur)) = (
                            pending_start,
                            attr_value(tag, "dur").and_then(|v| parse_seconds_ts(&v)),
                        ) {
                            pending_end = Some(s + dur);
                        }
                    }
                }
                "window" => {
                    // Opener already in extradata; `</window>` is a stream end.
                }
                _ => {
                    // Part of the cue body — keep the raw tag text.
                    pending_body.push('<');
                    pending_body.push_str(tag);
                    pending_body.push('>');
                }
            }
        } else {
            let next = match text[i..].find('<') {
                Some(p) => i + p,
                None => bytes_s.len(),
            };
            pending_body.push_str(&text[i..next]);
            i = next;
        }
    }

    if let (Some(s), Some(e)) = (pending_start, pending_end) {
        let segments = parse_inline_html(pending_body.trim());
        track.cues.push(SubtitleCue {
            start_us: s,
            end_us: e,
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
        out.push_str("<window>\n");
    }

    for cue in &track.cues {
        out.push_str(&format!(
            "<time begin=\"{}\" end=\"{}\"/>\n",
            format_seconds(cue.start_us),
            format_seconds(cue.end_us)
        ));
        out.push_str(&render_segments(&cue.segments));
        out.push('\n');
    }

    out.push_str("</window>\n");
    Ok(out.into_bytes())
}

pub fn probe(buf: &[u8]) -> u8 {
    if looks_like_realtext(buf) {
        95
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a realtext codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(RtDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not a realtext codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(RtEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

struct RtDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for RtDecoder {
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

struct RtEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for RtEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("realtext encoder: expected Frame::Subtitle")),
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

pub(crate) fn looks_like_realtext(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    let lc = text.to_ascii_lowercase();
    lc.contains("<window") && lc.contains("<time")
}

// ---------------------------------------------------------------------------
// Inline HTML parsing — a minimal subset tailored to RealText.

fn parse_inline_html(body: &str) -> Vec<Segment> {
    let mut parser = RtParser { src: body, pos: 0 };
    parser.parse_until(None)
}

struct RtParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> RtParser<'a> {
    fn parse_until(&mut self, stop: Option<&str>) -> Vec<Segment> {
        let mut out: Vec<Segment> = Vec::new();
        let mut text_buf = String::new();
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() {
            if bytes[self.pos] == b'<' {
                let tag_end = match self.src[self.pos..].find('>') {
                    Some(e) => e,
                    None => {
                        text_buf.push_str(&self.src[self.pos..]);
                        self.pos = bytes.len();
                        continue;
                    }
                };
                let tag = &self.src[self.pos + 1..self.pos + tag_end];
                self.pos += tag_end + 1;

                // Closer?
                let trimmed = tag.trim();
                let is_close = trimmed.starts_with('/');
                let plain_body = if is_close {
                    trimmed[1..].trim_start()
                } else {
                    trimmed
                };
                let self_closing = trimmed.ends_with('/');
                let (name, _rest) = split_tag_name(plain_body);
                let name_lc = name.to_ascii_lowercase();

                if is_close {
                    if let Some(stop_name) = stop {
                        if name_lc == stop_name {
                            if !text_buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut text_buf)));
                            }
                            return out;
                        }
                    }
                    // Stray closer — drop.
                    continue;
                }

                if !text_buf.is_empty() {
                    out.push(Segment::Text(std::mem::take(&mut text_buf)));
                }

                match name_lc.as_str() {
                    "br" => out.push(Segment::LineBreak),
                    "b" => {
                        if self_closing {
                            continue;
                        }
                        let children = self.parse_until(Some("b"));
                        out.push(Segment::Bold(children));
                    }
                    "i" => {
                        if self_closing {
                            continue;
                        }
                        let children = self.parse_until(Some("i"));
                        out.push(Segment::Italic(children));
                    }
                    "u" => {
                        if self_closing {
                            continue;
                        }
                        let children = self.parse_until(Some("u"));
                        out.push(Segment::Underline(children));
                    }
                    "font" => {
                        if self_closing {
                            continue;
                        }
                        let children = self.parse_until(Some("font"));
                        out.push(classify_font(tag, children));
                    }
                    _ => {
                        out.push(Segment::Raw(format!("<{}>", tag)));
                    }
                }
            } else {
                text_buf.push(bytes[self.pos] as char);
                self.pos += 1;
            }
        }
        if !text_buf.is_empty() {
            out.push(Segment::Text(text_buf));
        }
        out
    }
}

fn split_tag_name(tag: &str) -> (&str, &str) {
    let t = tag.trim().trim_end_matches('/').trim_end();
    match t.find(char::is_whitespace) {
        Some(i) => (&t[..i], t[i..].trim()),
        None => (t, ""),
    }
}

fn classify_font(tag: &str, children: Vec<Segment>) -> Segment {
    if let Some(col) = attr_value(tag, "color") {
        if let Some(rgb) = parse_hex_color(&col) {
            return Segment::Color { rgb, children };
        }
    }
    let family = attr_value(tag, "face");
    let size = attr_value(tag, "size").and_then(|s| s.parse::<f32>().ok());
    Segment::Font {
        family,
        size,
        children,
    }
}

/// Extract `name="..."` / `name='...'` / `name=bareword` from a tag body.
fn attr_value(tag: &str, name: &str) -> Option<String> {
    let name_lc = name.to_ascii_lowercase();
    let bytes = tag.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        // Read a key (alphanumerics + `-`).
        let key_start = i;
        while i < bytes.len() && {
            let c = bytes[i] as char;
            c.is_ascii_alphanumeric() || c == '-' || c == '_'
        } {
            i += 1;
        }
        if key_start == i {
            // No key — maybe a `/` or `>` we should skip over.
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        let key = &tag[key_start..i];
        let key_lc = key.to_ascii_lowercase();
        // Skip `=`?
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            // Attribute without a value — skip.
            continue;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        // Read the value.
        let (val, next) = if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
            let q = bytes[i];
            let val_start = i + 1;
            let mut j = val_start;
            while j < bytes.len() && bytes[j] != q {
                j += 1;
            }
            (&tag[val_start..j], j.saturating_add(1).min(bytes.len()))
        } else {
            let val_start = i;
            let mut j = val_start;
            while j < bytes.len() {
                let c = bytes[j] as char;
                if c.is_whitespace() || c == '>' || c == '/' {
                    break;
                }
                j += 1;
            }
            (&tag[val_start..j], j)
        };
        if key_lc == name_lc {
            return Some(val.to_string());
        }
        i = next;
    }
    None
}

fn parse_hex_color(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim().trim_matches(|c: char| c == '"' || c == '\'');
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
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
            Segment::LineBreak => out.push_str("<br/>"),
            Segment::Bold(c) => {
                out.push_str("<b>");
                append_segments(c, out);
                out.push_str("</b>");
            }
            Segment::Italic(c) => {
                out.push_str("<i>");
                append_segments(c, out);
                out.push_str("</i>");
            }
            Segment::Underline(c) => {
                out.push_str("<u>");
                append_segments(c, out);
                out.push_str("</u>");
            }
            Segment::Strike(c) => append_segments(c, out),
            Segment::Color { rgb, children } => {
                out.push_str(&format!(
                    "<font color=\"#{:02X}{:02X}{:02X}\">",
                    rgb.0, rgb.1, rgb.2
                ));
                append_segments(children, out);
                out.push_str("</font>");
            }
            Segment::Font {
                family,
                size,
                children,
            } => {
                let mut open = String::from("<font");
                if let Some(f) = family {
                    open.push_str(&format!(" face=\"{}\"", f));
                }
                if let Some(sz) = size {
                    open.push_str(&format!(" size=\"{}\"", sz));
                }
                open.push('>');
                out.push_str(&open);
                append_segments(children, out);
                out.push_str("</font>");
            }
            Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_segments(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

// ---------------------------------------------------------------------------
// Time helpers — RealText times are plain decimal seconds.

fn parse_seconds_ts(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.contains(':') {
        let (hms, frac) = match s.find('.') {
            Some(i) => (&s[..i], &s[i + 1..]),
            None => (s, ""),
        };
        let parts: Vec<&str> = hms.split(':').collect();
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
            _ => return None,
        };
        let base = (h * 3600 + m * 60 + sec) * 1_000_000;
        let frac_us = parse_fraction_us(frac);
        return Some(base + frac_us);
    }
    let f: f64 = s.parse().ok()?;
    Some((f * 1_000_000.0).round() as i64)
}

fn parse_fraction_us(frac: &str) -> i64 {
    if frac.is_empty() {
        return 0;
    }
    let f = format!("0.{}", frac);
    let v: f64 = f.parse().unwrap_or(0.0);
    (v * 1_000_000.0).round() as i64
}

fn format_seconds(us: i64) -> String {
    let us = us.max(0);
    let whole = us / 1_000_000;
    let frac = (us % 1_000_000) / 1_000; // 3 decimal digits
    if frac == 0 {
        format!("{}.0", whole)
    } else {
        format!("{}.{:03}", whole, frac)
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
    s.push_str(&format!(
        "<time begin=\"{}\" end=\"{}\"/>",
        format_seconds(cue.start_us),
        format_seconds(cue.end_us)
    ));
    s.push_str(&render_segments(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let lc = text.to_ascii_lowercase();
    let t_start = lc
        .find("<time")
        .ok_or_else(|| Error::invalid("RealText: missing <time> in cue payload"))?;
    let t_end_rel = text[t_start..]
        .find('>')
        .ok_or_else(|| Error::invalid("RealText: unclosed <time> tag"))?;
    let tag = &text[t_start + 1..t_start + t_end_rel];
    let begin = attr_value(tag, "begin")
        .and_then(|v| parse_seconds_ts(&v))
        .ok_or_else(|| Error::invalid("RealText: <time> missing begin"))?;
    let end = attr_value(tag, "end")
        .and_then(|v| parse_seconds_ts(&v))
        .or_else(|| {
            attr_value(tag, "dur")
                .and_then(|v| parse_seconds_ts(&v))
                .map(|dur| begin + dur)
        })
        .ok_or_else(|| Error::invalid("RealText: <time> missing end/dur"))?;
    let body = &text[t_start + t_end_rel + 1..];
    let segments = parse_inline_html(body.trim());
    Ok(SubtitleCue {
        start_us: begin,
        end_us: end,
        style_ref: None,
        positioning: None,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "<window type=\"generic\" duration=\"30.0\">
<time begin=\"0.0\" end=\"3.0\"/>
<font color=\"#FF0000\">Hello</font><br/>
<time begin=\"3.0\" end=\"5.0\"/>
World
</window>
";

    #[test]
    fn parses_two_cues() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 0);
        assert_eq!(t.cues[0].end_us, 3_000_000);
        assert_eq!(t.cues[1].start_us, 3_000_000);
    }

    #[test]
    fn parses_font_color() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let mut saw = false;
        for s in &t.cues[0].segments {
            if let Segment::Color { rgb, .. } = s {
                assert_eq!(*rgb, (0xFF, 0, 0));
                saw = true;
            }
        }
        assert!(saw, "expected color segment");
    }

    #[test]
    fn roundtrips() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let out = write(&t).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("<time begin="));
        assert!(s.contains("</window>"));
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
