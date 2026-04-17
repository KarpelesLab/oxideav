//! TTML (W3C Timed Text Markup Language) parser and writer.
//!
//! Handles TTML v1 and v2 XML. Elements:
//!
//! * `<tt>` — root
//! * `<head>`, `<styling>`, `<style>` — named style table
//! * `<body>`, `<div>` — structural containers
//! * `<p>` — a cue (begin/end/dur attributes + inline children)
//! * `<span>` — inline span with styling attributes (may be nested)
//! * `<br/>` — line break
//!
//! Timing attributes: `begin` / `end` / `dur` accept `HH:MM:SS.mmm`,
//! `HH:MM:SS`, `HH:MM:SS:FF` (frames truncated — treated as 0), `<n>s`,
//! `<n>ms`, `<n>m`, `<n>h` shorthand. Styling attributes in the TTML
//! styling namespace (`tts:`): `color`, `backgroundColor`, `fontFamily`,
//! `fontSize`, `fontWeight`, `fontStyle`, `textDecoration`.
//!
//! The XML parser is a tiny hand-rolled one — no deps.

use oxideav_core::{Error, Result, Segment, SubtitleCue, SubtitleStyle};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id string.
pub const CODEC_ID: &str = "ttml";

/// Parse a TTML payload into a [`SubtitleTrack`].
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let nodes = parse_xml(&text)?;
    let tt = find_element(&nodes, "tt").ok_or_else(|| Error::invalid("TTML: missing <tt> root"))?;

    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt), // closest stable enum — rewritten below
        ..SubtitleTrack::default()
    };
    // Override to a more appropriate flavour in metadata (we don't have a
    // TTML variant in SourceFormat yet).
    track.metadata.push(("source_format".into(), "ttml".into()));

    // Capture xml:lang for round-trip.
    if let Some(lang) = tt.attrs.iter().find(|(k, _)| k == "xml:lang") {
        track.metadata.push(("xml:lang".into(), lang.1.clone()));
    }

    // Parse styles out of <head><styling><style .../></styling></head>.
    if let Some(head) = find_element(&tt.children, "head") {
        if let Some(styling) = find_element(&head.children, "styling") {
            for child in &styling.children {
                if let Node::Element(e) = child {
                    if tag_local(&e.name) == "style" {
                        if let Some(s) = build_style(e) {
                            track.styles.push(s);
                        }
                    }
                }
            }
        }
    }

    // Walk <body> collecting <p> cues (optionally nested in <div>s).
    if let Some(body) = find_element(&tt.children, "body") {
        collect_cues(&body.children, &mut track, 0);
    }

    // Keep the original source as extradata so round-trip can replay the
    // header style table when re-emitting.
    track.extradata = text.into_bytes();

    Ok(track)
}

/// Write a track as a minimal TTML document.
pub fn write(track: &SubtitleTrack) -> Vec<u8> {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    let lang = track
        .metadata
        .iter()
        .find(|(k, _)| k == "xml:lang")
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| "en".into());
    out.push_str(&format!(
        "<tt xmlns=\"http://www.w3.org/ns/ttml\" xmlns:tts=\"http://www.w3.org/ns/ttml#styling\" xml:lang=\"{}\">\n",
        escape_attr(&lang)
    ));

    // Head / styling.
    if !track.styles.is_empty() {
        out.push_str("  <head>\n    <styling>\n");
        for s in &track.styles {
            out.push_str(&format!("      <style xml:id=\"{}\"", escape_attr(&s.name)));
            if let Some((r, g, b, a)) = s.primary_color {
                out.push_str(&format!(
                    " tts:color=\"#{:02X}{:02X}{:02X}{:02X}\"",
                    r, g, b, a
                ));
            }
            if let Some((r, g, b, a)) = s.back_color {
                out.push_str(&format!(
                    " tts:backgroundColor=\"#{:02X}{:02X}{:02X}{:02X}\"",
                    r, g, b, a
                ));
            }
            if let Some(fam) = &s.font_family {
                out.push_str(&format!(" tts:fontFamily=\"{}\"", escape_attr(fam)));
            }
            if let Some(sz) = s.font_size {
                out.push_str(&format!(" tts:fontSize=\"{}px\"", sz));
            }
            if s.bold {
                out.push_str(" tts:fontWeight=\"bold\"");
            }
            if s.italic {
                out.push_str(" tts:fontStyle=\"italic\"");
            }
            if s.underline {
                out.push_str(" tts:textDecoration=\"underline\"");
            }
            out.push_str("/>\n");
        }
        out.push_str("    </styling>\n  </head>\n");
    }

    out.push_str("  <body>\n    <div>\n");
    for cue in &track.cues {
        out.push_str("      <p");
        out.push_str(&format!(" begin=\"{}\"", format_ts(cue.start_us)));
        out.push_str(&format!(" end=\"{}\"", format_ts(cue.end_us)));
        if let Some(s) = &cue.style_ref {
            out.push_str(&format!(" style=\"{}\"", escape_attr(s)));
        }
        out.push('>');
        write_segments(&cue.segments, &mut out);
        out.push_str("</p>\n");
    }
    out.push_str("    </div>\n  </body>\n");
    out.push_str("</tt>\n");
    out.into_bytes()
}

/// Probe — returns a confidence score (0..=100).
pub fn probe(buf: &[u8]) -> u8 {
    looks_like_ttml(buf)
}

/// Containers dispatch — return the score used by this format's probe.
pub fn looks_like_ttml(buf: &[u8]) -> u8 {
    let head = &buf[..buf.len().min(4096)];
    let text = String::from_utf8_lossy(head);
    let lc = text.to_ascii_lowercase();
    let mut score: u8 = 0;
    if lc.contains("<?xml") {
        score += 15;
    }
    if lc.contains("<tt ") || lc.contains("<tt>") || lc.contains(":tt ") {
        score += 40;
    }
    if lc.contains("http://www.w3.org/ns/ttml") {
        score += 45;
    }
    if lc.contains("tts:") {
        score = score.saturating_add(10);
    }
    score.min(100)
}

/// Decoder factory. Delegates to [`crate::codec::make_decoder`] when
/// wired by lib.rs — this stub satisfies the required surface area when
/// the caller hasn't plugged the codec switch yet.
pub fn make_decoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Decoder>> {
    crate::codec::make_decoder(params)
}

/// Encoder factory — same shape as [`make_decoder`].
pub fn make_encoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Encoder>> {
    crate::codec::make_encoder(params)
}

// ---------------------------------------------------------------------------
// Cue assembly.

fn collect_cues(nodes: &[Node], track: &mut SubtitleTrack, parent_offset_us: i64) {
    for node in nodes {
        if let Node::Element(e) = node {
            let local = tag_local(&e.name);
            match local.as_str() {
                "div" => {
                    let begin = attr(e, "begin")
                        .and_then(|v| parse_ttml_time(&v))
                        .unwrap_or(0);
                    collect_cues(&e.children, track, parent_offset_us + begin);
                }
                "p" => {
                    let begin = attr(e, "begin")
                        .and_then(|v| parse_ttml_time(&v))
                        .unwrap_or(0);
                    let end_attr = attr(e, "end").and_then(|v| parse_ttml_time(&v));
                    let dur_attr = attr(e, "dur").and_then(|v| parse_ttml_time(&v));
                    let start_us = parent_offset_us + begin;
                    let end_us = if let Some(e_us) = end_attr {
                        parent_offset_us + e_us
                    } else if let Some(d) = dur_attr {
                        start_us + d
                    } else {
                        start_us
                    };
                    let style_ref = attr(e, "style");
                    let segments = collect_segments(&e.children);
                    track.cues.push(SubtitleCue {
                        start_us,
                        end_us,
                        style_ref,
                        positioning: None,
                        segments,
                    });
                }
                _ => {
                    // Unknown structural element — recurse.
                    collect_cues(&e.children, track, parent_offset_us);
                }
            }
        }
    }
}

fn collect_segments(nodes: &[Node]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for node in nodes {
        match node {
            Node::Text(s) => {
                if !s.is_empty() {
                    out.push(Segment::Text(s.clone()));
                }
            }
            Node::Element(e) => {
                let local = tag_local(&e.name);
                match local.as_str() {
                    "br" => out.push(Segment::LineBreak),
                    "span" => {
                        let children = collect_segments(&e.children);
                        out.push(wrap_with_style(e, children));
                    }
                    _ => {
                        // Unknown inline element — flatten children.
                        out.extend(collect_segments(&e.children));
                    }
                }
            }
        }
    }
    out
}

/// Wrap `children` based on the styling attributes (`tts:color`, etc.) on
/// `el`. Emits the tightest matching [`Segment`] variants.
fn wrap_with_style(el: &Element, mut children: Vec<Segment>) -> Segment {
    let weight = attr(el, "tts:fontWeight").unwrap_or_default();
    let style_a = attr(el, "tts:fontStyle").unwrap_or_default();
    let deco = attr(el, "tts:textDecoration").unwrap_or_default();
    let color = attr(el, "tts:color");
    let fam = attr(el, "tts:fontFamily");
    let sz = attr(el, "tts:fontSize").and_then(|v| {
        v.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
            .parse::<f32>()
            .ok()
    });

    if weight.eq_ignore_ascii_case("bold") {
        children = vec![Segment::Bold(children)];
    }
    if style_a.eq_ignore_ascii_case("italic") || style_a.eq_ignore_ascii_case("oblique") {
        children = vec![Segment::Italic(children)];
    }
    let deco_lc = deco.to_ascii_lowercase();
    if deco_lc.contains("underline") {
        children = vec![Segment::Underline(children)];
    }
    if deco_lc.contains("line-through") || deco_lc.contains("strike") {
        children = vec![Segment::Strike(children)];
    }
    if let Some(c) = color {
        if let Some(rgb) = parse_ttml_color_rgb(&c) {
            children = vec![Segment::Color { rgb, children }];
        }
    }
    if fam.is_some() || sz.is_some() {
        children = vec![Segment::Font {
            family: fam,
            size: sz,
            children,
        }];
    }
    if children.len() == 1 {
        children.pop().unwrap()
    } else {
        // Wrap in a Font with no attrs so callers still traverse.
        Segment::Font {
            family: None,
            size: None,
            children,
        }
    }
}

fn build_style(e: &Element) -> Option<SubtitleStyle> {
    let id = attr(e, "xml:id").or_else(|| attr(e, "id"))?;
    let mut s = SubtitleStyle::new(id);
    if let Some(c) = attr(e, "tts:color") {
        s.primary_color = parse_ttml_color_rgba(&c);
    }
    if let Some(c) = attr(e, "tts:backgroundColor") {
        s.back_color = parse_ttml_color_rgba(&c);
    }
    if let Some(f) = attr(e, "tts:fontFamily") {
        s.font_family = Some(f);
    }
    if let Some(v) = attr(e, "tts:fontSize") {
        let num: String = v
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        s.font_size = num.parse::<f32>().ok();
    }
    if attr(e, "tts:fontWeight")
        .map(|v| v.eq_ignore_ascii_case("bold"))
        .unwrap_or(false)
    {
        s.bold = true;
    }
    if attr(e, "tts:fontStyle")
        .map(|v| v.eq_ignore_ascii_case("italic") || v.eq_ignore_ascii_case("oblique"))
        .unwrap_or(false)
    {
        s.italic = true;
    }
    if let Some(d) = attr(e, "tts:textDecoration") {
        let lc = d.to_ascii_lowercase();
        if lc.contains("underline") {
            s.underline = true;
        }
        if lc.contains("line-through") || lc.contains("strike") {
            s.strike = true;
        }
    }
    Some(s)
}

fn write_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => out.push_str(&escape_text(s)),
            Segment::LineBreak => out.push_str("<br/>"),
            Segment::Bold(c) => {
                out.push_str("<span tts:fontWeight=\"bold\">");
                write_segments(c, out);
                out.push_str("</span>");
            }
            Segment::Italic(c) => {
                out.push_str("<span tts:fontStyle=\"italic\">");
                write_segments(c, out);
                out.push_str("</span>");
            }
            Segment::Underline(c) => {
                out.push_str("<span tts:textDecoration=\"underline\">");
                write_segments(c, out);
                out.push_str("</span>");
            }
            Segment::Strike(c) => {
                out.push_str("<span tts:textDecoration=\"lineThrough\">");
                write_segments(c, out);
                out.push_str("</span>");
            }
            Segment::Color { rgb, children } => {
                out.push_str(&format!(
                    "<span tts:color=\"#{:02X}{:02X}{:02X}\">",
                    rgb.0, rgb.1, rgb.2
                ));
                write_segments(children, out);
                out.push_str("</span>");
            }
            Segment::Font {
                family,
                size,
                children,
            } => {
                out.push_str("<span");
                if let Some(f) = family {
                    out.push_str(&format!(" tts:fontFamily=\"{}\"", escape_attr(f)));
                }
                if let Some(s) = size {
                    out.push_str(&format!(" tts:fontSize=\"{}px\"", s));
                }
                out.push('>');
                write_segments(children, out);
                out.push_str("</span>");
            }
            Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => write_segments(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(&escape_text(s)),
        }
    }
}

// ---------------------------------------------------------------------------
// Cue <-> bytes helpers (used by the codec wiring).

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let mut s = String::new();
    s.push_str("<p");
    s.push_str(&format!(" begin=\"{}\"", format_ts(cue.start_us)));
    s.push_str(&format!(" end=\"{}\"", format_ts(cue.end_us)));
    if let Some(sr) = &cue.style_ref {
        s.push_str(&format!(" style=\"{}\"", escape_attr(sr)));
    }
    s.push('>');
    write_segments(&cue.segments, &mut s);
    s.push_str("</p>");
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let nodes = parse_xml(&text)?;
    let p = find_element(&nodes, "p").ok_or_else(|| Error::invalid("TTML cue: missing <p>"))?;
    let start_us = attr(p, "begin")
        .and_then(|v| parse_ttml_time(&v))
        .unwrap_or(0);
    let end_us = attr(p, "end")
        .and_then(|v| parse_ttml_time(&v))
        .or_else(|| {
            attr(p, "dur")
                .and_then(|v| parse_ttml_time(&v))
                .map(|d| start_us + d)
        })
        .unwrap_or(start_us);
    let style_ref = attr(p, "style");
    let segments = collect_segments(&p.children);
    Ok(SubtitleCue {
        start_us,
        end_us,
        style_ref,
        positioning: None,
        segments,
    })
}

// ---------------------------------------------------------------------------
// Time helpers.

/// Parse a TTML time expression into microseconds. Supports `HH:MM:SS`,
/// `HH:MM:SS.mmm`, `HH:MM:SS.mmmmmm`, `HH:MM:SS:FF`, `<n>s`, `<n>ms`,
/// `<n>m`, `<n>h`.
fn parse_ttml_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Offset-time with unit.
    if let Some(n) = s.strip_suffix("ms") {
        let v: f64 = n.trim().parse().ok()?;
        return Some((v * 1_000.0) as i64);
    }
    if let Some(n) = s.strip_suffix('s') {
        let v: f64 = n.trim().parse().ok()?;
        return Some((v * 1_000_000.0) as i64);
    }
    if let Some(n) = s.strip_suffix('m') {
        let v: f64 = n.trim().parse().ok()?;
        return Some((v * 60_000_000.0) as i64);
    }
    if let Some(n) = s.strip_suffix('h') {
        let v: f64 = n.trim().parse().ok()?;
        return Some((v * 3_600_000_000.0) as i64);
    }
    if let Some(n) = s.strip_suffix('f') {
        // Frames without known fps — reject.
        let _ = n;
        return None;
    }
    // Clock time: hh:mm:ss[.fraction] or hh:mm:ss:frames.
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let h: i64 = parts[0].parse().ok()?;
    let m: i64 = parts[1].parse().ok()?;
    let mut sec_us: i64 = 0;
    if parts.len() >= 3 {
        let sp = parts[2];
        if let Some((int_p, frac_p)) = sp.split_once('.') {
            let sec: i64 = int_p.parse().ok()?;
            // Fractional seconds — pad / truncate to 6 digits.
            let frac: String = frac_p.chars().take(6).collect();
            let pad_len = 6 - frac.len();
            let mut pad = String::new();
            for _ in 0..pad_len {
                pad.push('0');
            }
            let frac_us: i64 = (frac + &pad).parse().unwrap_or(0);
            sec_us = sec * 1_000_000 + frac_us;
        } else {
            let sec: i64 = sp.parse().ok()?;
            sec_us = sec * 1_000_000;
        }
    }
    // parts[3] (frames) — we don't know fps; ignore.
    Some(h * 3_600_000_000 + m * 60_000_000 + sec_us)
}

fn format_ts(us: i64) -> String {
    let us = us.max(0);
    let total_ms = us / 1_000;
    let ms = (total_ms % 1_000) as u32;
    let total_s = total_ms / 1_000;
    let s = (total_s % 60) as u32;
    let m = ((total_s / 60) % 60) as u32;
    let h = (total_s / 3_600) as u32;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

// ---------------------------------------------------------------------------
// Color helpers.

fn parse_ttml_color_rgb(s: &str) -> Option<(u8, u8, u8)> {
    parse_ttml_color_rgba(s).map(|(r, g, b, _)| (r, g, b))
}

fn parse_ttml_color_rgba(s: &str) -> Option<(u8, u8, u8, u8)> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        return match hex.len() {
            3 => Some((
                u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?,
                u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?,
                u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?,
                255,
            )),
            6 => Some((
                u8::from_str_radix(&hex[0..2], 16).ok()?,
                u8::from_str_radix(&hex[2..4], 16).ok()?,
                u8::from_str_radix(&hex[4..6], 16).ok()?,
                255,
            )),
            8 => Some((
                u8::from_str_radix(&hex[0..2], 16).ok()?,
                u8::from_str_radix(&hex[2..4], 16).ok()?,
                u8::from_str_radix(&hex[4..6], 16).ok()?,
                u8::from_str_radix(&hex[6..8], 16).ok()?,
            )),
            _ => None,
        };
    }
    if let Some(rest) = s.strip_prefix("rgb(").and_then(|r| r.strip_suffix(')')) {
        let p: Vec<&str> = rest.split(',').map(|v| v.trim()).collect();
        if p.len() == 3 {
            return Some((
                p[0].parse().ok()?,
                p[1].parse().ok()?,
                p[2].parse().ok()?,
                255,
            ));
        }
    }
    named(s)
}

fn named(s: &str) -> Option<(u8, u8, u8, u8)> {
    match s.to_ascii_lowercase().as_str() {
        "black" => Some((0, 0, 0, 255)),
        "white" => Some((255, 255, 255, 255)),
        "red" => Some((255, 0, 0, 255)),
        "green" => Some((0, 128, 0, 255)),
        "lime" => Some((0, 255, 0, 255)),
        "blue" => Some((0, 0, 255, 255)),
        "yellow" => Some((255, 255, 0, 255)),
        "cyan" | "aqua" => Some((0, 255, 255, 255)),
        "magenta" | "fuchsia" => Some((255, 0, 255, 255)),
        "silver" => Some((192, 192, 192, 255)),
        "gray" | "grey" => Some((128, 128, 128, 255)),
        "transparent" => Some((0, 0, 0, 0)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tiny XML parser.

#[derive(Clone, Debug)]
pub(crate) struct Element {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

#[derive(Clone, Debug)]
pub(crate) enum Node {
    Element(Element),
    Text(String),
}

pub(crate) fn parse_xml(src: &str) -> Result<Vec<Node>> {
    let mut p = XmlParser {
        src: src.as_bytes(),
        pos: 0,
    };
    p.skip_prolog();
    let mut out: Vec<Node> = Vec::new();
    while p.pos < p.src.len() {
        match p.parse_node() {
            Some(Ok(node)) => out.push(node),
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    Ok(out)
}

struct XmlParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> XmlParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len()
            && matches!(self.src[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn skip_prolog(&mut self) {
        self.skip_ws();
        while self.pos < self.src.len() {
            if self.src[self.pos..].starts_with(b"<?") {
                // Processing instruction.
                let end = find_seq(self.src, self.pos, b"?>")
                    .map(|e| e + 2)
                    .unwrap_or(self.src.len());
                self.pos = end;
            } else if self.src[self.pos..].starts_with(b"<!--") {
                let end = find_seq(self.src, self.pos, b"-->")
                    .map(|e| e + 3)
                    .unwrap_or(self.src.len());
                self.pos = end;
            } else if self.src[self.pos..].starts_with(b"<!DOCTYPE")
                || self.src[self.pos..].starts_with(b"<!")
            {
                // Ignore DOCTYPE up to matching >.
                let end = find_seq(self.src, self.pos, b">")
                    .map(|e| e + 1)
                    .unwrap_or(self.src.len());
                self.pos = end;
            } else {
                break;
            }
            self.skip_ws();
        }
    }

    /// Parse one top-level node.
    fn parse_node(&mut self) -> Option<Result<Node>> {
        // Collect leading text up to `<`, then an element.
        let start = self.pos;
        while self.pos < self.src.len() && self.src[self.pos] != b'<' {
            self.pos += 1;
        }
        if self.pos > start {
            let raw = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
            let decoded = decode_entities(raw);
            if !decoded.trim().is_empty() {
                return Some(Ok(Node::Text(decoded)));
            }
            // Pure whitespace — keep a single space if inline; else skip.
            // We keep whitespace because it's significant inside <p>.
            return Some(Ok(Node::Text(decoded)));
        }
        if self.pos >= self.src.len() {
            return None;
        }
        // self.src[self.pos] == b'<'
        if self.src[self.pos..].starts_with(b"<!--") {
            let end = find_seq(self.src, self.pos, b"-->")
                .map(|e| e + 3)
                .unwrap_or(self.src.len());
            self.pos = end;
            return self.parse_node();
        }
        if self.src[self.pos..].starts_with(b"<![CDATA[") {
            let data_start = self.pos + b"<![CDATA[".len();
            let end = find_seq(self.src, data_start, b"]]>").unwrap_or(self.src.len());
            let raw = std::str::from_utf8(&self.src[data_start..end]).unwrap_or("");
            self.pos = end + 3;
            return Some(Ok(Node::Text(raw.to_string())));
        }
        if self.src[self.pos..].starts_with(b"</") {
            // Unexpected close — caller handles.
            return None;
        }
        // Opening tag.
        match self.parse_element() {
            Ok(e) => Some(Ok(Node::Element(e))),
            Err(err) => Some(Err(err)),
        }
    }

    fn parse_element(&mut self) -> Result<Element> {
        debug_assert_eq!(self.src[self.pos], b'<');
        self.pos += 1;
        // Read name.
        let name_start = self.pos;
        while self.pos < self.src.len()
            && !matches!(
                self.src[self.pos],
                b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/'
            )
        {
            self.pos += 1;
        }
        let name = std::str::from_utf8(&self.src[name_start..self.pos])
            .map_err(|_| Error::invalid("XML: bad UTF-8 in tag name"))?
            .to_string();
        if name.is_empty() {
            return Err(Error::invalid("XML: empty tag name"));
        }
        // Attributes.
        let mut attrs: Vec<(String, String)> = Vec::new();
        self.skip_ws();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            if b == b'>' {
                self.pos += 1;
                // Parse children until matching close.
                let children = self.parse_children(&name)?;
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if b == b'/' {
                // Self-closing.
                self.pos += 1;
                self.skip_ws();
                if self.pos < self.src.len() && self.src[self.pos] == b'>' {
                    self.pos += 1;
                    return Ok(Element {
                        name,
                        attrs,
                        children: Vec::new(),
                    });
                }
                return Err(Error::invalid("XML: malformed self-closing tag"));
            }
            // Attribute: name = value.
            let attr_name_start = self.pos;
            while self.pos < self.src.len()
                && !matches!(
                    self.src[self.pos],
                    b' ' | b'\t' | b'\n' | b'\r' | b'=' | b'>' | b'/'
                )
            {
                self.pos += 1;
            }
            let attr_name = std::str::from_utf8(&self.src[attr_name_start..self.pos])
                .map_err(|_| Error::invalid("XML: bad UTF-8 in attr name"))?
                .to_string();
            self.skip_ws();
            if self.pos >= self.src.len() || self.src[self.pos] != b'=' {
                // Valueless attr.
                if !attr_name.is_empty() {
                    attrs.push((attr_name, String::new()));
                }
                self.skip_ws();
                continue;
            }
            self.pos += 1; // skip '='
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Err(Error::invalid("XML: attribute missing value"));
            }
            let quote = self.src[self.pos];
            let (val_start, val_end) = if quote == b'"' || quote == b'\'' {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != quote {
                    self.pos += 1;
                }
                let end = self.pos;
                if self.pos < self.src.len() {
                    self.pos += 1;
                }
                (start, end)
            } else {
                // Unquoted.
                let start = self.pos;
                while self.pos < self.src.len()
                    && !matches!(
                        self.src[self.pos],
                        b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/'
                    )
                {
                    self.pos += 1;
                }
                (start, self.pos)
            };
            let raw = std::str::from_utf8(&self.src[val_start..val_end])
                .map_err(|_| Error::invalid("XML: bad UTF-8 in attr value"))?;
            attrs.push((attr_name, decode_entities(raw)));
            self.skip_ws();
        }
        Err(Error::invalid("XML: truncated element"))
    }

    fn parse_children(&mut self, name: &str) -> Result<Vec<Node>> {
        let mut children: Vec<Node> = Vec::new();
        while self.pos < self.src.len() {
            // Check for close tag.
            if self.src[self.pos..].starts_with(b"</") {
                let tag_end = find_seq(self.src, self.pos, b">")
                    .ok_or_else(|| Error::invalid("XML: truncated close tag"))?;
                let close_name = std::str::from_utf8(&self.src[self.pos + 2..tag_end])
                    .map_err(|_| Error::invalid("XML: bad UTF-8 in close tag"))?
                    .trim();
                self.pos = tag_end + 1;
                if close_name.eq_ignore_ascii_case(name) {
                    return Ok(children);
                }
                // Mismatched close — tolerate by stopping here.
                return Ok(children);
            }
            match self.parse_node() {
                Some(Ok(node)) => children.push(node),
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
        Ok(children)
    }
}

fn find_seq(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| from + p)
}

fn find_element<'a>(nodes: &'a [Node], name: &str) -> Option<&'a Element> {
    for n in nodes {
        if let Node::Element(e) = n {
            if tag_local(&e.name).eq_ignore_ascii_case(name) {
                return Some(e);
            }
        }
    }
    None
}

fn tag_local(name: &str) -> String {
    match name.rsplit_once(':') {
        Some((_, local)) => local.to_ascii_lowercase(),
        None => name.to_ascii_lowercase(),
    }
}

fn attr(el: &Element, name: &str) -> Option<String> {
    el.attrs
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

// ---------------------------------------------------------------------------
// Entity / escaping helpers.

fn decode_entities(s: &str) -> String {
    let mut out2 = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '&' {
            let mut ent = String::new();
            let mut terminated = false;
            while let Some(&nc) = chars.peek() {
                if nc == ';' {
                    chars.next();
                    terminated = true;
                    break;
                }
                if nc.is_whitespace() || nc == '&' || nc == '<' {
                    break;
                }
                ent.push(nc);
                chars.next();
                if ent.len() > 16 {
                    break;
                }
            }
            if terminated {
                if let Some(dec) = lookup_entity(&ent) {
                    out2.push(dec);
                    continue;
                }
                // Not recognised — emit as-is.
                out2.push('&');
                out2.push_str(&ent);
                out2.push(';');
                continue;
            }
            out2.push('&');
            out2.push_str(&ent);
        } else {
            out2.push(c);
        }
    }
    out2
}

fn lookup_entity(name: &str) -> Option<char> {
    if let Some(rest) = name.strip_prefix('#') {
        let code = if let Some(hex) = rest.strip_prefix('x').or_else(|| rest.strip_prefix('X')) {
            u32::from_str_radix(hex, 16).ok()?
        } else {
            rest.parse::<u32>().ok()?
        };
        return char::from_u32(code);
    }
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some('\u{00A0}'),
        _ => None,
    }
}

fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn decode_utf8_lossy_stripping_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let src = r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xml:lang="en">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:03.000">Hello</p>
      <p begin="00:00:04.500" end="00:00:06.000">Line one<br/>Line two</p>
    </div>
  </body>
</tt>"#;
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_000_000);
        assert_eq!(t.cues[1].start_us, 4_500_000);
    }

    #[test]
    fn parse_offset_time() {
        let src = r#"<?xml version="1.0"?><tt><body><div>
            <p begin="1.5s" end="3s">hi</p>
            <p begin="4000ms" dur="1s">ho</p>
        </div></body></tt>"#;
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 1_500_000);
        assert_eq!(t.cues[1].start_us, 4_000_000);
        assert_eq!(t.cues[1].end_us, 5_000_000);
    }

    #[test]
    fn styling_roundtrip() {
        // Raw string terminator `"#` cannot appear in content; use `##` delim.
        let src = r##"<?xml version="1.0"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <head>
    <styling>
      <style xml:id="s1" tts:color="#FF0000" tts:fontWeight="bold"/>
    </styling>
  </head>
  <body><div>
    <p begin="0s" end="1s" style="s1"><span tts:color="#00FF00">green</span></p>
  </div></body>
</tt>"##;
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.styles.len(), 1);
        assert_eq!(t.styles[0].name, "s1");
        assert!(t.styles[0].bold);
        let out = write(&t);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("<p"));
        assert!(s.contains("begin=\"00:00:00.000\""));
    }

    #[test]
    fn probe_positive() {
        let src = br#"<?xml version="1.0"?><tt xmlns="http://www.w3.org/ns/ttml"/>"#;
        assert!(probe(src) > 60);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"WEBVTT\n"), 0);
    }
}
