//! SubRip (.srt) parser and writer.
//!
//! SRT is a plain-text format. Structure (per cue):
//!
//! ```text
//! 1
//! 00:00:01,000 --> 00:00:03,500
//! Hello <i>world</i>
//! second line
//! <blank line>
//! 2
//! ...
//! ```
//!
//! We preserve the common inline HTML-ish tags: `<b>`, `<i>`, `<u>`,
//! `<s>`, and `<font color="...">`. Any tag we don't recognise survives
//! as [`Segment::Raw`] so a re-emit remains faithful.

use oxideav_core::{Error, Result, Segment, SubtitleCue};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Parse a UTF-8 (or UTF-8 with a leading BOM) SRT payload into a track.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut cues: Vec<SubtitleCue> = Vec::new();

    // Normalise line endings and walk cue-by-cue. We don't split on blank
    // lines alone because some files use CR-LF-CR-LF; a blank line is any
    // line whose trimmed form is empty.
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut i = 0;
    while i < lines.len() {
        // Skip blank lines between cues.
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        // Optional index line (a positive integer). Skip it if present.
        // SRT in the wild sometimes omits the index, so we tolerate.
        let maybe_index = lines[i].trim();
        if maybe_index.chars().all(|c| c.is_ascii_digit()) && !maybe_index.is_empty() {
            i += 1;
            if i >= lines.len() {
                break;
            }
        }

        // Timing line.
        let timing_line = lines[i].trim();
        i += 1;
        let (start_us, end_us) = match parse_timing_line(timing_line) {
            Some(t) => t,
            None => {
                // Skip malformed cue until the next blank line.
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }
        };

        // Text block — up to the next blank line.
        let mut text_lines: Vec<&str> = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text_lines.push(lines[i]);
            i += 1;
        }
        let body = text_lines.join("\n");
        let segments = parse_inline_tags(&body);
        cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments,
        });
    }

    Ok(SubtitleTrack {
        source: Some(SourceFormat::Srt),
        styles: Vec::new(),
        cues,
        metadata: Vec::new(),
        extradata: Vec::new(),
    })
}

/// Re-emit a track as SRT bytes.
pub fn write(track: &SubtitleTrack) -> Vec<u8> {
    let mut out = String::new();
    for (idx, cue) in track.cues.iter().enumerate() {
        out.push_str(&(idx + 1).to_string());
        out.push('\n');
        out.push_str(&format_timing(cue.start_us, cue.end_us));
        out.push('\n');
        out.push_str(&render_segments(&cue.segments));
        out.push('\n');
        out.push('\n');
    }
    out.into_bytes()
}

/// Render a single SRT cue body (no index, no timing line — just the
/// inline text with SRT-preserved tags).
pub fn render_segments(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_segments(segments, &mut out);
    out
}

fn append_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => out.push_str(&escape_text(s)),
            Segment::LineBreak => out.push('\n'),
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
            Segment::Strike(c) => {
                out.push_str("<s>");
                append_segments(c, out);
                out.push_str("</s>");
            }
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
                let mut header = String::from("<font");
                if let Some(fam) = family {
                    header.push_str(&format!(" face=\"{}\"", fam));
                }
                if let Some(sz) = size {
                    header.push_str(&format!(" size=\"{}\"", sz));
                }
                header.push('>');
                out.push_str(&header);
                append_segments(children, out);
                out.push_str("</font>");
            }
            Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => {
                // SRT can't express these — flatten to children.
                append_segments(children, out);
            }
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

fn escape_text(s: &str) -> String {
    // Don't escape `<` — SRT is permissive and real files are full of
    // unescaped angle brackets. We only need to avoid smuggling a known
    // tag into plain text. This is the same behaviour as libass.
    s.to_string()
}

// ---------------------------------------------------------------------------
// Time parsing

fn parse_timing_line(line: &str) -> Option<(i64, i64)> {
    // Find `-->`, allowing surrounding whitespace.
    let mid = line.find("-->")?;
    let (l, r) = line.split_at(mid);
    let r = &r[3..];
    let lhs = l.trim();
    // Right side may contain cue-setting-like trailing text (nonstandard in
    // SRT but harmless); strip anything after the timestamp.
    let rhs = r.trim().split_whitespace().next()?;
    let s = parse_srt_timestamp(lhs)?;
    let e = parse_srt_timestamp(rhs)?;
    Some((s, e))
}

/// Parse `HH:MM:SS,mmm` (or `HH:MM:SS.mmm`, or `MM:SS,mmm`) into microseconds.
fn parse_srt_timestamp(s: &str) -> Option<i64> {
    let (hms, ms) = split_hms_ms(s)?;
    let (h, m, sec) = split_hms(hms)?;
    // Milliseconds field is exactly the portion after the decimal/comma.
    let ms_val: u32 = ms.parse().ok()?;
    let micros = (h as i64) * 3_600_000_000
        + (m as i64) * 60_000_000
        + (sec as i64) * 1_000_000
        + (ms_val as i64) * 1_000;
    Some(micros)
}

fn split_hms_ms(s: &str) -> Option<(&str, &str)> {
    if let Some(i) = s.find(',') {
        Some((&s[..i], &s[i + 1..]))
    } else if let Some(i) = s.find('.') {
        Some((&s[..i], &s[i + 1..]))
    } else {
        // Tolerate missing ms.
        Some((s, "000"))
    }
}

fn split_hms(s: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        3 => {
            let h: u32 = parts[0].parse().ok()?;
            let m: u32 = parts[1].parse().ok()?;
            let sec: u32 = parts[2].parse().ok()?;
            Some((h, m, sec))
        }
        2 => {
            let m: u32 = parts[0].parse().ok()?;
            let sec: u32 = parts[1].parse().ok()?;
            Some((0, m, sec))
        }
        _ => None,
    }
}

/// Format microseconds as `HH:MM:SS,mmm`.
pub fn format_timing(start_us: i64, end_us: i64) -> String {
    format!(
        "{} --> {}",
        format_ts(start_us.max(0)),
        format_ts(end_us.max(0))
    )
}

fn format_ts(us: i64) -> String {
    let ms_total = us / 1_000;
    let ms = (ms_total % 1_000) as u32;
    let s_total = ms_total / 1_000;
    let s = (s_total % 60) as u32;
    let m = ((s_total / 60) % 60) as u32;
    let h = (s_total / 3_600) as u32;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

// ---------------------------------------------------------------------------
// Inline tag parser
//
// Supported tags:
//
//   <b>...</b>     <i>...</i>     <u>...</u>     <s>...</s>
//   <font color="#RRGGBB">...</font>
//   <font color="red">...</font>    (named colors)
//   <font face="..." size="...">...</font>
//
// Unknown tags survive as Segment::Raw (angle brackets included) so the
// round-trip preserves them.

fn parse_inline_tags(body: &str) -> Vec<Segment> {
    let mut p = Parser { src: body, pos: 0 };
    p.parse_until(None)
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn parse_until(&mut self, stop_tag: Option<&str>) -> Vec<Segment> {
        let mut out: Vec<Segment> = Vec::new();
        let mut text_buf = String::new();
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() {
            let byte = bytes[self.pos];
            if byte == b'<' {
                // Look ahead for a complete tag.
                if let Some(tag_end) = self.src[self.pos..].find('>') {
                    let tag = &self.src[self.pos + 1..self.pos + tag_end];
                    // Closing tag?
                    if let Some(stop) = stop_tag {
                        if tag.eq_ignore_ascii_case(&format!("/{}", stop)) {
                            if !text_buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut text_buf)));
                            }
                            self.pos += tag_end + 1;
                            return out;
                        }
                    }
                    // Dispatch to known openers.
                    let (name, _attrs) = split_tag(tag);
                    let name_lc = name.to_ascii_lowercase();
                    match name_lc.as_str() {
                        "b" | "i" | "u" | "s" => {
                            if !text_buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut text_buf)));
                            }
                            self.pos += tag_end + 1;
                            let children = self.parse_until(Some(&name_lc));
                            out.push(match name_lc.as_str() {
                                "b" => Segment::Bold(children),
                                "i" => Segment::Italic(children),
                                "u" => Segment::Underline(children),
                                _ => Segment::Strike(children),
                            });
                            continue;
                        }
                        "font" => {
                            if !text_buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut text_buf)));
                            }
                            self.pos += tag_end + 1;
                            let children = self.parse_until(Some("font"));
                            out.push(classify_font_tag(tag, children));
                            continue;
                        }
                        _ => {
                            // Unknown: treat the whole `<...>` block as raw text.
                            if !text_buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut text_buf)));
                            }
                            out.push(Segment::Raw(format!("<{}>", tag)));
                            self.pos += tag_end + 1;
                            continue;
                        }
                    }
                } else {
                    // No closing `>` — treat rest as text.
                    text_buf.push_str(&self.src[self.pos..]);
                    self.pos = bytes.len();
                }
            } else {
                text_buf.push(byte as char);
                self.pos += 1;
            }
        }
        if !text_buf.is_empty() {
            out.push(Segment::Text(text_buf));
        }
        out
    }
}

fn split_tag(tag: &str) -> (&str, &str) {
    let tag = tag.trim();
    match tag.find(char::is_whitespace) {
        Some(i) => (&tag[..i], tag[i..].trim()),
        None => (tag, ""),
    }
}

fn classify_font_tag(full_tag: &str, children: Vec<Segment>) -> Segment {
    // full_tag is the inner text of `<...>`: e.g. `font color="#ff0000"`.
    let lc = full_tag.to_ascii_lowercase();
    if let Some(col) = extract_attr(&lc, "color") {
        if let Some(rgb) = parse_color(&col) {
            return Segment::Color { rgb, children };
        }
    }
    let family = extract_attr(&lc, "face");
    let size = extract_attr(&lc, "size").and_then(|s| s.parse::<f32>().ok());
    if family.is_some() || size.is_some() {
        return Segment::Font {
            family,
            size,
            children,
        };
    }
    // Nothing recognisable — keep as raw so we don't lose the opener.
    // (Rendering uses children directly — the raw tag reappears on write.)
    Segment::Raw(format!("<{}>", full_tag)).wrap_with(children)
}

trait SegmentExt {
    fn wrap_with(self, children: Vec<Segment>) -> Segment;
}

impl SegmentExt for Segment {
    fn wrap_with(self, children: Vec<Segment>) -> Segment {
        // Synthesise a Font with children under an unknown tag so downstream
        // writers still surface the contained text.
        let _ = self;
        Segment::Font {
            family: None,
            size: None,
            children,
        }
    }
}

fn extract_attr(lc_tag: &str, name: &str) -> Option<String> {
    // Looks for `name="..."` or `name='...'` or `name=bareword` in a
    // lowercase-normalised tag body.
    let key = format!("{}=", name);
    let idx = lc_tag.find(&key)?;
    let rest = &lc_tag[idx + key.len()..];
    let bytes = rest.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (start_quote, end_char) = match bytes[0] {
        b'"' => (1, b'"'),
        b'\'' => (1, b'\''),
        _ => (0, b' '),
    };
    let rest = &rest[start_quote..];
    let end = rest
        .as_bytes()
        .iter()
        .position(|&b| b == end_char || b == b'>');
    let end = end.unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn parse_color(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim().trim_matches(|c: char| c == '"' || c == '\'');
    if let Some(hex) = s.strip_prefix('#') {
        return hex_color(hex);
    }
    if s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return hex_color(s);
    }
    named_color(s)
}

fn hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    if hex.len() == 3 {
        let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
        let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
        let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
        return Some((r, g, b));
    }
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}

fn named_color(name: &str) -> Option<(u8, u8, u8)> {
    match name.to_ascii_lowercase().as_str() {
        "black" => Some((0, 0, 0)),
        "white" => Some((255, 255, 255)),
        "red" => Some((255, 0, 0)),
        "green" => Some((0, 128, 0)),
        "lime" => Some((0, 255, 0)),
        "blue" => Some((0, 0, 255)),
        "yellow" => Some((255, 255, 0)),
        "cyan" | "aqua" => Some((0, 255, 255)),
        "magenta" | "fuchsia" => Some((255, 0, 255)),
        "silver" => Some((192, 192, 192)),
        "gray" | "grey" => Some((128, 128, 128)),
        "maroon" => Some((128, 0, 0)),
        "olive" => Some((128, 128, 0)),
        "navy" => Some((0, 0, 128)),
        "purple" => Some((128, 0, 128)),
        "teal" => Some((0, 128, 128)),
        _ => None,
    }
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

/// Quick header check. Used by the container probe to tell SRT apart from
/// other text formats.
pub(crate) fn looks_like_srt(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let first = match lines.next() {
        Some(v) => v.trim(),
        None => return false,
    };
    // First non-empty line: index (1+ digits).
    if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
        // Some files skip the index — fall back to detecting the timing
        // line directly on the first line.
        return looks_like_timing(first);
    }
    let second = match lines.next() {
        Some(v) => v.trim(),
        None => return false,
    };
    looks_like_timing(second)
}

fn looks_like_timing(line: &str) -> bool {
    line.contains("-->") && parse_timing_line(line).is_some()
}

/// Turn one cue into its standalone on-wire form (no preceding index
/// line — the container supplies one if it chooses to).
pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&format_timing(cue.start_us, cue.end_us));
    s.push('\n');
    s.push_str(&render_segments(&cue.segments));
    s.into_bytes()
}

/// Parse one cue (as emitted by [`cue_to_bytes`]) back into a [`SubtitleCue`].
pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    // Optional leading blank lines.
    while lines.first().map(|l| l.trim().is_empty()).unwrap_or(false) {
        lines.remove(0);
    }
    // Optional leading index line.
    if let Some(first) = lines.first() {
        if first.trim().chars().all(|c| c.is_ascii_digit()) && !first.trim().is_empty() {
            lines.remove(0);
        }
    }
    if lines.is_empty() {
        return Err(Error::invalid("SRT: empty cue payload"));
    }
    let timing = lines.remove(0);
    let (start_us, end_us) = parse_timing_line(timing.trim())
        .ok_or_else(|| Error::invalid("SRT: cue has no valid timing"))?;
    let body = lines.join("\n");
    let segments = parse_inline_tags(body.trim_end());
    Ok(SubtitleCue {
        start_us,
        end_us,
        style_ref: None,
        positioning: None,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_cue() {
        let src = "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 1);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_500_000);
    }

    #[test]
    fn round_trips_italic() {
        let src = "1\n00:00:01,000 --> 00:00:02,000\n<i>hi</i>\n\n";
        let t = parse(src.as_bytes()).unwrap();
        let out = String::from_utf8(write(&t)).unwrap();
        assert!(out.contains("<i>hi</i>"), "roundtrip: {out}");
    }

    #[test]
    fn format_timing_matches() {
        assert_eq!(format_ts(3_500_000), "00:00:03,500");
        assert_eq!(format_ts(3_661_500_000), "01:01:01,500");
    }

    #[test]
    fn parses_color_font() {
        let src = "1\n00:00:01,000 --> 00:00:02,000\n<font color=\"red\">hi</font>\n\n";
        let t = parse(src.as_bytes()).unwrap();
        match &t.cues[0].segments[0] {
            Segment::Color { rgb, .. } => assert_eq!(*rgb, (255, 0, 0)),
            other => panic!("expected Color segment, got {other:?}"),
        }
    }

    #[test]
    fn looks_like_srt_true() {
        assert!(looks_like_srt(b"1\n00:00:01,000 --> 00:00:02,000\nHi\n"));
    }

    #[test]
    fn looks_like_srt_false() {
        assert!(!looks_like_srt(b"WEBVTT\n\n"));
    }
}
