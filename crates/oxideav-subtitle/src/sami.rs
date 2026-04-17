//! SAMI (Synchronized Accessible Media Interchange, Microsoft) parser and
//! writer.
//!
//! SAMI is HTML-like tag soup. High-level shape:
//!
//! ```text
//! <SAMI>
//! <HEAD>
//!   <STYLE TYPE="text/css">
//!   <!--
//!     .ENUSCC { Name: English; lang: en-US; }
//!     .FRCC   { Name: French;  lang: fr-FR; color: yellow; }
//!   -->
//!   </STYLE>
//! </HEAD>
//! <BODY>
//! <SYNC Start=1000>
//! <P Class="ENUSCC">Hello <B>world</B></P>
//! <SYNC Start=3000>
//! <P Class="ENUSCC">&nbsp;</P>
//! </BODY>
//! </SAMI>
//! ```
//!
//! Parsing strategy: hand-rolled tag-soup walker. Each `<SYNC Start=ms>`
//! starts a new cue and closes the previous one (end time = this Start).
//! A body containing only `&nbsp;` is a clear — the previous cue stays
//! with its computed end but nothing new is emitted until the next
//! populated SYNC.

use oxideav_core::{Result, Segment, SubtitleCue, SubtitleStyle};

use crate::ir::{SourceFormat, SubtitleTrack};

/// Codec id string.
pub const CODEC_ID: &str = "sami";

/// Parse a SAMI payload into a [`SubtitleTrack`].
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt), // SAMI not in enum — annotate metadata
        ..SubtitleTrack::default()
    };
    track.metadata.push(("source_format".into(), "sami".into()));

    // Grab the first <STYLE ...> block so we can parse CSS classes into
    // style rows.
    let lc = text.to_ascii_lowercase();
    if let Some(sty_start) = lc.find("<style") {
        if let Some(body_start_rel) = lc[sty_start..].find('>') {
            let body_start = sty_start + body_start_rel + 1;
            if let Some(end_rel) = lc[body_start..].find("</style>") {
                let raw = &text[body_start..body_start + end_rel];
                track.styles.extend(parse_sami_css(raw));
            }
        }
    }

    // Collect sync points.
    //
    // We stream `<SYNC Start=N>` markers and slice the text between them.
    let mut syncs: Vec<(i64, usize)> = Vec::new();
    let mut i = 0usize;
    while i < text.len() {
        let slice = &text[i..];
        let lc_slice = slice.to_ascii_lowercase();
        if let Some(idx) = lc_slice.find("<sync") {
            let abs = i + idx;
            // Find closing '>'.
            if let Some(close_rel) = text[abs..].find('>') {
                let tag = &text[abs..abs + close_rel];
                if let Some(ms) = parse_sync_start(tag) {
                    let body_start = abs + close_rel + 1;
                    syncs.push((ms * 1_000, body_start));
                }
                i = abs + close_rel + 1;
                continue;
            }
            break;
        } else {
            break;
        }
    }

    // Default cue duration when a SAMI file ends without a trailing clear
    // SYNC: 4 seconds.
    const DEFAULT_DUR_US: i64 = 4_000_000;

    let syncs_len = syncs.len();
    for (idx, &(start_us, body_start)) in syncs.iter().enumerate() {
        let body_end = if idx + 1 < syncs_len {
            // Go up to just before the next `<SYNC`.
            let next_start = syncs[idx + 1].1;
            // Walk back to the `<SYNC` itself.
            let slice = &text[..next_start];
            let sync_idx = slice
                .to_ascii_lowercase()
                .rfind("<sync")
                .unwrap_or(slice.len());
            sync_idx
        } else {
            text.len()
        };
        let body = &text[body_start..body_end];
        // Feed the whole body (including the outer <P>) to the inline
        // parser so it can capture the class attribute.
        let is_clear = {
            let inner_only = extract_p_content(body).unwrap_or_else(|| body.to_string());
            is_clear_body(inner_only.trim())
        };
        if is_clear {
            // If we have a previous cue that hasn't been closed yet,
            // close it at this time.
            if let Some(last) = track.cues.last_mut() {
                if last.end_us == 0 || last.end_us > start_us {
                    last.end_us = start_us;
                }
            }
            continue;
        }
        let (class_name, segments) = parse_inline_sami(body);
        // Close the previous cue, if any, at this new start.
        let mut end_us = start_us + DEFAULT_DUR_US;
        if let Some(last) = track.cues.last_mut() {
            if last.end_us == 0 || last.end_us > start_us {
                last.end_us = start_us;
            }
        }
        // If there's a next sync, extend end to it initially.
        if idx + 1 < syncs_len {
            end_us = syncs[idx + 1].0;
        }
        track.cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: class_name,
            positioning: None,
            segments,
        });
    }

    // Stash the entire source as extradata for round-trip.
    track.extradata = text.into_bytes();
    Ok(track)
}

/// Re-emit a track as SAMI bytes.
pub fn write(track: &SubtitleTrack) -> Vec<u8> {
    let mut out = String::new();
    out.push_str("<SAMI>\n<HEAD>\n<STYLE TYPE=\"text/css\">\n<!--\n");
    out.push_str("P { font-family: Arial; font-size: 16px; color: white; }\n");
    for s in &track.styles {
        out.push_str(&format!(".{} {{", s.name));
        if let Some((r, g, b, _)) = s.primary_color {
            out.push_str(&format!(" color: #{:02X}{:02X}{:02X};", r, g, b));
        }
        if let Some(fam) = &s.font_family {
            out.push_str(&format!(" font-family: {};", fam));
        }
        if let Some(sz) = s.font_size {
            out.push_str(&format!(" font-size: {}px;", sz));
        }
        if s.bold {
            out.push_str(" font-weight: bold;");
        }
        if s.italic {
            out.push_str(" font-style: italic;");
        }
        out.push_str(" }\n");
    }
    out.push_str("-->\n</STYLE>\n</HEAD>\n<BODY>\n");

    for cue in &track.cues {
        out.push_str(&format!("<SYNC Start={}>\n", cue.start_us / 1_000));
        let class = cue.style_ref.as_deref().unwrap_or("");
        if class.is_empty() {
            out.push_str("<P>");
        } else {
            out.push_str(&format!("<P Class=\"{}\">", class));
        }
        write_segments(&cue.segments, &mut out);
        out.push_str("</P>\n");
        // Emit a clear SYNC at cue.end_us so the display clears.
        out.push_str(&format!("<SYNC Start={}>\n", cue.end_us / 1_000));
        out.push_str("<P>&nbsp;</P>\n");
    }
    out.push_str("</BODY>\n</SAMI>\n");
    out.into_bytes()
}

/// Probe score (0..=100).
pub fn probe(buf: &[u8]) -> u8 {
    looks_like_sami(buf)
}

pub fn looks_like_sami(buf: &[u8]) -> u8 {
    let head = &buf[..buf.len().min(4096)];
    let text = String::from_utf8_lossy(head).to_ascii_lowercase();
    let mut score = 0u8;
    if text.contains("<sami") {
        score += 60;
    }
    if text.contains("<sync") {
        score += 30;
    }
    if text.contains("<style") && text.contains("text/css") {
        score = score.saturating_add(5);
    }
    score.min(100)
}

pub fn make_decoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Decoder>> {
    crate::codec::make_decoder(params)
}

pub fn make_encoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Encoder>> {
    crate::codec::make_encoder(params)
}

// ---------------------------------------------------------------------------
// Cue <-> bytes helpers.

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&format!("<SYNC Start={}>\n", cue.start_us / 1_000));
    let class = cue.style_ref.as_deref().unwrap_or("");
    if class.is_empty() {
        s.push_str("<P>");
    } else {
        s.push_str(&format!("<P Class=\"{}\">", class));
    }
    write_segments(&cue.segments, &mut s);
    s.push_str("</P>");
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> oxideav_core::Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    // Expect "<SYNC Start=...>\n<P ...>...</P>".
    let start_ms = find_attr_u64(&text, "start")
        .ok_or_else(|| oxideav_core::Error::invalid("SAMI cue: missing Start attribute"))?;
    let inner = extract_p_content(&text).unwrap_or_default();
    let (class_name, segments) = parse_inline_sami(&inner);
    let start_us = start_ms as i64 * 1_000;
    Ok(SubtitleCue {
        start_us,
        end_us: start_us + 4_000_000,
        style_ref: class_name,
        positioning: None,
        segments,
    })
}

// ---------------------------------------------------------------------------
// SYNC / tag extraction.

fn parse_sync_start(tag: &str) -> Option<i64> {
    find_attr_u64(tag, "start").map(|v| v as i64)
}

/// Find a named attribute's unsigned-integer value inside a raw tag or a
/// small text blob (case-insensitive). Accepts `name=123`, `name="123"`,
/// `name='123'`.
fn find_attr_u64(s: &str, name: &str) -> Option<u64> {
    let lc = s.to_ascii_lowercase();
    let lc_name = name.to_ascii_lowercase();
    let mut search_from = 0usize;
    while let Some(idx) = lc[search_from..].find(&lc_name) {
        let abs = search_from + idx;
        // Ensure preceded by whitespace or `<` so we don't match inside
        // another attr name.
        let ok_before = abs == 0
            || matches!(
                lc.as_bytes()[abs - 1],
                b' ' | b'\t' | b'\n' | b'\r' | b'<' | b'"' | b'\'' | b';'
            );
        let after = abs + lc_name.len();
        if ok_before && after < lc.len() {
            // Skip spaces then expect '='.
            let mut i = after;
            while i < lc.len() && matches!(lc.as_bytes()[i], b' ' | b'\t') {
                i += 1;
            }
            if i < lc.len() && lc.as_bytes()[i] == b'=' {
                i += 1;
                while i < lc.len() && matches!(lc.as_bytes()[i], b' ' | b'\t') {
                    i += 1;
                }
                // Optional quote.
                let quote = if i < lc.len() { lc.as_bytes()[i] } else { 0 };
                if quote == b'"' || quote == b'\'' {
                    i += 1;
                }
                // Read digits.
                let digit_start = i;
                while i < lc.len() && lc.as_bytes()[i].is_ascii_digit() {
                    i += 1;
                }
                if i > digit_start {
                    return s[digit_start..i].parse().ok();
                }
            }
        }
        search_from = abs + lc_name.len();
    }
    None
}

fn extract_p_content(body: &str) -> Option<String> {
    let lc = body.to_ascii_lowercase();
    let start = lc.find("<p")?;
    let after = body[start..].find('>')? + start + 1;
    // Scan for closing </P> but also stop at next <SYNC.
    let rest_lc = lc[after..].to_string();
    let close = rest_lc.find("</p>");
    let next_sync = rest_lc.find("<sync");
    let end_rel = match (close, next_sync) {
        (Some(c), Some(s)) => c.min(s),
        (Some(c), None) => c,
        (None, Some(s)) => s,
        (None, None) => rest_lc.len(),
    };
    Some(body[after..after + end_rel].to_string())
}

fn is_clear_body(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Decode entities & strip tags first — if what's left is empty /
    // single NBSP, it's a clear.
    let (_class, segs) = parse_inline_sami(trimmed);
    let plain: String = segs
        .iter()
        .filter_map(|s| match s {
            Segment::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    plain.trim().is_empty() || plain.trim() == "\u{00A0}"
}

// ---------------------------------------------------------------------------
// Inline tag-soup parser.

/// Parse the text inside a `<P>` into segments. Returns the class name
/// (from an outer `<P Class="...">`) if found.
fn parse_inline_sami(src: &str) -> (Option<String>, Vec<Segment>) {
    // If the source still contains an outer `<P ...>...</P>` wrapper,
    // peel it off while remembering the class.
    let mut class_name: Option<String> = None;
    let mut body = src.to_string();
    let lc = body.to_ascii_lowercase();
    if let Some(p_open) = lc.find("<p") {
        if let Some(tag_end) = body[p_open..].find('>') {
            let open_tag = &body[p_open..p_open + tag_end];
            class_name = extract_class_attr(open_tag);
            // Slice to inside.
            let start = p_open + tag_end + 1;
            let lc_rest = lc[start..].to_string();
            let close = lc_rest.find("</p>");
            let end = close.map(|c| start + c).unwrap_or(body.len());
            body = body[start..end].to_string();
        }
    }

    let mut p = TagSoup {
        src: body.as_bytes(),
        pos: 0,
    };
    (class_name, p.parse_until(None))
}

fn extract_class_attr(tag: &str) -> Option<String> {
    let lc = tag.to_ascii_lowercase();
    let idx = lc.find("class")?;
    let rest = &tag[idx..];
    // Find '=' after 'class'.
    let eq = rest.find('=')?;
    let v = rest[eq + 1..].trim_start();
    let v_bytes = v.as_bytes();
    if v_bytes.is_empty() {
        return None;
    }
    let (quote, start_off) = match v_bytes[0] {
        b'"' => (b'"', 1),
        b'\'' => (b'\'', 1),
        _ => (b' ', 0),
    };
    let v = &v[start_off..];
    let end = v
        .as_bytes()
        .iter()
        .position(|&b| b == quote || b == b'>' || b == b' ' || b == b'\t' || b == b'\n');
    let end = end.unwrap_or(v.len());
    let val = v[..end].trim();
    if val.is_empty() {
        None
    } else {
        Some(val.to_string())
    }
}

struct TagSoup<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> TagSoup<'a> {
    fn parse_until(&mut self, stop: Option<&str>) -> Vec<Segment> {
        let mut out: Vec<Segment> = Vec::new();
        let mut buf = String::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            if b == b'<' {
                // Find closing '>'.
                let tag_end = match memchr_byte(self.src, self.pos, b'>') {
                    Some(e) => e,
                    None => {
                        // No closer — treat rest as text.
                        let s = std::str::from_utf8(&self.src[self.pos..]).unwrap_or("");
                        buf.push_str(&decode_entities(s));
                        self.pos = self.src.len();
                        break;
                    }
                };
                let inner = std::str::from_utf8(&self.src[self.pos + 1..tag_end]).unwrap_or("");
                // Comment / PI — skip.
                if inner.starts_with('!') || inner.starts_with('?') {
                    self.pos = tag_end + 1;
                    continue;
                }
                let (name, _rest) = split_tag(inner);
                let name_lc = name.trim_start_matches('/').to_ascii_lowercase();
                let is_close = name.starts_with('/');
                // End of file / body / sync — stop.
                if is_close {
                    if let Some(stop_tag) = stop {
                        if name_lc == stop_tag {
                            if !buf.is_empty() {
                                out.push(Segment::Text(std::mem::take(&mut buf)));
                            }
                            self.pos = tag_end + 1;
                            return out;
                        }
                    }
                    // Generic close — just skip.
                    self.pos = tag_end + 1;
                    continue;
                }
                if name_lc == "br" {
                    if !buf.is_empty() {
                        out.push(Segment::Text(std::mem::take(&mut buf)));
                    }
                    out.push(Segment::LineBreak);
                    self.pos = tag_end + 1;
                    continue;
                }
                if name_lc == "sync" || name_lc == "body" || name_lc == "sami" || name_lc == "p" {
                    // A nested <P> / <SYNC> ends the current cue body.
                    if !buf.is_empty() {
                        out.push(Segment::Text(std::mem::take(&mut buf)));
                    }
                    // Rewind so the caller sees this tag.
                    return out;
                }
                match name_lc.as_str() {
                    "b" => {
                        if !buf.is_empty() {
                            out.push(Segment::Text(std::mem::take(&mut buf)));
                        }
                        self.pos = tag_end + 1;
                        let kids = self.parse_until(Some("b"));
                        out.push(Segment::Bold(kids));
                    }
                    "i" => {
                        if !buf.is_empty() {
                            out.push(Segment::Text(std::mem::take(&mut buf)));
                        }
                        self.pos = tag_end + 1;
                        let kids = self.parse_until(Some("i"));
                        out.push(Segment::Italic(kids));
                    }
                    "u" => {
                        if !buf.is_empty() {
                            out.push(Segment::Text(std::mem::take(&mut buf)));
                        }
                        self.pos = tag_end + 1;
                        let kids = self.parse_until(Some("u"));
                        out.push(Segment::Underline(kids));
                    }
                    "s" | "strike" => {
                        if !buf.is_empty() {
                            out.push(Segment::Text(std::mem::take(&mut buf)));
                        }
                        self.pos = tag_end + 1;
                        let kids = self.parse_until(Some(&name_lc));
                        out.push(Segment::Strike(kids));
                    }
                    "font" => {
                        if !buf.is_empty() {
                            out.push(Segment::Text(std::mem::take(&mut buf)));
                        }
                        let col = extract_attr_val(inner, "color");
                        let face = extract_attr_val(inner, "face");
                        let size =
                            extract_attr_val(inner, "size").and_then(|v| v.parse::<f32>().ok());
                        self.pos = tag_end + 1;
                        let kids = self.parse_until(Some("font"));
                        if let Some(c) = col.as_deref().and_then(parse_color_rgb) {
                            out.push(Segment::Color {
                                rgb: c,
                                children: kids,
                            });
                        } else if face.is_some() || size.is_some() {
                            out.push(Segment::Font {
                                family: face,
                                size,
                                children: kids,
                            });
                        } else {
                            out.extend(kids);
                        }
                    }
                    _ => {
                        // Unknown — skip tag, flatten children.
                        self.pos = tag_end + 1;
                    }
                }
            } else {
                // Accumulate characters by taking the slice up to next `<`.
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != b'<' {
                    self.pos += 1;
                }
                let raw = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
                buf.push_str(&decode_entities(raw));
            }
        }
        if !buf.is_empty() {
            out.push(Segment::Text(buf));
        }
        out
    }
}

fn memchr_byte(haystack: &[u8], from: usize, needle: u8) -> Option<usize> {
    haystack[from..]
        .iter()
        .position(|&b| b == needle)
        .map(|p| from + p)
}

fn split_tag(tag: &str) -> (&str, &str) {
    let tag = tag.trim();
    match tag.find(char::is_whitespace) {
        Some(i) => (&tag[..i], tag[i..].trim()),
        None => (tag, ""),
    }
}

fn extract_attr_val(tag: &str, name: &str) -> Option<String> {
    let lc = tag.to_ascii_lowercase();
    let idx = lc.find(&format!("{}=", name.to_ascii_lowercase()))?;
    let rest = &tag[idx + name.len() + 1..];
    let bytes = rest.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (quote, start) = match bytes[0] {
        b'"' => (b'"', 1),
        b'\'' => (b'\'', 1),
        _ => (b' ', 0),
    };
    let rest = &rest[start..];
    let end = rest
        .as_bytes()
        .iter()
        .position(|&b| b == quote || b == b'>' || b == b' ' || b == b'\t');
    let end = end.unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

// ---------------------------------------------------------------------------
// Simple CSS class parser (for the SAMI <STYLE> block).

fn parse_sami_css(src: &str) -> Vec<SubtitleStyle> {
    // Strip HTML comment markers.
    let cleaned = src.replace("<!--", "").replace("-->", "");
    let mut out: Vec<SubtitleStyle> = Vec::new();
    let mut i = 0;
    while i < cleaned.len() {
        let rest = &cleaned[i..];
        let dot = match rest.find('.') {
            Some(p) => p,
            None => break,
        };
        let after = i + dot + 1;
        // Read class name.
        let name_start = after;
        let name_bytes = cleaned.as_bytes();
        let mut j = name_start;
        while j < name_bytes.len()
            && (name_bytes[j].is_ascii_alphanumeric()
                || name_bytes[j] == b'_'
                || name_bytes[j] == b'-')
        {
            j += 1;
        }
        if j == name_start {
            i = after;
            continue;
        }
        let class_name = &cleaned[name_start..j];
        // Find `{`.
        let open = cleaned[j..].find('{').map(|p| j + p);
        let close = open.and_then(|o| cleaned[o..].find('}').map(|p| o + p));
        if let (Some(o), Some(c)) = (open, close) {
            let body = &cleaned[o + 1..c];
            let mut st = SubtitleStyle::new(class_name);
            for decl in body.split(';') {
                let decl = decl.trim();
                if let Some((k, v)) = decl.split_once(':') {
                    let k = k.trim().to_ascii_lowercase();
                    let v = v.trim();
                    match k.as_str() {
                        "color" => {
                            if let Some(rgb) = parse_color_rgb(v) {
                                st.primary_color = Some((rgb.0, rgb.1, rgb.2, 255));
                            }
                        }
                        "background-color" | "background" => {
                            if let Some(rgb) = parse_color_rgb(v) {
                                st.back_color = Some((rgb.0, rgb.1, rgb.2, 255));
                            }
                        }
                        "font-family" => st.font_family = Some(v.trim_matches('"').to_string()),
                        "font-size" => {
                            let num: String = v
                                .chars()
                                .take_while(|c| c.is_ascii_digit() || *c == '.')
                                .collect();
                            st.font_size = num.parse::<f32>().ok();
                        }
                        "font-weight" => {
                            if v.eq_ignore_ascii_case("bold") {
                                st.bold = true;
                            }
                        }
                        "font-style" => {
                            if v.eq_ignore_ascii_case("italic") {
                                st.italic = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            out.push(st);
            i = c + 1;
        } else {
            break;
        }
    }
    out
}

fn parse_color_rgb(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim().trim_matches(|c: char| c == '"' || c == '\'');
    if let Some(hex) = s.strip_prefix('#') {
        return hex_to_rgb(hex);
    }
    match s.to_ascii_lowercase().as_str() {
        "black" => Some((0, 0, 0)),
        "white" => Some((255, 255, 255)),
        "red" => Some((255, 0, 0)),
        "green" => Some((0, 128, 0)),
        "lime" => Some((0, 255, 0)),
        "blue" => Some((0, 0, 255)),
        "yellow" => Some((255, 255, 0)),
        "cyan" | "aqua" => Some((0, 255, 255)),
        "magenta" | "fuchsia" => Some((255, 0, 255)),
        _ => None,
    }
}

fn hex_to_rgb(hex: &str) -> Option<(u8, u8, u8)> {
    match hex.len() {
        3 => Some((
            u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?,
            u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?,
            u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?,
        )),
        6 => Some((
            u8::from_str_radix(&hex[0..2], 16).ok()?,
            u8::from_str_radix(&hex[2..4], 16).ok()?,
            u8::from_str_radix(&hex[4..6], 16).ok()?,
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Segment writer.

fn write_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => out.push_str(&escape_text(s)),
            Segment::LineBreak => out.push_str("<BR>"),
            Segment::Bold(c) => {
                out.push_str("<B>");
                write_segments(c, out);
                out.push_str("</B>");
            }
            Segment::Italic(c) => {
                out.push_str("<I>");
                write_segments(c, out);
                out.push_str("</I>");
            }
            Segment::Underline(c) => {
                out.push_str("<U>");
                write_segments(c, out);
                out.push_str("</U>");
            }
            Segment::Strike(c) => {
                out.push_str("<S>");
                write_segments(c, out);
                out.push_str("</S>");
            }
            Segment::Color { rgb, children } => {
                out.push_str(&format!(
                    "<FONT color=\"#{:02X}{:02X}{:02X}\">",
                    rgb.0, rgb.1, rgb.2
                ));
                write_segments(children, out);
                out.push_str("</FONT>");
            }
            Segment::Font {
                family,
                size,
                children,
            } => {
                out.push_str("<FONT");
                if let Some(f) = family {
                    out.push_str(&format!(" face=\"{}\"", f));
                }
                if let Some(s) = size {
                    out.push_str(&format!(" size=\"{}\"", s));
                }
                out.push('>');
                write_segments(children, out);
                out.push_str("</FONT>");
            }
            Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => write_segments(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(s) => out.push_str(s),
        }
    }
}

fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\u{00A0}' => out.push_str("&nbsp;"),
            _ => out.push(c),
        }
    }
    out
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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
                if let Some(ch) = lookup_entity(&ent) {
                    out.push(ch);
                    continue;
                }
                out.push('&');
                out.push_str(&ent);
                out.push(';');
                continue;
            }
            out.push('&');
            out.push_str(&ent);
        } else {
            out.push(c);
        }
    }
    out
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
    match name.to_ascii_lowercase().as_str() {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some('\u{00A0}'),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "<SAMI>\n<HEAD>\n<STYLE TYPE=\"text/css\">\n<!--\n.ENUSCC { Name: English; lang: en-US; color: yellow; }\n-->\n</STYLE>\n</HEAD>\n<BODY>\n<SYNC Start=1000>\n<P Class=\"ENUSCC\">Hello <B>world</B></P>\n<SYNC Start=3000>\n<P Class=\"ENUSCC\">&nbsp;</P>\n<SYNC Start=5000>\n<P Class=\"ENUSCC\">Second line</P>\n</BODY>\n</SAMI>\n";

    #[test]
    fn parses_syncs() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2, "cues: {:?}", t.cues);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_000_000); // closed by clear SYNC
        assert_eq!(t.cues[1].start_us, 5_000_000);
    }

    #[test]
    fn parses_style_class() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert!(t.styles.iter().any(|s| s.name == "ENUSCC"));
        assert_eq!(t.cues[0].style_ref.as_deref(), Some("ENUSCC"));
    }

    #[test]
    fn probe_positive() {
        assert!(probe(SAMPLE.as_bytes()) > 60);
    }

    #[test]
    fn write_roundtrip_preserves_inline_bold() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        let out = write(&t);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("<B>world</B>"), "missing bold: {s}");
        assert!(s.contains("SYNC Start=1000"));
    }
}
