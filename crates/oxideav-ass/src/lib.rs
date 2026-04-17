//! ASS/SSA subtitle codec + container for oxideav.
//!
//! This crate hosts the parser, writer, codec, and container for the
//! Advanced SubStation Alpha (`.ass`) and SubStation Alpha (`.ssa`) text
//! subtitle formats. It is a sibling to `oxideav-subtitle`, which hosts
//! the "lightweight" text formats (SRT, WebVTT) and the shared subtitle
//! IR re-exports.
//!
//! | Format  | Codec id | Container name | Extensions      |
//! |---------|----------|----------------|-----------------|
//! | ASS/SSA | `ass`    | `ass`          | `.ass`, `.ssa`  |
//!
//! The public API mirrors `oxideav-webp` / `oxideav-gif`:
//!
//! * [`register_codecs`] — add the ASS codec.
//! * [`register_containers`] — add the ASS container + probe.
//! * [`register`] — do both.
//!
//! Format-to-format converters between ASS and the SRT/WebVTT formats
//! from `oxideav-subtitle` live in [`transform`].

pub mod codec;
pub mod container;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId, MediaType};

pub use transform::{ass_to_srt, ass_to_webvtt, srt_to_ass, webvtt_to_ass};

// ---------------------------------------------------------------------------
// Parser / writer (moved verbatim from oxideav-subtitle::ass).

use oxideav_core::{CuePosition, Error, Result, Segment, SubtitleCue, SubtitleStyle, TextAlign};
use oxideav_subtitle::ir::{SourceFormat, SubtitleTrack};

pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let mut track = SubtitleTrack {
        source: Some(SourceFormat::AssOrSsa),
        ..SubtitleTrack::default()
    };

    let mut current_section = String::new();
    let mut style_format: Vec<String> = Vec::new();
    let mut event_format: Vec<String> = Vec::new();
    let mut is_ssa = false;

    // extradata: collect everything up to (and including) the [Events] Format line
    let mut extradata = String::new();

    for line_raw in text.split('\n') {
        let line = line_raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Keep empty lines in extradata while we're still in header.
            if !is_events_body(&current_section, &event_format) {
                extradata.push_str(line);
                extradata.push('\n');
            }
            continue;
        }
        if trimmed.starts_with(';') || trimmed.starts_with('!') {
            // Comment (inside Script Info); preserve in extradata.
            if !is_events_body(&current_section, &event_format) {
                extradata.push_str(line);
                extradata.push('\n');
            }
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].to_ascii_lowercase();
            if current_section == "v4 styles" {
                is_ssa = true;
            }
            if !is_events_body(&current_section, &event_format) {
                extradata.push_str(line);
                extradata.push('\n');
            }
            continue;
        }

        match current_section.as_str() {
            "script info" => {
                if let Some((k, v)) = trimmed.split_once(':') {
                    track.metadata.push((
                        k.trim().to_ascii_lowercase().replace(' ', "_"),
                        v.trim().to_string(),
                    ));
                }
                extradata.push_str(line);
                extradata.push('\n');
            }
            "v4+ styles" | "v4 styles" => {
                extradata.push_str(line);
                extradata.push('\n');
                if let Some(rest) = strip_prefix_case(trimmed, "Format:") {
                    style_format = rest.split(',').map(|s| s.trim().to_string()).collect();
                } else if let Some(rest) = strip_prefix_case(trimmed, "Style:") {
                    if let Some(style) = parse_style_line(rest, &style_format, is_ssa) {
                        track.styles.push(style);
                    }
                }
            }
            "events" => {
                if let Some(rest) = strip_prefix_case(trimmed, "Format:") {
                    event_format = rest.split(',').map(|s| s.trim().to_string()).collect();
                    extradata.push_str(line);
                    extradata.push('\n');
                } else if let Some(rest) = strip_prefix_case(trimmed, "Dialogue:") {
                    if let Some(cue) = parse_event_line(rest, &event_format) {
                        track.cues.push(cue);
                    }
                } else if let Some(_rest) = strip_prefix_case(trimmed, "Comment:") {
                    // Ignore comment events.
                } else {
                    // Unknown event-section line — drop.
                }
            }
            "fonts" | "graphics" => {
                // Skip UU-encoded data blocks.
            }
            _ => {}
        }
    }

    track.extradata = extradata.into_bytes();
    Ok(track)
}

fn is_events_body(section: &str, event_format: &[String]) -> bool {
    section == "events" && !event_format.is_empty()
}

fn strip_prefix_case<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if line.len() < prefix.len() {
        return None;
    }
    if line[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(line[prefix.len()..].trim_start())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Style parsing

fn parse_style_line(line: &str, fmt: &[String], is_ssa: bool) -> Option<SubtitleStyle> {
    let fields: Vec<&str> = split_csv(line, fmt.len());
    if fields.len() < fmt.len() {
        return None;
    }
    let mut style = SubtitleStyle::default();
    for (k, v) in fmt.iter().zip(fields.iter()) {
        let key = k.to_ascii_lowercase().replace(' ', "");
        let val = v.trim();
        match key.as_str() {
            "name" => style.name = val.to_string(),
            "fontname" => style.font_family = Some(val.to_string()),
            "fontsize" => style.font_size = val.parse().ok(),
            "primarycolour" | "primarycolor" => {
                style.primary_color = parse_ass_color(val);
            }
            "outlinecolour" | "outlinecolor" => {
                style.outline_color = parse_ass_color(val);
            }
            "backcolour" | "backcolor" => {
                style.back_color = parse_ass_color(val);
            }
            "bold" => style.bold = parse_bool_flag(val),
            "italic" => style.italic = parse_bool_flag(val),
            "underline" => style.underline = parse_bool_flag(val),
            "strikeout" | "strikethrough" => style.strike = parse_bool_flag(val),
            "alignment" => {
                style.align = if is_ssa {
                    ssa_alignment_to_textalign(val.parse().unwrap_or(2))
                } else {
                    ass_alignment_to_textalign(val.parse().unwrap_or(2))
                };
            }
            "marginl" => style.margin_l = val.parse().ok(),
            "marginr" => style.margin_r = val.parse().ok(),
            "marginv" => style.margin_v = val.parse().ok(),
            "outline" => style.outline = val.parse().ok(),
            "shadow" => style.shadow = val.parse().ok(),
            _ => {}
        }
    }
    if style.name.is_empty() {
        style.name = "Default".into();
    }
    Some(style)
}

/// ASS `\an<N>`: 1=bl, 2=bc, 3=br, 4=ml, 5=mc, 6=mr, 7=tl, 8=tc, 9=tr.
fn ass_alignment_to_textalign(n: i32) -> TextAlign {
    match n {
        1 | 4 | 7 => TextAlign::Left,
        2 | 5 | 8 => TextAlign::Center,
        3 | 6 | 9 => TextAlign::Right,
        _ => TextAlign::Center,
    }
}

/// SSA alignment: low nibble = L/C/R (1/2/3), high bit = top/mid/bot.
fn ssa_alignment_to_textalign(n: i32) -> TextAlign {
    match n & 0x03 {
        1 => TextAlign::Left,
        3 => TextAlign::Right,
        _ => TextAlign::Center,
    }
}

fn parse_bool_flag(s: &str) -> bool {
    let v: i32 = s.parse().unwrap_or(0);
    v != 0
}

/// ASS colors: `&HAABBGGRR&` or `&HBBGGRR` (no alpha) or `&H...`.
/// Return RGBA. Alpha in ASS is 00 = fully opaque, FF = fully transparent.
fn parse_ass_color(s: &str) -> Option<(u8, u8, u8, u8)> {
    let s = s.trim().trim_matches('&');
    let s = s.trim_start_matches(|c| c == 'H' || c == 'h');
    let s = s.trim_start_matches("0x");
    // Trim trailing `&` or whitespace.
    let s = s.trim_end_matches('&').trim();
    if s.is_empty() {
        return None;
    }
    // Parse as hex, pad to 8 chars.
    let mut v: u32 = u32::from_str_radix(s, 16).ok()?;
    let has_alpha = s.len() > 6;
    if !has_alpha {
        // Pad alpha to 00 (opaque).
        v &= 0x00FF_FFFF;
    }
    let a = ((v >> 24) & 0xFF) as u8;
    let b = ((v >> 16) & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let r = (v & 0xFF) as u8;
    // Invert ASS "transparency" to canonical alpha where 255 is opaque.
    Some((r, g, b, 255_u8.saturating_sub(a)))
}

/// Split a comma-separated field list but **only into the first N-1 commas**;
/// the tail is left whole (dialogue Text may contain commas).
fn split_csv(line: &str, n: usize) -> Vec<&str> {
    if n == 0 {
        return vec![line];
    }
    let mut out: Vec<&str> = Vec::with_capacity(n);
    let mut cursor = line;
    for _ in 0..n - 1 {
        if let Some(i) = cursor.find(',') {
            out.push(&cursor[..i]);
            cursor = &cursor[i + 1..];
        } else {
            out.push(cursor);
            cursor = "";
        }
    }
    out.push(cursor);
    out
}

// ---------------------------------------------------------------------------
// Event parsing

fn parse_event_line(line: &str, fmt: &[String]) -> Option<SubtitleCue> {
    if fmt.is_empty() {
        return None;
    }
    let fields = split_csv(line, fmt.len());
    if fields.len() < fmt.len() {
        return None;
    }
    let mut start_us: i64 = 0;
    let mut end_us: i64 = 0;
    let mut style_ref: Option<String> = None;
    let mut text: &str = "";
    for (k, v) in fmt.iter().zip(fields.iter()) {
        let key = k.to_ascii_lowercase();
        let val = v.trim();
        match key.as_str() {
            "start" => start_us = parse_ass_timestamp(val).unwrap_or(0),
            "end" => end_us = parse_ass_timestamp(val).unwrap_or(0),
            "style" => {
                if !val.is_empty() {
                    style_ref = Some(val.to_string());
                }
            }
            "text" => text = v,
            _ => {}
        }
    }
    let (segments, positioning) = parse_ass_text(text);
    Some(SubtitleCue {
        start_us,
        end_us,
        style_ref,
        positioning,
        segments,
    })
}

/// ASS timestamp: `H:MM:SS.cc` (centiseconds) — sometimes with extra digits.
fn parse_ass_timestamp(s: &str) -> Option<i64> {
    let (hms, frac) = match s.find('.') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, "0"),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (
            parts[0].parse::<u32>().ok()?,
            parts[1].parse::<u32>().ok()?,
            parts[2].parse::<u32>().ok()?,
        ),
        2 => (
            0u32,
            parts[0].parse::<u32>().ok()?,
            parts[1].parse::<u32>().ok()?,
        ),
        _ => return None,
    };
    // `frac` is centiseconds (2 digits) but be robust to 1-3 digit forms.
    let cs_str = if frac.len() > 2 { &frac[..2] } else { frac };
    let cs: u32 = if cs_str.is_empty() {
        0
    } else {
        cs_str.parse().ok()?
    };
    // Pad to 2 digits if only 1 was given.
    let cs = if frac.len() == 1 { cs * 10 } else { cs };
    Some(
        (h as i64) * 3_600_000_000
            + (m as i64) * 60_000_000
            + (sec as i64) * 1_000_000
            + (cs as i64) * 10_000,
    )
}

fn format_ass_ts(us: i64) -> String {
    let us = us.max(0);
    let cs_total = us / 10_000;
    let cs = (cs_total % 100) as u32;
    let s_total = cs_total / 100;
    let s = (s_total % 60) as u32;
    let m = ((s_total / 60) % 60) as u32;
    let h = (s_total / 3_600) as u32;
    format!("{}:{:02}:{:02}.{:02}", h, m, s, cs)
}

// ---------------------------------------------------------------------------
// ASS override-tag parser

fn parse_ass_text(text: &str) -> (Vec<Segment>, Option<CuePosition>) {
    let mut out: Vec<Segment> = Vec::new();
    let mut state = AssState::default();
    let mut positioning: Option<CuePosition> = None;

    let mut cursor = 0;
    let bytes = text.as_bytes();
    let mut text_buf = String::new();

    while cursor < bytes.len() {
        if bytes[cursor] == b'{' {
            // Flush accumulated text.
            if !text_buf.is_empty() {
                out.push(state.wrap(Segment::Text(std::mem::take(&mut text_buf))));
            }
            let end = match text[cursor..].find('}') {
                Some(e) => cursor + e,
                None => {
                    text_buf.push('{');
                    cursor += 1;
                    continue;
                }
            };
            let overrides = &text[cursor + 1..end];
            handle_overrides(overrides, &mut state, &mut positioning, &mut out);
            cursor = end + 1;
            continue;
        }
        if bytes[cursor] == b'\\' && cursor + 1 < bytes.len() {
            let c = bytes[cursor + 1] as char;
            if c == 'N' {
                if !text_buf.is_empty() {
                    out.push(state.wrap(Segment::Text(std::mem::take(&mut text_buf))));
                }
                out.push(Segment::LineBreak);
                cursor += 2;
                continue;
            }
            if c == 'n' {
                // Soft line break — treat like a space in text when word-wrap is on.
                text_buf.push(' ');
                cursor += 2;
                continue;
            }
            if c == 'h' {
                // Hard space.
                text_buf.push('\u{00A0}');
                cursor += 2;
                continue;
            }
        }
        text_buf.push(bytes[cursor] as char);
        cursor += 1;
    }
    if !text_buf.is_empty() {
        out.push(state.wrap(Segment::Text(text_buf)));
    }

    (out, positioning)
}

#[derive(Clone, Debug, Default)]
struct AssState {
    bold: bool,
    italic: bool,
    underline: bool,
    strike: bool,
    color: Option<(u8, u8, u8)>,
    font_family: Option<String>,
    font_size: Option<f32>,
}

impl AssState {
    fn wrap(&self, seg: Segment) -> Segment {
        let mut s = seg;
        if self.bold {
            s = Segment::Bold(vec![s]);
        }
        if self.italic {
            s = Segment::Italic(vec![s]);
        }
        if self.underline {
            s = Segment::Underline(vec![s]);
        }
        if self.strike {
            s = Segment::Strike(vec![s]);
        }
        if let Some(rgb) = self.color {
            s = Segment::Color {
                rgb,
                children: vec![s],
            };
        }
        if self.font_family.is_some() || self.font_size.is_some() {
            s = Segment::Font {
                family: self.font_family.clone(),
                size: self.font_size,
                children: vec![s],
            };
        }
        s
    }
}

fn handle_overrides(
    block: &str,
    state: &mut AssState,
    positioning: &mut Option<CuePosition>,
    out: &mut Vec<Segment>,
) {
    // Walk the block splitting on backslashes (each `\tag...` is one override).
    // We can't just split on `\` because `\pos(x,y)` contains commas — but
    // splits on `\` not inside parens are fine.
    let mut i = 0;
    let bytes = block.as_bytes();
    // Skip leading whitespace.
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    // We preserve the original text as a fallback Raw so the round-trip
    // keeps the block intact even if we don't understand every override.
    let mut any_understood = false;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        i += 1;
        // Read override identifier. Names are alphabetic or start with a
        // small digit followed by letters (e.g. `1c`, `2c`, `3c`, `4a`).
        let start = i;
        if i < bytes.len() && bytes[i].is_ascii_digit() {
            // Digit-prefixed name — take the digit + any following letters.
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else {
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        }
        let name = &block[start..i];
        // Parameter — either parenthesised or ends at next `\` or end.
        let param_start = i;
        let param = if i < bytes.len() && bytes[i] == b'(' {
            let end = match block[i..].find(')') {
                Some(e) => i + e,
                None => block.len(),
            };
            let p = &block[i + 1..end];
            i = (end + 1).min(block.len());
            p.to_string()
        } else {
            // Until next `\`.
            while i < bytes.len() && bytes[i] != b'\\' {
                i += 1;
            }
            block[param_start..i].to_string()
        };
        let name_lc = name.to_ascii_lowercase();
        match name_lc.as_str() {
            "b" => {
                state.bold = parse_bool_flag(&param);
                any_understood = true;
            }
            "i" => {
                state.italic = parse_bool_flag(&param);
                any_understood = true;
            }
            "u" => {
                state.underline = parse_bool_flag(&param);
                any_understood = true;
            }
            "s" => {
                state.strike = parse_bool_flag(&param);
                any_understood = true;
            }
            "c" | "1c" => {
                if let Some((r, g, b, _)) = parse_ass_color(&param) {
                    state.color = Some((r, g, b));
                }
                any_understood = true;
            }
            "fn" => {
                state.font_family = Some(param.trim().to_string());
                any_understood = true;
            }
            "fs" => {
                state.font_size = param.trim().parse().ok();
                any_understood = true;
            }
            "pos" => {
                let parts: Vec<&str> = param.split(',').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    let cp = positioning.get_or_insert_with(CuePosition::default);
                    cp.x = parts[0].parse().ok();
                    cp.y = parts[1].parse().ok();
                    any_understood = true;
                }
            }
            "an" => {
                let n: i32 = param.trim().parse().unwrap_or(2);
                let cp = positioning.get_or_insert_with(CuePosition::default);
                cp.align = ass_alignment_to_textalign(n);
                any_understood = true;
            }
            "k" | "kf" | "ko" => {
                let cs: u32 = param.trim().parse().unwrap_or(0);
                // Emit an empty karaoke segment now; the next text chunk
                // will be appended to a later karaoke span in a richer
                // implementation. Here we push a marker Karaoke with empty
                // children — consumers that care about karaoke can detect
                // it.
                out.push(Segment::Karaoke {
                    cs,
                    children: Vec::new(),
                });
                any_understood = true;
            }
            _ => {
                // Unknown override — leave fallback in place below.
            }
        }
    }
    // Nothing understood → preserve the block verbatim. (But avoid
    // duplicating state that we successfully applied.)
    if !any_understood {
        out.push(Segment::Raw(format!("{{{}}}", block)));
    }
}

// ---------------------------------------------------------------------------
// Writer

pub fn write(track: &SubtitleTrack) -> Vec<u8> {
    // Re-use extradata if present (keeps the user's original script-info
    // and style rows intact). Otherwise synthesise a minimal header.
    let mut out = String::new();
    if !track.extradata.is_empty() {
        out.push_str(&String::from_utf8_lossy(&track.extradata));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    } else {
        out.push_str("[Script Info]\n");
        out.push_str("ScriptType: v4.00+\n");
        for (k, v) in &track.metadata {
            let cap = capitalise_key(k);
            out.push_str(&format!("{}: {}\n", cap, v));
        }
        out.push('\n');
        out.push_str("[V4+ Styles]\n");
        out.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, Alignment, MarginL, MarginR, MarginV, Outline, Shadow\n");
        let has_default = track.styles.iter().any(|s| s.name == "Default");
        if !has_default {
            out.push_str(&default_style_line());
        }
        for s in &track.styles {
            out.push_str(&style_row(s));
        }
        out.push('\n');
        out.push_str("[Events]\n");
        out.push_str(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        );
    }
    for cue in &track.cues {
        let txt = render_event_text(cue);
        let style = cue.style_ref.clone().unwrap_or_else(|| "Default".into());
        out.push_str(&format!(
            "Dialogue: 0,{},{},{},,0,0,0,,{}\n",
            format_ass_ts(cue.start_us),
            format_ass_ts(cue.end_us),
            style,
            txt
        ));
    }
    out.into_bytes()
}

fn capitalise_key(k: &str) -> String {
    // Convert `play_res_x` -> `PlayResX`. Rough heuristic.
    k.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn default_style_line() -> String {
    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,2,10,10,10,1,0\n"
        .into()
}

fn style_row(s: &SubtitleStyle) -> String {
    let col = s
        .primary_color
        .map(|(r, g, b, a)| format_ass_color(r, g, b, a))
        .unwrap_or_else(|| "&H00FFFFFF".into());
    let outline = s
        .outline_color
        .map(|(r, g, b, a)| format_ass_color(r, g, b, a))
        .unwrap_or_else(|| "&H00000000".into());
    let back = s
        .back_color
        .map(|(r, g, b, a)| format_ass_color(r, g, b, a))
        .unwrap_or_else(|| "&H00000000".into());
    let fn_ = s.font_family.clone().unwrap_or_else(|| "Arial".into());
    let fs = s.font_size.unwrap_or(20.0);
    let align = match s.align {
        TextAlign::Left | TextAlign::Start => 1,
        TextAlign::Center => 2,
        TextAlign::Right | TextAlign::End => 3,
    };
    format!(
        "Style: {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
        s.name,
        fn_,
        fs,
        col,
        outline,
        back,
        s.bold as u8,
        s.italic as u8,
        s.underline as u8,
        s.strike as u8,
        align,
        s.margin_l.unwrap_or(10),
        s.margin_r.unwrap_or(10),
        s.margin_v.unwrap_or(10),
        s.outline.unwrap_or(1.0),
        s.shadow.unwrap_or(0.0),
    )
}

fn format_ass_color(r: u8, g: u8, b: u8, a: u8) -> String {
    // Invert our alpha (255=opaque) back to ASS transparency (00=opaque).
    let inv_a = 255_u8.saturating_sub(a);
    format!("&H{:02X}{:02X}{:02X}{:02X}", inv_a, b, g, r)
}

fn render_event_text(cue: &SubtitleCue) -> String {
    let mut out = String::new();
    if let Some(p) = &cue.positioning {
        if let (Some(x), Some(y)) = (p.x, p.y) {
            out.push_str(&format!("{{\\pos({},{})}}", x as i32, y as i32));
        }
    }
    append_ass_segments(&cue.segments, &mut out);
    out
}

fn append_ass_segments(segments: &[Segment], out: &mut String) {
    for seg in segments {
        match seg {
            Segment::Text(s) => {
                // Escape `{`, `}`, and newlines.
                for c in s.chars() {
                    match c {
                        '\n' => out.push_str("\\N"),
                        '{' | '}' => out.push(c),
                        _ => out.push(c),
                    }
                }
            }
            Segment::LineBreak => out.push_str("\\N"),
            Segment::Bold(c) => {
                out.push_str("{\\b1}");
                append_ass_segments(c, out);
                out.push_str("{\\b0}");
            }
            Segment::Italic(c) => {
                out.push_str("{\\i1}");
                append_ass_segments(c, out);
                out.push_str("{\\i0}");
            }
            Segment::Underline(c) => {
                out.push_str("{\\u1}");
                append_ass_segments(c, out);
                out.push_str("{\\u0}");
            }
            Segment::Strike(c) => {
                out.push_str("{\\s1}");
                append_ass_segments(c, out);
                out.push_str("{\\s0}");
            }
            Segment::Color { rgb, children } => {
                out.push_str(&format!(
                    "{{\\c&H{:02X}{:02X}{:02X}&}}",
                    rgb.2, rgb.1, rgb.0
                ));
                append_ass_segments(children, out);
                out.push_str("{\\c}");
            }
            Segment::Font {
                family,
                size,
                children,
            } => {
                if let Some(fam) = family {
                    out.push_str(&format!("{{\\fn{}}}", fam));
                }
                if let Some(sz) = size {
                    out.push_str(&format!("{{\\fs{}}}", sz));
                }
                append_ass_segments(children, out);
            }
            Segment::Voice { children, .. } | Segment::Class { children, .. } => {
                append_ass_segments(children, out);
            }
            Segment::Karaoke { cs, children } => {
                out.push_str(&format!("{{\\k{}}}", cs));
                append_ass_segments(children, out);
            }
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

pub(crate) fn looks_like_ass(buf: &[u8]) -> bool {
    let text = decode_utf8_lossy_stripping_bom(buf);
    // Look at the first 2048 chars for `[Script Info]` (case-insensitive).
    let head: String = text.chars().take(2048).collect();
    let head_lc = head.to_ascii_lowercase();
    head_lc.contains("[script info]")
}

/// Serialise one cue for single-packet container emission.
pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let style = cue.style_ref.clone().unwrap_or_else(|| "Default".into());
    let txt = render_event_text(cue);
    let line = format!(
        "Dialogue: 0,{},{},{},,0,0,0,,{}",
        format_ass_ts(cue.start_us),
        format_ass_ts(cue.end_us),
        style,
        txt
    );
    line.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = decode_utf8_lossy_stripping_bom(bytes);
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| Error::invalid("ASS: empty cue"))?;
    let rest = strip_prefix_case(line.trim(), "Dialogue:")
        .ok_or_else(|| Error::invalid("ASS: cue missing Dialogue prefix"))?;
    // Use the default format ordering — Layer,Start,End,Style,Name,ML,MR,MV,Effect,Text
    let fmt = [
        "Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();
    parse_event_line(rest, &fmt).ok_or_else(|| Error::invalid("ASS: bad Dialogue line"))
}

// ---------------------------------------------------------------------------
// Registration entry points

/// Register the ASS codec (decoder + encoder).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities {
        decode: false,
        encode: false,
        media_type: MediaType::Subtitle,
        intra_only: true,
        lossy: false,
        lossless: true,
        hardware_accelerated: false,
        implementation: "ass_sw".into(),
        max_width: None,
        max_height: None,
        max_bitrate: None,
        max_sample_rate: None,
        max_channels: None,
        priority: 100,
        accepted_pixel_formats: Vec::new(),
    };
    reg.register_both(
        CodecId::new(codec::ASS_CODEC_ID),
        caps,
        codec::make_decoder,
        codec::make_encoder,
    );
}

/// Register the ASS container (demuxer + muxer + probe).
pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}

/// Convenience combined registration.
pub fn register(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r"[Script Info]
Title: Test
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\b1}Hello{\b0} world
";

    #[test]
    fn parse_sample() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 1);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_000_000);
        assert_eq!(t.cues[0].style_ref.as_deref(), Some("Default"));
        assert!(t.styles.iter().any(|s| s.name == "Default"));
    }

    #[test]
    fn parse_override() {
        let t = parse(SAMPLE.as_bytes()).unwrap();
        // First segment should be Bold wrapping "Hello".
        let s0 = &t.cues[0].segments[0];
        match s0 {
            Segment::Bold(inner) => match &inner[0] {
                Segment::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected text in bold, got {other:?}"),
            },
            other => panic!("expected bold, got {other:?}"),
        }
    }

    #[test]
    fn ass_color_parse() {
        // &H00FF0000 → alpha 00 (opaque), B=FF, G=00, R=00 → blue opaque
        let c = parse_ass_color("&H00FF0000").unwrap();
        assert_eq!(c, (0, 0, 255, 255));
    }

    #[test]
    fn ass_timestamp() {
        let t = parse_ass_timestamp("0:00:01.50").unwrap();
        assert_eq!(t, 1_500_000);
    }

    #[test]
    fn looks_like() {
        assert!(looks_like_ass(SAMPLE.as_bytes()));
        assert!(!looks_like_ass(b"WEBVTT\n"));
    }
}
