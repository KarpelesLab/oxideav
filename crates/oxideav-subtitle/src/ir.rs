//! Format-independent intermediate representation for a whole subtitle track.
//!
//! A [`SubtitleTrack`] is what every parser produces and every writer
//! consumes. The unified segment / style / cue types live in
//! `oxideav-core` so downstream crates can consume them without pulling
//! this crate in.

pub use oxideav_core::{CuePosition, Segment, SubtitleCue, SubtitleStyle, TextAlign};

/// Which on-disk flavour the track was parsed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFormat {
    Srt,
    WebVtt,
    AssOrSsa,
}

/// A fully parsed subtitle file: track-level metadata + styles + cues.
///
/// `metadata` keys follow a loose convention:
/// * `title`, `author` — script info (ASS).
/// * `play_res_x`, `play_res_y`, `wrap_style`, `scaled_border_and_shadow`,
///   `timer`, `script_type` — ASS script-info extras.
/// * `header` — raw WebVTT header trailing text after `WEBVTT`.
#[derive(Clone, Debug, Default)]
pub struct SubtitleTrack {
    pub source: Option<SourceFormat>,
    pub styles: Vec<SubtitleStyle>,
    pub cues: Vec<SubtitleCue>,
    pub metadata: Vec<(String, String)>,
    /// Raw extradata (e.g. the WebVTT header block, ASS `[Script Info]` +
    /// `[V4+ Styles]` + `[Events]` lead-in). Preserved so remuxers can
    /// replay byte-for-byte where that's valuable.
    pub extradata: Vec<u8>,
}

impl SubtitleTrack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_source(mut self, src: SourceFormat) -> Self {
        self.source = Some(src);
        self
    }

    /// Look up a style by name.
    pub fn style(&self, name: &str) -> Option<&SubtitleStyle> {
        self.styles.iter().find(|s| s.name == name)
    }
}

/// Collect all plain-text content from a segment tree, separating
/// [`Segment::LineBreak`] with `\n`. Useful for the most aggressive
/// downconversion (e.g. producing an SRT line without style markup).
pub fn plain_text(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_plain(segments, &mut out);
    out
}

fn append_plain(segments: &[Segment], out: &mut String) {
    for seg in segments {
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
