//! Subtitle compositor: turns a [`SubtitleCue`] into an RGBA bitmap
//! suitable for compositing as a video plane.
//!
//! Pipeline:
//!
//! 1. Walk the segment tree, flattening it into a stream of styled runs
//!    (text, face, italic, color).
//! 2. Word-wrap those runs into lines that each fit within `width` pixels
//!    — breaking at spaces first, hard-breaking only when a single word
//!    is larger than the line.
//! 3. Stack the lines from the bottom of the canvas upwards (honouring
//!    `bottom_margin_px`) and horizontally centre-align each line — unless
//!    the cue's own `positioning.align` tells us to left/right-align.
//! 4. Blit each glyph with a black outline drawn first (4 one-pixel
//!    offsets in `outline_color`) and the run's foreground on top.
//!    Italic renders via a per-row horizontal shear of `cell_w / 4`
//!    pixels across the glyph height.
//!
//! The output is always a fresh RGBA `Vec<u8>` of size `width*height*4`,
//! starting zeroed (fully transparent). A paired [`Compositor::render_into`]
//! reuses a caller-provided buffer.
//!
//! Intentional non-features (left for later):
//!
//! * No TrueType shaping — we use the embedded 8×16 bitmap font only.
//! * No CJK, no BiDi, no combining marks beyond Latin-1 precomposed.
//! * No animation / karaoke timing (the Karaoke segment is rendered as
//!   plain text).
//! * No absolute positioning (ASS `\pos`, WebVTT `x%,y%`). Everything
//!   sits at the bottom-centre (or bottom-left / bottom-right).

use oxideav_core::{Segment, SubtitleCue, TextAlign};

use crate::font::BitmapFont;

/// Bottom-centered, bitmap-font subtitle renderer.
pub struct Compositor {
    pub width: u32,
    pub height: u32,
    /// Default foreground RGBA. Runs without an explicit `Color` use this.
    pub default_color: [u8; 4],
    /// Outline RGBA drawn underneath every glyph.
    pub outline_color: [u8; 4],
    /// Distance between baselines of consecutive lines, in pixels.
    pub line_height_px: u32,
    /// Spacing between the bottom edge of the canvas and the baseline of
    /// the last line.
    pub bottom_margin_px: u32,
    /// Outline thickness (0..=2). Larger values are clamped in `render`.
    pub outline_px: u32,
}

impl Compositor {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            default_color: [255, 255, 255, 255],
            outline_color: [0, 0, 0, 255],
            line_height_px: 20,
            bottom_margin_px: 24,
            outline_px: 1,
        }
    }

    /// Render a cue into a freshly-allocated RGBA buffer of
    /// `width * height * 4` bytes (all pixels initially transparent).
    pub fn render(&self, cue: &SubtitleCue) -> Vec<u8> {
        let mut buf = vec![0u8; (self.width as usize) * (self.height as usize) * 4];
        self.render_into(cue, &mut buf);
        buf
    }

    /// Render a cue into a pre-allocated RGBA buffer. The buffer is
    /// cleared to transparent first, so callers can reuse it across
    /// frames without manual zeroing.
    pub fn render_into(&self, cue: &SubtitleCue, dst: &mut [u8]) {
        // Zero the canvas.
        let required = (self.width as usize) * (self.height as usize) * 4;
        if dst.len() < required {
            return;
        }
        for b in dst[..required].iter_mut() {
            *b = 0;
        }

        // 1. Flatten segments into runs.
        let runs = flatten_segments(&cue.segments, RunStyle::default_with(self.default_color));

        // 2. Wrap runs into lines by greedy word-break.
        let regular = BitmapFont::default_regular();
        let cell_w = regular.cell_w;
        let max_cols = (self.width / cell_w).max(1);
        let lines = wrap_runs(&runs, max_cols as usize);
        if lines.is_empty() {
            return;
        }

        // 3. Position lines from bottom up.
        let cell_h = regular.cell_h;
        let bearing_y = regular.bearing_y;
        let line_h = self.line_height_px.max(cell_h);
        let outline = self.outline_px.min(2);
        let last_baseline = self
            .height
            .saturating_sub(self.bottom_margin_px)
            .saturating_sub((cell_h - bearing_y).min(cell_h));
        let last_baseline = last_baseline as i32;

        // Honour cue-level alignment. Default: Center.
        let align = cue
            .positioning
            .as_ref()
            .map(|p| p.align)
            .unwrap_or(TextAlign::Center);

        // 4. Blit each line.
        let n_lines = lines.len();
        for (i, line) in lines.iter().enumerate() {
            let baseline = last_baseline - ((n_lines - 1 - i) as i32) * line_h as i32;
            let line_width_px = measure_line(line, cell_w) as i32;
            let x = match align {
                TextAlign::Left | TextAlign::Start => 8,
                TextAlign::Right | TextAlign::End => self.width as i32 - line_width_px - 8,
                TextAlign::Center => (self.width as i32 - line_width_px) / 2,
            };
            self.draw_line(line, dst, x, baseline, outline);
        }
    }

    fn draw_line(&self, line: &Line, dst: &mut [u8], start_x: i32, baseline: i32, outline: u32) {
        let regular = BitmapFont::default_regular();
        let bold = BitmapFont::default_bold();
        let mut x = start_x;
        for piece in &line.pieces {
            let font = if piece.style.bold { bold } else { regular };
            let shear = if piece.style.italic {
                font.cell_w as f32 / 4.0
            } else {
                0.0
            };
            for ch in piece.text.chars() {
                // Draw outline first (4-offset smear).
                if outline > 0 {
                    for dy in -(outline as i32)..=(outline as i32) {
                        for dx in -(outline as i32)..=(outline as i32) {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            // Only cardinal + diagonals at exactly `outline` distance
                            // to keep the outline sharp, not bloomed.
                            if dx.abs().max(dy.abs()) != outline as i32 {
                                continue;
                            }
                            font.draw_glyph_sheared(
                                ch,
                                dst,
                                self.width,
                                self.height,
                                x + dx,
                                baseline + dy,
                                self.outline_color,
                                shear,
                            );
                        }
                    }
                }
                // Foreground on top.
                font.draw_glyph_sheared(
                    ch,
                    dst,
                    self.width,
                    self.height,
                    x,
                    baseline,
                    piece.style.color,
                    shear,
                );
                x += font.cell_w as i32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Run / line model
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct RunStyle {
    bold: bool,
    italic: bool,
    color: [u8; 4],
}

impl RunStyle {
    fn default_with(color: [u8; 4]) -> Self {
        Self {
            bold: false,
            italic: false,
            color,
        }
    }
}

#[derive(Clone, Debug)]
struct Run {
    text: String,
    style: RunStyle,
}

#[derive(Clone, Debug, Default)]
struct Line {
    pieces: Vec<Run>,
}

fn flatten_segments(segments: &[Segment], style: RunStyle) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    walk(segments, style, &mut out);
    // Collapse adjacent runs that have the same style, to keep lines tidy.
    let mut merged: Vec<Run> = Vec::with_capacity(out.len());
    for run in out {
        if let Some(last) = merged.last_mut() {
            if same_style(&last.style, &run.style) {
                last.text.push_str(&run.text);
                continue;
            }
        }
        merged.push(run);
    }
    merged
}

fn same_style(a: &RunStyle, b: &RunStyle) -> bool {
    a.bold == b.bold && a.italic == b.italic && a.color == b.color
}

fn walk(segments: &[Segment], style: RunStyle, out: &mut Vec<Run>) {
    for seg in segments {
        match seg {
            Segment::Text(s) => {
                out.push(Run {
                    text: s.clone(),
                    style,
                });
            }
            Segment::LineBreak => {
                // Encode as a newline character; the wrapper splits on it.
                out.push(Run {
                    text: "\n".to_string(),
                    style,
                });
            }
            Segment::Bold(children) => {
                let mut s = style;
                s.bold = true;
                walk(children, s, out);
            }
            Segment::Italic(children) => {
                let mut s = style;
                s.italic = true;
                walk(children, s, out);
            }
            Segment::Underline(children) | Segment::Strike(children) => {
                // No glyph-level support — render as plain text.
                walk(children, style, out);
            }
            Segment::Color { rgb, children } => {
                let mut s = style;
                s.color = [rgb.0, rgb.1, rgb.2, 255];
                walk(children, s, out);
            }
            Segment::Font { children, .. } => {
                walk(children, style, out);
            }
            Segment::Voice { name, children } => {
                out.push(Run {
                    text: format!("{name}: "),
                    style,
                });
                walk(children, style, out);
            }
            Segment::Class { children, .. } => {
                walk(children, style, out);
            }
            Segment::Karaoke { children, .. } => {
                // Future: could highlight the active beat. For now, plain text.
                walk(children, style, out);
            }
            Segment::Timestamp { .. } => {
                // Nothing visible.
            }
            Segment::Raw(s) => {
                out.push(Run {
                    text: s.clone(),
                    style,
                });
            }
        }
    }
}

/// Greedy word-wrap. `max_cols` is in *glyph cells* (assumes fixed-width
/// font). Splits first on embedded `\n`, then on spaces. Words longer
/// than `max_cols` are hard-broken.
fn wrap_runs(runs: &[Run], max_cols: usize) -> Vec<Line> {
    // Step 1: split at \n to get raw logical lines, each still a Vec<Run>.
    let mut raw_lines: Vec<Vec<Run>> = vec![Vec::new()];
    for run in runs {
        // Split the text on \n, preserving styles.
        let mut iter = run.text.split('\n').peekable();
        while let Some(piece) = iter.next() {
            if !piece.is_empty() {
                raw_lines.last_mut().unwrap().push(Run {
                    text: piece.to_string(),
                    style: run.style,
                });
            }
            if iter.peek().is_some() {
                raw_lines.push(Vec::new());
            }
        }
    }

    // Step 2: for each logical line, wrap to max_cols.
    let mut out: Vec<Line> = Vec::new();
    for logical in raw_lines {
        // Walk the runs, emitting tokens (word | space) with styles. Greedy
        // accumulate into the current visual line; on overflow, start a new one.
        let tokens = tokenise(&logical);
        let mut current = Line::default();
        let mut current_cols = 0usize;
        for tok in tokens {
            let tok_cols = visible_cols(&tok.text);
            if tok_cols == 0 {
                continue;
            }
            if current_cols == 0 && tok.is_space {
                // Skip leading whitespace on a wrap.
                continue;
            }
            if current_cols + tok_cols > max_cols && current_cols > 0 {
                // Wrap.
                out.push(std::mem::take(&mut current));
                current_cols = 0;
                if tok.is_space {
                    continue;
                }
            }
            // Word larger than a line: hard-break across multiple lines.
            if tok_cols > max_cols && !tok.is_space {
                for chunk in hard_break(&tok.text, max_cols) {
                    if current_cols > 0 {
                        out.push(std::mem::take(&mut current));
                    }
                    append_run(
                        &mut current,
                        Run {
                            text: chunk,
                            style: tok.style,
                        },
                    );
                    current_cols = visible_cols(&current.pieces.last().unwrap().text);
                    if current_cols >= max_cols {
                        out.push(std::mem::take(&mut current));
                        current_cols = 0;
                    }
                }
                continue;
            }
            append_run(
                &mut current,
                Run {
                    text: tok.text,
                    style: tok.style,
                },
            );
            current_cols += tok_cols;
        }
        out.push(current);
    }
    // Prune trailing spaces and fully-empty lines at the tail.
    for line in out.iter_mut() {
        trim_trailing_space(line);
    }
    while out.last().map(|l| is_empty_line(l)).unwrap_or(false) {
        out.pop();
    }
    out
}

fn append_run(line: &mut Line, run: Run) {
    if let Some(last) = line.pieces.last_mut() {
        if same_style(&last.style, &run.style) {
            last.text.push_str(&run.text);
            return;
        }
    }
    line.pieces.push(run);
}

fn trim_trailing_space(line: &mut Line) {
    while let Some(last) = line.pieces.last_mut() {
        let trimmed = last.text.trim_end_matches(' ').to_string();
        if trimmed.is_empty() {
            line.pieces.pop();
        } else {
            last.text = trimmed;
            break;
        }
    }
}

fn is_empty_line(line: &Line) -> bool {
    line.pieces.iter().all(|r| r.text.is_empty())
}

#[derive(Clone, Debug)]
struct Token {
    text: String,
    is_space: bool,
    style: RunStyle,
}

fn tokenise(runs: &[Run]) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    for run in runs {
        let mut buf = String::new();
        let mut buf_is_space: Option<bool> = None;
        for ch in run.text.chars() {
            let is_sp = ch == ' ' || ch == '\t';
            match buf_is_space {
                None => {
                    buf.push(ch);
                    buf_is_space = Some(is_sp);
                }
                Some(prev) if prev == is_sp => buf.push(ch),
                Some(_) => {
                    out.push(Token {
                        text: std::mem::take(&mut buf),
                        is_space: buf_is_space.unwrap(),
                        style: run.style,
                    });
                    buf.push(ch);
                    buf_is_space = Some(is_sp);
                }
            }
        }
        if !buf.is_empty() {
            out.push(Token {
                text: buf,
                is_space: buf_is_space.unwrap_or(false),
                style: run.style,
            });
        }
    }
    out
}

fn visible_cols(s: &str) -> usize {
    // Each char is one cell (fixed-width bitmap font). Tabs count as 1
    // for now; control chars count as 0.
    s.chars().filter(|c| !c.is_control() || *c == '\t').count()
}

fn hard_break(s: &str, max_cols: usize) -> Vec<String> {
    if max_cols == 0 {
        return vec![s.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cols = 0usize;
    for ch in s.chars() {
        if cols >= max_cols {
            out.push(std::mem::take(&mut cur));
            cols = 0;
        }
        cur.push(ch);
        cols += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn measure_line(line: &Line, cell_w: u32) -> u32 {
    let cols: usize = line.pieces.iter().map(|r| visible_cols(&r.text)).sum();
    cols as u32 * cell_w
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CuePosition, Segment, SubtitleCue, TextAlign};

    fn make_cue(segs: Vec<Segment>) -> SubtitleCue {
        SubtitleCue {
            start_us: 0,
            end_us: 1_000_000,
            style_ref: None,
            positioning: None,
            segments: segs,
        }
    }

    #[test]
    fn renders_plain_text() {
        let comp = Compositor::new(320, 240);
        let cue = make_cue(vec![Segment::Text("Hi".to_string())]);
        let buf = comp.render(&cue);
        assert_eq!(buf.len(), 320 * 240 * 4);
        // Some pixel somewhere has alpha > 0.
        assert!(buf.chunks(4).any(|p| p[3] > 0), "no lit pixels");
    }

    #[test]
    fn alignment_right() {
        let mut cue = make_cue(vec![Segment::Text("X".to_string())]);
        cue.positioning = Some(CuePosition {
            align: TextAlign::Right,
            ..Default::default()
        });
        let comp = Compositor::new(200, 100);
        let buf = comp.render(&cue);
        // Right-aligned: lit pixel should exist in right half.
        let lit_right = (0..buf.len() / 4)
            .filter(|i| buf[i * 4 + 3] > 0)
            .filter(|i| {
                let x = i % 200;
                x > 100
            })
            .count();
        assert!(lit_right > 0, "no lit pixels on right side");
    }

    #[test]
    fn handles_empty_cue() {
        let comp = Compositor::new(64, 32);
        let cue = make_cue(vec![]);
        let buf = comp.render(&cue);
        assert!(buf.iter().all(|&b| b == 0));
    }
}
