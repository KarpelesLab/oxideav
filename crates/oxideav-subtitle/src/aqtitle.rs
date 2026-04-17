//! AQTitle (.aqt) parser and writer.
//!
//! Minimalist frame-based format:
//!
//! ```text
//! -->> 25
//! Hello world
//! -->> 75
//! Second cue
//! line two
//! -->> 150
//! ```
//!
//! Each `-->> N` line starts a cue at frame `N`, and ends the previous
//! cue at that same frame. If there's no trailing `-->> N` after the
//! last cue, the writer appends a 3-second (in frames) tail. 25 fps is
//! assumed.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::SubtitleTrack;

pub const CODEC_ID: &str = "aqtitle";
pub const DEFAULT_FPS: f64 = 25.0;

/// Frames added to the final cue when the file has no trailing marker.
const TRAILING_FRAMES: i64 = 75; // ~3 s at 25 fps

/// Parse an AQTitle payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    let fps = DEFAULT_FPS;

    // Pass 1: build a sequence of (start_frame, body_lines).
    let mut blocks: Vec<(i64, Vec<String>)> = Vec::new();
    let mut current: Option<(i64, Vec<String>)> = None;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if let Some(frame) = parse_marker(trimmed) {
            if let Some(prev) = current.take() {
                blocks.push(prev);
            }
            current = Some((frame, Vec::new()));
        } else if let Some(cur) = &mut current {
            if trimmed.is_empty() {
                // Blank line: treat as separator only if cur already has
                // text — else ignore.
                if !cur.1.is_empty() {
                    // keep blank as-is? The AQT spec joins all following
                    // non-marker lines into one cue. We emit a blank
                    // "line" to preserve intent on re-write.
                    cur.1.push(String::new());
                }
            } else {
                cur.1.push(trimmed.to_string());
            }
        }
        // Lines before the first marker are ignored (format has no header).
    }
    if let Some(last) = current {
        blocks.push(last);
    }

    // Trim any trailing empty strings from each body.
    for (_, body) in &mut blocks {
        while body.last().map(|s| s.is_empty()).unwrap_or(false) {
            body.pop();
        }
    }

    // Pass 2: end frame = next block's start, or trailing fallback.
    let mut cues: Vec<SubtitleCue> = Vec::with_capacity(blocks.len());
    for i in 0..blocks.len() {
        let (start_frame, body) = &blocks[i];
        // An empty-body marker can be used as a sentinel to end the
        // preceding cue; skip it from the output.
        if body.is_empty() {
            continue;
        }
        let end_frame = if i + 1 < blocks.len() {
            blocks[i + 1].0.max(*start_frame + 1)
        } else {
            *start_frame + TRAILING_FRAMES
        };
        cues.push(SubtitleCue {
            start_us: frame_to_us(*start_frame, fps),
            end_us: frame_to_us(end_frame, fps),
            style_ref: None,
            positioning: None,
            segments: lines_to_segments(body),
        });
    }

    Ok(SubtitleTrack {
        cues,
        ..SubtitleTrack::default()
    })
}

/// Re-emit a track as AQTitle bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let fps = DEFAULT_FPS;
    let mut out = String::new();
    for cue in &track.cues {
        let sf = us_to_frame(cue.start_us, fps);
        out.push_str(&format!("-->> {}\n", sf));
        out.push_str(&render_body(&cue.segments));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    // Emit a sentinel marker for the last cue's end, if any cues exist.
    if let Some(last) = track.cues.last() {
        let ef = us_to_frame(last.end_us, fps);
        out.push_str(&format!("-->> {}\n", ef));
    }
    Ok(out.into_bytes())
}

/// Quick probe — look for a `-->>` marker near the top.
pub fn probe(buf: &[u8]) -> u8 {
    let text = strip_bom(buf);
    let mut checked = 0;
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        checked += 1;
        if parse_marker(line).is_some() {
            return 80;
        }
        if checked >= 5 {
            break;
        }
    }
    0
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an aqtitle codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(AqtDecoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an aqtitle codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(AqtEncoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------
// Packet shape: a single cue encoded as:
//
//   -->> <start_frame>
//   body line 1
//   body line 2
//   -->> <end_frame>
//
// so the decoder can recover both endpoints from the payload alone.

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let fps = DEFAULT_FPS;
    let sf = us_to_frame(cue.start_us, fps);
    let ef = us_to_frame(cue.end_us, fps);
    let mut s = format!("-->> {}\n", sf);
    s.push_str(&render_body(&cue.segments));
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s.push_str(&format!("-->> {}", ef));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
    let mut start_frame: Option<i64> = None;
    let mut end_frame: Option<i64> = None;
    let mut body: Vec<String> = Vec::new();
    for line in &lines {
        let t = line.trim();
        if let Some(f) = parse_marker(t) {
            if start_frame.is_none() {
                start_frame = Some(f);
            } else {
                end_frame = Some(f);
                break;
            }
        } else if start_frame.is_some() {
            if !t.is_empty() {
                body.push(t.to_string());
            }
        }
    }
    let sf = start_frame.ok_or_else(|| Error::invalid("aqtitle: missing start marker"))?;
    let ef = end_frame.unwrap_or(sf + TRAILING_FRAMES);
    Ok(SubtitleCue {
        start_us: frame_to_us(sf, DEFAULT_FPS),
        end_us: frame_to_us(ef, DEFAULT_FPS),
        style_ref: None,
        positioning: None,
        segments: lines_to_segments(&body),
    })
}

// ---------------------------------------------------------------------------

fn parse_marker(line: &str) -> Option<i64> {
    // Canonical form is `-->> N`; accept optional whitespace and a
    // variable number of `-` / `>` so real-world variants still parse.
    let t = line.trim();
    let t = t.strip_prefix("-->>").or_else(|| t.strip_prefix("-->"))?;
    let t = t.trim_start();
    let num: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if num.is_empty() {
        return None;
    }
    num.parse().ok()
}

fn lines_to_segments(lines: &[String]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            out.push(Segment::LineBreak);
        }
        if !l.is_empty() {
            out.push(Segment::Text(l.clone()));
        }
    }
    out
}

fn render_body(segments: &[Segment]) -> String {
    let mut out = String::new();
    append_flat(segments, &mut out);
    out
}

fn append_flat(segs: &[Segment], out: &mut String) {
    for seg in segs {
        match seg {
            Segment::Text(s) => out.push_str(s),
            Segment::LineBreak => out.push('\n'),
            Segment::Bold(c) | Segment::Italic(c) | Segment::Underline(c) | Segment::Strike(c) => {
                append_flat(c, out)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => append_flat(children, out),
            Segment::Timestamp { .. } => {}
            Segment::Raw(_) => {}
        }
    }
}

fn frame_to_us(f: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((f as f64 / fps) * 1_000_000.0).round() as i64
}

fn us_to_frame(us: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return 0;
    }
    ((us as f64 / 1_000_000.0) * fps).round() as i64
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

struct AqtDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for AqtDecoder {
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

struct AqtEncoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for AqtEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("aqtitle encoder: expected Frame::Subtitle")),
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
    fn parse_basic() {
        let src = "-->> 25\nHello world\n-->> 75\nSecond line\n-->> 150\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, frame_to_us(25, DEFAULT_FPS));
        assert_eq!(t.cues[0].end_us, frame_to_us(75, DEFAULT_FPS));
        assert_eq!(t.cues[1].end_us, frame_to_us(150, DEFAULT_FPS));
    }

    #[test]
    fn parse_multiline_body() {
        let src = "-->> 10\nline1\nline2\n-->> 30\n";
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
        assert!(probe(b"-->> 25\nhi\n") > 0);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"random text\n"), 0);
    }
}
