//! MPL2 (.mpl) parser and writer.
//!
//! Simple line-based format with deciseconds (1/10 s) timing:
//!
//! ```text
//! [0][25]Hello|world
//! [30][60]/italic line
//! ```
//!
//! * `[start_ds][end_ds]text` per line.
//! * `|` is a hard line break.
//! * A leading `/` on any line (after a `|` split) means that line is
//!   italic.
//!
//! No other inline formatting; unknown content is preserved verbatim.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, Segment, SubtitleCue,
    TimeBase,
};

use crate::ir::SubtitleTrack;

pub const CODEC_ID: &str = "mpl2";

/// Parse an MPL2 payload.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    let text = strip_bom(bytes);
    let mut cues: Vec<SubtitleCue> = Vec::new();
    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let (start_ds, end_ds, rest) = match parse_header(line) {
            Some(v) => v,
            None => continue,
        };
        cues.push(SubtitleCue {
            start_us: ds_to_us(start_ds),
            end_us: ds_to_us(end_ds),
            style_ref: None,
            positioning: None,
            segments: parse_inline(rest),
        });
    }
    Ok(SubtitleTrack {
        cues,
        ..SubtitleTrack::default()
    })
}

/// Re-emit a track as MPL2 bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let mut out = String::new();
    for cue in &track.cues {
        let sd = us_to_ds(cue.start_us);
        let ed = us_to_ds(cue.end_us);
        out.push_str(&format!("[{}][{}]", sd, ed));
        out.push_str(&render_inline(&cue.segments));
        out.push('\n');
    }
    Ok(out.into_bytes())
}

/// Quick probe: look for the `[n][n]...` shape.
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
        if parse_header(line).is_some() {
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
        40
    } else {
        0
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an mpl2 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    Ok(Box::new(Mpl2Decoder {
        codec_id: params.codec_id.clone(),
        pending: VecDeque::new(),
        eof: false,
    }))
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    if params.codec_id.as_str() != CODEC_ID {
        return Err(Error::unsupported(format!(
            "not an mpl2 codec id: {}",
            params.codec_id.as_str()
        )));
    }
    let mut p = params.clone();
    p.media_type = MediaType::Subtitle;
    Ok(Box::new(Mpl2Encoder {
        params: p,
        pending: VecDeque::new(),
    }))
}

// ---------------------------------------------------------------------------
// Packet shape

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    let sd = us_to_ds(cue.start_us);
    let ed = us_to_ds(cue.end_us);
    let mut s = format!("[{}][{}]", sd, ed);
    s.push_str(&render_inline(&cue.segments));
    s.into_bytes()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    let text = strip_bom(bytes);
    let line = text
        .lines()
        .map(|l| l.trim_end_matches('\r').trim())
        .find(|l| !l.is_empty())
        .ok_or_else(|| Error::invalid("mpl2: empty cue"))?;
    let (sd, ed, rest) =
        parse_header(line).ok_or_else(|| Error::invalid("mpl2: bad cue header"))?;
    Ok(SubtitleCue {
        start_us: ds_to_us(sd),
        end_us: ds_to_us(ed),
        style_ref: None,
        positioning: None,
        segments: parse_inline(rest),
    })
}

// ---------------------------------------------------------------------------

fn parse_header(line: &str) -> Option<(i64, i64, &str)> {
    let line = line.trim_start();
    let rest = line.strip_prefix('[')?;
    let e1 = rest.find(']')?;
    let ds1: i64 = rest[..e1].trim().parse().ok()?;
    let after1 = &rest[e1 + 1..];
    let after1 = after1.strip_prefix('[')?;
    let e2 = after1.find(']')?;
    let ds2: i64 = after1[..e2].trim().parse().ok()?;
    Some((ds1, ds2, &after1[e2 + 1..]))
}

fn parse_inline(body: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (idx, piece) in body.split('|').enumerate() {
        if idx > 0 {
            out.push(Segment::LineBreak);
        }
        if let Some(rest) = piece.strip_prefix('/') {
            // Italicise this line only.
            let mut inner: Vec<Segment> = Vec::new();
            if !rest.is_empty() {
                inner.push(Segment::Text(rest.to_string()));
            }
            out.push(Segment::Italic(inner));
        } else if !piece.is_empty() {
            out.push(Segment::Text(piece.to_string()));
        }
    }
    out
}

fn render_inline(segments: &[Segment]) -> String {
    // Render line-by-line so the `/` italic marker lands on a fresh
    // segment of text. We flatten nested styles to plain text with the
    // single `/` italic marker when applicable.
    let mut lines: Vec<String> = vec![String::new()];
    let mut line_italic: Vec<bool> = vec![false];
    append_flat(segments, &mut lines, &mut line_italic, false);
    let mut out = String::new();
    for (i, (line, is_italic)) in lines.iter().zip(line_italic.iter()).enumerate() {
        if i > 0 {
            out.push('|');
        }
        if *is_italic && !line.is_empty() {
            out.push('/');
        }
        // `|` in text is reserved as a line break; sanitise.
        out.push_str(&line.replace('|', "/"));
    }
    out
}

fn append_flat(
    segments: &[Segment],
    lines: &mut Vec<String>,
    line_italic: &mut Vec<bool>,
    italic: bool,
) {
    for seg in segments {
        match seg {
            Segment::Text(s) => {
                let idx = lines.len() - 1;
                lines[idx].push_str(s);
                if italic && !s.is_empty() {
                    line_italic[idx] = true;
                }
            }
            Segment::LineBreak => {
                lines.push(String::new());
                line_italic.push(false);
            }
            Segment::Italic(c) => append_flat(c, lines, line_italic, true),
            Segment::Bold(c) | Segment::Underline(c) | Segment::Strike(c) => {
                append_flat(c, lines, line_italic, italic)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => {
                append_flat(children, lines, line_italic, italic)
            }
            Segment::Timestamp { .. } => {}
            Segment::Raw(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------

fn ds_to_us(ds: i64) -> i64 {
    ds * 100_000
}

fn us_to_ds(us: i64) -> i64 {
    (us + 50_000) / 100_000
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

struct Mpl2Decoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for Mpl2Decoder {
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

struct Mpl2Encoder {
    params: CodecParameters,
    pending: VecDeque<Packet>,
}

impl Encoder for Mpl2Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let cue = match frame {
            Frame::Subtitle(c) => c,
            _ => return Err(Error::invalid("mpl2 encoder: expected Frame::Subtitle")),
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
        let src = "[0][25]Hello world\n[30][60]Second line\n";
        let t = parse(src.as_bytes()).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 0);
        assert_eq!(t.cues[0].end_us, 2_500_000);
        assert_eq!(t.cues[1].start_us, 3_000_000);
    }

    #[test]
    fn parse_italic_prefix() {
        let src = "[0][10]/just italic\n";
        let t = parse(src.as_bytes()).unwrap();
        match &t.cues[0].segments[0] {
            Segment::Italic(_) => {}
            other => panic!("expected italic: {other:?}"),
        }
    }

    #[test]
    fn line_break() {
        let src = "[0][10]line1|line2\n";
        let t = parse(src.as_bytes()).unwrap();
        assert!(t.cues[0]
            .segments
            .iter()
            .any(|s| matches!(s, Segment::LineBreak)));
    }

    #[test]
    fn probe_positive() {
        assert!(probe(b"[0][10]hi\n") > 0);
    }

    #[test]
    fn probe_negative() {
        assert_eq!(probe(b"random\ntext\n"), 0);
    }
}
