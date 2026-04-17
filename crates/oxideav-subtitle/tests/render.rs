//! Integration tests for the subtitle compositor + RenderedSubtitleDecoder.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Error, Frame, Packet, PixelFormat, Result, Segment, SubtitleCue};
use oxideav_subtitle::{make_rendered_decoder, Compositor, RenderedSubtitleDecoder};

fn mkcue(segs: Vec<Segment>) -> SubtitleCue {
    SubtitleCue {
        start_us: 1_000_000,
        end_us: 2_000_000,
        style_ref: None,
        positioning: None,
        segments: segs,
    }
}

fn rgba_alpha(buf: &[u8], w: u32, x: u32, y: u32) -> u8 {
    let idx = (y as usize * w as usize + x as usize) * 4 + 3;
    buf.get(idx).copied().unwrap_or(0)
}

fn bounding_box(buf: &[u8], w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;
    for y in 0..h {
        for x in 0..w {
            if rgba_alpha(buf, w, x, y) > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }
    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

#[test]
fn renders_plain_text_cue() {
    let comp = Compositor::new(640, 480);
    let cue = mkcue(vec![Segment::Text("Hello".to_string())]);
    let buf = comp.render(&cue);
    assert_eq!(buf.len(), 640 * 480 * 4);

    let (min_x, min_y, max_x, max_y) =
        bounding_box(&buf, 640, 480).expect("rendered cue had no lit pixels");

    // Lower-middle region.
    assert!(
        min_y > 240,
        "text should be below mid-height; bbox y={min_y}..{max_y}"
    );
    // Centered horizontally — bbox should straddle the centre column or
    // at least start before it.
    assert!(
        min_x < 320 && max_x > 320,
        "text should straddle horizontal centre; bbox x={min_x}..{max_x}"
    );
    // Upper area (y < 200): no lit pixels at all.
    for y in 0..200 {
        for x in 0..640 {
            assert_eq!(
                rgba_alpha(&buf, 640, x, y),
                0,
                "unexpected lit pixel at ({x}, {y})"
            );
        }
    }
}

#[test]
fn bold_italic_color() {
    let comp = Compositor::new(320, 240);
    let cue = mkcue(vec![
        Segment::Bold(vec![Segment::Text("Bold".to_string())]),
        Segment::Text(" ".to_string()),
        Segment::Italic(vec![Segment::Text("Italic".to_string())]),
        Segment::Text(" ".to_string()),
        Segment::Color {
            rgb: (255, 0, 0),
            children: vec![Segment::Text("Red".to_string())],
        },
    ]);
    let buf = comp.render(&cue);
    assert_eq!(buf.len(), 320 * 240 * 4);
    let lit = buf.chunks(4).filter(|p| p[3] > 0).count();
    assert!(lit > 0, "bold/italic/color cue produced no pixels");
}

#[test]
fn linebreak_multiple_lines() {
    let comp = Compositor::new(320, 240);
    // Single-line cue first.
    let cue_one = mkcue(vec![Segment::Text("Line".to_string())]);
    let buf_one = comp.render(&cue_one);
    let (_, one_min_y, _, one_max_y) = bounding_box(&buf_one, 320, 240).expect("single line");

    // Two-line cue.
    let cue_two = mkcue(vec![
        Segment::Text("Line".to_string()),
        Segment::LineBreak,
        Segment::Text("Two".to_string()),
    ]);
    let buf_two = comp.render(&cue_two);
    let (_, two_min_y, _, two_max_y) = bounding_box(&buf_two, 320, 240).expect("two lines");

    let one_height = one_max_y - one_min_y;
    let two_height = two_max_y - two_min_y;
    assert!(
        two_height > one_height,
        "two-line cue bbox height ({two_height}) must exceed single-line ({one_height})"
    );
}

#[test]
fn wrap_long_line() {
    let long = "x".repeat(400);
    let cue = mkcue(vec![Segment::Text(long)]);
    let comp = Compositor::new(320, 240);
    let buf = comp.render(&cue);
    // Count rows that have at least one lit pixel.
    let mut non_empty_rows = 0u32;
    for y in 0..240 {
        let any = (0..320).any(|x| rgba_alpha(&buf, 320, x, y) > 0);
        if any {
            non_empty_rows += 1;
        }
    }
    // Each bitmap line spans cell_h = 16 rows; multiple visual lines
    // should produce well above 16 lit rows.
    assert!(
        non_empty_rows > 20,
        "wrapped 400-char cue produced only {non_empty_rows} lit rows; \
         expected multi-line wrap to fill many more"
    );
}

// ------------------------------------------------------------------
// Wrapper-level test
// ------------------------------------------------------------------

/// A tiny fake decoder that emits a fixed queue of Frames regardless of
/// packets. Useful for testing wrappers.
struct CannedDecoder {
    codec_id: CodecId,
    queue: VecDeque<Frame>,
    flushed: bool,
}

impl CannedDecoder {
    fn new(frames: Vec<Frame>) -> Self {
        Self {
            codec_id: CodecId::new("test_canned"),
            queue: frames.into_iter().collect(),
            flushed: false,
        }
    }
}

impl Decoder for CannedDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.queue.pop_front() {
            Ok(f)
        } else if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.flushed = true;
        Ok(())
    }
}

#[test]
fn wrapper_deduplicates_identical_cues() {
    let cue = mkcue(vec![Segment::Text("Same".to_string())]);
    let inner = Box::new(CannedDecoder::new(vec![
        Frame::Subtitle(cue.clone()),
        Frame::Subtitle(cue),
    ]));
    let mut wrapper = RenderedSubtitleDecoder::new(inner, 160, 120);

    // First call: fresh cue → Frame::Video.
    match wrapper.receive_frame() {
        Ok(Frame::Video(vf)) => {
            assert_eq!(vf.format, PixelFormat::Rgba);
            assert_eq!(vf.width, 160);
            assert_eq!(vf.height, 120);
            assert_eq!(vf.pts, Some(1_000_000));
            assert_eq!(vf.planes.len(), 1);
            assert_eq!(vf.planes[0].stride, 160 * 4);
            assert_eq!(vf.planes[0].data.len(), 160 * 120 * 4);
        }
        other => panic!("expected Frame::Video on first cue; got {other:?}"),
    }

    // Second call: duplicate → NeedMore.
    match wrapper.receive_frame() {
        Err(Error::NeedMore) => {}
        other => panic!("expected NeedMore on duplicate; got {other:?}"),
    }
}

#[test]
fn wrapper_emits_new_frame_on_content_change() {
    let cue_a = mkcue(vec![Segment::Text("A".to_string())]);
    let mut cue_b = cue_a.clone();
    cue_b.segments = vec![Segment::Text("B".to_string())];
    cue_b.start_us = 2_000_000;
    cue_b.end_us = 3_000_000;

    let inner = Box::new(CannedDecoder::new(vec![
        Frame::Subtitle(cue_a),
        Frame::Subtitle(cue_b),
    ]));
    let mut wrapper = RenderedSubtitleDecoder::new(inner, 160, 120);
    assert!(matches!(wrapper.receive_frame(), Ok(Frame::Video(_))));
    assert!(matches!(wrapper.receive_frame(), Ok(Frame::Video(_))));
}

#[test]
fn make_rendered_decoder_factory() {
    let cue = mkcue(vec![Segment::Text("X".to_string())]);
    let inner: Box<dyn Decoder> = Box::new(CannedDecoder::new(vec![Frame::Subtitle(cue)]));
    let mut wrapper = make_rendered_decoder(inner, 64, 64);
    assert!(matches!(wrapper.receive_frame(), Ok(Frame::Video(_))));
}
