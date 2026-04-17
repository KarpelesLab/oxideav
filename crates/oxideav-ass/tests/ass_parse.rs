//! ASS parsing, writing, and round-trip.

use oxideav_ass as ass;
use oxideav_core::Segment;
use oxideav_subtitle::ir::plain_text;

const SAMPLE: &str = r"[Script Info]
; Authored by test
Title: Test Show
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1
Style: Caption,Verdana,18,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1,0,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\b1}Hello{\b0} world
Dialogue: 0,0:00:04.00,0:00:05.50,Caption,,0,0,0,,{\pos(100,200)}positioned, line with, commas
Dialogue: 0,0:00:06.00,0:00:08.00,Default,,0,0,0,,{\i1}line one{\i0}\Nline two
";

#[test]
fn parses_script_info_and_styles() {
    let t = ass::parse(SAMPLE.as_bytes()).unwrap();
    assert!(t
        .metadata
        .iter()
        .any(|(k, v)| k == "title" && v == "Test Show"));
    assert!(t
        .metadata
        .iter()
        .any(|(k, v)| k == "playresx" && v == "384"));
    let default = t.styles.iter().find(|s| s.name == "Default").unwrap();
    assert_eq!(default.font_size, Some(20.0));
    let caption = t.styles.iter().find(|s| s.name == "Caption").unwrap();
    assert!(caption.bold);
}

#[test]
fn parses_dialogue_and_overrides() {
    let t = ass::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 3);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_000_000);
    // First segment is Bold wrapping "Hello".
    match &t.cues[0].segments[0] {
        Segment::Bold(inner) => match &inner[0] {
            Segment::Text(s) => assert_eq!(s, "Hello"),
            other => panic!("expected text, got {other:?}"),
        },
        other => panic!("expected bold, got {other:?}"),
    }
}

#[test]
fn parses_pos_override() {
    let t = ass::parse(SAMPLE.as_bytes()).unwrap();
    let c1 = &t.cues[1];
    let pos = c1.positioning.as_ref().unwrap();
    assert_eq!(pos.x, Some(100.0));
    assert_eq!(pos.y, Some(200.0));
    // Commas in text preserved.
    let plain = plain_text(&c1.segments);
    assert!(plain.contains("line with, commas"), "got: {plain}");
}

#[test]
fn parses_linebreak() {
    let t = ass::parse(SAMPLE.as_bytes()).unwrap();
    let c2 = &t.cues[2];
    let mut saw_break = false;
    visit(&c2.segments, &mut |s| {
        if matches!(s, Segment::LineBreak) {
            saw_break = true;
        }
    });
    assert!(saw_break);
}

#[test]
fn write_preserves_events_and_styles() {
    let t = ass::parse(SAMPLE.as_bytes()).unwrap();
    let out = String::from_utf8(ass::write(&t)).unwrap();
    // Events re-emitted.
    assert!(out.contains("[Events]"));
    assert!(out.contains("Dialogue: 0,0:00:01.00,0:00:03.00,Default"));
    assert!(out.contains("Style: Caption"));
    // Bold override re-emitted.
    assert!(out.contains("{\\b1}"));
    // Positioning preserved.
    assert!(out.contains("\\pos(100,200)"));

    // Reparsing yields same cue count + timings + style refs.
    let t2 = ass::parse(out.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), t.cues.len());
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
        assert_eq!(a.style_ref, b.style_ref);
    }
}

fn visit<F: FnMut(&Segment)>(segs: &[Segment], f: &mut F) {
    for s in segs {
        f(s);
        match s {
            Segment::Bold(c) | Segment::Italic(c) | Segment::Underline(c) | Segment::Strike(c) => {
                visit(c, f)
            }
            Segment::Color { children, .. }
            | Segment::Font { children, .. }
            | Segment::Voice { children, .. }
            | Segment::Class { children, .. }
            | Segment::Karaoke { children, .. } => visit(children, f),
            _ => {}
        }
    }
}
