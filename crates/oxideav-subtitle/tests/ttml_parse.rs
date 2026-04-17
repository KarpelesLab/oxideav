//! TTML parsing, writing, and round-trip.
//!
//! Targets the public `oxideav_subtitle::ttml` module — available once
//! the caller adds `pub mod ttml;` to `lib.rs`.

use oxideav_core::Segment;
use oxideav_subtitle::ttml;

const SAMPLE: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<tt xmlns=\"http://www.w3.org/ns/ttml\" xmlns:tts=\"http://www.w3.org/ns/ttml#styling\" xml:lang=\"en\">\n\
  <head>\n\
    <styling>\n\
      <style xml:id=\"s1\" tts:color=\"yellow\" tts:fontWeight=\"bold\"/>\n\
    </styling>\n\
  </head>\n\
  <body>\n\
    <div>\n\
      <p begin=\"00:00:01.000\" end=\"00:00:03.000\" style=\"s1\">Hello <span tts:color=\"#FF0000\">world</span></p>\n\
      <p begin=\"00:00:04.500\" end=\"00:00:06.000\">Line one<br/>Line two</p>\n\
    </div>\n\
  </body>\n\
</tt>\n";

#[test]
fn parses_two_cues() {
    let t = ttml::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 2);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_000_000);
    assert_eq!(t.cues[0].style_ref.as_deref(), Some("s1"));
    assert_eq!(t.cues[1].start_us, 4_500_000);
    assert_eq!(t.cues[1].end_us, 6_000_000);
}

#[test]
fn parses_named_style() {
    let t = ttml::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.styles.len(), 1);
    assert_eq!(t.styles[0].name, "s1");
    assert!(t.styles[0].bold);
}

#[test]
fn preserves_inline_color_span() {
    let t = ttml::parse(SAMPLE.as_bytes()).unwrap();
    let segs = &t.cues[0].segments;
    let mut saw_color = false;
    visit(segs, &mut |s| {
        if let Segment::Color { rgb, .. } = s {
            if *rgb == (255, 0, 0) {
                saw_color = true;
            }
        }
    });
    assert!(saw_color, "expected #FF0000 color span");
}

#[test]
fn preserves_linebreak_in_second_cue() {
    let t = ttml::parse(SAMPLE.as_bytes()).unwrap();
    let segs = &t.cues[1].segments;
    let mut saw_br = false;
    visit(segs, &mut |s| {
        if matches!(s, Segment::LineBreak) {
            saw_br = true;
        }
    });
    assert!(saw_br, "expected LineBreak from <br/>");
}

#[test]
fn write_roundtrips_basic_shape() {
    let t = ttml::parse(SAMPLE.as_bytes()).unwrap();
    let out = ttml::write(&t);
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("<tt"));
    assert!(s.contains("<body>"));
    assert!(s.contains("begin=\"00:00:01.000\""));
    assert!(s.contains("begin=\"00:00:04.500\""));

    // Reparse the output and confirm timing fidelity.
    let t2 = ttml::parse(s.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), 2);
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
    }
}

#[test]
fn probe_positive_and_negative() {
    assert!(ttml::probe(SAMPLE.as_bytes()) > 60);
    assert_eq!(ttml::probe(b"WEBVTT\n"), 0);
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
