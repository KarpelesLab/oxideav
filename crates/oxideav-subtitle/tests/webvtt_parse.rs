//! WebVTT parsing, writing, and round-trip.

use oxideav_core::Segment;
use oxideav_subtitle::webvtt;

const SAMPLE: &str = "WEBVTT Language: en

STYLE
::cue(.yellow) {
  color: yellow;
  font-weight: bold;
}

STYLE
::cue(.blue) {
  color: blue;
  font-style: italic;
}

REGION
id:speaker
width:40%

00:00:01.000 --> 00:00:03.500 position:25% line:90% align:center
<v Alice>Hello <c.yellow>world</c></v>

cue-2
00:00:04.000 --> 00:00:05.500
<b>bold</b> then <i>italic</i>
second line
";

#[test]
fn parses_header_and_style() {
    let t = webvtt::parse(SAMPLE.as_bytes()).unwrap();
    // Header trailing stored in metadata.
    assert!(t
        .metadata
        .iter()
        .any(|(k, v)| k == "header" && v == "Language: en"));
    // Two style classes + one region.
    let yellow = t.styles.iter().find(|s| s.name == "yellow").unwrap();
    assert!(yellow.bold);
    let blue = t.styles.iter().find(|s| s.name == "blue").unwrap();
    assert!(blue.italic);
    assert!(t.styles.iter().any(|s| s.name == "region:speaker"));
}

#[test]
fn parses_voice_and_class() {
    let t = webvtt::parse(SAMPLE.as_bytes()).unwrap();
    let c0 = &t.cues[0];
    assert_eq!(c0.start_us, 1_000_000);
    assert_eq!(c0.end_us, 3_500_000);
    // Positioning present.
    let pos = c0.positioning.as_ref().unwrap();
    assert_eq!(pos.x, Some(25.0));
    assert_eq!(pos.y, Some(90.0));
    // Voice + Class in the segment tree.
    let mut saw_voice = false;
    let mut saw_class = false;
    visit(&c0.segments, &mut |s| match s {
        Segment::Voice { name, .. } if name == "Alice" => saw_voice = true,
        Segment::Class { name, .. } if name == "yellow" => saw_class = true,
        _ => {}
    });
    assert!(saw_voice);
    assert!(saw_class);
}

#[test]
fn parses_b_i_multiline() {
    let t = webvtt::parse(SAMPLE.as_bytes()).unwrap();
    let c1 = &t.cues[1];
    let mut saw_bold = false;
    let mut saw_italic = false;
    visit(&c1.segments, &mut |s| match s {
        Segment::Bold(_) => saw_bold = true,
        Segment::Italic(_) => saw_italic = true,
        _ => {}
    });
    assert!(saw_bold);
    assert!(saw_italic);
}

#[test]
fn write_roundtrips_signatures() {
    let t = webvtt::parse(SAMPLE.as_bytes()).unwrap();
    let out = String::from_utf8(webvtt::write(&t)).unwrap();
    assert!(out.starts_with("WEBVTT"));
    assert!(out.contains("00:00:01.000 --> 00:00:03.500"));
    assert!(out.contains("<v Alice>"));

    let t2 = webvtt::parse(out.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), t.cues.len());
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
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
