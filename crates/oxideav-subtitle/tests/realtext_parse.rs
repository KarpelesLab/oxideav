//! RealText parsing, writing, probe.

use oxideav_core::Segment;
use oxideav_subtitle::realtext;

const SAMPLE: &str = "<window type=\"generic\" duration=\"30.0\">
<time begin=\"0.0\" end=\"2.5\"/>
<font color=\"#FF0000\">Hello</font><br/>
<time begin=\"2.5\" end=\"5.0\"/>
<b>World</b>
<time begin=\"5.0\" end=\"7.5\"/>
<i>italic</i>
<time begin=\"7.5\" end=\"10.0\"/>
<u>under</u>
<time begin=\"10.0\" end=\"12.0\"/>
plain text
</window>
";

#[test]
fn five_cues() {
    let t = realtext::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 5);
    assert_eq!(t.cues[0].start_us, 0);
    assert_eq!(t.cues[0].end_us, 2_500_000);
    assert_eq!(t.cues[1].start_us, 2_500_000);
    assert_eq!(t.cues[4].end_us, 12_000_000);
}

#[test]
fn parses_color_bold_italic_underline() {
    let t = realtext::parse(SAMPLE.as_bytes()).unwrap();
    let mut saw_color = false;
    let mut saw_bold = false;
    let mut saw_italic = false;
    let mut saw_u = false;
    let mut saw_lb = false;
    for cue in &t.cues {
        visit(&cue.segments, &mut |s| match s {
            Segment::Color { rgb, .. } => {
                saw_color = true;
                assert_eq!(*rgb, (0xFF, 0, 0));
            }
            Segment::Bold(_) => saw_bold = true,
            Segment::Italic(_) => saw_italic = true,
            Segment::Underline(_) => saw_u = true,
            Segment::LineBreak => saw_lb = true,
            _ => {}
        });
    }
    assert!(saw_color);
    assert!(saw_bold);
    assert!(saw_italic);
    assert!(saw_u);
    assert!(saw_lb);
}

#[test]
fn write_roundtrips() {
    let t = realtext::parse(SAMPLE.as_bytes()).unwrap();
    let out = realtext::write(&t).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("<window"));
    assert!(s.contains("</window>"));
    assert!(s.contains("<time begin="));
    let t2 = realtext::parse(s.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), t.cues.len());
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
    }
}

#[test]
fn probe_positive() {
    assert!(realtext::probe(SAMPLE.as_bytes()) > 0);
}

#[test]
fn probe_rejects_other() {
    assert_eq!(realtext::probe(b"WEBVTT\n"), 0);
    assert_eq!(
        realtext::probe(b"1\n00:00:01,000 --> 00:00:02,000\nhi\n"),
        0
    );
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
