//! SRT parsing, writing, and round-trip.

use oxideav_core::Segment;
use oxideav_subtitle::srt;

const SAMPLE: &str = "\u{FEFF}1
00:00:01,000 --> 00:00:03,500
<b>Hello</b>, <i>world</i>!
<u>second line</u>

2
00:00:05,250 --> 00:00:06,750
<font color=\"#FF0000\">red</font>
<s>strike</s>

3
00:00:10,000 --> 00:00:12,000
plain
multi
line
";

#[test]
fn three_cues() {
    let t = srt::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 3);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_500_000);
    assert_eq!(t.cues[1].start_us, 5_250_000);
    assert_eq!(t.cues[1].end_us, 6_750_000);
    assert_eq!(t.cues[2].start_us, 10_000_000);
    assert_eq!(t.cues[2].end_us, 12_000_000);
}

#[test]
fn preserves_bold_italic_underline() {
    let t = srt::parse(SAMPLE.as_bytes()).unwrap();
    let segs = &t.cues[0].segments;
    assert!(matches!(&segs[0], Segment::Bold(_)), "expected bold first");
    // Find italic somewhere in the segment tree.
    let mut saw_italic = false;
    let mut saw_underline = false;
    visit(segs, &mut |s| match s {
        Segment::Italic(_) => saw_italic = true,
        Segment::Underline(_) => saw_underline = true,
        _ => {}
    });
    assert!(saw_italic, "expected italic segment");
    assert!(saw_underline, "expected underline segment");
}

#[test]
fn preserves_color_and_strike() {
    let t = srt::parse(SAMPLE.as_bytes()).unwrap();
    let segs = &t.cues[1].segments;
    let mut saw_color = false;
    let mut saw_strike = false;
    visit(segs, &mut |s| match s {
        Segment::Color { rgb, .. } => {
            saw_color = true;
            assert_eq!(*rgb, (255, 0, 0));
        }
        Segment::Strike(_) => saw_strike = true,
        _ => {}
    });
    assert!(saw_color);
    assert!(saw_strike);
}

#[test]
fn write_roundtrips_tags() {
    let t = srt::parse(SAMPLE.as_bytes()).unwrap();
    let out = srt::write(&t);
    let out_str = String::from_utf8(out).unwrap();
    assert!(out_str.contains("<b>Hello</b>"));
    assert!(out_str.contains("<i>world</i>"));
    // Indexes renumbered starting at 1.
    assert!(
        out_str.starts_with("1\n"),
        "want leading `1\\n`, got: {out_str}"
    );
    assert!(out_str.contains("\n2\n"));
    assert!(out_str.contains("\n3\n"));

    // Reparsing the output produces the same cue count + timings.
    let t2 = srt::parse(out_str.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), 3);
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
    }
}

#[test]
fn missing_index_is_tolerated() {
    let src = "00:00:01,000 --> 00:00:02,000\nhi\n\n";
    let t = srt::parse(src.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 1);
    assert_eq!(t.cues[0].start_us, 1_000_000);
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
