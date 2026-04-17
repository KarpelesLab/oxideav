//! JACOsub parsing, writing, probe.

use oxideav_core::Segment;
use oxideav_subtitle::jacosub;

const SAMPLE: &str = "#TITLE Demo show
#AUTHOR Nobody
#TIMERES 100
#SHIFT 0

@0:00:01.00 0:00:03.00 D hello world
@0:00:04.50 0:00:05.50 D \\Bbold\\b then \\Iitalic\\i
@0:00:06.00 0:00:08.00 D \\Uunderline\\u here
@0:00:10.00 0:00:12.00 D line one\\nline two
@0:00:13.00 0:00:14.00 D trailing cue
";

#[test]
fn five_cues() {
    let t = jacosub::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 5);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_000_000);
    assert_eq!(t.cues[1].start_us, 4_500_000);
    assert_eq!(t.cues[4].end_us, 14_000_000);
}

#[test]
fn parses_bold_italic_underline_and_line_break() {
    let t = jacosub::parse(SAMPLE.as_bytes()).unwrap();
    let mut saw_bold = false;
    let mut saw_italic = false;
    let mut saw_u = false;
    let mut saw_lb = false;
    for cue in &t.cues {
        visit(&cue.segments, &mut |s| match s {
            Segment::Bold(_) => saw_bold = true,
            Segment::Italic(_) => saw_italic = true,
            Segment::Underline(_) => saw_u = true,
            Segment::LineBreak => saw_lb = true,
            _ => {}
        });
    }
    assert!(saw_bold, "expected bold");
    assert!(saw_italic, "expected italic");
    assert!(saw_u, "expected underline");
    assert!(saw_lb, "expected line-break");
}

#[test]
fn write_roundtrips() {
    let t = jacosub::parse(SAMPLE.as_bytes()).unwrap();
    let out = jacosub::write(&t).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(
        s.contains("@0:00:01.00 0:00:03.00 D hello world"),
        "got: {s}"
    );
    // Headers come from the captured extradata.
    assert!(s.contains("#TITLE Demo show"));
    // Reparse yields the same cue count + timings.
    let t2 = jacosub::parse(s.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), t.cues.len());
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
    }
}

#[test]
fn probe_positive() {
    assert!(jacosub::probe(SAMPLE.as_bytes()) > 0);
}

#[test]
fn probe_rejects_webvtt() {
    assert_eq!(
        jacosub::probe(b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"),
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
