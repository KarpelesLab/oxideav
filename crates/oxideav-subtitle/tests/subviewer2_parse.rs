//! SubViewer 2.0 parsing, writing, probe.

use oxideav_core::Segment;
use oxideav_subtitle::subviewer2;

const SAMPLE: &str = "[INFORMATION]
[TITLE]
Example Show
[AUTHOR]
Anonymous
[SOURCE]
Test
[END INFORMATION]
[SUBTITLE]

00:00:01.00,00:00:03.50
Hello|World

00:00:04.00,00:00:06.00
Second cue|line two

00:00:07.25,00:00:08.00
Short one

00:00:09.00,00:00:11.00
multi
line
body

00:00:12.00,00:00:13.50
Last
";

#[test]
fn five_cues() {
    let t = subviewer2::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 5);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_500_000);
    assert_eq!(t.cues[2].start_us, 7_250_000);
    assert_eq!(t.cues[4].end_us, 13_500_000);
}

#[test]
fn parses_metadata() {
    let t = subviewer2::parse(SAMPLE.as_bytes()).unwrap();
    assert!(t
        .metadata
        .iter()
        .any(|(k, v)| k == "title" && v == "Example Show"));
    assert!(t
        .metadata
        .iter()
        .any(|(k, v)| k == "author" && v == "Anonymous"));
    assert!(t.metadata.iter().any(|(k, v)| k == "source" && v == "Test"));
}

#[test]
fn pipe_is_line_break() {
    let t = subviewer2::parse(SAMPLE.as_bytes()).unwrap();
    let mut lb = 0;
    for s in &t.cues[0].segments {
        if matches!(s, Segment::LineBreak) {
            lb += 1;
        }
    }
    assert_eq!(lb, 1);
}

#[test]
fn write_roundtrips() {
    let t = subviewer2::parse(SAMPLE.as_bytes()).unwrap();
    let out = subviewer2::write(&t).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("[INFORMATION]"));
    assert!(s.contains("[END INFORMATION]"));
    assert!(s.contains("[SUBTITLE]"));
    assert!(s.contains("00:00:01.00,00:00:03.50"));
    assert!(s.contains("Hello|World"));
    let t2 = subviewer2::parse(s.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), t.cues.len());
    for (a, b) in t.cues.iter().zip(t2.cues.iter()) {
        assert_eq!(a.start_us, b.start_us);
        assert_eq!(a.end_us, b.end_us);
    }
}

#[test]
fn probe_positive() {
    assert!(subviewer2::probe(SAMPLE.as_bytes()) > 0);
}

#[test]
fn probe_rejects_webvtt() {
    assert_eq!(subviewer2::probe(b"WEBVTT\n"), 0);
}
