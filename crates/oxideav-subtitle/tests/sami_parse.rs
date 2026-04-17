//! SAMI parsing, writing, and round-trip.
//!
//! Targets the public `oxideav_subtitle::sami` module — available once
//! the caller adds `pub mod sami;` to `lib.rs`.

use oxideav_core::Segment;
use oxideav_subtitle::sami;

const SAMPLE: &str = "<SAMI>\n\
<HEAD>\n\
<STYLE TYPE=\"text/css\">\n\
<!--\n\
.ENUSCC { Name: English; lang: en-US; color: yellow; }\n\
.FRCC   { Name: French;  lang: fr-FR; color: red; }\n\
-->\n\
</STYLE>\n\
</HEAD>\n\
<BODY>\n\
<SYNC Start=1000>\n\
<P Class=\"ENUSCC\">Hello <B>world</B></P>\n\
<SYNC Start=3000>\n\
<P Class=\"ENUSCC\">&nbsp;</P>\n\
<SYNC Start=5000>\n\
<P Class=\"FRCC\">Bonjour <I>monde</I></P>\n\
</BODY>\n\
</SAMI>\n";

#[test]
fn two_cues_bracketed_by_clear() {
    let t = sami::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues.len(), 2);
    assert_eq!(t.cues[0].start_us, 1_000_000);
    assert_eq!(t.cues[0].end_us, 3_000_000);
    assert_eq!(t.cues[1].start_us, 5_000_000);
}

#[test]
fn preserves_classes() {
    let t = sami::parse(SAMPLE.as_bytes()).unwrap();
    assert_eq!(t.cues[0].style_ref.as_deref(), Some("ENUSCC"));
    assert_eq!(t.cues[1].style_ref.as_deref(), Some("FRCC"));
    let names: Vec<&str> = t.styles.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"ENUSCC"));
    assert!(names.contains(&"FRCC"));
}

#[test]
fn preserves_inline_formatting() {
    let t = sami::parse(SAMPLE.as_bytes()).unwrap();
    let mut saw_bold = false;
    visit(&t.cues[0].segments, &mut |s| {
        if matches!(s, Segment::Bold(_)) {
            saw_bold = true;
        }
    });
    assert!(saw_bold);
    let mut saw_italic = false;
    visit(&t.cues[1].segments, &mut |s| {
        if matches!(s, Segment::Italic(_)) {
            saw_italic = true;
        }
    });
    assert!(saw_italic);
}

#[test]
fn write_roundtrips_syncs() {
    let t = sami::parse(SAMPLE.as_bytes()).unwrap();
    let out = sami::write(&t);
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("<SAMI>"));
    assert!(s.contains("SYNC Start=1000"));
    assert!(s.contains("SYNC Start=5000"));
    assert!(s.contains("<B>world</B>"));
    assert!(s.contains("<I>monde</I>"));

    // Reparse.
    let t2 = sami::parse(s.as_bytes()).unwrap();
    assert_eq!(t2.cues.len(), 2);
    assert_eq!(t2.cues[0].start_us, 1_000_000);
    assert_eq!(t2.cues[1].start_us, 5_000_000);
}

#[test]
fn probe_positive_and_negative() {
    assert!(sami::probe(SAMPLE.as_bytes()) > 60);
    assert_eq!(sami::probe(b"1\n00:00:01,000 --> 00:00:02,000\n"), 0);
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
