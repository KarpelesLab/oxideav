//! Container probe: PNG magic bytes at offset 0 → score 100. Anything else
//! → 0.

use oxideav_container::ProbeData;

#[test]
fn probe_recognises_png_magic() {
    let buf = b"\x89PNG\r\n\x1a\n_____";
    let p = ProbeData { buf, ext: None };
    assert_eq!(oxideav_png::container::probe(&p), 100);
}

#[test]
fn probe_rejects_non_png() {
    let buf = b"NOPENGMAGIC";
    let p = ProbeData { buf, ext: None };
    assert_eq!(oxideav_png::container::probe(&p), 0);
}

#[test]
fn probe_rejects_short_buffer() {
    let buf = b"\x89PNG";
    let p = ProbeData { buf, ext: None };
    assert_eq!(oxideav_png::container::probe(&p), 0);
}
