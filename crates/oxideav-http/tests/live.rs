//! Live HTTP tests against a public host.
//!
//! Off by default — set `OXIDEAV_LIVE_HTTP_TESTS=1` to enable.
//! Skipped (with a printed note) when off so default `cargo test` does
//! not depend on the network.

use std::io::{Read, Seek, SeekFrom};

use oxideav_http::HttpSource;

fn live_enabled() -> bool {
    std::env::var("OXIDEAV_LIVE_HTTP_TESTS")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

const URL: &str =
    "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov";

#[test]
fn head_reports_length_and_range_support() {
    if !live_enabled() {
        eprintln!("skip: set OXIDEAV_LIVE_HTTP_TESTS=1 to enable");
        return;
    }
    let src = HttpSource::open(URL).expect("HEAD ok");
    let len = src.len();
    assert!(len > 1_000_000, "expected >1 MiB asset, got {len}");
}

#[test]
fn read_first_kb_then_seek_and_compare() {
    if !live_enabled() {
        eprintln!("skip: set OXIDEAV_LIVE_HTTP_TESTS=1 to enable");
        return;
    }
    let mut src = HttpSource::open(URL).expect("HEAD ok");

    // Pull the first 1 KiB.
    let mut a = vec![0u8; 1024];
    src.read_exact(&mut a).expect("read first 1k");
    // The asset is a MOV/MP4 — first 4 bytes after a leading size are
    // typically `ftyp`.
    let has_ftyp = a.windows(4).any(|w| w == b"ftyp");
    assert!(has_ftyp, "no ftyp in first 1 KiB — wrong asset?");

    // Re-seek to byte 0 and read the same 1 KiB; bytes must match.
    src.seek(SeekFrom::Start(0)).unwrap();
    let mut b = vec![0u8; 1024];
    src.read_exact(&mut b).expect("re-read first 1k");
    assert_eq!(a, b, "Range re-fetch returned different bytes");

    // Jump to the middle and grab 4 KiB; just assert we got something.
    let mid = src.len() / 2;
    src.seek(SeekFrom::Start(mid)).unwrap();
    let mut c = vec![0u8; 4096];
    src.read_exact(&mut c).expect("read mid 4k");
}

#[test]
fn registry_dispatch_via_https_scheme() {
    if !live_enabled() {
        eprintln!("skip: set OXIDEAV_LIVE_HTTP_TESTS=1 to enable");
        return;
    }
    let mut reg = oxideav_source::SourceRegistry::with_defaults();
    oxideav_http::register(&mut reg);
    let mut s = reg.open(URL).expect("registry open");
    let mut head = [0u8; 32];
    s.read_exact(&mut head).expect("read");
    assert!(head.iter().any(|&b| b != 0));
}
