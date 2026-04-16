//! File-driver round-trip tests.

use std::io::{Read, Seek, SeekFrom};

use oxideav_source::SourceRegistry;

#[test]
fn open_bare_path_reads_first_bytes() {
    let reg = SourceRegistry::with_defaults();
    let mut f = reg.open("Cargo.toml").expect("open");
    let mut head = [0u8; 4];
    f.read_exact(&mut head).expect("read");
    assert_eq!(&head, b"[pac"); // start of a Cargo.toml: "[package]"
}

#[test]
fn open_file_url_reads_first_bytes() {
    let reg = SourceRegistry::with_defaults();
    let cwd = std::env::current_dir().unwrap();
    let url = format!("file://{}/Cargo.toml", cwd.display());
    let mut f = reg.open(&url).expect("open");
    let mut head = [0u8; 4];
    f.read_exact(&mut head).expect("read");
    assert_eq!(&head, b"[pac");
}

#[test]
fn open_supports_seek() {
    let reg = SourceRegistry::with_defaults();
    let mut f = reg.open("Cargo.toml").unwrap();
    let end = f.seek(SeekFrom::End(0)).unwrap();
    assert!(end > 0);
    f.seek(SeekFrom::Start(0)).unwrap();
    let pos = f.stream_position().unwrap();
    assert_eq!(pos, 0);
}

#[test]
fn unknown_scheme_with_no_driver_errors() {
    let mut reg = SourceRegistry::new();
    reg.register("file", oxideav_source::open_file);
    let r = reg.open("https://example.com/x");
    // No https driver registered — falls through to file driver, which
    // will then fail to open a file with that path.
    assert!(r.is_err());
}
