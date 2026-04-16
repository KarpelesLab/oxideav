//! BufferedSource correctness — exercise the prefetch ring without
//! depending on an external source.

use std::io::{Cursor, Read, Seek, SeekFrom};

use oxideav_source::BufferedSource;

fn ramp(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i & 0xff) as u8).collect()
}

#[test]
fn sequential_read_matches_inner() {
    let data = ramp(2 * 1024 * 1024);
    let inner = Box::new(Cursor::new(data.clone()));
    let mut buf = BufferedSource::new(inner, 1024 * 1024).unwrap();
    let mut out = vec![0u8; data.len()];
    buf.read_exact(&mut out).unwrap();
    assert_eq!(out, data);
}

#[test]
fn read_at_eof_returns_zero() {
    let data = ramp(4096);
    let inner = Box::new(Cursor::new(data));
    let mut buf = BufferedSource::new(inner, 0).unwrap();
    let mut out = vec![0u8; 4096];
    buf.read_exact(&mut out).unwrap();
    let mut tail = [0u8; 16];
    assert_eq!(buf.read(&mut tail).unwrap(), 0);
}

#[test]
fn seek_within_window_does_not_block_or_lose_bytes() {
    let data = ramp(4 * 1024 * 1024);
    let inner = Box::new(Cursor::new(data.clone()));
    let mut buf = BufferedSource::new(inner, 2 * 1024 * 1024).unwrap();
    // Read 64 KiB, seek back 32 KiB, read again — should match.
    let mut a = vec![0u8; 64 * 1024];
    buf.read_exact(&mut a).unwrap();
    buf.seek(SeekFrom::Current(-32 * 1024)).unwrap();
    let mut b = vec![0u8; 32 * 1024];
    buf.read_exact(&mut b).unwrap();
    assert_eq!(b[..], data[32 * 1024..64 * 1024]);
}

#[test]
fn seek_outside_window_restarts_prefetch() {
    let data = ramp(4 * 1024 * 1024);
    let inner = Box::new(Cursor::new(data.clone()));
    let mut buf = BufferedSource::new(inner, 256 * 1024).unwrap();
    // Read 8 KiB, then jump to 3 MiB and read a chunk.
    let mut a = vec![0u8; 8 * 1024];
    buf.read_exact(&mut a).unwrap();
    let target: u64 = 3 * 1024 * 1024;
    buf.seek(SeekFrom::Start(target)).unwrap();
    let mut b = vec![0u8; 16 * 1024];
    buf.read_exact(&mut b).unwrap();
    assert_eq!(b[..], data[target as usize..(target as usize + 16 * 1024)]);
}

#[test]
fn seek_to_end_then_read_returns_zero() {
    use std::time::{Duration, Instant};
    let data = ramp(64 * 1024);
    let inner = Box::new(Cursor::new(data));
    let mut buf = BufferedSource::new(inner, 0).unwrap();
    let end = buf.seek(SeekFrom::End(0)).unwrap();
    assert_eq!(end, 64 * 1024);
    let mut out = [0u8; 8];
    // Must not block on the worker — at EOF the read should return
    // immediately, not wait the prefetch timeout.
    let t0 = Instant::now();
    assert_eq!(buf.read(&mut out).unwrap(), 0);
    assert!(t0.elapsed() < Duration::from_secs(2));
}

#[test]
fn drop_terminates_worker_promptly() {
    use std::time::{Duration, Instant};
    let data = ramp(8 * 1024 * 1024);
    let inner = Box::new(Cursor::new(data));
    let buf = BufferedSource::new(inner, 4 * 1024 * 1024).unwrap();
    // Drop the buffer; the worker should exit and join() should return
    // well under a second.
    let t0 = Instant::now();
    drop(buf);
    assert!(t0.elapsed() < Duration::from_secs(1));
}

#[test]
fn len_reports_total() {
    let data = ramp(12345);
    let inner = Box::new(Cursor::new(data));
    let buf = BufferedSource::new(inner, 0).unwrap();
    assert_eq!(buf.len(), Some(12345));
}
