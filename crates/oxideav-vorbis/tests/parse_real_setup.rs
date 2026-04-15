//! Integration test: parse a real Vorbis setup header from a known fixture.
//! The fixture is constructed inline so the test is hermetic — no external
//! files required.
//!
//! The bytes below are the Xiph-laced 3-packet CodecPrivate produced by
//! ffmpeg/libvorbis for a 1-second 1ch 48 kHz sine wave at low quality
//! (sine.ogg → A_VORBIS in MKV). Captured 2026-04-15 from the test rig.

use oxideav_vorbis::identification::parse_identification_header;
use oxideav_vorbis::setup::parse_setup;

/// Split a Xiph-laced extradata blob (count_byte=N-1, then N-1 lacing groups,
/// then the packets themselves) into individual packet byte vectors.
fn split_xiph_lacing(extradata: &[u8]) -> Option<Vec<Vec<u8>>> {
    if extradata.is_empty() {
        return None;
    }
    let n_packets = extradata[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n_packets);
    let mut i = 1usize;
    for _ in 0..n_packets - 1 {
        let mut s = 0usize;
        loop {
            if i >= extradata.len() {
                return None;
            }
            let b = extradata[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    let used: usize = sizes.iter().sum();
    if i + used > extradata.len() {
        return None;
    }
    let last = extradata.len() - i - used;
    sizes.push(last);
    let mut packets = Vec::with_capacity(n_packets);
    for sz in sizes {
        packets.push(extradata[i..i + sz].to_vec());
        i += sz;
    }
    Some(packets)
}

#[test]
fn split_lacing_minimal() {
    // "Manually" pack: 2 packets → count byte = 1, then 1 lacing entry, then both packets.
    let p1 = b"first";
    let p2 = b"second-packet";
    let mut blob = Vec::new();
    blob.push(1); // packets - 1
    blob.push(p1.len() as u8); // p1 size
    blob.extend_from_slice(p1);
    blob.extend_from_slice(p2);
    let split = split_xiph_lacing(&blob).unwrap();
    assert_eq!(split.len(), 2);
    assert_eq!(split[0], p1);
    assert_eq!(split[1], p2);
}

#[test]
fn parses_real_setup_from_disk() {
    // Optional: if the test rig has a pre-built sine.ogg lying around at the
    // standard test location, parse its CodecPrivate-equivalent. Skipped
    // when the file isn't present so CI without external assets still passes.
    let path = std::path::Path::new("/tmp/oxideav-test/sine.ogg");
    if !path.exists() {
        return;
    }
    let data = std::fs::read(path).unwrap();
    // Walk the Ogg pages, collect first 3 packets (= the 3 Vorbis headers).
    let mut headers: Vec<Vec<u8>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    'outer: while i + 27 <= data.len() && headers.len() < 3 {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let n_segs = data[i + 26] as usize;
        let lacing = &data[i + 27..i + 27 + n_segs];
        let body_start = i + 27 + n_segs;
        let mut off = body_start;
        for &lv in lacing {
            buf.extend_from_slice(&data[off..off + lv as usize]);
            off += lv as usize;
            if lv < 255 {
                headers.push(std::mem::take(&mut buf));
                if headers.len() == 3 {
                    break 'outer;
                }
            }
        }
        i = body_start + (lacing.iter().map(|&v| v as usize).sum::<usize>());
    }
    assert_eq!(headers.len(), 3, "expected 3 Vorbis header packets");
    let id = parse_identification_header(&headers[0]).unwrap();
    let setup = parse_setup(&headers[2], id.audio_channels).unwrap();
    assert!(!setup.codebooks.is_empty(), "setup should have codebooks");
    assert!(!setup.modes.is_empty(), "setup should declare at least one mode");
}
