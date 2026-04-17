//! Integration test for Ogg demuxer bisection seek.
//!
//! Builds a synthetic Vorbis-in-Ogg stream with 20 data pages of
//! monotonically increasing granules, then seeks to a mid-granule and
//! validates the landed page granule is <= target and the next data page's
//! granule is > landed.

use std::io::Cursor;

use oxideav_container::ReadSeek;
use oxideav_ogg::page::{self, flags, lace, Page};

/// Minimal valid Vorbis identification packet (30 bytes).
fn vorbis_id_packet() -> Vec<u8> {
    let mut p = Vec::with_capacity(30);
    p.push(0x01); // packet type
    p.extend_from_slice(b"vorbis");
    p.extend_from_slice(&0u32.to_le_bytes()); // version
    p.push(2); // channels
    p.extend_from_slice(&48000u32.to_le_bytes()); // sample rate
    p.extend_from_slice(&0i32.to_le_bytes()); // br_max
    p.extend_from_slice(&128000i32.to_le_bytes()); // br_nom
    p.extend_from_slice(&0i32.to_le_bytes()); // br_min
    p.push(0xB8); // blocksize_0 | blocksize_1 (packed nibbles) -- nominal
    p.push(0x01); // framing bit
    assert_eq!(p.len(), 30);
    p
}

/// Minimal valid Vorbis comment packet: 0x03 "vorbis" + empty vendor +
/// zero user comments + framing bit.
fn vorbis_comment_packet() -> Vec<u8> {
    let mut p = Vec::new();
    p.push(0x03);
    p.extend_from_slice(b"vorbis");
    p.extend_from_slice(&0u32.to_le_bytes()); // vendor string length
    p.extend_from_slice(&0u32.to_le_bytes()); // user comment count
    p.push(0x01); // framing bit
    p
}

/// Stub Vorbis setup packet. The demuxer doesn't parse the setup body —
/// it only needs the 0x05 + "vorbis" signature to count this as the third
/// header packet.
fn vorbis_setup_packet() -> Vec<u8> {
    let mut p = Vec::new();
    p.push(0x05);
    p.extend_from_slice(b"vorbis");
    // 16 bytes of placeholder setup data; contents are irrelevant to this test.
    p.extend_from_slice(&[0u8; 16]);
    p
}

/// Build a single Ogg page carrying one packet.
fn build_page(flags_byte: u8, granule: i64, serial: u32, seq: u32, packet: &[u8]) -> Vec<u8> {
    let lacing = lace(packet.len());
    let page = Page {
        flags: flags_byte,
        granule_position: granule,
        serial,
        seq_no: seq,
        lacing,
        data: packet.to_vec(),
    };
    page.to_bytes()
}

/// Build a synthetic Vorbis-in-Ogg blob with 20 data pages whose granules
/// are 960, 1920, 2880, ... , 19200 (one 20ms audio frame at 48 kHz apiece).
fn build_synthetic_ogg() -> Vec<u8> {
    let serial: u32 = 0xBADC_0FFE;
    let mut out = Vec::new();
    let mut seq = 0u32;

    // BOS page: Vorbis identification packet, FIRST_PAGE flag, granule 0.
    out.extend(build_page(
        flags::FIRST_PAGE,
        0,
        serial,
        seq,
        &vorbis_id_packet(),
    ));
    seq += 1;

    // Comment packet page.
    out.extend(build_page(0, 0, serial, seq, &vorbis_comment_packet()));
    seq += 1;

    // Setup packet page.
    out.extend(build_page(0, 0, serial, seq, &vorbis_setup_packet()));
    seq += 1;

    // 20 data pages. Each carries a tiny dummy "packet" (2 bytes) and has
    // a monotonically increasing granule at 48 kHz.
    for i in 1..=20i64 {
        let granule = 960 * i; // 20 ms at 48 kHz per step
        let is_last = i == 20;
        let flag = if is_last { flags::LAST_PAGE } else { 0 };
        let dummy_packet: [u8; 2] = [0xAA, (i as u8).wrapping_add(1)];
        out.extend(build_page(flag, granule, serial, seq, &dummy_packet));
        seq += 1;
    }

    // Sanity: capture pattern must appear exactly 3 + 20 = 23 times.
    let count = out.windows(4).filter(|w| *w == b"OggS").count();
    assert_eq!(count, 23);
    out
}

#[test]
fn seek_to_bisects_to_page_with_granule_le_target() {
    let blob = build_synthetic_ogg();
    let reader: Box<dyn ReadSeek> = Box::new(Cursor::new(blob.clone()));
    let mut demux = oxideav_ogg::demux::open(reader).expect("open synthetic ogg");
    let streams = demux.streams();
    assert_eq!(streams.len(), 1, "synthetic file has one stream");
    assert_eq!(streams[0].params.codec_id.as_str(), "vorbis");

    // Target a mid-stream granule that falls exactly on page 10 (granule 9600).
    // Seeking to 9600 should land on granule 9600 or an earlier page.
    let target: i64 = 9600;
    let landed = demux.seek_to(0, target).expect("seek_to ok");
    assert!(
        landed <= target,
        "landed granule {landed} must be <= target {target}"
    );

    // After seek, the next packet drained should have pts <= target ...
    let pkt = demux.next_packet().expect("next_packet after seek");
    if let Some(pts) = pkt.pts {
        assert!(
            pts <= target || pts == landed,
            "first delivered packet pts {pts} should be <= target {target}"
        );
    }

    // ... and the subsequent packet's pts should be strictly > landed,
    // confirming we didn't rewind past the boundary.
    let mut saw_higher = false;
    for _ in 0..5 {
        match demux.next_packet() {
            Ok(p) => {
                if let Some(pts) = p.pts {
                    if pts > landed {
                        saw_higher = true;
                        break;
                    }
                }
            }
            Err(_) => break,
        }
    }
    assert!(
        saw_higher,
        "no packet with pts > landed ({landed}) found — seek did not advance"
    );
}

#[test]
fn seek_to_granule_at_midpoint_is_close_to_target() {
    let blob = build_synthetic_ogg();
    let reader: Box<dyn ReadSeek> = Box::new(Cursor::new(blob));
    let mut demux = oxideav_ogg::demux::open(reader).unwrap();

    // Data pages have granules 960, 1920, ... 19200 (step 960). A target
    // between two pages should land on the lower one; the delta must be
    // within one page's granule step.
    let target: i64 = 12345;
    let landed = demux.seek_to(0, target).unwrap();
    assert!(landed <= target);
    assert!(
        target - landed < 2 * 960,
        "landed granule {landed} is more than one step below target {target}"
    );
}

#[test]
fn seek_to_unknown_stream_is_out_of_range() {
    let blob = build_synthetic_ogg();
    let reader: Box<dyn ReadSeek> = Box::new(Cursor::new(blob));
    let mut demux = oxideav_ogg::demux::open(reader).unwrap();

    let err = demux.seek_to(99, 0).unwrap_err();
    match err {
        oxideav_core::Error::InvalidData(_) => {}
        other => panic!("expected InvalidData, got {other:?}"),
    }
}

// Silence unused-import warnings for `page::CAPTURE_PATTERN` if the
// synthetic builder ever stops referencing the `page` module directly.
#[allow(dead_code)]
const _SANITY: [u8; 4] = page::CAPTURE_PATTERN;
