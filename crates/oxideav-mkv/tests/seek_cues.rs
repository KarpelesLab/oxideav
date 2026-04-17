//! Integration tests for `Demuxer::seek_to` on the Matroska demuxer.
//!
//! These craft a minimal MKV by hand (EBML header + Segment + Info +
//! Tracks + Cues + 3 Clusters) so the Cues element's `CueClusterPosition`
//! offsets are known exactly. We then seek to various timestamps and
//! assert:
//!
//! * `landed_pts <= target_pts`  (never overshoots the request)
//! * the first packet read after the seek carries `pts >= landed_pts`
//!   (we landed on the cluster the cue pointed at, not an earlier one)
//! * a missing-Cues file returns `Error::Unsupported`.
//!
//! The file uses a 1 ms timecode scale so Matroska ticks and stream-side
//! pts are numerically identical — makes the assertions easy to read.
//!
//! Matroska element IDs reference: <https://www.matroska.org/technical/elements.html>

use std::io::Cursor;

use oxideav_container::ReadSeek;
use oxideav_core::Error;

// Re-export the EBML primitives we need — the test drops down below the
// muxer API and writes raw elements because we want to control the Cues
// byte-layout precisely.
use oxideav_mkv::ebml::{write_element_id, write_vint};
use oxideav_mkv::ids;

/// Helper: encode a uint-valued EBML element.
fn elem_uint(id: u32, value: u64) -> Vec<u8> {
    // Minimum-width payload (1..=8 bytes).
    let n = if value == 0 {
        1
    } else {
        (64 - value.leading_zeros()).div_ceil(8) as usize
    };
    let mut out = Vec::new();
    out.extend_from_slice(&write_element_id(id));
    out.extend_from_slice(&write_vint(n as u64, 0));
    for i in (0..n).rev() {
        out.push(((value >> (i * 8)) & 0xFF) as u8);
    }
    out
}

fn elem_str(id: u32, s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&write_element_id(id));
    out.extend_from_slice(&write_vint(s.len() as u64, 0));
    out.extend_from_slice(s.as_bytes());
    out
}

fn elem_float_be_f64(id: u32, value: f64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&write_element_id(id));
    out.extend_from_slice(&write_vint(8, 0));
    out.extend_from_slice(&value.to_be_bytes());
    out
}

fn elem_master(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&write_element_id(id));
    out.extend_from_slice(&write_vint(body.len() as u64, 0));
    out.extend_from_slice(body);
    out
}

/// Build a SimpleBlock element carrying one raw payload byte. The `tc`
/// offset is relative to the enclosing Cluster's Timecode, signed i16.
/// Track number is a 1-byte VINT (tracks 1..=126 fit).
fn simple_block(track: u8, tc_offset: i16, keyframe: bool, payload: u8) -> Vec<u8> {
    // Body: track (vint, 1 byte for 1..=126) + i16 BE + flags + payload.
    let mut body = Vec::new();
    body.extend_from_slice(&write_vint(track as u64, 0));
    body.extend_from_slice(&tc_offset.to_be_bytes());
    body.push(if keyframe { 0x80 } else { 0x00 });
    body.push(payload);
    let mut out = Vec::new();
    out.extend_from_slice(&write_element_id(ids::SIMPLE_BLOCK));
    out.extend_from_slice(&write_vint(body.len() as u64, 0));
    out.extend_from_slice(&body);
    out
}

/// Build a minimal self-contained MKV:
///   EBML header
///   Segment (KNOWN size — critical for the test's offset math)
///     Info (TimecodeScale=1ms, Duration=3000)
///     Tracks (one PCM track, TrackNumber=1)
///     Cues (3 CuePoints at 0, 1000, 2000)
///     Cluster (timecode=0)       ← 1 block @ pts 0
///     Cluster (timecode=1000)    ← 1 block @ pts 1000
///     Cluster (timecode=2000)    ← 1 block @ pts 2000
///
/// Returns the bytes + the (start, offset) pair for each cluster, with
/// offsets measured relative to the Segment payload start (what Cues
/// `CueClusterPosition` is specified to store).
fn build_mkv_with_cues(include_cues: bool) -> (Vec<u8>, Vec<u64>) {
    // --- EBML header ---
    let mut ebml_body = Vec::new();
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_VERSION, 1));
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_READ_VERSION, 1));
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_MAX_ID_LENGTH, 4));
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_MAX_SIZE_LENGTH, 8));
    ebml_body.extend_from_slice(&elem_str(ids::EBML_DOC_TYPE, "matroska"));
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_DOC_TYPE_VERSION, 4));
    ebml_body.extend_from_slice(&elem_uint(ids::EBML_DOC_TYPE_READ_VERSION, 2));
    let ebml_header = elem_master(ids::EBML_HEADER, &ebml_body);

    // --- Info ---
    let mut info_body = Vec::new();
    info_body.extend_from_slice(&elem_uint(ids::TIMECODE_SCALE, 1_000_000));
    info_body.extend_from_slice(&elem_float_be_f64(ids::DURATION, 3000.0));
    info_body.extend_from_slice(&elem_str(ids::MUXING_APP, "oxideav-test"));
    info_body.extend_from_slice(&elem_str(ids::WRITING_APP, "oxideav-test"));
    let info = elem_master(ids::INFO, &info_body);

    // --- Tracks: one PCM S16LE stereo 48k track, TrackNumber=1 ---
    let mut track_body = Vec::new();
    track_body.extend_from_slice(&elem_uint(ids::TRACK_NUMBER, 1));
    track_body.extend_from_slice(&elem_uint(ids::TRACK_UID, 1));
    track_body.extend_from_slice(&elem_uint(ids::TRACK_TYPE, ids::TRACK_TYPE_AUDIO));
    track_body.extend_from_slice(&elem_str(ids::CODEC_ID, "A_PCM/INT/LIT"));
    let mut audio_body = Vec::new();
    audio_body.extend_from_slice(&elem_float_be_f64(ids::SAMPLING_FREQUENCY, 48_000.0));
    audio_body.extend_from_slice(&elem_uint(ids::CHANNELS, 2));
    audio_body.extend_from_slice(&elem_uint(ids::BIT_DEPTH, 16));
    track_body.extend_from_slice(&elem_master(ids::AUDIO, &audio_body));
    let track_entry = elem_master(ids::TRACK_ENTRY, &track_body);
    let tracks = elem_master(ids::TRACKS, &track_entry);

    // --- Clusters: we need their byte sizes to compute the Cues offsets.
    // Build them as concrete bytes first and only *later* wrap them in a
    // Segment master element.
    let mut cluster_a_body = Vec::new();
    cluster_a_body.extend_from_slice(&elem_uint(ids::TIMECODE, 0));
    cluster_a_body.extend_from_slice(&simple_block(1, 0, true, 0xAA));
    let cluster_a = elem_master(ids::CLUSTER, &cluster_a_body);

    let mut cluster_b_body = Vec::new();
    cluster_b_body.extend_from_slice(&elem_uint(ids::TIMECODE, 1000));
    cluster_b_body.extend_from_slice(&simple_block(1, 0, true, 0xBB));
    let cluster_b = elem_master(ids::CLUSTER, &cluster_b_body);

    let mut cluster_c_body = Vec::new();
    cluster_c_body.extend_from_slice(&elem_uint(ids::TIMECODE, 2000));
    cluster_c_body.extend_from_slice(&simple_block(1, 0, true, 0xCC));
    let cluster_c = elem_master(ids::CLUSTER, &cluster_c_body);

    // Cluster offsets are measured from the Segment *payload* start. We
    // build Cues (if requested) to sit *before* the clusters in the
    // Segment body, so we must size-and-offset Cues first.
    //
    // Strategy: tentatively build Cues with placeholder offsets, then
    // compute the Segment layout and *rewrite* the Cues with correct
    // offsets. Cluster positions depend on the Cues element's length, so
    // this is inherently circular — but Cues encodes offsets as VINT
    // uints, and we only need to ensure the VINT width doesn't change
    // when we rewrite. We avoid the churn by using explicit 8-byte uints
    // for the CueClusterPosition field.

    // Helper: encode a uint forced to exactly 8 bytes.
    fn u64_fixed8(id: u32, value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&write_element_id(id));
        out.extend_from_slice(&write_vint(8, 0));
        out.extend_from_slice(&value.to_be_bytes());
        out
    }

    let build_cues = |off_a: u64, off_b: u64, off_c: u64| -> Vec<u8> {
        let mk_cue = |time: u64, offset: u64| -> Vec<u8> {
            let mut ctp_body = Vec::new();
            ctp_body.extend_from_slice(&elem_uint(ids::CUE_TRACK, 1));
            ctp_body.extend_from_slice(&u64_fixed8(ids::CUE_CLUSTER_POSITION, offset));
            let ctp = elem_master(ids::CUE_TRACK_POSITIONS, &ctp_body);
            let mut cp_body = Vec::new();
            cp_body.extend_from_slice(&elem_uint(ids::CUE_TIME, time));
            cp_body.extend_from_slice(&ctp);
            elem_master(ids::CUE_POINT, &cp_body)
        };
        let mut body = Vec::new();
        body.extend_from_slice(&mk_cue(0, off_a));
        body.extend_from_slice(&mk_cue(1000, off_b));
        body.extend_from_slice(&mk_cue(2000, off_c));
        elem_master(ids::CUES, &body)
    };

    // Build Cues with placeholder offsets (the byte length is stable
    // because we encoded CueClusterPosition at fixed 8 bytes).
    let cues_bytes = if include_cues {
        build_cues(0, 0, 0)
    } else {
        Vec::new()
    };

    // Now compute the real cluster offsets (relative to Segment payload
    // start). Segment body layout: Info ++ Tracks ++ Cues ++ A ++ B ++ C.
    let off_info = 0u64;
    let off_tracks = off_info + info.len() as u64;
    let off_cues = off_tracks + tracks.len() as u64;
    let off_a = off_cues + cues_bytes.len() as u64;
    let off_b = off_a + cluster_a.len() as u64;
    let off_c = off_b + cluster_b.len() as u64;

    // Rewrite Cues with real offsets.
    let cues_bytes = if include_cues {
        build_cues(off_a, off_b, off_c)
    } else {
        Vec::new()
    };

    // Assemble Segment body.
    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cues_bytes);
    seg_body.extend_from_slice(&cluster_a);
    seg_body.extend_from_slice(&cluster_b);
    seg_body.extend_from_slice(&cluster_c);
    let segment = elem_master(ids::SEGMENT, &seg_body);

    // Final file: EBML header ++ Segment.
    let mut out = Vec::new();
    out.extend_from_slice(&ebml_header);
    out.extend_from_slice(&segment);

    (out, vec![off_a, off_b, off_c])
}

#[test]
fn seek_to_lands_on_exact_cue() {
    let (bytes, _offsets) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");
    assert_eq!(dmx.streams().len(), 1);

    // Seek to exactly the middle cue (pts=1000). The timebase is 1ms per
    // tick (TimecodeScale=1e6 ns → time_base = 1e6/1e9 = 1/1000 s), so
    // Matroska ticks and stream-pts are numerically equal in this file.
    let landed = dmx.seek_to(0, 1000).expect("seek_to mid");
    assert_eq!(landed, 1000, "should land exactly on cue @ t=1000");

    let pkt = dmx.next_packet().expect("packet after seek");
    assert_eq!(
        pkt.pts,
        Some(1000),
        "first packet after seek should be cluster B (pts=1000)"
    );
    assert_eq!(pkt.data, vec![0xBB], "payload sanity check");
}

#[test]
fn seek_to_between_cues_picks_earlier() {
    let (bytes, _) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    // Request pts=1500 — no exact cue, should land on the cue @ 1000.
    let landed = dmx.seek_to(0, 1500).expect("seek_to between");
    assert!(
        landed <= 1500,
        "landed ({landed}) must not overshoot target (1500)"
    );
    assert_eq!(landed, 1000, "should snap back to the nearest earlier cue");

    let pkt = dmx.next_packet().expect("packet after seek");
    assert!(
        pkt.pts.unwrap() >= landed,
        "first packet pts ({:?}) should be >= landed ({landed})",
        pkt.pts
    );
}

#[test]
fn seek_to_past_end_lands_on_last_cue() {
    let (bytes, _) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    // Way past the end — should land on the last cue (2000).
    let landed = dmx.seek_to(0, 100_000).expect("seek past end");
    assert_eq!(landed, 2000);
    let pkt = dmx.next_packet().expect("packet after seek past end");
    assert_eq!(pkt.pts, Some(2000));
    assert_eq!(pkt.data, vec![0xCC]);
}

#[test]
fn seek_to_before_start_uses_first_cue() {
    let (bytes, _) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    // Negative / zero target — the first cue is at t=0 so we always
    // have a valid landing point.
    let landed = dmx.seek_to(0, -10).expect("seek before start");
    assert_eq!(landed, 0, "should land on first cue (t=0)");
    let pkt = dmx.next_packet().expect("packet after seek-to-start");
    assert_eq!(pkt.pts, Some(0));
    assert_eq!(pkt.data, vec![0xAA]);
}

#[test]
fn seek_to_without_cues_is_unsupported() {
    let (bytes, _) = build_mkv_with_cues(false);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    match dmx.seek_to(0, 1000) {
        Err(Error::Unsupported(_)) => {}
        other => panic!("expected Error::Unsupported without Cues, got {other:?}"),
    }
}

#[test]
fn seek_to_invalid_stream_is_invalid() {
    let (bytes, _) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    match dmx.seek_to(99, 0) {
        Err(Error::InvalidData(_)) => {}
        other => panic!("expected Error::InvalidData for bad stream index, got {other:?}"),
    }
}

#[test]
fn seek_to_resets_pending_packets() {
    // Regression: if the demuxer had packets in its out-queue from a
    // prior next_packet() call, seek_to must discard them so the first
    // post-seek packet is from the cue target, not from the stale queue.
    let (bytes, _) = build_mkv_with_cues(true);
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux open");

    // Consume the first cluster's packet to populate demuxer state.
    let first = dmx.next_packet().expect("first");
    assert_eq!(first.pts, Some(0));

    // Seek forward and verify we don't see stale packets.
    let landed = dmx.seek_to(0, 2000).expect("seek");
    assert_eq!(landed, 2000);
    let after = dmx.next_packet().expect("after seek");
    assert_eq!(after.pts, Some(2000));
    assert_eq!(after.data, vec![0xCC]);
}
