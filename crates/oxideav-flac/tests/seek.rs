//! FLAC demuxer `seek_to` smoke test.
//!
//! Builds a synthetic FLAC prefix (`fLaC` + STREAMINFO + SEEKTABLE) followed
//! by three hand-rolled fixed-blocking frames whose headers parse and
//! CRC-8-verify. We seek into the middle of the file via the SEEKTABLE and
//! assert that the next emitted packet carries the expected sample number.

use std::io::Cursor;

use oxideav_container::ContainerRegistry;
use oxideav_flac::crc::crc8;

/// Pack a FLAC metadata-block header: last-flag (1 bit) + type (7 bits) +
/// 24-bit big-endian length.
fn block_header(is_last: bool, block_type: u8, length: usize) -> [u8; 4] {
    let b0 = if is_last { 0x80 } else { 0x00 } | (block_type & 0x7F);
    [
        b0,
        ((length >> 16) & 0xFF) as u8,
        ((length >> 8) & 0xFF) as u8,
        (length & 0xFF) as u8,
    ]
}

/// Build a STREAMINFO block. Fixed block size 192, 48kHz mono 16-bit,
/// total_samples unknown (0).
fn build_streaminfo() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&192u16.to_be_bytes()); // min block size
    block[2..4].copy_from_slice(&192u16.to_be_bytes()); // max block size
                                                        // packed: sr(20)=48000, ch-1(3)=0 (mono), bps-1(5)=15 (16-bit), total(36)=0
    let packed: u64 = (48_000u64 << 44) | (15u64 << 36);
    block[10..18].copy_from_slice(&packed.to_be_bytes());
    block
}

/// Build a single FLAC fixed-blocking frame whose header carries
/// `frame_number`. Returns (frame_bytes, first_sample_of_frame).
///
/// Layout mirrors the parser in `frame.rs`:
/// * bytes 0..2: sync `0xFF 0xF8` (fixed blocking, reserved bits 0)
/// * byte 2:     block_size_code=1 (192 samples) <<4 | sample_rate_code=10 (48000)
/// * byte 3:     channel_code=0 (mono) <<4 | sample_size_code=4 (16bps) <<1 | reserved 0
/// * byte 4:     UTF-8 coded frame number (single byte for small values)
/// * byte 5:     CRC-8 over the previous 5 bytes
/// * 32 bytes of zero body, chosen to contain no 0xFF 0xF8/0xF9 pair.
fn build_frame(frame_number: u8) -> (Vec<u8>, u64) {
    assert!(frame_number < 0x80, "extend UTF-8 encoding for >=128");
    let mut bytes = Vec::with_capacity(6 + 32);
    bytes.push(0xFF);
    bytes.push(0xF8);
    bytes.push((1u8 << 4) | 10u8); // block_size_code=1, sample_rate_code=10
    bytes.push(4u8 << 1); // channel_code=0, sample_size_code=4, reserved=0
    bytes.push(frame_number); // UTF-8 coded number (1 byte for <=0x7F)
    let c = crc8(&bytes);
    bytes.push(c);
    // Frame body: 32 zero bytes. No 0xFF so no false sync.
    bytes.extend_from_slice(&[0u8; 32]);
    (bytes, frame_number as u64 * 192)
}

/// Pack a single SEEKPOINT: sample_number (u64 BE) + offset (u64 BE)
/// + frame_samples (u16 BE).
fn seek_point(sample: u64, offset: u64, frame_samples: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(18);
    out.extend_from_slice(&sample.to_be_bytes());
    out.extend_from_slice(&offset.to_be_bytes());
    out.extend_from_slice(&frame_samples.to_be_bytes());
    out
}

#[test]
fn flac_seek_lands_on_expected_frame() {
    // 1. Build three frames with known first samples.
    let (f0, s0) = build_frame(0); // samples [0, 192)
    let (f1, s1) = build_frame(1); // samples [192, 384)
    let (f2, s2) = build_frame(2); // samples [384, 576)
    assert_eq!(s0, 0);
    assert_eq!(s1, 192);
    assert_eq!(s2, 384);

    let o1 = f0.len() as u64;
    let o2 = (f0.len() + f1.len()) as u64;

    // 2. Build a SEEKTABLE pointing at each frame.
    let mut seektable = Vec::new();
    seektable.extend_from_slice(&seek_point(s0, 0, 192));
    seektable.extend_from_slice(&seek_point(s1, o1, 192));
    seektable.extend_from_slice(&seek_point(s2, o2, 192));
    // Also include a placeholder — the parser should drop it.
    seektable.extend_from_slice(&seek_point(0xFFFF_FFFF_FFFF_FFFF, 0, 0));

    // 3. Assemble the file.
    let streaminfo = build_streaminfo();
    let mut file = Vec::new();
    file.extend_from_slice(b"fLaC");
    file.extend_from_slice(&block_header(false, 0, streaminfo.len()));
    file.extend_from_slice(&streaminfo);
    file.extend_from_slice(&block_header(true, 3, seektable.len()));
    file.extend_from_slice(&seektable);
    file.extend_from_slice(&f0);
    file.extend_from_slice(&f1);
    file.extend_from_slice(&f2);

    // 4. Open through the container registry.
    let mut reg = ContainerRegistry::new();
    oxideav_flac::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let mut demuxer = reg.open_demuxer("flac", cursor).expect("open flac");

    // 5. Seek to target_pts = 300 (inside frame 1 which covers [192, 384)).
    let landed = demuxer.seek_to(0, 300).expect("seek");
    assert_eq!(landed, 192, "expected to land on frame 1's first sample");
    assert!(landed <= 300, "landing pts must be <= target");

    // 6. Next packet should have pts == 192 (frame 1's first sample).
    let pkt = demuxer.next_packet().expect("packet after seek");
    assert_eq!(pkt.pts, Some(192));
    assert_eq!(pkt.duration, Some(192));

    // 7. Following packet should be frame 2 at sample 384 — strictly > target - epsilon.
    let pkt = demuxer.next_packet().expect("second packet after seek");
    assert_eq!(pkt.pts, Some(384));
}

#[test]
fn flac_seek_before_first_seekpoint_lands_on_frame_zero() {
    // Same synthetic file, but we seek to a negative-looking pts. This
    // should clamp to 0 and land on the very first frame.
    let (f0, _) = build_frame(0);
    let (f1, _) = build_frame(1);
    let (f2, _) = build_frame(2);

    // SEEKTABLE that intentionally skips sample 0 — first entry is s1.
    // That way a seek to anything below s1 exercises the "before first
    // seekpoint" fallback path.
    let o1 = f0.len() as u64;
    let mut seektable = Vec::new();
    seektable.extend_from_slice(&seek_point(192, o1, 192));

    let streaminfo = build_streaminfo();
    let mut file = Vec::new();
    file.extend_from_slice(b"fLaC");
    file.extend_from_slice(&block_header(false, 0, streaminfo.len()));
    file.extend_from_slice(&streaminfo);
    file.extend_from_slice(&block_header(true, 3, seektable.len()));
    file.extend_from_slice(&seektable);
    file.extend_from_slice(&f0);
    file.extend_from_slice(&f1);
    file.extend_from_slice(&f2);

    let mut reg = ContainerRegistry::new();
    oxideav_flac::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let mut demuxer = reg.open_demuxer("flac", cursor).expect("open flac");

    let landed = demuxer.seek_to(0, 0).expect("seek");
    assert_eq!(landed, 0);
    let pkt = demuxer.next_packet().expect("first packet");
    assert_eq!(pkt.pts, Some(0));
}

#[test]
fn flac_seek_without_seektable_is_unsupported() {
    let streaminfo = build_streaminfo();
    let mut file = Vec::new();
    file.extend_from_slice(b"fLaC");
    file.extend_from_slice(&block_header(true, 0, streaminfo.len()));
    file.extend_from_slice(&streaminfo);

    let mut reg = ContainerRegistry::new();
    oxideav_flac::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let mut demuxer = reg.open_demuxer("flac", cursor).expect("open flac");

    match demuxer.seek_to(0, 0) {
        Err(oxideav_core::Error::Unsupported(msg)) => {
            assert!(
                msg.contains("SEEKTABLE"),
                "expected SEEKTABLE hint, got {msg:?}"
            );
        }
        other => panic!("expected Unsupported(SEEKTABLE hint), got {other:?}"),
    }
}
