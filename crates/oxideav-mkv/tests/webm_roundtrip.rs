//! Integration tests for the WebM muxer mode.
//!
//! These exercise the public surface — `mux::open_webm` and the WebM
//! codec whitelist — by writing a small VP9+Opus stream, round-tripping
//! it through the same-crate demuxer, and verifying the on-disk
//! `DocType` matches. A second test checks the whitelist rejects
//! non-WebM codecs up front.

use std::io::{Cursor, Read, SeekFrom};

use oxideav_container::{ReadSeek, WriteSeek};
use oxideav_core::{CodecId, CodecParameters, Error, MediaType, Packet, StreamInfo, TimeBase};

/// Build a minimal OpusHead extradata blob (RFC 7845 §5.1) for a stereo
/// 48 kHz stream.
fn opus_head(channels: u8, sample_rate: u32, pre_skip: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(b"OpusHead"); // magic
    out.push(1); // version
    out.push(channels);
    out.extend_from_slice(&pre_skip.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&0i16.to_le_bytes()); // output gain
    out.push(0); // mapping family
    out
}

/// A valid-enough Opus packet (1 frame, code 0, config 16 = SILK-NB 10 ms)
/// carrying arbitrary payload bytes for byte-identity checks.
fn opus_packet(payload_byte: u8) -> Vec<u8> {
    // TOC: config=16 (SILK-NB 10 ms), stereo=0, code=0 → 0x80
    // NB: config is bits 3..7, so 16<<3 = 0x80.
    let mut out = vec![0x80u8];
    out.extend_from_slice(&[payload_byte; 32]);
    out
}

/// A stand-in for a VP9 frame — the muxer is payload-agnostic, and we
/// only read the bytes back for equality comparison.
fn vp9_frame(marker: u8, len: usize) -> Vec<u8> {
    let mut v = vec![marker; len];
    v[0] = marker;
    v
}

fn build_vp9_stream() -> StreamInfo {
    let mut p = CodecParameters::video(CodecId::new("vp9"));
    p.width = Some(320);
    p.height = Some(240);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1000),
        duration: None,
        start_time: Some(0),
        params: p,
    }
}

fn build_opus_stream() -> StreamInfo {
    let mut p = CodecParameters::audio(CodecId::new("opus"));
    p.sample_rate = Some(48_000);
    p.channels = Some(2);
    p.extradata = opus_head(2, 48_000, 312);
    StreamInfo {
        index: 1,
        time_base: TimeBase::new(1, 48_000),
        duration: None,
        start_time: Some(0),
        params: p,
    }
}

/// In-memory cursor that looks like a file to the Muxer + Demuxer.
struct MemFile {
    inner: Cursor<Vec<u8>>,
}

impl std::io::Write for MemFile {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.inner.write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl std::io::Read for MemFile {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(b)
    }
}

impl std::io::Seek for MemFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[test]
fn webm_muxer_advertises_webm_format_name() {
    let video = build_vp9_stream();
    let audio = build_opus_stream();
    let streams = vec![video, audio];
    let mem = MemFile {
        inner: Cursor::new(Vec::new()),
    };
    let ws: Box<dyn WriteSeek> = Box::new(mem);
    let mux = oxideav_mkv::mux::open_webm(ws, &streams).unwrap();
    assert_eq!(mux.format_name(), "webm");
}

#[test]
fn webm_rejects_h264() {
    let mut params = CodecParameters::video(CodecId::new("h264"));
    params.width = Some(320);
    params.height = Some(240);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1000),
        duration: None,
        start_time: Some(0),
        params,
    };
    let mem = MemFile {
        inner: Cursor::new(Vec::new()),
    };
    let ws: Box<dyn WriteSeek> = Box::new(mem);
    let err = match oxideav_mkv::mux::open_webm(ws, std::slice::from_ref(&stream)) {
        Ok(_) => panic!("h264 must be rejected by the WebM muxer"),
        Err(e) => e,
    };
    match err {
        Error::Unsupported(msg) => {
            assert!(
                msg.contains("h264"),
                "error should mention the offending codec id, got: {msg}"
            );
            assert!(
                msg.contains("WebM"),
                "error should mention WebM context, got: {msg}"
            );
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}

#[test]
fn webm_rejects_flac() {
    let mut params = CodecParameters::audio(CodecId::new("flac"));
    params.sample_rate = Some(44_100);
    params.channels = Some(2);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 44_100),
        duration: None,
        start_time: Some(0),
        params,
    };
    let mem = MemFile {
        inner: Cursor::new(Vec::new()),
    };
    let ws: Box<dyn WriteSeek> = Box::new(mem);
    assert!(matches!(
        oxideav_mkv::mux::open_webm(ws, std::slice::from_ref(&stream)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn matroska_muxer_accepts_flac() {
    // Sanity check: the plain Matroska muxer must still accept codecs
    // outside the WebM whitelist.
    let mut params = CodecParameters::audio(CodecId::new("flac"));
    params.sample_rate = Some(44_100);
    params.channels = Some(2);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 44_100),
        duration: None,
        start_time: Some(0),
        params,
    };
    let mem = MemFile {
        inner: Cursor::new(Vec::new()),
    };
    let ws: Box<dyn WriteSeek> = Box::new(mem);
    let _mux = oxideav_mkv::mux::open(ws, std::slice::from_ref(&stream))
        .expect("matroska muxer must accept flac");
}

#[test]
fn webm_file_roundtrip_through_filesystem() {
    // Write a VP9+Opus WebM to a tmp file, re-open with the demuxer,
    // verify DocType + codec IDs + frame bytes.
    let video = build_vp9_stream();
    let audio = build_opus_stream();
    let streams = vec![video.clone(), audio.clone()];

    let tmp = std::env::temp_dir().join("oxideav-mkv-webm-roundtrip.webm");
    let v_frames: Vec<Vec<u8>> = (0..5).map(|i| vp9_frame(i as u8, 64 + i)).collect();
    let a_packets: Vec<Vec<u8>> = (0..5).map(|i| opus_packet((i + 100) as u8)).collect();
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_mkv::mux::open_webm(ws, &streams).unwrap();
        mux.write_header().unwrap();
        for i in 0..5 {
            let mut vp = Packet::new(0, video.time_base, v_frames[i].clone());
            vp.pts = Some((i as i64) * 40);
            vp.duration = Some(40);
            vp.flags.keyframe = i == 0;
            mux.write_packet(&vp).unwrap();

            let mut ap = Packet::new(1, audio.time_base, a_packets[i].clone());
            ap.pts = Some((i as i64) * 960);
            ap.duration = Some(960);
            ap.flags.keyframe = true;
            mux.write_packet(&ap).unwrap();
        }
        mux.write_trailer().unwrap();
    }

    // Peek at the EBML header bytes: should contain the ASCII "webm".
    let mut f = std::fs::File::open(&tmp).unwrap();
    let mut head = vec![0u8; 64];
    let n = f.read(&mut head).unwrap();
    head.truncate(n);
    assert!(
        head.windows(4).any(|w| w == b"webm"),
        "DocType 'webm' should appear in the first 64 bytes of the file"
    );

    // Re-open with the demuxer and verify.
    let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = oxideav_mkv::demux::open(rs).expect("demux");
    assert_eq!(dmx.streams().len(), 2);
    let (video_idx, audio_idx) = {
        let s = dmx.streams();
        let vi = s
            .iter()
            .find(|x| x.params.media_type == MediaType::Video)
            .expect("video stream present");
        let ai = s
            .iter()
            .find(|x| x.params.media_type == MediaType::Audio)
            .expect("audio stream present");
        assert_eq!(vi.params.codec_id, CodecId::new("vp9"));
        assert_eq!(ai.params.codec_id, CodecId::new("opus"));
        (vi.index, ai.index)
    };

    // Drain packets and collect by stream.
    let mut got_video: Vec<Vec<u8>> = Vec::new();
    let mut got_audio: Vec<Vec<u8>> = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => {
                if p.stream_index == video_idx {
                    got_video.push(p.data);
                } else if p.stream_index == audio_idx {
                    got_audio.push(p.data);
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    assert_eq!(got_video, v_frames, "VP9 frame bytes must round-trip");
    assert_eq!(got_audio, a_packets, "Opus packet bytes must round-trip");

    let _ = std::fs::remove_file(&tmp);
}

/// Optional ffmpeg interop: ask ffprobe to describe the file we just
/// wrote. Skipped if `ffprobe` isn't on `$PATH`. ffprobe should report
/// "matroska,webm" as the format (its demuxer covers both) and the two
/// streams should come back as vp9 + opus.
#[test]
fn webm_ffprobe_interop() {
    if std::process::Command::new("ffprobe")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("ffprobe not available, skipping interop test");
        return;
    }

    let video = build_vp9_stream();
    let audio = build_opus_stream();
    let streams = vec![video.clone(), audio.clone()];
    let tmp = std::env::temp_dir().join("oxideav-mkv-webm-ffprobe.webm");
    let v_frames: Vec<Vec<u8>> = (0..5).map(|i| vp9_frame(i as u8, 64 + i)).collect();
    let a_packets: Vec<Vec<u8>> = (0..5).map(|i| opus_packet((i + 100) as u8)).collect();
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_mkv::mux::open_webm(ws, &streams).unwrap();
        mux.write_header().unwrap();
        for i in 0..5 {
            let mut vp = Packet::new(0, video.time_base, v_frames[i].clone());
            vp.pts = Some((i as i64) * 40);
            vp.duration = Some(40);
            vp.flags.keyframe = i == 0;
            mux.write_packet(&vp).unwrap();
            let mut ap = Packet::new(1, audio.time_base, a_packets[i].clone());
            ap.pts = Some((i as i64) * 960);
            ap.duration = Some(960);
            ap.flags.keyframe = true;
            mux.write_packet(&ap).unwrap();
        }
        mux.write_trailer().unwrap();
    }

    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name:stream=codec_name",
            "-of",
            "default=nw=1",
        ])
        .arg(&tmp)
        .output()
        .expect("ffprobe executes");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // ffprobe reports matroska/webm format together because its demuxer
    // covers both. What matters: no errors, and codec names match.
    assert!(
        out.status.success(),
        "ffprobe failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("codec_name=vp9"),
        "ffprobe should identify vp9:\n{stdout}"
    );
    assert!(
        stdout.contains("codec_name=opus"),
        "ffprobe should identify opus:\n{stdout}"
    );
    assert!(
        stdout.contains("webm") || stdout.contains("matroska"),
        "ffprobe format field should mention webm/matroska:\n{stdout}"
    );

    let _ = std::fs::remove_file(&tmp);
}
