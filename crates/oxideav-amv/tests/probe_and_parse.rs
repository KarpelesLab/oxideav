//! Integration tests for the AMV crate.
//!
//! Two tiers — an always-on hand-crafted AMV byte stream that exercises
//! the probe, header parser, and first chunks of `next_packet`; and an
//! ffmpeg-gated end-to-end test that generates a tiny real AMV file with
//! ffmpeg's AMV encoder and runs it through the demuxer plus the audio
//! and video decoders. The ffmpeg-gated test is skipped cleanly when
//! `/usr/bin/ffmpeg` is missing, matching the convention used by
//! `oxideav-opus/tests/roundtrip.rs`.

use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use oxideav_codec::CodecRegistry;
use oxideav_container::{ContainerRegistry, ProbeData};
use oxideav_core::{Error, Frame, MediaType};

const FFMPEG: &str = "/usr/bin/ffmpeg";
const REF_PATH: &str = "/tmp/oxideav-amv-ref.amv";

fn ffmpeg_available() -> bool {
    Path::new(FFMPEG).exists()
}

fn ensure_ref() -> bool {
    if !ffmpeg_available() {
        return false;
    }
    // Discard previous zero-byte attempts: ffmpeg's amv encoder rejects
    // many parameter combinations and leaves an empty file behind. We
    // require a non-empty file or we re-generate.
    if let Ok(meta) = std::fs::metadata(REF_PATH) {
        if meta.len() > 0 {
            return true;
        }
        let _ = std::fs::remove_file(REF_PATH);
    }
    // Combine `testsrc` with a silent mono audio source so the AMV muxer
    // gets the two streams it requires. The block_size must match the
    // sample rate / frame rate ratio (22050 / 15 = 1470).
    let status = Command::new(FFMPEG)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=15:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-shortest",
            "-c:v",
            "amv",
            "-c:a",
            "adpcm_ima_amv",
            "-ar",
            "22050",
            "-block_size",
            "1470",
            REF_PATH,
        ])
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        return false;
    }
    matches!(std::fs::metadata(REF_PATH), Ok(m) if m.len() > 0)
}

// ---- Hand-crafted AMV ------------------------------------------------------

fn put_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Build a tiny in-memory AMV byte stream large enough to exercise probe +
/// header parse + at least one video and one audio chunk.
fn make_minimal_amv() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    // RIFF <0> AMV
    buf.extend_from_slice(b"RIFF");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"AMV ");

    // LIST <0> hdrl
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"hdrl");

    // amvh <56> body — only width@32, height@36, fps_den@40, fps_num@44 matter.
    buf.extend_from_slice(b"amvh");
    put_u32_le(&mut buf, 56);
    let mut amvh = vec![0u8; 56];
    // width = 16 @ offset 32
    amvh[32..36].copy_from_slice(&16u32.to_le_bytes());
    // height = 16 @ offset 36
    amvh[36..40].copy_from_slice(&16u32.to_le_bytes());
    // fps_den = 1 @ offset 40
    amvh[40..44].copy_from_slice(&1u32.to_le_bytes());
    // fps_num = 15 @ offset 44
    amvh[44..48].copy_from_slice(&15u32.to_le_bytes());
    buf.extend_from_slice(&amvh);

    // LIST <0> strl (video, ignored)
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"strl");
    // strh 56 zero
    buf.extend_from_slice(b"strh");
    put_u32_le(&mut buf, 56);
    buf.extend_from_slice(&[0u8; 56]);
    // strf 36 zero (video)
    buf.extend_from_slice(b"strf");
    put_u32_le(&mut buf, 36);
    buf.extend_from_slice(&[0u8; 36]);

    // LIST <0> strl (audio, must come first to be the audio strf)
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"strl");
    // strh 48 zero
    buf.extend_from_slice(b"strh");
    put_u32_le(&mut buf, 48);
    buf.extend_from_slice(&[0u8; 48]);
    // strf 20 = WAVEFORMATEX (audio)
    buf.extend_from_slice(b"strf");
    put_u32_le(&mut buf, 20);
    let mut wfx = vec![0u8; 20];
    put_u16_le(&mut Vec::new(), 0); // unused helper
    wfx[0..2].copy_from_slice(&1u16.to_le_bytes()); // wFormatTag
    wfx[2..4].copy_from_slice(&1u16.to_le_bytes()); // channels = 1
    wfx[4..8].copy_from_slice(&22_050u32.to_le_bytes()); // sample_rate
    wfx[8..12].copy_from_slice(&22_050u32.to_le_bytes()); // byte_rate (not validated)
    wfx[12..14].copy_from_slice(&1u16.to_le_bytes()); // block_align
    wfx[14..16].copy_from_slice(&16u16.to_le_bytes()); // bps
    buf.extend_from_slice(&wfx);

    // LIST <0> movi
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"movi");

    // 00dc <4> [0xFF 0xD8 0xFF 0xD9]  ← minimal SOI+EOI video chunk
    buf.extend_from_slice(b"00dc");
    put_u32_le(&mut buf, 4);
    buf.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xD9]);

    // 01wb <12> [pred lo hi, step_idx, 5 reserved, then 4 nibble bytes]
    buf.extend_from_slice(b"01wb");
    put_u32_le(&mut buf, 12);
    let mut audio = vec![0u8; 8];
    audio[0] = 0; // predictor lo
    audio[1] = 0; // predictor hi
    audio[2] = 0; // step_index = 0
                  // 4 zero data bytes → 8 silent samples
    audio.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(&audio);

    // Trailer.
    buf.extend_from_slice(b"AMV_");
    buf.extend_from_slice(b"END_");

    buf
}

// ---- Always-on tests -------------------------------------------------------

#[test]
fn probe_recognises_amv_magic() {
    let blob = make_minimal_amv();
    {
        let _pd = ProbeData {
            buf: &blob,
            ext: None,
        };
    }
    // Register and probe via the public registration helper.
    let mut reg = ContainerRegistry::new();
    oxideav_amv::register_containers(&mut reg);
    // Use the public probe function indirectly: probe_input wants a reader.
    let mut rs = Cursor::new(blob.clone());
    let name = reg
        .probe_input(&mut rs, Some("amv"))
        .expect("AMV probe should succeed");
    assert_eq!(name, "amv");
    // Sanity: confirm we can open the resolved demuxer.
    let _demux = reg
        .open_demuxer(&name, Box::new(Cursor::new(blob)))
        .expect("open_demuxer should succeed");
}

#[test]
fn open_and_read_minimal_amv() {
    let blob = make_minimal_amv();
    let mut reg = ContainerRegistry::new();
    oxideav_amv::register_containers(&mut reg);
    let mut demux = reg
        .open_demuxer("amv", Box::new(Cursor::new(blob)))
        .expect("open should succeed");
    assert_eq!(demux.format_name(), "amv");
    let streams = demux.streams().to_vec();
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].params.media_type, MediaType::Video);
    assert_eq!(streams[0].params.width, Some(16));
    assert_eq!(streams[0].params.height, Some(16));
    assert_eq!(streams[1].params.media_type, MediaType::Audio);
    assert_eq!(streams[1].params.channels, Some(1));
    assert_eq!(streams[1].params.sample_rate, Some(22_050));

    // First packet should be the video chunk.
    let v_pkt = demux.next_packet().expect("video packet");
    assert_eq!(v_pkt.stream_index, 0);
    assert_eq!(v_pkt.data.len(), 4);
    assert_eq!(&v_pkt.data, &[0xFF, 0xD8, 0xFF, 0xD9]);

    // Second packet should be the audio chunk.
    let a_pkt = demux.next_packet().expect("audio packet");
    assert_eq!(a_pkt.stream_index, 1);
    assert_eq!(a_pkt.data.len(), 12);

    // Then EOF.
    match demux.next_packet() {
        Err(Error::Eof) => {}
        other => panic!("expected Eof, got {:?}", other),
    }
}

// ---- ffmpeg-gated end-to-end test ------------------------------------------

#[test]
fn ffmpeg_roundtrip_decodes_video_and_audio() {
    if !ensure_ref() {
        eprintln!(
            "ffmpeg-gated AMV test skipped — set up {} via ffmpeg amv encoder to enable.",
            REF_PATH
        );
        return;
    }
    let bytes = std::fs::read(REF_PATH).expect("read ref AMV");
    let mut creg = ContainerRegistry::new();
    let mut codecs = CodecRegistry::new();
    oxideav_amv::register(&mut codecs, &mut creg);

    let mut demux = creg
        .open_demuxer("amv", Box::new(Cursor::new(bytes)))
        .expect("open ffmpeg AMV file");
    let streams = demux.streams().to_vec();
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].params.media_type, MediaType::Video);
    assert_eq!(streams[0].params.width, Some(64));
    assert_eq!(streams[0].params.height, Some(64));
    assert_eq!(streams[1].params.media_type, MediaType::Audio);
    assert_eq!(streams[1].params.channels, Some(1));
    assert_eq!(streams[1].params.sample_rate, Some(22_050));

    let mut vdec = codecs
        .make_decoder(&streams[0].params)
        .expect("make video decoder");
    let mut adec = codecs
        .make_decoder(&streams[1].params)
        .expect("make audio decoder");

    let mut decoded_video_frames = 0usize;
    let mut decoded_audio_frames = 0usize;
    let mut total_audio_samples = 0usize;

    loop {
        match demux.next_packet() {
            Ok(pkt) => {
                if pkt.stream_index == 0 {
                    vdec.send_packet(&pkt).expect("send video packet");
                    let frame = vdec.receive_frame().expect("receive video frame");
                    if let Frame::Video(vf) = frame {
                        assert_eq!(vf.width, 64);
                        assert_eq!(vf.height, 64);
                        assert!(!vf.planes.is_empty());
                        decoded_video_frames += 1;
                    } else {
                        panic!("expected video frame");
                    }
                } else if pkt.stream_index == 1 {
                    adec.send_packet(&pkt).expect("send audio packet");
                    let frame = adec.receive_frame().expect("receive audio frame");
                    if let Frame::Audio(af) = frame {
                        assert_eq!(af.sample_rate, 22_050);
                        assert_eq!(af.channels, 1);
                        decoded_audio_frames += 1;
                        total_audio_samples += af.samples as usize;
                    } else {
                        panic!("expected audio frame");
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {:?}", e),
        }
    }
    assert!(
        decoded_video_frames > 0,
        "no video frames decoded from ffmpeg AMV file"
    );
    assert!(
        decoded_audio_frames > 0,
        "no audio frames decoded from ffmpeg AMV file"
    );
    assert!(total_audio_samples > 0, "no audio samples decoded");
}
