//! Integration tests for the Asterisk signed-linear (`.sln*`) container.
//!
//! Covers:
//!   * Mux → demux byte-for-byte round trip at the two Asterisk rates that
//!     the demuxer must infer from the file extension (`.sln` = 8 kHz,
//!     `.sln48` = 48 kHz).
//!   * Probe scoring: `sln` extension hint must score > 0 but weakly
//!     (≤ `PROBE_SCORE_EXTENSION` = 25), reflecting that the probe is
//!     extension-only and has no content signature to corroborate.

use std::io::Cursor;

use oxideav_basic::{register_containers, slin};
use oxideav_container::{ContainerRegistry, ProbeData, WriteSeek, PROBE_SCORE_EXTENSION};
use oxideav_core::{
    CodecId, CodecParameters, MediaType, Packet, SampleFormat, StreamInfo, TimeBase,
};

/// Build a 1-channel S16LE stream description for the muxer.
fn mono_s16_stream(sample_rate: u32) -> StreamInfo {
    let mut params = CodecParameters::audio(CodecId::new("pcm_s16le"));
    params.media_type = MediaType::Audio;
    params.channels = Some(1);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, sample_rate as i64),
        duration: None,
        start_time: Some(0),
        params,
    }
}

/// Synth 100 ms of S16LE sawtooth so the byte pattern has structure and
/// a stuck-byte demuxer bug would be obvious.
fn synth_100ms_s16(sample_rate: u32) -> Vec<u8> {
    let n = (sample_rate as usize) / 10; // 100 ms
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let s = (((i as i32) * 327) % 65536 - 32768) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn roundtrip_through_file(sample_rate: u32, ext: &str) {
    let payload = synth_100ms_s16(sample_rate);
    assert_eq!(payload.len(), (sample_rate as usize / 10) * 2);

    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);

    // Mux.
    let tmp = std::env::temp_dir().join(format!("oxideav-basic-slin-{sample_rate}.{ext}"));
    let _ = std::fs::remove_file(&tmp);
    let stream = mono_s16_stream(sample_rate);
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = reg
            .open_muxer("slin", ws, std::slice::from_ref(&stream))
            .expect("open slin muxer");
        mux.write_header().unwrap();
        let pkt = Packet::new(0, stream.time_base, payload.clone());
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }

    // File size must match the payload byte-for-byte (no header, no trailer).
    let on_disk = std::fs::read(&tmp).unwrap();
    assert_eq!(
        on_disk, payload,
        "slin file body must equal raw S16LE payload"
    );

    // Demux via the explicit-sample-rate entry point — this keeps the test
    // honest about what rate is expected regardless of the registry's
    // default.
    let f = std::fs::File::open(&tmp).unwrap();
    let mut dmx = slin::open_demuxer_with_rate(Box::new(f), sample_rate).expect("open demuxer");
    assert_eq!(dmx.format_name(), "slin");
    assert_eq!(dmx.streams().len(), 1);
    assert_eq!(
        dmx.streams()[0].params.sample_rate,
        Some(sample_rate),
        "demuxer should report the sample rate it was opened at"
    );
    assert_eq!(dmx.streams()[0].params.channels, Some(1));
    assert_eq!(
        dmx.streams()[0].params.sample_format,
        Some(SampleFormat::S16)
    );

    let mut out = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => out.extend_from_slice(&p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e}"),
        }
    }
    assert_eq!(out, payload, "round-tripped PCM bytes must match input");

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn roundtrip_sln_8k() {
    roundtrip_through_file(8_000, "sln");
}

#[test]
fn roundtrip_sln48_48k() {
    roundtrip_through_file(48_000, "sln48");
}

#[test]
fn probe_sln_extension_is_weak_but_positive() {
    // Empty buffer — raw PCM has no signature, so only the extension hint
    // can drive a positive score.
    let data = ProbeData {
        buf: &[],
        ext: Some("sln"),
    };
    let score = slin::probe(&data);
    assert!(score > 0, "slin probe must fire on .sln extension");
    assert!(
        score <= PROBE_SCORE_EXTENSION,
        "slin probe must stay at or below PROBE_SCORE_EXTENSION ({PROBE_SCORE_EXTENSION}), got {score}"
    );
    assert!(score <= 25, "slin probe must be weak (<= 25), got {score}");
}

#[test]
fn probe_without_extension_returns_zero() {
    let data = ProbeData {
        buf: &[0xAA; 512],
        ext: None,
    };
    assert_eq!(slin::probe(&data), 0);
}

#[test]
fn probe_unrelated_extension_returns_zero() {
    let data = ProbeData {
        buf: &[],
        ext: Some("wav"),
    };
    assert_eq!(slin::probe(&data), 0);
}

#[test]
fn registry_probe_input_picks_slin_on_extension() {
    let mut reg = ContainerRegistry::new();
    register_containers(&mut reg);
    let mut cur = Cursor::new(vec![0u8; 16]);
    let name = reg.probe_input(&mut cur, Some("sln16")).expect("probe");
    assert_eq!(name, "slin");
}
