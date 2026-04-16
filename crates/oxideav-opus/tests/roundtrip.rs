//! Integration tests using ffmpeg-produced reference clips.
//!
//! These tests are skipped gracefully if `/usr/bin/ffmpeg` or the reference
//! files are missing — consistent with other crates in the workspace.
//!
//! Scope today:
//!
//! * Mode detection via the TOC parser on real ffmpeg-produced packets.
//! * SILK/Hybrid rejection (the decoder must return `Unsupported` and not
//!   panic or emit garbage).
//! * CELT packet-framing invariants (total packet duration ≤ 120 ms).
//!
//! Full CELT audio decoding is not yet implemented; see
//! `oxideav-opus/src/decoder.rs` for the current scope.

use std::path::Path;
use std::process::Command;

use oxideav_container::{Demuxer, ReadSeek};
use oxideav_core::{Error, Frame};
use oxideav_opus::toc::{parse_packet, OpusMode, Toc};

const FFMPEG: &str = "/usr/bin/ffmpeg";

fn ffmpeg_available() -> bool {
    Path::new(FFMPEG).exists()
}

fn ensure_ref(path: &str, args: &[&str]) -> bool {
    if !ffmpeg_available() {
        return false;
    }
    if Path::new(path).exists() {
        return true;
    }
    let status = Command::new(FFMPEG)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(args)
        .arg(path)
        .status();
    matches!(status, Ok(s) if s.success()) && Path::new(path).exists()
}

fn ensure_celt_mono() -> Option<&'static str> {
    let path = "/tmp/ref-opus-celt-mono.opus";
    if ensure_ref(
        path,
        &[
            "-f",
            "lavfi",
            "-i",
            "sine=f=1000:d=1:sample_rate=48000",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            "-application",
            "audio",
        ],
    ) {
        Some(path)
    } else {
        None
    }
}

fn ensure_celt_stereo() -> Option<&'static str> {
    let path = "/tmp/ref-opus-celt-stereo.opus";
    if ensure_ref(
        path,
        &[
            "-f",
            "lavfi",
            "-i",
            "sine=f=1000:d=1:sample_rate=48000",
            "-ac",
            "2",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            "-application",
            "audio",
        ],
    ) {
        Some(path)
    } else {
        None
    }
}

fn ensure_voip_mono() -> Option<&'static str> {
    let path = "/tmp/ref-opus-voip-mono.opus";
    if ensure_ref(
        path,
        &[
            "-f",
            "lavfi",
            "-i",
            "sine=f=300:d=1:sample_rate=16000",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "16k",
            "-application",
            "voip",
        ],
    ) {
        Some(path)
    } else {
        None
    }
}

fn open_ogg(path: &str) -> Box<dyn Demuxer> {
    let f = std::fs::File::open(path).expect("open ref");
    let rs: Box<dyn ReadSeek> = Box::new(f);
    oxideav_ogg::demux::open(rs).expect("open ogg demuxer")
}

/// Mode-detection check: CELT-only reference TOC reports CELT-only.
#[test]
fn toc_reports_celt_only_for_music() {
    let Some(path) = ensure_celt_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let pkt = dmx.next_packet().expect("packet");
    let toc = Toc::parse(pkt.data[0]);
    assert_eq!(toc.mode, OpusMode::CeltOnly);
    assert_eq!(toc.frame_samples_48k, 960);
    assert!(!toc.stereo, "mono reference");
}

#[test]
fn toc_reports_celt_only_for_stereo_music() {
    let Some(path) = ensure_celt_stereo() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let pkt = dmx.next_packet().expect("packet");
    let toc = Toc::parse(pkt.data[0]);
    assert_eq!(toc.mode, OpusMode::CeltOnly);
    assert!(toc.stereo, "stereo reference");
}

#[test]
fn toc_reports_silk_only_for_voip() {
    let Some(path) = ensure_voip_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let pkt = dmx.next_packet().expect("packet");
    let toc = Toc::parse(pkt.data[0]);
    assert_eq!(toc.mode, OpusMode::SilkOnly);
}

/// Packet parse invariant: we can successfully split every real-world
/// packet from a music clip into frames, and every frame is non-empty.
#[test]
fn celt_mono_packets_parse_cleanly() {
    let Some(path) = ensure_celt_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let mut n = 0usize;
    loop {
        match dmx.next_packet() {
            Ok(pkt) => {
                let parsed = parse_packet(&pkt.data).expect("TOC parse");
                assert!(!parsed.frames.is_empty(), "packet #{} has zero frames", n);
                // ffmpeg produces code-0 (single frame) for CELT at 128 kbps.
                assert_eq!(parsed.toc.code, 0);
                n += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {}", e),
        }
    }
    assert!(n > 40, "expected >40 packets from a 1-second clip, got {n}");
}

/// Decode 50 SILK-only voip packets and assert we reject each one with a
/// clean `Unsupported` error — no panics, no garbage.
#[test]
fn silk_packets_are_rejected_not_crashed() {
    let Some(path) = ensure_voip_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let params = dmx.streams()[0].params.clone();
    let mut dec = oxideav_opus::decoder::make_decoder(&params).expect("make decoder");

    let mut rejected = 0usize;
    for _ in 0..50 {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {}", e),
        };
        dec.send_packet(&pkt).expect("send");
        let r = dec.receive_frame();
        match r {
            Err(Error::Unsupported(msg)) => {
                assert!(
                    msg.to_lowercase().contains("silk") || msg.to_lowercase().contains("hybrid"),
                    "unexpected Unsupported message: {}",
                    msg
                );
                rejected += 1;
            }
            Ok(_) => panic!("SILK packet unexpectedly decoded"),
            Err(e) => panic!("unexpected error type: {:?}", e),
        }
    }
    assert!(rejected >= 10, "expected ≥10 rejections, got {rejected}");
}

/// CELT-only packets with full audio content currently return
/// `Unsupported` after the front-of-frame header (silence/post-filter/
/// CELT decode pipeline runs end-to-end without panicking on real
/// ffmpeg-produced packets. Audio quality is gated separately by the
/// `#[ignore]`'d Goertzel test below — this test only pins the contract
/// that the structure is in place: every packet either produces a real
/// AudioFrame at the expected rate/length, or returns a CELT-tagged
/// `Unsupported` (e.g. for a stage we haven't bit-exact'd yet).
#[test]
fn celt_pipeline_runs_end_to_end() {
    let Some(path) = ensure_celt_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let params = dmx.streams()[0].params.clone();
    let mut dec = oxideav_opus::decoder::make_decoder(&params).expect("make decoder");

    let mut tested = 0usize;
    let mut saw_audio = false;
    for _ in 0..20 {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("demux: {}", e),
        };
        dec.send_packet(&pkt).expect("send");
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                assert_eq!(a.sample_rate, 48_000);
                assert_eq!(a.samples, 960);
                saw_audio = true;
            }
            Ok(Frame::Video(_)) => panic!("audio decoder returned video frame"),
            Err(Error::Unsupported(msg)) => {
                let lc = msg.to_lowercase();
                assert!(
                    lc.contains("celt") || lc.contains("silk") || lc.contains("hybrid"),
                    "Unsupported message should mention codec mode: {}",
                    msg
                );
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
        tested += 1;
    }
    assert!(tested > 0, "no packets tested");
    assert!(
        saw_audio,
        "expected at least one CELT packet to produce audio"
    );
}

/// Acceptance bar for the full CELT decoder. A 1-second 1 kHz sine-wave
/// CELT-only Opus mono clip should decode to PCM with a Goertzel ratio
/// at least 5× over the noise floor at 1 kHz.
///
/// Ignored until the decoder lands coarse energy, bit allocation, PVQ
/// shape decode, anti-collapse, IMDCT, and post-filter. Run via:
///   `cargo test -p oxideav-opus --test roundtrip -- --include-ignored`.
#[test]
#[ignore = "celt audio output not yet landed: needs §4.3.2 + §4.3.3 + §4.3.4 + §4.3.5 + §4.3.7 + §4.3.8"]
fn celt_mono_decodes_to_audible_sine() {
    let Some(path) = ensure_celt_mono() else {
        eprintln!("skip: ffmpeg / reference unavailable");
        return;
    };
    let mut dmx = open_ogg(path);
    let params = dmx.streams()[0].params.clone();
    let mut dec = oxideav_opus::decoder::make_decoder(&params).expect("make decoder");

    let mut pcm: Vec<f32> = Vec::with_capacity(48_000);
    loop {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("demux: {}", e),
        };
        dec.send_packet(&pkt).expect("send");
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                let bytes = &a.data[0];
                for chunk in bytes.chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    pcm.push(s as f32 / 32768.0);
                }
            }
            Ok(_) => panic!("expected audio"),
            Err(e) => panic!("decode error: {:?}", e),
        }
    }
    assert!(
        pcm.len() > 40_000,
        "expected ≥40k samples, got {}",
        pcm.len()
    );

    // RMS over the whole clip should be > 0.05 (a quiet sine is ~0.7×).
    let rms = (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32).sqrt();
    assert!(rms > 0.05, "RMS too low: {rms}");

    // Goertzel at 1 kHz vs 5 kHz (noise reference).
    let g_signal = goertzel(&pcm, 48_000.0, 1_000.0);
    let g_noise = goertzel(&pcm, 48_000.0, 5_000.0);
    assert!(
        g_signal > 5.0 * g_noise,
        "Goertzel ratio too small: 1kHz={g_signal}, 5kHz={g_noise}"
    );
}

/// Single-frequency Goertzel magnitude. Used by the audio acceptance test.
#[allow(dead_code)]
fn goertzel(samples: &[f32], sample_rate: f32, target_hz: f32) -> f32 {
    let k = (samples.len() as f32 * target_hz / sample_rate).round();
    let omega = 2.0 * std::f32::consts::PI * k / samples.len() as f32;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    for &x in samples {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    (s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2).sqrt()
}
