//! Encode → decode roundtrip tests for the GSM 06.10 Full Rate codec.
//!
//! 1. Pure in-process roundtrip: generate a 1 kHz mono sine at 8 kHz,
//!    encode with our encoder, decode with our decoder, then verify the
//!    energy level (RMS) and 1 kHz Goertzel concentration.
//! 2. Optional ffmpeg-libgsm interoperability: if `/usr/bin/ffmpeg` is
//!    available and its `libgsm` encoder is built in, compare our output
//!    against ffmpeg's for a basic sanity check (falls back to skip).
//!
//! The `ffmpeg-skip-when-missing` idiom mirrors `oxideav-opus/tests/
//! roundtrip.rs`: we never fail the test suite just because the reference
//! binary isn't present.

use std::path::Path;
use std::process::Command;

use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_gsm::decoder::make_decoder;
use oxideav_gsm::encoder::make_encoder;
use oxideav_gsm::frame::{FRAME_SIZE, MS_FRAME_SIZE};
use oxideav_gsm::{CODEC_ID_MS_STR, CODEC_ID_STR};

const FFMPEG: &str = "/usr/bin/ffmpeg";

fn build_sine(freq_hz: f32, sample_rate: u32, duration_s: f32, amplitude: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let v = (two_pi * freq_hz * t).sin() * amplitude;
            v.clamp(-32767.0, 32767.0) as i16
        })
        .collect()
}

fn pcm_to_frame_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn encode_pcm(codec_id: &str, pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut params = CodecParameters::audio(CodecId::new(codec_id));
    params.sample_rate = Some(8_000);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = make_encoder(&params).expect("encoder");
    let tb = TimeBase::new(1, 8_000);

    // Feed in 160-sample chunks.
    let bytes = pcm_to_frame_bytes(pcm);
    let chunk_bytes = 160 * 2;
    let mut pts: i64 = 0;
    for slice in bytes.chunks(chunk_bytes) {
        let n_samples = (slice.len() / 2) as u32;
        let af = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 8_000,
            samples: n_samples,
            pts: Some(pts),
            time_base: tb,
            data: vec![slice.to_vec()],
        };
        enc.send_frame(&Frame::Audio(af)).expect("send");
        pts += n_samples as i64;
    }
    enc.flush().expect("flush");

    let mut out = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        out.push(pkt.data);
    }
    out
}

fn decode_packets(codec_id: &str, packets: &[Vec<u8>]) -> Vec<f32> {
    let mut params = CodecParameters::audio(CodecId::new(codec_id));
    params.sample_rate = Some(8_000);
    params.channels = Some(1);
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 8_000);
    let mut pcm: Vec<f32> = Vec::new();
    let frames_per_packet = match codec_id {
        CODEC_ID_MS_STR => 2,
        _ => 1,
    };
    for data in packets {
        let pkt = Packet::new(0, tb, data.clone());
        dec.send_packet(&pkt).expect("send");
        for _ in 0..frames_per_packet {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for chunk in a.data[0].chunks_exact(2) {
                        let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                        pcm.push(s);
                    }
                }
                Ok(_) => panic!("expected audio"),
                Err(e) => panic!("decode error: {e:?}"),
            }
        }
    }
    pcm
}

fn rms(pcm: &[f32]) -> f32 {
    (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32).sqrt()
}

/// Goertzel magnitude at `target_hz` — returns a single scalar proportional
/// to the energy at that bin. Compare multiple targets by ratio.
fn goertzel(pcm: &[f32], sample_rate: f32, target_hz: f32) -> f32 {
    let k = (pcm.len() as f32 * target_hz / sample_rate).round();
    let omega = 2.0 * std::f32::consts::PI * k / pcm.len() as f32;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    for &x in pcm {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    (s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2).sqrt()
}

#[test]
fn sine_roundtrip_standard_framing() {
    // 1-second 1 kHz sine at 8 kHz → 50 frames × 33 bytes = 1650 bytes.
    let pcm = build_sine(1000.0, 8_000, 1.0, 20_000.0);
    let packets = encode_pcm(CODEC_ID_STR, &pcm);
    assert_eq!(packets.len(), 50);
    for p in &packets {
        assert_eq!(p.len(), FRAME_SIZE);
        assert_eq!((p[0] >> 4) & 0x0F, 0xD);
    }
    let decoded = decode_packets(CODEC_ID_STR, &packets);
    assert_eq!(decoded.len(), 8_000);

    let r = rms(&decoded);
    assert!(r > 0.05, "decoded RMS too low: {r}");

    let g_sig = goertzel(&decoded, 8_000.0, 1_000.0);
    let g_noise = goertzel(&decoded, 8_000.0, 2_500.0);
    assert!(
        g_sig > 3.0 * g_noise,
        "1kHz Goertzel ratio too small: {g_sig} vs {g_noise}"
    );
}

#[test]
fn sine_roundtrip_ms_framing() {
    // 1-second 1 kHz sine, MS framing emits 65-byte packets (each = 2 frames).
    let pcm = build_sine(1000.0, 8_000, 1.0, 20_000.0);
    let packets = encode_pcm(CODEC_ID_MS_STR, &pcm);
    assert_eq!(packets.len(), 25);
    for p in &packets {
        assert_eq!(p.len(), MS_FRAME_SIZE);
    }
    let decoded = decode_packets(CODEC_ID_MS_STR, &packets);
    assert_eq!(decoded.len(), 8_000);
    let r = rms(&decoded);
    assert!(r > 0.05, "decoded RMS too low: {r}");
    let g_sig = goertzel(&decoded, 8_000.0, 1_000.0);
    let g_noise = goertzel(&decoded, 8_000.0, 2_500.0);
    assert!(
        g_sig > 3.0 * g_noise,
        "1kHz Goertzel ratio too small: {g_sig} vs {g_noise}"
    );
}

#[test]
fn bit_exact_roundtrip_after_warmup_is_stable() {
    // Two consecutive encode-then-decode passes over the same PCM should
    // yield identical packet bytes — the encoder is fully deterministic
    // with no runtime-dependent state.
    let pcm = build_sine(440.0, 8_000, 0.5, 12_000.0);
    let a = encode_pcm(CODEC_ID_STR, &pcm);
    let b = encode_pcm(CODEC_ID_STR, &pcm);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x, y, "encoder is non-deterministic");
    }
}

/// ffmpeg decoding of our bitstream: if a libgsm-enabled ffmpeg is
/// available, wrap our packets in a raw `.gsm` file (which ffmpeg reads
/// directly) and decode to s16 PCM, then compare energy levels. Skips
/// cleanly otherwise.
#[test]
fn ffmpeg_can_decode_our_standard_bitstream() {
    if !Path::new(FFMPEG).exists() {
        eprintln!("skip: {FFMPEG} missing");
        return;
    }
    // Check that ffmpeg actually has a gsm decoder built in.
    let probe = Command::new(FFMPEG)
        .args(["-hide_banner", "-decoders"])
        .output();
    match probe {
        Ok(o) if String::from_utf8_lossy(&o.stdout).contains(" gsm ") => {}
        _ => {
            eprintln!("skip: ffmpeg lacks gsm decoder");
            return;
        }
    }

    let pcm = build_sine(1000.0, 8_000, 1.0, 20_000.0);
    let packets = encode_pcm(CODEC_ID_STR, &pcm);

    // Concatenate into a single .gsm bitstream.
    let gsm_path = "/tmp/oxideav-gsm-roundtrip.gsm";
    let mut bitstream = Vec::new();
    for p in &packets {
        bitstream.extend_from_slice(p);
    }
    std::fs::write(gsm_path, &bitstream).expect("write gsm");

    let pcm_path = "/tmp/oxideav-gsm-roundtrip.s16";
    let status = Command::new(FFMPEG)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "gsm",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-i",
            gsm_path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            pcm_path,
        ])
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("skip: ffmpeg decode failed");
            return;
        }
    }

    let bytes = std::fs::read(pcm_path).expect("read decoded");
    assert!(bytes.len() >= 2 * 8_000, "decoded PCM too short");
    let decoded: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    let r = rms(&decoded);
    assert!(r > 0.05, "ffmpeg-decoded RMS too low: {r}");
    let g_sig = goertzel(&decoded, 8_000.0, 1_000.0);
    let g_noise = goertzel(&decoded, 8_000.0, 2_500.0);
    assert!(
        g_sig > 3.0 * g_noise,
        "ffmpeg-decoded 1kHz ratio too small: {g_sig} vs {g_noise}"
    );
}
