//! Integration tests that decode our generated 1-second AAC-LC fixtures.
//!
//! Acceptance: dominant frequency ≈ 440 Hz at ratio ≥ 5× over off-bin
//! reference, RMS above silence threshold.

use std::path::Path;

use oxideav_aac::adts::{parse_adts_header, ADTS_HEADER_NO_CRC};
use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

fn decoder_for(codec_id: &str, sr: u32, channels: u16) -> Box<dyn Decoder> {
    let params = CodecParameters {
        sample_rate: Some(sr),
        channels: Some(channels),
        ..CodecParameters::audio(CodecId::new(codec_id))
    };
    oxideav_aac::decoder::make_decoder(&params).expect("make decoder")
}

/// Iterate ADTS frames in `bytes`, yielding (offset, frame_length) tuples.
fn iter_adts(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + ADTS_HEADER_NO_CRC < bytes.len() {
        if bytes[i] != 0xFF || (bytes[i + 1] & 0xF0) != 0xF0 {
            i += 1;
            continue;
        }
        match parse_adts_header(&bytes[i..]) {
            Ok(h) => {
                if h.frame_length == 0 || i + h.frame_length > bytes.len() {
                    break;
                }
                out.push((i, h.frame_length));
                i += h.frame_length;
            }
            Err(_) => i += 1,
        }
    }
    out
}

/// Decode the entire ADTS file and return interleaved S16 samples.
fn decode_file(path: impl AsRef<Path>) -> (Vec<i16>, u32, u16, usize) {
    let bytes = std::fs::read(path).expect("fixture missing");
    let frames = iter_adts(&bytes);
    assert!(!frames.is_empty(), "no ADTS frames found");

    let first = parse_adts_header(&bytes[frames[0].0..]).unwrap();
    let sr = first.sample_rate().unwrap();
    let ch = first.channel_configuration as u16;

    let mut dec = decoder_for("aac", sr, ch);
    let mut all = Vec::<i16>::new();
    let tb = TimeBase::new(1, sr as i64);
    let mut decoded_frames = 0usize;
    for (i, &(off, len)) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, bytes[off..off + len].to_vec()).with_pts(i as i64 * 1024);
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(e) => panic!("send_packet at frame {i}: {e}"),
        }
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => {
                decoded_frames += 1;
                let bytes = &af.data[0];
                for chunk in bytes.chunks_exact(2) {
                    let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                    all.push(v);
                }
            }
            Ok(_) => panic!("expected audio frame"),
            Err(e) => panic!("receive_frame at frame {i}: {e}"),
        }
    }
    (all, sr, ch, decoded_frames)
}

/// Goertzel algorithm — single bin energy estimator.
fn goertzel(samples: &[f32], sample_rate: f32, target_freq: f32) -> f32 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let k = (0.5 + (n as f32 * target_freq) / sample_rate).floor();
    let omega = (2.0 * std::f32::consts::PI * k) / n as f32;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0;
    let mut s_prev2 = 0.0;
    for &x in samples {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    let power = s_prev2.powi(2) + s_prev.powi(2) - coeff * s_prev * s_prev2;
    power.sqrt()
}

fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for &s in samples {
        let v = s as f64 / 32768.0;
        acc += v * v;
    }
    (acc / samples.len() as f64).sqrt() as f32
}

#[test]
fn decode_mono_sine_fixture() {
    let path = Path::new("/tmp/aac_lc_mono.aac");
    if !path.exists() {
        eprintln!("fixture not present — skipping");
        return;
    }
    let (samples, sr, ch, frames) = decode_file(path);
    assert_eq!(ch, 1);
    assert!(frames >= 40, "expected many frames; got {frames}");
    assert!(
        samples.len() >= 30 * 1024,
        "too few samples: {}",
        samples.len()
    );

    let rms = rms_i16(&samples);
    eprintln!("mono: frames={frames} samples={} rms={rms}", samples.len());
    assert!(rms > 0.05, "RMS too low: {rms} — decoder produced silence?");

    // Skip a few warmup frames so transient/overlap state stabilises.
    let warm = 4 * 1024;
    let analysis: Vec<f32> = samples[warm..]
        .iter()
        .map(|&s| s as f32 / 32768.0)
        .collect();
    let g440 = goertzel(&analysis, sr as f32, 440.0);
    let g_off1 = goertzel(&analysis, sr as f32, 220.0);
    let g_off2 = goertzel(&analysis, sr as f32, 1000.0);
    let g_off3 = goertzel(&analysis, sr as f32, 100.0);
    let off_max = g_off1.max(g_off2).max(g_off3);
    let ratio = g440 / off_max.max(1e-9);
    eprintln!("mono: goertzel 440={g440}, off_max={off_max}, ratio={ratio}");
    assert!(
        ratio >= 5.0,
        "440 Hz energy ratio {ratio} insufficient (g440={g440}, off={off_max})"
    );
}

#[test]
fn decode_stereo_sine_fixture() {
    let path = Path::new("/tmp/aac_lc_stereo.aac");
    if !path.exists() {
        eprintln!("fixture not present — skipping");
        return;
    }
    let (samples, sr, ch, frames) = decode_file(path);
    assert_eq!(ch, 2);
    assert!(frames >= 40, "expected many frames; got {frames}");
    assert!(samples.len() >= 30 * 1024 * 2, "too few samples");

    let rms = rms_i16(&samples);
    eprintln!(
        "stereo: frames={frames} samples={} rms={rms}",
        samples.len()
    );
    assert!(rms > 0.05, "RMS too low: {rms}");

    // De-interleave left channel for analysis.
    let left: Vec<f32> = samples
        .chunks_exact(2)
        .skip(4 * 1024)
        .map(|p| p[0] as f32 / 32768.0)
        .collect();
    let g440 = goertzel(&left, sr as f32, 440.0);
    let g_off = goertzel(&left, sr as f32, 1000.0);
    let ratio = g440 / g_off.max(1e-9);
    eprintln!("stereo (L): goertzel 440={g440}, off={g_off}, ratio={ratio}");
    assert!(ratio >= 5.0, "440 Hz ratio {ratio} insufficient");
}
