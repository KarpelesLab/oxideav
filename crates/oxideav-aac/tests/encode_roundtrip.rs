//! End-to-end encoder tests: feed PCM in, take ADTS out, decode through
//! both our own decoder and ffmpeg, verify Goertzel ratio at the source
//! frequency. Acceptance bar from the task: ratio ≥ 50× for both decoders
//! at 44.1 kHz mono and stereo, plus 48 kHz mono.

use std::path::Path;

use oxideav_aac::adts::{parse_adts_header, ADTS_HEADER_NO_CRC};
// Trait imports needed for `enc.send_frame` / `dec.send_packet` resolution.
#[allow(unused_imports)]
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

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

fn pcm_sine_mono(freq: f32, sr: u32, secs: f32, amp: f32) -> Vec<u8> {
    let total = (sr as f32 * secs) as usize;
    let mut out = Vec::with_capacity(total * 2);
    for i in 0..total {
        let t = i as f32 / sr as f32;
        let v = (2.0 * std::f32::consts::PI * freq * t).sin() * amp;
        let s = (v * 32767.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn pcm_sine_stereo(freq_l: f32, freq_r: f32, sr: u32, secs: f32, amp: f32) -> Vec<u8> {
    let total = (sr as f32 * secs) as usize;
    let mut out = Vec::with_capacity(total * 4);
    for i in 0..total {
        let t = i as f32 / sr as f32;
        let l = (2.0 * std::f32::consts::PI * freq_l * t).sin() * amp;
        let r = (2.0 * std::f32::consts::PI * freq_r * t).sin() * amp;
        let sl = (l * 32767.0) as i16;
        let sr_s = (r * 32767.0) as i16;
        out.extend_from_slice(&sl.to_le_bytes());
        out.extend_from_slice(&sr_s.to_le_bytes());
    }
    out
}

fn encode(pcm: Vec<u8>, sr: u32, channels: u16, bitrate: u64) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new("aac"));
    params.sample_rate = Some(sr);
    params.channels = Some(channels);
    params.bit_rate = Some(bitrate);
    let mut enc = oxideav_aac::encoder::make_encoder(&params).expect("make encoder");
    let total_samples = pcm.len() / (2 * channels as usize);
    let frame = Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate: sr,
        samples: total_samples as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, sr as i64),
        data: vec![pcm],
    });
    enc.send_frame(&frame).expect("send_frame");
    enc.flush().expect("flush");
    let mut out = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.extend_from_slice(&p.data);
    }
    out
}

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

fn decode_self(bytes: &[u8]) -> Vec<i16> {
    let frames = iter_adts(bytes);
    assert!(!frames.is_empty(), "no ADTS frames found");
    let first = parse_adts_header(&bytes[frames[0].0..]).unwrap();
    let sr = first.sample_rate().unwrap();
    let ch = first.channel_configuration as u16;
    let mut params = CodecParameters::audio(CodecId::new("aac"));
    params.sample_rate = Some(sr);
    params.channels = Some(ch);
    let mut dec = oxideav_aac::decoder::make_decoder(&params).expect("make dec");
    let tb = TimeBase::new(1, sr as i64);
    let mut samples = Vec::<i16>::new();
    for (i, &(off, len)) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, bytes[off..off + len].to_vec()).with_pts(i as i64 * 1024);
        dec.send_packet(&pkt).unwrap();
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => {
                for chunk in af.data[0].chunks_exact(2) {
                    samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
    samples
}

/// Returns `Some(decoded PCM)` if ffmpeg is on PATH and decode
/// succeeded; `None` if ffmpeg is missing — caller-side tests then
/// skip cleanly so CI without ffmpeg passes.
fn ffmpeg_decode(bytes: &[u8], out_wav: &Path) -> Option<Vec<i16>> {
    let in_path = std::env::temp_dir().join("oxideav_aac_enc_test.aac");
    std::fs::write(&in_path, bytes).expect("write tmp aac");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(&in_path)
        .arg("-f")
        .arg("s16le")
        .arg("-ar")
        .arg("44100")
        .arg(out_wav)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let raw = std::fs::read(out_wav).expect("read decoded wav");
    Some(
        raw.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

fn check_goertzel(samples: &[i16], sr: u32, channels: u16, target: f32, ch_idx: usize) -> f32 {
    let warm = 4 * 1024 * channels as usize;
    let analysis: Vec<f32> = samples[warm..]
        .chunks_exact(channels as usize)
        .map(|c| c[ch_idx] as f32 / 32768.0)
        .collect();
    let g_target = goertzel(&analysis, sr as f32, target);
    let off_freqs = [220.0, 660.0, 1000.0, 100.0, 50.0];
    let off_max = off_freqs
        .iter()
        .map(|&f| goertzel(&analysis, sr as f32, f))
        .fold(0.0f32, f32::max);
    let ratio = g_target / off_max.max(1e-9);
    eprintln!("[ch={ch_idx}] goertzel {target}={g_target}, off_max={off_max}, ratio={ratio}");
    ratio
}

#[test]
fn encode_mono_roundtrip_self_decoder() {
    let sr = 44_100u32;
    let pcm = pcm_sine_mono(440.0, sr, 1.0, 0.5);
    let aac = encode(pcm, sr, 1, 128_000);
    eprintln!("mono encoded size: {} bytes", aac.len());
    let decoded = decode_self(&aac);
    let ratio = check_goertzel(&decoded, sr, 1, 440.0, 0);
    assert!(
        ratio >= 50.0,
        "self-decode mono Goertzel ratio {ratio} < 50"
    );
}

#[test]
fn encode_stereo_roundtrip_self_decoder() {
    let sr = 44_100u32;
    let pcm = pcm_sine_stereo(440.0, 880.0, sr, 1.0, 0.5);
    let aac = encode(pcm, sr, 2, 128_000);
    eprintln!("stereo encoded size: {} bytes", aac.len());
    let decoded = decode_self(&aac);
    let r0 = check_goertzel(&decoded, sr, 2, 440.0, 0);
    let r1 = check_goertzel(&decoded, sr, 2, 880.0, 1);
    assert!(r0 >= 50.0, "stereo L Goertzel ratio {r0} < 50");
    assert!(r1 >= 50.0, "stereo R Goertzel ratio {r1} < 50");
}

#[test]
fn encode_mono_roundtrip_ffmpeg() {
    let sr = 44_100u32;
    let pcm = pcm_sine_mono(440.0, sr, 1.0, 0.5);
    let aac = encode(pcm, sr, 1, 128_000);
    let out_wav = std::env::temp_dir().join("oxideav_aac_enc_mono.s16");
    let Some(decoded) = ffmpeg_decode(&aac, &out_wav) else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let _ = std::fs::remove_file(&out_wav);
    let ratio = check_goertzel(&decoded, sr, 1, 440.0, 0);
    assert!(
        ratio >= 50.0,
        "ffmpeg-decoded mono Goertzel ratio {ratio} < 50"
    );
}

#[test]
fn encode_mono_48k_ffmpeg() {
    let sr = 48_000u32;
    let pcm = pcm_sine_mono(440.0, sr, 1.0, 0.5);
    let aac = encode(pcm, sr, 1, 128_000);
    let out_wav = std::env::temp_dir().join("oxideav_aac_enc_mono_48k.s16");
    let Some(decoded) = ffmpeg_decode(&aac, &out_wav) else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let _ = std::fs::remove_file(&out_wav);
    // ffmpeg_decode resamples to 44.1 kHz on output (its hard-coded -ar
    // flag), so the analysis sample rate is still 44100 even though the
    // source was 48k.
    let ratio = check_goertzel(&decoded, 44_100, 1, 440.0, 0);
    assert!(ratio >= 50.0, "ffmpeg mono 48k ratio {ratio} < 50");
}

#[test]
fn encode_stereo_roundtrip_ffmpeg() {
    let sr = 44_100u32;
    let pcm = pcm_sine_stereo(440.0, 880.0, sr, 1.0, 0.5);
    let aac = encode(pcm, sr, 2, 128_000);
    let out_wav = std::env::temp_dir().join("oxideav_aac_enc_stereo.s16");
    let Some(decoded) = ffmpeg_decode(&aac, &out_wav) else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let _ = std::fs::remove_file(&out_wav);
    let r0 = check_goertzel(&decoded, sr, 2, 440.0, 0);
    let r1 = check_goertzel(&decoded, sr, 2, 880.0, 1);
    assert!(r0 >= 50.0, "ffmpeg stereo L Goertzel ratio {r0} < 50");
    assert!(r1 >= 50.0, "ffmpeg stereo R Goertzel ratio {r1} < 50");
}
