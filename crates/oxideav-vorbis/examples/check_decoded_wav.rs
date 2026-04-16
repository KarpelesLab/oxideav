//! Read a 16-bit PCM WAV (mono or stereo) and run a Goertzel detection at
//! 1 kHz vs an off-frequency. Prints the ratio. Used to verify ffmpeg's
//! decode of our encoded output contains the expected sine.
//!
//! Usage: `cargo run --example check_decoded_wav -- <wav_path> [freq_hz]`

use std::env;
use std::fs;

fn goertzel(samples: &[i16], freq: f64, sr: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * freq / sr;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0f64;
    let mut s_prev2 = 0f64;
    for &s in samples {
        let s_now = s as f64 + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s_now;
    }
    (s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2).sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .expect("usage: check_decoded_wav <wav> [freq_hz]");
    let freq: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000.0);
    let bytes = fs::read(path).expect("read wav");

    // Minimal WAV parser: scan for "fmt " and "data" chunks.
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut i = 12usize;
    let mut sr: u32 = 0;
    let mut nch: u16 = 0;
    let mut data_off = 0usize;
    let mut data_len = 0usize;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            nch = u16::from_le_bytes(bytes[i + 10..i + 12].try_into().unwrap());
            sr = u32::from_le_bytes(bytes[i + 12..i + 16].try_into().unwrap());
        } else if id == b"data" {
            data_off = i + 8;
            data_len = sz;
            break;
        }
        i += 8 + sz;
    }
    println!("WAV: {} Hz, {} ch, {} bytes data", sr, nch, data_len);
    let pcm: Vec<i16> = bytes[data_off..data_off + data_len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    if nch == 1 {
        let g_t = goertzel(&pcm, freq, sr as f64);
        let g_o = goertzel(&pcm, freq * 5.0, sr as f64);
        let rms = (pcm.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / pcm.len() as f64).sqrt();
        let peak = pcm.iter().map(|&s| s.abs() as i32).max().unwrap_or(0);
        println!(
            "mono: target {:.1} Hz = {:.0}, off {:.1} Hz = {:.0}, ratio {:.1}x, rms={:.1}, peak={}",
            freq,
            g_t,
            freq * 5.0,
            g_o,
            g_t / g_o.max(1.0),
            rms,
            peak
        );
    } else if nch == 2 {
        let mut l = Vec::with_capacity(pcm.len() / 2);
        let mut r = Vec::with_capacity(pcm.len() / 2);
        for chunk in pcm.chunks_exact(2) {
            l.push(chunk[0]);
            r.push(chunk[1]);
        }
        let g_l = goertzel(&l, freq, sr as f64);
        let g_r = goertzel(&r, freq, sr as f64);
        let g_lo = goertzel(&l, freq * 5.0, sr as f64);
        let g_ro = goertzel(&r, freq * 5.0, sr as f64);
        let rms_l = (l.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / l.len() as f64).sqrt();
        let rms_r = (r.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
        let peak_l = l.iter().map(|&s| s.abs() as i32).max().unwrap_or(0);
        let peak_r = r.iter().map(|&s| s.abs() as i32).max().unwrap_or(0);
        println!(
            "stereo L: target={:.0}, off={:.0}, ratio {:.1}x, rms={:.1}, peak={}",
            g_l,
            g_lo,
            g_l / g_lo.max(1.0),
            rms_l,
            peak_l
        );
        println!(
            "stereo R: target={:.0}, off={:.0}, ratio {:.1}x, rms={:.1}, peak={}",
            g_r,
            g_ro,
            g_r / g_ro.max(1.0),
            rms_r,
            peak_r
        );
    }
}
