//! Shared test helpers for cross-codec roundtrip comparison tests.
//!
//! Every codec test in this crate follows the same pattern:
//!
//! 1. Generate a known test signal (ffmpeg lavfi for audio, testsrc for video)
//! 2. **Encoder test**: encode with ours → decode with ffmpeg → compare
//!    against ffmpeg-encode → ffmpeg-decode of the same input
//! 3. **Decoder test**: encode with ffmpeg → decode with ours → compare
//!    against ffmpeg's own decode
//!
//! All tests skip gracefully when ffmpeg is absent.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const FFMPEG: &str = "/usr/bin/ffmpeg";

pub fn ffmpeg_available() -> bool {
    Path::new(FFMPEG).exists()
}

pub fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Run ffmpeg and return true on success.
pub fn ffmpeg(args: &[&str]) -> bool {
    let mut cmd_args = vec!["-y", "-hide_banner", "-loglevel", "error"];
    cmd_args.extend_from_slice(args);
    matches!(
        Command::new(FFMPEG).args(&cmd_args).status(),
        Ok(s) if s.success()
    )
}

/// Run ffmpeg with PathBuf args.
pub fn ffmpeg_paths(args: &[&std::ffi::OsStr]) -> bool {
    let base: Vec<&std::ffi::OsStr> = ["-y", "-hide_banner", "-loglevel", "error"]
        .iter()
        .map(|s| std::ffi::OsStr::new(*s))
        .collect();
    let mut all = base;
    all.extend_from_slice(args);
    matches!(
        Command::new(FFMPEG).args(&all).status(),
        Ok(s) if s.success()
    )
}

// ── Audio helpers ─────────────────────────────────────────────────────

/// Write interleaved s16le PCM to a file.
pub fn write_pcm_s16le(path: &Path, pcm: &[i16]) {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write pcm");
}

/// Read s16le PCM from a file.
pub fn read_pcm_s16le(path: &Path) -> Vec<i16> {
    let data = std::fs::read(path).expect("read pcm");
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Compute RMS difference between two PCM buffers (normalised to [-1, 1]).
pub fn audio_rms_diff(a: &[i16], b: &[i16]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::INFINITY;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let da = a[i] as f64 / 32768.0;
        let db = b[i] as f64 / 32768.0;
        let d = da - db;
        sum += d * d;
    }
    (sum / n as f64).sqrt()
}

/// Compute PSNR between two audio PCM buffers.
pub fn audio_psnr(a: &[i16], b: &[i16]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        mse += d * d;
    }
    mse /= n as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (32767.0f64 * 32767.0f64 / mse).log10()
}

/// Generate a deterministic test signal: 440 Hz sine + chirp + click.
pub fn generate_audio_signal(sample_rate: u32, channels: u16, duration_secs: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let mut pcm = Vec::with_capacity(n * channels as usize);
    for i in 0..n {
        let t = i as f64 / sample_rate as f64;
        let dur = duration_secs as f64;
        let sine = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
        let chirp_f = 800.0 + 400.0 * (t / dur);
        let chirp = 0.3 * (2.0 * std::f64::consts::PI * chirp_f * t).sin();
        let click_pos = sample_rate as usize;
        let click = if i >= click_pos && i < click_pos + 5 {
            0.8
        } else {
            0.0
        };
        let sample = ((sine * 0.5 + chirp + click).clamp(-1.0, 1.0) * 30000.0) as i16;
        for _ in 0..channels {
            pcm.push(sample);
        }
    }
    pcm
}

// ── Video helpers ─────────────────────────────────────────────────────

/// Read raw YUV420P from a file. Returns (Y, Cb, Cr) planes.
pub fn read_yuv420p(path: &Path, width: u32, height: u32) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;
    let frame_size = (width * height * 3 / 2) as usize;
    if data.len() < frame_size {
        return None;
    }
    Some(data)
}

/// Compute Y-plane PSNR between two YUV420P buffers.
pub fn video_y_psnr(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    let n = (width * height) as usize;
    if a.len() < n || b.len() < n {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        mse += d * d;
    }
    mse /= n as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0f64 / mse).log10()
}

/// Report line for a comparison result.
pub fn report(label: &str, rms: f64, psnr: f64, ours_len: usize, ref_len: usize) {
    eprintln!("  [{label}] RMS={rms:.6}  PSNR={psnr:.1} dB  ours={ours_len} ref={ref_len} samples");
}
