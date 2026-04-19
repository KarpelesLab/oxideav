//! Audio format conversion helpers shared between output drivers.
//!
//! The `oxideav` core audio model admits many sample formats (U8/S8/
//! S16/S24/S32/F32/F64 in interleaved or planar layouts) and decoders
//! emit whatever their native format is. Both the SDL2 and the
//! winit+sysaudio drivers feed the OS a single normalised f32 interleaved
//! stream, so the conversion + resampling + up/down-mix logic lives
//! here once.

use oxideav_core::{AudioFrame, SampleFormat};

/// Convert any [`AudioFrame`] payload to f32 interleaved at
/// `out_channels` (1 = mono, 2 = stereo). Mono destination averages
/// input channels; stereo destination duplicates mono or picks the
/// first two channels verbatim.
pub fn to_f32_interleaved(frame: &AudioFrame, out_channels: u16) -> Vec<f32> {
    let in_ch = frame.channels.max(1) as usize;
    let n = frame.samples as usize;
    let out_ch = out_channels.max(1) as usize;
    let mut out = Vec::with_capacity(n * out_ch);

    // Pull one (channel, sample) value as f32 in [-1, 1] from the source.
    let sample_at = |ch: usize, i: usize| -> f32 {
        match frame.format {
            SampleFormat::U8 => {
                let b = frame.data[0][i * in_ch + ch];
                (b as f32 - 128.0) / 128.0
            }
            SampleFormat::S8 => {
                let b = frame.data[0][i * in_ch + ch] as i8;
                b as f32 / 128.0
            }
            SampleFormat::S16 => {
                let off = (i * in_ch + ch) * 2;
                let v = i16::from_le_bytes([frame.data[0][off], frame.data[0][off + 1]]);
                v as f32 / 32768.0
            }
            SampleFormat::S24 => {
                let off = (i * in_ch + ch) * 3;
                let b0 = frame.data[0][off] as i32;
                let b1 = frame.data[0][off + 1] as i32;
                let b2 = frame.data[0][off + 2] as i32;
                let mut v = b0 | (b1 << 8) | (b2 << 16);
                if v & 0x80_0000 != 0 {
                    v |= !0xFF_FFFF;
                }
                v as f32 / 8_388_608.0
            }
            SampleFormat::S32 => {
                let off = (i * in_ch + ch) * 4;
                let v = i32::from_le_bytes([
                    frame.data[0][off],
                    frame.data[0][off + 1],
                    frame.data[0][off + 2],
                    frame.data[0][off + 3],
                ]);
                v as f32 / 2_147_483_648.0
            }
            SampleFormat::F32 => {
                let off = (i * in_ch + ch) * 4;
                f32::from_le_bytes([
                    frame.data[0][off],
                    frame.data[0][off + 1],
                    frame.data[0][off + 2],
                    frame.data[0][off + 3],
                ])
            }
            SampleFormat::F64 => {
                let off = (i * in_ch + ch) * 8;
                let v = f64::from_le_bytes([
                    frame.data[0][off],
                    frame.data[0][off + 1],
                    frame.data[0][off + 2],
                    frame.data[0][off + 3],
                    frame.data[0][off + 4],
                    frame.data[0][off + 5],
                    frame.data[0][off + 6],
                    frame.data[0][off + 7],
                ]);
                v as f32
            }
            SampleFormat::U8P => {
                let b = frame.data[ch][i];
                (b as f32 - 128.0) / 128.0
            }
            SampleFormat::S16P => {
                let off = i * 2;
                let v = i16::from_le_bytes([frame.data[ch][off], frame.data[ch][off + 1]]);
                v as f32 / 32768.0
            }
            SampleFormat::S32P => {
                let off = i * 4;
                let v = i32::from_le_bytes([
                    frame.data[ch][off],
                    frame.data[ch][off + 1],
                    frame.data[ch][off + 2],
                    frame.data[ch][off + 3],
                ]);
                v as f32 / 2_147_483_648.0
            }
            SampleFormat::F32P => {
                let off = i * 4;
                f32::from_le_bytes([
                    frame.data[ch][off],
                    frame.data[ch][off + 1],
                    frame.data[ch][off + 2],
                    frame.data[ch][off + 3],
                ])
            }
            SampleFormat::F64P => {
                let off = i * 8;
                let v = f64::from_le_bytes([
                    frame.data[ch][off],
                    frame.data[ch][off + 1],
                    frame.data[ch][off + 2],
                    frame.data[ch][off + 3],
                    frame.data[ch][off + 4],
                    frame.data[ch][off + 5],
                    frame.data[ch][off + 6],
                    frame.data[ch][off + 7],
                ]);
                v as f32
            }
        }
    };

    // Up/down-mix by duplicating or averaging channels.
    for i in 0..n {
        for oc in 0..out_ch {
            let src_ch = if in_ch == 1 {
                0
            } else if out_ch == 1 {
                // Mono: average input channels.
                let mut acc = 0.0f32;
                for ic in 0..in_ch {
                    acc += sample_at(ic, i);
                }
                out.push(acc / in_ch as f32);
                continue;
            } else {
                oc.min(in_ch - 1)
            };
            out.push(sample_at(src_ch, i));
        }
    }
    out
}

/// Dumb linear-interpolation resampler over f32 interleaved input.
/// Good enough for playback; not used for transcoding (see
/// `oxideav-audio-filter` for that).
pub fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32, channels: usize) -> Vec<f32> {
    if src.is_empty() || channels == 0 || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    let in_frames = src.len() / channels;
    if in_frames == 0 {
        return Vec::new();
    }
    let out_frames = (in_frames as u64 * dst_rate as u64 / src_rate as u64) as usize;
    let mut out = Vec::with_capacity(out_frames * channels);
    for i in 0..out_frames {
        let pos = (i as f64) * (src_rate as f64) / (dst_rate as f64);
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let idx_a = idx.min(in_frames - 1);
        let idx_b = (idx + 1).min(in_frames - 1);
        for c in 0..channels {
            let a = src[idx_a * channels + c];
            let b = src[idx_b * channels + c];
            out.push(a + (b - a) * frac);
        }
    }
    out
}
