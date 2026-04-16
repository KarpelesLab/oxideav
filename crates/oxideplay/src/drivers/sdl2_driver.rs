//! SDL2-backed audio + video output driver.
//!
//! Audio is the master clock: the SDL2 audio callback advances a sample
//! counter under a mutex, and `master_clock_pos` returns the total
//! samples consumed divided by the output rate.
//!
//! Video (when enabled) uses a YUV texture; incoming `VideoFrame`s are
//! converted on the fly to `Yuv420P` if needed and uploaded with
//! `update_yuv`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, PixelFormat, Result, SampleFormat, VideoFrame};
use sdl2::audio::{AudioCallback, AudioDevice, AudioSpecDesired};
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Mod};
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};
use sdl2::{AudioSubsystem, EventPump, Sdl, VideoSubsystem};

use crate::driver::{OutputDriver, PlayerEvent, SeekDir};

/// Shared state the audio callback reads from and the decode loop writes to.
struct AudioShared {
    /// Interleaved f32 samples pending playback.
    ring: Mutex<VecDeque<f32>>,
    /// Total samples consumed by the device since start (per channel).
    samples_played: AtomicU64,
    /// Audio output sample rate.
    sample_rate: u32,
    /// Channel count of the output device.
    channels: u16,
    /// 0.0..=1.0 volume.
    volume: Mutex<f32>,
    /// When true, callback emits silence (still advances the clock so the
    /// pause UI feels "correct" — but we override: see comments below).
    paused: AtomicBool,
}

struct SinkCallback {
    shared: Arc<AudioShared>,
}

impl AudioCallback for SinkCallback {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        let paused = self.shared.paused.load(Ordering::Acquire);
        if paused {
            for s in out.iter_mut() {
                *s = 0.0;
            }
            return;
        }
        let vol = *self.shared.volume.lock().unwrap();
        let mut ring = self.shared.ring.lock().unwrap();
        let mut produced: usize = 0;
        for dst in out.iter_mut() {
            match ring.pop_front() {
                Some(v) => {
                    *dst = v * vol;
                    produced += 1;
                }
                None => *dst = 0.0,
            }
        }
        // samples_played is in *frames per channel*; each iteration above
        // consumed one interleaved f32 (= one sample from one channel).
        let ch = self.shared.channels.max(1) as u64;
        self.shared
            .samples_played
            .fetch_add((produced as u64) / ch, Ordering::Release);
    }
}

/// Video sub-state that only exists when a window is open.
///
/// The `'static` lifetime on `Texture<'static>` is safe because we pair
/// it with the `TextureCreator` in the same struct; they're dropped
/// together when the `VideoState` is dropped. We use a raw pointer-like
/// trick: the texture creator is held via `Arc` in rust-sdl2 internally,
/// but `TextureCreator<WindowContext>` does *not* hand out a Clone, so
/// we own the creator and use unsafe to lie about the lifetime.
struct VideoState {
    canvas: Canvas<Window>,
    tex_creator: TextureCreator<WindowContext>,
    texture: Option<TextureBundle>,
}

/// A YUV texture paired with the dimensions it was built for.
///
/// SAFETY: `texture`'s real lifetime is tied to `tex_creator` inside
/// `VideoState`. We extend it to `'static` via transmute on creation
/// because we always drop the texture before the creator. See
/// `(re)create texture` in `present_video` for the tight coupling.
struct TextureBundle {
    texture: Texture<'static>,
    width: u32,
    height: u32,
}

pub struct Sdl2Driver {
    #[allow(dead_code)]
    sdl: Sdl,
    #[allow(dead_code)]
    audio_sub: AudioSubsystem,
    #[allow(dead_code)]
    video_sub: Option<VideoSubsystem>,
    event_pump: EventPump,
    audio_dev: AudioDevice<SinkCallback>,
    shared: Arc<AudioShared>,
    video: Option<VideoState>,
    output_sample_rate: u32,
    output_channels: u16,
}

impl Sdl2Driver {
    /// Build a driver. If `video` is `Some((w, h))`, a window of that size
    /// is created. Audio is always initialised.
    pub fn new(
        audio_sample_rate: u32,
        audio_channels: u16,
        video: Option<(u32, u32)>,
    ) -> Result<Self> {
        let sdl = sdl2::init().map_err(Error::other)?;
        let audio_sub = sdl.audio().map_err(Error::other)?;

        let channels = audio_channels.clamp(1, 2);
        let shared = Arc::new(AudioShared {
            ring: Mutex::new(VecDeque::with_capacity(
                (audio_sample_rate as usize) * channels as usize,
            )),
            samples_played: AtomicU64::new(0),
            sample_rate: audio_sample_rate,
            channels,
            volume: Mutex::new(1.0),
            paused: AtomicBool::new(false),
        });

        let desired = AudioSpecDesired {
            freq: Some(audio_sample_rate as i32),
            channels: Some(channels as u8),
            samples: Some(1024),
        };
        let shared_cb = shared.clone();
        let audio_dev = audio_sub
            .open_playback(None, &desired, move |_spec| SinkCallback {
                shared: shared_cb,
            })
            .map_err(Error::other)?;
        audio_dev.resume();

        let (video_sub, video, event_pump) = match video {
            Some((w, h)) => {
                let vs = sdl.video().map_err(Error::other)?;
                let window = vs
                    .window("oxideplay", w.max(1), h.max(1))
                    .position_centered()
                    .resizable()
                    .build()
                    .map_err(|e| Error::other(e.to_string()))?;
                let canvas = window
                    .into_canvas()
                    .build()
                    .map_err(|e| Error::other(e.to_string()))?;
                let tex_creator = canvas.texture_creator();
                let ep = sdl.event_pump().map_err(Error::other)?;
                (
                    Some(vs),
                    Some(VideoState {
                        canvas,
                        tex_creator,
                        texture: None,
                    }),
                    ep,
                )
            }
            None => {
                // Even audio-only mode needs an event pump for the SDL2
                // audio subsystem to pump its queue on some platforms.
                let ep = sdl.event_pump().map_err(Error::other)?;
                (None, None, ep)
            }
        };

        Ok(Self {
            sdl,
            audio_sub,
            video_sub,
            event_pump,
            audio_dev,
            shared,
            video,
            output_sample_rate: audio_sample_rate,
            output_channels: channels,
        })
    }
}

fn to_f32_interleaved(frame: &AudioFrame, out_channels: u16) -> Vec<f32> {
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

/// Map one of our PixelFormat variants to an SDL_PIXELFORMAT enum.
fn sdl_pixel_format(fmt: PixelFormat) -> PixelFormatEnum {
    match fmt {
        PixelFormat::Yuv420P => PixelFormatEnum::IYUV,
        PixelFormat::Yuv422P | PixelFormat::Yuv444P => PixelFormatEnum::IYUV, // converted
        PixelFormat::Rgb24 => PixelFormatEnum::RGB24,
        PixelFormat::Rgba => PixelFormatEnum::RGBA32,
        PixelFormat::Gray8 => PixelFormatEnum::IYUV,
    }
}

/// Subsample YUV422P or YUV444P planes down to YUV420P planes.
/// Output stride for Y = w, for U/V = w/2 (even w required; odd w rounded down).
fn to_yuv420p(frame: &VideoFrame) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = frame.width as usize;
    let h = frame.height as usize;
    match frame.format {
        PixelFormat::Yuv420P => {
            let y = plane_tight(&frame.planes[0].data, frame.planes[0].stride, w, h);
            let u = plane_tight(&frame.planes[1].data, frame.planes[1].stride, w / 2, h / 2);
            let v = plane_tight(&frame.planes[2].data, frame.planes[2].stride, w / 2, h / 2);
            (y, u, v)
        }
        PixelFormat::Yuv422P => {
            // 4:2:2 → 4:2:0 by vertical 2× subsample on chroma.
            let y = plane_tight(&frame.planes[0].data, frame.planes[0].stride, w, h);
            let u_src = &frame.planes[1];
            let v_src = &frame.planes[2];
            let u = downsample_vertical(u_src, w / 2, h);
            let v = downsample_vertical(v_src, w / 2, h);
            (y, u, v)
        }
        PixelFormat::Yuv444P => {
            let y = plane_tight(&frame.planes[0].data, frame.planes[0].stride, w, h);
            // 4:4:4 → 4:2:0 = 2× horizontal + 2× vertical subsample.
            let u = downsample_2x2(&frame.planes[1], w, h);
            let v = downsample_2x2(&frame.planes[2], w, h);
            (y, u, v)
        }
        PixelFormat::Gray8 => {
            let y = plane_tight(&frame.planes[0].data, frame.planes[0].stride, w, h);
            let chroma = vec![128u8; (w / 2) * (h / 2)];
            (y, chroma.clone(), chroma)
        }
        _ => {
            // Fallback: build a flat grey image.
            let y = vec![128u8; w * h];
            let chroma = vec![128u8; (w / 2) * (h / 2)];
            (y, chroma.clone(), chroma)
        }
    }
}

fn plane_tight(src: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    if stride == w {
        return src[..w * h.min(src.len() / stride.max(1))].to_vec();
    }
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        let off = row * stride;
        if off + w > src.len() {
            break;
        }
        out.extend_from_slice(&src[off..off + w]);
    }
    out
}

fn downsample_vertical(plane: &oxideav_core::VideoPlane, out_w: usize, in_h: usize) -> Vec<u8> {
    let out_h = in_h / 2;
    let mut out = Vec::with_capacity(out_w * out_h);
    for row in 0..out_h {
        let src_row = row * 2;
        let off = src_row * plane.stride;
        if off + out_w > plane.data.len() {
            break;
        }
        out.extend_from_slice(&plane.data[off..off + out_w]);
    }
    out
}

fn downsample_2x2(plane: &oxideav_core::VideoPlane, in_w: usize, in_h: usize) -> Vec<u8> {
    let out_w = in_w / 2;
    let out_h = in_h / 2;
    let mut out = Vec::with_capacity(out_w * out_h);
    for row in 0..out_h {
        let src_row = row * 2;
        let off = src_row * plane.stride;
        if off + in_w > plane.data.len() {
            break;
        }
        for col in 0..out_w {
            let src_col = col * 2;
            out.push(plane.data[off + src_col]);
        }
    }
    out
}

impl OutputDriver for Sdl2Driver {
    fn present_video(&mut self, frame: &VideoFrame) -> Result<()> {
        let Some(v) = self.video.as_mut() else {
            return Ok(());
        };
        let w = frame.width;
        let h = frame.height;
        if w == 0 || h == 0 {
            return Ok(());
        }

        // (Re)create the texture if dimensions changed.
        let need_new = match &v.texture {
            Some(tb) => tb.width != w || tb.height != h,
            None => true,
        };
        if need_new {
            let tex = v
                .tex_creator
                .create_texture_streaming(sdl_pixel_format(frame.format), w, h)
                .map_err(|e| Error::other(e.to_string()))?;
            // SAFETY: The texture borrows from `v.tex_creator`. We store
            // both together in `VideoState` and only drop the texture when
            // the struct is torn down (or when replacing on resize). The
            // lifetime is effectively the same as `v`, but we can't name
            // that in a struct field, so we extend to `'static` and
            // guarantee via the struct layout that the creator outlives
            // the texture.
            let tex_static: Texture<'static> = unsafe { std::mem::transmute(tex) };
            v.texture = Some(TextureBundle {
                texture: tex_static,
                width: w,
                height: h,
            });
        }

        let (y, u, vplane) = to_yuv420p(frame);
        let yp = w as usize;
        let up = (w / 2) as usize;
        let vp = (w / 2) as usize;
        if let Some(tb) = v.texture.as_mut() {
            tb.texture
                .update_yuv(None, &y, yp, &u, up, &vplane, vp)
                .map_err(|e| Error::other(e.to_string()))?;
            v.canvas.clear();
            v.canvas
                .copy(&tb.texture, None, None)
                .map_err(Error::other)?;
            v.canvas.present();
        }
        Ok(())
    }

    fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.samples == 0 {
            return Ok(());
        }
        let buf = to_f32_interleaved(frame, self.output_channels);
        // Simple sample-rate adaptation: if rates differ, linearly
        // resample per-channel. Avoids a full resampler dep for v1.
        let final_buf = if frame.sample_rate == self.output_sample_rate {
            buf
        } else {
            resample_linear(
                &buf,
                frame.sample_rate,
                self.output_sample_rate,
                self.output_channels as usize,
            )
        };
        let mut ring = self.shared.ring.lock().unwrap();
        ring.extend(final_buf);
        Ok(())
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        let mut out = Vec::new();
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => out.push(PlayerEvent::Quit),
                Event::KeyDown {
                    keycode: Some(kc),
                    keymod,
                    ..
                } => {
                    if let Some(ev) = map_sdl_key(kc, keymod) {
                        out.push(ev);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn master_clock_pos(&self) -> Duration {
        let played = self.shared.samples_played.load(Ordering::Acquire);
        let sr = self.shared.sample_rate.max(1) as u64;
        let secs = played / sr;
        let frac = played % sr;
        let nanos = (frac * 1_000_000_000) / sr;
        Duration::new(secs, nanos as u32)
    }

    fn set_paused(&mut self, paused: bool) {
        self.shared.paused.store(paused, Ordering::Release);
        if paused {
            self.audio_dev.pause();
        } else {
            self.audio_dev.resume();
        }
    }

    fn set_volume(&mut self, vol: f32) {
        *self.shared.volume.lock().unwrap() = vol.clamp(0.0, 1.0);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        let ring = self.shared.ring.lock().unwrap();
        (ring.len() as u64) / (self.output_channels.max(1) as u64)
    }
}

fn map_sdl_key(kc: Keycode, keymod: Mod) -> Option<PlayerEvent> {
    let shift = keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD);
    match kc {
        Keycode::Q | Keycode::Escape => Some(PlayerEvent::Quit),
        Keycode::Space => Some(PlayerEvent::TogglePause),
        Keycode::Left => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Back))
        }
        Keycode::Right => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Forward))
        }
        Keycode::Up => Some(PlayerEvent::VolumeDelta(5)),
        Keycode::Down => Some(PlayerEvent::VolumeDelta(-5)),
        _ => None,
    }
}

/// Dumb linear-interpolation resampler, interleaved.
fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32, channels: usize) -> Vec<f32> {
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
