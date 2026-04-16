//! SDL2-backed audio + video output driver, talking to a runtime-loaded
//! libSDL2 (`crate::drivers::sdl2_loader`) instead of linking against
//! `sdl2-sys` at build time.
//!
//! Audio uses the SDL queue API (`SDL_QueueAudio` / `SDL_GetQueuedAudioSize`)
//! rather than callbacks, which means we don't have to thread a Rust
//! callback through the FFI boundary. The "master clock" is derived
//! from `(total_queued_bytes - currently_queued_bytes) / bytes_per_second`.
//!
//! Video (when enabled) uses a YUV streaming texture; incoming
//! `VideoFrame`s are converted on the fly to `Yuv420P` if needed and
//! uploaded with `SDL_UpdateYUVTexture`.

use std::ffi::{c_int, c_void, CString};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, PixelFormat, Result, SampleFormat, VideoFrame};

use crate::driver::{OutputDriver, PlayerEvent, SeekDir};
use crate::drivers::sdl2_loader::{
    self as ldr, SDL_AudioDeviceID, SDL_AudioSpec, SDL_Event, Sdl2Lib,
};

/// RAII guard around `SDL_Init` / `SDL_Quit`.
struct SdlGuard {
    lib: Arc<Sdl2Lib>,
}

impl Drop for SdlGuard {
    fn drop(&mut self) {
        // SAFETY: matches the SDL_Init done in `Sdl2Driver::new`.
        unsafe { (self.lib.SDL_Quit)() };
    }
}

/// Video sub-state that only exists when a window is open.
struct VideoState {
    lib: Arc<Sdl2Lib>,
    window: *mut c_void,
    renderer: *mut c_void,
    /// Currently bound texture (if any) plus its dimensions.
    texture: Option<TextureBundle>,
}

impl Drop for VideoState {
    fn drop(&mut self) {
        // Texture goes first, then renderer, then window — same order
        // rust-sdl2 enforces internally and what SDL2 itself documents.
        if let Some(tb) = self.texture.take() {
            // SAFETY: pointer was returned by SDL_CreateTexture; SDL is
            // still loaded because `lib` is held by Arc.
            unsafe { (self.lib.SDL_DestroyTexture)(tb.texture) };
        }
        if !self.renderer.is_null() {
            unsafe { (self.lib.SDL_DestroyRenderer)(self.renderer) };
        }
        if !self.window.is_null() {
            unsafe { (self.lib.SDL_DestroyWindow)(self.window) };
        }
    }
}

struct TextureBundle {
    texture: *mut c_void,
    width: u32,
    height: u32,
}

/// Audio device + bookkeeping for the queue-based clock.
struct AudioState {
    lib: Arc<Sdl2Lib>,
    dev: SDL_AudioDeviceID,
    /// Output sample rate.
    sample_rate: u32,
    /// Total bytes ever pushed to SDL via SDL_QueueAudio. Used to
    /// derive samples_played = (total_queued - still_queued) / bps.
    total_queued_bytes: u64,
    /// Bytes per output sample frame (channels * sizeof(f32) = ch * 4).
    bytes_per_frame: u32,
    /// Current playback volume (0.0..=1.0).
    volume: f32,
    /// True if SDL_PauseAudioDevice was called with pause_on=1.
    paused: bool,
}

impl Drop for AudioState {
    fn drop(&mut self) {
        if self.dev != 0 {
            // SAFETY: dev was returned from SDL_OpenAudioDevice.
            unsafe { (self.lib.SDL_CloseAudioDevice)(self.dev) };
        }
    }
}

pub struct Sdl2Driver {
    lib: Arc<Sdl2Lib>,
    /// Holds the SDL_Init, dropped *last* (after audio + video state).
    _guard: SdlGuard,
    audio: AudioState,
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
        let lib = Arc::new(Sdl2Lib::try_load().map_err(|e| {
            Error::other(format!(
                "SDL2 library not found at runtime — install libSDL2 to enable audio/video output ({e})"
            ))
        })?);

        let init_flags = ldr::SDL_INIT_AUDIO
            | ldr::SDL_INIT_EVENTS
            | if video.is_some() {
                ldr::SDL_INIT_VIDEO
            } else {
                0
            };

        // SAFETY: SDL_Init is the canonical initialisation entry point
        // and is safe to call from the main thread.
        let rc = unsafe { (lib.SDL_Init)(init_flags) };
        if rc != 0 {
            return Err(Error::other(format!(
                "SDL_Init failed: {}",
                lib.last_error()
            )));
        }
        let guard = SdlGuard { lib: lib.clone() };

        let channels = audio_channels.clamp(1, 2);
        let bytes_per_frame = (channels as u32) * 4; // f32 samples
        let desired = SDL_AudioSpec {
            freq: audio_sample_rate as c_int,
            format: ldr::AUDIO_F32,
            channels: channels as u8,
            silence: 0,
            samples: 1024,
            padding: 0,
            size: 0,
            // None = use the queue API (SDL_QueueAudio).
            callback: None,
            userdata: ptr::null_mut(),
        };
        let mut obtained: SDL_AudioSpec = SDL_AudioSpec {
            freq: 0,
            format: 0,
            channels: 0,
            silence: 0,
            samples: 0,
            padding: 0,
            size: 0,
            callback: None,
            userdata: ptr::null_mut(),
        };

        // SAFETY: `desired` and `&mut obtained` outlive the call. NULL
        // device name = default playback device. allowed_changes=0 forces
        // SDL to convert internally if the device differs.
        let dev = unsafe {
            (lib.SDL_OpenAudioDevice)(
                ptr::null(),
                0,
                &desired as *const _,
                &mut obtained as *mut _,
                0,
            )
        };
        if dev == 0 {
            return Err(Error::other(format!(
                "SDL_OpenAudioDevice failed: {}",
                lib.last_error()
            )));
        }
        // Resume = unpause (pause_on=0).
        unsafe { (lib.SDL_PauseAudioDevice)(dev, 0) };

        let audio = AudioState {
            lib: lib.clone(),
            dev,
            sample_rate: audio_sample_rate,
            total_queued_bytes: 0,
            bytes_per_frame,
            volume: 1.0,
            paused: false,
        };

        let video = match video {
            Some((w, h)) => Some(open_video(&lib, w.max(1), h.max(1))?),
            None => None,
        };

        Ok(Self {
            lib,
            _guard: guard,
            audio,
            video,
            output_sample_rate: audio_sample_rate,
            output_channels: channels,
        })
    }
}

fn open_video(lib: &Arc<Sdl2Lib>, w: u32, h: u32) -> Result<VideoState> {
    let title = CString::new("oxideplay").unwrap();
    // SAFETY: title is NUL-terminated and lives for the call. SDL2 copies it.
    let window = unsafe {
        (lib.SDL_CreateWindow)(
            title.as_ptr(),
            ldr::SDL_WINDOWPOS_CENTERED,
            ldr::SDL_WINDOWPOS_CENTERED,
            w as c_int,
            h as c_int,
            ldr::SDL_WINDOW_RESIZABLE,
        )
    };
    if window.is_null() {
        return Err(Error::other(format!(
            "SDL_CreateWindow failed: {}",
            lib.last_error()
        )));
    }
    // -1 = "first driver supporting the requested flags"; 0 = no extra flags
    // (lets SDL pick hardware acceleration when available).
    // SAFETY: window was just created; SDL is initialised.
    let renderer = unsafe { (lib.SDL_CreateRenderer)(window, -1, 0) };
    if renderer.is_null() {
        let err = lib.last_error();
        unsafe { (lib.SDL_DestroyWindow)(window) };
        return Err(Error::other(format!("SDL_CreateRenderer failed: {err}")));
    }

    Ok(VideoState {
        lib: lib.clone(),
        window,
        renderer,
        texture: None,
    })
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

/// Map one of our PixelFormat variants to an SDL2 pixel-format int.
fn sdl_pixel_format(fmt: PixelFormat) -> u32 {
    match fmt {
        PixelFormat::Yuv420P => ldr::SDL_PIXELFORMAT_IYUV,
        PixelFormat::Yuv422P | PixelFormat::Yuv444P => ldr::SDL_PIXELFORMAT_IYUV, // converted
        PixelFormat::Rgb24 => ldr::SDL_PIXELFORMAT_RGB24,
        PixelFormat::Rgba => ldr::SDL_PIXELFORMAT_RGBA32,
        PixelFormat::Gray8 => ldr::SDL_PIXELFORMAT_IYUV,
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
            // Drop the old one first so SDL can release its GPU side.
            if let Some(old) = v.texture.take() {
                unsafe { (self.lib.SDL_DestroyTexture)(old.texture) };
            }
            // SAFETY: renderer is non-null and from this lib.
            let tex = unsafe {
                (self.lib.SDL_CreateTexture)(
                    v.renderer,
                    sdl_pixel_format(frame.format),
                    ldr::SDL_TEXTUREACCESS_STREAMING,
                    w as c_int,
                    h as c_int,
                )
            };
            if tex.is_null() {
                return Err(Error::other(format!(
                    "SDL_CreateTexture failed: {}",
                    self.lib.last_error()
                )));
            }
            v.texture = Some(TextureBundle {
                texture: tex,
                width: w,
                height: h,
            });
        }

        let (yp_buf, up_buf, vp_buf) = to_yuv420p(frame);
        let yp = w as c_int;
        let up = (w / 2) as c_int;
        let vp = (w / 2) as c_int;
        if let Some(tb) = v.texture.as_ref() {
            // SAFETY: rect=NULL means update the whole texture; pitches
            // match the buffer widths we just produced.
            let rc = unsafe {
                (self.lib.SDL_UpdateYUVTexture)(
                    tb.texture,
                    ptr::null(),
                    yp_buf.as_ptr(),
                    yp,
                    up_buf.as_ptr(),
                    up,
                    vp_buf.as_ptr(),
                    vp,
                )
            };
            if rc != 0 {
                return Err(Error::other(format!(
                    "SDL_UpdateYUVTexture failed: {}",
                    self.lib.last_error()
                )));
            }
            unsafe { (self.lib.SDL_RenderClear)(v.renderer) };
            unsafe {
                (self.lib.SDL_RenderCopy)(v.renderer, tb.texture, ptr::null(), ptr::null());
            }
            unsafe { (self.lib.SDL_RenderPresent)(v.renderer) };
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
        let mut final_buf = if frame.sample_rate == self.output_sample_rate {
            buf
        } else {
            resample_linear(
                &buf,
                frame.sample_rate,
                self.output_sample_rate,
                self.output_channels as usize,
            )
        };
        // Apply volume gain in-place before handing the buffer to SDL —
        // we no longer have a callback, so this is the only place it can
        // happen.
        let vol = self.audio.volume;
        if (vol - 1.0).abs() > f32::EPSILON {
            for s in final_buf.iter_mut() {
                *s *= vol;
            }
        }

        let byte_len = (final_buf.len() * std::mem::size_of::<f32>()) as u32;
        // SAFETY: data points into `final_buf`, valid for the call. SDL
        // copies the bytes synchronously into its internal queue.
        let rc = unsafe {
            (self.lib.SDL_QueueAudio)(
                self.audio.dev,
                final_buf.as_ptr() as *const c_void,
                byte_len,
            )
        };
        if rc != 0 {
            return Err(Error::other(format!(
                "SDL_QueueAudio failed: {}",
                self.lib.last_error()
            )));
        }
        self.audio.total_queued_bytes += byte_len as u64;
        Ok(())
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        let mut out = Vec::new();
        loop {
            let mut ev = SDL_Event::zeroed();
            // SAFETY: ev is valid; SDL writes into the passed-in struct.
            let got = unsafe { (self.lib.SDL_PollEvent)(&mut ev as *mut _) };
            if got == 0 {
                break;
            }
            match ev.r#type {
                ldr::SDL_QUIT => out.push(PlayerEvent::Quit),
                ldr::SDL_KEYDOWN => {
                    // SAFETY: type-discriminant matches the union variant.
                    let key = unsafe { ev.as_key() };
                    if let Some(pe) = map_sdl_key(key.keysym.sym, key.keysym.r#mod) {
                        out.push(pe);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn master_clock_pos(&self) -> Duration {
        // Played frames = (total_queued_bytes - currently_queued_bytes) / bytes_per_frame.
        // SAFETY: dev is valid for as long as `audio` is alive.
        let queued = unsafe { (self.lib.SDL_GetQueuedAudioSize)(self.audio.dev) } as u64;
        let bpf = self.audio.bytes_per_frame.max(1) as u64;
        let played_frames = self.audio.total_queued_bytes.saturating_sub(queued) / bpf;
        let sr = self.audio.sample_rate.max(1) as u64;
        let secs = played_frames / sr;
        let frac = played_frames % sr;
        let nanos = (frac * 1_000_000_000) / sr;
        Duration::new(secs, nanos as u32)
    }

    fn set_paused(&mut self, paused: bool) {
        if self.audio.paused == paused {
            return;
        }
        self.audio.paused = paused;
        // SAFETY: dev is valid.
        unsafe {
            (self.lib.SDL_PauseAudioDevice)(self.audio.dev, if paused { 1 } else { 0 });
        }
    }

    fn set_volume(&mut self, vol: f32) {
        self.audio.volume = vol.clamp(0.0, 1.0);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        // SAFETY: dev is valid.
        let queued_bytes = unsafe { (self.lib.SDL_GetQueuedAudioSize)(self.audio.dev) } as u64;
        let bpf = self.audio.bytes_per_frame.max(1) as u64;
        queued_bytes / bpf
    }
}

fn map_sdl_key(sym: i32, modmask: u16) -> Option<PlayerEvent> {
    let shift = (modmask & ldr::KMOD_SHIFT) != 0;
    match sym {
        x if x == ldr::SDLK_q || x == ldr::SDLK_ESCAPE => Some(PlayerEvent::Quit),
        x if x == ldr::SDLK_SPACE => Some(PlayerEvent::TogglePause),
        x if x == ldr::SDLK_LEFT => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Back))
        }
        x if x == ldr::SDLK_RIGHT => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Forward))
        }
        x if x == ldr::SDLK_UP => Some(PlayerEvent::VolumeDelta(5)),
        x if x == ldr::SDLK_DOWN => Some(PlayerEvent::VolumeDelta(-5)),
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
