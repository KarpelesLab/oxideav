//! SDL2 video output engine. Owns an SDL window + renderer, streams
//! decoded video frames into a YUV texture, and pumps SDL events
//! (keyboard + close-requested) into [`PlayerEvent`]s.
//!
//! The SDL2 library is loaded + ref-counted through
//! [`crate::drivers::sdl2_root`]; this file never calls `SDL_Init` or
//! `SDL_Quit` directly. The audio side is a separate engine
//! ([`crate::drivers::sdl2_audio::SdlAudioEngine`]); the two can be
//! composed with each other or with non-SDL counterparts via
//! [`crate::drivers::engine::Composite`].

use std::ffi::{c_int, c_void, CString};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{Error, PixelFormat, Result, VideoFrame};

use crate::driver::{PlayerEvent, SeekDir};
use crate::drivers::engine::VideoEngine;
use crate::drivers::sdl2_loader::{self as ldr, SDL_Event, Sdl2Lib};
use crate::drivers::sdl2_root::{self, SubsystemGuard};
use crate::drivers::video_convert::to_yuv420p;

struct TextureBundle {
    texture: *mut c_void,
    width: u32,
    height: u32,
}

pub struct SdlVideoEngine {
    lib: Arc<Sdl2Lib>,
    // Dropped after window/renderer/texture → SDL_QuitSubSystem runs
    // last, matching how SDL itself wants teardown ordered.
    _guard: SubsystemGuard,
    window: *mut c_void,
    renderer: *mut c_void,
    texture: Option<TextureBundle>,
    /// Initial window size — frozen at creation for `info()`; the
    /// real surface size is queried lazily via
    /// `SDL_GetRendererOutputSize` on each present.
    initial_dims: (u32, u32),
}

// The raw pointers never leave this thread in the current player
// design (the main loop calls `present`), but we declare Send so
// `Composite` can carry us as `Box<dyn VideoEngine>`.
unsafe impl Send for SdlVideoEngine {}

impl SdlVideoEngine {
    pub fn new(dims: (u32, u32)) -> Result<Self> {
        let guard = sdl2_root::acquire(sdl2_root::VIDEO_MASK)?;
        let lib = guard.lib().clone();
        let (w, h) = (dims.0.max(1), dims.1.max(1));
        let title = CString::new("oxideplay").unwrap();
        // SAFETY: title is NUL-terminated and lives for the call.
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
        // -1 = first supporting driver; 0 = no extra flags (SDL picks
        // hardware acceleration when it can).
        let renderer = unsafe { (lib.SDL_CreateRenderer)(window, -1, 0) };
        if renderer.is_null() {
            let err = lib.last_error();
            unsafe { (lib.SDL_DestroyWindow)(window) };
            return Err(Error::other(format!("SDL_CreateRenderer failed: {err}")));
        }
        Ok(Self {
            lib,
            _guard: guard,
            window,
            renderer,
            texture: None,
            initial_dims: (w, h),
        })
    }

    fn toggle_fullscreen(&mut self) {
        // SAFETY: window is non-null for the engine's lifetime.
        let flags = unsafe { (self.lib.SDL_GetWindowFlags)(self.window) };
        let next = if flags & ldr::SDL_WINDOW_FULLSCREEN_DESKTOP != 0 {
            0
        } else {
            ldr::SDL_WINDOW_FULLSCREEN_DESKTOP
        };
        unsafe {
            (self.lib.SDL_SetWindowFullscreen)(self.window, next);
        }
    }
}

impl Drop for SdlVideoEngine {
    fn drop(&mut self) {
        if let Some(tb) = self.texture.take() {
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

impl VideoEngine for SdlVideoEngine {
    fn present(&mut self, frame: &VideoFrame) -> Result<()> {
        let w = frame.width;
        let h = frame.height;
        if w == 0 || h == 0 {
            return Ok(());
        }
        let need_new = match &self.texture {
            Some(tb) => tb.width != w || tb.height != h,
            None => true,
        };
        if need_new {
            if let Some(old) = self.texture.take() {
                unsafe { (self.lib.SDL_DestroyTexture)(old.texture) };
            }
            let tex = unsafe {
                (self.lib.SDL_CreateTexture)(
                    self.renderer,
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
            self.texture = Some(TextureBundle {
                texture: tex,
                width: w,
                height: h,
            });
        }

        let (yp_buf, up_buf, vp_buf) = to_yuv420p(frame);
        let yp = w as c_int;
        let up = (w / 2) as c_int;
        let vp = (w / 2) as c_int;
        if let Some(tb) = self.texture.as_ref() {
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
            let mut out_w: c_int = 0;
            let mut out_h: c_int = 0;
            unsafe {
                (self.lib.SDL_GetRendererOutputSize)(self.renderer, &mut out_w, &mut out_h);
            }
            let dst = fit_rect(w as i32, h as i32, out_w as i32, out_h as i32);
            unsafe {
                (self.lib.SDL_RenderClear)(self.renderer);
                (self.lib.SDL_RenderCopy)(
                    self.renderer,
                    tb.texture,
                    ptr::null(),
                    &dst as *const _,
                );
                (self.lib.SDL_RenderPresent)(self.renderer);
            }
        }
        Ok(())
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        let mut out = Vec::new();
        loop {
            let mut ev = SDL_Event::zeroed();
            let got = unsafe { (self.lib.SDL_PollEvent)(&mut ev as *mut _) };
            if got == 0 {
                break;
            }
            match ev.r#type {
                ldr::SDL_QUIT => out.push(PlayerEvent::Quit),
                ldr::SDL_KEYDOWN => {
                    let key = unsafe { ev.as_key() };
                    if key.keysym.sym == ldr::SDLK_f {
                        self.toggle_fullscreen();
                    } else if let Some(pe) = map_sdl_key(key.keysym.sym, key.keysym.r#mod) {
                        out.push(pe);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn info(&self) -> String {
        // SDL2 hides the renderer choice behind its own heuristic
        // (SDL_GetRendererInfo would tell us "opengl" / "direct3d" /
        // "software" / "metal" but the loader doesn't bind that
        // entry point today). Keep the banner honest: just the
        // backend name + initial window size; the actual output is
        // always IYUV for our uploaded planes.
        let (w, h) = self.initial_dims;
        format!("sdl2  window: {w}x{h}  upload: IYUV (4:2:0)")
    }
}

fn sdl_pixel_format(fmt: PixelFormat) -> u32 {
    match fmt {
        PixelFormat::Yuv420P => ldr::SDL_PIXELFORMAT_IYUV,
        PixelFormat::Yuv422P | PixelFormat::Yuv444P => ldr::SDL_PIXELFORMAT_IYUV,
        PixelFormat::Rgb24 => ldr::SDL_PIXELFORMAT_RGB24,
        PixelFormat::Rgba => ldr::SDL_PIXELFORMAT_RGBA32,
        PixelFormat::Gray8 => ldr::SDL_PIXELFORMAT_IYUV,
        _ => ldr::SDL_PIXELFORMAT_IYUV,
    }
}

fn fit_rect(src_w: i32, src_h: i32, dst_w: i32, dst_h: i32) -> ldr::SDL_Rect {
    if src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 {
        return ldr::SDL_Rect {
            x: 0,
            y: 0,
            w: dst_w.max(0),
            h: dst_h.max(0),
        };
    }
    let src_ar = src_w as f64 / src_h as f64;
    let dst_ar = dst_w as f64 / dst_h as f64;
    let (w, h) = if src_ar > dst_ar {
        let w = dst_w;
        let h = (dst_w as f64 / src_ar).round() as i32;
        (w, h)
    } else {
        let h = dst_h;
        let w = (dst_h as f64 * src_ar).round() as i32;
        (w, h)
    };
    let x = (dst_w - w) / 2;
    let y = (dst_h - h) / 2;
    ldr::SDL_Rect { x, y, w, h }
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
