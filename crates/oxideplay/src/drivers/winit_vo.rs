//! winit + wgpu video output engine.
//!
//! Extracted from the old combined `winit_driver.rs`. Owns the winit
//! `EventLoop` plus a [`VideoRenderer`] (wgpu YUV→RGB) inside its own
//! `ApplicationHandler`. `poll_events` pumps the event loop
//! non-blocking; `present` renders through wgpu. Audio is a separate
//! concern handled by [`crate::drivers::sysaudio_ao::SysAudioEngine`]
//! (or any other [`AudioEngine`]).

use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{Error, Result, VideoFrame};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use crate::driver::{PlayerEvent, SeekDir};
use crate::drivers::engine::VideoEngine;
use crate::drivers::winit_video::VideoRenderer;

pub struct WinitVideoEngine {
    /// Taken out of the `Option` on first use. winit requires the
    /// event loop to be pumped from the same thread that created it.
    event_loop: Option<EventLoop<()>>,
    app: WinitApp,
    /// Requested content dimensions, remembered so `info()` can show
    /// them even before `resumed()` has run and the renderer exists.
    requested_dims: Option<(u32, u32)>,
}

struct WinitApp {
    /// Initialized lazily inside `resumed()` — on macOS and some
    /// Wayland configurations the window handle is only valid after
    /// the platform signals `Resumed`.
    window: Option<Arc<Window>>,
    video: Option<VideoRenderer>,
    video_dims: Option<(u32, u32)>,
    /// Accumulated PlayerEvents — drained by
    /// `WinitVideoEngine::poll_events`.
    pending: Vec<PlayerEvent>,
    /// Set by `window_event` when the user closes the window; also
    /// translated into a `PlayerEvent::Quit`.
    quit: bool,
}

impl WinitVideoEngine {
    pub fn new(video_dims: Option<(u32, u32)>) -> Result<Self> {
        let mut event_loop =
            EventLoop::new().map_err(|e| Error::other(format!("winit: EventLoop::new: {e}")))?;
        let mut app = WinitApp {
            window: None,
            video: None,
            video_dims,
            pending: Vec::new(),
            quit: false,
        };
        // Prime the event loop once so `resumed()` fires and the
        // window + wgpu renderer are actually constructed before we
        // hand the engine back. Without this, `info()` would only
        // know "winit is about to start"; the adapter/GPU summary
        // wouldn't be available until the first `poll_events` tick.
        // On X11/Wayland/Windows this fires Resumed on first pump;
        // on macOS a Resumed event is scheduled by AppKit too.
        let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
        Ok(Self {
            event_loop: Some(event_loop),
            app,
            requested_dims: video_dims,
        })
    }
}

// winit keeps a bunch of non-Send state on the app struct (window
// handles, event loops); we don't actually move this across threads,
// but `Box<dyn VideoEngine>` needs `Send` for `Composite`. In practice
// the player only touches the engine from the main thread — this impl
// is a marker to satisfy the trait, not a claim of real Send safety.
unsafe impl Send for WinitVideoEngine {}

impl ApplicationHandler for WinitApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = WindowAttributes::default().with_title("oxideplay");
        if let Some((w, h)) = self.video_dims {
            attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::new(w.max(1), h.max(1)));
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("oxideplay: create_window failed: {e}");
                self.quit = true;
                self.pending.push(PlayerEvent::Quit);
                return;
            }
        };
        let video = match VideoRenderer::new(window.clone()) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("oxideplay: wgpu init failed: {e}");
                self.quit = true;
                self.pending.push(PlayerEvent::Quit);
                None
            }
        };
        self.window = Some(window);
        self.video = video;
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.quit = true;
                self.pending.push(PlayerEvent::Quit);
            }
            WindowEvent::Resized(size) => {
                if let Some(v) = self.video.as_mut() {
                    v.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event: key, .. }
                if key.state == ElementState::Pressed && !key.repeat =>
            {
                // F toggles borderless fullscreen. Handled locally —
                // the player core has no business knowing about window
                // chrome. Check the logical key so AZERTY/DVORAK
                // layouts bind the letter F rather than the physical
                // position of "f" on a US keyboard.
                if is_logical_char(&key, 'f') {
                    if let Some(w) = self.window.as_ref() {
                        let next = if w.fullscreen().is_some() {
                            None
                        } else {
                            Some(Fullscreen::Borderless(None))
                        };
                        w.set_fullscreen(next);
                    }
                } else if let Some(pe) = map_key(&key) {
                    self.pending.push(pe);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {}
}

fn map_key(ev: &KeyEvent) -> Option<PlayerEvent> {
    if is_logical_char(ev, 'q') {
        return Some(PlayerEvent::Quit);
    }
    if is_logical_char(ev, '*') {
        return Some(PlayerEvent::VolumeDelta(5));
    }
    if is_logical_char(ev, '/') {
        return Some(PlayerEvent::VolumeDelta(-5));
    }
    if let Key::Named(named) = &ev.logical_key {
        match named {
            NamedKey::Escape => return Some(PlayerEvent::Quit),
            NamedKey::Space => return Some(PlayerEvent::TogglePause),
            NamedKey::ArrowLeft => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(10),
                    SeekDir::Back,
                ))
            }
            NamedKey::ArrowRight => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(10),
                    SeekDir::Forward,
                ))
            }
            NamedKey::ArrowUp => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(60),
                    SeekDir::Forward,
                ))
            }
            NamedKey::ArrowDown => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(60),
                    SeekDir::Back,
                ))
            }
            NamedKey::PageUp => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(600),
                    SeekDir::Forward,
                ))
            }
            NamedKey::PageDown => {
                return Some(PlayerEvent::SeekRelative(
                    Duration::from_secs(600),
                    SeekDir::Back,
                ))
            }
            _ => {}
        }
    }

    let code = match ev.physical_key {
        PhysicalKey::Code(c) => c,
        _ => return None,
    };
    match code {
        KeyCode::Escape => Some(PlayerEvent::Quit),
        KeyCode::Space => Some(PlayerEvent::TogglePause),
        KeyCode::ArrowLeft => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(10),
            SeekDir::Back,
        )),
        KeyCode::ArrowRight => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(10),
            SeekDir::Forward,
        )),
        KeyCode::ArrowUp => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(60),
            SeekDir::Forward,
        )),
        KeyCode::ArrowDown => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(60),
            SeekDir::Back,
        )),
        KeyCode::PageUp => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(600),
            SeekDir::Forward,
        )),
        KeyCode::PageDown => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(600),
            SeekDir::Back,
        )),
        KeyCode::NumpadMultiply => Some(PlayerEvent::VolumeDelta(5)),
        KeyCode::NumpadDivide => Some(PlayerEvent::VolumeDelta(-5)),
        _ => None,
    }
}

fn is_logical_char(ev: &KeyEvent, expected: char) -> bool {
    let Key::Character(s) = &ev.logical_key else {
        return false;
    };
    s.chars()
        .next()
        .is_some_and(|c| c.eq_ignore_ascii_case(&expected))
}

impl VideoEngine for WinitVideoEngine {
    fn present(&mut self, frame: &VideoFrame) -> Result<()> {
        if let Some(v) = self.app.video.as_mut() {
            v.render(frame)?;
        }
        Ok(())
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        let Some(event_loop) = self.event_loop.as_mut() else {
            return Vec::new();
        };
        match event_loop.pump_app_events(Some(Duration::ZERO), &mut self.app) {
            PumpStatus::Continue => {}
            PumpStatus::Exit(_) => {
                self.app.quit = true;
                self.app.pending.push(PlayerEvent::Quit);
            }
        }
        std::mem::take(&mut self.app.pending)
    }

    fn info(&self) -> String {
        // If the resumed() pump in `new()` had a chance to fire, we
        // know the real GPU adapter + surface; otherwise we fall
        // back to the requested content dims.
        match self.app.video.as_ref() {
            Some(v) => format!("winit+wgpu  {}", v.adapter_summary()),
            None => match self.requested_dims {
                Some((w, h)) => format!("winit+wgpu  requested: {w}x{h} (renderer not up yet)"),
                None => "winit+wgpu  (no dims, renderer not up yet)".into(),
            },
        }
    }
}
