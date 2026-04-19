//! Pure-Rust `OutputDriver` implementation using winit (windowing),
//! wgpu (rendering), and oxideav-sysaudio (audio — runtime-loaded
//! ALSA / PulseAudio / WASAPI / CoreAudio).
//!
//! The driver owns the winit `EventLoop` and a persistent
//! `ApplicationHandler`. `poll_events` pumps the event loop
//! non-blocking; key/close/resize events are mapped to `PlayerEvent`
//! and returned. Presenting a decoded frame renders through wgpu.

use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, Result, VideoFrame};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use crate::driver::{OutputDriver, PlayerEvent, SeekDir};
use crate::drivers::winit_audio::AudioOut;
use crate::drivers::winit_video::VideoRenderer;

pub struct WinitWgpuDriver {
    /// Taken out of the `Option` on first use. winit requires the event
    /// loop to be pumped from the same thread that created it.
    event_loop: Option<EventLoop<()>>,
    app: WinitApp,
    audio: AudioOut,
}

struct WinitApp {
    /// Initialized lazily inside `resumed()` — on macOS and some
    /// Wayland configurations the window handle is only valid after
    /// the platform signals `Resumed`.
    window: Option<Arc<Window>>,
    video: Option<VideoRenderer>,
    video_dims: Option<(u32, u32)>,
    /// Accumulated PlayerEvents — drained by `WinitWgpuDriver::poll_events`.
    pending: Vec<PlayerEvent>,
    /// Set by `ApplicationHandler::window_event` when the user closes
    /// the window; also translated into a `PlayerEvent::Quit`.
    quit: bool,
}

impl WinitWgpuDriver {
    pub fn new(sample_rate: u32, channels: u16, video_dims: Option<(u32, u32)>) -> Result<Self> {
        let event_loop =
            EventLoop::new().map_err(|e| Error::other(format!("winit: EventLoop::new: {e}")))?;
        let audio = AudioOut::new(sample_rate, channels)?;
        let app = WinitApp {
            window: None,
            video: None,
            video_dims,
            pending: Vec::new(),
            quit: false,
        };
        Ok(Self {
            event_loop: Some(event_loop),
            app,
            audio,
        })
    }
}

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
                // F toggles borderless fullscreen. Handled locally in
                // the driver — the player core has no business knowing
                // about window chrome. Check the logical key so AZERTY
                // / DVORAK / etc. layouts bind the letter F rather than
                // the physical position of "f" on a US keyboard.
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

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        // No-op: we drive redraws from the player loop via
        // `present_video`, not from winit timing. The default
        // behavior (no RedrawRequested) is what we want.
    }
}

fn map_key(ev: &KeyEvent) -> Option<PlayerEvent> {
    // Character bindings: match the *logical* key so the typed letter
    // is what binds, independent of keyboard layout. E.g. on AZERTY,
    // pressing the "a" key still types 'a' and the "q" key still types
    // 'q' — `PhysicalKey::KeyQ` would however be where "a" lives.
    if is_logical_char(ev, 'q') {
        return Some(PlayerEvent::Quit);
    }
    if is_logical_char(ev, '*') {
        return Some(PlayerEvent::VolumeDelta(5));
    }
    if is_logical_char(ev, '/') {
        return Some(PlayerEvent::VolumeDelta(-5));
    }

    // Named / positional keys: look up by logical `NamedKey` when set;
    // otherwise fall through to the physical code (handles some layouts
    // that don't report a logical name for arrows / PageUp etc.).
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

/// Check whether the event's *logical* key is the given character
/// (case-insensitive). Uses `key_without_modifiers` so Shift+Q also
/// matches 'q'.
fn is_logical_char(ev: &KeyEvent, expected: char) -> bool {
    let Key::Character(s) = &ev.logical_key else {
        return false;
    };
    s.chars()
        .next()
        .is_some_and(|c| c.eq_ignore_ascii_case(&expected))
}

impl OutputDriver for WinitWgpuDriver {
    fn present_video(&mut self, frame: &VideoFrame) -> Result<()> {
        if let Some(v) = self.app.video.as_mut() {
            v.render(frame)?;
        }
        Ok(())
    }

    fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        self.audio.queue_audio(frame)
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        let Some(event_loop) = self.event_loop.as_mut() else {
            return Vec::new();
        };
        // Non-blocking pump — drain whatever's queued and return.
        match event_loop.pump_app_events(Some(Duration::ZERO), &mut self.app) {
            PumpStatus::Continue => {}
            PumpStatus::Exit(_) => {
                self.app.quit = true;
                self.app.pending.push(PlayerEvent::Quit);
            }
        }
        std::mem::take(&mut self.app.pending)
    }

    fn master_clock_pos(&self) -> Duration {
        self.audio.master_clock_pos()
    }

    fn set_paused(&mut self, paused: bool) {
        self.audio.set_paused(paused);
    }

    fn set_volume(&mut self, vol: f32) {
        self.audio.set_volume(vol);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        self.audio.audio_queue_len_samples()
    }
}
