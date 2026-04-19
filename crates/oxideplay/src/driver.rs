//! Output-driver abstraction.
//!
//! An `OutputDriver` is the thing that actually puts decoded audio + video
//! on the user's speakers / screen. The player core pushes frames to it,
//! polls events from it, and asks it what the current master-clock position
//! is (which is usually driven by the audio output rate).

use oxideav_core::{AudioFrame, Result, VideoFrame};
use std::time::Duration;

/// A user action emitted by the TUI, the window, or any other input surface.
///
/// The player merges these into a single event queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerEvent {
    /// Quit the player.
    Quit,
    /// Toggle paused / playing.
    TogglePause,
    /// Seek forward or backward by a duration.
    SeekRelative(Duration, SeekDir),
    /// Nudge the volume up or down (percentage points, -100..=100).
    VolumeDelta(i32),
}

/// Direction for relative seeks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekDir {
    Forward,
    Back,
}

/// Abstraction over "the thing that draws frames + plays samples".
///
/// For the SDL2 implementation, both audio and video live in the same
/// driver. A future ALSA-only or wgpu driver could implement this trait
/// too without the player caring.
pub trait OutputDriver {
    /// Present one decoded video frame. A headless/audio-only driver
    /// should simply discard the frame.
    fn present_video(&mut self, frame: &VideoFrame) -> Result<()>;

    /// Queue a decoded audio frame for playback. The driver is expected
    /// to own its audio callback and consume this buffer as samples are
    /// needed by the output device.
    fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()>;

    /// Poll any user-input events (window keys, gamepad buttons, etc.).
    fn poll_events(&mut self) -> Vec<PlayerEvent>;

    /// Current position of the master clock. Implementations should use the
    /// audio clock when available; otherwise a wall-clock-based monotonic
    /// counter is acceptable.
    fn master_clock_pos(&self) -> Duration;

    /// Pause / resume both audio and video output.
    fn set_paused(&mut self, paused: bool);

    /// Set the output volume (0.0 = mute, 1.0 = unity gain).
    fn set_volume(&mut self, vol: f32);

    /// Approximate number of samples currently buffered in the audio queue.
    /// The player uses this to throttle decoding so we don't run ahead of
    /// the output and starve memory.
    fn audio_queue_len_samples(&self) -> u64 {
        0
    }

    /// Remaining samples the audio backend can accept before it starts
    /// dropping. See `AudioEngine::audio_headroom_samples`.
    fn audio_headroom_samples(&self) -> u64 {
        u64::MAX
    }

    /// Per-engine one-liner descriptions for the startup banner —
    /// `(video, audio)`. `None` on either side means that engine is
    /// disabled (e.g. `--vo null`). Default returns `(None, None)`;
    /// `Composite` overrides to surface the real engines' `info()`.
    fn engine_info(&self) -> (Option<String>, Option<String>) {
        (None, None)
    }
}

/// Blanket impl so `Box<dyn OutputDriver>` can stand in for a concrete
/// `D: OutputDriver` everywhere the player is generic. This lets `main`
/// pick between SDL2 and winit at runtime by building different
/// boxed drivers and passing them into the same `Player<Box<dyn _>>`.
impl<D: OutputDriver + ?Sized> OutputDriver for Box<D> {
    fn present_video(&mut self, frame: &VideoFrame) -> Result<()> {
        (**self).present_video(frame)
    }
    fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        (**self).queue_audio(frame)
    }
    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        (**self).poll_events()
    }
    fn master_clock_pos(&self) -> Duration {
        (**self).master_clock_pos()
    }
    fn set_paused(&mut self, paused: bool) {
        (**self).set_paused(paused)
    }
    fn set_volume(&mut self, vol: f32) {
        (**self).set_volume(vol)
    }
    fn audio_queue_len_samples(&self) -> u64 {
        (**self).audio_queue_len_samples()
    }
    fn audio_headroom_samples(&self) -> u64 {
        (**self).audio_headroom_samples()
    }
    fn engine_info(&self) -> (Option<String>, Option<String>) {
        (**self).engine_info()
    }
}
