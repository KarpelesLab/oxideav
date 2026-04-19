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

    /// Pre-roll gate: the driver starts with its audio output paused
    /// (so SDL doesn't begin draining an empty queue the instant the
    /// device opens). The player calls this on every tick; the first
    /// call where the buffered-sample count is ≥ `min_samples` flips
    /// the device into playing state. Subsequent calls are no-ops
    /// (returns `true`).
    ///
    /// Default implementation: immediately "prerolled" — headless /
    /// dummy drivers don't need a pre-roll step.
    fn try_start_audio(&mut self, _min_samples: u64) -> bool {
        true
    }
}
