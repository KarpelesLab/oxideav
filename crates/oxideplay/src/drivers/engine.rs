//! Split-responsibility output traits.
//!
//! The old `OutputDriver` trait handled audio *and* video in one object,
//! which forced every driver (SDL2, winit) to own both halves. The
//! `--vo` / `--ao` CLI flags need the opposite: any video engine
//! composed with any audio engine. We split into:
//!
//! - [`VideoEngine`]: presents frames and pumps window-input events.
//! - [`AudioEngine`]: consumes audio frames and owns the master clock.
//!
//! A [`Composite`] struct carries an `Option<Box<dyn VideoEngine>>` and
//! an `Option<Box<dyn AudioEngine>>`, and implements the player's
//! original [`OutputDriver`] trait on top of them. That way the player
//! core (and every call-site that already speaks `OutputDriver`) keeps
//! working unchanged — the composition happens at `build_driver` time.

use std::time::Duration;

use oxideav_core::{AudioFrame, Result, VideoFrame};

use crate::driver::{OutputDriver, PlayerEvent};

/// Per-frame video presentation + window-input pump. Implementations:
/// `SdlVideoEngine`, `WinitVideoEngine`, `NullVideoEngine` (used when
/// `--vo none`).
pub trait VideoEngine: Send {
    fn present(&mut self, frame: &VideoFrame) -> Result<()>;
    /// Drain any queued user-input events (keyboard, close button).
    /// Audio-only engines return an empty Vec.
    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        Vec::new()
    }
    /// The video path gets a chance to react to pause, e.g. by
    /// pausing a GPU render timer. Most just don't care.
    fn set_paused(&mut self, _paused: bool) {}
    /// One-line human-readable description — driver name, GPU / render
    /// backend, initial surface size, pixel format, etc. Printed by
    /// `oxideplay` at startup so users can confirm they got the
    /// backend they expected.
    fn info(&self) -> String {
        "unknown".into()
    }
}

/// Audio output + master-clock owner. Implementations: `SdlAudioEngine`,
/// `SysAudioEngine`, `NullAudioEngine` (used when `--ao none`).
pub trait AudioEngine: Send {
    fn queue(&mut self, frame: &AudioFrame) -> Result<()>;

    /// Current position of the master clock — typically
    /// `samples_played / sample_rate`.
    fn master_clock_pos(&self) -> Duration;

    fn set_paused(&mut self, paused: bool);
    fn set_volume(&mut self, vol: f32);

    /// Approximate samples still queued to the device. Used by the
    /// player to throttle the decoder.
    fn audio_queue_len_samples(&self) -> u64 {
        0
    }
    /// Output-side latency reported by the backend, if available.
    /// See `oxideav_sysaudio::Stream::latency` — over Bluetooth /
    /// network sinks this matters for A/V sync compensation.
    #[allow(dead_code)] // consumed once A/V-sync compensation lands in the sync layer
    fn latency(&self) -> Option<Duration> {
        None
    }
    /// One-line human-readable description — driver name, device
    /// sample rate / channels / format, and a note on how the
    /// backend measures `latency()` (end-to-end vs. driver-queue vs.
    /// software-estimate). Printed by `oxideplay` at startup.
    fn info(&self) -> String {
        "unknown".into()
    }
}

/// Combines an optional video engine with an optional audio engine into
/// the player's original [`OutputDriver`] trait. `--vo none` → `None`
/// for the video slot (present is a no-op, poll_events returns empty);
/// `--ao none` → `None` for audio (clock ticks from a wall-clock
/// fallback).
pub struct Composite {
    pub video: Option<Box<dyn VideoEngine>>,
    pub audio: Option<Box<dyn AudioEngine>>,
    /// Fallback wall-clock start used when `audio` is None. Set to
    /// `None` while paused so elapsed doesn't keep accumulating.
    wall_start: Option<std::time::Instant>,
    wall_accum: Duration,
}

impl Composite {
    pub fn new(
        video: Option<Box<dyn VideoEngine>>,
        audio: Option<Box<dyn AudioEngine>>,
    ) -> Self {
        Self {
            video,
            audio,
            wall_start: Some(std::time::Instant::now()),
            wall_accum: Duration::ZERO,
        }
    }

}

impl OutputDriver for Composite {
    fn present_video(&mut self, frame: &VideoFrame) -> Result<()> {
        match self.video.as_mut() {
            Some(v) => v.present(frame),
            None => Ok(()),
        }
    }

    fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        match self.audio.as_mut() {
            Some(a) => a.queue(frame),
            None => Ok(()),
        }
    }

    fn poll_events(&mut self) -> Vec<PlayerEvent> {
        match self.video.as_mut() {
            Some(v) => v.poll_events(),
            None => Vec::new(),
        }
    }

    fn master_clock_pos(&self) -> Duration {
        if let Some(a) = self.audio.as_ref() {
            return a.master_clock_pos();
        }
        // No audio output → walk wall-clock time. Accurate enough for
        // video-only playback; the decoder pacing doesn't need sample
        // precision.
        match self.wall_start {
            Some(t) => self.wall_accum + t.elapsed(),
            None => self.wall_accum,
        }
    }

    fn set_paused(&mut self, paused: bool) {
        if let Some(a) = self.audio.as_mut() {
            a.set_paused(paused);
        }
        if let Some(v) = self.video.as_mut() {
            v.set_paused(paused);
        }
        // Freeze / unfreeze the wall-clock fallback regardless of
        // whether the audio engine exists — it's only consulted when
        // audio is absent but should behave consistently.
        if paused {
            if let Some(t) = self.wall_start.take() {
                self.wall_accum += t.elapsed();
            }
        } else if self.wall_start.is_none() {
            self.wall_start = Some(std::time::Instant::now());
        }
    }

    fn set_volume(&mut self, vol: f32) {
        if let Some(a) = self.audio.as_mut() {
            a.set_volume(vol);
        }
    }

    fn audio_queue_len_samples(&self) -> u64 {
        self.audio
            .as_ref()
            .map(|a| a.audio_queue_len_samples())
            .unwrap_or(0)
    }

    fn engine_info(&self) -> (Option<String>, Option<String>) {
        (
            self.video.as_ref().map(|v| v.info()),
            self.audio.as_ref().map(|a| a.info()),
        )
    }
}

// `--vo none` and `--ao none` are handled by passing `None` for the
// respective slot in `Composite`; no stub engine needed.
