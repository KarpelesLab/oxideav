//! SDL2 audio output engine. Uses the queue API
//! (`SDL_QueueAudio` + `SDL_GetQueuedAudioSize`) rather than
//! callbacks, so no Rust closure has to cross the FFI boundary. The
//! master clock comes from `(total_queued - currently_queued) /
//! bytes_per_second`.
//!
//! SDL2 is loaded + ref-counted through
//! [`crate::drivers::sdl2_root`]; video lives in a separate engine.

use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, Result};

use crate::drivers::audio_convert::{resample_linear, to_f32_interleaved};
use crate::drivers::engine::AudioEngine;
use crate::drivers::sdl2_loader::{self as ldr, SDL_AudioDeviceID, SDL_AudioSpec, Sdl2Lib};
use crate::drivers::sdl2_root::{self, SubsystemGuard};

pub struct SdlAudioEngine {
    lib: Arc<Sdl2Lib>,
    _guard: SubsystemGuard,
    dev: SDL_AudioDeviceID,
    /// Output sample rate (what SDL negotiated with the device).
    sample_rate: u32,
    /// Bytes per output sample frame (channels * sizeof(f32) = ch * 4).
    bytes_per_frame: u32,
    output_channels: u16,
    /// Total bytes ever pushed to SDL via `SDL_QueueAudio`.
    total_queued_bytes: u64,
    volume: f32,
    paused: bool,
}

unsafe impl Send for SdlAudioEngine {}

impl SdlAudioEngine {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let guard = sdl2_root::acquire(sdl2_root::AUDIO_MASK)?;
        let lib = guard.lib().clone();
        let channels = channels.clamp(1, 2);
        let bytes_per_frame = (channels as u32) * 4;
        let desired = SDL_AudioSpec {
            freq: sample_rate as c_int,
            format: ldr::AUDIO_F32,
            channels: channels as u8,
            silence: 0,
            samples: 1024,
            padding: 0,
            size: 0,
            // Queue API — no callback.
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
        // Start the device immediately — it outputs silence until the
        // main thread feeds it samples, which keeps the open/close
        // cycle cheap.
        unsafe { (lib.SDL_PauseAudioDevice)(dev, 0) };
        Ok(Self {
            lib,
            _guard: guard,
            dev,
            sample_rate,
            bytes_per_frame,
            output_channels: channels,
            total_queued_bytes: 0,
            volume: 1.0,
            paused: false,
        })
    }
}

impl Drop for SdlAudioEngine {
    fn drop(&mut self) {
        if self.dev != 0 {
            unsafe { (self.lib.SDL_CloseAudioDevice)(self.dev) };
        }
    }
}

impl AudioEngine for SdlAudioEngine {
    fn queue(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.samples == 0 {
            return Ok(());
        }
        let buf = to_f32_interleaved(frame, self.output_channels);
        let mut final_buf = if frame.sample_rate == self.sample_rate {
            buf
        } else {
            resample_linear(
                &buf,
                frame.sample_rate,
                self.sample_rate,
                self.output_channels as usize,
            )
        };
        // Apply volume in-place — queue API has no callback, so this
        // is the only hook we get.
        let vol = self.volume;
        if (vol - 1.0).abs() > f32::EPSILON {
            for s in final_buf.iter_mut() {
                *s *= vol;
            }
        }
        let byte_len = (final_buf.len() * std::mem::size_of::<f32>()) as u32;
        let rc = unsafe {
            (self.lib.SDL_QueueAudio)(
                self.dev,
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
        self.total_queued_bytes += byte_len as u64;
        Ok(())
    }

    fn master_clock_pos(&self) -> Duration {
        let queued = unsafe { (self.lib.SDL_GetQueuedAudioSize)(self.dev) } as u64;
        let bpf = self.bytes_per_frame.max(1) as u64;
        let played_frames = self.total_queued_bytes.saturating_sub(queued) / bpf;
        let sr = self.sample_rate.max(1) as u64;
        let secs = played_frames / sr;
        let frac = played_frames % sr;
        let nanos = (frac * 1_000_000_000) / sr;
        Duration::new(secs, nanos as u32)
    }

    fn set_paused(&mut self, paused: bool) {
        if self.paused == paused {
            return;
        }
        self.paused = paused;
        unsafe {
            (self.lib.SDL_PauseAudioDevice)(self.dev, if paused { 1 } else { 0 });
        }
    }

    fn set_volume(&mut self, vol: f32) {
        self.volume = vol.clamp(0.0, 1.0);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        let queued_bytes = unsafe { (self.lib.SDL_GetQueuedAudioSize)(self.dev) } as u64;
        let bpf = self.bytes_per_frame.max(1) as u64;
        queued_bytes / bpf
    }

    fn audio_headroom_samples(&self) -> u64 {
        // SDL2's queue has no hard cap — it'll grow unbounded if we
        // keep pushing. Impose a ~4s soft target so the player
        // applies back-pressure the same way it does against
        // oxideav-sysaudio.
        let target = self.sample_rate as u64 * 4;
        target.saturating_sub(self.audio_queue_len_samples())
    }

    fn info(&self) -> String {
        // SDL2 reports "how much is queued"; we can derive an
        // estimate of latency at this instant but it's not a
        // hardware property, just our own queue depth. Flag that
        // clearly so users don't confuse it with e.g. PulseAudio's
        // server-reported latency.
        format!(
            "sdl2 @ {} Hz {}ch f32 — latency: derived from SDL queue size (software-side only)",
            self.sample_rate, self.output_channels
        )
    }
}
