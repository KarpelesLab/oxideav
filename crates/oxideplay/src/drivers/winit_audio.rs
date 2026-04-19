//! `oxideav-sysaudio`-backed audio output for the winit driver.
//!
//! `sysaudio` dlopen's the native audio API at runtime (ALSA,
//! PulseAudio, WASAPI, CoreAudio, …) so `oxideplay` stops listing
//! `libasound.so.2` in its ELF NEEDED entries. The backend gives us a
//! pull-callback; we feed it from a lock-free SPSC ring buffer that
//! `queue_audio` fills on the main thread. The callback increments a
//! `samples_played` atomic that the driver reports as the audio master
//! clock.
//!
//! The default driver is picked by `oxideav_sysaudio::probe()` — on
//! Linux that's PulseAudio when a server is running, otherwise ALSA.
//! Users can force a specific driver via the `OXIDEPLAY_AUDIO_DRIVER`
//! environment variable (set to `pulse`, `alsa`, `wasapi`, …).

use std::env;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, Result};
use oxideav_sysaudio::{self as sysaudio, StreamRequest};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};

use crate::drivers::audio_convert::{resample_linear, to_f32_interleaved};

pub struct AudioOut {
    // Kept alive so the sysaudio callback keeps running. No direct
    // access to the inner data.
    _stream: sysaudio::Stream,

    producer: ringbuf::HeapProd<f32>,

    /// Device-side sample rate; may differ from the decoder's rate.
    /// When it does, `queue_audio` resamples before pushing.
    device_rate: u32,
    /// Device-side channel count (clamped to 1 or 2).
    device_channels: u16,

    /// Samples per channel consumed by the device. Grows monotonically
    /// while the stream is playing.
    samples_played: Arc<AtomicU64>,
    /// 0.0..=1.0 volume, bit-packed so we can load/store without a mutex.
    volume: Arc<AtomicU32>,
    /// Mirror of the paused state so we don't call `pause/play`
    /// repeatedly.
    paused: bool,
    /// If the decoder's rate differed from the device rate, we
    /// resample with a dumb linear interpolator.
    resample_from: Option<u32>,

    /// Latches `true` if the callback is ever invoked. Used nowhere
    /// critical; primarily a diagnostic hook.
    #[allow(dead_code)]
    callback_ran: Arc<AtomicBool>,
}

impl AudioOut {
    /// Open an output stream on the platform's preferred audio backend.
    /// Respects `OXIDEPLAY_AUDIO_DRIVER=<name>` for explicit override.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let channels = channels.clamp(1, 2);
        let driver = select_driver()?;

        // The backend decides the actual device rate; we'll resample on
        // the producer side if it doesn't match `sample_rate`.
        let req = StreamRequest::new(sample_rate, channels);

        // Pre-allocate the ring buffer. We don't yet know the device
        // rate until after open(), so size conservatively — worst-case
        // 192 kHz × 2 ch × 4 s = 1.5M samples ≈ 6 MB of f32. Cheap.
        let capacity = ((sample_rate.max(48_000) as usize) * channels as usize * 4).max(8192);
        let rb = HeapRb::<f32>::new(capacity);
        let (producer, mut consumer) = rb.split();

        let samples_played = Arc::new(AtomicU64::new(0));
        let samples_played_cb = samples_played.clone();
        let volume = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let volume_cb = volume.clone();
        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_ran_cb = callback_ran.clone();

        // We need device_channels inside the callback, but we don't
        // learn it until `open` returns. Capture `channels` (our
        // requested count) here and patch it up after the fact if the
        // backend negotiated something else.
        let ch_cb = channels as usize;

        let stream = sysaudio::open(driver, req, move |out, _info| {
            callback_ran_cb.store(true, Ordering::Relaxed);
            let v = f32::from_bits(volume_cb.load(Ordering::Relaxed));
            let written = consumer.pop_slice(out);
            for s in out[..written].iter_mut() {
                *s *= v;
            }
            // Underrun = silence; don't leak stale buffer memory.
            out[written..].fill(0.0);
            samples_played_cb.fetch_add((written / ch_cb) as u64, Ordering::Relaxed);
        })
        .map_err(|e| Error::other(format!("sysaudio: open({}): {e}", driver.name())))?;

        let fmt = stream.format();
        let resample_from = (fmt.sample_rate != sample_rate).then_some(sample_rate);

        Ok(Self {
            _stream: stream,
            producer,
            device_rate: fmt.sample_rate,
            device_channels: fmt.channels,
            samples_played,
            volume,
            paused: false,
            resample_from,
            callback_ran,
        })
    }

    pub fn queue_audio(&mut self, frame: &AudioFrame) -> Result<()> {
        // 1. Normalise to f32 interleaved at the device channel count.
        let mut buf = to_f32_interleaved(frame, self.device_channels);
        // 2. Resample if rates disagree.
        if let Some(src_rate) = self.resample_from {
            buf = resample_linear(
                &buf,
                src_rate,
                self.device_rate,
                self.device_channels as usize,
            );
        }
        // 3. Push into the ring. If the ring is full we drop — same
        // behaviour as SDL_QueueAudio: the device keeps consuming what's
        // already queued. In practice the producer beats the consumer.
        let _ = self.producer.push_slice(&buf);
        Ok(())
    }

    pub fn master_clock_pos(&self) -> Duration {
        let samples = self.samples_played.load(Ordering::Relaxed);
        let rate = self.device_rate.max(1) as u64;
        let secs = samples / rate;
        let nanos = ((samples % rate) * 1_000_000_000 / rate) as u32;
        Duration::new(secs, nanos)
    }

    /// Current output-side latency as reported by the sysaudio backend:
    /// how much audio sits "in flight" between `samples_played` being
    /// incremented and the user actually hearing it. Bluetooth and
    /// network sinks push this into the hundreds of milliseconds, so
    /// the video path should subtract this from `master_clock_pos()`
    /// for accurate A/V sync.
    #[allow(dead_code)] // consumed by the sync layer once A/V-sync compensation lands
    pub fn audio_latency(&self) -> Option<Duration> {
        self._stream.latency()
    }

    pub fn set_paused(&mut self, paused: bool) {
        if paused == self.paused {
            return;
        }
        if paused {
            let _ = self._stream.pause();
        } else {
            let _ = self._stream.play();
        }
        self.paused = paused;
    }

    pub fn set_volume(&mut self, v: f32) {
        let clamped = v.clamp(0.0, 1.0);
        self.volume.store(clamped.to_bits(), Ordering::Relaxed);
    }

    pub fn audio_queue_len_samples(&self) -> u64 {
        // `occupied_len` counts f32 slots; divide by channel count to
        // match the "samples" unit the player expects.
        (self.producer.occupied_len() / self.device_channels.max(1) as usize) as u64
    }
}

/// Resolve the `oxideav-sysaudio` driver to use. Priority:
///   1. `OXIDEPLAY_AUDIO_DRIVER=<name>` — exact match on backend name,
///      regardless of whether `probe()` thinks it's ready.
///   2. `oxideav_sysaudio::default_driver()` — first entry of probe().
fn select_driver() -> Result<sysaudio::Driver> {
    if let Ok(name) = env::var("OXIDEPLAY_AUDIO_DRIVER") {
        let name = name.trim();
        if !name.is_empty() {
            return sysaudio::driver_by_name(name).ok_or_else(|| {
                Error::other(format!(
                    "OXIDEPLAY_AUDIO_DRIVER={name} — no such sysaudio backend"
                ))
            });
        }
    }
    sysaudio::default_driver()
        .ok_or_else(|| Error::other("sysaudio: no audio backend is available"))
}
