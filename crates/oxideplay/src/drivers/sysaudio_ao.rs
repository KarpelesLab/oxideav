//! `oxideav-sysaudio`-backed audio output engine.
//!
//! `sysaudio` dlopen's the native audio API at runtime (ALSA,
//! PulseAudio, WASAPI, CoreAudio, …) so `oxideplay` has no
//! `libasound.so.2` / `libpulse.so.0` / `ole32.dll` / `AudioToolbox`
//! entry in its dynamic-link requirements. The backend gives us a
//! pull-callback; we feed it from a lock-free SPSC ring buffer that
//! `queue` fills. The callback increments a `samples_played` atomic
//! that the engine reports as the master clock.
//!
//! Driver selection: by default we pick `probe()`'s top choice (e.g.
//! PipeWire → PulseAudio → ALSA → OSS on Linux). The [`Self::with_driver`]
//! constructor lets the CLI force `pulse`, `alsa`, `wasapi`, etc.
//! directly.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oxideav_core::{AudioFrame, Error, Result};
use oxideav_sysaudio::{self as sysaudio, Driver, StreamRequest};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};

use crate::drivers::audio_convert::{resample_linear, to_f32_interleaved};
use crate::drivers::engine::AudioEngine;

pub struct SysAudioEngine {
    /// Kept alive so the sysaudio callback keeps running.
    _stream: sysaudio::Stream,

    producer: ringbuf::HeapProd<f32>,

    /// Device-side sample rate; may differ from the decoder's rate.
    device_rate: u32,
    /// Device-side channel count (clamped to 1 or 2).
    device_channels: u16,

    /// Samples per channel consumed by the device — monotonic.
    samples_played: Arc<AtomicU64>,
    /// 0.0..=1.0 volume, bit-packed so we can load/store without a mutex.
    volume: Arc<AtomicU32>,
    /// Mirror of the paused state so we don't call `pause/play`
    /// repeatedly.
    paused: bool,
    /// If the decoder's rate differed from the device rate, `queue`
    /// resamples with a dumb linear interpolator before pushing.
    resample_from: Option<u32>,

    /// Name of the sysaudio backend we landed on (e.g. `"pulse"`,
    /// `"alsa"`). Remembered so `info()` can tell the user which
    /// driver ended up active after `auto`.
    backend_name: &'static str,

    /// Diagnostic — latches true once the callback has run.
    #[allow(dead_code)]
    callback_ran: Arc<AtomicBool>,
}

impl SysAudioEngine {
    /// Open on `probe()`'s top pick. See [`Self::with_driver`] for
    /// explicit selection.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let driver = sysaudio::default_driver()
            .ok_or_else(|| Error::other("sysaudio: no audio backend is available"))?;
        Self::open(driver, sample_rate, channels)
    }

    /// Open using a specific sysaudio driver name (`"pulse"`,
    /// `"alsa"`, `"wasapi"`, …). The CLI uses this when `--ao <name>`
    /// is passed.
    pub fn with_driver(name: &str, sample_rate: u32, channels: u16) -> Result<Self> {
        let driver = sysaudio::driver_by_name(name).ok_or_else(|| {
            Error::other(format!(
                "sysaudio: no backend named '{name}' — try one of: {}",
                sysaudio::drivers()
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        Self::open(driver, sample_rate, channels)
    }

    fn open(driver: Driver, sample_rate: u32, channels: u16) -> Result<Self> {
        let channels = channels.clamp(1, 2);
        let req = StreamRequest::new(sample_rate, channels);

        // Size the ring for ~4 s worst-case at 192 kHz.
        let capacity = ((sample_rate.max(48_000) as usize) * channels as usize * 4).max(8192);
        let rb = HeapRb::<f32>::new(capacity);
        let (producer, mut consumer) = rb.split();

        let samples_played = Arc::new(AtomicU64::new(0));
        let samples_played_cb = samples_played.clone();
        let volume = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let volume_cb = volume.clone();
        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_ran_cb = callback_ran.clone();
        let ch_cb = channels as usize;

        let stream = sysaudio::open(driver, req, move |out, _info| {
            callback_ran_cb.store(true, Ordering::Relaxed);
            let v = f32::from_bits(volume_cb.load(Ordering::Relaxed));
            let written = consumer.pop_slice(out);
            for s in out[..written].iter_mut() {
                *s *= v;
            }
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
            backend_name: driver.name(),
            callback_ran,
        })
    }
}

/// Short description of how reliably a given sysaudio backend reports
/// `Stream::latency()`. The copy here mirrors what `oxideav-sysaudio`
/// implements (see its README).
fn latency_quality(backend: &str) -> &'static str {
    match backend {
        "pulse" => "end-to-end (server-reported, catches BT + network)",
        "alsa" => "driver-queue depth (partial — misses BT hops above ALSA)",
        "wasapi" => "stream latency + padding (end-to-end, BT-aware)",
        "coreaudio" => "software estimate — hardware/BT latency not included yet",
        _ => "backend-specific",
    }
}

impl AudioEngine for SysAudioEngine {
    fn queue(&mut self, frame: &AudioFrame) -> Result<()> {
        let mut buf = to_f32_interleaved(frame, self.device_channels);
        if let Some(src_rate) = self.resample_from {
            buf = resample_linear(
                &buf,
                src_rate,
                self.device_rate,
                self.device_channels as usize,
            );
        }
        let _ = self.producer.push_slice(&buf);
        Ok(())
    }

    fn master_clock_pos(&self) -> Duration {
        let samples = self.samples_played.load(Ordering::Relaxed);
        let rate = self.device_rate.max(1) as u64;
        let secs = samples / rate;
        let nanos = ((samples % rate) * 1_000_000_000 / rate) as u32;
        Duration::new(secs, nanos)
    }

    fn set_paused(&mut self, paused: bool) {
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

    fn set_volume(&mut self, v: f32) {
        let clamped = v.clamp(0.0, 1.0);
        self.volume.store(clamped.to_bits(), Ordering::Relaxed);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        (self.producer.occupied_len() / self.device_channels.max(1) as usize) as u64
    }

    fn latency(&self) -> Option<Duration> {
        self._stream.latency()
    }

    fn info(&self) -> String {
        format!(
            "sysaudio/{} @ {} Hz {}ch f32 — latency: {}",
            self.backend_name,
            self.device_rate,
            self.device_channels,
            latency_quality(self.backend_name)
        )
    }
}
