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
    /// The underlying sysaudio stream. Kept alive so the callback
    /// keeps running; we also drive it through `play()` / `pause()`
    /// for the preroll gate.
    stream: sysaudio::Stream,

    producer: ringbuf::HeapProd<f32>,

    /// Device-side sample rate; may differ from the decoder's rate.
    device_rate: u32,
    /// Device-side channel count (clamped to 1 or 2).
    device_channels: u16,

    /// Samples per channel consumed by the device — monotonic.
    samples_played: Arc<AtomicU64>,
    /// 0.0..=1.0 volume, bit-packed so we can load/store without a mutex.
    volume: Arc<AtomicU32>,
    /// User-requested pause state (Space in the player). Combined
    /// with `preroll_done` to decide whether the stream should
    /// actually be playing.
    user_paused: bool,
    /// Samples-per-channel we want buffered before we start playing
    /// the first time. Smooths startup when the decoder barely
    /// matches real-time (e.g. slow codec in debug builds, or a
    /// modem-speed network input).
    preroll_target: u64,
    /// True once we've unpaused after hitting `preroll_target`.
    /// Latches; after this `pause()` / `play()` pass through to the
    /// stream unchanged.
    preroll_done: bool,
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

        let mut stream = sysaudio::open(driver, req, move |out, _info| {
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

        // Start paused so the callback writes silence (and
        // `samples_played` doesn't advance) until `queue()` has
        // primed the ring with ~1 s of audio. Without this the
        // callback opens immediately on an empty ring and underruns
        // on the first beat, which sounds like chopping or a stutter
        // on any codec where decode-per-packet is close to the
        // packet duration.
        let _ = stream.pause();

        let fmt = stream.format();
        let resample_from = (fmt.sample_rate != sample_rate).then_some(sample_rate);
        // 1 s preroll target — generous enough to smooth routine
        // decoder jitter without feeling laggy.
        let preroll_target = fmt.sample_rate as u64;

        Ok(Self {
            stream,
            producer,
            device_rate: fmt.sample_rate,
            device_channels: fmt.channels,
            samples_played,
            volume,
            user_paused: false,
            preroll_target,
            preroll_done: false,
            resample_from,
            backend_name: driver.name(),
            callback_ran,
        })
    }

    /// Decide whether the stream should be playing right now, given
    /// the combination of `user_paused` + `preroll_done`. Called from
    /// the state transitions below.
    fn apply_play_state(&mut self) {
        let should_play = self.preroll_done && !self.user_paused;
        if should_play {
            let _ = self.stream.play();
        } else {
            let _ = self.stream.pause();
        }
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
        // Push what fits. The player's back-pressure path
        // (`audio_headroom_samples`) means we normally never come
        // close to a full ring — so any partial-push here signals
        // the back-pressure threshold is set too loose, not that we
        // should silently drop. The caller owns that decision.
        let _ = self.producer.push_slice(&buf);

        // First push that crosses the preroll threshold flips the
        // stream into the playing state.
        if !self.preroll_done
            && self.producer.occupied_len() as u64
                >= self.preroll_target * self.device_channels.max(1) as u64
        {
            self.preroll_done = true;
            self.apply_play_state();
        }
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
        if paused == self.user_paused {
            return;
        }
        self.user_paused = paused;
        self.apply_play_state();
    }

    fn set_volume(&mut self, v: f32) {
        let clamped = v.clamp(0.0, 1.0);
        self.volume.store(clamped.to_bits(), Ordering::Relaxed);
    }

    fn audio_queue_len_samples(&self) -> u64 {
        (self.producer.occupied_len() / self.device_channels.max(1) as usize) as u64
    }

    fn audio_headroom_samples(&self) -> u64 {
        // The ringbuf stores f32 slots; divide by channels to report
        // per-channel sample headroom, matching what the player
        // thinks of as "samples". Once this hits ~0 the player stops
        // pulling audio, the decoder channel fills, the decoder
        // blocks, and the demuxer naturally back-pressures.
        (self.producer.vacant_len() as u64) / self.device_channels.max(1) as u64
    }

    fn latency(&self) -> Option<Duration> {
        self.stream.latency()
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
