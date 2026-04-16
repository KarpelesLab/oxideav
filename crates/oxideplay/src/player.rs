//! Player pipeline driver.
//!
//! Owns the demuxer, decoders, and output driver. Runs a simple cooperative
//! loop: read packet → decode → push frames to driver → poll events →
//! sleep if buffer is getting full. Audio is the master clock.

use std::time::{Duration, Instant};

use oxideav::Registries;
use oxideav_codec::Decoder;
use oxideav_container::{Demuxer, ReadSeek};
use oxideav_core::{AudioFrame, CodecParameters, Error, Frame, MediaType, Result, StreamInfo};
use oxideav_source::{BufferedSource, SourceRegistry};

use crate::driver::{OutputDriver, PlayerEvent, SeekDir};

/// All the state the play loop needs.
pub struct Player<D: OutputDriver> {
    pub driver: D,
    demuxer: Box<dyn Demuxer>,
    audio_decoder: Option<Box<dyn Decoder>>,
    video_decoder: Option<Box<dyn Decoder>>,
    audio_stream: Option<StreamInfo>,
    video_stream: Option<StreamInfo>,
    /// Where the audio master clock was *set* to (from a seek or start).
    /// Added to driver.master_clock_pos() to get the "logical" position.
    clock_origin: Duration,
    /// Samples already consumed by the driver at the moment of the last
    /// seek. Used to offset the driver's monotonic sample counter.
    clock_baseline_samples: u64,
    output_sample_rate: u32,
    paused: bool,
    volume: f32,
    eof: bool,
}

/// Summary info about what we'll play.
pub struct OpenedMedia {
    pub audio: Option<StreamInfo>,
    pub video: Option<StreamInfo>,
    pub duration: Option<Duration>,
    pub format_name: String,
}

/// Default prefetch buffer for playback (bytes). Sized to absorb a few
/// seconds of typical home-broadband jitter on HD streams.
pub const DEFAULT_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// Open `input` (URI or bare path) through the source registry, wrap in
/// a [`BufferedSource`] of `buffer_bytes`, then probe the container.
/// Returns the detected format name plus the buffered handle, ready
/// for `open_demuxer`.
fn detect_input_format(
    registries: &Registries,
    sources: &SourceRegistry,
    input: &str,
    buffer_bytes: usize,
) -> Result<(String, Box<dyn ReadSeek>)> {
    let raw = sources.open(input)?;
    let buffered = BufferedSource::new(raw, buffer_bytes)?;
    let mut handle: Box<dyn ReadSeek> = Box::new(buffered);
    let ext = ext_from_uri(input);
    let format = registries
        .containers
        .probe_input(&mut *handle, ext.as_deref())?;
    let _ = Error::FormatNotFound; // keep the import live; no fallback needed.
    Ok((format, handle))
}

/// Best-effort extension hint from a URI: takes everything after the
/// last `/`-segment's `.`, ignoring `?…` query strings.
fn ext_from_uri(uri: &str) -> Option<String> {
    let last_segment = uri.rsplit('/').next().unwrap_or(uri);
    let last_segment = last_segment.split('?').next().unwrap_or(last_segment);
    let dot = last_segment.rfind('.')?;
    Some(last_segment[dot + 1..].to_ascii_lowercase())
}

/// Probe the input and return its streams without touching SDL2.
/// Used for `--dry-run` and for determining whether to open a video window.
pub fn probe(
    registries: &Registries,
    sources: &SourceRegistry,
    input: &str,
) -> Result<OpenedMedia> {
    // Probe doesn't need a fat buffer — keep memory low.
    let (format, file) = detect_input_format(registries, sources, input, 1 << 20)?;
    let demuxer = registries.containers.open_demuxer(&format, file)?;
    let (audio, video) = pick_streams(demuxer.streams());
    let duration = audio
        .as_ref()
        .or(video.as_ref())
        .and_then(|s| s.duration.map(|d| secs_of(s, d)));
    Ok(OpenedMedia {
        audio,
        video,
        duration,
        format_name: demuxer.format_name().to_owned(),
    })
}

fn pick_streams(streams: &[StreamInfo]) -> (Option<StreamInfo>, Option<StreamInfo>) {
    let audio = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Audio)
        .cloned();
    let video = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .cloned();
    (audio, video)
}

fn secs_of(s: &StreamInfo, ticks: i64) -> Duration {
    let secs = s.time_base.seconds_of(ticks);
    if secs.is_finite() && secs > 0.0 {
        Duration::from_secs_f64(secs)
    } else {
        Duration::ZERO
    }
}

impl<D: OutputDriver> Player<D> {
    /// Open the file, build decoders, and return a `Player` that's ready
    /// to run. `build_driver` receives the (audio sample rate, audio
    /// channels, optional video (w,h)) and returns a driver — this lets
    /// the caller pick headless vs. SDL2 etc.
    pub fn open<F>(
        registries: &Registries,
        sources: &SourceRegistry,
        input: &str,
        buffer_bytes: usize,
        build_driver: F,
    ) -> Result<(Self, OpenedMedia)>
    where
        F: FnOnce(u32, u16, Option<(u32, u32)>) -> Result<D>,
    {
        let (format, file) = detect_input_format(registries, sources, input, buffer_bytes)?;
        let demuxer = registries.containers.open_demuxer(&format, file)?;
        let (audio, video) = pick_streams(demuxer.streams());
        let duration = audio
            .as_ref()
            .or(video.as_ref())
            .and_then(|s| s.duration.map(|d| secs_of(s, d)));
        let format_name = demuxer.format_name().to_owned();

        let (audio_decoder, audio_sample_rate, audio_channels) = match &audio {
            Some(s) => {
                let dec = registries.codecs.make_decoder(&s.params)?;
                (
                    Some(dec),
                    s.params.sample_rate.unwrap_or(48_000),
                    s.params.channels.unwrap_or(2),
                )
            }
            None => (None, 48_000, 2),
        };

        let (video_decoder, video_dims) = match &video {
            Some(s) => match registries.codecs.make_decoder(&s.params) {
                Ok(d) => {
                    let w = s.params.width.unwrap_or(640);
                    let h = s.params.height.unwrap_or(480);
                    (Some(d), Some((w, h)))
                }
                Err(e) => {
                    eprintln!(
                        "oxideplay: video decoder unavailable for {}: {}",
                        s.params.codec_id, e
                    );
                    (None, None)
                }
            },
            None => (None, None),
        };

        let driver = build_driver(audio_sample_rate, audio_channels, video_dims)?;

        let opened = OpenedMedia {
            audio: audio.clone(),
            video: video.clone(),
            duration,
            format_name,
        };

        Ok((
            Self {
                driver,
                demuxer,
                audio_decoder,
                video_decoder,
                audio_stream: audio,
                video_stream: video,
                clock_origin: Duration::ZERO,
                clock_baseline_samples: 0,
                output_sample_rate: audio_sample_rate,
                paused: false,
                volume: 1.0,
                eof: false,
            },
            opened,
        ))
    }

    pub fn position(&self) -> Duration {
        let raw = self.driver.master_clock_pos();
        // Subtract the pre-seek baseline then add the origin.
        let base = Duration::from_secs_f64(
            self.clock_baseline_samples as f64 / self.output_sample_rate.max(1) as f64,
        );
        self.clock_origin + raw.saturating_sub(base)
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    #[allow(dead_code)]
    pub fn audio_stream(&self) -> Option<&StreamInfo> {
        self.audio_stream.as_ref()
    }

    #[allow(dead_code)]
    pub fn video_stream(&self) -> Option<&StreamInfo> {
        self.video_stream.as_ref()
    }

    #[allow(dead_code)]
    pub fn can_seek(&self) -> bool {
        // Placeholder: we learn seek support lazily. `main.rs` flips the
        // flag off when the demuxer returns `Unsupported`.
        false
    }

    /// Apply a user event. Returns true if the player should keep running.
    pub fn apply_event(&mut self, ev: PlayerEvent, seek_supported: &mut bool) -> bool {
        match ev {
            PlayerEvent::Quit => return false,
            PlayerEvent::TogglePause => {
                self.paused = !self.paused;
                self.driver.set_paused(self.paused);
            }
            PlayerEvent::SeekRelative(d, dir) => {
                if !*seek_supported {
                    return true;
                }
                let cur = self.position();
                let target = match dir {
                    SeekDir::Forward => cur + d,
                    SeekDir::Back => cur.saturating_sub(d),
                };
                match self.seek_to(target) {
                    Ok(()) => {}
                    Err(Error::Unsupported(_)) => {
                        *seek_supported = false;
                    }
                    Err(e) => eprintln!("oxideplay: seek failed: {e}"),
                }
            }
            PlayerEvent::VolumeDelta(d) => {
                self.volume = (self.volume + (d as f32) / 100.0).clamp(0.0, 1.0);
                self.driver.set_volume(self.volume);
            }
        }
        true
    }

    /// Attempt to seek to an absolute position.
    pub fn seek_to(&mut self, target: Duration) -> Result<()> {
        // Prefer seeking on the audio stream (master clock); fall back to
        // video if no audio is present.
        let (stream_idx, tb) = if let Some(a) = &self.audio_stream {
            (a.index, a.time_base)
        } else if let Some(v) = &self.video_stream {
            (v.index, v.time_base)
        } else {
            return Err(Error::unsupported("nothing to seek"));
        };
        let pts = (target.as_secs_f64() / tb.as_rational().as_f64()).round() as i64;
        let landed = self.demuxer.seek_to(stream_idx, pts)?;
        let landed_secs = tb.seconds_of(landed);
        let landed_dur = if landed_secs.is_finite() && landed_secs > 0.0 {
            Duration::from_secs_f64(landed_secs)
        } else {
            Duration::ZERO
        };

        // Reset decoder queues so we don't serve pre-seek frames.
        if let Some(dec) = self.audio_decoder.as_mut() {
            // Best-effort flush; any error here just means some buffered
            // frames remain, which will be harmless.
            let _ = dec.flush();
            drain_decoder(dec.as_mut());
        }
        if let Some(dec) = self.video_decoder.as_mut() {
            let _ = dec.flush();
            drain_decoder(dec.as_mut());
        }
        self.clock_baseline_samples =
            self.driver
                .master_clock_pos()
                .as_secs_f64()
                .max(0.0)
                .mul_add(self.output_sample_rate as f64, 0.0) as u64;
        self.clock_origin = landed_dur;
        self.eof = false;
        Ok(())
    }

    /// Drive one loop iteration: read a packet, decode, push frames out.
    /// Returns Ok(()) normally; Err at EOF or unrecoverable error.
    pub fn pump_once(&mut self) -> Result<bool> {
        if self.paused {
            return Ok(false);
        }
        if self.eof {
            return Ok(false);
        }
        // Read next packet.
        let pkt = match self.demuxer.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => {
                self.eof = true;
                // Flush decoders to drain.
                if let Some(d) = self.audio_decoder.as_mut() {
                    let _ = d.flush();
                    while let Ok(Frame::Audio(af)) = d.receive_frame() {
                        let _ = self.driver.queue_audio(&af);
                    }
                }
                return Ok(false);
            }
            Err(e) => return Err(e),
        };

        // Route packet to the right decoder.
        let audio_idx = self.audio_stream.as_ref().map(|s| s.index);
        let video_idx = self.video_stream.as_ref().map(|s| s.index);
        if Some(pkt.stream_index) == audio_idx {
            if let Some(dec) = self.audio_decoder.as_mut() {
                if let Err(e) = dec.send_packet(&pkt) {
                    if !matches!(e, Error::NeedMore) {
                        eprintln!("oxideplay: audio decode error: {e}");
                    }
                }
                loop {
                    match dec.receive_frame() {
                        Ok(Frame::Audio(af)) => {
                            self.driver.queue_audio(&af)?;
                        }
                        Ok(Frame::Video(_)) => {}
                        Err(Error::NeedMore) => break,
                        Err(Error::Eof) => {
                            self.eof = true;
                            break;
                        }
                        Err(e) => {
                            eprintln!("oxideplay: audio recv error: {e}");
                            break;
                        }
                    }
                }
            }
        } else if Some(pkt.stream_index) == video_idx {
            // Collect decoded video frames first, then present them
            // (avoids a double-borrow of &mut self through video_decoder
            // and &self through position()/driver).
            let mut decoded: Vec<oxideav_core::VideoFrame> = Vec::new();
            if let Some(dec) = self.video_decoder.as_mut() {
                if let Err(e) = dec.send_packet(&pkt) {
                    if !matches!(e, Error::NeedMore) {
                        eprintln!("oxideplay: video decode error: {e}");
                    }
                }
                loop {
                    match dec.receive_frame() {
                        Ok(Frame::Video(vf)) => decoded.push(vf),
                        Ok(Frame::Audio(_)) => {}
                        Err(Error::NeedMore) => break,
                        Err(Error::Eof) => {
                            self.eof = true;
                            break;
                        }
                        Err(e) => {
                            eprintln!("oxideplay: video recv error: {e}");
                            break;
                        }
                    }
                }
            }
            let video_tb = self.video_stream.as_ref().map(|s| s.time_base);
            for vf in decoded {
                let pts_secs = match (vf.pts, video_tb) {
                    (Some(p), Some(tb)) => tb.seconds_of(p),
                    _ => 0.0,
                };
                let target = if pts_secs.is_finite() && pts_secs > 0.0 {
                    Duration::from_secs_f64(pts_secs)
                } else {
                    Duration::ZERO
                };
                let now = self.position();
                let epsilon = Duration::from_millis(50);
                if target + epsilon < now {
                    continue;
                }
                if target > now + epsilon {
                    let wait = target - now;
                    let capped = std::cmp::min(wait, Duration::from_millis(50));
                    std::thread::sleep(capped);
                }
                self.driver.present_video(&vf)?;
            }
        }
        Ok(true)
    }

    pub fn eof_reached(&self) -> bool {
        self.eof
    }

    /// Has playback of the queued audio caught up to end-of-stream? Used
    /// to decide when to exit after demuxer EOF.
    pub fn audio_drained(&self) -> bool {
        self.driver.audio_queue_len_samples() == 0
    }

    /// Run the whole playback loop with the given callback invoked
    /// roughly once per UI tick (~16ms). Callback returns events to apply
    /// and a bool: should we keep running. It is also responsible for
    /// drawing the TUI/progress, since only the caller knows whether
    /// stdout is a TTY.
    #[allow(dead_code)]
    pub fn run<Tick>(mut self, mut tick: Tick) -> Result<()>
    where
        Tick: FnMut(&mut Player<D>, Vec<PlayerEvent>) -> bool,
    {
        let tick_interval = Duration::from_millis(16);
        let mut last_tick = Instant::now();
        let mut seek_supported = true;
        loop {
            // Gather events (from driver + tui via caller).
            let driver_events = self.driver.poll_events();
            let keep = tick(&mut self, driver_events);
            if !keep {
                break;
            }
            let mut running = true;
            // We shouldn't apply events twice — `tick` was given the driver
            // events but the contract is that `tick` returns true/false and
            // is responsible for calling `apply_event` itself. See main.rs.
            let _ = &mut running;
            let _ = &mut seek_supported;

            // Pump the pipeline if we're not too far ahead.
            let max_buffer_secs = Duration::from_secs(2);
            let buffered = Duration::from_secs_f64(
                self.driver.audio_queue_len_samples() as f64
                    / self.output_sample_rate.max(1) as f64,
            );
            if !self.paused && !self.eof && buffered < max_buffer_secs {
                let _ = self.pump_once()?;
            }

            if self.eof && self.audio_drained() && !self.paused {
                break;
            }

            // Throttle to ~60Hz.
            let now = Instant::now();
            let elapsed = now - last_tick;
            if elapsed < tick_interval {
                std::thread::sleep(tick_interval - elapsed);
            }
            last_tick = Instant::now();
        }
        Ok(())
    }
}

/// Consume all frames currently sitting in a decoder (after flush),
/// discarding them. Used to clear buffers after seek.
fn drain_decoder(dec: &mut dyn Decoder) {
    loop {
        match dec.receive_frame() {
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(_) => break,
        }
    }
}

/// Compute how many channels + what sample rate the driver should be
/// initialised with given a stream's parameters. Provided as a free
/// function so tests can cover it without standing up SDL2.
#[allow(dead_code)]
pub fn driver_dims(audio: &Option<CodecParameters>) -> (u32, u16) {
    match audio {
        Some(p) => (
            p.sample_rate.unwrap_or(48_000),
            p.channels.unwrap_or(2).clamp(1, 2),
        ),
        None => (48_000, 2),
    }
}

/// Convert samples-played + sample-rate to a Duration. Extracted so it
/// can be tested without involving SDL2.
#[allow(dead_code)]
pub fn samples_to_duration(samples: u64, sample_rate: u32) -> Duration {
    let sr = sample_rate.max(1) as u64;
    Duration::new(samples / sr, ((samples % sr) * 1_000_000_000 / sr) as u32)
}

/// Convenience: given position + total duration, produce a normalized 0..1.
/// Clamped. Returns 0.0 for unknown totals.
#[allow(dead_code)]
pub fn progress_fraction(pos: Duration, total: Option<Duration>) -> f64 {
    match total {
        Some(t) if t.as_secs_f64() > 0.0 => (pos.as_secs_f64() / t.as_secs_f64()).clamp(0.0, 1.0),
        _ => 0.0,
    }
}

/// AudioFrame → (samples, duration). Used when estimating how much audio
/// we already have queued so we don't run away with the decode loop.
#[allow(dead_code)]
pub fn audio_frame_duration(frame: &AudioFrame) -> Duration {
    let sr = frame.sample_rate.max(1);
    Duration::from_secs_f64(frame.samples as f64 / sr as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_to_duration_exact_second() {
        assert_eq!(samples_to_duration(48_000, 48_000), Duration::from_secs(1));
    }

    #[test]
    fn samples_to_duration_sub_second() {
        let d = samples_to_duration(24_000, 48_000);
        assert_eq!(d, Duration::from_millis(500));
    }

    #[test]
    fn progress_fraction_basic() {
        let p = progress_fraction(Duration::from_secs(30), Some(Duration::from_secs(60)));
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn progress_fraction_unknown_total() {
        let p = progress_fraction(Duration::from_secs(30), None);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn driver_dims_picks_defaults() {
        let (sr, ch) = driver_dims(&None);
        assert_eq!(sr, 48_000);
        assert_eq!(ch, 2);
    }
}
