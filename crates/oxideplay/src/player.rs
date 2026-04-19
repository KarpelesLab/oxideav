//! Player pipeline driver.
//!
//! Owns the demuxer, decoders, and output driver. Runs a simple cooperative
//! loop: read packet → decode → push frames to driver → poll events →
//! sleep if buffer is getting full. Audio is the master clock.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use oxideav::Registries;
use oxideav_container::ReadSeek;
use oxideav_core::{AudioFrame, CodecParameters, Error, MediaType, Result, StreamInfo, VideoFrame};
use oxideav_source::{BufferedSource, SourceRegistry};

use crate::decode_worker::{DecodeWorker, DecodedCtl, DecodedUnit};
use crate::driver::{OutputDriver, PlayerEvent, SeekDir};

/// Soft cap on the number of decoded video frames buffered in the
/// main-thread queue. When we have this many, `pump_once` stops draining
/// from the decoder's output channel — the channel fills up, the decoder
/// blocks on send, and the pipeline becomes paced by actual presentation
/// rather than decode speed. This provides backpressure without dropping
/// future content (which is what a crude queue trim would do).
const VIDEO_QUEUE_SOFT_CAP: usize = 60;

/// How stale a decoded video frame can get before we skip it rather
/// than present it. Post-decode drop: the decoder still sees every
/// packet (so the reference-frame chain stays intact), we just
/// don't pay the SDL render cost for a frame that's already too late
/// to land on screen at its proper pts.
const VIDEO_FRAME_MAX_BEHIND: Duration = Duration::from_millis(100);

/// All the state the play loop needs.
pub struct Player<D: OutputDriver> {
    pub driver: D,
    /// Background thread doing demux + decode. Produces `DecodedUnit`s
    /// into `out_rx` (owned by the worker) for the main thread to drain.
    /// Dropped on Player drop; the worker joins cleanly.
    worker: DecodeWorker,
    audio_stream: Option<StreamInfo>,
    video_stream: Option<StreamInfo>,
    /// Decoded video frames pending presentation. Popped on every tick
    /// in pts-order as wallclock catches up to their pts.
    video_queue: VecDeque<VideoFrame>,
    /// Where the audio master clock was *set* to (from a seek or start).
    /// Added to driver.master_clock_pos() to get the "logical" position.
    clock_origin: Duration,
    /// Samples already consumed by the driver at the moment of the last
    /// seek. Used to offset the driver's monotonic sample counter.
    clock_baseline_samples: u64,
    output_sample_rate: u32,
    paused: bool,
    volume: f32,
    /// True once the worker has signalled [`DecodedUnit::Eof`].
    eof: bool,
    /// Tracks whether we're still waiting for a `Seeked` acknowledgement
    /// after issuing a seek. While `true`, Audio/Video units arriving
    /// in the channel predate the seek and must be discarded.
    seek_pending: bool,
    /// Cumulative duration of audio ever queued to the driver. This is
    /// the sum of each `AudioFrame`'s `samples / sample_rate` and is
    /// independent of whatever weird container time_base the audio
    /// packets were tagged with. `A` in the drift display equals
    /// `last_audio_end - master`.
    last_audio_end: Duration,
    /// pts of the most recent video frame pushed into `video_queue`
    /// (i.e. the newest thing the worker produced, not the newest
    /// frame presented).
    last_video_pts: Option<i64>,
    /// pts of the most recent video frame actually handed to
    /// `driver.present_video`.
    last_video_presented_pts: Option<i64>,
    /// TimeBase used for the most recently issued seek command. Seeks
    /// against the video stream (so we snap to a keyframe) return a
    /// landed pts in *video* time-base, which is different from audio's.
    /// Remembered here so `on_seeked` can convert correctly.
    seek_tb: Option<oxideav_core::TimeBase>,
}

/// Diagnostic timestamps for the TUI. All Durations are relative to
/// the start of the stream (i.e. `tb.seconds_of(pts)`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerTimings {
    pub master: Duration,
    pub audio: Option<Duration>,
    pub video_decoded: Option<Duration>,
    pub video_presented: Option<Duration>,
    pub video_queue_len: usize,
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
    let demuxer = registries
        .containers
        .open_demuxer(&format, file, &registries.codecs)?;
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
    ///
    /// `want_audio` / `want_video` tell the pipeline whether to bring
    /// up a decoder for that track. When either is `false`, the
    /// matching decoder is skipped and the demux thread drops that
    /// stream's packets on the floor — no CPU wasted decoding a stream
    /// the user isn't going to hear or see. We also hint the demuxer
    /// with `set_active_streams` so container-level skipping (MKV
    /// cluster-aware, MP4 trak-aware) can avoid reading the bytes too
    /// when the container supports it. For interleaved containers
    /// where individual streams can't be skipped, the demuxer falls
    /// back to reading and discarding.
    pub fn open<F>(
        registries: &Registries,
        sources: &SourceRegistry,
        input: &str,
        buffer_bytes: usize,
        want_audio: bool,
        want_video: bool,
        build_driver: F,
    ) -> Result<(Self, OpenedMedia)>
    where
        F: FnOnce(u32, u16, Option<(u32, u32)>) -> Result<D>,
    {
        let (format, file) = detect_input_format(registries, sources, input, buffer_bytes)?;
        let mut demuxer = registries
            .containers
            .open_demuxer(&format, file, &registries.codecs)?;
        let (audio, video) = pick_streams(demuxer.streams());
        let duration = audio
            .as_ref()
            .or(video.as_ref())
            .and_then(|s| s.duration.map(|d| secs_of(s, d)));
        let format_name = demuxer.format_name().to_owned();

        let (audio_decoder, audio_sample_rate, audio_channels) = match (&audio, want_audio) {
            (Some(s), true) => {
                let dec = registries.codecs.make_decoder(&s.params)?;
                (
                    Some(dec),
                    s.params.sample_rate.unwrap_or(48_000),
                    s.params.channels.unwrap_or(2),
                )
            }
            (Some(_), false) | (None, _) => (None, 48_000, 2),
        };

        let (video_decoder, video_dims) = match (&video, want_video) {
            (Some(s), true) => match registries.codecs.make_decoder(&s.params) {
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
            (Some(_), false) | (None, _) => (None, None),
        };

        // Tell the demuxer which streams we actually care about so it
        // can skip the others at the container level when possible.
        let mut active: Vec<u32> = Vec::new();
        if audio_decoder.is_some() {
            if let Some(a) = audio.as_ref() {
                active.push(a.index);
            }
        }
        if video_decoder.is_some() {
            if let Some(v) = video.as_ref() {
                active.push(v.index);
            }
        }
        if !active.is_empty() {
            demuxer.set_active_streams(&active);
        }

        let driver = build_driver(audio_sample_rate, audio_channels, video_dims)?;

        let opened = OpenedMedia {
            audio: audio.clone(),
            video: video.clone(),
            duration,
            format_name,
        };

        // Only hand the stream indices the worker should actually
        // route. Passing `None` for a stream index makes the demux
        // thread discard those packets without sending them to a
        // decoder (see `DemuxCtx::run`).
        let audio_idx = audio.as_ref().filter(|_| audio_decoder.is_some()).map(|s| s.index);
        let video_idx = video.as_ref().filter(|_| video_decoder.is_some()).map(|s| s.index);
        let worker =
            DecodeWorker::spawn(demuxer, audio_decoder, video_decoder, audio_idx, video_idx);

        Ok((
            Self {
                driver,
                worker,
                audio_stream: audio,
                video_stream: video,
                video_queue: VecDeque::new(),
                clock_origin: Duration::ZERO,
                clock_baseline_samples: 0,
                output_sample_rate: audio_sample_rate,
                paused: false,
                volume: 1.0,
                eof: false,
                seek_pending: false,
                last_audio_end: Duration::ZERO,
                last_video_pts: None,
                last_video_presented_pts: None,
                seek_tb: None,
            },
            opened,
        ))
    }

    /// Snapshot of per-stream timestamps for the TUI's drift display.
    pub fn timings(&self) -> PlayerTimings {
        fn to_dur(pts: Option<i64>, s: Option<&StreamInfo>) -> Option<Duration> {
            let (p, s) = (pts?, s?);
            let secs = s.time_base.seconds_of(p);
            if secs.is_finite() && secs >= 0.0 {
                Some(Duration::from_secs_f64(secs))
            } else {
                None
            }
        }
        PlayerTimings {
            master: self.position(),
            audio: if self.last_audio_end > Duration::ZERO {
                Some(self.last_audio_end)
            } else {
                None
            },
            video_decoded: to_dur(self.last_video_pts, self.video_stream.as_ref()),
            video_presented: to_dur(self.last_video_presented_pts, self.video_stream.as_ref()),
            video_queue_len: self.video_queue.len(),
        }
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
    ///
    /// Seeks prefer the video stream so the demuxer's "last keyframe ≤
    /// target" logic snaps the landing pts to the nearest I-frame.
    /// That matters because on seek the decoder wipes its reference
    /// state; resuming on a P-frame would produce garbage until the
    /// next I-frame. Audio has no equivalent constraint, so if there's
    /// no video we seek by audio.
    pub fn seek_to(&mut self, target: Duration) -> Result<()> {
        let (stream_idx, tb) = if let Some(v) = &self.video_stream {
            (v.index, v.time_base)
        } else if let Some(a) = &self.audio_stream {
            (a.index, a.time_base)
        } else {
            return Err(Error::unsupported("nothing to seek"));
        };
        let pts = (target.as_secs_f64() / tb.as_rational().as_f64()).round() as i64;
        // Clear the video queue — anything in it is pre-seek.
        self.video_queue.clear();
        if !self.worker.seek(stream_idx, pts) {
            return Err(Error::other("decode worker exited"));
        }
        self.seek_pending = true;
        self.seek_tb = Some(tb);
        self.eof = false;
        Ok(())
    }

    /// Called when the worker emits [`DecodedUnit::Seeked`]. Recomputes
    /// the master-clock origin so `position()` lines up with the
    /// landed pts (which has been snapped to a keyframe by the demuxer).
    fn on_seeked(&mut self, landed_pts: i64) {
        let tb = self.seek_tb.take().or_else(|| {
            self.video_stream
                .as_ref()
                .map(|s| s.time_base)
                .or_else(|| self.audio_stream.as_ref().map(|s| s.time_base))
        });
        let landed_dur = match tb {
            Some(tb) => {
                let s = tb.seconds_of(landed_pts);
                if s.is_finite() && s > 0.0 {
                    Duration::from_secs_f64(s)
                } else {
                    Duration::ZERO
                }
            }
            None => Duration::ZERO,
        };
        self.clock_baseline_samples =
            self.driver
                .master_clock_pos()
                .as_secs_f64()
                .max(0.0)
                .mul_add(self.output_sample_rate as f64, 0.0) as u64;
        self.clock_origin = landed_dur;
        // Audio-ahead bookkeeping is cumulative. After a seek, fresh
        // audio frames start flowing from the new landing pts, so
        // rebase so the drift display doesn't read "A -20s" until
        // last_audio_end catches up naturally.
        self.last_audio_end = landed_dur;
        self.seek_pending = false;
    }

    /// Drive one loop iteration:
    ///   1. Drain everything the decode worker has produced into our
    ///      audio/video queues (audio goes straight to SDL's queue,
    ///      video into `self.video_queue`).
    ///   2. Pop any video frames whose pts has reached the wallclock
    ///      and present them.
    ///
    /// Returns `Ok(true)` if something was moved; `Ok(false)` when
    /// paused / EOF / nothing available.
    pub fn pump_once(&mut self) -> Result<bool> {
        let mut activity = false;

        // Phase 1 — drain the worker's output channels.
        //
        // Both streams are throttled with a "soft cap" check so the
        // player stops pulling before the downstream would drop or
        // balloon:
        //
        //  * Audio: `audio_headroom_samples()` — how many samples the
        //    backend can still accept. Once the ringbuf / SDL queue
        //    has less than ~250 ms of headroom left we stop pulling
        //    audio, the decoder channel fills, the decoder blocks,
        //    the packet channel fills, and finally the demuxer blocks
        //    on audio routing. That's the natural backpressure chain
        //    that keeps front-loaded containers (non-interleaved
        //    AVI, MKV clusters with many audio blocks) from racing
        //    ahead and making queue_audio silently drop samples.
        //
        //  * Video: `video_queue.len()` — same shape, main-thread
        //    queue size instead of device headroom.
        //
        // Ctl messages always drain.
        let audio_rate = self.output_sample_rate.max(1) as u64;
        // ~250 ms of audio headroom kept as slack so brief decoder
        // stalls don't cause underruns. Once the backend reports
        // less than this free, we pause audio-pull.
        let audio_headroom_floor = audio_rate / 4;
        loop {
            let want_video = self.video_queue.len() < VIDEO_QUEUE_SOFT_CAP;
            let want_audio =
                self.driver.audio_headroom_samples() > audio_headroom_floor;
            let unit = self.worker.try_recv_subset(want_audio, want_video);
            let Some(unit) = unit else { break };
            activity = true;
            match unit {
                DecodedUnit::Audio(af) => {
                    if self.seek_pending {
                        // Pre-seek payload; discard.
                        continue;
                    }
                    if af.sample_rate > 0 {
                        self.last_audio_end +=
                            Duration::from_secs_f64(af.samples as f64 / af.sample_rate as f64);
                    }
                    self.driver.queue_audio(&af)?;
                }
                DecodedUnit::Video(vf) => {
                    if self.seek_pending {
                        continue;
                    }
                    if let Some(p) = vf.pts {
                        self.last_video_pts = Some(p);
                    }
                    self.video_queue.push_back(vf);
                }
                DecodedUnit::Ctl(DecodedCtl::Seeked(landed)) => {
                    self.on_seeked(landed);
                }
                DecodedUnit::Ctl(DecodedCtl::Eof) => {
                    self.eof = true;
                }
                DecodedUnit::Ctl(DecodedCtl::Err(msg)) => {
                    // Don't end playback on a single decoder hiccup — a
                    // bad packet at startup shouldn't kill the stream.
                    // The demux thread always follows any fatal exit
                    // with `Eof`, so real end-of-data still drains us.
                    eprintln!("oxideplay: decode worker error: {msg}");
                }
            }
        }

        if self.paused {
            return Ok(activity);
        }

        // Phase 2 — trim stale frames (post-decode drop) then present
        // at most ONE video frame per tick whose pts has reached the
        // wallclock. SDL's present path can take 5-20 ms per call
        // (YUV upload + renderer present); presenting several in a
        // row would block this thread for >100 ms, during which the
        // worker's bounded output channel fills, the audio decoder
        // blocks on send, and SDL underruns its audio queue.
        //
        // If the decoder has produced a lot of future frames the
        // extras sit in `video_queue` and come out one per tick.
        // Anything more than `VIDEO_FRAME_MAX_BEHIND` behind master is
        // dropped by `trim_video_queue` without paying the render
        // cost.
        self.trim_video_queue();
        let now = self.position();
        let video_tb = self.video_stream.as_ref().map(|s| s.time_base);
        let epsilon = Duration::from_millis(50);
        if let Some(vf) = self.video_queue.front() {
            let pts_secs = match (vf.pts, video_tb) {
                (Some(p), Some(tb)) => tb.seconds_of(p),
                _ => 0.0,
            };
            let target = if pts_secs.is_finite() && pts_secs > 0.0 {
                Duration::from_secs_f64(pts_secs)
            } else {
                Duration::ZERO
            };
            if target <= now + epsilon {
                let vf = self.video_queue.pop_front().unwrap();
                self.last_video_presented_pts = vf.pts;
                self.driver.present_video(&vf)?;
                activity = true;
            }
        }

        Ok(activity)
    }

    /// Post-decode drop: walk the front of the video queue and pop any
    /// frames that have fallen more than `VIDEO_FRAME_MAX_BEHIND` behind
    /// the master clock. The decoder already saw those packets — the
    /// reference chain is intact — we just skip the render cost for
    /// frames that would land visibly late.
    ///
    /// Backpressure (not dropping) is what bounds queue size on the
    /// future side now; see `pump_once`.
    fn trim_video_queue(&mut self) {
        let Some(tb) = self.video_stream.as_ref().map(|s| s.time_base) else {
            return;
        };
        let now = self.position();
        while let Some(front) = self.video_queue.front() {
            let pts_secs = front.pts.map(|p| tb.seconds_of(p)).unwrap_or(0.0);
            let target = Duration::from_secs_f64(pts_secs.max(0.0));
            if target + VIDEO_FRAME_MAX_BEHIND < now {
                self.video_queue.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn eof_reached(&self) -> bool {
        self.eof
    }

    /// Has playback of the queued audio caught up to end-of-stream? Used
    /// to decide when to exit after demuxer EOF.
    ///
    /// At startup, before any audio has been queued, SDL's queue is empty
    /// which would otherwise report "drained". That combined with an
    /// early EOF (e.g. from a transient decoder error) used to exit
    /// playback instantly. Require `last_audio_end > 0` so "drained"
    /// means "had audio, now finished" rather than "never started".
    pub fn audio_drained(&self) -> bool {
        if self.audio_stream.is_none() {
            return true;
        }
        self.last_audio_end > Duration::ZERO && self.driver.audio_queue_len_samples() == 0
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

            // Drain decoded frames from the worker + present any due
            // video. Backpressure is the worker's bounded output
            // channel — when we stop consuming (pause, slow render),
            // the worker blocks inside its send(). No gating needed
            // here.
            let _ = self.pump_once()?;

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
