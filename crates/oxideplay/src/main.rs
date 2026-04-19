//! `oxideplay` — reference media player built on `oxideav`.
//!
//! This binary is the *only* place in the workspace where SDL2 (and its
//! transitive C dep via `sdl2-sys`) is allowed. The library half of
//! oxideav remains pure Rust.

mod decode_worker;
mod driver;
mod drivers;
mod events;
mod job_sink;
mod player;
mod tui;

use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::Parser;
use oxideav::Registries;
use oxideav_source::SourceRegistry;

use crate::driver::{OutputDriver, PlayerEvent};
use crate::drivers::sdl2_driver::Sdl2Driver;
use crate::player::{Player, DEFAULT_BUFFER_BYTES};

#[derive(Parser)]
#[command(
    name = "oxideplay",
    version,
    about = "Play a media file via the oxideav library (SDL2 audio + video)"
)]
struct Cli {
    /// Input media URI: a local path, file:// URL, or http(s):// URL.
    /// Not required when `--job` is given.
    #[arg(required_unless_present = "job")]
    input: Option<String>,

    /// Run a JSON-described job. The job must declare exactly one
    /// `@display` or `@out` sink — that sink is bound to SDL2.
    /// `-` reads the JSON from stdin.
    #[arg(long)]
    job: Option<String>,

    /// Probe the input, print stream info, and exit without touching SDL2.
    #[arg(long)]
    dry_run: bool,

    /// Start muted.
    #[arg(long)]
    mute: bool,

    /// Force audio-only mode even if the file has a video track.
    #[arg(long)]
    no_video: bool,

    /// Prefetch buffer size in MiB (default 64).
    #[arg(long, default_value_t = (DEFAULT_BUFFER_BYTES / (1 << 20)) as u32)]
    buffer_mib: u32,

    /// Thread budget for `--job` execution. `0` = auto (logical CPUs or
    /// the job's own `threads` field). Ignored in non-`--job` mode.
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("oxideplay: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Build the source registry with the file driver and (when compiled in)
/// HTTP/HTTPS support.
fn build_sources() -> SourceRegistry {
    let mut reg = SourceRegistry::with_defaults();
    #[cfg(feature = "http")]
    {
        oxideav::http::register(&mut reg);
    }
    reg
}

fn run(cli: Cli) -> oxideav_core::Result<()> {
    let registries = Registries::with_all_features();
    let sources = build_sources();

    if let Some(job_src) = cli.job.as_deref() {
        return run_job(
            &registries,
            &sources,
            job_src,
            cli.mute,
            !cli.no_video,
            cli.threads,
        );
    }

    let input = cli
        .input
        .as_deref()
        .ok_or_else(|| oxideav_core::Error::invalid("no input URI (pass a path or --job)"))?;

    if cli.dry_run {
        return dry_run(&registries, &sources, input);
    }

    let want_video = !cli.no_video;
    let buffer_bytes = (cli.buffer_mib as usize).saturating_mul(1 << 20);

    let (mut play, media) = Player::open(
        &registries,
        &sources,
        input,
        buffer_bytes,
        |sr, ch, video_dims| {
            let video_dims = if want_video { video_dims } else { None };
            Sdl2Driver::new(sr, ch, video_dims)
        },
    )?;

    if cli.mute {
        play.driver.set_volume(0.0);
    }

    // Print stream summary to stderr so stdout is free for the TUI.
    eprintln!(
        "oxideplay: playing {} (format: {}){}",
        input,
        media.format_name,
        match &media.duration {
            Some(d) => format!(", duration {}", tui::format_duration(*d)),
            None => String::new(),
        }
    );
    if let Some(a) = &media.audio {
        eprintln!(
            "  audio: {} {}ch @ {} Hz",
            a.params.codec_id,
            a.params.channels.unwrap_or(0),
            a.params.sample_rate.unwrap_or(0)
        );
    }
    if let Some(v) = &media.video {
        eprintln!(
            "  video: {} {}x{}",
            v.params.codec_id,
            v.params.width.unwrap_or(0),
            v.params.height.unwrap_or(0)
        );
    }

    let tty = tui::stdout_is_tty();
    let mut tui_guard: Option<tui::TuiGuard> = if tty {
        tui::TuiGuard::enter().ok()
    } else {
        None
    };

    let mut last_status = Instant::now();
    // Assume seek is supported; flipped off lazily in `apply_event` on
    // first `Unsupported` error from the demuxer.
    let mut seek_supported = true;

    let result = run_loop(
        &mut play,
        &media,
        &mut tui_guard,
        &mut last_status,
        &mut seek_supported,
    );

    // Explicitly drop TUI guard before exit so terminal is restored.
    drop(tui_guard);

    result
}

fn run_loop<D: OutputDriver>(
    play: &mut Player<D>,
    media: &player::OpenedMedia,
    tui_guard: &mut Option<tui::TuiGuard>,
    last_status: &mut Instant,
    seek_supported: &mut bool,
) -> oxideav_core::Result<()> {
    let tick_interval = Duration::from_millis(16);
    let status_interval = Duration::from_secs(1);

    loop {
        // Gather events from driver + tui. tui::poll_events with a
        // zero-duration first poll now drains any pending key events
        // properly (Duration::ZERO is fine — it's the "non-blocking
        // check" mode, and the rest of the loop keeps ticking).
        let mut events = play.driver.poll_events();
        if tui_guard.is_some() {
            events.extend(tui::poll_events(Duration::ZERO));
        }
        let mut keep_going = true;
        for ev in events {
            if !play.apply_event(ev, seek_supported) {
                keep_going = false;
                break;
            }
        }
        if !keep_going {
            break;
        }

        // Drain the decode worker on every tick. Back-pressure already
        // lives in the worker's bounded output channel — gating the
        // drain by the audio-queue depth here would starve the channel
        // and stall the audio decoder, which in turn starves SDL and
        // causes choppy playback.
        let _ = play.pump_once()?;

        // Pre-roll gate: keep the audio device paused until enough
        // samples are buffered (0.25 s at the stream's sample rate).
        // This avoids the "first samples glitch" where SDL starts
        // draining the queue before demux + decode have produced
        // anything. Keep this value below what MP2/AAC typical
        // frame sizes pile up to in <10 packets so the check triggers
        // reliably.
        let preroll_samples = media
            .audio
            .as_ref()
            .and_then(|a| a.params.sample_rate)
            .unwrap_or(48_000) as u64
            / 4;
        let _ = play.driver.try_start_audio(preroll_samples);

        if play.eof_reached() && play.audio_drained() && !play.paused() {
            break;
        }

        // Status output.
        let now = Instant::now();
        let snap = play.timings();
        let tui_snap = tui::PlayerTimings {
            audio: snap.audio,
            video_decoded: snap.video_decoded,
            video_presented: snap.video_presented,
            video_queue_len: snap.video_queue_len,
        };
        let drift_str = tui::format_drift(snap.master, &tui_snap);
        if now.duration_since(*last_status) >= status_interval {
            if tui_guard.is_some() {
                let _ = tui::draw_status(
                    play.position(),
                    media.duration,
                    play.paused(),
                    play.volume(),
                    *seek_supported,
                    Some(&drift_str),
                );
            } else {
                let dur = media
                    .duration
                    .map(tui::format_duration)
                    .unwrap_or_else(|| "?".into());
                eprintln!(
                    "oxideplay: {} / {}  vol {:>3}%  {}{}",
                    tui::format_duration(play.position()),
                    dur,
                    (play.volume() * 100.0).round() as i32,
                    drift_str,
                    if play.paused() { "  [paused]" } else { "" },
                );
            }
            *last_status = now;
        } else if tui_guard.is_some() {
            // Still update the status bar frequently so time ticks smoothly.
            let _ = tui::draw_status(
                play.position(),
                media.duration,
                play.paused(),
                play.volume(),
                *seek_supported,
                Some(&drift_str),
            );
        }

        // Sleep a tick.
        std::thread::sleep(tick_interval);
    }
    Ok(())
}

fn run_job(
    registries: &Registries,
    sources: &SourceRegistry,
    job_src: &str,
    mute: bool,
    want_video: bool,
    threads: usize,
) -> oxideav_core::Result<()> {
    use oxideav::job::{Executor, Job};

    // Load the job JSON.
    let raw = if job_src == "-" {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(job_src)?
    };
    let job = Job::from_json(&raw)?;
    job.validate()?;

    // Pick the first @display/@out target. No loop concurrency yet —
    // playback is fire-and-forget (no pause/seek).
    let target = ["@display", "@out"]
        .iter()
        .find(|k| job.outputs.contains_key(**k))
        .copied()
        .ok_or_else(|| {
            oxideav_core::Error::invalid(
                "oxideplay --job: expected a @display or @out output in the job",
            )
        })?;

    let sink = Box::new(job_sink::PlayerSink::new(mute, want_video));
    let stats = Executor::new(&job, &registries.codecs, &registries.containers, sources)
        .with_sink_override(target, sink)
        .with_threads(threads)
        .run()?;
    eprintln!(
        "oxideplay: job finished ({} pkts read, {} frames decoded, {} frames played)",
        stats.packets_read, stats.frames_decoded, stats.frames_written
    );
    Ok(())
}

fn dry_run(
    registries: &Registries,
    sources: &SourceRegistry,
    input: &str,
) -> oxideav_core::Result<()> {
    let media = player::probe(registries, sources, input)?;
    println!("Input: {input}");
    println!("Format: {}", media.format_name);
    if let Some(d) = media.duration {
        println!("Duration: {}", tui::format_duration(d));
    }
    if let Some(a) = &media.audio {
        println!(
            "Audio: stream #{} codec={} channels={} rate={}",
            a.index,
            a.params.codec_id,
            a.params.channels.unwrap_or(0),
            a.params.sample_rate.unwrap_or(0),
        );
    }
    if let Some(v) = &media.video {
        println!(
            "Video: stream #{} codec={} {}x{}",
            v.index,
            v.params.codec_id,
            v.params.width.unwrap_or(0),
            v.params.height.unwrap_or(0),
        );
    }
    if media.audio.is_none() && media.video.is_none() {
        println!("(no audio or video streams)");
    }
    let _ = PlayerEvent::Quit; // suppress unused warning
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_help_builds() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_parses_dry_run() {
        let cli = Cli::try_parse_from(["oxideplay", "--dry-run", "x.mp4"]).unwrap();
        assert!(cli.dry_run);
        assert_eq!(cli.input.as_deref(), Some("x.mp4"));
    }
}
