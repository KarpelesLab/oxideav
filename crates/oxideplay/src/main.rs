//! `oxideplay` — reference media player built on `oxideav`.
//!
//! This binary is the *only* place in the workspace where SDL2 (and its
//! transitive C dep via `sdl2-sys`) is allowed. The library half of
//! oxideav remains pure Rust.

mod driver;
mod drivers;
mod events;
mod player;
mod tui;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::Parser;
use oxideav::Registries;

use crate::driver::{OutputDriver, PlayerEvent};
use crate::drivers::sdl2_driver::Sdl2Driver;
use crate::player::Player;

#[derive(Parser)]
#[command(
    name = "oxideplay",
    version,
    about = "Play a media file via the oxideav library (SDL2 audio + video)"
)]
struct Cli {
    /// Input media file.
    input: PathBuf,

    /// Probe the file, print stream info, and exit without touching SDL2.
    #[arg(long)]
    dry_run: bool,

    /// Start muted.
    #[arg(long)]
    mute: bool,

    /// Force audio-only mode even if the file has a video track.
    #[arg(long)]
    no_video: bool,
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

fn run(cli: Cli) -> oxideav_core::Result<()> {
    let registries = Registries::with_all_features();

    if cli.dry_run {
        return dry_run(&registries, &cli.input);
    }

    let want_video = !cli.no_video;

    let (mut play, media) = Player::open(&registries, &cli.input, |sr, ch, video_dims| {
        let video_dims = if want_video { video_dims } else { None };
        Sdl2Driver::new(sr, ch, video_dims)
    })?;

    if cli.mute {
        play.driver.set_volume(0.0);
    }

    // Print stream summary to stderr so stdout is free for the TUI.
    eprintln!(
        "oxideplay: playing {} (format: {}){}",
        cli.input.display(),
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
    let max_buffer = Duration::from_secs(2);

    loop {
        // Gather events from driver + tui.
        let mut events = play.driver.poll_events();
        if tui_guard.is_some() {
            events.extend(tui::poll_events(Duration::from_millis(0)));
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

        // Pump the pipeline.
        let buffered_secs = Duration::from_secs_f64(
            play.driver.audio_queue_len_samples() as f64
                / media
                    .audio
                    .as_ref()
                    .and_then(|a| a.params.sample_rate)
                    .unwrap_or(48_000)
                    .max(1) as f64,
        );
        if !play.paused() && !play.eof_reached() && buffered_secs < max_buffer {
            let _ = play.pump_once()?;
        }

        if play.eof_reached() && play.audio_drained() && !play.paused() {
            break;
        }

        // Status output.
        let now = Instant::now();
        if now.duration_since(*last_status) >= status_interval {
            if tui_guard.is_some() {
                let _ = tui::draw_status(
                    play.position(),
                    media.duration,
                    play.paused(),
                    play.volume(),
                    *seek_supported,
                );
            } else {
                let dur = media
                    .duration
                    .map(tui::format_duration)
                    .unwrap_or_else(|| "?".into());
                eprintln!(
                    "oxideplay: {} / {}  vol {:>3}%{}",
                    tui::format_duration(play.position()),
                    dur,
                    (play.volume() * 100.0).round() as i32,
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
            );
        }

        // Sleep a tick.
        std::thread::sleep(tick_interval);
    }
    Ok(())
}

fn dry_run(registries: &Registries, input: &std::path::Path) -> oxideav_core::Result<()> {
    let media = player::probe(registries, input)?;
    println!("Input: {}", input.display());
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
        assert_eq!(cli.input.to_string_lossy(), "x.mp4");
    }
}
