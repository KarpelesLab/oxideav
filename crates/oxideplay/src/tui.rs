//! Crossterm-based one-line TUI.
//!
//! Drawn on the final terminal row when stdout is a TTY. Falls through to
//! plain stderr progress lines otherwise.

use std::io::{self, IsTerminal, Write};
use std::time::Duration;

use crossterm::event::{poll, read, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{cursor, queue, style, terminal};

use crate::driver::{PlayerEvent, SeekDir};

/// Returns true when stdout looks like an interactive terminal.
pub fn stdout_is_tty() -> bool {
    io::stdout().is_terminal()
}

/// Idempotent terminal-setup guard. On drop, restores the terminal state.
pub struct TuiGuard {
    active: bool,
}

impl TuiGuard {
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut out = io::stdout();
        queue!(out, cursor::Hide)?;
        out.flush()?;
        Ok(Self { active: true })
    }

    #[allow(dead_code)]
    pub fn active(&self) -> bool {
        self.active
    }
}

impl Drop for TuiGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = disable_raw_mode();
        let mut out = io::stdout();
        let _ = queue!(
            out,
            cursor::Show,
            terminal::Clear(terminal::ClearType::CurrentLine),
            style::ResetColor
        );
        let _ = writeln!(out);
        let _ = out.flush();
    }
}

/// Render one status line at the current cursor row. Overwrites any
/// previous content on the line.
///
/// `drift` — when `Some`, a formatted per-stream timestamp string is
/// appended after the volume. The caller formats via
/// [`format_drift`] so this module doesn't depend on the player's
/// internal types.
pub fn draw_status(
    position: Duration,
    duration: Option<Duration>,
    paused: bool,
    volume: f32,
    seek_enabled: bool,
    drift: Option<&str>,
) -> io::Result<()> {
    let mut out = io::stdout();
    queue!(
        out,
        cursor::MoveToColumn(0),
        terminal::Clear(terminal::ClearType::CurrentLine)
    )?;
    let dur_str = duration.map(format_duration).unwrap_or_else(|| "?".into());
    let state = if paused { "PAUSED " } else { "PLAYING" };
    let hints = if seek_enabled {
        "[q]quit [space]pause [←/→]10s [↓/↑]1m [pgdn/pgup]10m [/ *]vol"
    } else {
        "[q]quit [space]pause [/ *]vol"
    };
    if let Some(d) = drift {
        write!(
            out,
            "{} {} / {}  vol {:>3}%  {}  {}",
            state,
            format_duration(position),
            dur_str,
            (volume * 100.0).round() as i32,
            d,
            hints,
        )?;
    } else {
        write!(
            out,
            "{} {} / {}  vol {:>3}%  {}",
            state,
            format_duration(position),
            dur_str,
            (volume * 100.0).round() as i32,
            hints,
        )?;
    }
    out.flush()?;
    Ok(())
}

/// Format a per-stream timing snapshot for the status line.
///
/// Output looks like `A +0.02 V +0.04 (dec +0.08, q=6)` — each
/// stream's offset relative to the master clock in seconds, with the
/// decoded-but-not-yet-presented video gap and the video queue depth
/// at the end.
pub fn format_drift(master: Duration, timings: &PlayerTimings) -> String {
    fn signed_secs(d: Duration, base: Duration) -> f64 {
        d.as_secs_f64() - base.as_secs_f64()
    }
    let mut out = String::new();
    out.push_str("A ");
    match timings.audio {
        Some(a) => out.push_str(&format!("{:+.2}", signed_secs(a, master))),
        None => out.push_str(" —  "),
    }
    out.push_str(" V ");
    match timings.video_presented {
        Some(v) => out.push_str(&format!("{:+.2}", signed_secs(v, master))),
        None => out.push_str(" —  "),
    }
    if let Some(v) = timings.video_decoded {
        out.push_str(&format!(
            " (dec {:+.2}, q={})",
            signed_secs(v, master),
            timings.video_queue_len
        ));
    }
    out
}

/// Subset of `player::PlayerTimings` the TUI cares about. Kept here
/// to avoid a circular dependency with the player module at compile
/// time — main.rs converts.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerTimings {
    pub audio: Option<Duration>,
    pub video_decoded: Option<Duration>,
    pub video_presented: Option<Duration>,
    pub video_queue_len: usize,
}

/// Format a Duration as `MM:SS.cc`.
pub fn format_duration(d: Duration) -> String {
    let total = d.as_secs_f64();
    let m = (total / 60.0) as u64;
    let s = total - (m as f64) * 60.0;
    format!("{:02}:{:05.2}", m, s)
}

/// Non-blocking poll of keyboard events from the terminal. Returns any
/// matched `PlayerEvent`s. `timeout == 0` means "drain any events that
/// are already pending without waiting"; a positive timeout blocks up
/// to that duration waiting for the first event, then drains.
pub fn poll_events(timeout: Duration) -> Vec<PlayerEvent> {
    let mut out = Vec::new();
    // First poll: honour the caller's requested timeout. If nothing is
    // ready within that window, return empty.
    match poll(timeout) {
        Ok(true) => {}
        _ => return out,
    }
    // Drain everything currently ready with zero-timeout follow-ups.
    loop {
        match read() {
            Ok(Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press | KeyEventKind::Repeat,
                ..
            })) => {
                if let Some(e) = map_key(code, modifiers) {
                    out.push(e);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
        match poll(Duration::ZERO) {
            Ok(true) => continue,
            _ => break,
        }
    }
    out
}

/// Pure key → event mapping, so it can be tested without a real terminal.
pub fn map_key(code: KeyCode, modifiers: KeyModifiers) -> Option<PlayerEvent> {
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    // In raw mode, Ctrl+C is delivered as a key event, not as SIGINT.
    // Intercept it explicitly so the player can exit.
    if ctrl {
        if let KeyCode::Char(c) = code {
            if c == 'c' || c == 'C' || c == 'd' || c == 'D' {
                return Some(PlayerEvent::Quit);
            }
        }
    }
    match code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => Some(PlayerEvent::Quit),
        KeyCode::Char(' ') => Some(PlayerEvent::TogglePause),
        KeyCode::Left => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(10),
            SeekDir::Back,
        )),
        KeyCode::Right => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(10),
            SeekDir::Forward,
        )),
        KeyCode::Up => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(60),
            SeekDir::Forward,
        )),
        KeyCode::Down => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(60),
            SeekDir::Back,
        )),
        KeyCode::PageUp => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(600),
            SeekDir::Forward,
        )),
        KeyCode::PageDown => Some(PlayerEvent::SeekRelative(
            Duration::from_secs(600),
            SeekDir::Back,
        )),
        KeyCode::Char('*') => Some(PlayerEvent::VolumeDelta(5)),
        KeyCode::Char('/') => Some(PlayerEvent::VolumeDelta(-5)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_basic() {
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00.00");
        assert_eq!(format_duration(Duration::from_secs(65)), "01:05.00");
        assert_eq!(format_duration(Duration::from_millis(12_340)), "00:12.34");
    }

    #[test]
    fn keybinds_quit() {
        assert_eq!(
            map_key(KeyCode::Char('q'), KeyModifiers::NONE),
            Some(PlayerEvent::Quit)
        );
        assert_eq!(
            map_key(KeyCode::Esc, KeyModifiers::NONE),
            Some(PlayerEvent::Quit)
        );
    }

    #[test]
    fn keybinds_seek_arrows_10s() {
        assert_eq!(
            map_key(KeyCode::Left, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(10),
                SeekDir::Back
            ))
        );
        assert_eq!(
            map_key(KeyCode::Right, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(10),
                SeekDir::Forward
            ))
        );
    }

    #[test]
    fn keybinds_seek_vertical_1min() {
        assert_eq!(
            map_key(KeyCode::Up, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(60),
                SeekDir::Forward
            ))
        );
        assert_eq!(
            map_key(KeyCode::Down, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(60),
                SeekDir::Back
            ))
        );
    }

    #[test]
    fn keybinds_seek_pageup_pagedown_10min() {
        assert_eq!(
            map_key(KeyCode::PageUp, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(600),
                SeekDir::Forward
            ))
        );
        assert_eq!(
            map_key(KeyCode::PageDown, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(600),
                SeekDir::Back
            ))
        );
    }

    #[test]
    fn keybinds_volume_slash_star() {
        assert_eq!(
            map_key(KeyCode::Char('*'), KeyModifiers::NONE),
            Some(PlayerEvent::VolumeDelta(5))
        );
        assert_eq!(
            map_key(KeyCode::Char('/'), KeyModifiers::NONE),
            Some(PlayerEvent::VolumeDelta(-5))
        );
    }

    #[test]
    fn keybinds_pause() {
        assert_eq!(
            map_key(KeyCode::Char(' '), KeyModifiers::NONE),
            Some(PlayerEvent::TogglePause)
        );
    }

    #[test]
    fn keybinds_ctrl_c_and_ctrl_d_quit() {
        // Raw mode delivers Ctrl+C and Ctrl+D as key events; the player
        // must exit on both (Ctrl+C is the canonical "interrupt",
        // Ctrl+D is the canonical EOF — either one should end playback).
        assert_eq!(
            map_key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(PlayerEvent::Quit)
        );
        assert_eq!(
            map_key(KeyCode::Char('C'), KeyModifiers::CONTROL),
            Some(PlayerEvent::Quit)
        );
        assert_eq!(
            map_key(KeyCode::Char('d'), KeyModifiers::CONTROL),
            Some(PlayerEvent::Quit)
        );
    }
}
