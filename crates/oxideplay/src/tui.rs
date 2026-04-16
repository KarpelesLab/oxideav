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
pub fn draw_status(
    position: Duration,
    duration: Option<Duration>,
    paused: bool,
    volume: f32,
    seek_enabled: bool,
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
        "[q]quit [space]pause [←/→]5s [shift+←/→]30s [↑/↓]vol"
    } else {
        "[q]quit [space]pause [↑/↓]vol"
    };
    write!(
        out,
        "{} {} / {}  vol {:>3}%  {}",
        state,
        format_duration(position),
        dur_str,
        (volume * 100.0).round() as i32,
        hints,
    )?;
    out.flush()?;
    Ok(())
}

/// Format a Duration as `MM:SS.cc`.
pub fn format_duration(d: Duration) -> String {
    let total = d.as_secs_f64();
    let m = (total / 60.0) as u64;
    let s = total - (m as f64) * 60.0;
    format!("{:02}:{:05.2}", m, s)
}

/// Non-blocking poll of keyboard events from the terminal. Returns any
/// matched `PlayerEvent`s.
pub fn poll_events(timeout: Duration) -> Vec<PlayerEvent> {
    let mut out = Vec::new();
    // Don't block the player loop; cap by given timeout.
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            break;
        }
        let remain = deadline - now;
        match poll(remain) {
            Ok(true) => {}
            _ => break,
        }
        match read() {
            Ok(Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press,
                ..
            })) => {
                if let Some(e) = map_key(code, modifiers) {
                    out.push(e);
                }
            }
            Ok(Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Repeat,
                ..
            })) => {
                if let Some(e) = map_key(code, modifiers) {
                    out.push(e);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    out
}

/// Pure key → event mapping, so it can be tested without a real terminal.
pub fn map_key(code: KeyCode, modifiers: KeyModifiers) -> Option<PlayerEvent> {
    let shift = modifiers.contains(KeyModifiers::SHIFT);
    match code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => Some(PlayerEvent::Quit),
        KeyCode::Char(' ') => Some(PlayerEvent::TogglePause),
        KeyCode::Left => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Back))
        }
        KeyCode::Right => {
            let d = if shift {
                Duration::from_secs(30)
            } else {
                Duration::from_secs(5)
            };
            Some(PlayerEvent::SeekRelative(d, SeekDir::Forward))
        }
        KeyCode::Up => Some(PlayerEvent::VolumeDelta(5)),
        KeyCode::Down => Some(PlayerEvent::VolumeDelta(-5)),
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
    fn keybinds_seek() {
        assert_eq!(
            map_key(KeyCode::Left, KeyModifiers::NONE),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(5),
                SeekDir::Back
            ))
        );
        assert_eq!(
            map_key(KeyCode::Right, KeyModifiers::SHIFT),
            Some(PlayerEvent::SeekRelative(
                Duration::from_secs(30),
                SeekDir::Forward
            ))
        );
    }

    #[test]
    fn keybinds_volume() {
        assert_eq!(
            map_key(KeyCode::Up, KeyModifiers::NONE),
            Some(PlayerEvent::VolumeDelta(5))
        );
        assert_eq!(
            map_key(KeyCode::Down, KeyModifiers::NONE),
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
}
