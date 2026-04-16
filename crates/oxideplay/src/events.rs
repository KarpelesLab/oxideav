//! Merge TUI + window events into one queue.
//!
//! This is deliberately tiny — the driver and TUI both hand back
//! `Vec<PlayerEvent>` and we just concat them.

use std::time::Duration;

use crate::driver::{OutputDriver, PlayerEvent};
use crate::tui;

/// Pull events from both the output driver (window focus) and the
/// terminal (when `tui_active` is true). The `tui_timeout` caps the
/// terminal-poll latency per call.
#[allow(dead_code)]
pub fn gather<D: OutputDriver>(
    driver: &mut D,
    tui_active: bool,
    tui_timeout: Duration,
) -> Vec<PlayerEvent> {
    let mut out = driver.poll_events();
    if tui_active {
        out.extend(tui::poll_events(tui_timeout));
    }
    out
}
