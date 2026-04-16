//! ST3 (S3M) playback engine.
//!
//! Per-channel state is kept in `Channel`. The mixer runs at a fixed
//! output sample rate (44.1 kHz), resampling each channel via linear
//! interpolation between adjacent sample frames, and applies per-channel
//! volume, global volume, and pan.
//!
//! Timing follows the ST3 conventions:
//! - `speed` = ticks per row (default 6).
//! - `bpm`   = tempo (default 125).
//! - samples_per_tick = `sample_rate * 2.5 / bpm`  (same formula as MOD).
//!
//! Output frequency (in Hz) for a given note N on an instrument whose
//! C-5 speed is `c5` is:
//!     freq = c5 * 2^((N - C5) / 12)
//! with N = 12 * octave + semitone. We compute this directly as a float.

use crate::header::{S3mHeader, PATTERN_ROWS};
use crate::pattern::{Cell, Pattern};
use crate::samples::SampleBody;

pub const DEFAULT_SPEED: u8 = 6;
pub const DEFAULT_BPM: u8 = 125;

/// Command letters from the ST3 spec.
/// Stored as 1..=26 in the pattern data; translating A=1, B=2, ... Z=26.
pub mod cmd {
    pub const A_SET_SPEED: u8 = 1;
    pub const B_POS_JUMP: u8 = 2;
    pub const C_PAT_BREAK: u8 = 3;
    pub const D_VOL_SLIDE: u8 = 4;
    pub const E_SLIDE_DOWN: u8 = 5;
    pub const F_SLIDE_UP: u8 = 6;
    pub const G_TONE_PORTA: u8 = 7;
    pub const H_VIBRATO: u8 = 8;
    pub const O_SAMPLE_OFFSET: u8 = 15;
    pub const T_SET_TEMPO: u8 = 20;
    pub const V_GLOBAL_VOL: u8 = 22;
}

/// Per-channel playback state.
#[derive(Clone, Debug)]
pub struct Channel {
    /// 1-based instrument index (0 = nothing triggered yet).
    pub instrument: u8,
    /// Current playback frequency in Hz (0 = silent).
    pub frequency: f32,
    /// Fractional read cursor into the sample body.
    pub sample_pos: f64,
    /// Current per-channel volume 0..=64.
    pub volume: u8,
    /// Pan value 0..=15 (0 = hard left, 15 = hard right).
    pub pan: u8,
    /// Whether this channel is currently emitting sound.
    pub active: bool,
    /// Remembered note for tone-portamento (G) target tracking.
    pub target_frequency: f32,
    /// Current effect command 1..=26 (0 = none).
    pub command: u8,
    /// Effect parameter byte.
    pub info: u8,
    /// Vibrato phase in table units (0..=63).
    pub vibrato_pos: u8,
}

impl Default for Channel {
    fn default() -> Self {
        Channel {
            instrument: 0,
            frequency: 0.0,
            sample_pos: 0.0,
            volume: 0,
            pan: 8,
            active: false,
            target_frequency: 0.0,
            command: 0,
            info: 0,
            vibrato_pos: 0,
        }
    }
}

/// Convert an S3M note byte (octave << 4 | semitone) and C5 speed (Hz)
/// into a playback frequency.
fn note_to_frequency(note: u8, c5_speed: u32) -> f32 {
    // Note index relative to C-5: ST3 convention is 5*12 = 60 is C-5.
    let octave = (note >> 4) as i32;
    let semitone = (note & 0x0F) as i32;
    let n = octave * 12 + semitone;
    let c5_n = 5 * 12;
    let delta = n - c5_n;
    // freq = c5_speed * 2^(delta/12).
    (c5_speed as f32) * 2.0f32.powf(delta as f32 / 12.0)
}

/// Simple 64-entry sine vibrato table (8-bit signed, range -64..=64).
fn vibrato_sine(pos: u8) -> i32 {
    // Classic ST3 vibrato table (approx sine). Using f32 to keep the
    // code small; precision isn't critical for smoke tests.
    let phase = (pos as f32) * (std::f32::consts::PI * 2.0 / 64.0);
    (phase.sin() * 64.0) as i32
}

/// Top-level player state.
pub struct PlayerState {
    pub samples: Vec<SampleBody>,
    pub patterns: Vec<Pattern>,
    pub order: Vec<u8>,

    pub channels: Vec<Channel>,
    /// Initial pan values copied from the header (0..=15).
    pub initial_pan: Vec<u8>,

    pub speed: u8,
    pub bpm: u8,
    pub global_volume: u8,

    pub order_index: u8,
    pub row: u8,
    pub tick: u8,
    pub tick_sample_cursor: u32,

    pub sample_rate: u32,
    pub ended: bool,

    pending_jump: Option<Jump>,
}

#[derive(Clone, Copy, Debug)]
struct Jump {
    order: Option<u8>,
    row: u8,
}

impl PlayerState {
    pub fn new(
        header: &S3mHeader,
        samples: Vec<SampleBody>,
        patterns: Vec<Pattern>,
        sample_rate: u32,
    ) -> Self {
        let n_channels = header.channels.len();
        let channels = (0..n_channels)
            .map(|i| Channel {
                pan: header.pans.get(i).copied().unwrap_or(8) & 0x0F,
                ..Channel::default()
            })
            .collect();
        let initial_pan = header.pans.to_vec();

        let speed = if header.initial_speed == 0 {
            DEFAULT_SPEED
        } else {
            header.initial_speed
        };
        let bpm = if header.initial_tempo == 0 {
            DEFAULT_BPM
        } else {
            header.initial_tempo
        };

        PlayerState {
            samples,
            patterns,
            order: header.order.clone(),
            channels,
            initial_pan,
            speed,
            bpm,
            global_volume: header.global_volume.min(64),
            order_index: 0,
            row: 0,
            tick: 0,
            tick_sample_cursor: 0,
            sample_rate,
            ended: false,
            pending_jump: None,
        }
    }

    pub fn samples_per_tick(&self) -> u32 {
        ((self.sample_rate as f32) * 2.5 / self.bpm.max(1) as f32) as u32
    }

    fn find_next_playable_order(&mut self) -> Option<u8> {
        // Walk past 0xFE (marker) entries; stop at 0xFF (end).
        while (self.order_index as usize) < self.order.len() {
            let v = self.order[self.order_index as usize];
            if v == 0xFF {
                return None;
            }
            if v == 0xFE {
                self.order_index = self.order_index.saturating_add(1);
                continue;
            }
            return Some(v);
        }
        None
    }

    fn enter_row(&mut self) {
        let pat_idx = match self.find_next_playable_order() {
            Some(v) => v as usize,
            None => {
                self.ended = true;
                return;
            }
        };
        if pat_idx >= self.patterns.len() {
            self.ended = true;
            return;
        }
        let row_cells: Vec<Cell> = self.patterns[pat_idx].rows[self.row as usize].clone();

        let mut row_speed: Option<u8> = None;
        let mut row_tempo: Option<u8> = None;
        let mut row_global_vol: Option<u8> = None;
        let mut row_jump: Option<Jump> = None;

        for (ch_idx, cell) in row_cells.iter().enumerate() {
            if ch_idx >= self.channels.len() {
                break;
            }
            let ch = &mut self.channels[ch_idx];
            ch.command = cell.command;
            ch.info = cell.info;

            // Instrument change reloads volume.
            if cell.instrument != 0 {
                ch.instrument = cell.instrument;
                if let Some(s) = self.samples.get(cell.instrument as usize - 1) {
                    ch.volume = s.volume;
                }
            }

            // Note cut.
            if cell.note == 0xFE {
                ch.active = false;
                ch.frequency = 0.0;
            } else if cell.note != 0xFF {
                // Trigger.
                let inst_idx = ch.instrument as usize;
                if inst_idx > 0 && inst_idx <= self.samples.len() {
                    let c5 = self.samples[inst_idx - 1].c5_speed.max(1);
                    let freq = note_to_frequency(cell.note, c5);
                    // Tone portamento (G): don't retrigger, set target.
                    if ch.command == cmd::G_TONE_PORTA && ch.frequency > 0.0 {
                        ch.target_frequency = freq;
                    } else {
                        ch.frequency = freq;
                        ch.target_frequency = freq;
                        // Re-apply Oxx sample offset if present.
                        if ch.command == cmd::O_SAMPLE_OFFSET {
                            let off = (ch.info as u64) * 256;
                            ch.sample_pos = off as f64;
                        } else {
                            ch.sample_pos = 0.0;
                        }
                        ch.active = true;
                        ch.vibrato_pos = 0;
                    }
                }
            }

            // Explicit volume column.
            if cell.volume != 0xFF {
                ch.volume = cell.volume.min(64);
            }

            // Tick-0 effects (instant / row-level).
            match ch.command {
                cmd::A_SET_SPEED => {
                    if ch.info != 0 {
                        row_speed = Some(ch.info);
                    }
                }
                cmd::B_POS_JUMP => {
                    row_jump = Some(Jump {
                        order: Some(ch.info),
                        row: 0,
                    });
                }
                cmd::C_PAT_BREAK => {
                    // Parameter is BCD (high nibble * 10 + low).
                    let r = ((ch.info >> 4) * 10 + (ch.info & 0x0F)).min(63);
                    row_jump = Some(Jump {
                        order: None,
                        row: r,
                    });
                }
                cmd::T_SET_TEMPO => {
                    if ch.info >= 0x20 {
                        row_tempo = Some(ch.info);
                    }
                }
                cmd::V_GLOBAL_VOL => {
                    row_global_vol = Some(ch.info.min(64));
                }
                _ => {}
            }
        }

        if let Some(s) = row_speed {
            self.speed = s;
        }
        if let Some(t) = row_tempo {
            self.bpm = t;
        }
        if let Some(g) = row_global_vol {
            self.global_volume = g;
        }
        if row_jump.is_some() {
            self.pending_jump = row_jump;
        }
    }

    fn apply_per_tick(&mut self) {
        for ch in &mut self.channels {
            let x = ch.info >> 4;
            let y = ch.info & 0x0F;
            match ch.command {
                cmd::D_VOL_SLIDE => {
                    // Dxy: volume slide. Fine slides (Dx0 where x=0xF or
                    // D0y where y=0xF) happen on tick 0 only and are
                    // skipped here; normal slides happen on tick > 0.
                    if x == 0xF || y == 0xF {
                        // Fine slide — ignore per-tick (TODO: tick-0 fine).
                    } else if x != 0 && y == 0 {
                        ch.volume = (ch.volume as u16 + x as u16).min(64) as u8;
                    } else if y != 0 && x == 0 {
                        ch.volume = ch.volume.saturating_sub(y);
                    }
                }
                cmd::E_SLIDE_DOWN => {
                    // Exx: portamento down. Each tick: freq *= 2^(-param/768).
                    // Fine / extra-fine slides (0xEy / 0xFy) are tick-0 only
                    // — skip on per-tick path.
                    if ch.info != 0 && ch.info < 0xE0 {
                        let f = 2.0f32.powf(-(ch.info as f32) / 768.0);
                        ch.frequency *= f;
                    }
                }
                cmd::F_SLIDE_UP => {
                    if ch.info != 0 && ch.info < 0xE0 {
                        let f = 2.0f32.powf((ch.info as f32) / 768.0);
                        ch.frequency *= f;
                    }
                }
                cmd::G_TONE_PORTA => {
                    // Gxx: slide toward target at rate info/per tick.
                    if ch.info != 0 && ch.target_frequency > 0.0 {
                        let step = ch.info as f32;
                        if ch.frequency < ch.target_frequency {
                            let f = 2.0f32.powf(step / 768.0);
                            ch.frequency = (ch.frequency * f).min(ch.target_frequency);
                        } else if ch.frequency > ch.target_frequency {
                            let f = 2.0f32.powf(-step / 768.0);
                            ch.frequency = (ch.frequency * f).max(ch.target_frequency);
                        }
                    }
                }
                cmd::H_VIBRATO => {
                    // Hxy: vibrato. x = speed, y = depth.
                    let speed = x;
                    let depth = y;
                    if speed != 0 || depth != 0 {
                        ch.vibrato_pos = (ch.vibrato_pos.wrapping_add(speed * 4)) & 0x3F;
                        let delta = (vibrato_sine(ch.vibrato_pos) * depth as i32) / 128;
                        let mult = 2.0f32.powf(delta as f32 / 48.0);
                        // Apply on a copy so we don't accumulate drift.
                        // We rebase off target_frequency (last triggered note).
                        if ch.target_frequency > 0.0 {
                            ch.frequency = ch.target_frequency * mult;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn advance_tick(&mut self) {
        if self.tick == 0 {
            self.enter_row();
        } else {
            self.apply_per_tick();
        }
    }

    fn next_row(&mut self) {
        if let Some(jump) = self.pending_jump.take() {
            if let Some(order) = jump.order {
                self.order_index = order;
            } else {
                self.order_index = self.order_index.saturating_add(1);
            }
            self.row = jump.row;
        } else {
            self.row += 1;
            if self.row as usize >= PATTERN_ROWS {
                self.row = 0;
                self.order_index = self.order_index.saturating_add(1);
            }
        }
        if self.order_index as usize >= self.order.len() {
            self.ended = true;
        }
    }

    /// Mix one sample from one channel. Returns (left, right) in -1..=1.
    fn mix_channel(ch: &mut Channel, samples: &[SampleBody], out_rate: f32) -> (f32, f32) {
        if !ch.active || ch.frequency <= 0.0 {
            return (0.0, 0.0);
        }
        let idx = ch.instrument as usize;
        if idx == 0 || idx > samples.len() {
            return (0.0, 0.0);
        }
        let body = &samples[idx - 1];
        if body.pcm.is_empty() {
            return (0.0, 0.0);
        }

        let len = body.pcm.len() as f64;
        if ch.sample_pos >= len {
            if body.is_looped() {
                let ls = body.loop_start as f64;
                let le = body.loop_end as f64;
                let span = le - ls;
                if span > 0.0 {
                    let over = ch.sample_pos - ls;
                    ch.sample_pos = ls + over.rem_euclid(span);
                } else {
                    ch.active = false;
                    return (0.0, 0.0);
                }
            } else {
                ch.active = false;
                return (0.0, 0.0);
            }
        }

        let i = ch.sample_pos as usize;
        let frac = (ch.sample_pos - i as f64) as f32;
        let n = body.pcm.len();
        let s0 = body.pcm[i.min(n - 1)] as f32 / 32768.0;
        let next_idx = if i + 1 < n {
            i + 1
        } else if body.is_looped() {
            body.loop_start as usize
        } else {
            i
        };
        let s1 = body.pcm[next_idx.min(n - 1)] as f32 / 32768.0;
        let interp = s0 + (s1 - s0) * frac;

        let v = (ch.volume as f32) / 64.0;
        let out = interp * v;

        // Advance position.
        let step = (ch.frequency as f64) / (out_rate as f64);
        ch.sample_pos += step;

        // Pan: 0 = left, 15 = right. Equal-power-ish linear split.
        let pan = (ch.pan as f32) / 15.0;
        let left = out * (1.0 - pan);
        let right = out * pan;
        (left, right)
    }

    fn render_one(&mut self, out: &mut [i16]) {
        let out_rate = self.sample_rate as f32;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        let n_ch = self.channels.len();
        for ch in &mut self.channels {
            let (cl, cr) = Self::mix_channel(ch, &self.samples, out_rate);
            l += cl;
            r += cr;
        }
        let norm = (n_ch as f32 / 2.0).max(1.0);
        let gv = (self.global_volume as f32) / 64.0;
        l = (l / norm * gv).clamp(-1.0, 1.0);
        r = (r / norm * gv).clamp(-1.0, 1.0);
        out[0] = (l * 32767.0) as i16;
        out[1] = (r * 32767.0) as i16;
    }

    /// Render up to `dst.len()/2` stereo frames.
    pub fn render(&mut self, dst: &mut [i16]) -> usize {
        assert!(dst.len() % 2 == 0);
        let mut produced = 0usize;
        let total_frames = dst.len() / 2;
        while produced < total_frames {
            if self.ended {
                break;
            }
            if self.tick_sample_cursor == 0 {
                self.advance_tick();
                if self.ended {
                    break;
                }
            }
            let spt = self.samples_per_tick().max(1);
            let remaining_in_tick = spt.saturating_sub(self.tick_sample_cursor);
            let want = (total_frames - produced).min(remaining_in_tick as usize);
            for _ in 0..want {
                let off = produced * 2;
                self.render_one(&mut dst[off..off + 2]);
                produced += 1;
            }
            self.tick_sample_cursor += want as u32;
            if self.tick_sample_cursor >= spt {
                self.tick_sample_cursor = 0;
                self.tick += 1;
                if self.tick >= self.speed {
                    self.tick = 0;
                    self.next_row();
                }
            }
        }
        produced
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn note_freq_c5_is_c5_speed() {
        // C-5 = octave 5 semi 0 = 0x50.
        let f = note_to_frequency(0x50, 8363);
        assert!((f - 8363.0).abs() < 0.5, "got {}", f);
    }

    #[test]
    fn note_freq_c6_doubles() {
        let f5 = note_to_frequency(0x50, 8363);
        let f6 = note_to_frequency(0x60, 8363);
        assert!((f6 / f5 - 2.0).abs() < 0.001);
    }
}
