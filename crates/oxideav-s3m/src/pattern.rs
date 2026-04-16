//! S3M pattern unpacker.
//!
//! Unlike MOD's fixed-size rows, S3M packs patterns with a per-row
//! channel-mask scheme:
//!
//! ```text
//! pattern := 2-byte length  (includes the length word itself)
//!           { row }
//! row     := { record }  0x00
//! record  := flags_byte
//!            [ note | instrument ]   (if flags & 0x20)
//!              — 1 byte note (C-4 = 0x40 ish; 0xFE = note cut; 0xFF = empty)
//!              — 1 byte instrument (0 = none; 1..=99 = sample index)
//!            [ volume ]              (if flags & 0x40)  1 byte  0..=64 or 0xFF
//!            [ command cmd info ]    (if flags & 0x80)  2 bytes (ST3 letter 'A'-'Z' stored as 1..=26)
//! ```
//!
//! The low 5 bits (`flags & 0x1F`) give the channel index 0..=31.

use crate::header::{S3mHeader, PATTERN_ROWS};

/// A decoded cell for one channel at one row.
///
/// Zero fields mean "no change" — a note-on is signalled by `note > 0`,
/// an instrument change by `instrument > 0`, and an effect by `command >
/// 0` (1-based A..Z letters).
#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    /// S3M note encoding: high nibble = octave, low nibble = semitone 0..=11.
    /// Special: 0x00 = "no note", 0xFE = note cut, 0xFF = empty.
    pub note: u8,
    /// 1..=99 sample index; 0 = no change.
    pub instrument: u8,
    /// Volume 0..=64 or 0xFF = no change.
    pub volume: u8,
    /// Effect command as stored (1 = A, 2 = B, ... 26 = Z). 0 = no effect.
    pub command: u8,
    /// Effect parameter byte.
    pub info: u8,
}

impl Cell {
    pub const EMPTY: Cell = Cell {
        note: 0xFF,
        instrument: 0,
        volume: 0xFF,
        command: 0,
        info: 0,
    };

    pub fn has_note(&self) -> bool {
        self.note != 0xFF && self.note != 0x00
    }

    pub fn note_is_cut(&self) -> bool {
        self.note == 0xFE
    }
}

/// Decoded pattern: 64 rows × 32 channels.
#[derive(Clone, Debug)]
pub struct Pattern {
    pub rows: Vec<Vec<Cell>>, // rows[row][channel]
}

impl Pattern {
    pub fn empty(channels: usize) -> Self {
        let rows = (0..PATTERN_ROWS)
            .map(|_| vec![Cell::EMPTY; channels])
            .collect();
        Pattern { rows }
    }
}

/// Unpack one pattern from a byte buffer. `offset` points at the
/// 2-byte length header.
pub fn unpack_pattern(bytes: &[u8], offset: usize, channels: usize) -> Pattern {
    let mut pat = Pattern::empty(channels);
    if offset + 2 > bytes.len() {
        return pat;
    }
    let length = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as usize;
    // Packed body follows the 2-byte length.
    let body_start = offset + 2;
    let body_end = (offset + length).min(bytes.len());
    if body_start >= body_end {
        return pat;
    }
    let body = &bytes[body_start..body_end];

    let mut pos = 0usize;
    let mut row = 0usize;
    while row < PATTERN_ROWS && pos < body.len() {
        let flags = body[pos];
        pos += 1;
        if flags == 0 {
            row += 1;
            continue;
        }
        let channel = (flags & 0x1F) as usize;

        let mut note: u8 = 0xFF;
        let mut instrument: u8 = 0;
        let mut volume: u8 = 0xFF;
        let mut command: u8 = 0;
        let mut info: u8 = 0;

        if flags & 0x20 != 0 {
            if pos + 2 > body.len() {
                break;
            }
            note = body[pos];
            instrument = body[pos + 1];
            pos += 2;
        }
        if flags & 0x40 != 0 {
            if pos >= body.len() {
                break;
            }
            volume = body[pos];
            pos += 1;
        }
        if flags & 0x80 != 0 {
            if pos + 2 > body.len() {
                break;
            }
            command = body[pos];
            info = body[pos + 1];
            pos += 2;
        }

        if channel < channels {
            let cell = &mut pat.rows[row][channel];
            cell.note = note;
            cell.instrument = instrument;
            cell.volume = volume;
            cell.command = command;
            cell.info = info;
        }
    }
    pat
}

/// Unpack every pattern referenced by the header. A parapointer of 0
/// signals an empty / absent pattern in the wild, so we skip those and
/// emit a blank pattern.
pub fn unpack_all(header: &S3mHeader, bytes: &[u8]) -> Vec<Pattern> {
    let channels = header.channels.len();
    header
        .pattern_offsets
        .iter()
        .map(|&off| {
            if off == 0 {
                Pattern::empty(channels)
            } else {
                unpack_pattern(bytes, off as usize, channels)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_row_terminator_is_skipped() {
        // Two empty rows followed by a note on channel 0 of row 2.
        let mut buf = Vec::new();
        // 2-byte length placeholder.
        buf.extend_from_slice(&0u16.to_le_bytes());
        // Row 0: 0x00
        buf.push(0x00);
        // Row 1: 0x00
        buf.push(0x00);
        // Row 2: channel 0 + note-instrument flag + volume flag
        //        flags = 0x20 | 0x40 | 0 = 0x60
        buf.push(0x60);
        buf.push(0x40); // note C-4 (octave 4, semi 0)
        buf.push(1); // instrument 1
        buf.push(48); // volume
        buf.push(0x00); // end of row 2
                        // Patch the length field.
        let len = buf.len();
        buf[0..2].copy_from_slice(&(len as u16).to_le_bytes());

        let pat = unpack_pattern(&buf, 0, 32);
        assert_eq!(pat.rows.len(), 64);
        assert_eq!(pat.rows[2][0].note, 0x40);
        assert_eq!(pat.rows[2][0].instrument, 1);
        assert_eq!(pat.rows[2][0].volume, 48);
        // Row 0 and 1 remain empty cells.
        assert_eq!(pat.rows[0][0].note, 0xFF);
    }
}
