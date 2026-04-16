//! Scream Tracker 3 Module (S3M) header parser.
//!
//! Unlike MOD, S3M is little-endian and uses "parapointers" (paragraph
//! pointers) — 16-bit values that must be left-shifted by 4 to obtain a
//! byte offset. The header layout:
//!
//! ```text
//! 0x00  28 bytes  Song name (null-padded ASCII)
//! 0x1C   1 byte   0x1A (EOF / end-of-text marker)
//! 0x1D   1 byte   Type (0x10 = S3M)
//! 0x1E   2 bytes  Reserved
//! 0x20   2 bytes  OrdNum   — entries in order table (even count)
//! 0x22   2 bytes  InsNum   — number of instruments
//! 0x24   2 bytes  PatNum   — number of patterns
//! 0x26   2 bytes  Flags
//! 0x28   2 bytes  CwtV     — tracker version
//! 0x2A   2 bytes  FFI      — file format info (1 = signed samples, 2 = unsigned)
//! 0x2C   4 bytes  "SCRM"   — signature
//! 0x30   1 byte   GV       — global volume
//! 0x31   1 byte   IS       — initial speed (ticks per row)
//! 0x32   1 byte   IT       — initial tempo (BPM)
//! 0x33   1 byte   MV       — master volume (bit 7 = stereo)
//! 0x34   1 byte   UC       — ultra-click removal
//! 0x35   1 byte   DP       — default pan flag (0xFC = use pan values below)
//! 0x36   8 bytes  Reserved
//! 0x3E   2 bytes  Special  — parapointer to special data (unused)
//! 0x40  32 bytes  Channel settings — 0..=7 left, 8..=15 right, 16..=31 adlib, 0xFF = disabled
//! 0x60  OrdNum    Order table — 0xFE = marker, 0xFF = end
//! ...   InsNum*2  Instrument parapointer table
//! ...   PatNum*2  Pattern parapointer table
//! ...   (optional) 32 bytes of default pan values (if DP == 0xFC)
//! ```
//!
//! Each instrument is 80 bytes starting at its parapointer, and each
//! pattern starts with a 2-byte length followed by packed rows (see
//! `pattern.rs`).

use oxideav_core::{Error, Result};

pub const S3M_SIGNATURE: &[u8; 4] = b"SCRM";
pub const PATTERN_ROWS: usize = 64;
pub const CHANNEL_COUNT: usize = 32;
pub const INSTRUMENT_HEADER_SIZE: usize = 80;

/// Sample type codes in the instrument header.
pub const INST_TYPE_EMPTY: u8 = 0;
pub const INST_TYPE_PCM: u8 = 1;
pub const INST_TYPE_ADLIB_MELODY: u8 = 2;
// 3..=7: AdLib drum types (ignored for now).

/// Sample flag bits (byte 0x1F of instrument header).
pub const SAMPLE_FLAG_LOOP: u8 = 0x01;
pub const SAMPLE_FLAG_STEREO: u8 = 0x02;
pub const SAMPLE_FLAG_16BIT: u8 = 0x04;

/// An S3M instrument / sample definition (80 bytes in the file).
#[derive(Clone, Debug, Default)]
pub struct Instrument {
    /// 1 = PCM sample, 0 = empty, 2..=7 = AdLib (unsupported).
    pub kind: u8,
    /// Original DOS filename (12 bytes).
    pub dos_name: String,
    /// Parapointer to sample data (shift-left-by-4 to get byte offset).
    /// High byte at offset 0x0D, low word at 0x0E..0x10 (LE).
    pub sample_parapointer: u32,
    /// Length in samples.
    pub length: u32,
    /// Loop start in samples.
    pub loop_start: u32,
    /// Loop end in samples.
    pub loop_end: u32,
    /// Default volume 0..=64.
    pub volume: u8,
    /// Packing scheme (should be 0 for uncompressed PCM).
    pub pack: u8,
    /// Flags: bit0 loop, bit1 stereo, bit2 16-bit.
    pub flags: u8,
    /// C5 (middle-C) playback rate in Hz.
    pub c5_speed: u32,
    /// Instrument display name (28 bytes).
    pub name: String,
    /// Last 4 bytes should be "SCRS" for PCM, "SCRI" for AdLib.
    pub tag: [u8; 4],
}

impl Instrument {
    pub fn is_pcm(&self) -> bool {
        self.kind == INST_TYPE_PCM
    }

    pub fn is_looped(&self) -> bool {
        self.flags & SAMPLE_FLAG_LOOP != 0 && self.loop_end > self.loop_start
    }

    pub fn is_16bit(&self) -> bool {
        self.flags & SAMPLE_FLAG_16BIT != 0
    }

    pub fn is_stereo(&self) -> bool {
        self.flags & SAMPLE_FLAG_STEREO != 0
    }

    /// Byte offset where sample bytes begin.
    pub fn sample_byte_offset(&self) -> usize {
        (self.sample_parapointer as usize) << 4
    }
}

/// Parsed S3M top-level header + order table + parapointer tables +
/// channel settings + default pan.
#[derive(Clone, Debug)]
pub struct S3mHeader {
    pub song_name: String,
    pub ord_num: u16,
    pub ins_num: u16,
    pub pat_num: u16,
    pub flags: u16,
    pub tracker_version: u16,
    /// 1 = signed samples, 2 = unsigned samples.
    pub ffi: u16,
    pub global_volume: u8,
    pub initial_speed: u8,
    pub initial_tempo: u8,
    pub master_volume: u8,
    /// True if bit 7 of master_volume is set.
    pub stereo: bool,
    /// Default-pan flag — 0xFC means the 32 pan bytes at end of header are valid.
    pub default_pan_flag: u8,
    /// Channel settings, 32 entries: bit 7 = disabled, low bits = map.
    pub channels: [u8; CHANNEL_COUNT],
    /// Pan values for 32 channels (0..=15); populated from the default
    /// pan block or synthesized from `channels` (left/right bank split).
    pub pans: [u8; CHANNEL_COUNT],
    /// Raw order list (0xFE marker rows and 0xFF end markers preserved).
    pub order: Vec<u8>,
    /// Per-instrument definitions (parsed from parapointers).
    pub instruments: Vec<Instrument>,
    /// Per-pattern byte offsets in the file (shifted parapointers).
    pub pattern_offsets: Vec<u32>,
    /// Number of enabled (non-0xFF) channels used by the module.
    pub enabled_channels: u8,
}

fn read_u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn read_u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn read_padded_ascii(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end])
        .trim_end()
        .to_string()
}

/// Validate the S3M signature and parse the full header.
pub fn parse_header(bytes: &[u8]) -> Result<S3mHeader> {
    if bytes.len() < 0x60 {
        return Err(Error::invalid("S3M: file shorter than minimum header"));
    }
    // 'SCRM' signature at 0x2C.
    if &bytes[0x2C..0x30] != S3M_SIGNATURE {
        return Err(Error::invalid(
            "S3M: missing 'SCRM' signature at offset 0x2C",
        ));
    }
    // Type byte at 0x1D must be 0x10.
    if bytes[0x1D] != 0x10 {
        return Err(Error::invalid(format!(
            "S3M: expected type byte 0x10, got 0x{:02X}",
            bytes[0x1D]
        )));
    }

    let song_name = read_padded_ascii(&bytes[0x00..0x1C]);
    let ord_num = read_u16_le(bytes, 0x20);
    let ins_num = read_u16_le(bytes, 0x22);
    let pat_num = read_u16_le(bytes, 0x24);
    let flags = read_u16_le(bytes, 0x26);
    let tracker_version = read_u16_le(bytes, 0x28);
    let ffi = read_u16_le(bytes, 0x2A);
    let global_volume = bytes[0x30];
    let initial_speed = bytes[0x31];
    let initial_tempo = bytes[0x32];
    let master_volume_raw = bytes[0x33];
    let default_pan_flag = bytes[0x35];

    let master_volume = master_volume_raw & 0x7F;
    let stereo = (master_volume_raw & 0x80) != 0;

    let mut channels = [0u8; CHANNEL_COUNT];
    channels.copy_from_slice(&bytes[0x40..0x40 + CHANNEL_COUNT]);

    // Order table starts at 0x60.
    let ord_start = 0x60usize;
    let ord_end = ord_start + ord_num as usize;
    if bytes.len() < ord_end {
        return Err(Error::invalid("S3M: file shorter than order table"));
    }
    let order: Vec<u8> = bytes[ord_start..ord_end].to_vec();

    // Instrument parapointer table.
    let ins_table_start = ord_end;
    let ins_table_end = ins_table_start + ins_num as usize * 2;
    if bytes.len() < ins_table_end {
        return Err(Error::invalid(
            "S3M: truncated instrument parapointer table",
        ));
    }
    let mut instruments = Vec::with_capacity(ins_num as usize);
    for i in 0..ins_num as usize {
        let pp_off = ins_table_start + i * 2;
        let parapointer = read_u16_le(bytes, pp_off) as u32;
        let inst_byte_off = (parapointer as usize) << 4;
        instruments.push(parse_instrument(bytes, inst_byte_off)?);
    }

    // Pattern parapointer table.
    let pat_table_start = ins_table_end;
    let pat_table_end = pat_table_start + pat_num as usize * 2;
    if bytes.len() < pat_table_end {
        return Err(Error::invalid("S3M: truncated pattern parapointer table"));
    }
    let mut pattern_offsets = Vec::with_capacity(pat_num as usize);
    for i in 0..pat_num as usize {
        let pp_off = pat_table_start + i * 2;
        let parapointer = read_u16_le(bytes, pp_off) as u32;
        pattern_offsets.push(parapointer << 4);
    }

    // Default pan block (32 bytes) if default_pan_flag == 0xFC.
    let mut pans = [0u8; CHANNEL_COUNT];
    if default_pan_flag == 0xFC && bytes.len() >= pat_table_end + CHANNEL_COUNT {
        for (i, p) in pans.iter_mut().enumerate() {
            let raw = bytes[pat_table_end + i];
            // Low nibble is pan 0..=15 if bit 5 is set; else default.
            *p = if raw & 0x20 != 0 { raw & 0x0F } else { 0x08 };
        }
    } else {
        // Derive from channel settings: 0..=7 → left (0x03), 8..=15 → right (0x0C).
        for (i, c) in channels.iter().enumerate() {
            if *c == 0xFF {
                pans[i] = 0x08; // doesn't matter, channel disabled
            } else {
                let slot = c & 0x0F;
                pans[i] = if slot < 8 { 0x03 } else { 0x0C };
            }
        }
    }

    let enabled_channels = channels.iter().filter(|&&c| c != 0xFF && c < 16).count() as u8;

    Ok(S3mHeader {
        song_name,
        ord_num,
        ins_num,
        pat_num,
        flags,
        tracker_version,
        ffi,
        global_volume,
        initial_speed,
        initial_tempo,
        master_volume,
        stereo,
        default_pan_flag,
        channels,
        pans,
        order,
        instruments,
        pattern_offsets,
        enabled_channels,
    })
}

/// Parse a single 80-byte instrument header starting at `off`.
///
/// Layout:
/// ```text
/// 0x00   1 byte   Type (1 = PCM)
/// 0x01  12 bytes  DOS filename
/// 0x0D   1 byte   MemSeg hi
/// 0x0E   2 bytes  MemSeg lo (LE)  — combined: (hi << 16) | lo = parapointer
/// 0x10   4 bytes  Length (LE)
/// 0x14   4 bytes  Loop start (LE)
/// 0x18   4 bytes  Loop end (LE)
/// 0x1C   1 byte   Default volume
/// 0x1D   1 byte   Reserved
/// 0x1E   1 byte   Pack (0 = unpacked)
/// 0x1F   1 byte   Flags
/// 0x20   4 bytes  C5 speed (LE)
/// 0x24  12 bytes  Reserved
/// 0x30  28 bytes  Sample name
/// 0x4C   4 bytes  "SCRS" tag (for PCM)
/// ```
pub fn parse_instrument(bytes: &[u8], off: usize) -> Result<Instrument> {
    if off == 0 {
        // Parapointer of 0 means "empty slot".
        return Ok(Instrument::default());
    }
    if bytes.len() < off + INSTRUMENT_HEADER_SIZE {
        return Err(Error::invalid("S3M: truncated instrument header"));
    }
    let h = &bytes[off..off + INSTRUMENT_HEADER_SIZE];
    let kind = h[0];
    let dos_name = read_padded_ascii(&h[0x01..0x0D]);
    let mem_hi = h[0x0D] as u32;
    let mem_lo = read_u16_le(h, 0x0E) as u32;
    let sample_parapointer = (mem_hi << 16) | mem_lo;
    let length = read_u32_le(h, 0x10);
    let loop_start = read_u32_le(h, 0x14);
    let loop_end = read_u32_le(h, 0x18);
    let volume = h[0x1C].min(64);
    let pack = h[0x1E];
    let flags = h[0x1F];
    let c5_speed = read_u32_le(h, 0x20);
    let name = read_padded_ascii(&h[0x30..0x4C]);
    let mut tag = [0u8; 4];
    tag.copy_from_slice(&h[0x4C..0x50]);
    Ok(Instrument {
        kind,
        dos_name,
        sample_parapointer,
        length,
        loop_start,
        loop_end,
        volume,
        pack,
        flags,
        c5_speed,
        name,
        tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_signature() {
        let mut bytes = vec![0u8; 0x100];
        bytes[0x1D] = 0x10;
        // Leave SCRM missing.
        assert!(parse_header(&bytes).is_err());
    }

    #[test]
    fn rejects_wrong_type_byte() {
        let mut bytes = vec![0u8; 0x100];
        bytes[0x2C..0x30].copy_from_slice(S3M_SIGNATURE);
        bytes[0x1D] = 0x00;
        assert!(parse_header(&bytes).is_err());
    }
}
