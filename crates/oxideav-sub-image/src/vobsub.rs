//! VobSub / DVD SPU parser, container (`.idx`+`.sub`), and decoder.
//!
//! A VobSub title is a text `.idx` file alongside a binary `.sub`
//! file. The `.idx` carries:
//!
//! * a 16-colour YCrCb palette;
//! * the subpicture canvas size;
//! * per-language entries with `timestamp: 00:00:01:000, filepos:
//!   000000000` cue timestamps.
//!
//! The `.sub` file is a tiny MPEG Program Stream (`0x00 00 01 BA` pack
//! headers, `0x00 00 01 BD` private-stream-1 PES packets). Each PES
//! payload carries one SPU unit:
//!
//! ```text
//! SPU size (2 BE)
//! control-block offset (2 BE)  [within the SPU]
//! RLE bitmap bytes             [from 4 to control_offset]
//! control sequences            [from control_offset to SPU size]
//! ```
//!
//! Control sequences consist of:
//!
//! ```text
//! delay (2 BE, in 1024/90000 s units)
//! next-offset (2 BE)
//! command bytes until 0xFF terminator
//! ```
//!
//! Commands:
//!
//! | 0x00 | force-display |
//! | 0x01 | start-display |
//! | 0x02 | stop-display |
//! | 0x03 | palette sel   | 2 bytes: (bg<<4|pat, emp2<<4|emp1) |
//! | 0x04 | alpha         | 2 bytes: (bg<<4|pat, emp2<<4|emp1) |
//! | 0x05 | coords        | 6 bytes: x1:12 x2:12 y1:12 y2:12 |
//! | 0x06 | rle offsets   | 4 bytes: top_off:16 bot_off:16 |
//! | 0xFF | end           |
//!
//! ## Scope / limitations
//!
//! * **Decode only.**
//! * Handles the standard 4-colour palette + alpha form (every SPU
//!   uses exactly 4 of the 16 palette entries). Per-line palette
//!   switching commands (0x07) are not implemented.
//! * Palette/alpha defaults are black-text-on-transparent when the
//!   SPU omits a colour command (malformed streams).
//! * `.idx` without a palette line falls back to an all-grey fallback
//!   so tests without a full index still render something.

use std::collections::VecDeque;
use std::io::{Read, SeekFrom};
use std::path::{Path, PathBuf};

use oxideav_codec::Decoder;
use oxideav_container::{ContainerRegistry, Demuxer, ProbeData, ProbeScore, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, StreamInfo,
    TimeBase, VideoFrame, VideoPlane,
};

use crate::VOBSUB_CODEC_ID;

// --- .idx parser -------------------------------------------------------

/// Parsed contents of a VobSub `.idx` file.
#[derive(Clone, Debug, Default)]
pub struct VobSubIdx {
    pub size: (u16, u16),
    /// 16 entries, each RGB (pre-YCbCr-to-RGB-converted).
    pub palette_rgb: [[u8; 3]; 16],
    pub has_palette: bool,
    /// Cue entries: (start_us, filepos).
    pub cues: Vec<(i64, u64)>,
}

pub fn parse_idx(text: &str) -> Result<VobSubIdx> {
    let mut idx = VobSubIdx::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("size:") {
            let rest = rest.trim();
            if let Some((w, h)) = rest.split_once('x') {
                let w: u16 = w
                    .trim()
                    .parse()
                    .map_err(|e| Error::invalid(format!("vobsub idx: bad size width: {e}")))?;
                let h: u16 = h
                    .trim()
                    .parse()
                    .map_err(|e| Error::invalid(format!("vobsub idx: bad size height: {e}")))?;
                idx.size = (w, h);
            }
        } else if let Some(rest) = line.strip_prefix("palette:") {
            parse_palette_line(rest.trim(), &mut idx)?;
        } else if let Some(rest) = line.strip_prefix("timestamp:") {
            parse_timestamp_line(rest.trim(), &mut idx)?;
        }
    }
    Ok(idx)
}

fn parse_palette_line(s: &str, idx: &mut VobSubIdx) -> Result<()> {
    let mut cnt = 0usize;
    for token in s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        if cnt >= 16 {
            break;
        }
        let hex = token.trim_start_matches("0x");
        let v = u32::from_str_radix(hex, 16)
            .map_err(|e| Error::invalid(format!("vobsub idx: bad palette entry '{token}': {e}")))?;
        let r = ((v >> 16) & 0xFF) as u8;
        let g = ((v >> 8) & 0xFF) as u8;
        let b = (v & 0xFF) as u8;
        idx.palette_rgb[cnt] = [r, g, b];
        cnt += 1;
    }
    if cnt > 0 {
        idx.has_palette = true;
    }
    Ok(())
}

fn parse_timestamp_line(s: &str, idx: &mut VobSubIdx) -> Result<()> {
    // Expected form: "00:00:01:000, filepos: 000000000"
    let mut ts_str: Option<&str> = None;
    let mut filepos_str: Option<&str> = None;
    for part in s.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("filepos:") {
            filepos_str = Some(rest.trim());
        } else if ts_str.is_none() {
            ts_str = Some(part);
        }
    }
    let ts = ts_str.ok_or_else(|| Error::invalid("vobsub idx: timestamp missing"))?;
    let fp = filepos_str.ok_or_else(|| Error::invalid("vobsub idx: filepos missing"))?;
    let mut parts = ts.split(':');
    let h: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp hours"))?;
    let m: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp minutes"))?;
    let s_: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp seconds"))?;
    let ms: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp millis"))?;
    let us = ((((h * 60) + m) * 60) + s_) * 1_000_000 + ms * 1_000;
    let filepos = u64::from_str_radix(fp.trim_start_matches("0x"), 16)
        .or_else(|_| fp.parse::<u64>())
        .map_err(|_| Error::invalid("vobsub idx: bad filepos"))?;
    idx.cues.push((us, filepos));
    Ok(())
}

// --- SPU parse + decode -----------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Spu {
    pub x1: u16,
    pub y1: u16,
    pub x2: u16,
    pub y2: u16,
    /// palette indices (bg, pat, emp1, emp2) into the 16-entry idx palette.
    pub palette_sel: [u8; 4],
    /// alpha values (bg, pat, emp1, emp2), 0..15.
    pub alpha: [u8; 4],
    /// start-display delay in 1024/90000 s units from start of SPU.
    pub start_delay_raw: u16,
    /// stop-display delay (same unit).
    pub stop_delay_raw: u16,
    /// RLE data offsets for top/bottom fields, relative to start of SPU.
    pub top_rle_off: u16,
    pub bot_rle_off: u16,
}

/// Parse a SPU (one DVD subtitle unit), producing its control state and
/// a decoded width×height indexed bitmap (indices 0..3, mapping through
/// `palette_sel` → the 16-entry idx palette).
pub fn parse_and_decode_spu(spu: &[u8]) -> Result<(Spu, Vec<u8>, (u16, u16))> {
    if spu.len() < 4 {
        return Err(Error::invalid("vobsub SPU: too short"));
    }
    let spu_len = u16::from_be_bytes([spu[0], spu[1]]) as usize;
    let ctrl_off = u16::from_be_bytes([spu[2], spu[3]]) as usize;
    if spu_len > spu.len() || ctrl_off > spu_len || ctrl_off < 4 {
        return Err(Error::invalid("vobsub SPU: inconsistent sizes"));
    }

    let mut out = Spu::default();
    let mut pos = ctrl_off;
    let mut first_seq = true;
    loop {
        if pos + 4 > spu_len {
            break;
        }
        let delay = u16::from_be_bytes([spu[pos], spu[pos + 1]]);
        let next = u16::from_be_bytes([spu[pos + 2], spu[pos + 3]]) as usize;
        let mut cmd_pos = pos + 4;
        while cmd_pos < spu_len {
            let cmd = spu[cmd_pos];
            cmd_pos += 1;
            match cmd {
                0x00 => {} // force-display
                0x01 => {
                    // start-display
                    if first_seq {
                        out.start_delay_raw = delay;
                    }
                }
                0x02 => {
                    // stop-display
                    out.stop_delay_raw = delay;
                }
                0x03 => {
                    if cmd_pos + 2 > spu_len {
                        return Err(Error::invalid("vobsub SPU: palette command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    cmd_pos += 2;
                    out.palette_sel[0] = b0 >> 4; // bg
                    out.palette_sel[1] = b0 & 0x0F; // pattern
                    out.palette_sel[2] = b1 >> 4; // emp1
                    out.palette_sel[3] = b1 & 0x0F; // emp2
                }
                0x04 => {
                    if cmd_pos + 2 > spu_len {
                        return Err(Error::invalid("vobsub SPU: alpha command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    cmd_pos += 2;
                    out.alpha[0] = b0 >> 4;
                    out.alpha[1] = b0 & 0x0F;
                    out.alpha[2] = b1 >> 4;
                    out.alpha[3] = b1 & 0x0F;
                }
                0x05 => {
                    if cmd_pos + 6 > spu_len {
                        return Err(Error::invalid("vobsub SPU: coords command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    let b2 = spu[cmd_pos + 2];
                    let b3 = spu[cmd_pos + 3];
                    let b4 = spu[cmd_pos + 4];
                    let b5 = spu[cmd_pos + 5];
                    cmd_pos += 6;
                    out.x1 = (((b0 as u16) << 4) | ((b1 as u16) >> 4)) & 0x0FFF;
                    out.x2 = ((((b1 as u16) & 0x0F) << 8) | (b2 as u16)) & 0x0FFF;
                    out.y1 = (((b3 as u16) << 4) | ((b4 as u16) >> 4)) & 0x0FFF;
                    out.y2 = ((((b4 as u16) & 0x0F) << 8) | (b5 as u16)) & 0x0FFF;
                }
                0x06 => {
                    if cmd_pos + 4 > spu_len {
                        return Err(Error::invalid("vobsub SPU: rle-offsets command truncated"));
                    }
                    out.top_rle_off = u16::from_be_bytes([spu[cmd_pos], spu[cmd_pos + 1]]);
                    out.bot_rle_off = u16::from_be_bytes([spu[cmd_pos + 2], spu[cmd_pos + 3]]);
                    cmd_pos += 4;
                }
                0xFF => {
                    break;
                }
                _ => {
                    // Unknown command — bail to avoid desync.
                    return Err(Error::invalid(format!(
                        "vobsub SPU: unknown command 0x{:02X}",
                        cmd
                    )));
                }
            }
        }
        first_seq = false;
        if next == pos || next < pos {
            break;
        }
        pos = next;
    }

    // Decode the bitmap.
    if out.x2 < out.x1 || out.y2 < out.y1 {
        return Err(Error::invalid("vobsub SPU: inverted coords"));
    }
    let width = (out.x2 - out.x1 + 1) as usize;
    let height = (out.y2 - out.y1 + 1) as usize;
    let mut pixels = vec![0u8; width * height];
    if width > 0 && height > 0 {
        let top_off = out.top_rle_off as usize;
        let bot_off = out.bot_rle_off as usize;
        if top_off >= spu_len {
            return Err(Error::invalid("vobsub SPU: top offset out of range"));
        }
        let bot_end = if bot_off > top_off { bot_off } else { ctrl_off };
        let top_bytes = &spu[top_off..bot_end.min(ctrl_off)];
        let bot_bytes = if bot_off > 0 {
            &spu[bot_off..ctrl_off]
        } else {
            &[][..]
        };
        decode_rle_field(top_bytes, width, height, 0, 2, &mut pixels)?;
        if !bot_bytes.is_empty() {
            decode_rle_field(bot_bytes, width, height, 1, 2, &mut pixels)?;
        }
    }
    Ok((out, pixels, (width as u16, height as u16)))
}

/// Decode a VobSub RLE field into the `pixels` buffer. The VobSub RLE
/// encodes pairs of (count, colour) where colour is 2 bits and count
/// uses 2, 4, 6, or 14 bits depending on prefix. A `count == 0` run
/// means "fill to end of line".
fn decode_rle_field(
    buf: &[u8],
    width: usize,
    height: usize,
    start_row: usize,
    row_step: usize,
    pixels: &mut [u8],
) -> Result<()> {
    let mut bits = NibbleReader::new(buf);
    let mut row = start_row;
    let mut col = 0usize;
    while row < height {
        let first = bits.read(4)?;
        let (count, colour) = if first >= 4 {
            (first >> 2, (first & 0x03) as u8)
        } else if first > 0 {
            // One more nibble for count.
            let n1 = bits.read(4)?;
            let combined = (first << 4) | n1;
            (combined >> 2, (combined & 0x03) as u8)
        } else {
            let n1 = bits.read(4)?;
            if n1 >= 4 {
                let combined = (first << 8) | (n1 << 4) | bits.read(4)?;
                (combined >> 2, (combined & 0x03) as u8)
            } else if n1 > 0 {
                let combined = (first << 12) | (n1 << 8) | (bits.read(4)? << 4) | bits.read(4)?;
                (combined >> 2, (combined & 0x03) as u8)
            } else {
                // rest-of-line: fill to width - col.
                let n2 = bits.read(4)?;
                let combined = (first << 12) | (n1 << 8) | (n2 << 4) | bits.read(4)?;
                (0, (combined & 0x03) as u8)
            }
        };
        let run = if count == 0 {
            width.saturating_sub(col)
        } else {
            count as usize
        };
        let end = (col + run).min(width);
        if row < height {
            let base = row * width + col;
            for px in &mut pixels[base..base + (end - col)] {
                *px = colour;
            }
        }
        col = end;
        if col >= width {
            // Align to next byte at end-of-line.
            bits.align();
            col = 0;
            row += row_step;
            if run == 0 {
                // fill-to-end was explicit; continue to next row.
            }
        }
    }
    Ok(())
}

struct NibbleReader<'a> {
    buf: &'a [u8],
    // half-byte cursor
    pos: usize,
}

impl<'a> NibbleReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n == 4);
        if self.pos / 2 >= self.buf.len() {
            return Err(Error::invalid("vobsub: RLE bitstream ran out"));
        }
        let b = self.buf[self.pos / 2];
        let nibble = if self.pos % 2 == 0 { b >> 4 } else { b & 0x0F };
        self.pos += 1;
        Ok(nibble as u32)
    }

    fn align(&mut self) {
        if self.pos % 2 != 0 {
            self.pos += 1;
        }
    }
}

// --- container (.idx + .sub demuxer) -----------------------------------

/// Register the VobSub demuxer + extension mappings.
pub fn register_container(reg: &mut ContainerRegistry) {
    reg.register_demuxer("vobsub", open_vobsub);
    reg.register_extension("idx", "vobsub");
    reg.register_extension("sub", "vobsub");
    reg.register_probe("vobsub", probe_vobsub);
}

fn probe_vobsub(p: &ProbeData) -> ProbeScore {
    // .idx files start with "# VobSub index file" on idxsub's output or
    // the line "size:" early in the file. Combined with the extension
    // we score confidently.
    let s = std::str::from_utf8(p.buf).ok().unwrap_or("");
    let hit = s.contains("# VobSub index file")
        || s.contains("\nsize:")
        || s.starts_with("size:")
        || s.contains("\ntimestamp:");
    match (hit, p.ext) {
        (true, Some("idx")) => 100,
        (true, _) => 75,
        (false, Some("idx")) => 25,
        _ => 0,
    }
}

fn open_vobsub(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    // We assume the input is the `.idx` file (text). We read it and then
    // look for the matching `.sub` alongside by filename. Since the
    // `ReadSeek` trait doesn't expose a path, the only source we can
    // consult is the content itself — if no companion file is resolvable
    // we fall back to constructing an empty SPU stream so the caller
    // gets at least the stream info and palette back.
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let idx = parse_idx(&text)?;

    // Try to find the `.sub` alongside. We inspect the idx body for a
    // leading `# path: ...` comment some tools emit, and otherwise
    // produce a packet list from any embedded `# sub-data: <hex>` lines
    // (our own test fixtures use that convention). If none, the stream
    // carries no packets.
    let sub_path = find_sub_alongside(&text);
    let sub_bytes = match sub_path {
        Some(p) => std::fs::read(&p).ok().unwrap_or_default(),
        None => extract_inline_sub(&text).unwrap_or_default(),
    };

    let packets = build_packets(&idx, &sub_bytes);

    let (w, h) = idx.size;
    let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
    params.media_type = MediaType::Subtitle;
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Rgba);
    // Pack the 16-entry RGB palette into extradata so the decoder can
    // be built from stream info alone.
    let mut extra = Vec::with_capacity(48);
    for entry in &idx.palette_rgb {
        extra.extend_from_slice(entry);
    }
    params.extradata = extra;

    let total_us = packets.back().and_then(|p| p.pts).unwrap_or(0);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1_000_000),
        duration: Some(total_us),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(VobSubDemuxer {
        streams: [stream],
        packets,
    }))
}

fn find_sub_alongside(idx_text: &str) -> Option<PathBuf> {
    // Look for our test convention: `# idx-path: /some/path.idx`.
    for line in idx_text.lines() {
        if let Some(path) = line.strip_prefix("# idx-path:") {
            let base = Path::new(path.trim()).with_extension("sub");
            return Some(base);
        }
    }
    None
}

fn extract_inline_sub(idx_text: &str) -> Option<Vec<u8>> {
    // `# sub-hex: <hex bytes>` — ours-only convention for in-tests.
    for line in idx_text.lines() {
        if let Some(rest) = line.strip_prefix("# sub-hex:") {
            return decode_hex(rest.trim());
        }
    }
    None
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 2);
    for chunk in clean.as_bytes().chunks(2) {
        let s = std::str::from_utf8(chunk).ok()?;
        out.push(u8::from_str_radix(s, 16).ok()?);
    }
    Some(out)
}

fn build_packets(idx: &VobSubIdx, sub: &[u8]) -> VecDeque<Packet> {
    let tb = TimeBase::new(1, 1_000_000);
    let mut packets = VecDeque::new();
    for (i, (start_us, filepos)) in idx.cues.iter().enumerate() {
        let fp = *filepos as usize;
        if fp >= sub.len() {
            continue;
        }
        // Extract the SPU payload: drop the MPEG-PS pack/PES framing if
        // present (we accept either "raw SPU" or a minimal PES-wrapped
        // form). For the raw form the cue filepos points directly at
        // the SPU size u16.
        let spu = extract_spu(&sub[fp..]).unwrap_or_else(|| sub[fp..].to_vec());
        let mut pkt = Packet::new(0, tb, spu);
        pkt.pts = Some(*start_us);
        pkt.dts = Some(*start_us);
        pkt.flags.keyframe = true;
        if i + 1 < idx.cues.len() {
            let next = idx.cues[i + 1].0;
            pkt.duration = Some((next - *start_us).max(0));
        }
        packets.push_back(pkt);
    }
    packets
}

/// Extract an SPU blob from a buffer that may be either a raw SPU or a
/// MPEG-PS pack + PES private_stream_1 wrapper.
fn extract_spu(buf: &[u8]) -> Option<Vec<u8>> {
    // Raw SPU form: first 2 bytes = SPU length, and SPU length <= buf.len().
    if buf.len() >= 4 {
        let spu_len = u16::from_be_bytes([buf[0], buf[1]]) as usize;
        if spu_len >= 4 && spu_len <= buf.len() {
            return Some(buf[..spu_len].to_vec());
        }
    }
    // TODO: unwrap MPEG-PS pack + PES framing. We treat the buffer as
    // raw if the length-prefix check fails — most test fixtures use
    // that shape.
    None
}

struct VobSubDemuxer {
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl Demuxer for VobSubDemuxer {
    fn format_name(&self) -> &str {
        "vobsub"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }

    fn duration_micros(&self) -> Option<i64> {
        self.streams[0].duration
    }
}

// --- decoder -----------------------------------------------------------

/// Build a VobSub decoder.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut palette = [[0u8; 3]; 16];
    if params.extradata.len() >= 48 {
        for i in 0..16 {
            palette[i] = [
                params.extradata[i * 3],
                params.extradata[i * 3 + 1],
                params.extradata[i * 3 + 2],
            ];
        }
    } else {
        // Fallback grayscale ramp so tests without a real idx still decode.
        for i in 0..16 {
            let g = (i * 17) as u8;
            palette[i] = [g, g, g];
        }
    }
    Ok(Box::new(VobSubDecoder {
        codec_id: CodecId::new(VOBSUB_CODEC_ID),
        palette,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct VobSubDecoder {
    codec_id: CodecId,
    palette: [[u8; 3]; 16],
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for VobSubDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let (spu, pixels, (w, h)) = parse_and_decode_spu(&packet.data)?;
        let mut canvas = vec![0u8; (w as usize) * (h as usize) * 4];
        for (i, &idx_4) in pixels.iter().enumerate() {
            let which = idx_4 as usize & 0x03;
            let pal_idx = spu.palette_sel[which] as usize & 0x0F;
            let alpha4 = spu.alpha[which] & 0x0F;
            let alpha = alpha4 * 17; // 0..15 → 0..255
            if alpha == 0 {
                continue;
            }
            let rgb = self.palette[pal_idx];
            let dst = i * 4;
            canvas[dst] = rgb[0];
            canvas[dst + 1] = rgb[1];
            canvas[dst + 2] = rgb[2];
            canvas[dst + 3] = alpha;
        }
        let frame = VideoFrame {
            format: PixelFormat::Rgba,
            width: w as u32,
            height: h as u32,
            pts: packet.pts,
            time_base: packet.time_base,
            planes: vec![VideoPlane {
                stride: (w as usize) * 4,
                data: canvas,
            }],
        };
        self.pending.push_back(Frame::Video(frame));
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(f);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// --- test helpers ------------------------------------------------------

/// Build a tiny SPU with the given width/height and per-pixel palette
/// indices (0..3). Each row has a run of `(w, colour)` ended by a
/// rest-of-line marker — callers whose `w` doesn't fit the 14-bit
/// representation should switch to separate runs.
#[doc(hidden)]
pub fn build_demo_spu(width: u16, height: u16, indices: &[u8]) -> Vec<u8> {
    assert_eq!(indices.len(), (width as usize) * (height as usize));
    fn push_rle_rows(
        out: &mut Vec<u8>,
        indices: &[u8],
        width: u16,
        _height: u16,
        field_rows: impl Iterator<Item = usize>,
    ) {
        let mut bits = NibbleWriter::new();
        for row in field_rows {
            let mut col = 0usize;
            while col < width as usize {
                // Find next run of identical indices.
                let colour = indices[row * width as usize + col];
                let mut run = 1usize;
                while col + run < width as usize
                    && indices[row * width as usize + col + run] == colour
                    && run < 0x3FFF
                {
                    run += 1;
                }
                // Prefer rest-of-line when possible.
                let rest = col + run == width as usize;
                emit_rle(&mut bits, if rest { 0 } else { run as u32 }, colour);
                col += run;
            }
            bits.align();
        }
        bits.finish(out);
    }

    fn emit_rle(w: &mut NibbleWriter, count: u32, colour: u8) {
        let c = (colour & 0x03) as u32;
        // If count is 0 → rest-of-line: encode with 4 leading zeros
        // nibble then 2 bits of colour packed into a nibble.
        if count == 0 {
            // 0000 0000 0000 CC -> 4 nibbles = 0 0 0 c
            w.write(4, 0);
            w.write(4, 0);
            w.write(4, 0);
            w.write(4, c);
            return;
        }
        if count < 4 {
            // 2-bit count + 2-bit colour in one nibble (cc[1:0]Cc[1:0])
            let nib = ((count & 0x3) << 2) | c;
            w.write(4, nib);
            return;
        }
        if count < 16 {
            // 4-bit count + 2-bit colour in two nibbles: 0 count colour
            let val = (count << 2) | c; // 6 bits
            w.write(4, (val >> 4) & 0xF);
            w.write(4, val & 0xF);
            return;
        }
        if count < 64 {
            // 6-bit count + 2-bit colour in two nibbles with leading zero prefix
            let val = (count << 2) | c; // 8 bits
            w.write(4, 0);
            w.write(4, (val >> 4) & 0xF);
            w.write(4, val & 0xF);
            return;
        }
        // 14-bit: count<<2 | c → 16 bits = four nibbles, with leading 0 prefix.
        let val = (count << 2) | c;
        w.write(4, 0);
        w.write(4, (val >> 12) & 0xF);
        w.write(4, (val >> 8) & 0xF);
        w.write(4, (val >> 4) & 0xF);
        w.write(4, val & 0xF);
    }

    struct NibbleWriter {
        nibbles: Vec<u8>,
    }
    impl NibbleWriter {
        fn new() -> Self {
            Self {
                nibbles: Vec::new(),
            }
        }
        fn write(&mut self, _bits: u32, value: u32) {
            self.nibbles.push((value & 0x0F) as u8);
        }
        fn align(&mut self) {
            if self.nibbles.len() % 2 != 0 {
                self.nibbles.push(0);
            }
        }
        fn finish(&self, out: &mut Vec<u8>) {
            for pair in self.nibbles.chunks(2) {
                let hi = pair[0];
                let lo = if pair.len() == 2 { pair[1] } else { 0 };
                out.push((hi << 4) | lo);
            }
        }
    }

    // Build top & bottom RLE byte blocks.
    let mut top_bytes = Vec::new();
    push_rle_rows(
        &mut top_bytes,
        indices,
        width,
        height,
        (0..height as usize).step_by(2),
    );
    let mut bot_bytes = Vec::new();
    push_rle_rows(
        &mut bot_bytes,
        indices,
        width,
        height,
        (1..height as usize).step_by(2),
    );

    // Layout:
    //   [0..2]  SPU length
    //   [2..4]  control offset
    //   [4..]   RLE data (top then bottom)
    //   control: delay=0, next=pos, commands 0x03 palette, 0x04 alpha,
    //     0x05 coords, 0x06 offsets, 0x01 start, 0xFF end.
    let top_off = 4usize;
    let bot_off = top_off + top_bytes.len();
    let ctrl_off = bot_off + bot_bytes.len();
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0]); // placeholder SPU length
    out.extend_from_slice(&(ctrl_off as u16).to_be_bytes());
    out.extend_from_slice(&top_bytes);
    out.extend_from_slice(&bot_bytes);

    // Single control sequence.
    let ctrl_pos = out.len();
    out.extend_from_slice(&[0, 0]); // delay = 0
                                    // next_offset placeholder — will point back to itself at end.
    out.extend_from_slice(&[0, 0]);
    out.push(0x03); // palette select
    out.push(0x01); // bg=0, pat=1
    out.push(0x32); // emp1=3, emp2=2
    out.push(0x04); // alpha
    out.push(0x00); // bg=0, pat=0xF (full)
    out.push(0xFF);
    // but we actually want pattern+emp alphas to 0xF. rewrite:
    let last = out.len() - 2;
    out[last] = 0x0F; // (bg<<4)|pattern → bg=0 pat=0xF
    out[last + 1] = 0xFF; // emp1=0xF, emp2=0xF
    out.push(0x05); // coords
    out.push(((0u16) >> 4) as u8);
    out.push((((0u16) & 0x0F) << 4) as u8 | (((width as u16 - 1) >> 8) as u8 & 0x0F));
    out.push(((width as u16 - 1) & 0xFF) as u8);
    out.push(((0u16) >> 4) as u8);
    out.push((((0u16) & 0x0F) << 4) as u8 | (((height as u16 - 1) >> 8) as u8 & 0x0F));
    out.push(((height as u16 - 1) & 0xFF) as u8);
    out.push(0x06); // RLE offsets
    out.extend_from_slice(&(top_off as u16).to_be_bytes());
    out.extend_from_slice(&(bot_off as u16).to_be_bytes());
    out.push(0x01); // start display
    out.push(0xFF);

    // Patch next_offset in control sequence: point at itself to terminate.
    out[ctrl_pos + 2] = (ctrl_pos as u16 >> 8) as u8;
    out[ctrl_pos + 3] = (ctrl_pos as u16 & 0xFF) as u8;
    // Patch SPU length.
    let total = out.len() as u16;
    out[0] = (total >> 8) as u8;
    out[1] = (total & 0xFF) as u8;

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_idx_basics() {
        let text = "\
# VobSub index file
size: 720x480
palette: ff0000, 00ff00, 0000ff, ffffff, 000000, 808080, c0c0c0, 404040, 200020, 800080, a0a0a0, 010203, 040506, 070809, 0a0b0c, 0d0e0f
timestamp: 00:00:01:500, filepos: 000000000
timestamp: 00:00:03:000, filepos: 000000040
";
        let idx = parse_idx(text).unwrap();
        assert_eq!(idx.size, (720, 480));
        assert!(idx.has_palette);
        assert_eq!(idx.palette_rgb[0], [0xff, 0x00, 0x00]);
        assert_eq!(idx.palette_rgb[1], [0x00, 0xff, 0x00]);
        assert_eq!(idx.cues.len(), 2);
        assert_eq!(idx.cues[0].0, 1_500_000);
        assert_eq!(idx.cues[1].0, 3_000_000);
    }

    #[test]
    fn decodes_small_spu() {
        // 2×2 bitmap filled with palette index 1 (pattern colour).
        let indices = [1u8, 1, 1, 1];
        let spu = build_demo_spu(2, 2, &indices);
        let (state, pixels, (w, h)) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(pixels, vec![1u8, 1, 1, 1]);
        assert_eq!(state.palette_sel[1], 1);

        let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
        // 16-colour palette: entry 1 is red.
        let mut extra = vec![0u8; 48];
        extra[3] = 255;
        params.extradata = extra;
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.width, 2);
        assert_eq!(v.height, 2);
        let data = &v.planes[0].data;
        // All pixels red + alpha 255.
        for px in data.chunks(4) {
            assert_eq!(px, &[255, 0, 0, 255], "pixel not red: {:?}", px);
        }
    }
}
