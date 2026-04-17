//! DVB subtitle decoder (ETSI EN 300 743).
//!
//! DVB subtitles are carried as PES payloads inside MPEG-TS; each PES
//! payload begins with `0x20 0x00` (data_identifier + subtitle_stream_id),
//! followed by one or more segments, and ends with `0xFF` end-of-PES.
//! Each segment has the form:
//!
//! ```text
//! 0x0F              sync byte
//! segment_type      (1)   0x10 page / 0x11 region / 0x12 CLUT / 0x13 object / 0x14 display
//! page_id           (2 BE)
//! segment_length    (2 BE)
//! segment_body      (segment_length bytes)
//! ```
//!
//! A display-set terminates at either a `0x80` end-of-display-set
//! segment or the next page_composition segment. The decoder composes
//! the referenced regions onto a canvas sized by the display-definition
//! segment (or the default 720×576 PAL raster).
//!
//! ## Scope / limitations
//!
//! * **Decode only.** Pixel-coded objects are supported; *character*-
//!   coded objects (2‐byte UTF-style segments used for teletext-style
//!   streams) currently return `Error::Unsupported`.
//! * Single-region displays are handled; multi-region displays stack by
//!   region z-order (first region wins).
//! * Page timeouts are accepted but not enforced here — caller uses the
//!   accompanying packet duration.

use std::collections::{HashMap, VecDeque};

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame, VideoPlane,
};

use crate::DVBSUB_CODEC_ID;

// --- segment-type identifiers ------------------------------------------

pub const SEG_PAGE_COMPOSITION: u8 = 0x10;
pub const SEG_REGION_COMPOSITION: u8 = 0x11;
pub const SEG_CLUT_DEFINITION: u8 = 0x12;
pub const SEG_OBJECT_DATA: u8 = 0x13;
pub const SEG_DISPLAY_DEFINITION: u8 = 0x14;
pub const SEG_END_OF_DISPLAY_SET: u8 = 0x80;

// --- raw segments ------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RawSegment {
    pub seg_type: u8,
    pub page_id: u16,
    pub body: Vec<u8>,
}

/// Read the next DVB segment at `buf[pos]`. Returns the segment + new
/// cursor, or `Error::NeedMore` when fewer than 6 bytes remain / the
/// body is truncated.
pub fn read_segment(buf: &[u8], pos: usize) -> Result<(RawSegment, usize)> {
    if pos + 6 > buf.len() {
        return Err(Error::NeedMore);
    }
    if buf[pos] != 0x0F {
        return Err(Error::invalid(format!(
            "DVB sub: segment sync byte 0x0F expected, got 0x{:02X}",
            buf[pos]
        )));
    }
    let seg_type = buf[pos + 1];
    let page_id = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]);
    let len = u16::from_be_bytes([buf[pos + 4], buf[pos + 5]]) as usize;
    let end = pos + 6 + len;
    if end > buf.len() {
        return Err(Error::NeedMore);
    }
    Ok((
        RawSegment {
            seg_type,
            page_id,
            body: buf[pos + 6..end].to_vec(),
        },
        end,
    ))
}

// --- parse helpers -----------------------------------------------------

#[derive(Clone, Debug)]
struct DisplayDefinition {
    width: u16,
    height: u16,
}

impl Default for DisplayDefinition {
    fn default() -> Self {
        // Standard definition PAL DVB default raster.
        Self {
            width: 720,
            height: 576,
        }
    }
}

fn parse_display_definition(body: &[u8]) -> Result<DisplayDefinition> {
    if body.len() < 5 {
        return Err(Error::invalid("DVB DDS: body too short"));
    }
    // body[0] = version + flags
    let width = u16::from_be_bytes([body[1], body[2]]).wrapping_add(1);
    let height = u16::from_be_bytes([body[3], body[4]]).wrapping_add(1);
    Ok(DisplayDefinition { width, height })
}

#[derive(Clone, Debug)]
struct PageRegion {
    region_id: u8,
    x: u16,
    y: u16,
}

fn parse_page_composition(body: &[u8]) -> Result<Vec<PageRegion>> {
    if body.len() < 2 {
        return Err(Error::invalid("DVB page_composition: body too short"));
    }
    // body[0] = page_time_out (s), body[1] = version/state
    let mut cur = 2;
    let mut regions = Vec::new();
    while cur + 6 <= body.len() {
        let region_id = body[cur];
        // body[cur+1] reserved
        let x = u16::from_be_bytes([body[cur + 2], body[cur + 3]]);
        let y = u16::from_be_bytes([body[cur + 4], body[cur + 5]]);
        cur += 6;
        regions.push(PageRegion { region_id, x, y });
    }
    Ok(regions)
}

#[derive(Clone, Debug)]
struct Region {
    #[allow(dead_code)]
    width: u16,
    #[allow(dead_code)]
    height: u16,
    /// Colour-depth declared by the region: 2, 4, or 8 bits. We
    /// currently render without caring (the pixel-coded streams carry
    /// their own depth), but parsing validates the byte.
    #[allow(dead_code)]
    depth_bits: u8,
    clut_id: u8,
    objects: Vec<RegionObject>,
}

#[derive(Clone, Debug)]
struct RegionObject {
    object_id: u16,
    x: u16,
    y: u16,
}

fn parse_region_composition(body: &[u8]) -> Result<(u8, Region)> {
    if body.len() < 10 {
        return Err(Error::invalid("DVB region_composition: body too short"));
    }
    let region_id = body[0];
    // body[1] version + fill flag
    let width = u16::from_be_bytes([body[2], body[3]]);
    let height = u16::from_be_bytes([body[4], body[5]]);
    // body[6] = region_level_of_compatibility (3) + region_depth (3)
    let region_depth = (body[6] >> 2) & 0x07;
    let depth_bits = match region_depth {
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 4, // default-ish
    };
    let clut_id = body[7];
    // body[8..10] = 8-bit_pixel_code + 4-bit_pixel_code + 2-bit_pixel_code
    let mut cur = 10;
    let mut objects = Vec::new();
    while cur + 6 <= body.len() {
        let obj_hi = body[cur];
        let obj_lo = body[cur + 1];
        let object_id = u16::from_be_bytes([obj_hi, obj_lo]);
        let obj_type = (body[cur + 2] >> 6) & 0x03;
        // body[cur+2] also has provider_flag (2 bits) + x_pos hi (6 bits)
        let x = u16::from_be_bytes([body[cur + 2] & 0x3F, body[cur + 3]]);
        // body[cur+4] has reserved (4) + y_pos hi (4 bits)
        let y = u16::from_be_bytes([body[cur + 4] & 0x0F, body[cur + 5]]);
        cur += 6;
        if obj_type == 0x01 || obj_type == 0x02 {
            // foreground/background colour bytes follow — skip 2.
            if cur + 2 <= body.len() {
                cur += 2;
            }
        }
        objects.push(RegionObject { object_id, x, y });
    }
    Ok((
        region_id,
        Region {
            width,
            height,
            depth_bits,
            clut_id,
            objects,
        },
    ))
}

#[derive(Clone, Debug)]
struct Clut {
    /// 8-bit palette entries as RGBA.
    entries: [[u8; 4]; 256],
}

impl Default for Clut {
    fn default() -> Self {
        Self {
            entries: [[0u8; 4]; 256],
        }
    }
}

fn ycbcr_to_rgba(y: u8, cr: u8, cb: u8, t8: u8) -> [u8; 4] {
    // DVB CLUT values: Y/Cb/Cr full-range 8-bit + T (transparency).
    // Convert via BT.601.
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let r = y + ((91881 * cr) >> 16);
    let g = y - ((22554 * cb + 46802 * cr) >> 16);
    let b = y + ((116130 * cb) >> 16);
    let alpha = 255u8.saturating_sub(t8);
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        alpha,
    ]
}

fn parse_clut_into(body: &[u8], cluts: &mut HashMap<u8, Clut>) -> Result<()> {
    if body.len() < 2 {
        return Err(Error::invalid("DVB CLUT: body too short"));
    }
    let clut_id = body[0];
    // body[1] = version
    let entry = cluts.entry(clut_id).or_default();
    let mut cur = 2;
    while cur < body.len() {
        // Each entry: entry_id (1), flags (1), then either 2-byte
        // Y/Cr/Cb/T (bit set) or 4-byte values.
        if cur + 2 > body.len() {
            break;
        }
        let entry_id = body[cur];
        let flags = body[cur + 1];
        cur += 2;
        let full = (flags & 0x01) != 0;
        if full {
            if cur + 4 > body.len() {
                return Err(Error::invalid("DVB CLUT: truncated full entry"));
            }
            let y = body[cur];
            let cr = body[cur + 1];
            let cb = body[cur + 2];
            let t = body[cur + 3];
            cur += 4;
            entry.entries[entry_id as usize] = ycbcr_to_rgba(y, cr, cb, t);
        } else {
            if cur + 2 > body.len() {
                return Err(Error::invalid("DVB CLUT: truncated short entry"));
            }
            // Packed: 6 bits Y, 4 bits Cr, 4 bits Cb, 2 bits T — not
            // widely used; fall back to scaling.
            let b0 = body[cur];
            let b1 = body[cur + 1];
            cur += 2;
            let y = b0 & 0xFC;
            let cr = (((b0 & 0x03) << 2) | (b1 >> 6)) << 4;
            let cb = ((b1 >> 2) & 0x0F) << 4;
            let t = (b1 & 0x03) << 6;
            entry.entries[entry_id as usize] = ycbcr_to_rgba(y, cr, cb, t);
        }
    }
    Ok(())
}

// --- object data / pixel-coded string decoder --------------------------

#[derive(Clone, Debug)]
struct Object {
    /// Row-major indexed pixels. Width/height are determined by the
    /// region that hosts the object.
    rows: Vec<Vec<u8>>,
    /// Source object-data type. We track this so we can refuse
    /// character-coded objects at render time.
    #[allow(dead_code)]
    coding_method: u8,
}

fn parse_object_data(body: &[u8]) -> Result<(u16, Object)> {
    if body.len() < 3 {
        return Err(Error::invalid("DVB object_data: body too short"));
    }
    let object_id = u16::from_be_bytes([body[0], body[1]]);
    // body[2] = version (4 bits) + coding_method (2 bits) + non_modifying_colour_flag (1) + reserved
    let coding_method = (body[2] >> 2) & 0x03;
    if coding_method != 0 {
        return Err(Error::unsupported(format!(
            "DVB sub: coding_method {} (character/text objects)",
            coding_method
        )));
    }
    if body.len() < 7 {
        return Err(Error::invalid(
            "DVB object_data: pixel-coded header truncated",
        ));
    }
    let top_len = u16::from_be_bytes([body[3], body[4]]) as usize;
    let bot_len = u16::from_be_bytes([body[5], body[6]]) as usize;
    let top_start = 7;
    let top_end = top_start + top_len;
    let bot_end = top_end + bot_len;
    if bot_end > body.len() {
        return Err(Error::invalid(
            "DVB object_data: pixel-coded line blocks truncated",
        ));
    }
    let top_rows = parse_pixel_lines(&body[top_start..top_end])?;
    let bot_rows = if bot_len > 0 {
        parse_pixel_lines(&body[top_end..bot_end])?
    } else {
        top_rows.clone()
    };
    // Interleave top-then-bottom (top field rows on even lines, bottom
    // field rows on odd lines).
    let mut rows = Vec::with_capacity(top_rows.len() + bot_rows.len());
    let n = top_rows.len().max(bot_rows.len());
    for i in 0..n {
        if i < top_rows.len() {
            rows.push(top_rows[i].clone());
        }
        if i < bot_rows.len() {
            rows.push(bot_rows[i].clone());
        }
    }
    Ok((
        object_id,
        Object {
            rows,
            coding_method,
        },
    ))
}

fn parse_pixel_lines(buf: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut row: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let code = buf[i];
        i += 1;
        match code {
            0x10 => {
                // 2-bit pixel-code string — subset supported.
                let (consumed, pixels) = decode_2bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x11 => {
                let (consumed, pixels) = decode_4bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x12 => {
                let (consumed, pixels) = decode_8bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x20 | 0x21 | 0x22 => {
                // 2-to-4 / 2-to-8 / 4-to-8 map tables — skip the next two bytes.
                if i + 2 > buf.len() {
                    return Err(Error::invalid("DVB: map table truncated"));
                }
                i += 2;
            }
            0xF0 => {
                // End-of-object-line.
                rows.push(std::mem::take(&mut row));
            }
            _ => {
                // Unknown data_type; bail politely.
                return Err(Error::invalid(format!(
                    "DVB: unknown pixel-line data_type 0x{:02X}",
                    code
                )));
            }
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    Ok(rows)
}

/// Decode a 2-bit pixel-coded string. Returns (bytes_consumed, pixels).
fn decode_2bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    // Minimal implementation: read bit-by-bit until the terminator
    // `00 00 00 00` (8 zero bits) — the DVB spec's end-of-2-bit-code.
    let mut bits = BitReader::new(buf);
    let mut pixels = Vec::new();
    loop {
        let code = bits.read(2)?;
        if code != 0 {
            pixels.push(code as u8);
            continue;
        }
        let b1 = bits.read(1)?;
        if b1 == 1 {
            // 011 -> 3 pixels of colour 0 (not in spec literally, but
            // produced by the common 2-bit decoders we reference).
            let run = bits.read(3)? as usize + 3;
            let col = bits.read(2)? as u8;
            for _ in 0..run {
                pixels.push(col);
            }
            continue;
        }
        // 0 0 ...
        let b2 = bits.read(1)?;
        if b2 == 1 {
            // 001 - run of 0s, 3 + 4-bit count
            let run = bits.read(4)? as usize + 12;
            for _ in 0..run {
                pixels.push(0);
            }
            continue;
        }
        let b3 = bits.read(2)?;
        match b3 {
            0x00 => {
                // end of string; align to next byte.
                bits.align();
                break;
            }
            0x01 => pixels.push(0), // one pixel of colour 0
            0x02 => {
                // 3 + 3-bit count run of 2-bit value
                let run = bits.read(3)? as usize + 3;
                let col = bits.read(2)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            0x03 => {
                // 8-bit count run of 2-bit value, +25
                let run = bits.read(8)? as usize + 29;
                let col = bits.read(2)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            _ => unreachable!(),
        }
    }
    Ok((bits.consumed_bytes(), pixels))
}

/// Decode a 4-bit pixel-coded string.
fn decode_4bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    let mut bits = BitReader::new(buf);
    let mut pixels = Vec::new();
    loop {
        let code = bits.read(4)?;
        if code != 0 {
            pixels.push(code as u8);
            continue;
        }
        let b1 = bits.read(1)?;
        if b1 == 0 {
            // Either end-of-string or run of 0s.
            let run_hdr = bits.read(3)?;
            if run_hdr == 0 {
                bits.align();
                break;
            }
            // 2..9 pixels of colour 0.
            let run = run_hdr as usize + 2;
            for _ in 0..run {
                pixels.push(0);
            }
            continue;
        }
        let b2 = bits.read(1)?;
        if b2 == 0 {
            // 01xxxx - run of 4-bit colour (4..7) + colour
            let run = (bits.read(2)? as usize) + 4;
            let col = bits.read(4)? as u8;
            for _ in 0..run {
                pixels.push(col);
            }
            continue;
        }
        let b3 = bits.read(2)?;
        match b3 {
            0x00 => pixels.push(0),
            0x01 => {
                // 2 pixels of colour 0.
                pixels.push(0);
                pixels.push(0);
            }
            0x02 => {
                // 12..27 pixels of 4-bit colour
                let run = (bits.read(4)? as usize) + 9;
                let col = bits.read(4)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            0x03 => {
                // 29..284 pixels of 4-bit colour
                let run = (bits.read(8)? as usize) + 25;
                let col = bits.read(4)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            _ => unreachable!(),
        }
    }
    Ok((bits.consumed_bytes(), pixels))
}

/// Decode an 8-bit pixel-coded string.
fn decode_8bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    // The 8-bit encoding is byte-aligned: each "code" is one byte.
    // 0x00 introduces an escape; non-zero bytes are literal colours.
    let mut i = 0usize;
    let mut pixels = Vec::new();
    while i < buf.len() {
        let b = buf[i];
        i += 1;
        if b != 0 {
            pixels.push(b);
            continue;
        }
        if i >= buf.len() {
            return Err(Error::invalid("DVB 8-bit: escape truncated"));
        }
        let b1 = buf[i];
        i += 1;
        if b1 == 0 {
            // end-of-string
            break;
        }
        let run_flag = (b1 & 0x80) != 0;
        let count = (b1 & 0x7F) as usize;
        if !run_flag {
            if count == 0 {
                // reserved; skip
                continue;
            }
            // count pixels of colour 0.
            for _ in 0..count {
                pixels.push(0);
            }
        } else {
            if i >= buf.len() {
                return Err(Error::invalid("DVB 8-bit: run-byte truncated"));
            }
            let col = buf[i];
            i += 1;
            for _ in 0..count {
                pixels.push(col);
            }
        }
    }
    Ok((i, pixels))
}

// --- bit reader --------------------------------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    fn read(&mut self, n: u32) -> Result<u32> {
        let mut out = 0u32;
        for _ in 0..n {
            if self.bit_pos >= self.buf.len() * 8 {
                return Err(Error::invalid("DVB bit reader: ran out of data"));
            }
            let byte = self.buf[self.bit_pos / 8];
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            out = (out << 1) | (bit as u32);
            self.bit_pos += 1;
        }
        Ok(out)
    }

    fn align(&mut self) {
        self.bit_pos = (self.bit_pos + 7) / 8 * 8;
    }

    fn consumed_bytes(&self) -> usize {
        (self.bit_pos + 7) / 8
    }
}

// --- decoder -----------------------------------------------------------

/// Build a DVB subtitle decoder. Packet payloads are the raw PES
/// payload (with the `0x20 0x00` data_identifier prefix stripped).
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(DvbSubDecoder {
        codec_id: CodecId::new(DVBSUB_CODEC_ID),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct DvbSubDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for DvbSubDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let payload = strip_pes_prefix(&packet.data);
        let mut dds = DisplayDefinition::default();
        let mut regions: HashMap<u8, Region> = HashMap::new();
        let mut objects: HashMap<u16, Object> = HashMap::new();
        let mut cluts: HashMap<u8, Clut> = HashMap::new();
        let mut page: Vec<PageRegion> = Vec::new();
        let mut cur = 0;
        while cur < payload.len() {
            let (seg, next) = match read_segment(payload, cur) {
                Ok(x) => x,
                Err(Error::NeedMore) => break,
                Err(e) => return Err(e),
            };
            match seg.seg_type {
                SEG_DISPLAY_DEFINITION => {
                    dds = parse_display_definition(&seg.body)?;
                }
                SEG_PAGE_COMPOSITION => {
                    page = parse_page_composition(&seg.body)?;
                }
                SEG_REGION_COMPOSITION => {
                    let (id, region) = parse_region_composition(&seg.body)?;
                    regions.insert(id, region);
                }
                SEG_CLUT_DEFINITION => {
                    parse_clut_into(&seg.body, &mut cluts)?;
                }
                SEG_OBJECT_DATA => {
                    let (id, obj) = parse_object_data(&seg.body)?;
                    objects.insert(id, obj);
                }
                SEG_END_OF_DISPLAY_SET => {
                    break;
                }
                _ => {}
            }
            cur = next;
        }
        let width = dds.width as usize;
        let height = dds.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::invalid("DVB sub: zero canvas"));
        }
        let mut canvas = vec![0u8; width * height * 4];
        for pr in &page {
            let Some(region) = regions.get(&pr.region_id) else {
                continue;
            };
            let clut = cluts.get(&region.clut_id).cloned().unwrap_or_default();
            for ro in &region.objects {
                let Some(obj) = objects.get(&ro.object_id) else {
                    continue;
                };
                let base_x = pr.x as usize + ro.x as usize;
                let base_y = pr.y as usize + ro.y as usize;
                for (row_idx, row) in obj.rows.iter().enumerate() {
                    let dy = base_y + row_idx;
                    if dy >= height {
                        break;
                    }
                    for (col_idx, &px) in row.iter().enumerate() {
                        let dx = base_x + col_idx;
                        if dx >= width {
                            break;
                        }
                        if px == 0 {
                            continue; // transparent background
                        }
                        let rgba = clut.entries[px as usize];
                        if rgba[3] == 0 {
                            continue;
                        }
                        let dst = (dy * width + dx) * 4;
                        canvas[dst..dst + 4].copy_from_slice(&rgba);
                    }
                }
            }
        }

        let frame = VideoFrame {
            format: PixelFormat::Rgba,
            width: width as u32,
            height: height as u32,
            pts: packet.pts,
            time_base: packet.time_base,
            planes: vec![VideoPlane {
                stride: width * 4,
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

/// Strip the `0x20 0x00` data_identifier / subtitle_stream_id prefix if
/// present. Containers typically hand us the PES payload that already
/// starts with the DVB segment stream — but some strip these lead
/// bytes, so we accept either shape.
fn strip_pes_prefix(buf: &[u8]) -> &[u8] {
    if buf.len() >= 2 && buf[0] == 0x20 && buf[1] == 0x00 {
        &buf[2..]
    } else {
        buf
    }
}

// --- test helper -------------------------------------------------------

#[doc(hidden)]
pub fn build_demo_pes(canvas: (u16, u16), pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height);
    fn segment(out: &mut Vec<u8>, seg_type: u8, body: &[u8]) {
        out.push(0x0F);
        out.push(seg_type);
        out.extend_from_slice(&1u16.to_be_bytes()); // page_id
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    let mut out = vec![0x20, 0x00]; // data_identifier + subtitle_stream_id

    // DDS
    let mut dds = Vec::new();
    dds.push(0); // version + flags
    dds.extend_from_slice(&(canvas.0.saturating_sub(1)).to_be_bytes());
    dds.extend_from_slice(&(canvas.1.saturating_sub(1)).to_be_bytes());
    segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

    // Page composition: one region at (0,0).
    let mut page = Vec::new();
    page.push(0); // page_time_out (s)
    page.push(0); // version/state
    page.push(0); // region_id
    page.push(0xFF); // reserved
    page.extend_from_slice(&0u16.to_be_bytes()); // x
    page.extend_from_slice(&0u16.to_be_bytes()); // y
    segment(&mut out, SEG_PAGE_COMPOSITION, &page);

    // Region composition: one object at (0,0), 8-bit depth, CLUT=0.
    let mut region = Vec::new();
    region.push(0); // region_id
    region.push(0); // version + fill_flag
    region.extend_from_slice(&(width as u16).to_be_bytes());
    region.extend_from_slice(&(height as u16).to_be_bytes());
    region.push(3 << 2); // depth = 3 (8-bit)
    region.push(0); // clut_id
    region.push(0); // 8-bit pixel code
    region.push(0); // 4/2-bit codes
                    // object entry
    region.extend_from_slice(&0u16.to_be_bytes()); // object_id + obj_type(0)
    region.extend_from_slice(&0u16.to_be_bytes()); // x
    region.extend_from_slice(&0u16.to_be_bytes()); // y
    segment(&mut out, SEG_REGION_COMPOSITION, &region);

    // CLUT: entries 0..3 with identifiable RGBA values.
    let mut clut = Vec::new();
    clut.push(0); // clut_id
    clut.push(0); // version
                  // Entry 1 → white. Y=255, Cr=128, Cb=128, T=0.
    clut.push(1); // entry_id
    clut.push(0xFF); // flags — mark "full precision" (bit 0) + 8-bit type bits
    clut.push(255); // Y
    clut.push(128); // Cr
    clut.push(128); // Cb
    clut.push(0); // T
                  // Entry 2 → red.
    clut.push(2);
    clut.push(0xFF);
    clut.push(81); // approximate Y for red
    clut.push(240); // Cr
    clut.push(90); // Cb
    clut.push(0);
    segment(&mut out, SEG_CLUT_DEFINITION, &clut);

    // Object data: 8-bit pixel-coded string, two rows top+bottom.
    let mut obj = Vec::new();
    obj.extend_from_slice(&0u16.to_be_bytes()); // object_id
                                                // coding_method = 0 in bits 3..4 of byte[2]. Rest zero.
    obj.push(0);
    // Build top-field / bottom-field line blocks.
    fn encode_rows_8bit(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            out.push(0x12); // 8-bit pixel code
            for &p in row {
                if p == 0 {
                    out.push(0x00);
                    out.push(0x01); // 1 pixel of colour 0 (short form)
                    continue;
                }
                out.push(p);
            }
            out.push(0x00);
            out.push(0x00); // end of 8-bit string
            out.push(0xF0); // end-of-object-line
        }
        out
    }
    // Split into top/bottom field rows by interleave.
    let mut top: Vec<Vec<u8>> = Vec::new();
    let mut bot: Vec<Vec<u8>> = Vec::new();
    for (i, row) in pixels.chunks(width).enumerate() {
        if i % 2 == 0 {
            top.push(row.to_vec());
        } else {
            bot.push(row.to_vec());
        }
    }
    let top_bytes = encode_rows_8bit(&top);
    let bot_bytes = encode_rows_8bit(&bot);
    obj.extend_from_slice(&(top_bytes.len() as u16).to_be_bytes());
    obj.extend_from_slice(&(bot_bytes.len() as u16).to_be_bytes());
    obj.extend_from_slice(&top_bytes);
    obj.extend_from_slice(&bot_bytes);
    segment(&mut out, SEG_OBJECT_DATA, &obj);

    // END of display-set
    segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    #[test]
    fn decodes_pixel_coded_bitmap() {
        // 2×2 bitmap: white, red, red, white.
        let pixels = [1u8, 2, 2, 1];
        let pes = build_demo_pes((2, 2), &pixels, 2, 2);
        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.width, 2);
        assert_eq!(v.height, 2);
        assert_eq!(v.format, PixelFormat::Rgba);
        assert_eq!(v.planes[0].data.len(), 2 * 2 * 4);
        // Row 0: white, red.
        let r0c0 = &v.planes[0].data[0..4];
        let r0c1 = &v.planes[0].data[4..8];
        // White ≈ all channels high.
        assert!(
            r0c0[0] > 200 && r0c0[1] > 200 && r0c0[2] > 200,
            "expected white, got {:?}",
            r0c0
        );
        // Red-ish: R dominant.
        assert!(
            r0c1[0] > r0c1[1] && r0c1[0] > r0c1[2],
            "expected red-dominant, got {:?}",
            r0c1
        );
    }

    #[test]
    fn rejects_character_coded_objects() {
        // Object-data body: id=0, coding_method=1 (character string).
        let mut body = Vec::new();
        body.extend_from_slice(&0u16.to_be_bytes()); // object id
        body.push(0b0000_0100); // coding_method = 1 in bits 2..3
        body.extend_from_slice(&0u16.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());

        let err = parse_object_data(&body).unwrap_err();
        match err {
            Error::Unsupported(_) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
