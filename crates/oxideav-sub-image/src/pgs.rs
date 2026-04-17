//! PGS / HDMV / Blu-ray `.sup` parser, container, and decoder.
//!
//! A PGS stream is a sequence of display-sets. Each display-set is made
//! of segments, all prefixed by the same on-disk header:
//!
//! ```text
//! "PG" magic (2)
//! PTS (4 BE, 90 kHz units)
//! DTS (4 BE, 90 kHz units)
//! segment type (1): 0x14 PDS, 0x15 ODS, 0x16 PCS, 0x17 WDS, 0x80 END
//! segment size (2 BE)
//! segment body (segment-size bytes)
//! ```
//!
//! A display-set is: one PCS, optional WDS/PDS, one or more ODS,
//! followed by an END segment. The display-set defines a palette + one
//! or more picture objects + the positions at which to compose them on
//! the video canvas.
//!
//! ## Decoder contract
//!
//! The decoder output is one [`oxideav_core::Frame::Video`] per
//! display-set ending with an END segment. The frame's dimensions are
//! the video canvas declared in PCS. The pixel format is
//! [`oxideav_core::PixelFormat::Rgba`]. Transparent background wherever
//! no object is composed; objects are YCbCr→RGB-converted and
//! alpha-composited at the PCS positions.
//!
//! ## Scope / limitations
//!
//! * **Decode only.**
//! * Objects referenced by a composition but not yet seen via ODS are
//!   skipped silently — PGS allows carrying only palette/WDS updates.
//! * ODS fragmentation is handled (an object carries `last_in_sequence`
//!   bits); a PDS version update is treated as a replace.
//! * Cropped compositions (PCS.object_cropped_flag) fall back to
//!   compositing the full object — crop rectangles are parsed but not
//!   applied.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, SeekFrom};

use oxideav_codec::Decoder;
use oxideav_container::{ContainerRegistry, Demuxer, ProbeData, ProbeScore, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, StreamInfo,
    TimeBase, VideoFrame, VideoPlane,
};

use crate::PGS_CODEC_ID;

// --- segment-type identifiers --------------------------------------------

pub const SEG_PDS: u8 = 0x14;
pub const SEG_ODS: u8 = 0x15;
pub const SEG_PCS: u8 = 0x16;
pub const SEG_WDS: u8 = 0x17;
pub const SEG_END: u8 = 0x80;

/// One parsed display-set segment, carrying only the bytes we need.
#[derive(Clone, Debug)]
pub struct RawSegment {
    pub pts_90k: u32,
    pub dts_90k: u32,
    pub seg_type: u8,
    pub body: Vec<u8>,
}

/// Parse the next raw segment starting at `buf[pos]`. Returns the parsed
/// segment and the new cursor position. `Error::NeedMore` is returned if
/// fewer than 13 bytes remain or the body is truncated — callers reading
/// a complete file typically treat that as EOF.
pub fn read_segment(buf: &[u8], pos: usize) -> Result<(RawSegment, usize)> {
    if pos + 13 > buf.len() {
        return Err(Error::NeedMore);
    }
    if &buf[pos..pos + 2] != b"PG" {
        return Err(Error::invalid("PGS: segment missing 'PG' magic"));
    }
    let pts_90k = u32::from_be_bytes([buf[pos + 2], buf[pos + 3], buf[pos + 4], buf[pos + 5]]);
    let dts_90k = u32::from_be_bytes([buf[pos + 6], buf[pos + 7], buf[pos + 8], buf[pos + 9]]);
    let seg_type = buf[pos + 10];
    let size = u16::from_be_bytes([buf[pos + 11], buf[pos + 12]]) as usize;
    let end = pos + 13 + size;
    if end > buf.len() {
        return Err(Error::NeedMore);
    }
    Ok((
        RawSegment {
            pts_90k,
            dts_90k,
            seg_type,
            body: buf[pos + 13..end].to_vec(),
        },
        end,
    ))
}

// --- PCS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PresentationComposition {
    pub width: u16,
    pub height: u16,
    pub composition_number: u16,
    pub objects: Vec<CompositionObject>,
}

#[derive(Clone, Debug)]
pub struct CompositionObject {
    pub object_id: u16,
    pub window_id: u8,
    pub cropped: bool,
    pub forced: bool,
    pub x: u16,
    pub y: u16,
}

fn parse_pcs(body: &[u8]) -> Result<PresentationComposition> {
    if body.len() < 11 {
        return Err(Error::invalid("PGS PCS: body too short"));
    }
    let width = u16::from_be_bytes([body[0], body[1]]);
    let height = u16::from_be_bytes([body[2], body[3]]);
    // body[4] = frame-rate (ignored)
    let composition_number = u16::from_be_bytes([body[5], body[6]]);
    // body[7] = composition state (ignored for now)
    // body[8] = palette_update_flag (ignored)
    // body[9] = palette_id (ignored — stored on the PDS itself)
    let n_objects = body[10] as usize;
    let mut cur = 11;
    let mut objects = Vec::with_capacity(n_objects);
    for _ in 0..n_objects {
        if cur + 8 > body.len() {
            return Err(Error::invalid("PGS PCS: object entry truncated"));
        }
        let object_id = u16::from_be_bytes([body[cur], body[cur + 1]]);
        let window_id = body[cur + 2];
        let flags = body[cur + 3];
        let cropped = (flags & 0x80) != 0;
        let forced = (flags & 0x40) != 0;
        let x = u16::from_be_bytes([body[cur + 4], body[cur + 5]]);
        let y = u16::from_be_bytes([body[cur + 6], body[cur + 7]]);
        cur += 8;
        if cropped {
            if cur + 8 > body.len() {
                return Err(Error::invalid("PGS PCS: cropped object missing crop rect"));
            }
            cur += 8;
        }
        objects.push(CompositionObject {
            object_id,
            window_id,
            cropped,
            forced,
            x,
            y,
        });
    }
    Ok(PresentationComposition {
        width,
        height,
        composition_number,
        objects,
    })
}

// --- PDS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Palette {
    /// 256 entries, defaulting to transparent black. Each entry holds
    /// pre-resolved RGBA bytes.
    pub entries: [[u8; 4]; 256],
}

impl Default for Palette {
    fn default() -> Self {
        Self {
            entries: [[0u8; 4]; 256],
        }
    }
}

fn ycbcr_to_rgba(y: u8, cr: u8, cb: u8, a: u8) -> [u8; 4] {
    // BT.601 — PGS uses 8-bit limited-range YCbCr.
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let r = y + ((91881 * cr) >> 16);
    let g = y - ((22554 * cb + 46802 * cr) >> 16);
    let b = y + ((116130 * cb) >> 16);
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        a,
    ]
}

fn parse_pds_into(body: &[u8], palette: &mut Palette) -> Result<()> {
    if body.len() < 2 {
        return Err(Error::invalid("PGS PDS: too short"));
    }
    // body[0] = palette_id, body[1] = palette_version. We fold all
    // palettes into one shared table — real streams rarely use more
    // than the id-0 palette per display-set.
    let mut cur = 2;
    while cur + 5 <= body.len() {
        let idx = body[cur] as usize;
        let y = body[cur + 1];
        let cr = body[cur + 2];
        let cb = body[cur + 3];
        let a = body[cur + 4];
        palette.entries[idx] = ycbcr_to_rgba(y, cr, cb, a);
        cur += 5;
    }
    Ok(())
}

// --- ODS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Object {
    pub width: u16,
    pub height: u16,
    /// Indexed-colour pixels (palette ids), row-major.
    pub pixels: Vec<u8>,
}

/// Assemble an ODS (handling the `last_in_sequence` fragmentation bits)
/// into a completed object.
fn parse_ods_into(
    body: &[u8],
    fragments: &mut HashMap<u16, Vec<u8>>,
    objects: &mut HashMap<u16, Object>,
) -> Result<()> {
    if body.len() < 4 {
        return Err(Error::invalid("PGS ODS: header too short"));
    }
    let object_id = u16::from_be_bytes([body[0], body[1]]);
    // body[2] = object_version
    let seq_flag = body[3];
    let first = (seq_flag & 0x80) != 0;
    let last = (seq_flag & 0x40) != 0;

    // First fragment carries: object_data_length (3 BE) + width (2) + height (2) + rle-data.
    // Middle/last fragments carry just rle-data.
    let entry = fragments.entry(object_id).or_default();
    if first {
        entry.clear();
    }
    entry.extend_from_slice(&body[4..]);

    if !last {
        return Ok(());
    }
    // Complete — decode. Pull out width/height + RLE stream.
    let full = fragments.remove(&object_id).unwrap_or_default();
    if full.len() < 7 {
        return Err(Error::invalid(
            "PGS ODS: assembled object data shorter than header",
        ));
    }
    // full[0..3] = object_data_length (u24 BE, informational)
    let width = u16::from_be_bytes([full[3], full[4]]);
    let height = u16::from_be_bytes([full[5], full[6]]);
    if width == 0 || height == 0 {
        return Err(Error::invalid("PGS ODS: zero width/height"));
    }
    let rle = &full[7..];
    let pixels = decode_rle(rle, width as usize, height as usize)?;
    objects.insert(
        object_id,
        Object {
            width,
            height,
            pixels,
        },
    );
    Ok(())
}

/// Decode a PGS run-length encoded bitmap into a width×height indexed
/// buffer. The PGS RLE uses these runs:
///
/// * `CCCCCCCC` (C ≠ 0) — single pixel of colour C.
/// * `00 00`    — end-of-line marker.
/// * `00 LLLLLLLL` (L < 0x40, L > 0) — L pixels of colour 0.
/// * `00 01LLLLLL LLLLLLLL` — (L, 6+8 bits) pixels of colour 0.
/// * `00 10LLLLLL CCCCCCCC` — L pixels of colour C (L < 0x40, L > 0).
/// * `00 11LLLLLL LLLLLLLL CCCCCCCC` — L pixels (14 bits) of colour C.
pub fn decode_rle(rle: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; width * height];
    let mut i = 0;
    let mut row = 0usize;
    let mut col = 0usize;
    while i < rle.len() {
        let b0 = rle[i];
        i += 1;
        let (run_len, colour, line_end) = if b0 != 0 {
            (1usize, b0, false)
        } else {
            if i >= rle.len() {
                return Err(Error::invalid("PGS RLE: truncated after 0x00"));
            }
            let b1 = rle[i];
            i += 1;
            if b1 == 0 {
                // end-of-line
                (0usize, 0u8, true)
            } else {
                let hi = b1 & 0xC0;
                let len_lo = (b1 & 0x3F) as usize;
                let (length, colour) = match hi {
                    0x00 => (len_lo, 0u8),
                    0x40 => {
                        if i >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated 14-bit length"));
                        }
                        let b2 = rle[i] as usize;
                        i += 1;
                        ((len_lo << 8) | b2, 0u8)
                    }
                    0x80 => {
                        if i >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated colour"));
                        }
                        let c = rle[i];
                        i += 1;
                        (len_lo, c)
                    }
                    _ => {
                        if i + 1 >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated 14-bit+colour run"));
                        }
                        let b2 = rle[i] as usize;
                        let c = rle[i + 1];
                        i += 2;
                        (((len_lo << 8) | b2), c)
                    }
                };
                (length, colour, false)
            }
        };
        if line_end {
            row += 1;
            col = 0;
            if row > height {
                return Err(Error::invalid("PGS RLE: too many lines"));
            }
            continue;
        }
        if row >= height {
            return Err(Error::invalid("PGS RLE: pixel past end of bitmap"));
        }
        // Clamp runs that would stray past the declared row — malformed
        // streams exist in the wild. Matching a row-exact end is not
        // strictly required: the RLE end-of-line marker is what moves
        // us to the next row.
        let end = col.saturating_add(run_len).min(width);
        let base = row * width + col;
        for px in &mut out[base..base + (end - col)] {
            *px = colour;
        }
        col = end;
    }
    Ok(out)
}

// --- Display-set accumulator + renderer --------------------------------

/// Internal state as we walk segments looking for an END.
#[derive(Default)]
struct DisplaySet {
    pcs: Option<PresentationComposition>,
    // Accepted last-known PTS in 90 kHz units.
    pts_90k: u32,
    palette: Palette,
    object_fragments: HashMap<u16, Vec<u8>>,
    objects: HashMap<u16, Object>,
    /// Last-known canvas size (carried over between display-sets when
    /// a PCS-less reset arrives).
    last_canvas: Option<(u16, u16)>,
}

impl DisplaySet {
    fn push(&mut self, seg: &RawSegment) -> Result<()> {
        self.pts_90k = seg.pts_90k;
        match seg.seg_type {
            SEG_PCS => {
                let pcs = parse_pcs(&seg.body)?;
                self.last_canvas = Some((pcs.width, pcs.height));
                self.pcs = Some(pcs);
            }
            SEG_WDS => {
                // We don't clip to the declared windows — parsing is
                // kept here so that we validate the length is sane.
                if seg.body.is_empty() {
                    return Err(Error::invalid("PGS WDS: empty body"));
                }
            }
            SEG_PDS => {
                parse_pds_into(&seg.body, &mut self.palette)?;
            }
            SEG_ODS => {
                parse_ods_into(&seg.body, &mut self.object_fragments, &mut self.objects)?;
            }
            SEG_END => {}
            _ => {
                // Unknown / reserved — ignore like ffmpeg does.
            }
        }
        Ok(())
    }

    /// Render the assembled set into an RGBA frame. Returns `Ok(None)`
    /// when the display-set is an "erase" (PCS with zero composition
    /// objects or no PCS at all) — in that case the caller emits a
    /// fully-transparent canvas so the pipeline can clear whatever was
    /// displayed before.
    fn render(&self) -> Result<Option<Vec<u8>>> {
        let Some(pcs) = &self.pcs else {
            return Ok(None);
        };
        let width = pcs.width as usize;
        let height = pcs.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::invalid("PGS PCS: zero-sized canvas"));
        }
        let mut canvas = vec![0u8; width * height * 4];
        for co in &pcs.objects {
            let Some(obj) = self.objects.get(&co.object_id) else {
                continue;
            };
            let ox = co.x as usize;
            let oy = co.y as usize;
            let ow = obj.width as usize;
            let oh = obj.height as usize;
            for row in 0..oh {
                let dy = oy + row;
                if dy >= height {
                    break;
                }
                for col in 0..ow {
                    let dx = ox + col;
                    if dx >= width {
                        break;
                    }
                    let idx = obj.pixels[row * ow + col] as usize;
                    let rgba = self.palette.entries[idx];
                    if rgba[3] == 0 {
                        continue;
                    }
                    let dst = (dy * width + dx) * 4;
                    canvas[dst..dst + 4].copy_from_slice(&rgba);
                }
            }
        }
        Ok(Some(canvas))
    }
}

// --- Container (.sup) --------------------------------------------------

/// File-extension / container name registration.
pub fn register_container(reg: &mut ContainerRegistry) {
    reg.register_demuxer("pgs", open_pgs);
    reg.register_extension("sup", "pgs");
    reg.register_probe("pgs", probe_pgs);
}

fn probe_pgs(p: &ProbeData) -> ProbeScore {
    // A PGS file opens with the ASCII "PG" magic and reasonable PTS
    // fields. Combined with the .sup extension we give it a high score.
    if p.buf.len() >= 13 && &p.buf[..2] == b"PG" {
        100
    } else if p.ext == Some("sup") {
        25
    } else {
        0
    }
}

fn open_pgs(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut buf = Vec::new();
    input.seek(SeekFrom::Start(0))?;
    input.read_to_end(&mut buf)?;

    // Walk the file, gathering segments into display-sets, emitting one
    // packet per display-set with pts = its first segment's PTS.
    let time_base = TimeBase::new(1, 90_000);
    let mut packets: VecDeque<Packet> = VecDeque::new();
    let mut cur = 0usize;
    let mut ds_start_pts: Option<u32> = None;
    let mut ds_buf: Vec<u8> = Vec::new();
    let mut last_canvas: Option<(u16, u16)> = None;
    while cur < buf.len() {
        let (seg, next) = match read_segment(&buf, cur) {
            Ok(x) => x,
            Err(_) => break,
        };
        if ds_start_pts.is_none() {
            ds_start_pts = Some(seg.pts_90k);
        }
        ds_buf.extend_from_slice(&buf[cur..next]);
        // Snoop on PCS to capture the canvas size for stream metadata.
        if seg.seg_type == SEG_PCS {
            if let Ok(pcs) = parse_pcs(&seg.body) {
                last_canvas = Some((pcs.width, pcs.height));
            }
        }
        cur = next;
        if seg.seg_type == SEG_END {
            let pts = ds_start_pts.take().unwrap_or(0);
            let mut packet = Packet::new(0, time_base, std::mem::take(&mut ds_buf));
            packet.pts = Some(pts as i64);
            packet.dts = Some(pts as i64);
            packet.flags.keyframe = true;
            packets.push_back(packet);
        }
    }
    // Give successive packets a synthetic duration (end - start) and
    // leave the final one's duration unset — we don't know when its cue
    // disappears until another display-set arrives.
    for i in 0..packets.len().saturating_sub(1) {
        let (Some(a), Some(b)) = (packets[i].pts, packets[i + 1].pts) else {
            continue;
        };
        packets[i].duration = Some((b - a).max(0));
    }

    let (w, h) = last_canvas.unwrap_or((0, 0));
    let mut params = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    params.media_type = MediaType::Subtitle;
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Rgba);

    let total = packets.back().and_then(|p| p.pts).unwrap_or(0) as i64;
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total),
        start_time: Some(0),
        params,
    };
    Ok(Box::new(PgsDemuxer {
        streams: [stream],
        packets,
    }))
}

struct PgsDemuxer {
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl Demuxer for PgsDemuxer {
    fn format_name(&self) -> &str {
        "pgs"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }

    fn duration_micros(&self) -> Option<i64> {
        // PTS is in 90 kHz — µs = pts * (1_000_000 / 90_000) = pts * 100 / 9.
        let pts = self.streams[0].duration?;
        Some(pts * 100 / 9)
    }
}

// --- Decoder -----------------------------------------------------------

/// Build a PGS decoder.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(PgsDecoder {
        codec_id: CodecId::new(PGS_CODEC_ID),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct PgsDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for PgsDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let mut ds = DisplaySet::default();
        let mut cur = 0;
        while cur < packet.data.len() {
            let (seg, next) = read_segment(&packet.data, cur)?;
            ds.push(&seg)?;
            cur = next;
        }
        let Some(pcs) = &ds.pcs else {
            // No PCS → nothing to render for this packet. The container
            // normally emits packets only when an END closes a set, so
            // getting here means a malformed or empty display-set.
            return Ok(());
        };
        let width = pcs.width as u32;
        let height = pcs.height as u32;
        let rendered = ds
            .render()?
            .unwrap_or_else(|| vec![0u8; (width as usize) * (height as usize) * 4]);
        let frame = VideoFrame {
            format: PixelFormat::Rgba,
            width,
            height,
            pts: packet.pts,
            time_base: packet.time_base,
            planes: vec![VideoPlane {
                stride: (width as usize) * 4,
                data: rendered,
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

// --- Test helper -------------------------------------------------------

/// Build a minimal PGS display-set carrying one PCS, one PDS, one ODS,
/// and the terminating END. Used by this crate's tests. Public so that
/// external integration tests can exercise the decoder without pulling
/// in `oxideav-tests`.
#[doc(hidden)]
pub fn build_demo_display_set(
    canvas: (u16, u16),
    object: (u16, u16),
    position: (u16, u16),
    palette: &[(u8, [u8; 4])],
    pixels: &[u8],
) -> Vec<u8> {
    fn segment(out: &mut Vec<u8>, pts_90k: u32, seg_type: u8, body: &[u8]) {
        out.extend_from_slice(b"PG");
        out.extend_from_slice(&pts_90k.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.push(seg_type);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    let mut out = Vec::new();

    // PCS.
    let mut pcs = Vec::new();
    pcs.extend_from_slice(&canvas.0.to_be_bytes());
    pcs.extend_from_slice(&canvas.1.to_be_bytes());
    pcs.push(0); // frame rate
    pcs.extend_from_slice(&1u16.to_be_bytes()); // composition number
    pcs.push(0); // composition state (normal)
    pcs.push(0); // palette update flag
    pcs.push(0); // palette id
    pcs.push(1); // one composition object
    pcs.extend_from_slice(&1u16.to_be_bytes()); // object id
    pcs.push(0); // window id
    pcs.push(0); // flags
    pcs.extend_from_slice(&position.0.to_be_bytes());
    pcs.extend_from_slice(&position.1.to_be_bytes());
    segment(&mut out, 0, SEG_PCS, &pcs);

    // WDS (one window covering the full canvas).
    let mut wds = Vec::new();
    wds.push(1); // one window
    wds.push(0); // window id
    wds.extend_from_slice(&0u16.to_be_bytes()); // x
    wds.extend_from_slice(&0u16.to_be_bytes()); // y
    wds.extend_from_slice(&canvas.0.to_be_bytes());
    wds.extend_from_slice(&canvas.1.to_be_bytes());
    segment(&mut out, 0, SEG_WDS, &wds);

    // PDS.
    let mut pds = Vec::new();
    pds.push(0); // palette id
    pds.push(0); // palette version
    for (idx, rgba) in palette {
        // Convert to YCbCr.
        let r = rgba[0] as i32;
        let g = rgba[1] as i32;
        let b = rgba[2] as i32;
        let y = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
        let cb = (((-43 * r - 84 * g + 127 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        let cr = (((127 * r - 106 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        pds.push(*idx);
        pds.push(y);
        pds.push(cr);
        pds.push(cb);
        pds.push(rgba[3]);
    }
    segment(&mut out, 0, SEG_PDS, &pds);

    // ODS (first+last sequence).
    let mut rle = Vec::new();
    // encode `pixels` as a sequence of single-pixel runs + end-of-line.
    let w = object.0 as usize;
    let h = object.1 as usize;
    for row in 0..h {
        for c in 0..w {
            let p = pixels[row * w + c];
            if p == 0 {
                // short run-of-zero form: 00 01 (1 pixel of colour 0).
                rle.push(0);
                rle.push(1);
            } else {
                rle.push(p);
            }
        }
        rle.push(0);
        rle.push(0);
    }
    let mut ods = Vec::new();
    ods.extend_from_slice(&1u16.to_be_bytes()); // object id
    ods.push(0); // object version
    ods.push(0xC0); // first + last sequence
    let obj_data_len = (rle.len() + 4) as u32; // width+height (4) + rle
    ods.push(((obj_data_len >> 16) & 0xFF) as u8);
    ods.push(((obj_data_len >> 8) & 0xFF) as u8);
    ods.push((obj_data_len & 0xFF) as u8);
    ods.extend_from_slice(&object.0.to_be_bytes());
    ods.extend_from_slice(&object.1.to_be_bytes());
    ods.extend_from_slice(&rle);
    segment(&mut out, 0, SEG_ODS, &ods);

    // END.
    segment(&mut out, 0, SEG_END, &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_round_trip_simple() {
        // A 4×1 bitmap: 1 1 0 0 — encoded as two one-pixel runs of colour
        // 1 plus a 2-pixel run of colour 0, then end-of-line.
        let rle: &[u8] = &[0x01, 0x01, 0x00, 0x02, 0x00, 0x00];
        let px = decode_rle(rle, 4, 1).unwrap();
        assert_eq!(px, vec![1, 1, 0, 0]);
    }

    #[test]
    fn decodes_tiny_display_set() {
        // 2×2 bitmap: red red / green blue.
        let pixels = [1u8, 1, 2, 3];
        let palette = [
            (0u8, [0u8, 0, 0, 0]), // index 0 → transparent
            (1u8, [255u8, 0, 0, 255]),
            (2u8, [0u8, 255, 0, 255]),
            (3u8, [0u8, 0, 255, 255]),
        ];
        let blob = build_demo_display_set((2, 2), (2, 2), (0, 0), &palette, &pixels);

        let mut dec = make_decoder(&CodecParameters::video(CodecId::new(PGS_CODEC_ID))).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), blob).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.width, 2);
        assert_eq!(v.height, 2);
        assert_eq!(v.format, PixelFormat::Rgba);
        let data = &v.planes[0].data;
        assert_eq!(data.len(), 2 * 2 * 4);
        // Row 0: red, red — the YCbCr round-trip is approximate, so
        // check R dominance + opaque alpha rather than exact bytes.
        let r0c0 = &data[0..4];
        let r0c1 = &data[4..8];
        assert!(
            r0c0[0] > 200 && r0c0[3] == 255,
            "not red-dominant: {:?}",
            r0c0
        );
        assert!(
            r0c1[0] > 200 && r0c1[3] == 255,
            "not red-dominant: {:?}",
            r0c1
        );
        // Row 1: green, blue — same dominance check.
        let g = &data[8..12];
        let b = &data[12..16];
        assert!(
            g[1] > g[0] && g[1] > g[2],
            "green pixel not dominant: {:?}",
            g
        );
        assert!(
            b[2] > b[0] && b[2] > b[1],
            "blue pixel not dominant: {:?}",
            b
        );
    }
}
