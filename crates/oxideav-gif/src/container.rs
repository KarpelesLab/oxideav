//! GIF container (demuxer + muxer).
//!
//! A GIF file is a header (signature + Logical Screen Descriptor + optional
//! Global Color Table) followed by an unbounded chain of blocks:
//!
//! ```text
//! 0x2C — Image Descriptor (+ optional Local Color Table + LZW image data)
//! 0x21 — Extension block (next byte = sub-type):
//!        0xF9 — Graphic Control Extension (delay, disposal, transparency)
//!        0xFF — Application Extension (NETSCAPE2.0 = loop count; others ignored)
//!        0xFE — Comment Extension (ignored — text)
//!        0x01 — Plain Text Extension (ignored — legacy)
//! 0x3B — Trailer (end of stream)
//! ```
//!
//! LZW image data is stored as a single minimum-code-size byte followed by
//! a sub-block chain: `<len><len bytes of compressed data>...<0>`. Extension
//! blocks use the same sub-block chain after their header.
//!
//! The demuxer emits one packet per image frame (there is exactly one image
//! block per frame; the preceding GCE binds to it). Every packet's payload
//! is a self-contained frame record the decoder consumes — see
//! [`encode_frame_payload`] / [`decode_frame_payload`] for the layout.
//!
//! The muxer accepts packets produced by the encoder and emits a GIF89a
//! file. When more than one packet is written it also emits a NETSCAPE2.0
//! application extension to set the loop count (0 = infinite).

use std::io::{Read, SeekFrom, Write};

use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, PixelFormat, Result, StreamInfo, TimeBase,
};

/// Codec id registered for GIF image frames.
pub const GIF_CODEC_ID: &str = "gif";

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("gif", open);
    reg.register_muxer("gif", open_muxer);
    reg.register_extension("gif", "gif");
    reg.register_probe("gif", probe);
}

fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() < 6 {
        return 0;
    }
    if &p.buf[0..6] == b"GIF87a" || &p.buf[0..6] == b"GIF89a" {
        return 100;
    }
    0
}

fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut buf = Vec::new();
    input.seek(SeekFrom::Start(0))?;
    input.read_to_end(&mut buf)?;
    drop(input);
    let parsed = parse_gif(&buf)?;

    let mut params = CodecParameters::video(CodecId::new(GIF_CODEC_ID));
    params.media_type = MediaType::Video;
    params.width = Some(parsed.canvas_w);
    params.height = Some(parsed.canvas_h);
    params.pixel_format = Some(PixelFormat::Pal8);
    // Extradata: global palette (RGBA bytes, 4 per entry) — empty when none.
    params.extradata = palette_to_extradata(&parsed.global_palette);

    // GIF durations are in hundredths of a second. Use 1/100 s time base.
    let time_base = TimeBase::new(1, 100);
    let total: i64 = parsed.frames.iter().map(|f| f.delay_cs.max(1) as i64).sum();
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total),
        start_time: Some(0),
        params,
    };

    let metadata: Vec<(String, String)> = if let Some(loop_count) = parsed.loop_count {
        vec![("loop_count".into(), loop_count.to_string())]
    } else {
        Vec::new()
    };

    Ok(Box::new(GifDemuxer {
        stream,
        packets: parsed.into_packets(time_base),
        pos: 0,
        metadata,
    }))
}

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::invalid(
            "GIF muxer: exactly one video stream is required",
        ));
    }
    let s = &streams[0];
    if s.params.codec_id.as_str() != GIF_CODEC_ID {
        return Err(Error::invalid(format!(
            "GIF muxer: expected codec id `gif`, got `{}`",
            s.params.codec_id
        )));
    }
    let w = s
        .params
        .width
        .ok_or_else(|| Error::invalid("GIF muxer: missing width"))?;
    let h = s
        .params
        .height
        .ok_or_else(|| Error::invalid("GIF muxer: missing height"))?;
    let gct = extradata_to_palette(&s.params.extradata);
    Ok(Box::new(GifMuxer {
        out: output,
        canvas: (w, h),
        gct,
        header_written: false,
        packets_written: 0,
        buffered_packets: Vec::new(),
    }))
}

// ---- Internal frame payload format ---------------------------------------

/// Serialise one parsed frame into a self-contained packet payload the
/// decoder consumes. Layout:
///
/// ```text
///   magic    "OGIF" (4)
///   version  1 byte (currently 1)
///   flags    1 byte
///              bit0 = has_local_palette
///              bit1 = has_transparent
///              bit2 = interlaced
///   disposal u8
///   reserved u8
///   canvas_w u16 LE
///   canvas_h u16 LE
///   x        u16 LE
///   y        u16 LE
///   w        u16 LE
///   h        u16 LE
///   delay_cs u16 LE
///   transp_i u8
///   min_code_size u8
///   palette_len u16 LE   (0 when !has_local_palette — global palette in use)
///   [ palette_len * 4 bytes RGBA ]    (only when has_local_palette)
///   lzw_len  u32 LE
///   [ lzw_len bytes LZW-compressed indices ]
/// ```
pub(crate) fn encode_frame_payload(f: &ParsedFrame, canvas: (u32, u32)) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + f.lzw_data.len() + f.local_palette.len() * 4);
    out.extend_from_slice(b"OGIF");
    out.push(1);
    let mut flags = 0u8;
    if !f.local_palette.is_empty() {
        flags |= 0x01;
    }
    if f.transparent_index.is_some() {
        flags |= 0x02;
    }
    if f.interlaced {
        flags |= 0x04;
    }
    out.push(flags);
    out.push(f.disposal);
    out.push(0); // reserved
    out.extend_from_slice(&(canvas.0 as u16).to_le_bytes());
    out.extend_from_slice(&(canvas.1 as u16).to_le_bytes());
    out.extend_from_slice(&(f.x as u16).to_le_bytes());
    out.extend_from_slice(&(f.y as u16).to_le_bytes());
    out.extend_from_slice(&(f.w as u16).to_le_bytes());
    out.extend_from_slice(&(f.h as u16).to_le_bytes());
    out.extend_from_slice(&f.delay_cs.to_le_bytes());
    out.push(f.transparent_index.unwrap_or(0));
    out.push(f.min_code_size);
    out.extend_from_slice(&(f.local_palette.len() as u16).to_le_bytes());
    for c in &f.local_palette {
        out.extend_from_slice(c);
    }
    out.extend_from_slice(&(f.lzw_data.len() as u32).to_le_bytes());
    out.extend_from_slice(&f.lzw_data);
    out
}

pub(crate) fn decode_frame_payload(buf: &[u8]) -> Result<DecodedFrame<'_>> {
    // Fixed-header size: magic 4 + ver 1 + flags 1 + disp 1 + rsv 1 +
    //                    6*u16 + u16 + u8 + u8 + u16 + u32 = 30
    const FIXED: usize = 4 + 1 + 1 + 1 + 1 + 6 * 2 + 2 + 1 + 1 + 2 + 4;
    if buf.len() < FIXED {
        return Err(Error::invalid("GIF: frame payload too short"));
    }
    if &buf[0..4] != b"OGIF" {
        return Err(Error::invalid("GIF: bad frame payload magic"));
    }
    if buf[4] != 1 {
        return Err(Error::invalid("GIF: unknown frame payload version"));
    }
    let flags = buf[5];
    let disposal = buf[6];
    // buf[7] reserved
    let mut p = 8;
    let read_u16 = |p: &mut usize, b: &[u8]| -> u16 {
        let v = u16::from_le_bytes([b[*p], b[*p + 1]]);
        *p += 2;
        v
    };
    let read_u32 = |p: &mut usize, b: &[u8]| -> u32 {
        let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
        *p += 4;
        v
    };
    let canvas_w = read_u16(&mut p, buf) as u32;
    let canvas_h = read_u16(&mut p, buf) as u32;
    let x = read_u16(&mut p, buf) as u32;
    let y = read_u16(&mut p, buf) as u32;
    let w = read_u16(&mut p, buf) as u32;
    let h = read_u16(&mut p, buf) as u32;
    let delay_cs = read_u16(&mut p, buf);
    let transp_i = buf[p];
    p += 1;
    let min_code_size = buf[p];
    p += 1;
    let palette_len = read_u16(&mut p, buf) as usize;
    let has_local = flags & 0x01 != 0;
    let local_palette = if has_local {
        if p + palette_len * 4 > buf.len() {
            return Err(Error::invalid("GIF: local palette truncated"));
        }
        let start = p;
        p += palette_len * 4;
        &buf[start..p]
    } else {
        &[][..]
    };
    let lzw_len = read_u32(&mut p, buf) as usize;
    if p + lzw_len > buf.len() {
        return Err(Error::invalid("GIF: LZW data truncated"));
    }
    let lzw = &buf[p..p + lzw_len];
    Ok(DecodedFrame {
        has_transparent: flags & 0x02 != 0,
        interlaced: flags & 0x04 != 0,
        disposal,
        canvas: (canvas_w, canvas_h),
        x,
        y,
        w,
        h,
        delay_cs,
        transparent_index: transp_i,
        min_code_size,
        local_palette,
        lzw,
    })
}

pub(crate) struct DecodedFrame<'a> {
    pub has_transparent: bool,
    #[allow(dead_code)]
    pub interlaced: bool,
    #[allow(dead_code)]
    pub disposal: u8,
    #[allow(dead_code)]
    pub canvas: (u32, u32),
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    #[allow(dead_code)]
    pub delay_cs: u16,
    pub transparent_index: u8,
    pub min_code_size: u8,
    /// Packed `RGBA`×N bytes (empty when the global palette is in use).
    pub local_palette: &'a [u8],
    pub lzw: &'a [u8],
}

// ---- Parser --------------------------------------------------------------

/// Frame extracted from the on-disk GIF file.
#[derive(Debug)]
pub(crate) struct ParsedFrame {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub delay_cs: u16,
    pub disposal: u8,
    pub transparent_index: Option<u8>,
    pub interlaced: bool,
    pub min_code_size: u8,
    pub local_palette: Vec<[u8; 4]>,
    pub lzw_data: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct ParsedFile {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub global_palette: Vec<[u8; 4]>,
    pub frames: Vec<ParsedFrame>,
    pub loop_count: Option<u16>,
}

impl ParsedFile {
    fn into_packets(self, tb: TimeBase) -> Vec<Packet> {
        let mut pkts = Vec::with_capacity(self.frames.len());
        let mut pts: i64 = 0;
        let canvas = (self.canvas_w, self.canvas_h);
        for (i, f) in self.frames.into_iter().enumerate() {
            let dur = f.delay_cs.max(1) as i64;
            let data = encode_frame_payload(&f, canvas);
            let mut pkt = Packet::new(0, tb, data);
            pkt.pts = Some(pts);
            pkt.dts = Some(pts);
            pkt.duration = Some(dur);
            pkt.flags.keyframe = i == 0;
            pts += dur;
            pkts.push(pkt);
        }
        pkts
    }
}

pub(crate) fn parse_gif(buf: &[u8]) -> Result<ParsedFile> {
    if buf.len() < 13 {
        return Err(Error::invalid("GIF: file too short"));
    }
    if &buf[0..6] != b"GIF87a" && &buf[0..6] != b"GIF89a" {
        return Err(Error::invalid("GIF: bad signature"));
    }
    let canvas_w = u16::from_le_bytes([buf[6], buf[7]]) as u32;
    let canvas_h = u16::from_le_bytes([buf[8], buf[9]]) as u32;
    let packed = buf[10];
    // buf[11] = bg index, buf[12] = aspect — we don't use either.
    let mut p = 13usize;

    let has_gct = packed & 0x80 != 0;
    let gct_size_exp = (packed & 0x07) as u32 + 1;
    let gct_len = 1usize << gct_size_exp;

    let global_palette = if has_gct {
        if p + gct_len * 3 > buf.len() {
            return Err(Error::invalid("GIF: global color table truncated"));
        }
        let pal = read_palette(&buf[p..p + gct_len * 3], gct_len);
        p += gct_len * 3;
        pal
    } else {
        Vec::new()
    };

    let mut frames: Vec<ParsedFrame> = Vec::new();
    let mut loop_count: Option<u16> = None;
    let mut pending_gce: Option<PendingGce> = None;

    while p < buf.len() {
        let marker = buf[p];
        p += 1;
        match marker {
            0x3B => break, // trailer
            0x21 => {
                // Extension.
                if p >= buf.len() {
                    return Err(Error::invalid("GIF: extension marker at EOF"));
                }
                let label = buf[p];
                p += 1;
                match label {
                    0xF9 => {
                        // Graphic Control Extension — always a single 4-byte
                        // sub-block followed by a 0 terminator.
                        let sub = read_sub_blocks(buf, &mut p)?;
                        if sub.len() < 4 {
                            return Err(Error::invalid("GIF: GCE too short"));
                        }
                        let flags = sub[0];
                        let delay = u16::from_le_bytes([sub[1], sub[2]]);
                        let transp = sub[3];
                        let has_transp = flags & 0x01 != 0;
                        let disposal = (flags >> 2) & 0x07;
                        pending_gce = Some(PendingGce {
                            delay_cs: delay,
                            disposal,
                            transparent_index: if has_transp { Some(transp) } else { None },
                        });
                    }
                    0xFF => {
                        // Application Extension. First sub-block holds the
                        // 11-byte identifier ("NETSCAPE2.0" + "2.0"). The
                        // remaining sub-blocks carry loop data.
                        let first = read_single_sub_block(buf, &mut p)?;
                        let data = read_sub_blocks(buf, &mut p)?;
                        if first == b"NETSCAPE2.0" && data.len() >= 3 && data[0] == 0x01 {
                            loop_count = Some(u16::from_le_bytes([data[1], data[2]]));
                        }
                    }
                    0xFE | 0x01 => {
                        // Comment or Plain Text — skip.
                        let _ = read_sub_blocks(buf, &mut p)?;
                    }
                    _ => {
                        // Unknown extension — skip defensively.
                        let _ = read_sub_blocks(buf, &mut p)?;
                    }
                }
            }
            0x2C => {
                // Image Descriptor.
                if p + 9 > buf.len() {
                    return Err(Error::invalid("GIF: image descriptor truncated"));
                }
                let x = u16::from_le_bytes([buf[p], buf[p + 1]]) as u32;
                let y = u16::from_le_bytes([buf[p + 2], buf[p + 3]]) as u32;
                let w = u16::from_le_bytes([buf[p + 4], buf[p + 5]]) as u32;
                let h = u16::from_le_bytes([buf[p + 6], buf[p + 7]]) as u32;
                let flags = buf[p + 8];
                p += 9;
                let has_lct = flags & 0x80 != 0;
                let interlaced = flags & 0x40 != 0;
                let lct_size_exp = (flags & 0x07) as u32 + 1;
                let lct_len = 1usize << lct_size_exp;
                let local_palette = if has_lct {
                    if p + lct_len * 3 > buf.len() {
                        return Err(Error::invalid("GIF: local color table truncated"));
                    }
                    let pal = read_palette(&buf[p..p + lct_len * 3], lct_len);
                    p += lct_len * 3;
                    pal
                } else {
                    Vec::new()
                };
                if p >= buf.len() {
                    return Err(Error::invalid("GIF: missing LZW min-code-size"));
                }
                let min_code_size = buf[p];
                p += 1;
                let lzw_data = read_sub_blocks(buf, &mut p)?;
                let gce = pending_gce.take().unwrap_or_default();
                frames.push(ParsedFrame {
                    x,
                    y,
                    w,
                    h,
                    delay_cs: gce.delay_cs,
                    disposal: gce.disposal,
                    transparent_index: gce.transparent_index,
                    interlaced,
                    min_code_size,
                    local_palette,
                    lzw_data,
                });
            }
            other => {
                return Err(Error::invalid(format!(
                    "GIF: unexpected block 0x{:02x}",
                    other
                )));
            }
        }
    }

    if frames.is_empty() {
        return Err(Error::invalid("GIF: no image frames"));
    }

    Ok(ParsedFile {
        canvas_w,
        canvas_h,
        global_palette,
        frames,
        loop_count,
    })
}

#[derive(Default)]
struct PendingGce {
    delay_cs: u16,
    disposal: u8,
    transparent_index: Option<u8>,
}

fn read_palette(bytes: &[u8], n: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = bytes[i * 3];
        let g = bytes[i * 3 + 1];
        let b = bytes[i * 3 + 2];
        out.push([r, g, b, 0xFF]);
    }
    out
}

/// Read a sub-block chain starting at `p`. A chain is a series of
/// `<len><len bytes>` pairs terminated by a zero-length block. All
/// bytes are concatenated into a single buffer.
fn read_sub_blocks(buf: &[u8], p: &mut usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        if *p >= buf.len() {
            return Err(Error::invalid("GIF: sub-block chain truncated"));
        }
        let len = buf[*p] as usize;
        *p += 1;
        if len == 0 {
            break;
        }
        if *p + len > buf.len() {
            return Err(Error::invalid("GIF: sub-block runs past EOF"));
        }
        out.extend_from_slice(&buf[*p..*p + len]);
        *p += len;
    }
    Ok(out)
}

/// Read only the *first* sub-block and skip nothing else. Useful for the
/// application-extension identifier, which is always the first block and
/// always 11 bytes.
fn read_single_sub_block(buf: &[u8], p: &mut usize) -> Result<Vec<u8>> {
    if *p >= buf.len() {
        return Err(Error::invalid("GIF: sub-block at EOF"));
    }
    let len = buf[*p] as usize;
    *p += 1;
    if *p + len > buf.len() {
        return Err(Error::invalid("GIF: sub-block runs past EOF"));
    }
    let out = buf[*p..*p + len].to_vec();
    *p += len;
    Ok(out)
}

// ---- Palette <-> extradata conversion ------------------------------------

/// Convert a palette into `extradata` bytes: `N:u16 LE` + `N * 4` RGBA bytes.
pub(crate) fn palette_to_extradata(pal: &[[u8; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + pal.len() * 4);
    out.extend_from_slice(&(pal.len() as u16).to_le_bytes());
    for c in pal {
        out.extend_from_slice(c);
    }
    out
}

pub(crate) fn extradata_to_palette(data: &[u8]) -> Vec<[u8; 4]> {
    if data.len() < 2 {
        return Vec::new();
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let mut out = Vec::with_capacity(n);
    let mut p = 2;
    for _ in 0..n {
        if p + 4 > data.len() {
            break;
        }
        out.push([data[p], data[p + 1], data[p + 2], data[p + 3]]);
        p += 4;
    }
    out
}

// ---- Demuxer -------------------------------------------------------------

struct GifDemuxer {
    stream: StreamInfo,
    packets: Vec<Packet>,
    pos: usize,
    metadata: Vec<(String, String)>,
}

impl Demuxer for GifDemuxer {
    fn format_name(&self) -> &str {
        "gif"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.pos >= self.packets.len() {
            return Err(Error::Eof);
        }
        let pkt = self.packets[self.pos].clone();
        self.pos += 1;
        Ok(pkt)
    }

    fn duration_micros(&self) -> Option<i64> {
        // Centiseconds → microseconds.
        self.stream.duration.map(|d| d * 10_000)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

// ---- Muxer ---------------------------------------------------------------

struct GifMuxer {
    out: Box<dyn WriteSeek>,
    canvas: (u32, u32),
    gct: Vec<[u8; 4]>,
    header_written: bool,
    packets_written: usize,
    buffered_packets: Vec<Packet>,
}

impl Muxer for GifMuxer {
    fn format_name(&self) -> &str {
        "gif"
    }

    fn write_header(&mut self) -> Result<()> {
        // Buffered — the muxer needs to know whether there will be more
        // than one packet before deciding whether to emit NETSCAPE2.0.
        // We delay the actual header write until write_trailer(); collect
        // packets in the meantime.
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid("GIF muxer: write_header not called"));
        }
        self.buffered_packets.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let (cw, ch) = self.canvas;
        let mut buf: Vec<u8> = Vec::new();
        // Signature.
        buf.extend_from_slice(b"GIF89a");
        // Logical Screen Descriptor.
        buf.extend_from_slice(&(cw as u16).to_le_bytes());
        buf.extend_from_slice(&(ch as u16).to_le_bytes());
        let gct_present = !self.gct.is_empty();
        let gct_size_exp = if gct_present {
            size_exp_for(self.gct.len())
        } else {
            0
        };
        let mut packed: u8 = 0;
        if gct_present {
            packed |= 0x80;
            packed |= 0x70; // color resolution = 7 (common default)
            packed |= (gct_size_exp as u8) & 0x07;
        }
        buf.push(packed);
        buf.push(0); // background color index
        buf.push(0); // pixel aspect ratio

        if gct_present {
            let padded_len = 1usize << (gct_size_exp + 1);
            write_palette(&mut buf, &self.gct, padded_len);
        }

        // NETSCAPE2.0 loop extension when we have >1 frame.
        if self.buffered_packets.len() > 1 {
            // 0x21 0xFF 0x0B "NETSCAPE2.0" 0x03 0x01 loop_lo loop_hi 0x00
            buf.push(0x21);
            buf.push(0xFF);
            buf.push(0x0B);
            buf.extend_from_slice(b"NETSCAPE2.0");
            buf.push(0x03);
            buf.push(0x01);
            buf.push(0x00);
            buf.push(0x00);
            buf.push(0x00);
        }

        for pkt in &self.buffered_packets {
            write_frame(&mut buf, pkt, gct_present)?;
        }
        // Trailer.
        buf.push(0x3B);
        self.out.write_all(&buf)?;
        self.packets_written = self.buffered_packets.len();
        self.buffered_packets.clear();
        Ok(())
    }
}

fn size_exp_for(n: usize) -> u32 {
    // GIF stores size-1 as `2^(size+1)` entries, so for N colours the
    // exponent is `ceil(log2(N)) - 1`, clamped to `[0, 7]`.
    if n <= 2 {
        0
    } else if n <= 4 {
        1
    } else if n <= 8 {
        2
    } else if n <= 16 {
        3
    } else if n <= 32 {
        4
    } else if n <= 64 {
        5
    } else if n <= 128 {
        6
    } else {
        7
    }
}

fn write_palette(buf: &mut Vec<u8>, pal: &[[u8; 4]], padded_len: usize) {
    for i in 0..padded_len {
        if i < pal.len() {
            buf.push(pal[i][0]);
            buf.push(pal[i][1]);
            buf.push(pal[i][2]);
        } else {
            buf.push(0);
            buf.push(0);
            buf.push(0);
        }
    }
}

fn write_frame(buf: &mut Vec<u8>, pkt: &Packet, gct_present: bool) -> Result<()> {
    let df = decode_frame_payload(&pkt.data)?;
    // Graphic Control Extension — emit always when we have animation info.
    buf.push(0x21);
    buf.push(0xF9);
    buf.push(0x04); // block size
    let mut flags = 0u8;
    flags |= (df.disposal & 0x07) << 2;
    if df.has_transparent {
        flags |= 0x01;
    }
    buf.push(flags);
    buf.extend_from_slice(&df.delay_cs.to_le_bytes());
    buf.push(df.transparent_index);
    buf.push(0); // block terminator

    // Image Descriptor.
    buf.push(0x2C);
    buf.extend_from_slice(&(df.x as u16).to_le_bytes());
    buf.extend_from_slice(&(df.y as u16).to_le_bytes());
    buf.extend_from_slice(&(df.w as u16).to_le_bytes());
    buf.extend_from_slice(&(df.h as u16).to_le_bytes());
    // Packed: LCT? / interlace / sort / reserved / LCT size.
    let has_local = !df.local_palette.is_empty();
    let mut packed: u8 = 0;
    let lct_exp = if has_local {
        let n = df.local_palette.len() / 4;
        size_exp_for(n)
    } else {
        0
    };
    if has_local {
        packed |= 0x80;
        packed |= (lct_exp as u8) & 0x07;
    }
    if df.interlaced {
        packed |= 0x40;
    }
    buf.push(packed);
    if has_local {
        let padded = 1usize << (lct_exp + 1);
        // Convert RGBA palette to RGB in-file.
        for i in 0..padded {
            if i * 4 + 3 < df.local_palette.len() {
                buf.push(df.local_palette[i * 4]);
                buf.push(df.local_palette[i * 4 + 1]);
                buf.push(df.local_palette[i * 4 + 2]);
            } else {
                buf.push(0);
                buf.push(0);
                buf.push(0);
            }
        }
    } else if !gct_present {
        return Err(Error::invalid(
            "GIF muxer: frame has no local palette and no global palette",
        ));
    }
    // LZW min-code-size + compressed sub-block chain.
    buf.push(df.min_code_size);
    write_sub_blocks(buf, df.lzw);
    Ok(())
}

fn write_sub_blocks(buf: &mut Vec<u8>, data: &[u8]) {
    let mut p = 0;
    while p < data.len() {
        let chunk = (data.len() - p).min(255);
        buf.push(chunk as u8);
        buf.extend_from_slice(&data[p..p + chunk]);
        p += chunk;
    }
    buf.push(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_gif89a() {
        let mut buf = vec![0u8; 16];
        buf[..6].copy_from_slice(b"GIF89a");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 100);
    }

    #[test]
    fn probe_gif87a() {
        let mut buf = vec![0u8; 16];
        buf[..6].copy_from_slice(b"GIF87a");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 100);
    }

    #[test]
    fn probe_other() {
        let buf = vec![0u8; 16];
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }
}
