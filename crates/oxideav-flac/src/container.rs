//! FLAC native container: `fLaC` magic + metadata blocks + frame stream.
//!
//! The demuxer walks the metadata blocks to populate
//! [`CodecParameters`] from STREAMINFO, then emits frames as packets by
//! scanning for the FLAC frame sync pattern (0xFF, 0xF8/0xF9). The full set of
//! metadata blocks (including their headers) is preserved verbatim in
//! `extradata` so the muxer can round-trip byte-identical output.
//!
//! Per-packet timestamps are not yet computed — that requires parsing the
//! variable-length frame header in detail. The decoder (forthcoming) will do
//! that work.

use std::io::{Read, Seek, SeekFrom};

use oxideav_container::{Demuxer, Muxer, ReadSeek, WriteSeek};
use oxideav_core::{
    AttachedPicture, CodecId, CodecParameters, Error, MediaType, Packet, PictureType, Result,
    SampleFormat, StreamInfo, TimeBase,
};

use crate::frame::{parse_frame_header, FrameHeader};
use crate::metadata::{
    parse_seektable, BlockHeader, BlockType, SeekPoint, StreamInfo as Si, FLAC_MAGIC,
};

pub fn register(reg: &mut oxideav_container::ContainerRegistry) {
    reg.register_demuxer("flac", open_demuxer);
    reg.register_muxer("flac", open_muxer);
    reg.register_extension("flac", "flac");
    reg.register_extension("fla", "flac");
    reg.register_probe("flac", probe);
}

/// `fLaC` magic at offset 0, or after an ID3v2 tag at offset 0.
fn probe(p: &oxideav_container::ProbeData) -> u8 {
    if p.buf.len() >= 4 && &p.buf[0..4] == b"fLaC" {
        return 100;
    }
    if p.buf.len() >= 14 && &p.buf[0..3] == b"ID3" {
        let size = ((p.buf[6] as usize) << 21)
            | ((p.buf[7] as usize) << 14)
            | ((p.buf[8] as usize) << 7)
            | (p.buf[9] as usize);
        let off = 10 + size;
        if off + 4 <= p.buf.len() && &p.buf[off..off + 4] == b"fLaC" {
            return 100;
        }
    }
    0
}

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut pictures: Vec<AttachedPicture> = Vec::new();
    read_id3v2_if_present(&mut input, &mut metadata, &mut pictures)?;

    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if magic != FLAC_MAGIC {
        return Err(Error::invalid("not a FLAC stream (missing fLaC magic)"));
    }

    let mut extradata = Vec::new();
    let mut streaminfo: Option<Si> = None;
    let mut seek_points: Vec<SeekPoint> = Vec::new();
    loop {
        let mut hdr = [0u8; 4];
        input.read_exact(&mut hdr)?;
        let parsed = BlockHeader::parse(&hdr)?;
        let mut payload = vec![0u8; parsed.length as usize];
        input.read_exact(&mut payload)?;
        if streaminfo.is_none() && parsed.block_type == BlockType::StreamInfo {
            streaminfo = Some(Si::parse(&payload)?);
        }
        if parsed.block_type == BlockType::SeekTable {
            // Only the first SEEKTABLE is meaningful per spec.
            if seek_points.is_empty() {
                seek_points = parse_seektable(&payload);
            }
        }
        if parsed.block_type == BlockType::VorbisComment {
            parse_vorbis_comment(&payload, &mut metadata);
        }
        if parsed.block_type == BlockType::Picture {
            if let Some(pic) = parse_flac_picture_block(&payload) {
                pictures.push(pic);
            }
        }
        extradata.extend_from_slice(&hdr);
        extradata.extend_from_slice(&payload);
        if parsed.last {
            break;
        }
    }
    let info = streaminfo.ok_or_else(|| Error::invalid("FLAC stream missing STREAMINFO block"))?;

    // Byte position immediately after the final metadata block — this
    // is where the frame stream begins. The demuxer's seek_to adds the
    // SEEKPOINT's offset to this value to land on a frame boundary.
    let first_frame_offset = input.stream_position()?;

    let sample_format = match info.bits_per_sample {
        8 => SampleFormat::U8,
        16 => SampleFormat::S16,
        24 => SampleFormat::S24,
        32 => SampleFormat::S32,
        other => {
            return Err(Error::unsupported(format!(
                "unsupported FLAC bit depth {other}"
            )));
        }
    };

    let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(info.channels as u16);
    params.sample_rate = Some(info.sample_rate);
    params.sample_format = Some(sample_format);
    params.extradata = extradata;

    let time_base = TimeBase::new(1, info.sample_rate as i64);
    let total = if info.total_samples == 0 {
        None
    } else {
        Some(info.total_samples as i64)
    };
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: total,
        start_time: Some(0),
        params,
    };

    let duration_micros: i64 = if info.sample_rate > 0 && info.total_samples > 0 {
        (info.total_samples as i128 * 1_000_000 / info.sample_rate as i128) as i64
    } else {
        0
    };

    Ok(Box::new(FlacDemuxer {
        input,
        streams: vec![stream],
        scan: FrameScanner::new(info.min_block_size as u32),
        eof: false,
        metadata,
        pictures,
        duration_micros,
        seek_points,
        first_frame_offset,
    }))
}

/// Parse a FLAC `METADATA_BLOCK_PICTURE` payload (block type 6). The
/// wire format is shared with the base64 Vorbis-comment variant; the
/// only difference is that the Vorbis version is base64-wrapped.
fn parse_flac_picture_block(buf: &[u8]) -> Option<AttachedPicture> {
    let mut i = 0usize;
    fn read_u32(buf: &[u8], i: &mut usize) -> Option<u32> {
        if *i + 4 > buf.len() {
            return None;
        }
        let v = u32::from_be_bytes([buf[*i], buf[*i + 1], buf[*i + 2], buf[*i + 3]]);
        *i += 4;
        Some(v)
    }
    let type_raw = read_u32(buf, &mut i)?;
    let mime_len = read_u32(buf, &mut i)? as usize;
    if i + mime_len > buf.len() {
        return None;
    }
    let mime_type = std::str::from_utf8(&buf[i..i + mime_len])
        .unwrap_or("")
        .to_string();
    i += mime_len;
    let desc_len = read_u32(buf, &mut i)? as usize;
    if i + desc_len > buf.len() {
        return None;
    }
    let description = std::str::from_utf8(&buf[i..i + desc_len])
        .unwrap_or("")
        .to_string();
    i += desc_len;
    // Width, height, depth, colour count — skip: 4 × u32.
    if i + 16 > buf.len() {
        return None;
    }
    i += 16;
    let data_len = read_u32(buf, &mut i)? as usize;
    if i + data_len > buf.len() {
        return None;
    }
    let data = buf[i..i + data_len].to_vec();
    Some(AttachedPicture {
        mime_type,
        picture_type: PictureType::from_u8((type_raw & 0xFF) as u8),
        description,
        data,
    })
}

/// Parse a Vorbis-comment block payload (FLAC block type 4 shares the
/// Vorbis comment wire format). Appends normalised (lowercase key, value)
/// pairs to `out`.
fn parse_vorbis_comment(buf: &[u8], out: &mut Vec<(String, String)>) {
    let mut i = 0usize;
    // Vendor string
    if buf.len() < 4 {
        return;
    }
    let vlen = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    i += 4;
    if i + vlen > buf.len() {
        return;
    }
    let vendor = String::from_utf8_lossy(&buf[i..i + vlen]).to_string();
    i += vlen;
    if !vendor.is_empty() {
        out.push(("vendor".into(), vendor));
    }
    // Comment count + KEY=VALUE entries
    if i + 4 > buf.len() {
        return;
    }
    let n = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
    i += 4;
    for _ in 0..n {
        if i + 4 > buf.len() {
            break;
        }
        let clen = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
        i += 4;
        if i + clen > buf.len() {
            break;
        }
        let entry = &buf[i..i + clen];
        i += clen;
        if let Some(eq) = entry.iter().position(|&b| b == b'=') {
            let key = String::from_utf8_lossy(&entry[..eq])
                .to_ascii_lowercase()
                .trim()
                .to_string();
            let value = String::from_utf8_lossy(&entry[eq + 1..]).trim().to_string();
            if !key.is_empty() && !value.is_empty() {
                out.push((key, value));
            }
        }
    }
}

/// If the file begins with an ID3v2 tag, parse it and advance past
/// it. Many FLAC files in the wild have one even though the FLAC spec
/// does not require support. The parsed metadata/pictures are merged
/// into the caller's accumulators; FLAC's own Vorbis-comment and
/// PICTURE blocks later take precedence for duplicate keys.
fn read_id3v2_if_present(
    input: &mut Box<dyn ReadSeek>,
    metadata: &mut Vec<(String, String)>,
    pictures: &mut Vec<AttachedPicture>,
) -> Result<()> {
    let mut head = [0u8; 10];
    let n = read_up_to(input, &mut head)?;
    if n < 10 || &head[0..3] != b"ID3" {
        // Rewind whatever we read and return — no tag.
        input.seek(std::io::SeekFrom::Current(-(n as i64)))?;
        return Ok(());
    }
    let flags = head[5];
    let size = ((head[6] as u32) << 21)
        | ((head[7] as u32) << 14)
        | ((head[8] as u32) << 7)
        | (head[9] as u32);
    let footer = if flags & 0x10 != 0 { 10 } else { 0 };
    let mut body = vec![0u8; size as usize];
    if input.read_exact(&mut body).is_err() {
        if footer > 0 {
            input.seek(std::io::SeekFrom::Current(footer as i64))?;
        }
        return Ok(());
    }
    let mut full = Vec::with_capacity(10 + body.len());
    full.extend_from_slice(&head);
    full.extend_from_slice(&body);
    if let Ok((tag, _)) = oxideav_id3::parse_tag(&full) {
        for (k, v) in oxideav_id3::to_key_value_pairs(&tag) {
            metadata.push((k, v));
        }
        pictures.extend(oxideav_id3::attached_pictures(&tag));
    }
    if footer > 0 {
        input.seek(std::io::SeekFrom::Current(footer as i64))?;
    }
    Ok(())
}

fn read_up_to(input: &mut Box<dyn ReadSeek>, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match input.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(got)
}

struct FlacDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    scan: FrameScanner,
    eof: bool,
    metadata: Vec<(String, String)>,
    pictures: Vec<AttachedPicture>,
    duration_micros: i64,
    /// Parsed SEEKTABLE entries (placeholders filtered out). Empty if
    /// the file carried no SEEKTABLE block.
    seek_points: Vec<SeekPoint>,
    /// Byte offset of the first frame in the stream (immediately
    /// after the metadata-block sequence).
    first_frame_offset: u64,
}

/// Buffered FLAC frame scanner.
///
/// Each candidate sync (0xFF + 0xF8/0xF9) is verified by parsing the frame
/// header and checking its CRC-8. Verified frames anchor the start of the
/// packet to emit; the next verified frame anchors the end. False positives
/// are filtered by header-CRC mismatch, so byte-identical-data syncs that
/// happen to occur inside encoded residuals don't trip us.
struct FrameScanner {
    buffer: Vec<u8>,
    /// Offset within `buffer` of the start of the current packet.
    head: usize,
    /// Frame header for the packet starting at `head` (None until first frame found).
    head_frame: Option<FrameHeader>,
    /// Block size to use for fixed-blocking pts calculation (from STREAMINFO).
    streaminfo_block_size: u32,
    /// Running sample counter — fallback when frame headers don't directly
    /// provide a sample number (or for sanity checking).
    samples_emitted: u64,
}

impl FrameScanner {
    fn new(streaminfo_block_size: u32) -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024),
            head: 0,
            head_frame: None,
            streaminfo_block_size,
            samples_emitted: 0,
        }
    }

    /// Find the next valid (CRC-8-verified) frame header at or after `start`.
    /// Returns its offset in `buffer` and the parsed header.
    fn next_valid_frame(&self, start: usize) -> Option<(usize, FrameHeader)> {
        let mut i = start;
        while i + 1 < self.buffer.len() {
            if self.buffer[i] == 0xFF && (self.buffer[i + 1] == 0xF8 || self.buffer[i + 1] == 0xF9)
            {
                if let Ok(h) = parse_frame_header(&self.buffer[i..]) {
                    return Some((i, h));
                }
            }
            i += 1;
        }
        None
    }

    /// Pop the next emittable packet, if one is fully available.
    fn try_take(&mut self, eof: bool) -> Option<EmittedFrame> {
        // Locate the first frame the first time we're called. Anchor the
        // search at `self.head` so that after we've emitted the final frame
        // (which leaves `head == buffer.len()`) we don't rediscover the very
        // first frame again.
        if self.head_frame.is_none() {
            let (off, h) = self.next_valid_frame(self.head)?;
            self.head = off;
            self.head_frame = Some(h);
        }

        let head_frame = self.head_frame.as_ref().unwrap().clone();
        let search_start = self.head + head_frame.header_byte_len;

        match self.next_valid_frame(search_start) {
            Some((end, next_h)) => {
                let data = self.buffer[self.head..end].to_vec();
                let pts = head_frame.first_sample(self.streaminfo_block_size);
                let block_size = head_frame.block_size;
                self.samples_emitted = pts + block_size as u64;
                self.head = end;
                self.head_frame = Some(next_h);
                Some(EmittedFrame {
                    data,
                    pts: pts as i64,
                    duration: block_size as i64,
                })
            }
            None if eof => {
                if self.head < self.buffer.len() {
                    let data = self.buffer[self.head..].to_vec();
                    let pts = head_frame.first_sample(self.streaminfo_block_size);
                    let block_size = head_frame.block_size;
                    self.samples_emitted = pts + block_size as u64;
                    self.head = self.buffer.len();
                    self.head_frame = None;
                    Some(EmittedFrame {
                        data,
                        pts: pts as i64,
                        duration: block_size as i64,
                    })
                } else {
                    None
                }
            }
            None => None,
        }
    }

    fn compact(&mut self) {
        if self.head > 64 * 1024 {
            self.buffer.drain(..self.head);
            self.head = 0;
        }
    }

    /// Drop any buffered bytes and forget the anchored head-frame so
    /// that subsequent `try_take` calls re-scan from the post-seek
    /// stream position. `samples_emitted` is forced to
    /// `anchor_sample`, which the scanner uses as a ground-truth pts
    /// for variable-blocking frames (and for the `samples_emitted`
    /// accounting).
    fn reset_to(&mut self, anchor_sample: u64) {
        self.buffer.clear();
        self.head = 0;
        self.head_frame = None;
        self.samples_emitted = anchor_sample;
    }
}

struct EmittedFrame {
    data: Vec<u8>,
    pts: i64,
    duration: i64,
}

impl Demuxer for FlacDemuxer {
    fn format_name(&self) -> &str {
        "flac"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        loop {
            if !self.scan.buffer.is_empty() {
                if let Some(emitted) = self.scan.try_take(self.eof) {
                    self.scan.compact();
                    let stream = &self.streams[0];
                    let mut pkt = Packet::new(0, stream.time_base, emitted.data);
                    pkt.pts = Some(emitted.pts);
                    pkt.dts = Some(emitted.pts);
                    pkt.duration = Some(emitted.duration);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
            }
            if self.eof {
                return Err(Error::Eof);
            }

            let mut chunk = [0u8; 8192];
            let n = read_up_to(&mut self.input, &mut chunk)?;
            if n == 0 {
                self.eof = true;
            } else {
                self.scan.buffer.extend_from_slice(&chunk[..n]);
            }
        }
    }

    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if stream_index != 0 {
            return Err(Error::invalid(format!(
                "FLAC: stream index {stream_index} out of range (only stream 0 exists)"
            )));
        }
        if self.seek_points.is_empty() {
            return Err(Error::unsupported(
                "FLAC: file has no SEEKTABLE metadata block; cannot seek",
            ));
        }
        let target = pts.max(0) as u64;
        // Binary-search for the last SeekPoint with sample_number <= target.
        // partition_point returns the first index where the predicate is
        // false, so `idx - 1` is our candidate.
        let idx = self
            .seek_points
            .partition_point(|sp| sp.sample_number <= target);
        let (anchor_sample, byte_offset) = if idx == 0 {
            // Target is before the first seek point — land on sample 0
            // at the start of the frame stream.
            (0u64, 0u64)
        } else {
            let sp = &self.seek_points[idx - 1];
            (sp.sample_number, sp.offset)
        };
        self.input
            .seek(SeekFrom::Start(self.first_frame_offset + byte_offset))?;
        self.scan.reset_to(anchor_sample);
        self.eof = false;
        Ok(anchor_sample as i64)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn attached_pictures(&self) -> &[AttachedPicture] {
        &self.pictures
    }

    fn duration_micros(&self) -> Option<i64> {
        if self.duration_micros > 0 {
            Some(self.duration_micros)
        } else {
            None
        }
    }
}

// --- Muxer -----------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported(
            "FLAC native container holds exactly one stream",
        ));
    }
    let s = &streams[0];
    if s.params.codec_id.as_str() != crate::CODEC_ID_STR {
        return Err(Error::invalid(format!(
            "FLAC muxer requires codec_id=flac (got {})",
            s.params.codec_id
        )));
    }
    if s.params.extradata.is_empty() {
        return Err(Error::invalid(
            "FLAC muxer needs extradata containing metadata blocks",
        ));
    }
    Ok(Box::new(FlacMuxer {
        output,
        extradata: s.params.extradata.clone(),
        header_written: false,
        trailer_written: false,
    }))
}

struct FlacMuxer {
    output: Box<dyn WriteSeek>,
    extradata: Vec<u8>,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for FlacMuxer {
    fn format_name(&self) -> &str {
        "flac"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("FLAC muxer: write_header called twice"));
        }
        use std::io::Write;
        self.output.write_all(&FLAC_MAGIC)?;
        self.output.write_all(&self.extradata)?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("FLAC muxer: write_header not called"));
        }
        use std::io::Write;
        self.output.write_all(&packet.data)?;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        use std::io::Write;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}
