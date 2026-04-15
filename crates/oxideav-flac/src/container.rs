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

use std::io::Read;

use oxideav_container::{Demuxer, Muxer, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::metadata::{BlockHeader, BlockType, StreamInfo as Si, FLAC_MAGIC};

pub fn register(reg: &mut oxideav_container::ContainerRegistry) {
    reg.register_demuxer("flac", open_demuxer);
    reg.register_muxer("flac", open_muxer);
    reg.register_extension("flac", "flac");
    reg.register_extension("fla", "flac");
}

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    skip_id3v2_if_present(&mut input)?;

    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if magic != FLAC_MAGIC {
        return Err(Error::invalid("not a FLAC stream (missing fLaC magic)"));
    }

    let mut extradata = Vec::new();
    let mut streaminfo: Option<Si> = None;
    loop {
        let mut hdr = [0u8; 4];
        input.read_exact(&mut hdr)?;
        let parsed = BlockHeader::parse(&hdr)?;
        let mut payload = vec![0u8; parsed.length as usize];
        input.read_exact(&mut payload)?;
        if streaminfo.is_none() && parsed.block_type == BlockType::StreamInfo {
            streaminfo = Some(Si::parse(&payload)?);
        }
        extradata.extend_from_slice(&hdr);
        extradata.extend_from_slice(&payload);
        if parsed.last {
            break;
        }
    }
    let info = streaminfo
        .ok_or_else(|| Error::invalid("FLAC stream missing STREAMINFO block"))?;

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

    Ok(Box::new(FlacDemuxer {
        input,
        streams: vec![stream],
        scan: FrameScanner::new(),
        eof: false,
    }))
}

/// If the file begins with an ID3v2 tag, advance past it. Many FLAC files in
/// the wild have one even though the FLAC spec does not technically require
/// support.
fn skip_id3v2_if_present(input: &mut Box<dyn ReadSeek>) -> Result<()> {
    let mut head = [0u8; 10];
    let n = read_up_to(input, &mut head)?;
    if n < 10 || &head[0..3] != b"ID3" {
        // Rewind whatever we read and return — no tag.
        input.seek(std::io::SeekFrom::Current(-(n as i64)))?;
        return Ok(());
    }
    let size = ((head[6] as u32) << 21)
        | ((head[7] as u32) << 14)
        | ((head[8] as u32) << 7)
        | (head[9] as u32);
    let mut skip = vec![0u8; size as usize];
    input.read_exact(&mut skip)?;
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
}

/// Buffered scanner that emits one packet per FLAC frame by looking for the
/// frame sync pattern (0xFF, 0xF8 or 0xF9). Adjacent identical-sync
/// false-positives are tolerable for remux because the concatenation of all
/// emitted packets reproduces the input byte-for-byte.
struct FrameScanner {
    buffer: Vec<u8>,
    /// Offset within `buffer` of the start of the next packet to emit.
    head: usize,
}

impl FrameScanner {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64 * 1024),
            head: 0,
        }
    }

    /// Try to extract the next packet from the buffer. Returns `None` when
    /// more bytes need to be read from the underlying input.
    fn try_take(&mut self, eof: bool) -> Option<Vec<u8>> {
        // Find the next frame sync at or after `self.head + 1` (skip the byte
        // that started this packet, if any, so we find the *next* frame).
        let search_start = if self.head < self.buffer.len() {
            // Make sure self.head points to a sync (it should).
            self.head + 1
        } else {
            return None;
        };
        let next = find_sync(&self.buffer, search_start);
        match next {
            Some(end) => {
                let pkt = self.buffer[self.head..end].to_vec();
                self.head = end;
                Some(pkt)
            }
            None if eof => {
                if self.head < self.buffer.len() {
                    let pkt = self.buffer[self.head..].to_vec();
                    self.head = self.buffer.len();
                    Some(pkt)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Compact the buffer to discard already-emitted bytes, freeing memory.
    fn compact(&mut self) {
        if self.head > 64 * 1024 {
            self.buffer.drain(..self.head);
            self.head = 0;
        }
    }

    /// Position the head at the first frame sync in the buffer (call once
    /// before the first take).
    fn align_to_first_sync(&mut self) -> Result<bool> {
        if let Some(off) = find_sync(&self.buffer, 0) {
            self.head = off;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Find the next byte index `i` in `buf` (i ≥ start) such that buf[i]==0xFF
/// and buf[i+1] in {0xF8, 0xF9}. Returns None if not found.
fn find_sync(buf: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 1 < buf.len() {
        if buf[i] == 0xFF && (buf[i + 1] == 0xF8 || buf[i + 1] == 0xF9) {
            return Some(i);
        }
        i += 1;
    }
    None
}

impl Demuxer for FlacDemuxer {
    fn format_name(&self) -> &str {
        "flac"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        // Lazy initialization: on first call, read enough bytes to find the
        // first sync.
        loop {
            // Try to take a packet first.
            if !self.scan.buffer.is_empty() {
                if let Some(data) = self.scan.try_take(self.eof) {
                    self.scan.compact();
                    let stream = &self.streams[0];
                    let mut pkt = Packet::new(0, stream.time_base, data);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
            }
            if self.eof {
                return Err(Error::Eof);
            }

            // Need more data.
            let mut chunk = [0u8; 8192];
            let n = read_up_to(&mut self.input, &mut chunk)?;
            if n == 0 {
                self.eof = true;
            } else {
                self.scan.buffer.extend_from_slice(&chunk[..n]);
            }

            // Establish initial alignment after the first read.
            if self.scan.head == 0 && !self.scan.buffer.is_empty() {
                if !self.scan.align_to_first_sync()? && self.eof {
                    return Err(Error::invalid(
                        "FLAC: no frame sync found in stream",
                    ));
                }
            }
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
