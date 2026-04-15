//! Ogg muxer: pack incoming packets into pages.
//!
//! Strategy: maintain one buffered page per logical stream. Pack a packet by
//! appending its bytes and lacing values. Flush the page whenever it reaches
//! the 255-segment limit, when an explicit flush is requested, or at trailer
//! time. Granule positions come from `Packet::pts` for non-header packets.

use std::collections::HashMap;
use std::io::Write;

use oxideav_container::{Muxer, WriteSeek};
use oxideav_core::{CodecId, Error, Packet, Result, StreamInfo};

use crate::codec_id;
use crate::page::{self, flags, lace, Page};

pub fn open(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    let mut per_stream = HashMap::with_capacity(streams.len());
    for s in streams {
        let serial = derive_serial(s);
        let headers_remaining = codec_id::header_packet_count(&s.params.codec_id);
        per_stream.insert(
            s.index,
            StreamWriter {
                serial,
                seq_no: 0,
                buffered: PageBuilder::new(),
                headers_remaining,
                bos_emitted: false,
                pending_bytes: None,
            },
        );
    }
    Ok(Box::new(OggMuxer {
        output,
        streams: streams.to_vec(),
        per_stream,
        stream_order: streams.iter().map(|s| s.index).collect(),
        header_written: false,
        trailer_written: false,
    }))
}

/// Derive a stable serial number for a stream. Real-world muxers use random
/// 32-bit numbers; we use the stream index for determinism (which makes
/// remux output byte-stable when the input numbering is also dense from 0).
fn derive_serial(s: &StreamInfo) -> u32 {
    s.index
}

struct OggMuxer {
    output: Box<dyn WriteSeek>,
    /// Stream descriptors retained so write_header can reconstruct the
    /// codec-specific setup packets from each stream's extradata.
    streams: Vec<StreamInfo>,
    per_stream: HashMap<u32, StreamWriter>,
    stream_order: Vec<u32>,
    header_written: bool,
    trailer_written: bool,
}

struct StreamWriter {
    serial: u32,
    seq_no: u32,
    buffered: PageBuilder,
    headers_remaining: usize,
    bos_emitted: bool,
    /// Bytes of the most recently finalized page, held back until either
    /// another page is flushed (in which case it's written) or the trailer
    /// runs (in which case it gets EOS set and its CRC patched). This makes
    /// the EOS marker sit on a real data page instead of an empty trailing one.
    pending_bytes: Option<Vec<u8>>,
}

#[derive(Default)]
struct PageBuilder {
    /// Lacing values for the page so far (≤ 255 entries).
    lacing: Vec<u8>,
    /// Concatenated segment data for the page so far.
    data: Vec<u8>,
    /// First-segment-on-page is the continuation of an unfinished packet
    /// from the previous page.
    starts_continued: bool,
    /// Granule position to record on this page — set to the most recent
    /// completed packet's pts. -1 means "no packet ends here".
    granule_position: i64,
}

impl PageBuilder {
    fn new() -> Self {
        Self {
            granule_position: -1,
            ..Default::default()
        }
    }

    fn is_empty(&self) -> bool {
        self.lacing.is_empty()
    }
}

impl OggMuxer {
    fn writer_for(&mut self, stream_index: u32) -> Result<&mut StreamWriter> {
        self.per_stream
            .get_mut(&stream_index)
            .ok_or_else(|| Error::invalid(format!("unknown stream index {stream_index}")))
    }

    /// Finalize the buffered page for `stream_index`. The newly built page
    /// becomes the writer's *pending* page; whatever was previously pending
    /// gets written out to the underlying sink.
    fn flush_page(&mut self, stream_index: u32, force: bool) -> Result<()> {
        let writer = self
            .per_stream
            .get_mut(&stream_index)
            .ok_or_else(|| Error::invalid(format!("unknown stream index {stream_index}")))?;
        if writer.buffered.is_empty() && !force {
            return Ok(());
        }
        let mut page_flags = 0u8;
        if writer.buffered.starts_continued {
            page_flags |= flags::CONTINUED;
        }
        if !writer.bos_emitted {
            page_flags |= flags::FIRST_PAGE;
            writer.bos_emitted = true;
        }
        let page = Page {
            flags: page_flags,
            granule_position: writer.buffered.granule_position,
            serial: writer.serial,
            seq_no: writer.seq_no,
            lacing: std::mem::take(&mut writer.buffered.lacing),
            data: std::mem::take(&mut writer.buffered.data),
        };
        writer.seq_no = writer.seq_no.wrapping_add(1);
        writer.buffered.starts_continued =
            page.lacing.last().copied() == Some(255);
        writer.buffered.granule_position = -1;
        let new_bytes = page.to_bytes();

        // Write whatever was pending before, then queue the new bytes.
        if let Some(prev) = writer.pending_bytes.take() {
            self.output.write_all(&prev)?;
        }
        let writer = self.writer_for(stream_index)?;
        writer.pending_bytes = Some(new_bytes);
        Ok(())
    }
}

impl Muxer for OggMuxer {
    fn format_name(&self) -> &str {
        "ogg"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("Ogg muxer: write_header called twice"));
        }
        self.header_written = true;
        // Reconstruct codec setup packets from each stream's extradata and
        // emit them. Each is written via the normal write_packet path so the
        // header_packet bookkeeping (one packet per page, granule = 0, BOS
        // flag for the first) kicks in automatically.
        let stream_clone = self.streams.clone();
        for s in &stream_clone {
            for hp in extract_codec_headers(&s.params.codec_id, &s.params.extradata) {
                let pkt = Packet::new(s.index, s.time_base, hp);
                self.write_packet(&pkt)?;
            }
        }
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("Ogg muxer: write_header not called"));
        }
        let stream_index = packet.stream_index;
        let lacing_for_packet = lace(packet.data.len());

        let writer = self.writer_for(stream_index)?;
        let is_header = writer.headers_remaining > 0;

        // Flush early if this packet's lacing wouldn't fit in 255 segments.
        if writer.buffered.lacing.len() + lacing_for_packet.len() > 255 {
            self.flush_page(stream_index, false)?;
        }

        let writer = self.writer_for(stream_index)?;
        writer.buffered.lacing.extend_from_slice(&lacing_for_packet);
        writer.buffered.data.extend_from_slice(&packet.data);

        if is_header {
            // Header packets each get their own page with granule 0.
            writer.headers_remaining -= 1;
            writer.buffered.granule_position = 0;
            self.flush_page(stream_index, true)?;
            return Ok(());
        }

        // Audio/video packet. The page's granule_position is set from the
        // most recent pts seen on this page (this packet's pts wins if
        // present; otherwise the buffered value carries through). A new
        // page is flushed when the source signaled a page boundary via
        // `unit_boundary`. This separates *pts-per-packet* (decoders care)
        // from *page boundaries* (Ogg cares).
        if let Some(pts) = packet.pts {
            writer.buffered.granule_position = pts;
        }
        if packet.flags.unit_boundary {
            self.flush_page(stream_index, true)?;
        }

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        let order = self.stream_order.clone();
        for idx in order {
            // Drain any in-progress builder into pending_bytes.
            let needs_flush = {
                let writer = self.writer_for(idx)?;
                !writer.buffered.is_empty()
            };
            if needs_flush {
                self.flush_page(idx, true)?;
            }
            // Whatever's in pending_bytes is the truly last page — set EOS,
            // recompute its CRC, write it.
            let writer = self.writer_for(idx)?;
            if let Some(mut bytes) = writer.pending_bytes.take() {
                if bytes.len() >= 27 {
                    bytes[5] |= flags::LAST_PAGE;
                    // Zero out checksum field, recompute, patch back.
                    bytes[22..26].fill(0);
                    let crc = crate::crc::checksum(&bytes);
                    bytes[22..26].copy_from_slice(&crc.to_le_bytes());
                }
                self.output.write_all(&bytes)?;
            }
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

// Keep imports honest for downstream consumers.
#[allow(dead_code)]
const _SANITY: () = {
    let _ = page::CAPTURE_PATTERN;
};

/// Inverse of `oxideav_ogg::demux::build_codec_private`: turn a stream's
/// extradata back into the per-codec sequence of header packets that an Ogg
/// stream needs at its start.
fn extract_codec_headers(codec_id: &CodecId, extradata: &[u8]) -> Vec<Vec<u8>> {
    if extradata.is_empty() {
        return Vec::new();
    }
    match codec_id.as_str() {
        "vorbis" => parse_xiph_lacing(extradata).unwrap_or_default(),
        "opus" => {
            // OpusHead followed by a synthetic minimal OpusTags. (Original
            // tags are dropped during demux — they're not load-bearing.)
            let head = extradata.to_vec();
            let mut tags = Vec::with_capacity(20);
            tags.extend_from_slice(b"OpusTags");
            tags.extend_from_slice(&0u32.to_le_bytes()); // vendor string length = 0
            tags.extend_from_slice(&0u32.to_le_bytes()); // user comment count = 0
            vec![head, tags]
        }
        _ => vec![extradata.to_vec()],
    }
}

/// Parse a Xiph-laced 3-packet header blob (Vorbis/Theora layout). The first
/// byte is `(packet_count - 1)`, followed by `(packet_count - 1)` lacing
/// records (each a series of 0xFF terminators ending in a value < 0xFF).
fn parse_xiph_lacing(buf: &[u8]) -> Option<Vec<Vec<u8>>> {
    if buf.is_empty() {
        return None;
    }
    let n_packets = buf[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n_packets);
    let mut i = 1usize;
    for _ in 0..n_packets - 1 {
        let mut s = 0usize;
        loop {
            if i >= buf.len() {
                return None;
            }
            let b = buf[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    let used: usize = sizes.iter().sum();
    if i + used > buf.len() {
        return None;
    }
    let last_size = buf.len() - i - used;
    sizes.push(last_size);
    let mut packets = Vec::with_capacity(n_packets);
    for sz in sizes {
        if i + sz > buf.len() {
            return None;
        }
        packets.push(buf[i..i + sz].to_vec());
        i += sz;
    }
    Some(packets)
}
