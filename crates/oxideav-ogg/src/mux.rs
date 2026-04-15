//! Ogg muxer: pack incoming packets into pages.
//!
//! Strategy: maintain one buffered page per logical stream. Pack a packet by
//! appending its bytes and lacing values. Flush the page whenever it reaches
//! the 255-segment limit, when an explicit flush is requested, or at trailer
//! time. Granule positions come from `Packet::pts` for non-header packets.

use std::collections::HashMap;
use std::io::Write;

use oxideav_container::{Muxer, WriteSeek};
use oxideav_core::{Error, Packet, Result, StreamInfo};

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
                eos_pending: false,
            },
        );
    }
    Ok(Box::new(OggMuxer {
        output,
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
    eos_pending: bool,
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

    /// Flush the buffered page for `stream_index` to the output.
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
        if writer.eos_pending {
            page_flags |= flags::LAST_PAGE;
            writer.eos_pending = false;
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
        // Whether the next page begins with a continuation depends on whether
        // the just-flushed page ended in a 255-byte segment (= unfinished
        // packet). PageBuilder tracks this for us via the lacing values we
        // appended in `write_packet`.
        writer.buffered.starts_continued =
            page.lacing.last().copied() == Some(255);
        writer.buffered.granule_position = -1;
        let bytes = page.to_bytes();
        self.output.write_all(&bytes)?;
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
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("Ogg muxer: write_header not called"));
        }
        let stream_index = packet.stream_index;
        let lacing_for_packet = lace(packet.data.len());

        // Header packets each go into their own page (per Vorbis/Opus mapping
        // conventions: BOS page carries identification by itself; remaining
        // header packets are flushed before the first audio packet).
        let writer = self.writer_for(stream_index)?;
        let is_header = writer.headers_remaining > 0;

        // If adding this packet would exceed 255 segments on the current page,
        // flush first.
        if writer.buffered.lacing.len() + lacing_for_packet.len() > 255 {
            self.flush_page(stream_index, false)?;
        }

        // Append packet to buffered page.
        let writer = self.writer_for(stream_index)?;
        writer.buffered.lacing.extend_from_slice(&lacing_for_packet);
        writer.buffered.data.extend_from_slice(&packet.data);
        // Granule position of a page is the granule of the *last* packet
        // ending on that page. Header packets contribute granule = 0.
        let granule_for_this_packet = if is_header {
            0
        } else {
            packet.pts.unwrap_or(writer.buffered.granule_position.max(0))
        };
        writer.buffered.granule_position = granule_for_this_packet;

        if is_header {
            writer.headers_remaining -= 1;
            // Force one packet per header page.
            self.flush_page(stream_index, true)?;
        }

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Flush remaining data and tag the final page of each stream as EOS.
        let order = self.stream_order.clone();
        for idx in order {
            let writer = self
                .per_stream
                .get_mut(&idx)
                .ok_or_else(|| Error::invalid(format!("unknown stream index {idx}")))?;
            writer.eos_pending = true;
            self.flush_page(idx, true)?;
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

// Keep imports honest for downstream consumers.
#[allow(dead_code)]
const _SANITY: () = {
    // Reference page module so it stays linked even if otherwise unused here.
    let _ = page::CAPTURE_PATTERN;
};
