//! Ogg demuxer: page reader → per-stream packet reassembly.

use std::collections::HashMap;
use std::io::Read;

use oxideav_container::{Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, StreamInfo, TimeBase,
};

use crate::codec_id;
use crate::page::{self, Page};

/// Open an Ogg bitstream.
pub fn open(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut state = OggDemuxer::new(input);
    state.read_bos_section()?;
    state.read_until_headers_collected()?;
    state.populate_extradata();
    Ok(Box::new(state))
}

struct LogicalStream {
    /// Index into the public `streams` vec.
    public_index: usize,
    /// Buffered partial-packet bytes from a previous page that ended without
    /// a terminator (lacing 255). Concatenated with the next page's leading
    /// segments to form a complete packet.
    pending: Vec<u8>,
    /// Number of header packets still to be absorbed (not delivered).
    headers_remaining: usize,
    /// Header packets accumulated so far — used to populate codec-specific
    /// extradata on the stream's `CodecParameters` once they're all in.
    header_packets: Vec<Vec<u8>>,
    granule_seen: i64,
}

struct OggDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    state_by_serial: HashMap<u32, LogicalStream>,
    /// Pages we've already read but not yet drained for packets.
    page_queue: std::collections::VecDeque<Page>,
    /// Packets ready to emit, in insertion order across all streams.
    out_queue: std::collections::VecDeque<Packet>,
    /// True once we've read past the BOS section and into the data pages.
    eof_reached: bool,
}

impl OggDemuxer {
    fn new(input: Box<dyn ReadSeek>) -> Self {
        Self {
            input,
            streams: Vec::new(),
            state_by_serial: HashMap::new(),
            page_queue: std::collections::VecDeque::new(),
            out_queue: std::collections::VecDeque::new(),
            eof_reached: false,
        }
    }

    /// Read pages until we leave the Beginning-Of-Stream section, registering
    /// every logical bitstream we discover. The pages we read are queued so
    /// `next_packet` can drain them in order.
    fn read_bos_section(&mut self) -> Result<()> {
        loop {
            let page = match self.read_page()? {
                Some(p) => p,
                None => {
                    self.eof_reached = true;
                    break;
                }
            };
            let is_bos = page.is_first();
            if is_bos {
                self.register_stream(&page)?;
            }
            self.page_queue.push_back(page);
            if !is_bos {
                // The first non-BOS page marks the end of the BOS section.
                break;
            }
        }
        if self.streams.is_empty() {
            return Err(Error::invalid("Ogg file contains no logical streams"));
        }
        Ok(())
    }

    fn register_stream(&mut self, bos_page: &Page) -> Result<()> {
        // The BOS page's first packet is the identification packet for the
        // codec. Identification packets must fit in a single BOS page (RFC
        // 5334 / codec mapping conventions).
        let segs = bos_page.packet_segments();
        if segs.is_empty() {
            return Err(Error::invalid("Ogg BOS page has no packets"));
        }
        let first = &bos_page.data[segs[0].data.clone()];
        let codec_id = codec_id::detect(first);
        let public_index = self.streams.len();
        let mut params = guess_params(&codec_id, first)?;
        params.extradata = first.to_vec();

        let time_base = match codec_id.as_str() {
            "vorbis" | "flac" => {
                if let Some(sr) = params.sample_rate {
                    TimeBase::new(1, sr as i64)
                } else {
                    TimeBase::new(1, 1_000_000)
                }
            }
            // Opus uses a 48 kHz timebase regardless of input sample rate.
            "opus" => TimeBase::new(1, 48_000),
            _ => TimeBase::new(1, 1_000_000),
        };

        self.streams.push(StreamInfo {
            index: public_index as u32,
            time_base,
            duration: None,
            start_time: Some(0),
            params,
        });
        self.state_by_serial.insert(
            bos_page.serial,
            LogicalStream {
                public_index,
                pending: Vec::new(),
                headers_remaining: codec_id::header_packet_count(&codec_id),
                header_packets: Vec::new(),
                granule_seen: 0,
            },
        );
        Ok(())
    }

    fn read_page(&mut self) -> Result<Option<Page>> {
        // Read a page header (27 bytes), then enough to read the segment table
        // and data. We detect EOF by getting 0 bytes back from the very first
        // read; partial-page data is treated as truncation.
        let mut hdr = [0u8; 27];
        if !read_exact_or_eof(&mut self.input, &mut hdr)? {
            return Ok(None);
        }
        if &hdr[0..4] != &page::CAPTURE_PATTERN {
            return Err(Error::invalid("Ogg: lost page sync (no 'OggS')"));
        }
        let n_segs = hdr[26] as usize;
        let mut lacing = vec![0u8; n_segs];
        self.input.read_exact(&mut lacing)?;
        let data_len: usize = lacing.iter().map(|&v| v as usize).sum();
        let mut data = vec![0u8; data_len];
        self.input.read_exact(&mut data)?;

        // Re-parse from the assembled bytes so CRC validation logic is shared.
        let mut full = Vec::with_capacity(27 + n_segs + data_len);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&lacing);
        full.extend_from_slice(&data);
        let (page, consumed) = Page::parse(&full)?;
        debug_assert_eq!(consumed, full.len());
        Ok(Some(page))
    }

    /// After the BOS section, keep reading pages and absorbing header packets
    /// until every logical stream has gathered all of its expected setup
    /// packets (3 for Vorbis, 2 for Opus, …). Audio/video packets read in the
    /// process are still queued; they'll be delivered by `next_packet` later.
    fn read_until_headers_collected(&mut self) -> Result<()> {
        loop {
            let any_pending = self
                .state_by_serial
                .values()
                .any(|s| s.headers_remaining > 0);
            if !any_pending {
                return Ok(());
            }
            // Drain queued pages from the BOS phase first; only then read more.
            let page = if let Some(p) = self.page_queue.pop_front() {
                p
            } else {
                match self.read_page()? {
                    Some(p) => p,
                    None => return Ok(()), // EOF before all headers — best-effort.
                }
            };
            self.process_page(page)?;
        }
    }

    /// Build codec-specific extradata for each stream from its accumulated
    /// header packets and write it back to the stream's `CodecParameters`.
    fn populate_extradata(&mut self) {
        for state in self.state_by_serial.values() {
            let codec_id = self.streams[state.public_index].params.codec_id.clone();
            let extra = build_codec_private(&codec_id, &state.header_packets);
            if !extra.is_empty() {
                self.streams[state.public_index].params.extradata = extra;
            }
        }
    }

    /// Drain the next packet from the queued pages, possibly reading more.
    fn drain_next(&mut self) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = self.out_queue.pop_front() {
                return Ok(Some(p));
            }
            // Need to consume another page.
            let page = match self.page_queue.pop_front() {
                Some(p) => p,
                None => match self.read_page()? {
                    Some(p) => p,
                    None => {
                        self.eof_reached = true;
                        return Ok(None);
                    }
                },
            };
            self.process_page(page)?;
        }
    }

    fn process_page(&mut self, page: Page) -> Result<()> {
        let Some(stream) = self.state_by_serial.get_mut(&page.serial) else {
            // Unknown serial — skip silently.
            return Ok(());
        };
        let public_index = stream.public_index;
        let stream_idx = self.streams[public_index].index;
        let time_base = self.streams[public_index].time_base;
        let segs = page.packet_segments();
        let was_continued = page.is_continued();

        // Collect every packet that terminates on this page; the page's
        // granule_position applies to the last such packet (per RFC 3533).
        let mut completed: Vec<Vec<u8>> = Vec::new();
        for (i, seg) in segs.iter().enumerate() {
            let payload = &page.data[seg.data.clone()];
            if i == 0 && was_continued {
                stream.pending.extend_from_slice(payload);
            } else {
                if !stream.pending.is_empty() {
                    stream.pending.clear(); // defensive
                }
                stream.pending.extend_from_slice(payload);
            }
            if seg.terminated {
                completed.push(std::mem::take(&mut stream.pending));
            }
        }

        let last_idx = completed.len().checked_sub(1);
        for (i, data) in completed.into_iter().enumerate() {
            if stream.headers_remaining > 0 {
                stream.header_packets.push(data);
                stream.headers_remaining -= 1;
                continue;
            }
            let is_last = Some(i) == last_idx;
            // pts on the last-on-page packet carries the page's granule
            // (Ogg's only timing signal); intermediate packets get None.
            // Container-aware muxers that need per-packet pts should derive
            // them from codec-specific knowledge (e.g. Opus TOC parsing).
            let pts = if is_last && page.granule_position >= 0 {
                Some(page.granule_position)
            } else {
                None
            };
            let mut pkt = Packet::new(stream_idx, time_base, data);
            pkt.pts = pts;
            pkt.dts = pts;
            pkt.flags.keyframe = true;
            pkt.flags.unit_boundary = is_last;
            self.out_queue.push_back(pkt);
        }

        // Track the most recently observed granule for debugging/analysis. Not
        // used to assign per-packet pts any more.
        if page.granule_position >= 0 {
            stream.granule_seen = page.granule_position;
        }
        Ok(())
    }
}

impl Demuxer for OggDemuxer {
    fn format_name(&self) -> &str {
        "ogg"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.drain_next()? {
            return Ok(p);
        }
        Err(Error::Eof)
    }
}

/// Build initial codec parameters from a known identification packet.
fn guess_params(codec_id: &CodecId, first: &[u8]) -> Result<CodecParameters> {
    let mut p = match codec_id.as_str() {
        "vorbis" => CodecParameters::audio(codec_id.clone()),
        "opus" => CodecParameters::audio(codec_id.clone()),
        "flac" => CodecParameters::audio(codec_id.clone()),
        "theora" => CodecParameters::video(codec_id.clone()),
        "speex" => CodecParameters::audio(codec_id.clone()),
        _ => {
            let mut p = CodecParameters::audio(codec_id.clone());
            p.media_type = MediaType::Unknown;
            p
        }
    };

    match codec_id.as_str() {
        "vorbis" => parse_vorbis_id(&mut p, first)?,
        "opus" => parse_opus_id(&mut p, first)?,
        _ => {}
    }
    Ok(p)
}

fn parse_vorbis_id(p: &mut CodecParameters, packet: &[u8]) -> Result<()> {
    if packet.len() < 30 {
        return Err(Error::invalid("Vorbis identification header too short"));
    }
    // packet[0]=0x01, packet[1..7]="vorbis", packet[7..11]=version (must be 0).
    let version = u32::from_le_bytes([packet[7], packet[8], packet[9], packet[10]]);
    if version != 0 {
        return Err(Error::unsupported(format!(
            "unsupported Vorbis version {version}"
        )));
    }
    let channels = packet[11];
    let sample_rate =
        u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
    let _br_max = i32::from_le_bytes([packet[16], packet[17], packet[18], packet[19]]);
    let br_nom = i32::from_le_bytes([packet[20], packet[21], packet[22], packet[23]]);
    let _br_min = i32::from_le_bytes([packet[24], packet[25], packet[26], packet[27]]);
    if channels == 0 || sample_rate == 0 {
        return Err(Error::invalid("Vorbis ID header has zero channels or rate"));
    }
    p.channels = Some(channels as u16);
    p.sample_rate = Some(sample_rate);
    if br_nom > 0 {
        p.bit_rate = Some(br_nom as u64);
    }
    Ok(())
}

fn parse_opus_id(p: &mut CodecParameters, packet: &[u8]) -> Result<()> {
    if packet.len() < 19 {
        return Err(Error::invalid("Opus identification header too short"));
    }
    let channels = packet[9];
    let input_rate =
        u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
    p.channels = Some(channels as u16);
    // Opus always decodes to 48 kHz; "input_sample_rate" is informational.
    p.sample_rate = Some(if input_rate > 0 { input_rate } else { 48_000 });
    Ok(())
}

/// Build the per-codec setup blob ("CodecPrivate" in Matroska, "esds"-equivalent
/// in MP4, etc.) from the header packets gathered out of an Ogg stream.
///
/// - Vorbis: Xiph-laced concatenation of all 3 header packets (id, comment, setup).
/// - Opus: just the OpusHead identification packet (OpusTags discarded).
/// - Anything else: concatenate the headers and let the codec sort it out.
fn build_codec_private(codec_id: &CodecId, packets: &[Vec<u8>]) -> Vec<u8> {
    match codec_id.as_str() {
        "vorbis" if packets.len() == 3 => {
            let mut out = Vec::with_capacity(
                1 + packets[0].len() / 255 + 1 + packets[1].len() / 255 + 1
                    + packets.iter().map(|p| p.len()).sum::<usize>(),
            );
            out.push(0x02); // 3 packets — 1
            out.extend(xiph_lace_size(packets[0].len()));
            out.extend(xiph_lace_size(packets[1].len()));
            out.extend_from_slice(&packets[0]);
            out.extend_from_slice(&packets[1]);
            out.extend_from_slice(&packets[2]);
            out
        }
        "opus" => packets.first().cloned().unwrap_or_default(),
        _ => packets.iter().flatten().copied().collect(),
    }
}

fn xiph_lace_size(mut n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n / 255 + 1);
    while n >= 255 {
        v.push(255);
        n -= 255;
    }
    v.push(n as u8);
    v
}

fn read_exact_or_eof(r: &mut dyn Read, buf: &mut [u8]) -> Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => {
                return if got == 0 {
                    Ok(false)
                } else {
                    Err(Error::invalid(format!(
                        "Ogg: truncated read ({}/{} bytes)",
                        got,
                        buf.len()
                    )))
                };
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}
