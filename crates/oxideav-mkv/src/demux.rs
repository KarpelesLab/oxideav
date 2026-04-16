//! Matroska demuxer.
//!
//! Strategy: read the EBML header, locate the Segment, parse Info + Tracks
//! up front. Then on each `next_packet` call, walk Cluster children one at a
//! time, extracting frames from `SimpleBlock` and `BlockGroup → Block`
//! elements (lacing-aware).

use std::io::{Read, Seek, SeekFrom};

use oxideav_container::{Demuxer, ReadSeek};
use oxideav_core::{
    CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::codec_id::{from_matroska, strip_bitmapinfoheader};
use crate::ebml::{
    read_bytes, read_element_header, read_float, read_string, read_uint, skip, VINT_UNKNOWN_SIZE,
};
use crate::ids;

pub fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    // Validate EBML header.
    let hdr = read_element_header(&mut *input)?;
    if hdr.id != ids::EBML_HEADER {
        return Err(Error::invalid(format!(
            "MKV: expected EBML header at start, got id 0x{:X}",
            hdr.id
        )));
    }
    let mut doc_type = String::from("matroska");
    let ebml_end = input.stream_position()? + hdr.size;
    while input.stream_position()? < ebml_end {
        let e = read_element_header(&mut *input)?;
        match e.id {
            ids::EBML_DOC_TYPE => {
                doc_type = read_string(&mut *input, e.size as usize)?;
            }
            _ => skip(&mut *input, e.size)?,
        }
    }
    if doc_type != "matroska" && doc_type != "webm" {
        return Err(Error::unsupported(format!(
            "MKV: unsupported DocType '{doc_type}'"
        )));
    }

    // Find Segment.
    let seg = read_element_header(&mut *input)?;
    if seg.id != ids::SEGMENT {
        return Err(Error::invalid(format!(
            "MKV: expected Segment after EBML header, got id 0x{:X}",
            seg.id
        )));
    }
    let segment_data_start = input.stream_position()?;
    let segment_data_end = if seg.size == VINT_UNKNOWN_SIZE {
        // Unknown segment size — use file end.
        let cur = input.stream_position()?;
        let end = input.seek(SeekFrom::End(0))?;
        input.seek(SeekFrom::Start(cur))?;
        end
    } else {
        segment_data_start + seg.size
    };

    // Walk segment children, recording where Tracks/Info/Cluster live.
    let mut info = SegmentInfo::default();
    let mut tracks: Vec<TrackEntry> = Vec::new();
    let mut first_cluster_offset: Option<u64> = None;

    while input.stream_position()? < segment_data_end {
        let e = read_element_header(&mut *input)?;
        let body_start = input.stream_position()?;
        let body_end_known = if e.size == VINT_UNKNOWN_SIZE {
            None
        } else {
            Some(body_start + e.size)
        };
        match e.id {
            ids::INFO => {
                let end = body_end_known.unwrap_or(segment_data_end);
                parse_info(&mut *input, end, &mut info)?;
            }
            ids::TRACKS => {
                let end = body_end_known.unwrap_or(segment_data_end);
                parse_tracks(&mut *input, end, &mut tracks)?;
            }
            ids::CLUSTER => {
                if first_cluster_offset.is_none() {
                    // Position the cluster pointer at the Cluster's own header,
                    // not its body — the demuxer state machine re-reads the
                    // header to discover children.
                    first_cluster_offset = Some(body_start - e.header_len as u64);
                }
                // Stop scanning here; per Matroska conventions Tracks/Info come
                // before Clusters, so we have what we need.
                input.seek(SeekFrom::Start(body_start - e.header_len as u64))?;
                break;
            }
            // Skip everything else: SeekHead, Cues, Attachments, Tags, etc.
            _ => {
                if let Some(end) = body_end_known {
                    input.seek(SeekFrom::Start(end))?;
                } else {
                    return Err(Error::unsupported(
                        "MKV: unknown-size element other than Cluster",
                    ));
                }
            }
        }
    }

    if tracks.is_empty() {
        return Err(Error::invalid("MKV: no tracks found"));
    }

    // Use 1ms timebase if not specified (default Matroska timecode_scale = 1_000_000 ns).
    let timecode_scale_ns = if info.timecode_scale == 0 {
        1_000_000
    } else {
        info.timecode_scale
    };
    // For simplicity expose every stream with the segment time base = scale/1e9 seconds per tick.
    // 1 tick = timecode_scale_ns nanoseconds. So time base = timecode_scale_ns / 1_000_000_000.
    let time_base = TimeBase::new(timecode_scale_ns as i64, 1_000_000_000);

    // Build public StreamInfo list, preserving the input track-number → output index mapping.
    let mut streams: Vec<StreamInfo> = Vec::new();
    let mut track_index_by_number: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::new();
    for t in &tracks {
        let idx = streams.len() as u32;
        track_index_by_number.insert(t.number, idx);
        let codec_id = from_matroska(&t.codec_id_string, &t.codec_private);
        let mut params = match t.track_type {
            ids::TRACK_TYPE_VIDEO => CodecParameters::video(codec_id.clone()),
            ids::TRACK_TYPE_AUDIO => CodecParameters::audio(codec_id.clone()),
            _ => {
                let mut p = CodecParameters::audio(codec_id.clone());
                p.media_type = MediaType::Data;
                p
            }
        };
        // Codec-specific CodecPrivate normalisation:
        //   * `V_MS/VFW/FOURCC`: the outer 40-byte BITMAPINFOHEADER wraps
        //     real codec extradata — strip it so decoders see their own
        //     config record.
        //   * `A_FLAC`: the CodecPrivate sometimes has a leading `"fLaC"`
        //     magic; our FLAC decoder expects metadata blocks only.
        let stripped = strip_bitmapinfoheader(&t.codec_id_string, &t.codec_private);
        params.extradata = match codec_id.as_str() {
            "flac" if stripped.starts_with(b"fLaC") => stripped[4..].to_vec(),
            _ => stripped,
        };
        if t.track_type == ids::TRACK_TYPE_AUDIO {
            params.sample_rate = Some(t.sample_rate.round() as u32);
            params.channels = Some(t.channels as u16);
            params.sample_format = match (params.codec_id.as_str(), t.bit_depth) {
                ("pcm_s16le", _) => Some(SampleFormat::S16),
                ("pcm_s16be", _) => Some(SampleFormat::S16),
                ("pcm_f32le", _) => Some(SampleFormat::F32),
                ("flac", 8) => Some(SampleFormat::U8),
                ("flac", 16) => Some(SampleFormat::S16),
                ("flac", 24) => Some(SampleFormat::S24),
                ("flac", 32) => Some(SampleFormat::S32),
                _ => None,
            };
        }
        if t.track_type == ids::TRACK_TYPE_VIDEO {
            params.width = Some(t.width as u32);
            params.height = Some(t.height as u32);
        }
        streams.push(StreamInfo {
            index: idx,
            time_base,
            duration: if info.duration > 0.0 {
                Some(info.duration as i64)
            } else {
                None
            },
            start_time: Some(0),
            params,
        });
    }

    // Position at the first Cluster.
    let cluster_pos = first_cluster_offset.ok_or_else(|| Error::invalid("MKV: no clusters"))?;
    input.seek(SeekFrom::Start(cluster_pos))?;

    Ok(Box::new(MkvDemuxer {
        input,
        streams,
        track_index_by_number,
        segment_data_end,
        cluster_state: ClusterState::Idle,
        out_queue: std::collections::VecDeque::new(),
        time_base,
    }))
}

#[derive(Default)]
struct SegmentInfo {
    timecode_scale: u64,
    duration: f64,
}

#[derive(Default)]
struct TrackEntry {
    number: u64,
    track_type: u64,
    codec_id_string: String,
    codec_private: Vec<u8>,
    sample_rate: f64,
    channels: u64,
    bit_depth: u64,
    width: u64,
    height: u64,
}

fn parse_info(r: &mut dyn ReadSeek, end: u64, out: &mut SegmentInfo) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TIMECODE_SCALE => out.timecode_scale = read_uint(r, e.size as usize)?,
            ids::DURATION => out.duration = read_float(r, e.size as usize)?,
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_tracks(r: &mut dyn ReadSeek, end: u64, out: &mut Vec<TrackEntry>) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TRACK_ENTRY => {
                let body_end = r.stream_position()? + e.size;
                let mut t = TrackEntry::default();
                parse_track_entry(r, body_end, &mut t)?;
                out.push(t);
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_track_entry(r: &mut dyn ReadSeek, end: u64, t: &mut TrackEntry) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TRACK_NUMBER => t.number = read_uint(r, e.size as usize)?,
            ids::TRACK_TYPE => t.track_type = read_uint(r, e.size as usize)?,
            ids::CODEC_ID => t.codec_id_string = read_string(r, e.size as usize)?,
            ids::CODEC_PRIVATE => t.codec_private = read_bytes(r, e.size as usize)?,
            ids::AUDIO => {
                let body_end = r.stream_position()? + e.size;
                parse_audio(r, body_end, t)?;
            }
            ids::VIDEO => {
                let body_end = r.stream_position()? + e.size;
                parse_video(r, body_end, t)?;
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_audio(r: &mut dyn ReadSeek, end: u64, t: &mut TrackEntry) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::SAMPLING_FREQUENCY => t.sample_rate = read_float(r, e.size as usize)?,
            ids::CHANNELS => t.channels = read_uint(r, e.size as usize)?,
            ids::BIT_DEPTH => t.bit_depth = read_uint(r, e.size as usize)?,
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_video(r: &mut dyn ReadSeek, end: u64, t: &mut TrackEntry) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::PIXEL_WIDTH => t.width = read_uint(r, e.size as usize)?,
            ids::PIXEL_HEIGHT => t.height = read_uint(r, e.size as usize)?,
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

// --- Demuxer state machine ------------------------------------------------

enum ClusterState {
    /// Not inside a cluster; the next read must start with a Cluster header.
    Idle,
    /// Inside a Cluster, reading children. `body_end` is where the cluster ends.
    InCluster {
        body_end: u64,
        cluster_timecode: i64,
    },
}

struct MkvDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    track_index_by_number: std::collections::HashMap<u64, u32>,
    segment_data_end: u64,
    cluster_state: ClusterState,
    out_queue: std::collections::VecDeque<Packet>,
    time_base: TimeBase,
}

impl Demuxer for MkvDemuxer {
    fn format_name(&self) -> &str {
        "matroska"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        loop {
            if let Some(p) = self.out_queue.pop_front() {
                return Ok(p);
            }
            self.advance()?;
        }
    }
}

impl MkvDemuxer {
    fn advance(&mut self) -> Result<()> {
        match self.cluster_state {
            ClusterState::Idle => {
                let pos = self.input.stream_position()?;
                if pos >= self.segment_data_end {
                    return Err(Error::Eof);
                }
                let e = read_element_header(&mut *self.input)?;
                match e.id {
                    ids::CLUSTER => {
                        let body_start = self.input.stream_position()?;
                        let body_end = if e.size == VINT_UNKNOWN_SIZE {
                            self.segment_data_end
                        } else {
                            body_start + e.size
                        };
                        self.cluster_state = ClusterState::InCluster {
                            body_end,
                            cluster_timecode: 0,
                        };
                        Ok(())
                    }
                    ids::CUES | ids::ATTACHMENTS | ids::CHAPTERS | ids::TAGS => {
                        // Skip — not packet data.
                        skip(&mut *self.input, e.size)?;
                        Ok(())
                    }
                    _ => {
                        // Unknown element at top level — skip it.
                        skip(&mut *self.input, e.size)?;
                        Ok(())
                    }
                }
            }
            ClusterState::InCluster {
                body_end,
                cluster_timecode,
            } => {
                let pos = self.input.stream_position()?;
                if pos >= body_end {
                    self.cluster_state = ClusterState::Idle;
                    return Ok(());
                }
                let e = read_element_header(&mut *self.input)?;
                match e.id {
                    ids::TIMECODE => {
                        let v = read_uint(&mut *self.input, e.size as usize)? as i64;
                        if let ClusterState::InCluster {
                            ref mut cluster_timecode,
                            ..
                        } = self.cluster_state
                        {
                            *cluster_timecode = v;
                        }
                    }
                    ids::SIMPLE_BLOCK => {
                        let bytes = read_bytes(&mut *self.input, e.size as usize)?;
                        self.queue_block_packets(&bytes, cluster_timecode, false)?;
                    }
                    ids::BLOCK_GROUP => {
                        let bg_end = self.input.stream_position()? + e.size;
                        self.parse_block_group(bg_end, cluster_timecode)?;
                    }
                    _ => skip(&mut *self.input, e.size)?,
                }
                Ok(())
            }
        }
    }

    fn parse_block_group(&mut self, end: u64, cluster_timecode: i64) -> Result<()> {
        let mut block_bytes: Option<Vec<u8>> = None;
        let mut duration: Option<i64> = None;
        let mut is_keyframe = true;
        while self.input.stream_position()? < end {
            let e = read_element_header(&mut *self.input)?;
            match e.id {
                ids::BLOCK => {
                    block_bytes = Some(read_bytes(&mut *self.input, e.size as usize)?);
                }
                ids::BLOCK_DURATION => {
                    duration = Some(read_uint(&mut *self.input, e.size as usize)? as i64);
                }
                ids::REFERENCE_BLOCK => {
                    is_keyframe = false;
                    skip(&mut *self.input, e.size)?;
                }
                _ => skip(&mut *self.input, e.size)?,
            }
        }
        if let Some(b) = block_bytes {
            // For BlockGroup, the lacing flags are in the same place as
            // SimpleBlock (the "keyframe" bit doesn't exist in plain Block —
            // keyframe-ness is inferred from absence of ReferenceBlock).
            self.queue_block_packets_with(&b, cluster_timecode, is_keyframe, duration)?;
        }
        Ok(())
    }

    fn queue_block_packets(
        &mut self,
        bytes: &[u8],
        cluster_timecode: i64,
        _hint: bool,
    ) -> Result<()> {
        // SimpleBlock: keyframe bit is bit 7 of flags byte.
        // BlockGroup/Block has the same layout but no keyframe bit.
        // We pass through whatever's set in the flags byte for SimpleBlock.
        self.queue_block_packets_with(bytes, cluster_timecode, true, None)
    }

    fn queue_block_packets_with(
        &mut self,
        bytes: &[u8],
        cluster_timecode: i64,
        default_keyframe: bool,
        explicit_duration: Option<i64>,
    ) -> Result<()> {
        let mut cur = std::io::Cursor::new(bytes);
        let (track_number, _) = crate::ebml::read_vint(&mut cur, false)?;
        let mut tc_buf = [0u8; 2];
        cur.read_exact(&mut tc_buf)?;
        let timecode_offset = i16::from_be_bytes(tc_buf) as i64;
        let mut flags_buf = [0u8; 1];
        cur.read_exact(&mut flags_buf)?;
        let flags = flags_buf[0];
        let lacing = (flags >> 1) & 0x03;
        let keyframe_flag = flags & 0x80 != 0;

        let stream_idx = match self.track_index_by_number.get(&track_number) {
            Some(i) => *i,
            None => return Ok(()), // Skip frames for unknown tracks.
        };

        // Frame data starts at current cur position.
        let body_start = cur.position() as usize;
        let body = &bytes[body_start..];

        let frames = match lacing {
            0 => vec![body.to_vec()],
            1 => parse_xiph_lacing(body)?,
            2 => parse_fixed_lacing(body)?,
            3 => parse_ebml_lacing(body)?,
            _ => unreachable!(),
        };

        let pts_base = cluster_timecode + timecode_offset;
        let n_frames = frames.len() as i64;
        let per_frame = explicit_duration.map(|d| d / n_frames.max(1));
        for (i, f) in frames.into_iter().enumerate() {
            let pts = pts_base + per_frame.unwrap_or(0) * i as i64;
            let mut pkt = Packet::new(stream_idx, self.time_base, f);
            pkt.pts = Some(pts);
            pkt.dts = Some(pts);
            pkt.duration = per_frame;
            pkt.flags.keyframe = keyframe_flag || default_keyframe;
            self.out_queue.push_back(pkt);
        }
        Ok(())
    }
}

// --- Lacing helpers --------------------------------------------------------

fn parse_xiph_lacing(body: &[u8]) -> Result<Vec<Vec<u8>>> {
    if body.is_empty() {
        return Ok(vec![]);
    }
    let n_frames = body[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n_frames);
    let mut i = 1;
    for _ in 0..n_frames - 1 {
        let mut s = 0usize;
        loop {
            if i >= body.len() {
                return Err(Error::invalid("MKV xiph lacing: truncated size"));
            }
            let b = body[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    // Last frame size is whatever's left.
    let used: usize = sizes.iter().sum();
    let last_size = body.len() - i - used;
    sizes.push(last_size);
    let mut frames = Vec::with_capacity(n_frames);
    for s in sizes {
        if i + s > body.len() {
            return Err(Error::invalid("MKV xiph lacing: frame exceeds body"));
        }
        frames.push(body[i..i + s].to_vec());
        i += s;
    }
    Ok(frames)
}

fn parse_fixed_lacing(body: &[u8]) -> Result<Vec<Vec<u8>>> {
    if body.is_empty() {
        return Ok(vec![]);
    }
    let n_frames = body[0] as usize + 1;
    let payload = &body[1..];
    if payload.len() % n_frames != 0 {
        return Err(Error::invalid("MKV fixed lacing: non-divisible payload"));
    }
    let frame_size = payload.len() / n_frames;
    let mut frames = Vec::with_capacity(n_frames);
    for c in payload.chunks_exact(frame_size) {
        frames.push(c.to_vec());
    }
    Ok(frames)
}

fn parse_ebml_lacing(body: &[u8]) -> Result<Vec<Vec<u8>>> {
    if body.is_empty() {
        return Ok(vec![]);
    }
    let mut cur = std::io::Cursor::new(body);
    let n_frames = {
        let mut buf = [0u8; 1];
        cur.read_exact(&mut buf)?;
        buf[0] as usize + 1
    };
    let mut sizes = Vec::with_capacity(n_frames);
    // First size: full VINT.
    let (first, _) = crate::ebml::read_vint(&mut cur, false)?;
    sizes.push(first as i64);
    // Remaining sizes: signed deltas (raw VINT minus mid-of-range bias).
    for _ in 0..n_frames - 2 {
        let (raw, w) = crate::ebml::read_vint(&mut cur, false)?;
        let bias = ((1i64) << (7 * w as i64 - 1)) - 1;
        let signed = (raw as i64) - bias;
        let prev = *sizes.last().unwrap();
        sizes.push(prev + signed);
    }
    // Last frame is whatever remains.
    let pos = cur.position() as usize;
    let used: i64 = sizes.iter().sum();
    let last = body.len() as i64 - pos as i64 - used;
    sizes.push(last);
    let mut frames = Vec::with_capacity(n_frames);
    let mut i = pos;
    for s in sizes {
        if s < 0 || i + s as usize > body.len() {
            return Err(Error::invalid("MKV ebml lacing: invalid frame size"));
        }
        frames.push(body[i..i + s as usize].to_vec());
        i += s as usize;
    }
    Ok(frames)
}
