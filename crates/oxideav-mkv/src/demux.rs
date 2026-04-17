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
    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut cues: Vec<CueEntry> = Vec::new();

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
                parse_info(&mut *input, end, &mut info, &mut metadata)?;
            }
            ids::TRACKS => {
                let end = body_end_known.unwrap_or(segment_data_end);
                parse_tracks(&mut *input, end, &mut tracks)?;
            }
            ids::TAGS => {
                let end = body_end_known.unwrap_or(segment_data_end);
                parse_tags(&mut *input, end, &mut metadata)?;
            }
            ids::CUES => {
                let end = body_end_known.unwrap_or(segment_data_end);
                parse_cues(&mut *input, end, &mut cues)?;
            }
            ids::CLUSTER => {
                if first_cluster_offset.is_none() {
                    first_cluster_offset = Some(body_start - e.header_len as u64);
                }
                input.seek(SeekFrom::Start(body_start - e.header_len as u64))?;
                break;
            }
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

    // Cues are often written after the final Cluster — if we haven't seen
    // them yet and the segment size is known, scan from the first cluster
    // to segment end looking for a top-level Cues element. We keep this
    // best-effort: any I/O error or parse problem leaves `cues` empty
    // and falls back to Unsupported at seek time.
    if cues.is_empty() {
        if let Some(first_cluster) = first_cluster_offset {
            let resume_pos = input.stream_position()?;
            if scan_cues_from(&mut *input, first_cluster, segment_data_end, &mut cues).is_err() {
                cues.clear();
            }
            // Restore reader position to the first cluster for next_packet().
            input.seek(SeekFrom::Start(resume_pos))?;
        }
    }

    // Sort cues by (track, time) for stable lookup.
    cues.sort_by(|a, b| a.track.cmp(&b.track).then(a.time.cmp(&b.time)));

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

    // Segment\Info\Duration is in Matroska timecode ticks (timecode_scale ns
    // per tick), stored as a float. Translate to microseconds.
    let duration_micros: i64 = if info.duration > 0.0 {
        (info.duration * (timecode_scale_ns as f64) / 1_000.0) as i64
    } else {
        0
    };

    // Build reverse map: stream index → MKV TrackNumber.
    let mut track_number_by_index: Vec<u64> = vec![0; streams.len()];
    for (num, &idx) in &track_index_by_number {
        track_number_by_index[idx as usize] = *num;
    }

    Ok(Box::new(MkvDemuxer {
        input,
        streams,
        track_index_by_number,
        track_number_by_index,
        segment_data_start,
        segment_data_end,
        cluster_state: ClusterState::Idle,
        out_queue: std::collections::VecDeque::new(),
        time_base,
        metadata,
        duration_micros,
        cues,
        timecode_scale_ns,
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

fn parse_info(
    r: &mut dyn ReadSeek,
    end: u64,
    out: &mut SegmentInfo,
    metadata: &mut Vec<(String, String)>,
) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TIMECODE_SCALE => out.timecode_scale = read_uint(r, e.size as usize)?,
            ids::DURATION => out.duration = read_float(r, e.size as usize)?,
            ids::TITLE => {
                let s = read_string(r, e.size as usize)?;
                if !s.is_empty() {
                    metadata.push(("title".into(), s));
                }
            }
            ids::MUXING_APP => {
                let s = read_string(r, e.size as usize)?;
                if !s.is_empty() {
                    metadata.push(("muxer".into(), s));
                }
            }
            ids::WRITING_APP => {
                let s = read_string(r, e.size as usize)?;
                if !s.is_empty() {
                    metadata.push(("encoder".into(), s));
                }
            }
            ids::DATE_UTC => {
                // 8-byte signed integer: nanoseconds since 2001-01-01 00:00:00 UTC.
                if e.size == 8 {
                    let ns = read_uint(r, 8)? as i64;
                    let secs_since_2001 = ns / 1_000_000_000;
                    let unix_2001: i64 = 978_307_200;
                    let unix = unix_2001 + secs_since_2001;
                    metadata.push(("date".into(), format_iso8601(unix)));
                } else {
                    skip(r, e.size)?;
                }
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_tags(r: &mut dyn ReadSeek, end: u64, metadata: &mut Vec<(String, String)>) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TAG => {
                let tag_end = r.stream_position()? + e.size;
                parse_tag(r, tag_end, metadata)?;
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_tag(r: &mut dyn ReadSeek, end: u64, metadata: &mut Vec<(String, String)>) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::SIMPLE_TAG => {
                let st_end = r.stream_position()? + e.size;
                parse_simple_tag(r, st_end, metadata)?;
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_simple_tag(
    r: &mut dyn ReadSeek,
    end: u64,
    metadata: &mut Vec<(String, String)>,
) -> Result<()> {
    let mut name: Option<String> = None;
    let mut value: Option<String> = None;
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::TAG_NAME => name = Some(read_string(r, e.size as usize)?),
            ids::TAG_STRING => value = Some(read_string(r, e.size as usize)?),
            _ => skip(r, e.size)?,
        }
    }
    if let (Some(n), Some(v)) = (name, value) {
        let key = n.to_ascii_lowercase();
        if !key.is_empty() && !v.is_empty() {
            metadata.push((key, v));
        }
    }
    Ok(())
}

/// Format a unix timestamp (seconds since 1970-01-01 UTC) as an ISO-8601 date.
/// Roughly ffprobe-compatible; ignores leap seconds.
fn format_iso8601(unix_secs: i64) -> String {
    let (y, m, d, hh, mm, ss) = civil_from_days_seconds(unix_secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, hh, mm, ss)
}

fn civil_from_days_seconds(unix_secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400) as u32;
    // Howard Hinnant's date algorithms — shift so that era 0 starts 0000-03-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    (year, m, d, hh, mm, ss)
}

/// One Cues → CuePoint entry, denormalised to (track, time, cluster_offset)
/// where `cluster_offset` is a byte offset relative to the Segment payload
/// start (i.e. add it to `segment_data_start` to get an absolute file pos).
#[derive(Clone, Debug)]
struct CueEntry {
    track: u64,
    /// Timestamp in Matroska ticks (timecode_scale ns per tick).
    time: u64,
    cluster_offset: u64,
}

fn parse_cues(r: &mut dyn ReadSeek, end: u64, out: &mut Vec<CueEntry>) -> Result<()> {
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::CUE_POINT => {
                let body_end = r.stream_position()? + e.size;
                parse_cue_point(r, body_end, out)?;
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_cue_point(r: &mut dyn ReadSeek, end: u64, out: &mut Vec<CueEntry>) -> Result<()> {
    let mut time: u64 = 0;
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::CUE_TIME => time = read_uint(r, e.size as usize)?,
            ids::CUE_TRACK_POSITIONS => {
                let body_end = r.stream_position()? + e.size;
                parse_cue_track_positions(r, body_end, time, out)?;
            }
            _ => skip(r, e.size)?,
        }
    }
    Ok(())
}

fn parse_cue_track_positions(
    r: &mut dyn ReadSeek,
    end: u64,
    time: u64,
    out: &mut Vec<CueEntry>,
) -> Result<()> {
    let mut track: u64 = 0;
    let mut cluster_offset: Option<u64> = None;
    while r.stream_position()? < end {
        let e = read_element_header(r)?;
        match e.id {
            ids::CUE_TRACK => track = read_uint(r, e.size as usize)?,
            ids::CUE_CLUSTER_POSITION => cluster_offset = Some(read_uint(r, e.size as usize)?),
            _ => skip(r, e.size)?,
        }
    }
    if let Some(off) = cluster_offset {
        out.push(CueEntry {
            track,
            time,
            cluster_offset: off,
        });
    }
    Ok(())
}

/// Best-effort scan of the byte range `[start, end)` looking for a top-level
/// Cues element whose header we can find intact. Used when the Cues element
/// appears after the last Cluster in the file (the common ffmpeg layout
/// when muxing in a single pass with index-at-end).
fn scan_cues_from(
    r: &mut dyn ReadSeek,
    start: u64,
    end: u64,
    out: &mut Vec<CueEntry>,
) -> Result<()> {
    r.seek(SeekFrom::Start(start))?;
    while r.stream_position()? < end {
        let pos = r.stream_position()?;
        let e = read_element_header(r)?;
        let body_start = r.stream_position()?;
        let body_end = if e.size == VINT_UNKNOWN_SIZE {
            end
        } else {
            body_start + e.size
        };
        if body_end > end {
            // Truncated or invalid — abort scan.
            r.seek(SeekFrom::Start(pos))?;
            return Ok(());
        }
        match e.id {
            ids::CUES => {
                parse_cues(r, body_end, out)?;
                return Ok(());
            }
            _ => {
                if e.size == VINT_UNKNOWN_SIZE {
                    // Can't blindly skip unknown-size elements that aren't Cues.
                    return Ok(());
                }
                r.seek(SeekFrom::Start(body_end))?;
            }
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
    /// Reverse of `track_index_by_number`: stream index → MKV TrackNumber.
    track_number_by_index: Vec<u64>,
    /// Byte offset of the Segment payload start (immediately after the
    /// Segment element's header). Cue `cluster_offset` values are relative
    /// to this position.
    segment_data_start: u64,
    segment_data_end: u64,
    cluster_state: ClusterState,
    out_queue: std::collections::VecDeque<Packet>,
    time_base: TimeBase,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
    /// Cue index entries, sorted by (track, time). Empty if the file has
    /// no Cues element — `seek_to` returns `Error::Unsupported` in that
    /// case.
    cues: Vec<CueEntry>,
    /// Nanoseconds per Matroska timecode tick (the Segment\Info\TimecodeScale
    /// value, defaulted to 1_000_000 when absent).
    timecode_scale_ns: u64,
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

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        if self.duration_micros > 0 {
            Some(self.duration_micros)
        } else {
            None
        }
    }

    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if stream_index as usize >= self.streams.len() {
            return Err(Error::invalid(format!(
                "MKV: stream index {stream_index} out of range"
            )));
        }
        if self.cues.is_empty() {
            return Err(Error::unsupported(
                "MKV: no Cues index in file — cannot seek",
            ));
        }
        let track_number = self.track_number_by_index[stream_index as usize];

        // Convert the stream's pts → Matroska ticks.
        //   pts_seconds  = pts * stream.time_base.num / stream.time_base.den
        //   ticks        = pts_seconds * 1e9 / timecode_scale_ns
        //                = pts * num * 1e9 / (den * timecode_scale_ns)
        // Every stream in this demuxer currently exposes the segment time
        // base (timecode_scale_ns / 1e9), so the conversion collapses to
        // a copy — but we still do the full calculation so behaviour is
        // correct when other time bases are supplied.
        let stream_tb = self.streams[stream_index as usize].time_base.as_rational();
        let target_ticks_i128: i128 = if stream_tb.num == 0 || stream_tb.den == 0 {
            pts as i128
        } else {
            let numer = pts as i128 * stream_tb.num as i128 * 1_000_000_000i128;
            let denom = stream_tb.den as i128 * self.timecode_scale_ns as i128;
            if denom == 0 {
                pts as i128
            } else {
                numer / denom
            }
        };
        let target_ticks: u64 = target_ticks_i128.max(0) as u64;

        // Find last cue entry for this track with time <= target_ticks.
        // Cues are sorted by (track, time); use a manual scan of the
        // contiguous track block to keep the code obvious and panic-free.
        let mut best: Option<&CueEntry> = None;
        for c in self.cues.iter().filter(|c| c.track == track_number) {
            if c.time <= target_ticks {
                best = Some(c);
            } else {
                break;
            }
        }
        // If target is before the first cue, fall back to the first cue
        // for this track (seek returns the actual landed pts).
        if best.is_none() {
            best = self.cues.iter().find(|c| c.track == track_number);
        }
        let cue = best.ok_or_else(|| {
            Error::unsupported(format!(
                "MKV: no Cues entries for track {track_number} (stream {stream_index})"
            ))
        })?;

        let abs = self.segment_data_start + cue.cluster_offset;
        self.input.seek(SeekFrom::Start(abs))?;
        // Reset cluster reader state + any previously queued packets.
        self.cluster_state = ClusterState::Idle;
        self.out_queue.clear();

        // Convert the landed ticks back into the stream's time base.
        let landed_pts: i64 = if stream_tb.num == 0 || stream_tb.den == 0 {
            cue.time as i64
        } else {
            let numer = cue.time as i128 * stream_tb.den as i128 * self.timecode_scale_ns as i128;
            let denom = stream_tb.num as i128 * 1_000_000_000i128;
            if denom == 0 {
                cue.time as i64
            } else {
                (numer / denom) as i64
            }
        };
        Ok(landed_pts)
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
