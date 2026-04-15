//! Matroska muxer.
//!
//! Layout produced:
//!
//! ```text
//! EBML header
//! Segment (unknown size)
//!   Info (timecode scale, muxing/writing app)
//!   Tracks (one TrackEntry per input stream)
//!   Cluster (one per ~5 s of media, or one per file for short input)
//!     Timecode
//!     SimpleBlock × N
//! ```
//!
//! Both Segment and Cluster use the EBML "unknown size" sentinel so the muxer
//! is streaming-friendly (no seek-back needed). All packet timestamps are
//! converted to milliseconds using the standard 1 ms `TIMECODE_SCALE`.

use std::io::Write;

use oxideav_container::{Muxer, WriteSeek};
use oxideav_core::{Error, MediaType, Packet, Result, StreamInfo};

use crate::codec_id;
use crate::ebml::{write_element_id, write_vint, VINT_UNKNOWN_SIZE};
use crate::ids;

/// Cluster every ~5 seconds (in MKV ms timecode units).
const CLUSTER_DURATION_MS: i64 = 5_000;

pub fn open(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.is_empty() {
        return Err(Error::invalid("MKV muxer: need at least one stream"));
    }
    let stream_track_numbers: Vec<u64> = (0..streams.len() as u64).map(|i| i + 1).collect();
    Ok(Box::new(MkvMuxer {
        output,
        streams: streams.to_vec(),
        track_numbers: stream_track_numbers,
        stream_pts: vec![0i64; streams.len()],
        cluster_open: false,
        cluster_timecode_ms: 0,
        header_written: false,
        trailer_written: false,
    }))
}

struct MkvMuxer {
    output: Box<dyn WriteSeek>,
    streams: Vec<StreamInfo>,
    /// Per-stream MKV track numbers (1-indexed).
    track_numbers: Vec<u64>,
    /// Per-stream running pts, in the stream's own time base. Used to
    /// synthesise per-packet timestamps when the input container only
    /// signals page/chunk granules (e.g. Ogg).
    stream_pts: Vec<i64>,
    cluster_open: bool,
    /// Timecode (in ms) at the start of the currently open cluster.
    cluster_timecode_ms: i64,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for MkvMuxer {
    fn format_name(&self) -> &str {
        "matroska"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("MKV muxer: write_header called twice"));
        }
        // EBML header element.
        let mut ebml_body = Vec::new();
        write_uint_element(&mut ebml_body, ids::EBML_VERSION, 1);
        write_uint_element(&mut ebml_body, ids::EBML_READ_VERSION, 1);
        write_uint_element(&mut ebml_body, ids::EBML_MAX_ID_LENGTH, 4);
        write_uint_element(&mut ebml_body, ids::EBML_MAX_SIZE_LENGTH, 8);
        write_string_element(&mut ebml_body, ids::EBML_DOC_TYPE, "matroska");
        write_uint_element(&mut ebml_body, ids::EBML_DOC_TYPE_VERSION, 4);
        write_uint_element(&mut ebml_body, ids::EBML_DOC_TYPE_READ_VERSION, 2);
        let mut all = Vec::new();
        write_master_element(&mut all, ids::EBML_HEADER, &ebml_body);

        // Segment with unknown size.
        all.extend_from_slice(&write_element_id(ids::SEGMENT));
        all.extend_from_slice(&write_vint(VINT_UNKNOWN_SIZE, 0));

        // Info element.
        let mut info_body = Vec::new();
        write_uint_element(&mut info_body, ids::TIMECODE_SCALE, 1_000_000); // 1 ms
        write_string_element(&mut info_body, ids::MUXING_APP, "oxideav");
        write_string_element(&mut info_body, ids::WRITING_APP, "oxideav");
        write_master_element(&mut all, ids::INFO, &info_body);

        // Tracks element.
        let mut tracks_body = Vec::new();
        for (i, s) in self.streams.iter().enumerate() {
            let track_number = self.track_numbers[i];
            let mut t = Vec::new();
            write_uint_element(&mut t, ids::TRACK_NUMBER, track_number);
            write_uint_element(&mut t, ids::TRACK_UID, track_number);
            let track_type = match s.params.media_type {
                MediaType::Audio => ids::TRACK_TYPE_AUDIO,
                MediaType::Video => ids::TRACK_TYPE_VIDEO,
                MediaType::Subtitle => ids::TRACK_TYPE_SUBTITLE,
                _ => 17, // treat as subtitle/data fallback
            };
            write_uint_element(&mut t, ids::TRACK_TYPE, track_type);
            write_uint_element(&mut t, ids::FLAG_LACING, 0);
            if let Some(name) = codec_id::to_matroska(&s.params.codec_id) {
                write_string_element(&mut t, ids::CODEC_ID, name);
            } else {
                // Fall back to a Matroska-style unknown id; players will reject
                // this but the file is otherwise valid.
                let raw = format!("X_{}", s.params.codec_id);
                write_string_element(&mut t, ids::CODEC_ID, &raw);
            }
            // CodecPrivate with codec-specific normalisation.
            let cp = encode_codec_private(&s.params.codec_id, &s.params.extradata);
            if !cp.is_empty() {
                write_bytes_element(&mut t, ids::CODEC_PRIVATE, &cp);
            }
            // Codec-specific timing fields (Opus uses CodecDelay = pre_skip in ns
            // and a recommended SeekPreRoll of 80 ms).
            if s.params.codec_id.as_str() == "opus" {
                let pre_skip_samples = parse_opus_pre_skip(&s.params.extradata);
                let codec_delay_ns = pre_skip_samples as u64 * 1_000_000_000 / 48_000;
                write_uint_element(&mut t, ids::CODEC_DELAY, codec_delay_ns);
                write_uint_element(&mut t, ids::SEEK_PRE_ROLL, 80_000_000);
            }
            if s.params.media_type == MediaType::Audio {
                let mut audio = Vec::new();
                if let Some(sr) = s.params.sample_rate {
                    write_float_element(&mut audio, ids::SAMPLING_FREQUENCY, sr as f64);
                }
                if let Some(ch) = s.params.channels {
                    write_uint_element(&mut audio, ids::CHANNELS, ch as u64);
                }
                if let Some(fmt) = s.params.sample_format {
                    let bd = (fmt.bytes_per_sample() * 8) as u64;
                    write_uint_element(&mut audio, ids::BIT_DEPTH, bd);
                }
                write_master_element(&mut t, ids::AUDIO, &audio);
            }
            if s.params.media_type == MediaType::Video {
                let mut video = Vec::new();
                if let Some(w) = s.params.width {
                    write_uint_element(&mut video, ids::PIXEL_WIDTH, w as u64);
                }
                if let Some(h) = s.params.height {
                    write_uint_element(&mut video, ids::PIXEL_HEIGHT, h as u64);
                }
                write_master_element(&mut t, ids::VIDEO, &video);
            }
            write_master_element(&mut tracks_body, ids::TRACK_ENTRY, &t);
        }
        write_master_element(&mut all, ids::TRACKS, &tracks_body);

        self.output.write_all(&all)?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("MKV muxer: write_header not called"));
        }
        let stream_idx = packet.stream_index as usize;
        if stream_idx >= self.streams.len() {
            return Err(Error::invalid(format!(
                "MKV muxer: unknown stream index {}",
                stream_idx
            )));
        }
        let stream = &self.streams[stream_idx];
        let track_number = self.track_numbers[stream_idx];

        // Effective per-packet pts. If the source set one, use it; otherwise
        // derive from accumulated stream_pts and codec-specific durations.
        let codec = stream.params.codec_id.as_str();
        let derived_duration: Option<i64> = match codec {
            "opus" => opus_packet_duration_samples(&packet.data).map(|s| s as i64),
            _ => packet.duration,
        };
        let effective_pts = match packet.pts {
            Some(v) => v,
            None => self.stream_pts[stream_idx],
        };
        // Advance the running counter for the next packet without an explicit pts.
        if let Some(d) = derived_duration {
            self.stream_pts[stream_idx] = effective_pts + d;
        } else if packet.pts.is_some() {
            self.stream_pts[stream_idx] = effective_pts;
        }

        let pts_ms = pts_to_ms(effective_pts, stream.time_base);

        // Decide whether to start a new cluster.
        if !self.cluster_open
            || pts_ms - self.cluster_timecode_ms > CLUSTER_DURATION_MS
            || pts_ms - self.cluster_timecode_ms > i16::MAX as i64
            || pts_ms - self.cluster_timecode_ms < 0
        {
            self.start_cluster(pts_ms)?;
        }

        let timecode_offset = pts_ms - self.cluster_timecode_ms;
        if timecode_offset < i16::MIN as i64 || timecode_offset > i16::MAX as i64 {
            return Err(Error::other(
                "MKV muxer: packet timecode delta exceeds i16 range",
            ));
        }
        let block_bytes =
            build_simple_block(track_number, timecode_offset as i16, packet, &packet.data);
        self.output.write_all(&block_bytes)?;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Cluster has unknown size, so just stop writing — the EBML reader on
        // the other end will hit EOF cleanly.
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

impl MkvMuxer {
    fn start_cluster(&mut self, timecode_ms: i64) -> Result<()> {
        // Write Cluster element id + unknown-size sentinel.
        self.output.write_all(&write_element_id(ids::CLUSTER))?;
        self.output.write_all(&write_vint(VINT_UNKNOWN_SIZE, 0))?;
        // Write Timecode child element.
        let mut tc = Vec::new();
        write_uint_element(&mut tc, ids::TIMECODE, timecode_ms.max(0) as u64);
        self.output.write_all(&tc)?;
        self.cluster_timecode_ms = timecode_ms.max(0);
        self.cluster_open = true;
        Ok(())
    }
}

/// Build a SimpleBlock element: track number (vint) + timecode (s16) + flags
/// + frame data, wrapped in id + size.
fn build_simple_block(track: u64, tc_offset: i16, packet: &Packet, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(&write_vint(track, 0));
    body.extend_from_slice(&tc_offset.to_be_bytes());
    let mut flags: u8 = 0;
    if packet.flags.keyframe {
        flags |= 0x80;
    }
    body.push(flags);
    body.extend_from_slice(data);
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&write_element_id(ids::SIMPLE_BLOCK));
    out.extend_from_slice(&write_vint(body.len() as u64, 0));
    out.extend_from_slice(&body);
    out
}

fn pts_to_ms(value: i64, tb: oxideav_core::TimeBase) -> i64 {
    let r = tb.as_rational();
    if r.den == 0 {
        return value;
    }
    // value * num / den (in seconds) * 1000 (to ms).
    // Use i128 to avoid overflow.
    let v = value as i128 * r.num as i128 * 1000;
    (v / r.den as i128) as i64
}

/// Decode the Opus TOC byte (and code-3 frame count byte if needed) to get
/// the packet's total decoded sample count at 48 kHz. Returns `None` if the
/// packet doesn't look like a valid Opus packet.
///
/// Reference: RFC 6716 §3.1, Table 2.
fn opus_packet_duration_samples(packet: &[u8]) -> Option<u32> {
    if packet.is_empty() {
        return None;
    }
    let toc = packet[0];
    let config = toc >> 3;
    let frame_size_48k: u32 = match config {
        0 | 4 | 8 => 480,
        1 | 5 | 9 => 960,
        2 | 6 | 10 => 1920,
        3 | 7 | 11 => 2880,
        12 | 14 => 480,
        13 | 15 => 960,
        16 | 20 | 24 | 28 => 120,
        17 | 21 | 25 | 29 => 240,
        18 | 22 | 26 | 30 => 480,
        19 | 23 | 27 | 31 => 960,
        _ => return None,
    };
    let n_frames: u32 = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        3 => {
            if packet.len() < 2 {
                return None;
            }
            (packet[1] & 0x3F) as u32
        }
        _ => unreachable!(),
    };
    Some(frame_size_48k * n_frames)
}

/// Read the 16-bit pre-skip field from an OpusHead packet (RFC 7845 §5.1
/// bytes 10..12 little-endian). Returns 0 if the buffer doesn't look like
/// a valid OpusHead.
fn parse_opus_pre_skip(extradata: &[u8]) -> u16 {
    if extradata.len() < 12 || &extradata[0..8] != b"OpusHead" {
        return 0;
    }
    u16::from_le_bytes([extradata[10], extradata[11]])
}

fn encode_codec_private(codec_id: &oxideav_core::CodecId, extradata: &[u8]) -> Vec<u8> {
    match codec_id.as_str() {
        // Matroska's A_FLAC mapping carries the leading "fLaC" magic in
        // CodecPrivate even though many docs imply it's optional. ffmpeg
        // expects it; we always prepend it on the muxer side.
        "flac" => {
            let mut out = Vec::with_capacity(4 + extradata.len());
            out.extend_from_slice(b"fLaC");
            out.extend_from_slice(extradata);
            out
        }
        _ => extradata.to_vec(),
    }
}

// --- Element-writing helpers ----------------------------------------------

fn write_uint_element(buf: &mut Vec<u8>, id: u32, value: u64) {
    let n = if value == 0 {
        1
    } else {
        ((64 - value.leading_zeros() + 7) / 8) as usize
    };
    buf.extend_from_slice(&write_element_id(id));
    buf.extend_from_slice(&write_vint(n as u64, 0));
    for i in (0..n).rev() {
        buf.push(((value >> (i * 8)) & 0xFF) as u8);
    }
}

fn write_string_element(buf: &mut Vec<u8>, id: u32, value: &str) {
    buf.extend_from_slice(&write_element_id(id));
    buf.extend_from_slice(&write_vint(value.len() as u64, 0));
    buf.extend_from_slice(value.as_bytes());
}

fn write_bytes_element(buf: &mut Vec<u8>, id: u32, value: &[u8]) {
    buf.extend_from_slice(&write_element_id(id));
    buf.extend_from_slice(&write_vint(value.len() as u64, 0));
    buf.extend_from_slice(value);
}

fn write_float_element(buf: &mut Vec<u8>, id: u32, value: f64) {
    buf.extend_from_slice(&write_element_id(id));
    buf.extend_from_slice(&write_vint(8, 0));
    buf.extend_from_slice(&value.to_be_bytes());
}

fn write_master_element(buf: &mut Vec<u8>, id: u32, body: &[u8]) {
    buf.extend_from_slice(&write_element_id(id));
    buf.extend_from_slice(&write_vint(body.len() as u64, 0));
    buf.extend_from_slice(body);
}
