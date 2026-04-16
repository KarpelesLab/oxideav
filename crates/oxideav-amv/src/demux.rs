//! AMV container demuxer.
//!
//! Layout (re-derived from real AMV files plus the chunk tree documented in
//! ffmpeg's `libavformat/amvenc.c`):
//!
//! ```text
//! RIFF <size> "AMV "
//!   LIST <size> "hdrl"
//!     "amvh" 56 [us_per_frame, 28 reserved, width, height, fps_den, fps_num,
//!                4 reserved, duration]   (all LE u32; "size" fields throughout
//!                                         the file are routinely zero)
//!     LIST <size> "strl"
//!       "strh" 56  (zeroed)
//!       "strf" 36  (zeroed)
//!     LIST <size> "strl"
//!       "strh" 48  (zeroed)
//!       "strf" 20  (audio WAVEFORMATEX: format=1, channels, sample_rate,
//!                                       byte_rate, block_align, bps=16, +pad)
//!   LIST <size> "movi"
//!     <fourcc 00dc | 01wb> <le32 payload size> <payload>   ← interleaved V/A
//!     ...
//!   "AMV_"
//!   "END_"
//! ```
//!
//! Conformance / leniency:
//!
//! * The "size" fields on top-level LIST/RIFF chunks are commonly zero
//!   ("some players break when they're set correctly" — ffmpeg amvenc
//!   comment). We don't trust them; we walk by tag and use the per-chunk
//!   size only on `00dc` / `01wb`.
//! * No 2-byte chunk padding between `00dc` / `01wb` (verified against
//!   real ffmpeg output — `vsubhash`-style AMV writes go straight on,
//!   regardless of payload parity).

use std::io::Read;

use oxideav_container::ContainerRegistry;
use oxideav_container::{Demuxer, ProbeData, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, PixelFormat, Rational, Result,
    SampleFormat, StreamInfo, TimeBase,
};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("amv", open);
    reg.register_extension("amv", "amv");
    reg.register_probe("amv", probe);
}

/// `RIFF????AMV ` — RIFF with form type `AMV ` (trailing space).
fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() >= 12 && &p.buf[0..4] == b"RIFF" && &p.buf[8..12] == b"AMV " {
        100
    } else if p.ext == Some("amv") {
        25
    } else {
        0
    }
}

fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut blob = Vec::new();
    input.read_to_end(&mut blob)?;
    if blob.len() < 12 {
        return Err(Error::invalid("AMV: file shorter than RIFF/AMV header"));
    }
    if &blob[0..4] != b"RIFF" || &blob[8..12] != b"AMV " {
        return Err(Error::invalid("AMV: not a RIFF/AMV file"));
    }

    let header = parse_amvh(&blob)?;

    // Locate the start of the `movi` body (just past the `LIST<size>movi`
    // sequence). ffmpeg writes a fixed top-level layout; we walk the body of
    // the RIFF chunk looking for the next LIST whose form-type is `movi`.
    let movi_start = find_movi_start(&blob)?;

    let v_params = {
        let mut p = CodecParameters::video(CodecId::new(crate::VIDEO_CODEC_ID_STR));
        p.media_type = MediaType::Video;
        p.width = Some(header.width);
        p.height = Some(header.height);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        if header.fps_den > 0 {
            p.frame_rate = Some(Rational::new(header.fps_num as i64, header.fps_den as i64));
        }
        p
    };
    let a_params = {
        let mut p = CodecParameters::audio(CodecId::new(crate::AUDIO_CODEC_ID_STR));
        p.media_type = MediaType::Audio;
        p.channels = Some(header.audio_channels.max(1));
        p.sample_rate = Some(header.audio_sample_rate);
        p.sample_format = Some(SampleFormat::S16);
        p
    };

    // Stream 0 = video, stream 1 = audio. AMV requires strict V-A
    // interleaving so this index assignment matches packet order.
    let v_tb = if header.fps_den > 0 && header.fps_num > 0 {
        TimeBase::new(header.fps_den as i64, header.fps_num as i64)
    } else {
        TimeBase::new(1, 30)
    };
    let a_tb = TimeBase::new(1, header.audio_sample_rate as i64);

    let streams = vec![
        StreamInfo {
            index: 0,
            time_base: v_tb,
            duration: None,
            start_time: Some(0),
            params: v_params,
        },
        StreamInfo {
            index: 1,
            time_base: a_tb,
            duration: None,
            start_time: Some(0),
            params: a_params,
        },
    ];

    Ok(Box::new(AmvDemuxer {
        blob,
        cursor: movi_start,
        streams,
        v_pts: 0,
        a_pts: 0,
        eof: false,
    }))
}

/// Locate the offset of the first byte inside the `movi` LIST body — i.e.
/// the byte right after the `"movi"` form-type.
fn find_movi_start(blob: &[u8]) -> Result<usize> {
    // Top-level RIFF body starts at offset 12.
    let mut i = 12usize;
    while i + 8 <= blob.len() {
        let tag = &blob[i..i + 4];
        let size =
            u32::from_le_bytes([blob[i + 4], blob[i + 5], blob[i + 6], blob[i + 7]]) as usize;
        i += 8;
        if tag == b"LIST" {
            if i + 4 > blob.len() {
                return Err(Error::invalid("AMV: truncated LIST"));
            }
            let form = &blob[i..i + 4];
            i += 4;
            if form == b"movi" {
                return Ok(i);
            }
            // Top-level sizes are commonly zero — fall back to walking the
            // children. For `hdrl` we don't actually need its body here so
            // skip via the size if non-zero, else jump to next RIFF tag by
            // walking child chunks.
            if size >= 4 && i + (size - 4) <= blob.len() {
                i += size - 4;
            } else {
                // size was bogus; walk children byte-by-byte until we find
                // the next plausible LIST tag.
                while i + 4 <= blob.len() && &blob[i..i + 4] != b"LIST" {
                    i += 1;
                }
            }
        } else {
            // Non-list top-level chunk (rare — amvh shouldn't be at top
            // level but accept it). Skip its declared size.
            if size > 0 && i + size <= blob.len() {
                i += size;
            } else {
                i += 1;
            }
        }
    }
    Err(Error::invalid("AMV: movi list not found"))
}

/// AMV main header (`amvh` chunk, 56 bytes).
#[derive(Clone, Copy, Debug, Default)]
struct AmvHeader {
    /// Width in pixels (must be >0).
    pub width: u32,
    /// Height in pixels (must be >0).
    pub height: u32,
    /// Frame-rate numerator (per ffmpeg's amvenc layout: stored at offset
    /// 44, paired with `fps_den` at 40).
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
}

/// Walk the file looking for the `amvh` chunk inside `LIST hdrl`, plus the
/// audio `strf` (WAVEFORMATEX) inside the second `strl`. Robust to the
/// common case where every LIST/RIFF size field is zero.
fn parse_amvh(blob: &[u8]) -> Result<AmvHeader> {
    let mut h = AmvHeader::default();

    // Linear scan for known 4-CCs. The container is small and the tag set
    // is unambiguous (no risk of `amvh` colliding with payload bytes
    // because we stop scanning once we've seen `movi`).
    let mut i = 12usize;
    let mut saw_audio_strf = false;
    while i + 4 <= blob.len() {
        let tag = &blob[i..i + 4];
        if tag == b"movi" {
            break;
        }
        if tag == b"amvh" {
            if i + 8 + 56 > blob.len() {
                return Err(Error::invalid("AMV: amvh truncated"));
            }
            let body = &blob[i + 8..i + 8 + 56];
            // microseconds-per-frame at offset 0 — mostly redundant with
            // the explicit fps numerator/denominator further down, so we
            // ignore it for time-base computation.
            h.width = u32::from_le_bytes(body[32..36].try_into().unwrap());
            h.height = u32::from_le_bytes(body[36..40].try_into().unwrap());
            h.fps_den = u32::from_le_bytes(body[40..44].try_into().unwrap());
            h.fps_num = u32::from_le_bytes(body[44..48].try_into().unwrap());
            i += 8 + 56;
            continue;
        }
        if tag == b"strf" {
            // Audio strf is 20 bytes (WAVEFORMATEX). The first `strf`
            // encountered is the video one (size 36) which we ignore.
            let size = u32::from_le_bytes(blob[i + 4..i + 8].try_into().unwrap()) as usize;
            if !saw_audio_strf && size == 20 {
                if i + 8 + 20 > blob.len() {
                    return Err(Error::invalid("AMV: audio strf truncated"));
                }
                let body = &blob[i + 8..i + 8 + 20];
                let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                h.audio_channels = if channels == 0 { 1 } else { channels };
                h.audio_sample_rate = if sample_rate == 0 {
                    22_050
                } else {
                    sample_rate
                };
                saw_audio_strf = true;
            } else if size == 36 {
                // video strf — typically zeroed in AMV; ignore.
            }
            i += 8 + size;
            continue;
        }
        i += 1;
    }
    if h.width == 0 || h.height == 0 {
        return Err(Error::invalid("AMV: missing amvh / zero width/height"));
    }
    if h.audio_sample_rate == 0 {
        h.audio_sample_rate = 22_050;
        h.audio_channels = 1;
    }
    Ok(h)
}

struct AmvDemuxer {
    blob: Vec<u8>,
    cursor: usize,
    streams: Vec<StreamInfo>,
    v_pts: i64,
    a_pts: i64,
    eof: bool,
}

impl Demuxer for AmvDemuxer {
    fn format_name(&self) -> &str {
        "amv"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.eof {
            return Err(Error::Eof);
        }
        loop {
            if self.cursor + 8 > self.blob.len() {
                self.eof = true;
                return Err(Error::Eof);
            }
            let tag = &self.blob[self.cursor..self.cursor + 4];
            // Trailer markers stop the walk.
            if tag == b"AMV_" || tag == b"END_" {
                self.eof = true;
                return Err(Error::Eof);
            }
            let size = u32::from_le_bytes(
                self.blob[self.cursor + 4..self.cursor + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            // A bogus huge size means we've fallen off the end of the
            // movi list — stop instead of looping forever.
            if size > self.blob.len().saturating_sub(self.cursor + 8) {
                self.eof = true;
                return Err(Error::Eof);
            }
            let payload_start = self.cursor + 8;
            let payload_end = payload_start + size;
            self.cursor = payload_end;
            // No padding: AMV writes chunks back-to-back regardless of
            // payload parity (verified empirically against ffmpeg-produced
            // files; the AVI spec says pad-to-even but AMV ignores it).
            match tag {
                b"00dc" => {
                    let pts = self.v_pts;
                    self.v_pts += 1;
                    let data = self.blob[payload_start..payload_end].to_vec();
                    let mut pkt = Packet::new(0, self.streams[0].time_base, data);
                    pkt.pts = Some(pts);
                    pkt.dts = Some(pts);
                    pkt.duration = Some(1);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
                b"01wb" => {
                    // Number of decoded samples per AMV audio chunk =
                    // (payload_size - 8) * 2 (8-byte header + packed
                    // nibbles, two samples per byte).
                    let n_samples = if size >= 8 {
                        ((size - 8) * 2) as i64
                    } else {
                        0
                    };
                    let pts = self.a_pts;
                    self.a_pts += n_samples;
                    let data = self.blob[payload_start..payload_end].to_vec();
                    let mut pkt = Packet::new(1, self.streams[1].time_base, data);
                    pkt.pts = Some(pts);
                    pkt.dts = Some(pts);
                    pkt.duration = Some(n_samples);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
                _ => {
                    // Unknown / junk chunk — skip.
                    continue;
                }
            }
        }
    }

    fn duration_micros(&self) -> Option<i64> {
        // amvh stores duration as a packed HH:MM:SS triple at the tail of
        // the chunk; ffmpeg even acknowledges its own writer leaves it
        // zero on most files. Fall back to None and let callers compute
        // from the longest stream.
        None
    }
}
