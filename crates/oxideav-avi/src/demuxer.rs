//! AVI (RIFF/AVI) demuxer.
//!
//! On `open()`:
//! 1. Verify the top-level `RIFF`…`AVI ` header.
//! 2. Locate the `hdrl` LIST, parse `avih` (main header) and each `strl`
//!    LIST → `strh` (stream header) + `strf` (stream format).
//! 3. Locate the `movi` LIST. Remember its start offset and size so we can
//!    walk packet chunks lazily.
//!
//! `next_packet()` walks chunks inside `movi`. Each payload chunk name is
//! `NNxx` where `NN` is a two-ASCII-digit stream index and `xx` is one of
//! `dc` (compressed video), `db` (uncompressed video), `wb` (audio), or
//! something else which we skip. Unknown or out-of-range indexes are skipped
//! so we can tolerate files with embedded junk (`JUNK`, `ix##`, unsupported
//! streams).

use std::io::{Seek, SeekFrom};

use oxideav_container::{Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Rational, Result, SampleFormat, StreamInfo,
    TimeBase,
};

use crate::codec_map::{audio_codec_id, video_codec_id};
use crate::riff::{read_chunk_header, read_form_type, skip_chunk, skip_pad, AVI_FORM, LIST, RIFF};
use crate::stream_format::{parse_bitmap_info_header, parse_waveformatex};

/// Factory registered with the container registry.
pub fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    // Top-level RIFF chunk.
    let top = match read_chunk_header(&mut *input)? {
        Some(h) => h,
        None => return Err(Error::invalid("AVI: empty file")),
    };
    if top.id != RIFF {
        return Err(Error::invalid("AVI: not a RIFF file"));
    }
    let form = read_form_type(&mut *input)?;
    if form != AVI_FORM {
        return Err(Error::invalid("AVI: RIFF form type is not AVI"));
    }

    // Walk top-level nested chunks until we've processed both hdrl and movi.
    let mut streams: Vec<StreamInfo> = Vec::new();
    let mut packet_chunk_suffix: Vec<[u8; 2]> = Vec::new();
    let mut movi_start: Option<u64> = None;
    let mut movi_end: Option<u64> = None;
    let mut avih: Option<AviMainHeader> = None;

    while let Some(hdr) = read_chunk_header(&mut *input)? {
        if hdr.id == LIST {
            let list_type = read_form_type(&mut *input)?;
            let body_len = hdr.size.saturating_sub(4);
            let body_start = input.stream_position()?;
            let body_end = body_start + body_len as u64;
            match &list_type {
                b"hdrl" => {
                    let (main, stream_infos, suffixes) = parse_hdrl(&mut *input, body_end)?;
                    avih = Some(main);
                    streams = stream_infos;
                    packet_chunk_suffix = suffixes;
                }
                b"movi" => {
                    movi_start = Some(body_start);
                    movi_end = Some(body_end);
                }
                _ => {}
            }
            // Jump to end of list (skips contents we didn't consume) + pad.
            input.seek(SeekFrom::Start(body_end))?;
            skip_pad(&mut *input, hdr.size)?;
        } else {
            // Non-list top-level chunks (JUNK, idx1, INFO at top, etc.).
            skip_chunk(&mut *input, &hdr)?;
        }
    }

    let movi_start = movi_start.ok_or_else(|| Error::invalid("AVI: missing movi list"))?;
    let movi_end = movi_end.ok_or_else(|| Error::invalid("AVI: missing movi list"))?;
    let _ = avih; // retained for future use (frame count etc.)
    if streams.is_empty() {
        return Err(Error::invalid("AVI: no streams"));
    }

    // Seek to start of movi body for next_packet.
    input.seek(SeekFrom::Start(movi_start))?;

    Ok(Box::new(AviDemuxer {
        input,
        streams,
        packet_chunk_suffix,
        movi_end,
        per_stream_counter: Vec::new(),
    }))
}

/// Decoded AVIMAINHEADER (dwMicroSecPerFrame / … struct).
///
/// Most fields are retained for future use (seek tables, buffer sizing) even
/// though the current demuxer consumes them only during parsing.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
struct AviMainHeader {
    micro_sec_per_frame: u32,
    #[allow(dead_code)]
    max_bytes_per_sec: u32,
    #[allow(dead_code)]
    flags: u32,
    total_frames: u32,
    #[allow(dead_code)]
    initial_frames: u32,
    streams: u32,
    #[allow(dead_code)]
    suggested_buffer_size: u32,
    width: u32,
    height: u32,
}

/// Parse the AVIMAINHEADER body (should be 56 bytes).
fn parse_avih(buf: &[u8]) -> Result<AviMainHeader> {
    if buf.len() < 40 {
        return Err(Error::invalid("AVI: avih too short"));
    }
    Ok(AviMainHeader {
        micro_sec_per_frame: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        max_bytes_per_sec: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        // dwPaddingGranularity at offset 8 is ignored.
        flags: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        total_frames: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        initial_frames: u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]),
        streams: u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]),
        suggested_buffer_size: u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]),
        width: u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]),
        height: u32::from_le_bytes([buf[36], buf[37], buf[38], buf[39]]),
    })
}

/// Parse the `hdrl` LIST body.
///
/// Reads `avih`, then walks each nested `strl` LIST to build one `StreamInfo`
/// per stream. Returns also the list of expected packet-chunk suffixes (e.g.
/// `b"dc"`, `b"wb"`) so the demuxer can recognise packets.
fn parse_hdrl<R: ReadSeek + ?Sized>(
    r: &mut R,
    end_pos: u64,
) -> Result<(AviMainHeader, Vec<StreamInfo>, Vec<[u8; 2]>)> {
    let mut main = AviMainHeader::default();
    let mut streams: Vec<StreamInfo> = Vec::new();
    let mut suffixes: Vec<[u8; 2]> = Vec::new();

    while r.stream_position()? < end_pos {
        let hdr = match read_chunk_header(r)? {
            Some(h) => h,
            None => break,
        };
        match &hdr.id {
            b"avih" => {
                let body = read_body_bounded(r, hdr.size)?;
                main = parse_avih(&body)?;
                skip_pad(r, hdr.size)?;
            }
            b"LIST" => {
                let list_type = read_form_type(r)?;
                let body_len = hdr.size.saturating_sub(4);
                let body_start = r.stream_position()?;
                let body_end = body_start + body_len as u64;
                if &list_type == b"strl" {
                    let (si, suf) = parse_strl(r, body_end, streams.len() as u32)?;
                    if let Some(si) = si {
                        streams.push(si);
                        suffixes.push(suf.unwrap_or(*b"xx"));
                    }
                }
                r.seek(SeekFrom::Start(body_end))?;
                skip_pad(r, hdr.size)?;
            }
            _ => {
                skip_chunk(r, &hdr)?;
            }
        }
    }
    Ok((main, streams, suffixes))
}

/// Parse a `strl` LIST. Returns the `StreamInfo` and expected packet suffix.
fn parse_strl<R: ReadSeek + ?Sized>(
    r: &mut R,
    end_pos: u64,
    index: u32,
) -> Result<(Option<StreamInfo>, Option<[u8; 2]>)> {
    let mut strh_buf: Option<Vec<u8>> = None;
    let mut strf_buf: Option<Vec<u8>> = None;
    while r.stream_position()? < end_pos {
        let hdr = match read_chunk_header(r)? {
            Some(h) => h,
            None => break,
        };
        match &hdr.id {
            b"strh" => {
                strh_buf = Some(read_body_bounded(r, hdr.size)?);
                skip_pad(r, hdr.size)?;
            }
            b"strf" => {
                strf_buf = Some(read_body_bounded(r, hdr.size)?);
                skip_pad(r, hdr.size)?;
            }
            _ => {
                skip_chunk(r, &hdr)?;
            }
        }
    }
    let strh = match strh_buf {
        Some(b) => b,
        None => return Ok((None, None)),
    };
    let strf = strf_buf.unwrap_or_default();
    let parsed = build_stream(index, &strh, &strf)?;
    Ok((Some(parsed.0), Some(parsed.1)))
}

/// Build a StreamInfo from strh + strf payloads.
fn build_stream(index: u32, strh: &[u8], strf: &[u8]) -> Result<(StreamInfo, [u8; 2])> {
    // AVISTREAMHEADER layout (56 bytes):
    //   0  fccType       [4]
    //   4  fccHandler    [4]
    //   8  dwFlags       u32
    //  12  wPriority     u16
    //  14  wLanguage     u16
    //  16  dwInitialFrames u32
    //  20  dwScale       u32
    //  24  dwRate        u32  (rate/scale = samples/sec)
    //  28  dwStart       u32
    //  32  dwLength      u32
    //  36  dwSuggestedBufferSize u32
    //  40  dwQuality     u32
    //  44  dwSampleSize  u32
    //  48  rcFrame       [4 * i16]
    if strh.len() < 48 {
        return Err(Error::invalid("AVI: strh too short"));
    }
    let mut fcc_type = [0u8; 4];
    fcc_type.copy_from_slice(&strh[0..4]);
    let mut fcc_handler = [0u8; 4];
    fcc_handler.copy_from_slice(&strh[4..8]);
    let scale = u32::from_le_bytes([strh[20], strh[21], strh[22], strh[23]]).max(1);
    let rate = u32::from_le_bytes([strh[24], strh[25], strh[26], strh[27]]).max(1);
    let length = u32::from_le_bytes([strh[32], strh[33], strh[34], strh[35]]);
    let sample_size = u32::from_le_bytes([strh[44], strh[45], strh[46], strh[47]]);

    let (media_type, codec_id, params, suffix) = match &fcc_type {
        b"vids" => {
            let bmih = if !strf.is_empty() {
                Some(parse_bitmap_info_header(strf)?)
            } else {
                None
            };
            let compression = bmih.as_ref().map(|b| b.compression).unwrap_or(fcc_handler);
            let codec_id = video_codec_id(&compression);
            let mut p = CodecParameters::video(codec_id.clone());
            if let Some(b) = &bmih {
                p.width = Some(b.width);
                p.height = Some(b.height);
                p.extradata = b.extradata.clone();
            }
            // Frame rate from scale/rate (rate/scale = fps).
            p.frame_rate = Some(Rational::new(rate as i64, scale as i64));
            // MJPEG packets from AVI should be flagged as standalone JPEGs.
            let suffix = if codec_id.as_str() == "rgb24" {
                *b"db"
            } else {
                *b"dc"
            };
            (MediaType::Video, codec_id, p, suffix)
        }
        b"auds" => {
            let wfx = if !strf.is_empty() {
                Some(parse_waveformatex(strf)?)
            } else {
                None
            };
            let format_tag = wfx.as_ref().map(|w| w.format_tag).unwrap_or(0);
            let codec_id = audio_codec_id(format_tag);
            let mut p = CodecParameters::audio(codec_id.clone());
            if let Some(w) = &wfx {
                p.channels = Some(w.channels);
                p.sample_rate = Some(w.samples_per_sec);
                p.extradata = w.extradata.clone();
                p.sample_format = match (codec_id.as_str(), w.bits_per_sample) {
                    ("pcm_s16le", _) => Some(SampleFormat::S16),
                    (_, 16) => Some(SampleFormat::S16),
                    (_, 8) => Some(SampleFormat::U8),
                    _ => None,
                };
                p.bit_rate = if w.avg_bytes_per_sec > 0 {
                    Some(w.avg_bytes_per_sec as u64 * 8)
                } else {
                    None
                };
            }
            (MediaType::Audio, codec_id, p, *b"wb")
        }
        _ => {
            // "txts", "mids", "dats" — represent as data.
            let codec_id = CodecId::new(format!(
                "avi:{}",
                std::str::from_utf8(&fcc_type).unwrap_or("????")
            ));
            let mut p = CodecParameters::audio(codec_id.clone());
            p.media_type = MediaType::Data;
            (MediaType::Data, codec_id, p, *b"xx")
        }
    };

    let _ = codec_id; // absorbed into params

    // Stream time base. For video: scale/rate seconds per frame. For audio
    // at rate/scale samples per second, pick 1/samples_per_sec (standard
    // choice). For anything else, fall back to 1/rate.
    let time_base = match media_type {
        MediaType::Video => TimeBase::new(scale as i64, rate as i64),
        MediaType::Audio => {
            // rate/scale = samples_per_sec for PCM.
            TimeBase::new(scale as i64, rate as i64)
        }
        _ => TimeBase::new(scale as i64, rate as i64),
    };

    let duration = if length > 0 {
        Some(length as i64)
    } else {
        None
    };
    let stream = StreamInfo {
        index,
        time_base,
        duration,
        start_time: Some(0),
        params,
    };
    let _ = sample_size;
    Ok((stream, suffix))
}

fn read_body_bounded<R: std::io::Read + ?Sized>(r: &mut R, size: u32) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; size as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// --- Demuxer runtime ------------------------------------------------------

struct AviDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    /// For each stream, the expected 2-byte chunk-name suffix in `movi`.
    packet_chunk_suffix: Vec<[u8; 2]>,
    /// Absolute end-of-movi offset.
    movi_end: u64,
    /// Running packet counter per stream — used to synthesise PTS.
    per_stream_counter: Vec<u64>,
}

impl Demuxer for AviDemuxer {
    fn format_name(&self) -> &str {
        "avi"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.per_stream_counter.len() != self.streams.len() {
            self.per_stream_counter = vec![0u64; self.streams.len()];
        }
        loop {
            let pos = self.input.stream_position()?;
            if pos >= self.movi_end {
                return Err(Error::Eof);
            }
            let hdr = match read_chunk_header(&mut *self.input)? {
                Some(h) => h,
                None => return Err(Error::Eof),
            };
            // `LIST rec ` is an optional grouping inside movi — some writers
            // cluster chunks this way. Recurse by entering the list body.
            if hdr.id == LIST {
                let _form = read_form_type(&mut *self.input)?; // likely "rec "
                                                               // Continue: next iteration will consume its nested chunks.
                continue;
            }
            // End of movi guard in case sizes disagree.
            let body_end = self.input.stream_position()? + hdr.size as u64;
            if body_end > self.movi_end {
                // Truncated or bad size — stop.
                return Err(Error::Eof);
            }
            if hdr.id == *b"JUNK" || hdr.id == *b"junk" {
                skip_chunk(&mut *self.input, &hdr)?;
                continue;
            }
            // Payload chunk format: "NNsf" where NN is two ASCII digits and
            // sf ∈ {"dc","db","wb","pc","tx"}.
            if let Some(idx) = parse_stream_index(&hdr.id) {
                if (idx as usize) < self.streams.len() {
                    let expected = self.packet_chunk_suffix[idx as usize];
                    let suffix = [hdr.id[2], hdr.id[3]];
                    // Accept expected suffix; skip "pc" (palette change) and others.
                    let accept = suffix == expected
                        || suffix == *b"dc"
                        || suffix == *b"db"
                        || suffix == *b"wb";
                    if accept {
                        let data = read_body_bounded(&mut *self.input, hdr.size)?;
                        skip_pad(&mut *self.input, hdr.size)?;
                        let stream = &self.streams[idx as usize];
                        let counter = self.per_stream_counter[idx as usize];
                        // PTS: for video the counter is a frame index in the
                        // stream's time_base. For audio we advance by the
                        // number of samples in this packet (PCM: block_align
                        // derived from bps*channels; other codecs we just use
                        // the packet counter in units of rate/scale).
                        let pts = counter as i64;
                        let mut pkt = Packet::new(idx, stream.time_base, data);
                        pkt.pts = Some(pts);
                        pkt.dts = Some(pts);
                        pkt.flags.keyframe = true;
                        // Bump counter.
                        let bump = packet_time_delta(stream, pkt.data.len());
                        self.per_stream_counter[idx as usize] = counter + bump;
                        return Ok(pkt);
                    } else {
                        skip_chunk(&mut *self.input, &hdr)?;
                        continue;
                    }
                } else {
                    skip_chunk(&mut *self.input, &hdr)?;
                    continue;
                }
            }
            skip_chunk(&mut *self.input, &hdr)?;
        }
    }
}

/// Parse "NNsf" where NN is two ASCII digits into the stream index.
fn parse_stream_index(name: &[u8; 4]) -> Option<u32> {
    let h = ascii_hex(name[0])?;
    let l = ascii_hex(name[1])?;
    Some((h as u32) * 16 + l as u32)
}

/// Decode a single ASCII hex digit (0-9, a-f, A-F).
fn ascii_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn packet_time_delta(stream: &StreamInfo, payload_len: usize) -> u64 {
    match stream.params.media_type {
        MediaType::Video => 1,
        MediaType::Audio => {
            // PCM: duration = frames = payload / block_align. Non-PCM: one
            // tick per packet is a reasonable fallback.
            let block_align = stream
                .params
                .channels
                .zip(stream.params.sample_format)
                .map(|(c, f)| (c as usize) * f.bytes_per_sample())
                .filter(|&v| v > 0)
                .unwrap_or(0);
            if block_align > 0 {
                (payload_len / block_align) as u64
            } else {
                1
            }
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_index_parses() {
        assert_eq!(parse_stream_index(b"00dc"), Some(0));
        assert_eq!(parse_stream_index(b"01wb"), Some(1));
        assert_eq!(parse_stream_index(b"0adb"), Some(10));
        assert_eq!(parse_stream_index(b"XXXX"), None);
    }
}
