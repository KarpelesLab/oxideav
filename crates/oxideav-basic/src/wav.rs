//! Minimal pure-Rust RIFF/WAVE container.
//!
//! Supports reading and writing linear PCM streams via the `pcm_*` codecs.

use oxideav_codec as _;
use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};
use std::io::{Read, Seek, SeekFrom, Write};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("wav", open_demuxer);
    reg.register_muxer("wav", open_muxer);
    reg.register_extension("wav", "wav");
    reg.register_extension("wave", "wav");
}

const FMT_PCM: u16 = 0x0001;
const FMT_IEEE_FLOAT: u16 = 0x0003;
const FMT_EXTENSIBLE: u16 = 0xFFFE;

// Data GUIDs for WAVE_FORMAT_EXTENSIBLE subformats.
const GUID_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const GUID_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut hdr = [0u8; 12];
    input.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err(Error::invalid("not a RIFF/WAVE file"));
    }

    // Walk chunks until we hit "data"; parse "fmt " and "LIST" along the way.
    let mut fmt: Option<WaveFmt> = None;
    let mut metadata: Vec<(String, String)> = Vec::new();
    let data_offset: u64;
    let data_size: u64;
    loop {
        let mut chdr = [0u8; 8];
        input.read_exact(&mut chdr)?;
        let id = &chdr[0..4];
        let size = u32::from_le_bytes([chdr[4], chdr[5], chdr[6], chdr[7]]) as u64;
        match id {
            b"fmt " => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                fmt = Some(parse_fmt(&buf)?);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"LIST" => {
                let mut buf = vec![0u8; size as usize];
                input.read_exact(&mut buf)?;
                parse_list_chunk(&buf, &mut metadata);
                if size % 2 == 1 {
                    input.seek(SeekFrom::Current(1))?;
                }
            }
            b"data" => {
                data_offset = input.stream_position()?;
                data_size = size;
                break;
            }
            _ => {
                let pad = size + (size % 2);
                input.seek(SeekFrom::Current(pad as i64))?;
            }
        }
    }
    let fmt = fmt.ok_or_else(|| Error::invalid("WAV missing fmt chunk"))?;

    let codec_id = resolve_codec(&fmt)?;
    let sample_fmt = super::pcm::sample_format_for(&codec_id)
        .ok_or_else(|| Error::unsupported(format!("unsupported WAV codec {}", codec_id)))?;

    let time_base = TimeBase::new(1, fmt.sample_rate as i64);
    let block_align = fmt.block_align.max(1) as u64;
    let total_samples = data_size / block_align;
    let duration_micros: i64 = if fmt.sample_rate > 0 {
        (total_samples as i128 * 1_000_000 / fmt.sample_rate as i128) as i64
    } else {
        0
    };

    let mut params = CodecParameters::audio(codec_id);
    params.channels = Some(fmt.channels);
    params.sample_rate = Some(fmt.sample_rate);
    params.sample_format = Some(sample_fmt);
    params.bit_rate = Some(
        (sample_fmt.bytes_per_sample() as u64)
            * 8
            * (fmt.channels as u64)
            * (fmt.sample_rate as u64),
    );

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_samples as i64),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(WavDemuxer {
        input,
        streams: vec![stream],
        data_end: data_offset + data_size,
        cursor: data_offset,
        block_align,
        chunk_frames: 1024,
        samples_emitted: 0,
        metadata,
        duration_micros,
    }))
}

/// Parse a RIFF LIST chunk body. If the list type is `INFO`, map its
/// `IART`/`INAM`/... subchunks to standard key names (`artist`, `title`,
/// …) and append to `out`.
fn parse_list_chunk(buf: &[u8], out: &mut Vec<(String, String)>) {
    if buf.len() < 4 {
        return;
    }
    if &buf[0..4] != b"INFO" {
        return;
    }
    let mut i = 4usize;
    while i + 8 <= buf.len() {
        let id: [u8; 4] = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
        let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        i += 8;
        if i + size > buf.len() {
            break;
        }
        let raw = &buf[i..i + size];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let value = String::from_utf8_lossy(&raw[..end]).trim().to_string();
        let key = info_id_to_key(&id);
        if !value.is_empty() {
            if let Some(k) = key {
                out.push((k.to_string(), value));
            }
        }
        i += size;
        if size % 2 == 1 {
            i += 1;
        }
    }
}

fn info_id_to_key(id: &[u8; 4]) -> Option<&'static str> {
    match id {
        b"INAM" => Some("title"),
        b"IART" => Some("artist"),
        b"IPRD" => Some("album"),
        b"ICMT" => Some("comment"),
        b"ICRD" => Some("date"),
        b"IGNR" => Some("genre"),
        b"ICOP" => Some("copyright"),
        b"IENG" => Some("engineer"),
        b"ITCH" => Some("technician"),
        b"ISFT" => Some("encoder"),
        b"ISBJ" => Some("subject"),
        b"ITRK" => Some("track"),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct WaveFmt {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    #[allow(dead_code)]
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    subformat: Option<[u8; 16]>,
}

fn parse_fmt(buf: &[u8]) -> Result<WaveFmt> {
    if buf.len() < 16 {
        return Err(Error::invalid("fmt chunk too small"));
    }
    let format_tag = u16::from_le_bytes([buf[0], buf[1]]);
    let channels = u16::from_le_bytes([buf[2], buf[3]]);
    let sample_rate = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let byte_rate = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let block_align = u16::from_le_bytes([buf[12], buf[13]]);
    let bits_per_sample = u16::from_le_bytes([buf[14], buf[15]]);
    let mut subformat = None;
    if format_tag == FMT_EXTENSIBLE && buf.len() >= 40 {
        let mut g = [0u8; 16];
        g.copy_from_slice(&buf[24..40]);
        subformat = Some(g);
    }
    Ok(WaveFmt {
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        subformat,
    })
}

fn resolve_codec(fmt: &WaveFmt) -> Result<CodecId> {
    let (is_float, bits) = match fmt.format_tag {
        FMT_PCM => (false, fmt.bits_per_sample),
        FMT_IEEE_FLOAT => (true, fmt.bits_per_sample),
        FMT_EXTENSIBLE => {
            let sub = fmt
                .subformat
                .ok_or_else(|| Error::invalid("extensible WAV missing subformat"))?;
            let is_float = match sub {
                GUID_PCM => false,
                GUID_IEEE_FLOAT => true,
                _ => {
                    return Err(Error::unsupported(
                        "unsupported WAVE_FORMAT_EXTENSIBLE subformat",
                    ));
                }
            };
            (is_float, fmt.bits_per_sample)
        }
        other => {
            return Err(Error::unsupported(format!(
                "unsupported WAV format tag 0x{:04x}",
                other
            )));
        }
    };
    let name = match (is_float, bits) {
        (false, 8) => "pcm_u8",
        (false, 16) => "pcm_s16le",
        (false, 24) => "pcm_s24le",
        (false, 32) => "pcm_s32le",
        (true, 32) => "pcm_f32le",
        (true, 64) => "pcm_f64le",
        (f, b) => {
            return Err(Error::unsupported(format!(
                "unsupported WAV bit depth: float={} bits={}",
                f, b
            )));
        }
    };
    Ok(CodecId::new(name))
}

struct WavDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    data_end: u64,
    cursor: u64,
    block_align: u64,
    chunk_frames: u64,
    samples_emitted: i64,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
}

impl Demuxer for WavDemuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.cursor >= self.data_end {
            return Err(Error::Eof);
        }
        let remaining = self.data_end - self.cursor;
        let want_bytes = (self.chunk_frames * self.block_align).min(remaining);
        let want_bytes = (want_bytes / self.block_align) * self.block_align;
        if want_bytes == 0 {
            return Err(Error::Eof);
        }

        // Ensure we're positioned correctly (if an upstream operation seeked us).
        self.input.seek(SeekFrom::Start(self.cursor))?;
        let mut buf = vec![0u8; want_bytes as usize];
        self.input.read_exact(&mut buf)?;
        self.cursor += want_bytes;

        let stream = &self.streams[0];
        let frames = want_bytes / self.block_align;
        let pts = self.samples_emitted;
        self.samples_emitted += frames as i64;

        let mut pkt = Packet::new(0, stream.time_base, buf);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(frames as i64);
        pkt.flags.keyframe = true;
        Ok(pkt)
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
}

// --- Muxer -----------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported("WAV supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("WAV stream must be audio"));
    }
    let fmt = sample_format_for_params(&s.params)?;
    let channels = s
        .params
        .channels
        .ok_or_else(|| Error::invalid("WAV muxer: missing channels"))?;
    let sample_rate = s
        .params
        .sample_rate
        .ok_or_else(|| Error::invalid("WAV muxer: missing sample rate"))?;
    Ok(Box::new(WavMuxer {
        output,
        channels,
        sample_rate,
        sample_format: fmt,
        riff_size_offset: 0,
        data_size_offset: 0,
        data_bytes: 0,
        header_written: false,
        trailer_written: false,
    }))
}

fn sample_format_for_params(p: &CodecParameters) -> Result<SampleFormat> {
    p.sample_format
        .or_else(|| super::pcm::sample_format_for(&p.codec_id))
        .ok_or_else(|| Error::unsupported(format!("WAV: unknown PCM codec {}", p.codec_id)))
}

struct WavMuxer {
    output: Box<dyn WriteSeek>,
    channels: u16,
    sample_rate: u32,
    sample_format: SampleFormat,
    riff_size_offset: u64,
    data_size_offset: u64,
    data_bytes: u64,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for WavMuxer {
    fn format_name(&self) -> &str {
        "wav"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("WAV header already written"));
        }
        let (format_tag, bits_per_sample) = match self.sample_format {
            SampleFormat::U8 => (FMT_PCM, 8u16),
            SampleFormat::S16 => (FMT_PCM, 16),
            SampleFormat::S24 => (FMT_PCM, 24),
            SampleFormat::S32 => (FMT_PCM, 32),
            SampleFormat::F32 => (FMT_IEEE_FLOAT, 32),
            SampleFormat::F64 => (FMT_IEEE_FLOAT, 64),
            other => {
                return Err(Error::unsupported(format!(
                    "WAV muxer cannot write sample format {:?}",
                    other
                )));
            }
        };
        let block_align = (bits_per_sample / 8) * self.channels;
        let byte_rate = self.sample_rate * block_align as u32;

        self.output.write_all(b"RIFF")?;
        self.riff_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder
        self.output.write_all(b"WAVE")?;

        self.output.write_all(b"fmt ")?;
        self.output.write_all(&16u32.to_le_bytes())?;
        self.output.write_all(&format_tag.to_le_bytes())?;
        self.output.write_all(&self.channels.to_le_bytes())?;
        self.output.write_all(&self.sample_rate.to_le_bytes())?;
        self.output.write_all(&byte_rate.to_le_bytes())?;
        self.output.write_all(&block_align.to_le_bytes())?;
        self.output.write_all(&bits_per_sample.to_le_bytes())?;

        self.output.write_all(b"data")?;
        self.data_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_le_bytes())?; // placeholder

        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("WAV muxer: write_header not called"));
        }
        self.output.write_all(&packet.data)?;
        self.data_bytes += packet.data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // Pad data chunk to even length.
        if self.data_bytes % 2 == 1 {
            self.output.write_all(&[0u8])?;
        }
        let end = self.output.stream_position()?;

        // Patch "data" chunk size.
        let data_size_u32: u32 = self
            .data_bytes
            .try_into()
            .map_err(|_| Error::other("WAV data chunk exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.data_size_offset))?;
        self.output.write_all(&data_size_u32.to_le_bytes())?;

        // Patch "RIFF" size: total file size minus 8 (RIFF + size fields).
        let riff_size_u32: u32 = (end - 8)
            .try_into()
            .map_err(|_| Error::other("WAV RIFF size exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.riff_size_offset))?;
        self.output.write_all(&riff_size_u32.to_le_bytes())?;

        self.output.seek(SeekFrom::Start(end))?;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecParameters, MediaType};

    fn make_stream(fmt: SampleFormat, ch: u16, sr: u32) -> StreamInfo {
        let mut params = CodecParameters::audio(super::super::pcm::codec_id_for(fmt).unwrap());
        params.media_type = MediaType::Audio;
        params.channels = Some(ch);
        params.sample_rate = Some(sr);
        params.sample_format = Some(fmt);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sr as i64),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    #[test]
    fn round_trip_s16_mono() {
        // Write then read back a small S16 mono WAV via the public demuxer/muxer paths.
        let samples: Vec<i16> = (0..1000).map(|i| ((i * 32) - 16000) as i16).collect();
        let mut payload = Vec::with_capacity(samples.len() * 2);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }

        // Mux to a temp file, then demux and compare.
        let stream = make_stream(SampleFormat::S16, 1, 48_000);
        let tmp = std::env::temp_dir().join("oxideav-basic-wav-test.wav");
        {
            let f = std::fs::File::create(&tmp).unwrap();
            let ws: Box<dyn WriteSeek> = Box::new(f);
            let mut mux = open_muxer(ws, std::slice::from_ref(&stream)).unwrap();
            mux.write_header().unwrap();
            let pkt = Packet::new(0, stream.time_base, payload.clone());
            mux.write_packet(&pkt).unwrap();
            mux.write_trailer().unwrap();
        }
        let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
        let mut dmx = open_demuxer(rs).unwrap();
        assert_eq!(dmx.format_name(), "wav");
        assert_eq!(dmx.streams().len(), 1);
        assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
        let mut out_bytes = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(p) => out_bytes.extend_from_slice(&p.data),
                Err(Error::Eof) => break,
                Err(e) => panic!("demux error: {e}"),
            }
        }
        assert_eq!(out_bytes, payload);
    }
}
