//! Asterisk-style headerless signed-linear PCM container (`.sln` / `.slin*`).
//!
//! An `.sln` file is the degenerate container: the entire file body is raw
//! interleaved 16-bit little-endian PCM at a sample rate implied by the
//! extension. No header, no magic, no trailer. Mono only in Asterisk's own
//! usage, and that is what this demuxer assumes.
//!
//! Extension → sample rate (the Asterisk convention):
//!
//! | extension                 | sample rate |
//! |---------------------------|-------------|
//! | `sln` / `slin`            |  8_000 Hz   |
//! | `sln8` / `slin8`          |  8_000 Hz   |
//! | `sln12` / `slin12`        | 12_000 Hz   |
//! | `sln16` / `slin16`        | 16_000 Hz   |
//! | `sln24` / `slin24`        | 24_000 Hz   |
//! | `sln32` / `slin32`        | 32_000 Hz   |
//! | `sln44` / `slin44`        | 44_100 Hz   |
//! | `sln48` / `slin48`        | 48_000 Hz   |
//! | `sln96` / `slin96`        | 96_000 Hz   |
//! | `sln192` / `slin192`      | 192_000 Hz  |
//!
//! The muxer writes raw S16LE bytes verbatim with no framing.

use oxideav_container::{
    ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek, PROBE_SCORE_EXTENSION,
};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};
use std::io::{Read, Seek, SeekFrom, Write};

/// `(extension_without_dot, sample_rate_hz)`. The demuxer uses this table
/// to infer the sample rate; the container registry uses it to map every
/// listed extension to the `"slin"` container name.
pub(crate) const EXTENSIONS: &[(&str, u32)] = &[
    ("sln", 8_000),
    ("slin", 8_000),
    ("sln8", 8_000),
    ("slin8", 8_000),
    ("sln12", 12_000),
    ("slin12", 12_000),
    ("sln16", 16_000),
    ("slin16", 16_000),
    ("sln24", 24_000),
    ("slin24", 24_000),
    ("sln32", 32_000),
    ("slin32", 32_000),
    ("sln44", 44_100),
    ("slin44", 44_100),
    ("sln48", 48_000),
    ("slin48", 48_000),
    ("sln96", 96_000),
    ("slin96", 96_000),
    ("sln192", 192_000),
    ("slin192", 192_000),
];

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("slin", open_demuxer);
    reg.register_muxer("slin", open_muxer);
    for (ext, _) in EXTENSIONS {
        reg.register_extension(ext, "slin");
    }
    reg.register_probe("slin", probe);
}

/// Raw PCM has no magic bytes, so this probe is only able to fire on the
/// file extension. Returns [`PROBE_SCORE_EXTENSION`] (25) when an
/// `.sln*` / `.slin*` extension is supplied and `0` otherwise — a weak
/// hint that any real container probe will outrank.
pub fn probe(p: &ProbeData) -> u8 {
    let Some(ext) = p.ext else {
        return 0;
    };
    if sample_rate_for_ext(ext).is_some() {
        PROBE_SCORE_EXTENSION
    } else {
        0
    }
}

/// Look up the sample rate implied by a bare extension (no leading dot,
/// case-insensitive).
fn sample_rate_for_ext(ext: &str) -> Option<u32> {
    let lower = ext.to_ascii_lowercase();
    EXTENSIONS
        .iter()
        .find_map(|(e, sr)| if *e == lower { Some(*sr) } else { None })
}

// --- Demuxer ---------------------------------------------------------------

/// Default: 8 kHz — matches plain `.sln` / `.slin` (the Asterisk 1990s
/// default). Callers that know better should open via an explicit
/// sample-rate-bearing extension.
const DEFAULT_SAMPLE_RATE: u32 = 8_000;

fn open_demuxer(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    open_demuxer_with_rate(input, DEFAULT_SAMPLE_RATE)
}

/// Programmatic entry point: open a `.sln*` stream at an explicit sample
/// rate. Useful when the caller demuxes from memory and cannot supply a
/// filename hint through the registry's extension lookup.
pub fn open_demuxer_with_rate(
    mut input: Box<dyn ReadSeek>,
    sample_rate: u32,
) -> Result<Box<dyn Demuxer>> {
    if sample_rate == 0 {
        return Err(Error::invalid("slin demuxer: sample_rate must be > 0"));
    }
    // Determine total size for duration reporting without consuming the
    // stream — non-seekable inputs will just get an empty duration.
    let start = input.stream_position()?;
    let end = input.seek(SeekFrom::End(0))?;
    input.seek(SeekFrom::Start(start))?;
    let total_bytes = end.saturating_sub(start);

    let channels: u16 = 1;
    let block_align: u64 = (SampleFormat::S16.bytes_per_sample() as u64) * channels as u64;
    let total_samples = total_bytes / block_align;
    let duration_micros: i64 = if sample_rate > 0 {
        (total_samples as i128 * 1_000_000 / sample_rate as i128) as i64
    } else {
        0
    };

    let codec_id = CodecId::new("pcm_s16le");
    let mut params = CodecParameters::audio(codec_id);
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = Some(
        (SampleFormat::S16.bytes_per_sample() as u64) * 8 * (channels as u64) * sample_rate as u64,
    );

    let time_base = TimeBase::new(1, sample_rate as i64);
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_samples as i64),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(SlinDemuxer {
        input,
        streams: vec![stream],
        data_start: start,
        data_end: end,
        cursor: start,
        block_align,
        chunk_frames: 1024,
        samples_emitted: 0,
        duration_micros,
    }))
}

struct SlinDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    #[allow(dead_code)]
    data_start: u64,
    data_end: u64,
    cursor: u64,
    block_align: u64,
    chunk_frames: u64,
    samples_emitted: i64,
    duration_micros: i64,
}

impl Demuxer for SlinDemuxer {
    fn format_name(&self) -> &str {
        "slin"
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
            // Trailing partial frame — drop it; raw PCM has no framing.
            return Err(Error::Eof);
        }

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
        return Err(Error::unsupported("slin supports exactly one audio stream"));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("slin stream must be audio"));
    }
    let fmt = s
        .params
        .sample_format
        .or_else(|| super::pcm::sample_format_for(&s.params.codec_id))
        .ok_or_else(|| Error::invalid("slin muxer: unknown sample format"))?;
    if fmt != SampleFormat::S16 {
        return Err(Error::unsupported(
            "slin muxer requires S16LE samples (pcm_s16le or slin* codec id)",
        ));
    }
    Ok(Box::new(SlinMuxer {
        output,
        header_written: false,
        trailer_written: false,
    }))
}

struct SlinMuxer {
    output: Box<dyn WriteSeek>,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for SlinMuxer {
    fn format_name(&self) -> &str {
        "slin"
    }

    fn write_header(&mut self) -> Result<()> {
        // No-op: slin is headerless. Tracked only so that ordering errors
        // (write_packet before write_header) stay symmetric with other
        // muxers in this crate.
        if self.header_written {
            return Err(Error::other("slin header already written"));
        }
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("slin muxer: write_header not called"));
        }
        self.output.write_all(&packet.data)?;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}
