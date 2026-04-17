//! FORM 8SVX — Amiga 8-bit sampled voice audio.
//!
//! Layout: `FORM` group chunk → 4-byte `8SVX` form type → children:
//! - `VHDR` (20 bytes): voice header (one-shot/repeat sample counts,
//!   samples per high-cycle, samples per second, octave count, compression
//!   code, 16.16 volume).
//! - optional `NAME`, `ANNO`, `AUTH`, `(c) `, `CHAN`, `ATAK`, `RLSE`.
//! - `BODY`: raw signed 8-bit samples (or Fibonacci-delta compressed).
//!
//! We expose an 8SVX file as a single audio stream with codec id
//! `pcm_s8` and emit the BODY bytes in reasonably-sized packets. Only
//! uncompressed (`sCompression = 0`), mono (single `CHAN`/no CHAN) is
//! supported today; Fibonacci-delta decoding and the CHAN=6 stereo layout
//! are straightforward follow-ups.

use std::io::{Read, Seek, SeekFrom, Write};

use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::chunk::{
    read_body, read_chunk_header, read_form_type, skip_chunk_body, ChunkHeader, GROUP_FORM,
};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("iff_8svx", open);
    reg.register_muxer("iff_8svx", open_muxer);
    reg.register_extension("8svx", "iff_8svx");
    reg.register_extension("iff", "iff_8svx");
    reg.register_probe("iff_8svx", probe);
}

/// `FORM....8SVX` — IFF group chunk with the 8SVX form type.
fn probe(p: &oxideav_container::ProbeData) -> u8 {
    if p.buf.len() >= 12 && &p.buf[0..4] == b"FORM" && &p.buf[8..12] == b"8SVX" {
        100
    } else {
        0
    }
}

// --- VHDR parsing ---------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // VHDR holds metadata that's informational for now
struct Vhdr {
    one_shot_hi_samples: u32,
    repeat_hi_samples: u32,
    samples_per_hi_cycle: u32,
    samples_per_sec: u16,
    ct_octave: u8,
    compression: u8,
    volume_fixed: u32,
}

fn parse_vhdr(body: &[u8]) -> Result<Vhdr> {
    if body.len() < 20 {
        return Err(Error::invalid("8SVX VHDR: need 20 bytes"));
    }
    Ok(Vhdr {
        one_shot_hi_samples: u32::from_be_bytes([body[0], body[1], body[2], body[3]]),
        repeat_hi_samples: u32::from_be_bytes([body[4], body[5], body[6], body[7]]),
        samples_per_hi_cycle: u32::from_be_bytes([body[8], body[9], body[10], body[11]]),
        samples_per_sec: u16::from_be_bytes([body[12], body[13]]),
        ct_octave: body[14],
        compression: body[15],
        volume_fixed: u32::from_be_bytes([body[16], body[17], body[18], body[19]]),
    })
}

// --- Demuxer --------------------------------------------------------------

fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    // Outer FORM.
    let hdr = read_chunk_header(&mut *input)?.ok_or_else(|| Error::invalid("8SVX: empty file"))?;
    if hdr.id != GROUP_FORM {
        return Err(Error::invalid(format!(
            "8SVX: expected FORM chunk, got {}",
            hdr.id_str()
        )));
    }
    let form_type = read_form_type(&mut *input)?;
    if &form_type != b"8SVX" {
        return Err(Error::invalid(format!(
            "IFF: not an 8SVX file (form type {:?})",
            std::str::from_utf8(&form_type).unwrap_or("????")
        )));
    }
    // hdr.size counts FORM-type + children bytes; body length = hdr.size - 4.
    let body_limit = input.stream_position()? + hdr.size as u64 - 4;

    let mut vhdr: Option<Vhdr> = None;
    let mut channels: u16 = 1;
    let mut body_offset: u64 = 0;
    let mut body_size: u64 = 0;
    let mut metadata: Vec<(String, String)> = Vec::new();

    while input.stream_position()? < body_limit {
        let c = match read_chunk_header(&mut *input)? {
            Some(c) => c,
            None => break,
        };
        match &c.id {
            b"VHDR" => {
                let body = read_body(&mut *input, &c)?;
                vhdr = Some(parse_vhdr(&body)?);
                pad_after(&mut *input, &c)?;
            }
            b"CHAN" => {
                // CHAN payload: 4 bytes BE. 2 = left, 4 = right, 6 = stereo.
                let body = read_body(&mut *input, &c)?;
                if body.len() >= 4 {
                    let v = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    channels = if v == 6 { 2 } else { 1 };
                }
                pad_after(&mut *input, &c)?;
            }
            b"NAME" | b"AUTH" | b"ANNO" | b"(c) " | b"CHRS" => {
                let body = read_body(&mut *input, &c)?;
                let key = match &c.id {
                    b"NAME" => "title",
                    b"AUTH" => "artist",
                    b"ANNO" => "comment",
                    b"(c) " => "copyright",
                    b"CHRS" => "characters",
                    _ => unreachable!(),
                };
                let end = body.iter().position(|&b| b == 0).unwrap_or(body.len());
                let value = String::from_utf8_lossy(&body[..end]).trim().to_string();
                if !value.is_empty() {
                    metadata.push((key.into(), value));
                }
                pad_after(&mut *input, &c)?;
            }
            b"BODY" => {
                body_offset = input.stream_position()?;
                body_size = c.size as u64;
                break;
            }
            _ => skip_chunk_body(&mut *input, &c)?,
        }
    }

    let vhdr = vhdr.ok_or_else(|| Error::invalid("8SVX: missing VHDR chunk"))?;
    if vhdr.compression != 0 {
        return Err(Error::unsupported(format!(
            "8SVX: compression {} (Fibonacci-delta / other) not yet implemented",
            vhdr.compression
        )));
    }
    if body_size == 0 {
        return Err(Error::invalid("8SVX: missing BODY chunk"));
    }

    let sample_rate = vhdr.samples_per_sec as u32;
    let time_base = TimeBase::new(1, sample_rate as i64);
    let bytes_per_frame = channels as u64;
    let total_frames = body_size / bytes_per_frame;

    let mut params = CodecParameters::audio(CodecId::new("pcm_s8"));
    params.media_type = MediaType::Audio;
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S8);
    params.bit_rate = Some(8 * channels as u64 * sample_rate as u64);

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_frames as i64),
        start_time: Some(0),
        params,
    };

    let duration_micros: i64 = if sample_rate > 0 {
        (total_frames as i128 * 1_000_000 / sample_rate as i128) as i64
    } else {
        0
    };

    input.seek(SeekFrom::Start(body_offset))?;
    Ok(Box::new(SvxDemuxer {
        input,
        streams: vec![stream],
        body_end: body_offset + body_size,
        cursor: body_offset,
        channels,
        frames_emitted: 0,
        metadata,
        duration_micros,
    }))
}

fn pad_after<R: Seek + ?Sized>(r: &mut R, c: &ChunkHeader) -> Result<()> {
    if c.size & 1 == 1 {
        r.seek(SeekFrom::Current(1))?;
    }
    Ok(())
}

struct SvxDemuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    body_end: u64,
    cursor: u64,
    channels: u16,
    frames_emitted: i64,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
}

const CHUNK_FRAMES: u64 = 4096;

impl Demuxer for SvxDemuxer {
    fn format_name(&self) -> &str {
        "iff_8svx"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.cursor >= self.body_end {
            return Err(Error::Eof);
        }
        let bytes_per_frame = self.channels as u64;
        let remaining = self.body_end - self.cursor;
        let want_bytes = (CHUNK_FRAMES * bytes_per_frame).min(remaining);
        let want_bytes = (want_bytes / bytes_per_frame) * bytes_per_frame;
        if want_bytes == 0 {
            return Err(Error::Eof);
        }

        self.input.seek(SeekFrom::Start(self.cursor))?;
        let mut buf = vec![0u8; want_bytes as usize];
        self.input.read_exact(&mut buf)?;
        self.cursor += want_bytes;

        let stream = &self.streams[0];
        let frames = want_bytes / bytes_per_frame;
        let pts = self.frames_emitted;
        self.frames_emitted += frames as i64;

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

// --- Muxer ---------------------------------------------------------------

/// Open a muxer through the [`ContainerRegistry`] with no container-level
/// metadata. For callers that need to write `NAME` / `AUTH` / `ANNO` /
/// `CHRS` chunks, construct [`SvxMuxer`] directly via
/// [`SvxMuxer::with_metadata`] — the `Muxer` trait doesn't currently carry
/// metadata through its opening hook.
fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(SvxMuxer::new(output, streams)?))
}

/// 8SVX container muxer. Wraps one stream of 8-bit signed PCM
/// (`pcm_s8` / [`SampleFormat::S8`]) in an IFF FORM/8SVX tree:
/// `VHDR` (20 bytes) + optional string metadata + `BODY` (the raw samples).
///
/// Construct via [`SvxMuxer::new`] for a bare voice, or
/// [`SvxMuxer::with_metadata`] to attach `NAME` / `AUTH` / `ANNO` / `CHRS`
/// chunks. `(c) ` (copyright) is **not** emitted — the FourCC's trailing
/// space and the demuxer's ASCII-trim make arbitrary UTF-8 copyright
/// strings awkward to round-trip, so we stay out of that chunk for now.
pub struct SvxMuxer {
    output: Box<dyn WriteSeek>,
    channels: u16,
    sample_rate: u32,
    /// Ordered (key, value) pairs. Recognised keys: `title` → `NAME`,
    /// `artist` → `AUTH`, `comment` → `ANNO`, `characters` → `CHRS`.
    metadata: Vec<(String, String)>,
    form_size_offset: u64,
    body_size_offset: u64,
    body_bytes: u64,
    header_written: bool,
    trailer_written: bool,
}

impl SvxMuxer {
    /// Build a muxer that only writes VHDR + BODY (no string chunks).
    pub fn new(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Self> {
        Self::with_metadata(output, streams, &[])
    }

    /// Build a muxer with container-level metadata. Only recognised keys
    /// are emitted; unknown keys are silently dropped. Values are written
    /// as NUL-terminated ASCII-ish text (non-ASCII passes through as raw
    /// bytes — the demuxer reads UTF-8 with lossy fallback).
    pub fn with_metadata(
        output: Box<dyn WriteSeek>,
        streams: &[StreamInfo],
        metadata: &[(String, String)],
    ) -> Result<Self> {
        if streams.len() != 1 {
            return Err(Error::unsupported("8SVX supports exactly one audio stream"));
        }
        let s = &streams[0];
        if s.params.media_type != MediaType::Audio {
            return Err(Error::invalid("8SVX stream must be audio"));
        }
        if s.params.codec_id != CodecId::new("pcm_s8") {
            return Err(Error::unsupported(format!(
                "8SVX muxer only accepts pcm_s8 (got {})",
                s.params.codec_id
            )));
        }
        if let Some(fmt) = s.params.sample_format {
            if fmt != SampleFormat::S8 {
                return Err(Error::unsupported(format!(
                    "8SVX muxer requires SampleFormat::S8 (got {:?})",
                    fmt
                )));
            }
        }
        let channels = s
            .params
            .channels
            .ok_or_else(|| Error::invalid("8SVX muxer: missing channels"))?;
        if channels != 1 {
            return Err(Error::unsupported(format!(
                "8SVX muxer: only mono is supported today (got {} channels)",
                channels
            )));
        }
        let sample_rate = s
            .params
            .sample_rate
            .ok_or_else(|| Error::invalid("8SVX muxer: missing sample rate"))?;
        if sample_rate > u16::MAX as u32 {
            return Err(Error::unsupported(format!(
                "8SVX VHDR.samplesPerSec is u16; {} Hz exceeds the range",
                sample_rate
            )));
        }
        Ok(Self {
            output,
            channels,
            sample_rate,
            metadata: metadata.to_vec(),
            form_size_offset: 0,
            body_size_offset: 0,
            body_bytes: 0,
            header_written: false,
            trailer_written: false,
        })
    }
}

/// Map a metadata key to its 8SVX FourCC. Unknown keys return `None`
/// and are dropped by the muxer.
fn metadata_fourcc(key: &str) -> Option<&'static [u8; 4]> {
    match key {
        "title" => Some(b"NAME"),
        "artist" => Some(b"AUTH"),
        "comment" => Some(b"ANNO"),
        "characters" => Some(b"CHRS"),
        _ => None,
    }
}

impl Muxer for SvxMuxer {
    fn format_name(&self) -> &str {
        "iff_8svx"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("8SVX muxer: write_header called twice"));
        }
        // FORM group chunk header. Size is patched in write_trailer once
        // we know how much we wrote.
        self.output.write_all(b"FORM")?;
        self.form_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_be_bytes())?; // placeholder
        self.output.write_all(b"8SVX")?;

        // VHDR (20 bytes). We synthesise a one-shot voice with no
        // sustain/loop and no upper octaves: oneShotHiSamples is the
        // total frame count (or 0 when the stream duration is unknown —
        // FORM sizes are patched at close anyway), repeatHiSamples = 0,
        // samplesPerHiCycle = 0, volume = 1.0 (0x00010000, 16.16 fixed).
        self.output.write_all(b"VHDR")?;
        self.output.write_all(&20u32.to_be_bytes())?;
        // Frame count isn't known yet; patched in write_trailer.
        self.output.write_all(&0u32.to_be_bytes())?; // oneShotHiSamples
        self.output.write_all(&0u32.to_be_bytes())?; // repeatHiSamples
        self.output.write_all(&0u32.to_be_bytes())?; // samplesPerHiCycle
        self.output
            .write_all(&(self.sample_rate as u16).to_be_bytes())?;
        self.output.write_all(&[1u8])?; // ctOctave
        self.output.write_all(&[0u8])?; // sCompression (none)
        self.output.write_all(&0x0001_0000u32.to_be_bytes())?; // volume 1.0

        // Optional metadata chunks. Preserve caller-supplied order so
        // round-trips are stable. The demuxer strips trailing NULs, so
        // we always NUL-terminate and pad to even length.
        for (k, v) in &self.metadata {
            let Some(fourcc) = metadata_fourcc(k) else {
                continue;
            };
            let bytes = v.as_bytes();
            // NUL-terminate: the demuxer splits on the first NUL.
            let mut payload = Vec::with_capacity(bytes.len() + 1);
            payload.extend_from_slice(bytes);
            payload.push(0);
            let size = payload.len() as u32;
            self.output.write_all(fourcc)?;
            self.output.write_all(&size.to_be_bytes())?;
            self.output.write_all(&payload)?;
            if size & 1 == 1 {
                self.output.write_all(&[0u8])?; // IFF pad byte
            }
        }

        // BODY chunk header; body size is patched in write_trailer.
        self.output.write_all(b"BODY")?;
        self.body_size_offset = self.output.stream_position()?;
        self.output.write_all(&0u32.to_be_bytes())?; // placeholder

        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("8SVX muxer: write_header not called"));
        }
        if self.trailer_written {
            return Err(Error::other("8SVX muxer: write_packet after trailer"));
        }
        // Payload is raw 8-bit signed PCM — one byte per mono frame.
        self.output.write_all(&packet.data)?;
        self.body_bytes += packet.data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        if !self.header_written {
            return Err(Error::other("8SVX muxer: write_header not called"));
        }
        // IFF chunks pad to even length; BODY is the last child chunk so
        // its pad byte (if any) also pads the enclosing FORM.
        let body_pad = (self.body_bytes & 1) as u64;
        if body_pad == 1 {
            self.output.write_all(&[0u8])?;
        }
        let end = self.output.stream_position()?;

        // Patch BODY chunk size.
        let body_size_u32: u32 = self
            .body_bytes
            .try_into()
            .map_err(|_| Error::other("8SVX BODY chunk exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.body_size_offset))?;
        self.output.write_all(&body_size_u32.to_be_bytes())?;

        // Patch VHDR.oneShotHiSamples with the total frame count
        // (mono, 1 byte per frame). `form_size_offset` points at the
        // FORM size field (4 bytes), then comes "8SVX" (4), "VHDR" (4),
        // VHDR size (4) — so oneShotHiSamples lives at
        // form_size_offset + 16. Writing this lets a decoder that
        // inspects VHDR know the full length of the voice even before
        // reaching BODY.
        let one_shot = (self.body_bytes / self.channels as u64) as u32;
        self.output
            .seek(SeekFrom::Start(self.form_size_offset + 16))?;
        self.output.write_all(&one_shot.to_be_bytes())?;

        // Patch FORM size: everything after the 8-byte FORM header.
        let form_size_u32: u32 = (end - (self.form_size_offset + 4))
            .try_into()
            .map_err(|_| Error::other("8SVX FORM size exceeds 4 GiB"))?;
        self.output.seek(SeekFrom::Start(self.form_size_offset))?;
        self.output.write_all(&form_size_u32.to_be_bytes())?;

        self.output.seek(SeekFrom::Start(end))?;
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Hand-craft a tiny 8SVX file: FORM 8SVX { VHDR, BODY = 10 signed bytes }.
    fn make_fixture() -> Vec<u8> {
        let mut out = Vec::new();
        // FORM header: ID + size (filled in below) + form type
        out.extend_from_slice(b"FORM");
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"8SVX");

        // VHDR (20 bytes)
        out.extend_from_slice(b"VHDR");
        out.extend_from_slice(&20u32.to_be_bytes());
        out.extend_from_slice(&10u32.to_be_bytes()); // oneShotHiSamples
        out.extend_from_slice(&0u32.to_be_bytes()); // repeatHiSamples
        out.extend_from_slice(&0u32.to_be_bytes()); // samplesPerHiCycle
        out.extend_from_slice(&8000u16.to_be_bytes()); // samplesPerSec
        out.push(1); // ctOctave
        out.push(0); // sCompression (none)
        out.extend_from_slice(&0x10000u32.to_be_bytes()); // volume = 1.0

        // BODY: 10 signed 8-bit samples (pad to even: 10 is even, no pad)
        out.extend_from_slice(b"BODY");
        out.extend_from_slice(&10u32.to_be_bytes());
        let samples: [i8; 10] = [0, 16, 32, 48, 64, 48, 32, 16, 0, -16];
        for s in &samples {
            out.push(*s as u8);
        }

        // Patch FORM size = total - 8 (ID + size field).
        let total = out.len() as u32;
        out[4..8].copy_from_slice(&(total - 8).to_be_bytes());
        out
    }

    #[test]
    fn demux_minimal_8svx() {
        let bytes = make_fixture();
        let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
        let mut dmx = open(rs).unwrap();
        assert_eq!(dmx.format_name(), "iff_8svx");
        let s = &dmx.streams()[0];
        assert_eq!(s.params.codec_id.as_str(), "pcm_s8");
        assert_eq!(s.params.channels, Some(1));
        assert_eq!(s.params.sample_rate, Some(8000));

        let pkt = dmx.next_packet().unwrap();
        assert_eq!(pkt.data.len(), 10);
        assert_eq!(pkt.data[0], 0);
        assert_eq!(pkt.data[9], 0xF0); // -16 as u8

        // End of stream.
        let err = dmx.next_packet().unwrap_err();
        assert!(matches!(err, Error::Eof));
    }
}
