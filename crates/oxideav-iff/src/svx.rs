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

use std::io::{Read, Seek, SeekFrom};

use oxideav_container::{ContainerRegistry, Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::chunk::{
    read_body, read_chunk_header, read_form_type, skip_chunk_body, ChunkHeader, GROUP_FORM,
};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("iff_8svx", open);
    reg.register_extension("8svx", "iff_8svx");
    reg.register_extension("iff", "iff_8svx");
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
            b"BODY" => {
                body_offset = input.stream_position()?;
                body_size = c.size as u64;
                // Leave the read cursor at body start — demuxer state handles it.
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

    input.seek(SeekFrom::Start(body_offset))?;
    Ok(Box::new(SvxDemuxer {
        input,
        streams: vec![stream],
        body_end: body_offset + body_size,
        cursor: body_offset,
        channels,
        frames_emitted: 0,
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
