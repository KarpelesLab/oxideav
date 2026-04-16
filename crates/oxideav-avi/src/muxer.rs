//! AVI (RIFF/AVI) muxer.
//!
//! Output layout:
//! ```text
//! RIFF(AVI )
//!   LIST(hdrl)
//!     avih                  ← main header
//!     LIST(strl) × N
//!       strh                ← stream header
//!       strf                ← BITMAPINFOHEADER or WAVEFORMATEX
//!   LIST(movi)              ← packet chunks: NNdc / NNwb / NNdb
//!   idx1                    ← legacy index (written in write_trailer)
//! ```
//!
//! - The public `Muxer` API is codec-agnostic. The only codec-aware file is
//!   `codec_map::build_strf`, which errors with `Unsupported` at `open()` for
//!   codecs we don't package. `write_packet` never branches on codec.
//! - OpenDML (`ix##`, super-index, > 2 GiB) is out of scope; exceeding the
//!   32-bit RIFF size returns an error from `write_trailer`.

use std::io::{Seek, SeekFrom, Write};

use oxideav_container::{Muxer, WriteSeek};
use oxideav_core::{Error, Packet, Result, StreamInfo};

use crate::codec_map::{build_strf, StrfEntry};
use crate::riff::{begin_list, finish_chunk, write_chunk, AVI_FORM, LIST, RIFF};

/// Bookkeeping for a single idx1 entry (legacy AVI 1.0 index).
#[derive(Clone, Copy, Debug)]
struct IndexEntry {
    ckid: [u8; 4],
    flags: u32,
    /// Offset from the start of the `movi` list body (see `idx1` format note).
    offset: u32,
    size: u32,
}

struct TrackState {
    stream: StreamInfo,
    entry: StrfEntry,
    /// 4-byte chunk FourCC used in movi for this stream (e.g. b"00dc").
    packet_fourcc: [u8; 4],
    /// Running packet count (used for avih.TotalFrames for the first video
    /// stream and length fields).
    packet_count: u32,
    /// Running total sample count for audio (frames for PCM, packets for VBR).
    sample_count: u64,
    /// Max chunk size seen so far (for strh.dwSuggestedBufferSize).
    max_chunk_size: u32,
    /// Max output bytes per packet (used for ffmpeg compatibility).
    total_bytes: u64,
}

/// Factory registered with the container registry.
pub fn open(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.is_empty() {
        return Err(Error::invalid("avi muxer: need at least one stream"));
    }
    if streams.len() > 99 {
        // We use 2 ASCII *decimal* digits 00..99 for the chunk index.
        return Err(Error::unsupported(
            "avi muxer: > 99 streams not supported in legacy index",
        ));
    }
    let mut tracks = Vec::with_capacity(streams.len());
    for (i, s) in streams.iter().enumerate() {
        let entry = build_strf(&s.params)?;
        let packet_fourcc = packet_fourcc_for(i as u32, entry.chunk_suffix);
        tracks.push(TrackState {
            stream: s.clone(),
            entry,
            packet_fourcc,
            packet_count: 0,
            sample_count: 0,
            max_chunk_size: 0,
            total_bytes: 0,
        });
    }
    Ok(Box::new(AviMuxer {
        output,
        tracks,
        riff_size_off: 0,
        movi_size_off: 0,
        movi_start_off: 0,
        index: Vec::new(),
        header_written: false,
        trailer_written: false,
    }))
}

fn packet_fourcc_for(index: u32, suffix: [u8; 2]) -> [u8; 4] {
    // 00dc-style: two ASCII decimal digits.
    let tens = (index / 10) as u8 + b'0';
    let ones = (index % 10) as u8 + b'0';
    [tens, ones, suffix[0], suffix[1]]
}

struct AviMuxer {
    output: Box<dyn WriteSeek>,
    tracks: Vec<TrackState>,
    /// Offset of the RIFF chunk size field.
    riff_size_off: u64,
    /// Offset of the movi LIST size field.
    movi_size_off: u64,
    /// Start offset of movi list body (i.e. of the "movi" form-type word).
    /// AVI idx1 entries are offsets *from this point*, specifically from the
    /// byte that is 4 bytes before the first chunk header (i.e. the `movi`
    /// form-type fourcc).
    movi_start_off: u64,
    /// Per-packet index entries, built as we write; emitted in `write_trailer`.
    index: Vec<IndexEntry>,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for AviMuxer {
    fn format_name(&self) -> &str {
        "avi"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("avi muxer: write_header called twice"));
        }
        // Start outer RIFF list.
        self.riff_size_off = begin_list(self.output.as_mut(), &RIFF, &AVI_FORM)?;

        // hdrl LIST with avih + strl*.
        let hdrl_size_off = begin_list(self.output.as_mut(), &LIST, b"hdrl")?;
        let avih = build_avih(&self.tracks);
        write_chunk(self.output.as_mut(), b"avih", &avih)?;
        for (i, t) in self.tracks.iter().enumerate() {
            write_strl(self.output.as_mut(), i as u32, t)?;
        }
        finish_chunk(self.output.as_mut(), hdrl_size_off)?;

        // movi LIST — size patched in write_trailer.
        self.movi_size_off = begin_list(self.output.as_mut(), &LIST, b"movi")?;
        // movi_start_off points at the "movi" form-type FourCC — i.e. 4 bytes
        // after the size field. idx1 offsets are relative to this byte (+ 4 =
        // first chunk header). Per the AVI 1.0 spec, idx1 offsets may be
        // relative to either the start of the file OR the start of the movi
        // LIST body (the 'movi' FourCC). Most decoders heuristically detect
        // which — by convention, we make them relative to 'movi'.
        self.movi_start_off = self.movi_size_off + 4; // skip past size → 'movi' fourcc
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("avi muxer: write_header not called"));
        }
        let idx = packet.stream_index as usize;
        if idx >= self.tracks.len() {
            return Err(Error::invalid(format!(
                "avi muxer: unknown stream index {idx}"
            )));
        }
        if packet.data.len() > u32::MAX as usize {
            return Err(Error::invalid("avi muxer: packet larger than 4 GiB"));
        }
        let fourcc = self.tracks[idx].packet_fourcc;
        // Record offset (relative to 'movi' fourcc) BEFORE writing the chunk.
        let chunk_off = self.output.stream_position()?;
        let rel_off = chunk_off
            .checked_sub(self.movi_start_off)
            .ok_or_else(|| Error::other("avi muxer: movi offset underflow"))?;
        if rel_off > u32::MAX as u64 {
            return Err(Error::unsupported(
                "avi muxer: movi > 4 GiB, use OpenDML (not supported)",
            ));
        }
        let size = packet.data.len() as u32;
        let flags = if packet.flags.keyframe {
            0x10 // AVIIF_KEYFRAME
        } else {
            0
        };

        write_chunk(self.output.as_mut(), &fourcc, &packet.data)?;

        let t = &mut self.tracks[idx];
        t.packet_count += 1;
        if size > t.max_chunk_size {
            t.max_chunk_size = size;
        }
        t.total_bytes += size as u64;
        // Sample count: for audio with block_align, add the frame count;
        // otherwise one sample per packet.
        t.sample_count += sample_count_of_packet(&t.stream, &t.entry, size);

        self.index.push(IndexEntry {
            ckid: fourcc,
            flags,
            offset: rel_off as u32,
            size,
        });

        // Enforce the 2 GiB ceiling (AVI v1 — no OpenDML). Real-world
        // players often choke between 2 and 4 GiB, so we flag at 2 GiB.
        let cur = self.output.stream_position()?;
        if cur > (2 * 1024 * 1024 * 1024) - 1024 {
            return Err(Error::unsupported(
                "avi muxer: file would exceed 2 GiB (OpenDML not supported)",
            ));
        }

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        if !self.header_written {
            return Err(Error::other("avi muxer: write_trailer before write_header"));
        }
        // Close movi LIST (patch its size).
        finish_chunk(self.output.as_mut(), self.movi_size_off)?;

        // idx1: 16 bytes per entry (ckid[4], flags[u32 LE], offset[u32 LE],
        // size[u32 LE]). Offsets are relative to the `movi` form-type FourCC.
        let mut idx_body = Vec::with_capacity(self.index.len() * 16);
        for e in &self.index {
            idx_body.extend_from_slice(&e.ckid);
            idx_body.extend_from_slice(&e.flags.to_le_bytes());
            idx_body.extend_from_slice(&e.offset.to_le_bytes());
            idx_body.extend_from_slice(&e.size.to_le_bytes());
        }
        write_chunk(self.output.as_mut(), b"idx1", &idx_body)?;

        // Close outer RIFF.
        finish_chunk(self.output.as_mut(), self.riff_size_off)?;

        // Optionally patch avih.dwTotalFrames and strh.dwLength now that we
        // know the packet counts. These are located at well-known offsets
        // relative to the RIFF start.
        self.patch_post_counts()?;

        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

impl AviMuxer {
    /// Patch avih/strh length fields after the trailer is written. We know
    /// the exact offsets because we laid out the header deterministically.
    fn patch_post_counts(&mut self) -> Result<()> {
        // avih total_frames = max video stream packet_count (or first
        // stream if no video). strh dwLength = per-stream packet_count for
        // video, sample_count for audio.
        //
        // Layout we wrote:
        //   RIFF(12): 4 + size + 4(AVI )         — offset 0..12
        //   LIST(8): 4 + size + 4(hdrl)           — offset 12..20
        //   "avih" chunk:
        //     header: 4(avih) + 4(size)           — offset 20..28
        //     body  : 56 bytes                    — offset 28..84
        //       total_frames at body offset 16    — file offset 44..48
        //   For each stream i:
        //     LIST(8): 4 + size + 4(strl)
        //     strh(8): 4 + 4 + 56
        //       dwLength at strh body offset 32
        //     strf(8+N) ...
        let total_video_frames = self
            .tracks
            .iter()
            .find(|t| &t.entry.strh_type == b"vids")
            .map(|t| t.packet_count)
            .unwrap_or_else(|| self.tracks.first().map(|t| t.packet_count).unwrap_or(0));

        let end_pos = self.output.stream_position()?;

        // avih.dwTotalFrames is at offset 20 (LIST hdrl header end) + 8
        // ("avih" chunk header) + 16 (body offset of TotalFrames) = 44.
        self.output.seek(SeekFrom::Start(44))?;
        self.output.write_all(&total_video_frames.to_le_bytes())?;

        // Walk through strl lists to patch each strh.dwLength.
        // We need to compute the offset of each strl LIST. The hdrl LIST has
        // size = 4 (form-type) + avih chunk (8+56=64) + sum(strl sizes).
        // So the first strl LIST starts at:
        //   20 (hdrl header end skipping "hdrl") + 4(form) +64(avih) = 88
        // But actually RIFF(8)+AVI_(4)+LIST(8)+hdrl(4)+avih(8+56)=4+8+4+8+4+64 = ... let me recompute:
        //   RIFF header: 8 bytes (0..8)
        //   AVI  form type: 4 bytes (8..12)
        //   LIST header: 8 bytes (12..20)
        //   hdrl form type: 4 bytes (20..24)
        //   avih chunk: 8+56 = 64 bytes (24..88)
        //   → first strl LIST starts at 88
        let mut strl_off: u64 = 88;
        for t in &self.tracks {
            // strl LIST layout:
            //   8 bytes LIST header (offset 0)
            //   4 bytes "strl" form-type (offset 8)
            //   strh chunk: 8 bytes header (offset 12) + 56 bytes body (offset 20)
            //   strf chunk: 8 bytes header + strf.len() bytes body (+ pad if odd)
            //     starting at offset 76
            let strh_body_off = strl_off + 20;
            // strh.dwLength is at body offset 32 → file offset strh_body_off + 32.
            let length = if &t.entry.strh_type == b"auds" {
                // For PCM we store sample_count (frames). For VBR we'd
                // normally use packet count, but we don't support VBR audio
                // in the mux yet.
                t.sample_count as u32
            } else {
                t.packet_count
            };
            self.output.seek(SeekFrom::Start(strh_body_off + 32))?;
            self.output.write_all(&length.to_le_bytes())?;

            // Also patch strh.dwSuggestedBufferSize at body offset 36.
            self.output.seek(SeekFrom::Start(strh_body_off + 36))?;
            self.output.write_all(&t.max_chunk_size.to_le_bytes())?;

            // Advance strl_off by the size of the strl LIST (8 header +
            // body). Body = 4 (form) + 64 (strh) + 8 + strf.len() + pad.
            let strf_padded = t.entry.strf.len() + (t.entry.strf.len() & 1);
            let strl_body = 4 + 64 + 8 + strf_padded;
            strl_off += 8 + strl_body as u64;
        }

        // Restore writer position.
        self.output.seek(SeekFrom::Start(end_pos))?;
        Ok(())
    }
}

/// AVIMAINHEADER (56 bytes): dwMicroSecPerFrame, dwMaxBytesPerSec,
/// dwPaddingGranularity, dwFlags, dwTotalFrames, dwInitialFrames, dwStreams,
/// dwSuggestedBufferSize, dwWidth, dwHeight, dwReserved[4].
fn build_avih(tracks: &[TrackState]) -> Vec<u8> {
    let (video_micro_per_frame, width, height) = tracks
        .iter()
        .find(|t| &t.entry.strh_type == b"vids")
        .map(|t| {
            // scale/rate = seconds per frame; micro_per_frame = 1_000_000 * scale/rate.
            let scale = t.entry.scale.max(1) as u64;
            let rate = t.entry.rate.max(1) as u64;
            let upf = (1_000_000u64 * scale / rate) as u32;
            let w = t.stream.params.width.unwrap_or(0);
            let h = t.stream.params.height.unwrap_or(0);
            (upf, w, h)
        })
        .unwrap_or((0, 0, 0));
    let flags: u32 = 0x00000810; // AVIF_ISINTERLEAVED | AVIF_HASINDEX
    let total_frames: u32 = 0; // patched post-hoc
    let streams = tracks.len() as u32;

    let mut body = Vec::with_capacity(56);
    body.extend_from_slice(&video_micro_per_frame.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // MaxBytesPerSec
    body.extend_from_slice(&0u32.to_le_bytes()); // PaddingGranularity
    body.extend_from_slice(&flags.to_le_bytes());
    body.extend_from_slice(&total_frames.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // InitialFrames
    body.extend_from_slice(&streams.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // SuggestedBufferSize
    body.extend_from_slice(&width.to_le_bytes());
    body.extend_from_slice(&height.to_le_bytes());
    body.extend_from_slice(&[0u8; 16]); // reserved[4]
    body
}

/// Build and write a `strl` LIST (strh + strf).
fn write_strl<W: Write + Seek + ?Sized>(w: &mut W, _index: u32, t: &TrackState) -> Result<()> {
    let strl_off = begin_list(w, &LIST, b"strl")?;

    // strh body (56 bytes).
    let mut strh = Vec::with_capacity(56);
    strh.extend_from_slice(&t.entry.strh_type); // fccType
    strh.extend_from_slice(&t.entry.handler_fourcc); // fccHandler
    strh.extend_from_slice(&0u32.to_le_bytes()); // flags
    strh.extend_from_slice(&0u16.to_le_bytes()); // priority
    strh.extend_from_slice(&0u16.to_le_bytes()); // language
    strh.extend_from_slice(&0u32.to_le_bytes()); // initial_frames
    strh.extend_from_slice(&t.entry.scale.to_le_bytes());
    strh.extend_from_slice(&t.entry.rate.to_le_bytes());
    strh.extend_from_slice(&0u32.to_le_bytes()); // start
    strh.extend_from_slice(&0u32.to_le_bytes()); // length (patched)
    strh.extend_from_slice(&0u32.to_le_bytes()); // suggested_buffer_size (patched)
    strh.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // quality = -1 (default)
    strh.extend_from_slice(&t.entry.sample_size.to_le_bytes());
    // rcFrame: left, top, right, bottom (i16 each). Use 0,0,width,height
    // for video; zeros for audio.
    if &t.entry.strh_type == b"vids" {
        let w_val = t.stream.params.width.unwrap_or(0) as i16;
        let h_val = t.stream.params.height.unwrap_or(0) as i16;
        strh.extend_from_slice(&0i16.to_le_bytes());
        strh.extend_from_slice(&0i16.to_le_bytes());
        strh.extend_from_slice(&w_val.to_le_bytes());
        strh.extend_from_slice(&h_val.to_le_bytes());
    } else {
        strh.extend_from_slice(&[0u8; 8]);
    }
    write_chunk(w, b"strh", &strh)?;

    // strf chunk.
    write_chunk(w, b"strf", &t.entry.strf)?;

    finish_chunk(w, strl_off)?;
    Ok(())
}

fn sample_count_of_packet(stream: &StreamInfo, entry: &StrfEntry, size: u32) -> u64 {
    if &entry.strh_type == b"auds" && entry.sample_size > 0 {
        (size as u64) / (entry.sample_size as u64)
    } else {
        let _ = stream;
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters};

    #[test]
    fn packet_fourcc_layout() {
        assert_eq!(packet_fourcc_for(0, *b"dc"), *b"00dc");
        assert_eq!(packet_fourcc_for(1, *b"wb"), *b"01wb");
        assert_eq!(packet_fourcc_for(12, *b"db"), *b"12db");
    }

    #[test]
    fn unsupported_codec_errors_at_open() {
        use oxideav_container::WriteSeek;
        use std::io::Cursor;
        let mut params = CodecParameters::audio(CodecId::new("opus"));
        params.channels = Some(2);
        params.sample_rate = Some(48_000);
        let stream = StreamInfo {
            index: 0,
            time_base: oxideav_core::TimeBase::new(1, 48_000),
            duration: None,
            start_time: Some(0),
            params,
        };
        let cursor: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
        match open(cursor, &[stream]) {
            Err(Error::Unsupported(_)) => {}
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Unsupported"),
        }
    }
}
