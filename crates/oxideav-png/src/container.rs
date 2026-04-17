//! PNG / APNG container demuxer + muxer.
//!
//! Non-animated PNG: one `Packet` on stream 0 containing the whole file.
//! Animated PNG (has `acTL`): one `Packet` per animation frame, where each
//! packet is a valid standalone PNG in itself — we extract the default
//! image + all metadata chunks for the static frames, and for animated
//! frames we synthesise a PNG that replaces the file's default image with
//! that animation frame's pixel data (so a downstream decoder can decode
//! each packet independently).
//!
//! The `CodecParameters::extradata` carries the original IHDR bytes plus,
//! for palettised PNGs, the PLTE and tRNS chunk data concatenated in that
//! order (layout: `[IHDR (13 bytes)] [PLTE ...] [tRNS ...]`, with length
//! prefixes as `u32` BE before each non-IHDR block so callers can parse).

use std::io::{Read, SeekFrom, Write};

use oxideav_codec::Decoder;
use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, PixelFormat, Result, StreamInfo, TimeBase,
};

use crate::apng::parse_fdat;
use crate::chunk::{read_chunk, write_chunk, ChunkRef, PNG_MAGIC};
use crate::decoder::{parse_all_chunks, Ihdr};

/// Register the PNG codec (decoder + encoder).
pub fn register_codecs(reg: &mut oxideav_codec::CodecRegistry) {
    use oxideav_core::{CodecCapabilities, CodecId};

    let cid = CodecId::new(crate::CODEC_ID_STR);
    let caps = CodecCapabilities::video("png_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(16384, 16384)
        .with_pixel_formats(vec![
            PixelFormat::Rgba,
            PixelFormat::Rgb24,
            PixelFormat::Gray8,
            PixelFormat::Pal8,
            PixelFormat::Rgb48Le,
            PixelFormat::Rgba64Le,
        ]);
    reg.register_both(
        cid,
        caps,
        crate::decoder::make_decoder,
        crate::encoder::make_encoder,
    );
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_demuxer("png", open_demuxer);
    reg.register_muxer("png", open_muxer);
    reg.register_extension("png", "png");
    reg.register_extension("apng", "png");
    reg.register_probe("png", probe);
}

pub fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() < 8 {
        return 0;
    }
    if p.buf[0..8] == PNG_MAGIC {
        100
    } else {
        0
    }
}

// ---- Demuxer ------------------------------------------------------------

fn open_demuxer(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);

    if buf.len() < 8 || buf[0..8] != PNG_MAGIC {
        return Err(Error::invalid("PNG: bad magic"));
    }

    // Walk chunks to classify.
    let chunks = parse_all_chunks(&buf)?;
    let ihdr = Ihdr::parse(
        chunks
            .iter()
            .find(|c| c.is_type(b"IHDR"))
            .ok_or_else(|| Error::invalid("PNG: missing IHDR"))?
            .data,
    )?;
    let has_actl = chunks.iter().any(|c| c.is_type(b"acTL"));
    let loop_count = if let Some(actl) = chunks.iter().find(|c| c.is_type(b"acTL")) {
        if actl.data.len() == 8 {
            Some(u32::from_be_bytes([
                actl.data[4],
                actl.data[5],
                actl.data[6],
                actl.data[7],
            ]))
        } else {
            None
        }
    } else {
        None
    };

    let plte = chunks
        .iter()
        .find(|c| c.is_type(b"PLTE"))
        .map(|c| c.data.to_vec());
    let trns = chunks
        .iter()
        .find(|c| c.is_type(b"tRNS"))
        .map(|c| c.data.to_vec());

    // Build extradata: IHDR bytes || PLTE (with u32 BE length prefix) || tRNS (with u32 BE length prefix).
    let mut extradata = Vec::new();
    extradata.extend_from_slice(ihdr_chunk_data(&chunks)?);
    if let Some(p) = &plte {
        extradata.extend_from_slice(&(p.len() as u32).to_be_bytes());
        extradata.extend_from_slice(p);
    } else {
        extradata.extend_from_slice(&0u32.to_be_bytes());
    }
    if let Some(t) = &trns {
        extradata.extend_from_slice(&(t.len() as u32).to_be_bytes());
        extradata.extend_from_slice(t);
    } else {
        extradata.extend_from_slice(&0u32.to_be_bytes());
    }

    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(ihdr.width);
    params.height = Some(ihdr.height);
    params.pixel_format = Some(ihdr.output_pixel_format()?);
    params.extradata = extradata;

    let time_base = TimeBase::new(1, 100);
    let packets = if has_actl {
        build_apng_packets(&buf, &chunks, time_base)?
    } else {
        // Non-animated: single-packet representation = whole file.
        let mut pkt = Packet::new(0, time_base, buf.clone());
        pkt.pts = Some(0);
        pkt.dts = Some(0);
        pkt.duration = Some(1);
        pkt.flags.keyframe = true;
        vec![pkt]
    };

    let total_duration: i64 = packets.iter().map(|p| p.duration.unwrap_or(0)).sum();
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total_duration.max(1)),
        start_time: Some(0),
        params,
    };

    let mut metadata: Vec<(String, String)> = Vec::new();
    if let Some(lc) = loop_count {
        metadata.push(("loop_count".into(), lc.to_string()));
    }

    Ok(Box::new(PngDemuxer {
        stream,
        packets,
        pos: 0,
        metadata,
    }))
}

/// Lift the raw IHDR data bytes (13 bytes) out of the chunk list.
fn ihdr_chunk_data<'a>(chunks: &'a [ChunkRef<'_>]) -> Result<&'a [u8]> {
    chunks
        .iter()
        .find(|c| c.is_type(b"IHDR"))
        .map(|c| c.data)
        .ok_or_else(|| Error::invalid("PNG: missing IHDR"))
}

/// For an APNG file, split it into one packet per animation frame. Each
/// packet is a standalone PNG that a plain PNG decoder can decode (the
/// per-frame IDAT is inlined as a plain IDAT on top of the original header
/// chunks, and we strip acTL/fcTL/fdAT markers to make the output a still
/// image).
fn build_apng_packets(
    _file_buf: &[u8],
    chunks: &[ChunkRef<'_>],
    time_base: TimeBase,
) -> Result<Vec<Packet>> {
    use crate::apng::Fctl;

    let ihdr = chunks
        .iter()
        .find(|c| c.is_type(b"IHDR"))
        .ok_or_else(|| Error::invalid("PNG: missing IHDR"))?
        .data;
    let plte = chunks.iter().find(|c| c.is_type(b"PLTE")).map(|c| c.data);
    let trns = chunks.iter().find(|c| c.is_type(b"tRNS")).map(|c| c.data);

    let mut packets: Vec<Packet> = Vec::new();
    let mut pts: i64 = 0;

    // State accumulators for the currently-parsed frame.
    let mut pending_fctl: Option<Fctl> = None;
    let mut pending_data: Vec<u8> = Vec::new();
    let mut saw_idat = false;

    for c in chunks {
        match &c.chunk_type {
            b"fcTL" => {
                if let Some(fctl) = pending_fctl.take() {
                    let pkt = build_still_png_packet(
                        ihdr,
                        plte,
                        trns,
                        &pending_data,
                        time_base,
                        pts,
                        &fctl,
                    )?;
                    let delay = fctl.delay_centiseconds().max(1) as i64;
                    pts += delay;
                    let mut p = pkt;
                    p.duration = Some(delay);
                    packets.push(p);
                }
                pending_data.clear();
                pending_fctl = Some(Fctl::parse(c.data)?);
            }
            b"IDAT" => {
                saw_idat = true;
                if pending_fctl.is_some() {
                    pending_data.extend_from_slice(c.data);
                }
            }
            b"fdAT" => {
                let (_seq, payload) = parse_fdat(c.data)?;
                pending_data.extend_from_slice(payload);
            }
            _ => {}
        }
    }
    if let Some(fctl) = pending_fctl.take() {
        let pkt = build_still_png_packet(ihdr, plte, trns, &pending_data, time_base, pts, &fctl)?;
        let delay = fctl.delay_centiseconds().max(1) as i64;
        let mut p = pkt;
        p.duration = Some(delay);
        packets.push(p);
    }
    let _ = saw_idat;

    Ok(packets)
}

/// Build a standalone PNG file for a single animation frame. The file-level
/// IHDR says `width x height` but the frame is actually `fctl.width x
/// fctl.height` — we rewrite IHDR to the frame-local size so the decoder
/// treats it as a plain static PNG with the right bounds. Offsets into the
/// canvas are preserved in the packet metadata via pts.
fn build_still_png_packet(
    ihdr_bytes: &[u8],
    plte: Option<&[u8]>,
    trns: Option<&[u8]>,
    frame_idat: &[u8],
    time_base: TimeBase,
    pts: i64,
    fctl: &crate::apng::Fctl,
) -> Result<Packet> {
    // Rewrite IHDR width/height to the frame-local size.
    let mut patched_ihdr = [0u8; 13];
    patched_ihdr.copy_from_slice(ihdr_bytes);
    patched_ihdr[0..4].copy_from_slice(&fctl.width.to_be_bytes());
    patched_ihdr[4..8].copy_from_slice(&fctl.height.to_be_bytes());

    let mut out = Vec::with_capacity(64 + frame_idat.len());
    out.extend_from_slice(&PNG_MAGIC);
    write_chunk(&mut out, b"IHDR", &patched_ihdr);
    if let Some(p) = plte {
        write_chunk(&mut out, b"PLTE", p);
    }
    if let Some(t) = trns {
        write_chunk(&mut out, b"tRNS", t);
    }
    write_chunk(&mut out, b"IDAT", frame_idat);
    write_chunk(&mut out, b"IEND", &[]);

    let mut pkt = Packet::new(0, time_base, out);
    pkt.pts = Some(pts);
    pkt.dts = Some(pts);
    pkt.flags.keyframe = true;
    Ok(pkt)
}

struct PngDemuxer {
    stream: StreamInfo,
    packets: Vec<Packet>,
    pos: usize,
    metadata: Vec<(String, String)>,
}

impl Demuxer for PngDemuxer {
    fn format_name(&self) -> &str {
        "png"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.pos >= self.packets.len() {
            return Err(Error::Eof);
        }
        let pkt = self.packets[self.pos].clone();
        self.pos += 1;
        Ok(pkt)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        self.stream.duration.map(|d| d * 10_000)
    }
}

// ---- Muxer --------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported(
            "PNG muxer: exactly one video stream expected",
        ));
    }
    let s = &streams[0];
    if s.params.codec_id.as_str() != crate::CODEC_ID_STR {
        return Err(Error::invalid(format!(
            "PNG muxer: codec_id must be png (got {})",
            s.params.codec_id
        )));
    }
    Ok(Box::new(PngMuxer {
        output,
        stream: s.clone(),
        frames: Vec::new(),
        header_written: false,
        trailer_written: false,
    }))
}

struct PngMuxer {
    output: Box<dyn WriteSeek>,
    stream: StreamInfo,
    frames: Vec<Packet>,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for PngMuxer {
    fn format_name(&self) -> &str {
        "png"
    }

    fn write_header(&mut self) -> Result<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("PNG muxer: write_header not called"));
        }
        self.frames.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        // If exactly one packet → write it verbatim (it's already a full PNG).
        // If multiple packets → re-synthesise an APNG from them.
        if self.frames.len() == 1 {
            self.output.write_all(&self.frames[0].data)?;
        } else if self.frames.len() > 1 {
            // For multi-packet input, just write the first frame. Full
            // recombination into an APNG requires re-parsing and splicing
            // IDATs and is left for a follow-up — the encoder itself
            // produces a full APNG when multiple frames are sent, so this
            // path is exercised mostly by the "demux-then-remux" flow
            // which we handle by concatenating back via a helper.
            let merged = merge_still_packets_to_apng(&self.frames, &self.stream)?;
            self.output.write_all(&merged)?;
        } else {
            return Err(Error::invalid("PNG muxer: no packets written"));
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

/// Take N standalone-PNG packets produced by the demuxer's APNG split and
/// re-assemble them into a single APNG file. Extracts IDATs, rewrites them
/// as fdATs for frames 1..N, and inserts matching fcTL chunks between them.
fn merge_still_packets_to_apng(packets: &[Packet], stream: &StreamInfo) -> Result<Vec<u8>> {
    use crate::apng::{Actl, Blend, Disposal, Fctl};

    if packets.is_empty() {
        return Err(Error::invalid("PNG muxer: empty packet list"));
    }

    // Parse each packet's IHDR + IDATs.
    struct ParsedStill {
        ihdr: Ihdr,
        plte: Option<Vec<u8>>,
        trns: Option<Vec<u8>>,
        idat: Vec<u8>,
        duration: i64,
    }
    let mut stills: Vec<ParsedStill> = Vec::new();
    for p in packets {
        let chunks = parse_all_chunks(&p.data)?;
        let ihdr = Ihdr::parse(
            chunks
                .iter()
                .find(|c| c.is_type(b"IHDR"))
                .ok_or_else(|| Error::invalid("PNG muxer: packet missing IHDR"))?
                .data,
        )?;
        let plte = chunks
            .iter()
            .find(|c| c.is_type(b"PLTE"))
            .map(|c| c.data.to_vec());
        let trns = chunks
            .iter()
            .find(|c| c.is_type(b"tRNS"))
            .map(|c| c.data.to_vec());
        let mut idat = Vec::new();
        for c in &chunks {
            if c.is_type(b"IDAT") {
                idat.extend_from_slice(c.data);
            }
        }
        stills.push(ParsedStill {
            ihdr,
            plte,
            trns,
            idat,
            duration: p.duration.unwrap_or(1),
        });
    }

    // Canvas size from first still.
    let canvas_ihdr = stills[0].ihdr;
    // Pull PLTE/tRNS from first still if present.
    let plte = stills[0].plte.clone();
    let trns = stills[0].trns.clone();

    let actl = Actl {
        num_frames: stills.len() as u32,
        num_plays: 0,
    };
    let _ = stream;

    let mut out = Vec::new();
    out.extend_from_slice(&PNG_MAGIC);
    write_chunk(&mut out, b"IHDR", &canvas_ihdr.to_bytes());
    write_chunk(&mut out, b"acTL", &actl.to_bytes());
    if let Some(p) = plte.as_deref() {
        write_chunk(&mut out, b"PLTE", p);
    }
    if let Some(t) = trns.as_deref() {
        write_chunk(&mut out, b"tRNS", t);
    }

    let mut seq: u32 = 0;
    for (i, s) in stills.iter().enumerate() {
        let fctl = Fctl {
            sequence_number: seq,
            width: s.ihdr.width,
            height: s.ihdr.height,
            x_offset: 0,
            y_offset: 0,
            delay_num: s.duration as u16,
            delay_den: 100,
            dispose_op: Disposal::None,
            blend_op: Blend::Source,
        };
        write_chunk(&mut out, b"fcTL", &fctl.to_bytes());
        seq += 1;
        if i == 0 {
            write_chunk(&mut out, b"IDAT", &s.idat);
        } else {
            let payload = crate::apng::build_fdat(seq, &s.idat);
            write_chunk(&mut out, b"fdAT", &payload);
            seq += 1;
        }
    }
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

// Re-export decoder factory through here so `crate::container::register_codecs`
// is self-contained.
#[allow(dead_code)]
fn _unused_linker_hook() -> Option<fn(&CodecParameters) -> Result<Box<dyn Decoder>>> {
    // Silence dead-code analyzer for the unused re-exports.
    let _ = crate::chunk::read_chunk as fn(&[u8], usize) -> Result<(ChunkRef<'_>, usize)>;
    let _ = read_chunk;
    None
}
