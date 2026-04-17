//! PNG + APNG encoder.
//!
//! The encoder accepts a single video frame per `send_frame` and emits a
//! full, standalone PNG file on the first `receive_packet`. If multiple
//! frames are submitted (`frame_rate` set, or multiple `send_frame` calls
//! before the first drain), the trailing frames are buffered and an APNG
//! is produced on `flush` — the single output packet then contains the
//! whole animation.
//!
//! Compression level is fixed at miniz_oxide default (6). All rows use the
//! PNG §12.8 "minimum sum of absolute differences" heuristic (i.e. try all
//! 5 filters, pick the one with the smallest absolute byte sum).

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use miniz_oxide::deflate::compress_to_vec_zlib;

use crate::apng::{build_fdat, Actl, Blend, Disposal, Fctl};
use crate::chunk::{write_chunk, PNG_MAGIC};
use crate::decoder::Ihdr;
use crate::filter::{choose_filter_heuristic, filter_row};

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("PNG encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("PNG encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Rgba);
    // Allowed pixel formats.
    match pix {
        PixelFormat::Rgba
        | PixelFormat::Rgb24
        | PixelFormat::Gray8
        | PixelFormat::Pal8
        | PixelFormat::Rgb48Le
        | PixelFormat::Rgba64Le
        | PixelFormat::Gray16Le
        | PixelFormat::Ya8 => {}
        other => {
            return Err(Error::unsupported(format!(
                "PNG encoder: pixel format {other:?} not supported"
            )))
        }
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(crate::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    // If frame_rate is set, time_base = 1/100 (centiseconds) — APNG native.
    let time_base = if params.frame_rate.is_some() {
        TimeBase::new(1, 100)
    } else {
        params
            .frame_rate
            .map(|r: Rational| TimeBase::new(r.den, r.num))
            .unwrap_or(TimeBase::new(1, 100))
    };

    let animated_hint = params.frame_rate.is_some();

    Ok(Box::new(PngEncoder {
        output_params,
        width,
        height,
        pix,
        time_base,
        frames: Vec::new(),
        pending_out: VecDeque::new(),
        frame_rate: params.frame_rate,
        palette: params.extradata.clone(),
        animated_hint,
        eof: false,
    }))
}

struct PngEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    time_base: TimeBase,
    frames: Vec<VideoFrame>,
    pending_out: VecDeque<Packet>,
    frame_rate: Option<Rational>,
    /// Raw palette + optional trns carried on `extradata`. Only used when
    /// encoding Pal8: layout is `PLTE_bytes || tRNS_bytes` per the container.
    palette: Vec<u8>,
    animated_hint: bool,
    eof: bool,
}

impl Encoder for PngEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                if v.width != self.width || v.height != self.height {
                    return Err(Error::invalid(
                        "PNG encoder: frame dimensions must match encoder config",
                    ));
                }
                if v.format != self.pix {
                    return Err(Error::invalid(format!(
                        "PNG encoder: frame format {:?} does not match encoder format {:?}",
                        v.format, self.pix
                    )));
                }
                self.frames.push(v.clone());
                // Non-animated shortcut: if we only ever get one frame and
                // there's no animation hint, we emit eagerly on flush. Keep
                // buffered.
                Ok(())
            }
            _ => Err(Error::invalid("PNG encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if !self.pending_out.is_empty() {
            return Ok(self.pending_out.pop_front().unwrap());
        }
        if self.eof {
            // Produce output now if we haven't already.
            if !self.frames.is_empty() {
                self.finalize()?;
                if let Some(p) = self.pending_out.pop_front() {
                    return Ok(p);
                }
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        // Build the packet up front so receive_packet can pick it up.
        if !self.frames.is_empty() && self.pending_out.is_empty() {
            self.finalize()?;
        }
        Ok(())
    }
}

impl PngEncoder {
    fn finalize(&mut self) -> Result<()> {
        let is_animated = self.frames.len() > 1 || self.animated_hint;
        let bytes = if is_animated {
            encode_apng(self)?
        } else {
            encode_single(&self.frames[0], self.pix, &self.palette)?
        };
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = self.frames[0].pts;
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.pending_out.push_back(pkt);
        self.frames.clear();
        Ok(())
    }
}

// ---- Single-image encode -----------------------------------------------

pub fn encode_single(frame: &VideoFrame, pix: PixelFormat, palette: &[u8]) -> Result<Vec<u8>> {
    let (ihdr, row_bytes, plte_bytes, trns_bytes) = ihdr_and_row_bytes(frame, pix, palette)?;
    let raw_pixels = flatten_and_normalise_pixels(frame, pix, row_bytes)?;
    let idat = deflate_encode_pixels(&raw_pixels, row_bytes, frame.height as usize, &ihdr)?;

    let mut out = Vec::with_capacity(64 + idat.len());
    out.extend_from_slice(&PNG_MAGIC);
    write_chunk(&mut out, b"IHDR", &ihdr.to_bytes());
    if let Some(p) = plte_bytes.as_deref() {
        write_chunk(&mut out, b"PLTE", p);
    }
    if let Some(t) = trns_bytes.as_deref() {
        write_chunk(&mut out, b"tRNS", t);
    }
    write_chunk(&mut out, b"IDAT", &idat);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// Given the encoder configuration + first frame, produce an IHDR + row byte
/// count + optional PLTE / tRNS chunk payloads.
fn ihdr_and_row_bytes(
    frame: &VideoFrame,
    pix: PixelFormat,
    palette: &[u8],
) -> Result<(Ihdr, usize, Option<Vec<u8>>, Option<Vec<u8>>)> {
    let w = frame.width;
    let h = frame.height as usize;
    let _ = h;
    let (bit_depth, colour_type, channels): (u8, u8, usize) = match pix {
        PixelFormat::Gray8 => (8, 0, 1),
        PixelFormat::Gray16Le => (16, 0, 1),
        PixelFormat::Rgb24 => (8, 2, 3),
        PixelFormat::Rgb48Le => (16, 2, 3),
        PixelFormat::Pal8 => (8, 3, 1),
        PixelFormat::Ya8 => (8, 4, 2),
        PixelFormat::Rgba => (8, 6, 4),
        PixelFormat::Rgba64Le => (16, 6, 4),
        other => {
            return Err(Error::unsupported(format!(
                "PNG encoder: unsupported pixel format {other:?}"
            )))
        }
    };
    let row_bytes = channels * (bit_depth as usize / 8) * w as usize;
    let ihdr = Ihdr {
        width: w,
        height: frame.height,
        bit_depth,
        colour_type,
        compression: 0,
        filter: 0,
        interlace: 0,
    };

    // Split palette bytes into PLTE + tRNS. Convention: if len is a multiple
    // of 3, the whole thing is PLTE. Otherwise, interpret as PLTE (first
    // ceil(len/3)*3 bytes) + tRNS (trailing bytes up to num_entries).
    let (plte, trns) = if colour_type == 3 {
        if palette.is_empty() {
            // Default: 1-entry black palette — useful fallback, but the test
            // harness will usually supply one.
            (Some(vec![0u8, 0, 0]), None)
        } else {
            // Encoder convention: caller passes `palette = PLTE || tRNS`.
            // We need to know where PLTE ends. Assume `palette` was packed
            // as `3*N RGB triples followed by M alpha bytes (M<=N)`. We
            // derive N from the plane data's max index + 1, but to keep
            // the interface simple we assume the whole `palette` is PLTE
            // iff its length is a multiple of 3. Otherwise the remainder
            // (palette.len() % 3 != 0) is interpreted as having trailing
            // tRNS bytes — but a cleaner, fully-unambiguous layout is: the
            // first N*3 bytes are PLTE and the rest are tRNS. Implement
            // that by scanning the frame for max index.
            let max_idx = frame
                .planes
                .first()
                .map(|p| p.data.iter().copied().max().unwrap_or(0))
                .unwrap_or(0) as usize;
            let n = max_idx + 1;
            let plte_len = (n * 3).min(palette.len());
            let trns_len = palette.len().saturating_sub(plte_len);
            let plte = palette[..plte_len].to_vec();
            let trns = if trns_len > 0 {
                Some(palette[plte_len..plte_len + trns_len].to_vec())
            } else {
                None
            };
            (Some(plte), trns)
        }
    } else {
        (None, None)
    };

    Ok((ihdr, row_bytes, plte, trns))
}

/// Pack `frame` into a flat BE-oriented row-major byte buffer that matches
/// the PNG wire format (before filtering / DEFLATE). `row_bytes` is the
/// expected byte count per row.
fn flatten_and_normalise_pixels(
    frame: &VideoFrame,
    pix: PixelFormat,
    row_bytes: usize,
) -> Result<Vec<u8>> {
    let h = frame.height as usize;
    let w = frame.width as usize;
    let src = &frame.planes[0];
    let mut out = vec![0u8; row_bytes * h];

    match pix {
        PixelFormat::Gray8
        | PixelFormat::Rgb24
        | PixelFormat::Rgba
        | PixelFormat::Pal8
        | PixelFormat::Ya8 => {
            // Row-by-row copy; honour source stride.
            for y in 0..h {
                let sstart = y * src.stride;
                let dstart = y * row_bytes;
                out[dstart..dstart + row_bytes]
                    .copy_from_slice(&src.data[sstart..sstart + row_bytes]);
            }
        }
        PixelFormat::Gray16Le => {
            // Source is LE per sample; PNG needs BE.
            for y in 0..h {
                for x in 0..w {
                    let lo = src.data[y * src.stride + x * 2];
                    let hi = src.data[y * src.stride + x * 2 + 1];
                    out[y * row_bytes + x * 2] = hi;
                    out[y * row_bytes + x * 2 + 1] = lo;
                }
            }
        }
        PixelFormat::Rgb48Le => {
            for y in 0..h {
                for i in 0..(w * 3) {
                    let lo = src.data[y * src.stride + i * 2];
                    let hi = src.data[y * src.stride + i * 2 + 1];
                    out[y * row_bytes + i * 2] = hi;
                    out[y * row_bytes + i * 2 + 1] = lo;
                }
            }
        }
        PixelFormat::Rgba64Le => {
            for y in 0..h {
                for i in 0..(w * 4) {
                    let lo = src.data[y * src.stride + i * 2];
                    let hi = src.data[y * src.stride + i * 2 + 1];
                    out[y * row_bytes + i * 2] = hi;
                    out[y * row_bytes + i * 2 + 1] = lo;
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "PNG encoder: flatten unsupported for {other:?}"
            )))
        }
    }
    Ok(out)
}

/// Filter each row (per the PNG spec's sum-of-abs heuristic), prepend the
/// filter-type byte, then zlib compress. Returns the compressed IDAT bytes.
fn deflate_encode_pixels(
    raw: &[u8],
    row_bytes: usize,
    height: usize,
    ihdr: &Ihdr,
) -> Result<Vec<u8>> {
    let bpp = ihdr.bpp_for_filter()?;
    // 1 filter byte + row_bytes per row.
    let mut filtered = vec![0u8; (1 + row_bytes) * height];
    let mut scratch = vec![0u8; row_bytes];
    let zero_row = vec![0u8; row_bytes];
    for y in 0..height {
        let row = &raw[y * row_bytes..(y + 1) * row_bytes];
        let prev: &[u8] = if y == 0 {
            &zero_row
        } else {
            &raw[(y - 1) * row_bytes..y * row_bytes]
        };
        let ft = if y == 0 && height == 1 {
            // First (and only) row has no predecessor — Sub is the most
            // useful filter and avoids the noop `None` for tiny images.
            // (Pick via heuristic regardless.)
            choose_filter_heuristic(row, prev, bpp, &mut scratch)
        } else {
            choose_filter_heuristic(row, prev, bpp, &mut scratch)
        };
        let dst_off = y * (1 + row_bytes);
        filtered[dst_off] = ft as u8;
        let data_slot = &mut filtered[dst_off + 1..dst_off + 1 + row_bytes];
        // Re-run the filter to write into the right slot (scratch held
        // the last tried filter during the heuristic).
        filter_row(ft, row, prev, bpp, data_slot);
    }
    Ok(compress_to_vec_zlib(&filtered, 6))
}

// ---- APNG encode --------------------------------------------------------

fn encode_apng(enc: &PngEncoder) -> Result<Vec<u8>> {
    if enc.frames.is_empty() {
        return Err(Error::invalid("PNG encoder: no frames for APNG"));
    }
    let pix = enc.pix;
    let (ihdr, row_bytes, plte, trns) = ihdr_and_row_bytes(&enc.frames[0], pix, &enc.palette)?;

    let num_plays: u32 = 0; // loop forever by default
    let actl = Actl {
        num_frames: enc.frames.len() as u32,
        num_plays,
    };

    // Default delay per frame: derived from frame_rate or 10cs = 10Hz.
    let default_delay: (u16, u16) = match enc.frame_rate {
        Some(r) if r.num > 0 && r.den > 0 => (r.den as u16, r.num as u16),
        _ => (10, 100),
    };

    let mut out = Vec::new();
    out.extend_from_slice(&PNG_MAGIC);
    write_chunk(&mut out, b"IHDR", &ihdr.to_bytes());
    write_chunk(&mut out, b"acTL", &actl.to_bytes());
    if let Some(p) = plte.as_deref() {
        write_chunk(&mut out, b"PLTE", p);
    }
    if let Some(t) = trns.as_deref() {
        write_chunk(&mut out, b"tRNS", t);
    }

    let mut seq: u32 = 0;
    for (idx, frame) in enc.frames.iter().enumerate() {
        let fctl = Fctl {
            sequence_number: seq,
            width: ihdr.width,
            height: ihdr.height,
            x_offset: 0,
            y_offset: 0,
            delay_num: default_delay.0,
            delay_den: default_delay.1,
            dispose_op: Disposal::None,
            blend_op: Blend::Source,
        };
        write_chunk(&mut out, b"fcTL", &fctl.to_bytes());
        seq += 1;

        let raw = flatten_and_normalise_pixels(frame, pix, row_bytes)?;
        let compressed = deflate_encode_pixels(&raw, row_bytes, ihdr.height as usize, &ihdr)?;

        if idx == 0 {
            // First frame is the default image → IDAT.
            write_chunk(&mut out, b"IDAT", &compressed);
        } else {
            let payload = build_fdat(seq, &compressed);
            write_chunk(&mut out, b"fdAT", &payload);
            seq += 1;
        }
    }

    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}
