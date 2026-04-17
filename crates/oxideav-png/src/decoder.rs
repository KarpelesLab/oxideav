//! PNG + APNG decoder.
//!
//! High level flow:
//!
//! 1. Verify the 8-byte magic, then read chunks until IEND.
//! 2. `IHDR` gives width / height / bit depth / colour type.
//! 3. `PLTE` + `tRNS` feed the palette for colour type 3 (and alpha for 0/2).
//! 4. `acTL` + `fcTL` + `fdAT` carry animation frame metadata / data.
//! 5. Each frame's compressed stream = concatenation of `IDAT` (for default
//!    image) or `fdAT[4..]` (for animation frames) → `miniz_oxide` zlib
//!    decode → reverse per-row filters → fill a `VideoFrame`.
//!
//! Output pixel formats (no internal conversion — the `PixConvert` graph
//! node handles that):
//!
//! - colour type 0 / 8-bit  → `Gray8`
//! - colour type 0 / 16-bit → `Gray16Le` (network byte order collapsed to LE on output)
//! - colour type 2 / 8-bit  → `Rgb24`
//! - colour type 2 / 16-bit → `Rgb48Le`
//! - colour type 3 / 8-bit  → `Pal8` (palette embedded into extradata)
//! - colour type 4 / 8-bit  → `Ya8` (gray + alpha)
//! - colour type 4 / 16-bit → `Rgba64Le` (PNG has no native Ya16 — we expand)
//! - colour type 6 / 8-bit  → `Rgba`
//! - colour type 6 / 16-bit → `Rgba64Le`

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use miniz_oxide::inflate::decompress_to_vec_zlib;

use crate::apng::{parse_fdat, Actl, Blend, Disposal, Fctl};
use crate::chunk::{read_chunk, ChunkRef, PNG_MAGIC};
use crate::filter::{unfilter_row, FilterType};

pub const CODEC_ID_STR: &str = "png";

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(PngDecoder {
        codec_id: params.codec_id.clone(),
        pending: None,
        eof: false,
    }))
}

struct PngDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for PngDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "PNG decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_png_to_frame(&pkt.data, pkt.pts, pkt.time_base)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- IHDR ---------------------------------------------------------------

/// Parsed IHDR chunk (13 bytes).
#[derive(Clone, Copy, Debug)]
pub struct Ihdr {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub colour_type: u8,
    pub compression: u8,
    pub filter: u8,
    pub interlace: u8,
}

impl Ihdr {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() != 13 {
            return Err(Error::invalid(format!(
                "PNG IHDR: expected 13 bytes, got {}",
                data.len()
            )));
        }
        let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        Ok(Self {
            width,
            height,
            bit_depth: data[8],
            colour_type: data[9],
            compression: data[10],
            filter: data[11],
            interlace: data[12],
        })
    }

    pub fn to_bytes(&self) -> [u8; 13] {
        let mut out = [0u8; 13];
        out[0..4].copy_from_slice(&self.width.to_be_bytes());
        out[4..8].copy_from_slice(&self.height.to_be_bytes());
        out[8] = self.bit_depth;
        out[9] = self.colour_type;
        out[10] = self.compression;
        out[11] = self.filter;
        out[12] = self.interlace;
        out
    }

    /// Number of channels implied by `colour_type`.
    pub fn channels(&self) -> Result<usize> {
        Ok(match self.colour_type {
            0 => 1, // grayscale
            2 => 3, // RGB
            3 => 1, // palette index
            4 => 2, // gray + alpha
            6 => 4, // RGBA
            other => return Err(Error::invalid(format!("PNG: bad colour type {other}"))),
        })
    }

    /// Bytes per full pixel (rounded up to at least 1 for filtering
    /// purposes). For sub-byte bit depths this is 1 regardless of channel
    /// count, per the PNG spec.
    pub fn bpp_for_filter(&self) -> Result<usize> {
        let channels = self.channels()?;
        let bits = channels * self.bit_depth as usize;
        Ok((bits + 7) / 8)
    }

    /// Bytes per row of unfiltered pixel data.
    pub fn row_bytes(&self) -> Result<usize> {
        let channels = self.channels()?;
        let bits_per_pixel = channels * self.bit_depth as usize;
        let bits_per_row = bits_per_pixel * self.width as usize;
        Ok((bits_per_row + 7) / 8)
    }

    pub fn output_pixel_format(&self) -> Result<PixelFormat> {
        Ok(match (self.colour_type, self.bit_depth) {
            (0, 8) => PixelFormat::Gray8,
            (0, 16) => PixelFormat::Gray16Le,
            (2, 8) => PixelFormat::Rgb24,
            (2, 16) => PixelFormat::Rgb48Le,
            (3, 8) => PixelFormat::Pal8,
            (4, 8) => PixelFormat::Ya8,
            (4, 16) => PixelFormat::Rgba64Le,
            (6, 8) => PixelFormat::Rgba,
            (6, 16) => PixelFormat::Rgba64Le,
            (ct, bd) => {
                return Err(Error::unsupported(format!(
                    "PNG: colour type {ct} bit depth {bd} not implemented"
                )))
            }
        })
    }
}

// ---- The actual decode --------------------------------------------------

/// Iterate chunks from `buf[8..]`, returning a vector. The signature is
/// borrowed into the returned references. Fails fast on CRC error.
pub(crate) fn parse_all_chunks(buf: &[u8]) -> Result<Vec<ChunkRef<'_>>> {
    if buf.len() < 8 || buf[0..8] != PNG_MAGIC {
        return Err(Error::invalid("PNG: missing magic bytes"));
    }
    let mut out = Vec::new();
    let mut pos = 8;
    loop {
        let (chunk, next) = read_chunk(buf, pos)?;
        let is_iend = chunk.chunk_type == *b"IEND";
        out.push(chunk);
        pos = next;
        if is_iend {
            break;
        }
        if pos >= buf.len() {
            return Err(Error::invalid("PNG: stream ended before IEND"));
        }
    }
    Ok(out)
}

/// Decode a single non-animated PNG packet (or the "default image" of an
/// APNG) into a [`VideoFrame`].
pub fn decode_png_to_frame(
    buf: &[u8],
    pts: Option<i64>,
    time_base: TimeBase,
) -> Result<VideoFrame> {
    let chunks = parse_all_chunks(buf)?;
    let ihdr_chunk = chunks
        .iter()
        .find(|c| c.is_type(b"IHDR"))
        .ok_or_else(|| Error::invalid("PNG: missing IHDR"))?;
    let ihdr = Ihdr::parse(ihdr_chunk.data)?;
    if ihdr.interlace != 0 {
        return Err(Error::unsupported(
            "PNG: interlaced (Adam7) not implemented",
        ));
    }
    if ihdr.compression != 0 {
        return Err(Error::invalid("PNG: unknown compression method"));
    }
    if ihdr.filter != 0 {
        return Err(Error::invalid("PNG: unknown filter method"));
    }

    let mut plte: Option<&[u8]> = None;
    let mut trns: Option<&[u8]> = None;
    let mut idat_total_len = 0usize;
    for c in &chunks {
        if c.is_type(b"PLTE") {
            plte = Some(c.data);
        } else if c.is_type(b"tRNS") {
            trns = Some(c.data);
        } else if c.is_type(b"IDAT") {
            idat_total_len += c.data.len();
        }
    }

    let mut idat_concat = Vec::with_capacity(idat_total_len);
    for c in &chunks {
        if c.is_type(b"IDAT") {
            idat_concat.extend_from_slice(c.data);
        }
    }
    if idat_concat.is_empty() {
        return Err(Error::invalid("PNG: no IDAT chunks"));
    }

    let pixels = decompress_to_vec_zlib(&idat_concat)
        .map_err(|e| Error::invalid(format!("PNG: zlib decompress failed: {e:?}")))?;

    let frame_pixels = reconstruct_filtered(&pixels, &ihdr)?;
    let vf = build_video_frame(&ihdr, &frame_pixels, plte, trns, pts, time_base)?;
    Ok(vf)
}

/// Apply the 5 per-row filters, returning a flat raw-pixel buffer of
/// `row_bytes * height` bytes.
pub(crate) fn reconstruct_filtered(filtered: &[u8], ihdr: &Ihdr) -> Result<Vec<u8>> {
    let row_bytes = ihdr.row_bytes()?;
    let bpp = ihdr.bpp_for_filter()?;
    let height = ihdr.height as usize;

    // Each row is 1 filter byte + row_bytes data.
    let expected = (1 + row_bytes) * height;
    if filtered.len() != expected {
        return Err(Error::invalid(format!(
            "PNG: decompressed length {} != expected {}",
            filtered.len(),
            expected
        )));
    }

    let mut raw = vec![0u8; row_bytes * height];
    let zero_row = vec![0u8; row_bytes];
    for y in 0..height {
        let src_start = y * (1 + row_bytes);
        let filter_byte = filtered[src_start];
        let filter_type = FilterType::from_u8(filter_byte)?;
        let data_start = src_start + 1;
        // Copy row's raw bytes into dst.
        let dst_start = y * row_bytes;
        raw[dst_start..dst_start + row_bytes]
            .copy_from_slice(&filtered[data_start..data_start + row_bytes]);
        // Unfilter in place.
        let (prev_rows, curr_rows) = raw.split_at_mut(dst_start);
        let curr = &mut curr_rows[..row_bytes];
        let prev: &[u8] = if y == 0 {
            &zero_row
        } else {
            &prev_rows[(y - 1) * row_bytes..(y - 1) * row_bytes + row_bytes]
        };
        unfilter_row(filter_type, curr, prev, bpp)?;
    }
    Ok(raw)
}

/// Pack the raw pixel buffer into a `VideoFrame` for the given IHDR output
/// pixel format. For 16-bit formats we swap big-endian samples to little-
/// endian because our `Le` pixel formats expect LE byte order. For colour
/// type 4 / 16-bit we explode `(gray, alpha)` into `(gray, gray, gray, alpha)`
/// because we have no native Ya16 pixel format.
fn build_video_frame(
    ihdr: &Ihdr,
    raw: &[u8],
    plte: Option<&[u8]>,
    trns: Option<&[u8]>,
    pts: Option<i64>,
    time_base: TimeBase,
) -> Result<VideoFrame> {
    let fmt = ihdr.output_pixel_format()?;
    let w = ihdr.width as usize;
    let h = ihdr.height as usize;

    let (stride, data) = match fmt {
        PixelFormat::Gray8 => {
            // 1 byte/pixel, already laid out.
            (w, raw.to_vec())
        }
        PixelFormat::Pal8 => {
            let _plte =
                plte.ok_or_else(|| Error::invalid("PNG: colour type 3 requires PLTE chunk"))?;
            let _ = trns;
            (w, raw.to_vec())
        }
        PixelFormat::Gray16Le => {
            // 2 BE bytes/pixel → LE.
            let mut out = vec![0u8; w * h * 2];
            for i in 0..w * h {
                let be = &raw[i * 2..i * 2 + 2];
                out[i * 2] = be[1];
                out[i * 2 + 1] = be[0];
            }
            (w * 2, out)
        }
        PixelFormat::Rgb24 => (w * 3, raw.to_vec()),
        PixelFormat::Rgba => (w * 4, raw.to_vec()),
        PixelFormat::Rgb48Le => {
            let mut out = vec![0u8; w * h * 6];
            for i in 0..w * h * 3 {
                out[i * 2] = raw[i * 2 + 1];
                out[i * 2 + 1] = raw[i * 2];
            }
            (w * 6, out)
        }
        PixelFormat::Rgba64Le => {
            // Two cases: genuinely RGBA 16 (ct=6, bd=16) or gray+alpha 16 (ct=4, bd=16).
            if ihdr.colour_type == 6 {
                let mut out = vec![0u8; w * h * 8];
                for i in 0..w * h * 4 {
                    out[i * 2] = raw[i * 2 + 1];
                    out[i * 2 + 1] = raw[i * 2];
                }
                (w * 8, out)
            } else {
                // colour type 4 + 16 bit → (G16, A16) in BE per sample.
                // Expand to (G,G,G,A) LE.
                let mut out = vec![0u8; w * h * 8];
                for i in 0..w * h {
                    let g_hi = raw[i * 4];
                    let g_lo = raw[i * 4 + 1];
                    let a_hi = raw[i * 4 + 2];
                    let a_lo = raw[i * 4 + 3];
                    // Each 16-bit sample stored LE.
                    for c in 0..3 {
                        out[i * 8 + c * 2] = g_lo;
                        out[i * 8 + c * 2 + 1] = g_hi;
                    }
                    out[i * 8 + 6] = a_lo;
                    out[i * 8 + 7] = a_hi;
                }
                (w * 8, out)
            }
        }
        PixelFormat::Ya8 => (w * 2, raw.to_vec()),
        other => {
            return Err(Error::unsupported(format!(
                "PNG: build_video_frame unhandled pixel format {:?}",
                other
            )))
        }
    };

    Ok(VideoFrame {
        format: fmt,
        width: ihdr.width,
        height: ihdr.height,
        pts,
        time_base,
        planes: vec![VideoPlane { stride, data }],
    })
}

// ---- APNG: multi-frame iterator ----------------------------------------

/// Static description of an APNG, including its per-frame compressed data
/// segments ready for decompression. The demuxer uses this to split a PNG
/// file into per-frame packets.
#[derive(Debug)]
pub struct ApngInfo {
    pub ihdr: Ihdr,
    pub plte: Option<Vec<u8>>,
    pub trns: Option<Vec<u8>>,
    pub actl: Actl,
    /// One entry per animation frame.
    pub frames: Vec<ApngFrame>,
    /// True if the default image (IDAT) is also the first animation frame —
    /// i.e. there's an `fcTL` that came before `IDAT`.
    pub first_frame_is_default: bool,
}

#[derive(Debug)]
pub struct ApngFrame {
    pub fctl: Fctl,
    /// Concatenated compressed data: IDAT payload or fdAT payloads stripped
    /// of their 4-byte sequence number.
    pub compressed: Vec<u8>,
}

/// Parse an APNG file and return metadata + per-frame compressed segments.
/// Returns `Err` if the file is a plain PNG without `acTL`.
pub fn parse_apng(buf: &[u8]) -> Result<ApngInfo> {
    let chunks = parse_all_chunks(buf)?;
    let ihdr = Ihdr::parse(
        chunks
            .iter()
            .find(|c| c.is_type(b"IHDR"))
            .ok_or_else(|| Error::invalid("PNG: missing IHDR"))?
            .data,
    )?;
    let actl = chunks
        .iter()
        .find(|c| c.is_type(b"acTL"))
        .ok_or_else(|| Error::invalid("PNG: not animated (no acTL)"))?;
    let actl = Actl::parse(actl.data)?;

    let mut plte: Option<Vec<u8>> = None;
    let mut trns: Option<Vec<u8>> = None;

    let mut frames: Vec<ApngFrame> = Vec::new();
    let mut current_fctl: Option<Fctl> = None;
    let mut current_compressed: Vec<u8> = Vec::new();
    let mut saw_idat = false;
    let mut first_frame_is_default = false;

    // Scan chunks in order.
    let mut seen_any_fctl = false;
    for c in &chunks {
        match &c.chunk_type {
            b"PLTE" => plte = Some(c.data.to_vec()),
            b"tRNS" => trns = Some(c.data.to_vec()),
            b"fcTL" => {
                // Flush the previous frame.
                if let Some(prev_fctl) = current_fctl.take() {
                    frames.push(ApngFrame {
                        fctl: prev_fctl,
                        compressed: std::mem::take(&mut current_compressed),
                    });
                }
                let f = Fctl::parse(c.data)?;
                // If we see fcTL before IDAT, the default image is also the
                // first animation frame.
                if !saw_idat {
                    first_frame_is_default = true;
                }
                current_fctl = Some(f);
                seen_any_fctl = true;
            }
            b"IDAT" => {
                // If we have an fcTL pending, these IDAT bytes belong to
                // that frame. Otherwise, they form the "default image"
                // which is not part of the animation.
                saw_idat = true;
                if current_fctl.is_some() {
                    current_compressed.extend_from_slice(c.data);
                }
            }
            b"fdAT" => {
                let (_seq, payload) = parse_fdat(c.data)?;
                current_compressed.extend_from_slice(payload);
            }
            _ => {}
        }
        let _ = seen_any_fctl;
    }
    // Final frame.
    if let Some(f) = current_fctl.take() {
        frames.push(ApngFrame {
            fctl: f,
            compressed: std::mem::take(&mut current_compressed),
        });
    }

    if frames.len() as u32 != actl.num_frames {
        // Survive mildly lying files — don't error, but note in a comment.
        // (Libpng accepts extra/missing frames.)
    }

    Ok(ApngInfo {
        ihdr,
        plte,
        trns,
        actl,
        frames,
        first_frame_is_default,
    })
}

/// Decompress + unfilter a single APNG animation frame into a `VideoFrame`.
/// `ihdr` is the file-level IHDR; the frame may be smaller than the canvas
/// (fcTL width/height < IHDR width/height). The returned frame has the
/// frame-local dimensions; callers wanting canvas-size frames must
/// composite using `fctl.x_offset / y_offset` + disposal state.
///
/// In this initial cut we composite into a canvas-sized frame so the top
/// level API is simpler for downstream consumers.
pub fn decode_apng_frames(info: &ApngInfo, time_base: TimeBase) -> Result<Vec<VideoFrame>> {
    let canvas_w = info.ihdr.width;
    let canvas_h = info.ihdr.height;
    let canvas_fmt = info.ihdr.output_pixel_format()?;
    let (bytes_per_pixel, stride_canvas) = bytes_per_pixel_and_stride(canvas_fmt, canvas_w)?;

    let mut canvas = vec![0u8; stride_canvas * canvas_h as usize];
    // For Disposal::Previous we snapshot the pre-draw state before writing
    // the new frame.
    let mut prev_snapshot: Option<Vec<u8>> = None;
    let mut out_frames: Vec<VideoFrame> = Vec::new();
    let mut pts: i64 = 0;

    for (_idx, frame) in info.frames.iter().enumerate() {
        // Build a synthetic IHDR-like block for the sub-frame dimensions.
        // Same colour_type / bit_depth / compression / filter / interlace.
        let sub_ihdr = Ihdr {
            width: frame.fctl.width,
            height: frame.fctl.height,
            bit_depth: info.ihdr.bit_depth,
            colour_type: info.ihdr.colour_type,
            compression: info.ihdr.compression,
            filter: info.ihdr.filter,
            interlace: info.ihdr.interlace,
        };
        let decompressed = decompress_to_vec_zlib(&frame.compressed)
            .map_err(|e| Error::invalid(format!("APNG: zlib failed: {e:?}")))?;
        let frame_raw = reconstruct_filtered(&decompressed, &sub_ihdr)?;
        let sub_frame = build_video_frame(
            &sub_ihdr,
            &frame_raw,
            info.plte.as_deref(),
            info.trns.as_deref(),
            None,
            time_base,
        )?;

        // Disposal: Previous → snapshot the region pre-draw.
        if frame.fctl.dispose_op == Disposal::Previous {
            prev_snapshot = Some(canvas.clone());
        }

        // Blend the sub_frame into the canvas.
        blit_sub_into_canvas(
            &mut canvas,
            stride_canvas,
            bytes_per_pixel,
            canvas_w as usize,
            canvas_h as usize,
            &sub_frame,
            frame.fctl.x_offset as usize,
            frame.fctl.y_offset as usize,
            frame.fctl.blend_op,
        );

        // Emit the composed canvas as this frame.
        let mut vf = VideoFrame {
            format: canvas_fmt,
            width: canvas_w,
            height: canvas_h,
            pts: Some(pts),
            time_base,
            planes: vec![VideoPlane {
                stride: stride_canvas,
                data: canvas.clone(),
            }],
        };
        let delay = frame.fctl.delay_centiseconds().max(1) as i64;
        vf.pts = Some(pts);
        pts += delay;
        out_frames.push(vf);

        // Apply disposal *after* emitting.
        match frame.fctl.dispose_op {
            Disposal::None => {}
            Disposal::Background => {
                // Clear the sub-region to zeros.
                clear_region(
                    &mut canvas,
                    stride_canvas,
                    bytes_per_pixel,
                    canvas_w as usize,
                    canvas_h as usize,
                    frame.fctl.x_offset as usize,
                    frame.fctl.y_offset as usize,
                    frame.fctl.width as usize,
                    frame.fctl.height as usize,
                );
            }
            Disposal::Previous => {
                if let Some(snap) = prev_snapshot.take() {
                    canvas = snap;
                }
            }
        }
    }

    Ok(out_frames)
}

fn bytes_per_pixel_and_stride(fmt: PixelFormat, w: u32) -> Result<(usize, usize)> {
    Ok(match fmt {
        PixelFormat::Gray8 | PixelFormat::Pal8 => (1, w as usize),
        PixelFormat::Ya8 => (2, w as usize * 2),
        PixelFormat::Rgb24 => (3, w as usize * 3),
        PixelFormat::Rgba => (4, w as usize * 4),
        PixelFormat::Gray16Le => (2, w as usize * 2),
        PixelFormat::Rgb48Le => (6, w as usize * 6),
        PixelFormat::Rgba64Le => (8, w as usize * 8),
        other => {
            return Err(Error::unsupported(format!(
                "PNG: bytes_per_pixel_and_stride unsupported for {other:?}"
            )))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn blit_sub_into_canvas(
    canvas: &mut [u8],
    stride_canvas: usize,
    bpp: usize,
    canvas_w: usize,
    canvas_h: usize,
    sub: &VideoFrame,
    x_off: usize,
    y_off: usize,
    blend: Blend,
) {
    let sub_stride = sub.planes[0].stride;
    let sub_w = sub.width as usize;
    let sub_h = sub.height as usize;
    for sy in 0..sub_h {
        let dy = y_off + sy;
        if dy >= canvas_h {
            break;
        }
        let row_cap = (canvas_w - x_off.min(canvas_w)).min(sub_w);
        for sx in 0..row_cap {
            let dx = x_off + sx;
            let src =
                &sub.planes[0].data[sy * sub_stride + sx * bpp..sy * sub_stride + (sx + 1) * bpp];
            let dst_start = dy * stride_canvas + dx * bpp;
            let dst = &mut canvas[dst_start..dst_start + bpp];
            match blend {
                Blend::Source => {
                    dst.copy_from_slice(src);
                }
                Blend::Over => {
                    // Only meaningful for formats with alpha. For formats
                    // without alpha we fall back to Source.
                    if bpp == 4 {
                        // 8-bit RGBA alpha compositing.
                        let a = src[3] as u32;
                        if a == 255 {
                            dst.copy_from_slice(src);
                        } else if a == 0 {
                            // Leave canvas alone.
                        } else {
                            let inv = 255 - a;
                            for c in 0..3 {
                                let fg = src[c] as u32 * a;
                                let bg = dst[c] as u32 * inv;
                                dst[c] = ((fg + bg + 127) / 255) as u8;
                            }
                            // Alpha over: a_out = a_src + a_dst * (1 - a_src)
                            let a_dst = dst[3] as u32;
                            dst[3] = (a + ((a_dst * inv + 127) / 255)) as u8;
                        }
                    } else if bpp == 2 && matches!(sub.format, PixelFormat::Ya8) {
                        let a = src[1] as u32;
                        if a == 255 {
                            dst.copy_from_slice(src);
                        } else if a != 0 {
                            let inv = 255 - a;
                            let fg = src[0] as u32 * a;
                            let bg = dst[0] as u32 * inv;
                            dst[0] = ((fg + bg + 127) / 255) as u8;
                            let a_dst = dst[1] as u32;
                            dst[1] = (a + ((a_dst * inv + 127) / 255)) as u8;
                        }
                    } else {
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn clear_region(
    canvas: &mut [u8],
    stride_canvas: usize,
    bpp: usize,
    canvas_w: usize,
    canvas_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) {
    for dy in y..(y + h).min(canvas_h) {
        let row_start = dy * stride_canvas + x * bpp;
        let row_end = row_start + ((w.min(canvas_w - x.min(canvas_w))) * bpp);
        for b in &mut canvas[row_start..row_end] {
            *b = 0;
        }
    }
}
