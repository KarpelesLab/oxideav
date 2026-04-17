//! GIF frame decoder.
//!
//! The demuxer hands the decoder one packet per frame, each one wrapping
//! an `OGIF` payload (see `container.rs`) that contains everything needed
//! to reconstruct a single paletted frame: LZW-compressed indices, the
//! frame's sub-rectangle, the min-code-size byte, transparency info, and
//! either a local palette or a reference to the global palette (carried
//! in `CodecParameters::extradata`).
//!
//! The decoder produces `Pal8` [`VideoFrame`]s sized to the logical
//! canvas. Each frame is composited onto an internal canvas respecting
//! the GIF disposal model:
//!
//! * Disposal 0/1 — keep the rendered pixels.
//! * Disposal 2  — restore the frame area to the background (transparent
//!   if a transparent index is set, otherwise index 0).
//! * Disposal 3  — restore to the previous canvas state.
//!
//! Transparent pixels skip the composite (classic "don't touch what's
//! already there"). Interlaced frames are unwoven from GIF's 4-pass
//! order into progressive row storage before compositing.

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::container::{decode_frame_payload, extradata_to_palette};
use crate::lzw::Lzw;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("GIF decoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("GIF decoder: missing height"))?;
    let global_palette = extradata_to_palette(&params.extradata);
    Ok(Box::new(GifDecoder {
        codec_id: params.codec_id.clone(),
        width,
        height,
        global_palette,
        canvas: vec![0u8; (width * height) as usize],
        prev_canvas: None,
        pending: Vec::new(),
        eof: false,
    }))
}

struct GifDecoder {
    codec_id: CodecId,
    width: u32,
    height: u32,
    global_palette: Vec<[u8; 4]>,
    canvas: Vec<u8>,
    prev_canvas: Option<Vec<u8>>,
    pending: Vec<Packet>,
    eof: bool,
}

impl Decoder for GifDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending.push(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.pending.is_empty() {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        }
        let pkt = self.pending.remove(0);
        let df = decode_frame_payload(&pkt.data)?;

        // Decode LZW into the frame's sub-rect indices.
        let lzw = Lzw::decoder(df.min_code_size)?;
        let decoded = lzw.read(df.lzw)?;
        let frame_area = (df.w as usize) * (df.h as usize);
        if decoded.len() < frame_area {
            return Err(Error::invalid(format!(
                "GIF: LZW output {} < expected {}",
                decoded.len(),
                frame_area
            )));
        }

        // Unweave interlacing.
        let indices = if df.interlaced {
            deinterlace(&decoded, df.w as usize, df.h as usize)
        } else {
            decoded[..frame_area].to_vec()
        };

        // Disposal for the *previous* frame happens before we draw this
        // one, but we need to know that frame's disposal; the packet
        // stream carries the *current* frame's disposal for *its* post
        // step. The easiest correct encoding of that is to apply
        // disposal before composite: we treat the pre-composite state
        // as "canvas as delivered to this frame" and honour this
        // frame's disposal *after* compositing. Per-frame disposal
        // then advances the state for the next frame.
        //
        // Implementation: composite now, then capture `prev_canvas` if
        // disposal = 3, and finally apply disposal = 2 at the output
        // stage (after emitting the frame, the canvas is reset).

        // Save snapshot if this frame's disposal is "restore to previous".
        if df.disposal == 3 {
            self.prev_canvas = Some(self.canvas.clone());
        }

        // Composite.
        let transp = df.transparent_index;
        let has_transp = df.has_transparent;
        let canvas_w = self.width as usize;
        let canvas_h = self.height as usize;
        let fw = df.w as usize;
        let fh = df.h as usize;
        let fx = df.x as usize;
        let fy = df.y as usize;
        for row in 0..fh {
            let dst_y = fy + row;
            if dst_y >= canvas_h {
                break;
            }
            for col in 0..fw {
                let dst_x = fx + col;
                if dst_x >= canvas_w {
                    break;
                }
                let px = indices[row * fw + col];
                if has_transp && px == transp {
                    continue;
                }
                self.canvas[dst_y * canvas_w + dst_x] = px;
            }
        }

        // Choose which palette this frame should ship out with. Pal8
        // frames carry their indices in plane 0 and the palette in
        // plane 1 (4 bytes/entry RGBA). Most downstream code uses
        // plane 1 as the palette; we stuff the currently-active palette
        // in that plane.
        let palette = if !df.local_palette.is_empty() {
            // Local palette is the source of truth for this frame.
            // Parse the packed RGBA bytes back out.
            let n = df.local_palette.len() / 4;
            let mut pal = Vec::with_capacity(n);
            for i in 0..n {
                pal.push([
                    df.local_palette[i * 4],
                    df.local_palette[i * 4 + 1],
                    df.local_palette[i * 4 + 2],
                    df.local_palette[i * 4 + 3],
                ]);
            }
            pal
        } else {
            self.global_palette.clone()
        };

        // Pack the palette into a plane of bytes (RGBA×N, padded to 256).
        let mut palette_plane = Vec::with_capacity(256 * 4);
        for i in 0..256 {
            if i < palette.len() {
                palette_plane.extend_from_slice(&palette[i]);
            } else {
                palette_plane.extend_from_slice(&[0, 0, 0, 0xFF]);
            }
        }

        let planes = vec![
            VideoPlane {
                stride: canvas_w,
                data: self.canvas.clone(),
            },
            VideoPlane {
                stride: 256 * 4,
                data: palette_plane,
            },
        ];

        let out = VideoFrame {
            format: PixelFormat::Pal8,
            width: self.width,
            height: self.height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes,
        };

        // Apply this frame's disposal to prepare the canvas for the
        // *next* frame.
        match df.disposal {
            2 => {
                // Restore frame area to background (transparent = 0).
                let clear_idx = if has_transp { transp } else { 0 };
                for row in 0..fh {
                    let dst_y = fy + row;
                    if dst_y >= canvas_h {
                        break;
                    }
                    for col in 0..fw {
                        let dst_x = fx + col;
                        if dst_x >= canvas_w {
                            break;
                        }
                        self.canvas[dst_y * canvas_w + dst_x] = clear_idx;
                    }
                }
            }
            3 => {
                if let Some(prev) = self.prev_canvas.take() {
                    self.canvas = prev;
                }
            }
            _ => {}
        }

        Ok(Frame::Video(out))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Expand the GIF interlace pass order into progressive row storage.
///
/// GIF's interlace transmits rows in four passes:
///     pass 1: every 8th row starting at 0
///     pass 2: every 8th row starting at 4
///     pass 3: every 4th row starting at 2
///     pass 4: every 2nd row starting at 1
fn deinterlace(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut src_row = 0usize;
    for &(start, step) in &[(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
        let mut dst_row = start;
        while dst_row < h {
            let src_off = src_row * w;
            if src_off + w > src.len() {
                // Source exhausted — leave rest as zeros.
                return out;
            }
            out[dst_row * w..dst_row * w + w].copy_from_slice(&src[src_off..src_off + w]);
            src_row += 1;
            dst_row += step;
        }
    }
    out
}

/// Convenience wrapper to pull a `TimeBase`-tagged frame without wiring
/// up the full decoder trait — used by the container's own smoke tests.
#[allow(dead_code)]
pub(crate) fn decode_packet_frame(params: &CodecParameters, pkt: &Packet) -> Result<VideoFrame> {
    let mut d = make_decoder(params)?;
    d.send_packet(pkt)?;
    match d.receive_frame()? {
        Frame::Video(v) => Ok(v),
        _ => Err(Error::invalid("GIF: decoder returned non-video frame")),
    }
}

#[allow(dead_code)]
pub(crate) const DEFAULT_TIME_BASE: TimeBase = TimeBase::new(1, 100);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterlace_roundtrip() {
        // Build a pattern where every row is filled with its row index.
        let w = 4usize;
        let h = 8usize;
        let mut progressive = Vec::with_capacity(w * h);
        for y in 0..h {
            for _ in 0..w {
                progressive.push(y as u8);
            }
        }
        // Interlace it.
        let mut interlaced = vec![0u8; w * h];
        let mut src_row = 0usize;
        for &(start, step) in &[(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
            let mut r = start;
            while r < h {
                interlaced[src_row * w..src_row * w + w]
                    .copy_from_slice(&progressive[r * w..r * w + w]);
                src_row += 1;
                r += step;
            }
        }
        let restored = deinterlace(&interlaced, w, h);
        assert_eq!(restored, progressive);
    }
}
