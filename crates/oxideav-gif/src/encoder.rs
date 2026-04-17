//! GIF frame encoder.
//!
//! Accepts `Pal8` video frames. Plane 0 carries palette indices (1 byte
//! per pixel); plane 1 carries a packed RGBA palette (4 bytes per entry,
//! up to 256 entries). The encoder:
//!
//! 1. Picks an LZW minimum code size (`ceil(log2(palette_len))`, clamped
//!    to `[2, 8]`).
//! 2. Compresses the index plane with [`Lzw`], padding the input stride
//!    out of the way so the packed output is plain
//!    `width*height` bytes of raw indices.
//! 3. Emits the result as an `OGIF` packet matching the container's
//!    [`decode_frame_payload`](crate::container::decode_frame_payload)
//!    expectations.
//!
//! The first frame emitted writes its palette into `output_params.extradata`
//! via the muxer path (muxer reads `CodecParameters::extradata`). Every
//! frame also emits a local palette entry by default — that's the simplest
//! way to stay correct when consumers stitch frames with different
//! palettes. Callers who want a global-palette-only GIF can clear the
//! local palette themselves between calls.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::container::{encode_frame_payload, palette_to_extradata, ParsedFrame};
use crate::lzw::Lzw;

/// Default frame delay when the caller doesn't specify one, in GIF time
/// units (1/100 s). 10 cs ≈ 10 fps — a sensible baseline that every
/// viewer handles.
pub const DEFAULT_DELAY_CS: u16 = 10;

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("GIF encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("GIF encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Pal8);
    if pix != PixelFormat::Pal8 {
        return Err(Error::unsupported(format!(
            "GIF encoder: pixel format {:?} not supported — feed Pal8",
            pix
        )));
    }
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::GIF_CODEC_ID);
    output_params.pixel_format = Some(PixelFormat::Pal8);
    output_params.width = Some(width);
    output_params.height = Some(height);

    // Time base: 1/100 s — matches GIF's native delay unit.
    let time_base = TimeBase::new(1, 100);

    Ok(Box::new(GifEncoder {
        output_params,
        width,
        height,
        time_base,
        pending: VecDeque::new(),
        frame_count: 0,
        delay_cs: DEFAULT_DELAY_CS,
        global_palette_set: false,
        buffered: None,
    }))
}

/// A frame that has been LZW-compressed but whose delay is still
/// unknown (we only know how long it's displayed once the NEXT frame
/// arrives with a later pts). We carry everything we need to serialise
/// it to an `OGIF` payload once the delay is resolved.
struct BufferedFrame {
    palette: Vec<[u8; 4]>,
    min_code_size: u8,
    lzw_data: Vec<u8>,
    pts_cs: i64,
}

struct GifEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    frame_count: u64,
    delay_cs: u16,
    global_palette_set: bool,
    /// Most recently received frame — held until either the next
    /// frame or a `flush()` establishes its display duration.
    buffered: Option<BufferedFrame>,
}

impl Encoder for GifEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                if v.format != PixelFormat::Pal8 {
                    return Err(Error::invalid(format!(
                        "GIF encoder: frame format {:?} != Pal8",
                        v.format
                    )));
                }
                if v.width != self.width || v.height != self.height {
                    return Err(Error::invalid(
                        "GIF encoder: frame dimensions do not match encoder config",
                    ));
                }
                let palette = extract_palette(v)?;
                let indices = pack_indices(v);
                let min_code_size = min_code_size_for(palette.len());

                let mut lzw_data = Vec::new();
                let mut enc = Lzw::encoder(min_code_size)?;
                enc.write(&indices, &mut lzw_data);
                enc.finish(&mut lzw_data);

                // Normalise pts into centiseconds — our encoder time base.
                let pts_cs = v
                    .pts
                    .map(|p| v.time_base.rescale(p, TimeBase::new(1, 100)))
                    .unwrap_or(self.frame_count as i64 * self.delay_cs as i64);

                // If we have a buffered previous frame, resolve its
                // delay from the pts-delta and emit it as a packet.
                if let Some(prev) = self.buffered.take() {
                    let delta = (pts_cs - prev.pts_cs).max(1);
                    let delay = delta.min(u16::MAX as i64) as u16;
                    self.emit(prev, delay);
                }

                self.buffered = Some(BufferedFrame {
                    palette,
                    min_code_size,
                    lzw_data,
                    pts_cs,
                });
                Ok(())
            }
            _ => Err(Error::invalid("GIF encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(prev) = self.buffered.take() {
            // The last frame uses the default delay — nothing after it
            // defines the display duration.
            self.emit(prev, self.delay_cs);
        }
        Ok(())
    }
}

impl GifEncoder {
    fn emit(&mut self, bf: BufferedFrame, delay_cs: u16) {
        let parsed = ParsedFrame {
            x: 0,
            y: 0,
            w: self.width,
            h: self.height,
            delay_cs,
            disposal: 0,
            transparent_index: None,
            interlaced: false,
            min_code_size: bf.min_code_size,
            local_palette: bf.palette.clone(),
            lzw_data: bf.lzw_data,
        };
        if !self.global_palette_set {
            self.output_params.extradata = palette_to_extradata(&bf.palette);
            self.global_palette_set = true;
        }
        let data = encode_frame_payload(&parsed, (self.width, self.height));
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(bf.pts_cs);
        pkt.dts = pkt.pts;
        pkt.duration = Some(delay_cs as i64);
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        self.frame_count += 1;
    }
}

/// Compute `ceil(log2(max(2, palette_len)))`, clamped to `[2, 8]`. GIF
/// requires the LZW initial-code-width to fit the whole palette plus
/// the two reserved codes, and the minimum alphabet width is 2 bits.
fn min_code_size_for(palette_len: usize) -> u8 {
    let n = palette_len.max(2) as u32;
    let bits = 32 - (n - 1).leading_zeros();
    bits.clamp(2, 8) as u8
}

fn extract_palette(v: &VideoFrame) -> Result<Vec<[u8; 4]>> {
    if v.planes.len() < 2 {
        return Err(Error::invalid(
            "GIF encoder: Pal8 frame missing palette plane",
        ));
    }
    let p = &v.planes[1];
    // The palette plane is RGBA bytes, padded to at most 256 entries.
    let n = p.data.len() / 4;
    let mut out = Vec::with_capacity(n.min(256));
    for i in 0..n.min(256) {
        out.push([
            p.data[i * 4],
            p.data[i * 4 + 1],
            p.data[i * 4 + 2],
            p.data[i * 4 + 3],
        ]);
    }
    // Trim trailing zero-alpha entries to shrink the effective palette
    // if the caller padded. We keep at least 2 entries to stay within
    // LZW's minimum code-size range.
    while out.len() > 2
        && out
            .last()
            .map(|c| c[3] == 0xFF && c[0] == 0 && c[1] == 0 && c[2] == 0)
            .unwrap_or(false)
    {
        // But don't aggressively strip — palette entries can legitimately
        // be black. Only trim if the caller provided an obvious sentinel
        // block of pure zeros. Conservative approach: stop trimming.
        break;
    }
    Ok(out)
}

fn pack_indices(v: &VideoFrame) -> Vec<u8> {
    let w = v.width as usize;
    let h = v.height as usize;
    let plane = &v.planes[0];
    let stride = plane.stride;
    if stride == w {
        plane.data[..w * h].to_vec()
    } else {
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            out.extend_from_slice(&plane.data[row * stride..row * stride + w]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_code_size_basics() {
        assert_eq!(min_code_size_for(2), 2);
        assert_eq!(min_code_size_for(4), 2);
        assert_eq!(min_code_size_for(5), 3);
        assert_eq!(min_code_size_for(8), 3);
        assert_eq!(min_code_size_for(16), 4);
        assert_eq!(min_code_size_for(256), 8);
    }
}
