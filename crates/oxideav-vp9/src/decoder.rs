//! VP9 decoder facade.
//!
//! Status: parses the §6.2 uncompressed header for every packet, populates
//! `CodecParameters` (width/height/pixel_format), and surfaces a clean
//! `Error::Unsupported` for the actual pixel-reconstruction step. This is
//! enough for higher layers to probe a VP9 stream, mux/remux it, and
//! enumerate frames.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, VideoFrame,
};

use crate::compressed_header::parse_compressed_header;
use crate::headers::{parse_uncompressed_header, ColorConfig, FrameType, UncompressedHeader};

/// Build a `CodecParameters` from a parsed uncompressed header.
pub fn codec_parameters_from_header(h: &UncompressedHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(h.width);
    params.height = Some(h.height);
    params.pixel_format = Some(pixel_format_from_color_config(&h.color_config));
    params
}

/// Map VP9 color_config (subsampling + bit depth) to the closest oxideav
/// `PixelFormat`. We only have unsubsampled / 4:2:0 in core today, so
/// 4:2:2 / 4:4:4 / 10-bit / 12-bit fall back to `Yuv420P` until core
/// gains the missing variants.
pub fn pixel_format_from_color_config(cc: &ColorConfig) -> PixelFormat {
    // Only 8-bit 4:2:0 maps cleanly today.
    let _ = cc;
    PixelFormat::Yuv420P
}

/// Factory used by the codec registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Vp9Decoder::new(params.codec_id.clone())))
}

pub struct Vp9Decoder {
    codec_id: CodecId,
    /// Last-seen color_config — needed for non-key frames (§6.2.1).
    last_color_config: Option<ColorConfig>,
    /// Last-parsed header, kept for inspection.
    last_header: Option<UncompressedHeader>,
    /// Decoded frames waiting to be drained. Always empty in the current
    /// scaffold (we don't reconstruct pixels yet).
    ready_frames: VecDeque<VideoFrame>,
    eof: bool,
}

impl Vp9Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            last_color_config: None,
            last_header: None,
            ready_frames: VecDeque::new(),
            eof: false,
        }
    }

    pub fn last_header(&self) -> Option<&UncompressedHeader> {
        self.last_header.as_ref()
    }

    /// Parse one packet and update internal state. Returns `Ok(())` on
    /// successful header parse — decoding the actual residual is
    /// `Unsupported` and reported by `receive_frame`.
    fn ingest(&mut self, packet: &Packet) -> Result<()> {
        let h = parse_uncompressed_header(&packet.data, self.last_color_config)?;
        if h.show_existing_frame {
            self.last_header = Some(h);
            return Ok(());
        }
        if h.frame_type == FrameType::Key || h.intra_only {
            self.last_color_config = Some(h.color_config);
        }
        // Best-effort compressed-header parse — failing here doesn't
        // invalidate the stream-level metadata.
        if h.header_size > 0 {
            let cmp_start = h.uncompressed_header_size;
            let cmp_end = cmp_start.saturating_add(h.header_size as usize);
            if cmp_end <= packet.data.len() {
                let _ = parse_compressed_header(&packet.data[cmp_start..cmp_end], &h);
            }
        }
        self.last_header = Some(h);
        Ok(())
    }
}

impl Decoder for Vp9Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.ingest(packet)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        // We parsed a header but cannot reconstruct pixels. Tile skeleton,
        // range decoder, intra primitives (DC/V/H) and 4×4 / 8×8 iDCT are
        // wired up as primitives; the missing piece is partition syntax.
        if self.last_header.is_some() {
            return Err(Error::unsupported(
                "vp9 tile_decode: partition syntax §6.4.3 not implemented — \
                 range decode + intra primitives (DC/V/H, §8.5.1) + \
                 iDCT 4×4/8×8 (§8.7.1) are available as primitives",
            ));
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Helper that returns frame_rate from container-supplied stream timing
/// when available — VP9 itself doesn't carry frame_rate in-band.
pub fn frame_rate_from_container(num: i64, den: i64) -> Option<Rational> {
    if num > 0 && den > 0 {
        Some(Rational::new(num, den))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    /// Build the same synthetic 64×64 key-frame header as the parser unit
    /// test, manually here so the decoder's test doesn't have to reach
    /// into the headers test module.
    fn synth_key_frame_header() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write(2, 2);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(1, 1);
        bw.write(0, 1);
        bw.write(0x49, 8);
        bw.write(0x83, 8);
        bw.write(0x42, 8);
        bw.write(1, 3);
        bw.write(0, 1);
        bw.write(63, 16);
        bw.write(63, 16);
        bw.write(0, 1);
        bw.write(1, 1);
        bw.write(0, 1);
        bw.write(0, 2);
        bw.write(0, 6);
        bw.write(0, 3);
        bw.write(0, 1);
        bw.write(60, 8);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 1);
        bw.write(0, 16);
        bw.finish()
    }

    struct BitWriter {
        out: Vec<u8>,
        cur: u8,
        bits: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                cur: 0,
                bits: 0,
            }
        }
        fn write(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                let b = ((value >> i) & 1) as u8;
                self.cur = (self.cur << 1) | b;
                self.bits += 1;
                if self.bits == 8 {
                    self.out.push(self.cur);
                    self.cur = 0;
                    self.bits = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bits > 0 {
                self.cur <<= 8 - self.bits;
                self.out.push(self.cur);
            }
            self.out.extend_from_slice(&[0u8; 4]);
            self.out
        }
    }

    #[test]
    fn unsupported_after_header_parse() {
        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let params = CodecParameters::video(codec_id);
        let mut d = make_decoder(&params).unwrap();
        let buf = synth_key_frame_header();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), buf);
        d.send_packet(&pkt).unwrap();
        match d.receive_frame() {
            Err(Error::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
