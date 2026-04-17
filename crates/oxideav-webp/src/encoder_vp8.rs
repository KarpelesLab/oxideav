//! `oxideav_codec::Encoder` adapter that produces a full `.webp` file
//! using the VP8 lossy path.
//!
//! Each `send_frame` accepts a single `Yuv420P` video frame, runs it
//! through [`oxideav_vp8::encoder::encode_keyframe`] to get a bare VP8
//! keyframe bitstream, then wraps the bytes in a RIFF/WEBP container
//! with a single `VP8 ` chunk. `receive_packet` returns the complete
//! `.webp` file bytes for that frame.
//!
//! Registered under the crate-level codec id [`crate::CODEC_ID_VP8`]
//! (`"webp_vp8"`), a sibling of the existing `webp_vp8l` lossless id.
//! The corresponding read path is the WebP container demuxer —
//! callers wanting to decode the output can feed the bytes directly
//! to [`crate::decode_webp`], which handles RIFF/WEBP with a `VP8 `
//! chunk out of the box.
//!
//! Scope (v1):
//!   * single-frame still images only (no animated `ANMF` chunks);
//!   * no separate `ALPH` chunk — VP8 is RGB-only and this encoder
//!     emits an opaque image. Callers needing alpha should use the
//!     VP8L (lossless) path, which preserves the alpha channel in
//!     its native RGBA output;
//!   * no `VP8X` extended header — the simple file format suffices
//!     for a baseline lossy encode.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame,
};

use oxideav_vp8::encoder::{encode_keyframe, DEFAULT_QINDEX};

use crate::CODEC_ID_VP8;

/// Factory used by [`crate::register_codecs`] for the `webp_vp8` codec id.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, DEFAULT_QINDEX)
}

/// Build a VP8-lossy WebP encoder with an explicit qindex (0..=127).
/// Lower values produce higher quality at the cost of file size.
pub fn make_encoder_with_qindex(params: &CodecParameters, qindex: u8) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("VP8 WebP encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("VP8 WebP encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(Error::invalid(format!(
            "VP8 WebP encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "VP8 WebP encoder: pixel format {pix:?} not supported — feed Yuv420P (use webp_vp8l for Rgba)"
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(1, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.frame_rate = Some(frame_rate);

    let time_base = TimeBase::new(1, 1000);

    Ok(Box::new(Vp8WebpEncoder {
        output_params,
        width,
        height,
        qindex: qindex.min(127),
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Vp8WebpEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    qindex: u8,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Vp8WebpEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("VP8 WebP encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(format!(
                "VP8 WebP encoder: frame dims {}x{} do not match encoder {}x{}",
                v.width, v.height, self.width, self.height
            )));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::unsupported(format!(
                "VP8 WebP encoder: frame format {:?} must be Yuv420P (use webp_vp8l for Rgba)",
                v.format
            )));
        }
        let bytes = encode_frame_to_webp(self.width, self.height, self.qindex, v)?;
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Run the VP8 keyframe encoder and wrap the output in a minimal
/// RIFF/WEBP container with a single `VP8 ` chunk. Returns the
/// complete `.webp` file bytes.
fn encode_frame_to_webp(
    width: u32,
    height: u32,
    qindex: u8,
    frame: &VideoFrame,
) -> Result<Vec<u8>> {
    let vp8_bytes = encode_keyframe(width, height, qindex, frame)?;
    Ok(wrap_vp8_in_riff(&vp8_bytes))
}

/// Build a simple-file-format WebP container around a pre-encoded VP8
/// keyframe bitstream.
///
/// Layout (all multi-byte ints little-endian):
/// ```text
/// "RIFF"           (4 bytes)
/// riff_size        (4 bytes, u32 LE) = 4 ("WEBP") + 8 (VP8  hdr) + vp8_len + pad
/// "WEBP"           (4 bytes)
/// "VP8 "           (4 bytes, trailing space is significant)
/// vp8_len          (4 bytes, u32 LE)
/// <vp8 keyframe bytes>
/// <0x00 pad byte>  (only if vp8_len is odd — RIFF chunk padding)
/// ```
fn wrap_vp8_in_riff(vp8_bytes: &[u8]) -> Vec<u8> {
    let vp8_len = vp8_bytes.len() as u32;
    let pad = (vp8_len & 1) as usize;
    // RIFF payload size = "WEBP" (4) + VP8 chunk header (8) + payload + pad.
    let riff_size = 4 + 8 + vp8_len as usize + pad;
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8 "); // trailing space is part of the FourCC
    out.extend_from_slice(&vp8_len.to_le_bytes());
    out.extend_from_slice(vp8_bytes);
    if pad == 1 {
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riff_wrapper_layout_even_payload() {
        let payload = vec![0xAAu8; 10];
        let out = wrap_vp8_in_riff(&payload);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        assert_eq!(&out[12..16], b"VP8 ");
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        // 4 (WEBP) + 8 (chunk hdr) + 10 (payload) = 22
        assert_eq!(riff_size, 22);
        let chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]);
        assert_eq!(chunk_len, 10);
        assert_eq!(&out[20..30], &payload[..]);
        // No pad byte for even-length payloads.
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn riff_wrapper_layout_odd_payload_pads() {
        let payload = vec![0x55u8; 11];
        let out = wrap_vp8_in_riff(&payload);
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        // 4 + 8 + 11 + 1 (pad) = 24
        assert_eq!(riff_size, 24);
        // Payload + 1 zero pad byte at the end.
        assert_eq!(out.len(), 32);
        assert_eq!(out[31], 0x00);
    }
}
