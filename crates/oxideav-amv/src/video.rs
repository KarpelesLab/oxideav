//! AMV video decoder + encoder.
//!
//! AMV stores each frame as a stripped-down JPEG that begins with `FFD8`
//! (SOI), continues with raw entropy-coded data, and ends with `FFD9` (EOI).
//! The standard JPEG segments (DQT, DHT, SOF0, SOS) are *omitted* — the
//! decoder is expected to know they're the Annex K standard tables for a
//! 4:2:0 baseline JPEG with whatever width/height the container reported.
//!
//! Decode strategy: synthesise a real baseline JPEG by sandwiching the AMV
//! entropy payload between a hand-built header and the original `FFD9`, then
//! feed the result to the standard `oxideav-mjpeg` decoder. After decode,
//! flip the frame vertically — AMV stores the picture upside-down on disk
//! (the one and only AMV-specific line in ffmpeg's `mjpegdec.c`).
//!
//! Encode strategy: vertically flip the input YUV420P frame's planes, run
//! the standard `oxideav-mjpeg` encoder with its default Annex-K tables at
//! Q50 (the tables the AMV decoder expects), then strip the header segments
//! (JFIF APP0, DQT, DHT, SOF0, SOS) from the emitted JPEG so only
//! `FFD8` + entropy scan + `FFD9` remains. That stripped blob is exactly
//! one AMV `00dc` video-chunk payload.

use std::collections::VecDeque;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use oxideav_mjpeg::decoder::make_decoder as make_mjpeg_decoder;
use oxideav_mjpeg::encoder::encode_jpeg;
use oxideav_mjpeg::jpeg::huffman::{
    STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS, STD_AC_LUMA_BITS, STD_AC_LUMA_VALS, STD_DC_CHROMA_BITS,
    STD_DC_CHROMA_VALS, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS,
};
use oxideav_mjpeg::jpeg::markers;
use oxideav_mjpeg::jpeg::quant::{DEFAULT_CHROMA_Q50, DEFAULT_LUMA_Q50};
use oxideav_mjpeg::jpeg::zigzag::ZIGZAG;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("AMV video: missing width in CodecParameters"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("AMV video: missing height in CodecParameters"))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid("AMV video: zero width/height"));
    }

    // Inner MJPEG decoder is built per-packet (it's stateless across packets
    // and only carries `pending`), but we keep one around for re-use.
    let mut mjpeg_params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
    mjpeg_params.width = Some(width);
    mjpeg_params.height = Some(height);
    mjpeg_params.pixel_format = Some(PixelFormat::Yuv420P);
    let inner = make_mjpeg_decoder(&mjpeg_params)?;

    Ok(Box::new(AmvVideoDecoder {
        codec_id: CodecId::new(crate::VIDEO_CODEC_ID_STR),
        width,
        height,
        inner,
        pending: None,
        eof: false,
    }))
}

struct AmvVideoDecoder {
    codec_id: CodecId,
    width: u32,
    height: u32,
    inner: Box<dyn Decoder>,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for AmvVideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "AMV video: receive_frame must be called before sending another packet",
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

        // Synthesise a real JPEG from the AMV chunk payload.
        let jpeg = synthesise_jpeg(&pkt.data, self.width, self.height)?;
        let mut shim = Packet::new(pkt.stream_index, pkt.time_base, jpeg);
        shim.pts = pkt.pts;
        shim.dts = pkt.dts;
        shim.duration = pkt.duration;
        shim.flags = pkt.flags;

        self.inner.send_packet(&shim)?;
        let frame = self.inner.receive_frame()?;
        match frame {
            Frame::Video(mut vf) => {
                flip_vertically(&mut vf);
                Ok(Frame::Video(vf))
            }
            other => Ok(other),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        self.inner.flush()
    }
}

/// Wrap an AMV entropy payload in a synthesised baseline-JPEG header so a
/// stock JPEG decoder will accept it. AMV chunks are guaranteed to start
/// with `FFD8` and end with `FFD9`; we strip the SOI, prepend our own
/// header, and re-emit everything from the original entropy data through
/// the EOI verbatim.
fn synthesise_jpeg(amv_payload: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    if amv_payload.len() < 4 {
        return Err(Error::invalid("AMV video: chunk shorter than 4 bytes"));
    }
    if amv_payload[0] != 0xFF || amv_payload[1] != markers::SOI {
        return Err(Error::invalid("AMV video: chunk does not start with SOI"));
    }
    // Tail must be EOI. We don't strictly require it (some encoders drop it)
    // but it's the documented layout.
    let entropy_start = 2usize;
    let entropy_end = amv_payload.len();

    let mut out: Vec<u8> = Vec::with_capacity(amv_payload.len() + 1024);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // DQT — luma + chroma, Q50, 8-bit precision. Reshuffle into zigzag for
    // the wire format.
    write_dqt(&mut out, 0, &DEFAULT_LUMA_Q50);
    write_dqt(&mut out, 1, &DEFAULT_CHROMA_Q50);
    // SOF0 — baseline, 8-bit, 3 components, 4:2:0 (Y H=2 V=2, Cb/Cr H=1 V=1).
    write_sof0(&mut out, width as u16, height as u16);
    // DHT — all four standard Annex K tables.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    // SOS — interleaved scan over all 3 components, Ss=0 Se=63 Ah=Al=0.
    write_sos(&mut out);
    // Entropy bytes (and trailing EOI from the AMV chunk) — copy verbatim.
    out.extend_from_slice(&amv_payload[entropy_start..entropy_end]);
    // If the source somehow lacked an EOI, append one defensively.
    let n = out.len();
    if n < 2 || out[n - 2] != 0xFF || out[n - 1] != markers::EOI {
        out.push(0xFF);
        out.push(markers::EOI);
    }
    Ok(out)
}

fn write_length_prefix(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let len = (payload.len() + 2) as u16;
    out.push(0xFF);
    out.push(marker);
    out.push((len >> 8) as u8);
    out.push(len as u8);
    out.extend_from_slice(payload);
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, nat_order: &[u16; 64]) {
    let mut payload = Vec::with_capacity(1 + 64);
    payload.push(table_id & 0x0F); // precision=0, id=table_id
                                   // DQT carries values in zigzag order, but `nat_order` is row-major; use
                                   // ZIGZAG to map zigzag-index k → natural-index.
    for k in 0..64 {
        payload.push(nat_order[ZIGZAG[k]].min(255) as u8);
    }
    write_length_prefix(out, markers::DQT, &payload);
}

fn write_sof0(out: &mut Vec<u8>, width: u16, height: u16) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // components
                     // Y: id=1, H=2 V=2 (4:2:0 luma oversampling), Q-table id 0.
    payload.push(1);
    payload.push((2u8 << 4) | 2u8);
    payload.push(0);
    // Cb: id=2, H=1 V=1, Q-table id 1.
    payload.push(2);
    payload.push(0x11);
    payload.push(1);
    // Cr: id=3, H=1 V=1, Q-table id 1.
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF0, &payload);
}

fn write_dht(out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]) {
    let mut payload = Vec::with_capacity(1 + 16 + values.len());
    payload.push(((class & 0x01) << 4) | (id & 0x0F));
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_length_prefix(out, markers::DHT, &payload);
}

fn write_sos(out: &mut Vec<u8>) {
    let payload: [u8; 10] = [
        3, // components
        1, 0x00, // Y uses DC=0 AC=0
        2, 0x11, // Cb uses DC=1 AC=1
        3, 0x11, // Cr uses DC=1 AC=1
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Flip a planar `VideoFrame`'s planes vertically in place. Works for
/// `Yuv420P`, `Yuv422P`, `Yuv444P`, and `Gray8` — the only formats the
/// underlying MJPEG decoder produces. Each plane is flipped row-by-row
/// using its own `stride`/height.
fn flip_vertically(vf: &mut VideoFrame) {
    let (luma_h, chroma_h) = match vf.format {
        PixelFormat::Yuv420P => (vf.height as usize, vf.height.div_ceil(2) as usize),
        PixelFormat::Yuv422P | PixelFormat::Yuv444P => (vf.height as usize, vf.height as usize),
        PixelFormat::Gray8 => (vf.height as usize, 0),
        _ => return,
    };
    for (i, plane) in vf.planes.iter_mut().enumerate() {
        let h = if i == 0 { luma_h } else { chroma_h };
        if h <= 1 {
            continue;
        }
        let stride = plane.stride;
        if stride == 0 {
            continue;
        }
        let data = &mut plane.data;
        let mut top = 0usize;
        let mut bot = h - 1;
        // Use a small scratch buffer for one row swap.
        let mut tmp = vec![0u8; stride];
        while top < bot {
            let (a_off, b_off) = (top * stride, bot * stride);
            // Bounds-check before slicing.
            if a_off + stride > data.len() || b_off + stride > data.len() {
                break;
            }
            tmp.copy_from_slice(&data[a_off..a_off + stride]);
            data.copy_within(b_off..b_off + stride, a_off);
            data[b_off..b_off + stride].copy_from_slice(&tmp);
            top += 1;
            bot -= 1;
        }
    }
}

// ---- Encoder -------------------------------------------------------------

/// Build an AMV video encoder. Accepts YUV420P input at the configured
/// resolution; each received `VideoFrame` is emitted as exactly one packet
/// whose payload is the AMV `00dc` chunk body (`FFD8` + raw entropy + `FFD9`,
/// no JPEG header segments).
///
/// Internally we call [`encode_jpeg`] at JPEG quality 50 — Q50 is the fixed
/// point of `scale_for_quality` and therefore emits the unscaled Annex K
/// "standard" tables, which are exactly the tables the AMV *decoder*
/// synthesises when it rebuilds the header for each chunk. Using any other
/// quality factor would produce scaled quant tables that the decoder's
/// hand-rolled header can't recover, destroying the roundtrip. That's a
/// protocol-level constraint of AMV, not a knob we expose.
///
/// Real-world AMV files from cheap MP3/MP4 players also use Q50 Annex K
/// quant tables — that's the whole reason AMV can get away with dropping
/// DQT/DHT/SOF/SOS from the wire format in the first place.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("AMV video encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("AMV video encoder: missing height"))?;
    if width == 0 || height == 0 {
        return Err(Error::invalid("AMV video encoder: zero width/height"));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "AMV video encoder: pixel format {:?} not supported (need Yuv420P)",
            pix
        )));
    }

    let mut output_params = params.clone();
    output_params.codec_id = CodecId::new(crate::VIDEO_CODEC_ID_STR);
    output_params.media_type = MediaType::Video;
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(PixelFormat::Yuv420P);

    let time_base = params
        .frame_rate
        .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));

    Ok(Box::new(AmvVideoEncoder {
        output_params,
        width,
        height,
        time_base,
        pending: VecDeque::new(),
    }))
}

/// Fixed JPEG quality factor — Q50 emits the unscaled Annex K standard
/// quant tables, which is what the AMV decoder hard-codes when synthesising
/// the stripped header back into a real JPEG.
const AMV_JPEG_QUALITY: u8 = 50;

struct AmvVideoEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
}

impl Encoder for AmvVideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid("AMV video encoder: video frames only"));
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "AMV video encoder: frame dimensions do not match encoder config",
            ));
        }
        if v.format != PixelFormat::Yuv420P {
            return Err(Error::invalid(format!(
                "AMV video encoder: frame format {:?} not supported (need Yuv420P)",
                v.format
            )));
        }

        // Clone + flip vertically so the JPEG bitstream carries the
        // upside-down picture AMV stores on disk. The decoder flips it
        // back on output.
        let mut flipped = v.clone();
        flip_vertically(&mut flipped);

        let jpeg = encode_jpeg(&flipped, AMV_JPEG_QUALITY)?;
        let amv_payload = strip_jpeg_headers(&jpeg)?;
        let mut out = Packet::new(0, self.time_base, amv_payload);
        out.pts = v.pts;
        out.dts = v.pts;
        out.flags.keyframe = true;
        self.pending.push_back(out);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // Stateless encoder: nothing buffered internally, nothing to drain.
        Ok(())
    }
}

/// Walk a standalone baseline JPEG and strip every header segment the AMV
/// decoder knows how to reconstruct from the container (JFIF APP0, DQT, DHT,
/// SOF0, DRI, SOS). Returns a buffer shaped like an AMV frame envelope:
/// `FF D8` + the raw entropy scan + `FF D9`.
///
/// Segment walker follows the standard Annex B rules: each marker is `FF xx`;
/// "stand-alone" markers (SOI / EOI / RSTn) carry no payload; all other
/// markers are length-prefixed (big-endian u16 including the length bytes
/// themselves); the entropy scan begins immediately after the SOS payload
/// and runs up to the next non-stuffed, non-RST marker.
fn strip_jpeg_headers(jpeg: &[u8]) -> Result<Vec<u8>> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != markers::SOI {
        return Err(Error::invalid(
            "AMV video encoder: MJPEG output missing SOI",
        ));
    }
    let mut out = Vec::with_capacity(jpeg.len());
    out.push(0xFF);
    out.push(markers::SOI);

    let mut i = 2usize;
    let mut scan_started = false;
    while i < jpeg.len() {
        if jpeg[i] != 0xFF {
            return Err(Error::invalid(
                "AMV video encoder: MJPEG output not marker-aligned",
            ));
        }
        // Skip any 0xFF fill bytes between markers.
        while i < jpeg.len() && jpeg[i] == 0xFF {
            i += 1;
        }
        if i >= jpeg.len() {
            break;
        }
        let marker = jpeg[i];
        i += 1;

        match marker {
            // 0x00 after 0xFF is a stuffed zero; shouldn't appear outside an
            // entropy scan. Treat defensively as error.
            0x00 => {
                return Err(Error::invalid(
                    "AMV video encoder: unexpected 0xFF 0x00 outside entropy scan",
                ));
            }
            markers::EOI => {
                out.push(0xFF);
                out.push(markers::EOI);
                return Ok(out);
            }
            m if markers::is_rst(m) => {
                // Restart markers belong to the scan; copy them through.
                if scan_started {
                    out.push(0xFF);
                    out.push(m);
                }
            }
            markers::SOS => {
                // Length-prefixed SOS payload; skip it, then the entropy scan
                // follows up to the next marker.
                let (payload_len, body_start) = read_segment_length(jpeg, i)?;
                let scan_start = body_start + payload_len;
                // Scan continues until we hit a non-stuffed, non-RST marker.
                let mut j = scan_start;
                while j < jpeg.len() {
                    if jpeg[j] == 0xFF && j + 1 < jpeg.len() {
                        let nxt = jpeg[j + 1];
                        if nxt == 0x00 || markers::is_rst(nxt) {
                            j += 2;
                            continue;
                        }
                        // Real marker — end the scan here.
                        break;
                    }
                    j += 1;
                }
                // Copy scan bytes verbatim (stuffed 0x00s, RSTn, everything).
                out.extend_from_slice(&jpeg[scan_start..j]);
                scan_started = true;
                i = j;
            }
            _ => {
                // Any other marker — DQT, DHT, SOF0, APP*, COM, DRI, … —
                // is length-prefixed. Skip its whole segment without copying.
                let (payload_len, body_start) = read_segment_length(jpeg, i)?;
                i = body_start + payload_len;
            }
        }
    }
    // Missing EOI: emit one defensively so the decoder is happy.
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Read a two-byte big-endian segment length at `i` (pointing at the first
/// length byte). Returns `(payload_bytes_after_length, body_start_offset)`
/// where body_start_offset is `i + 2`.
fn read_segment_length(jpeg: &[u8], i: usize) -> Result<(usize, usize)> {
    if i + 2 > jpeg.len() {
        return Err(Error::invalid(
            "AMV video encoder: truncated JPEG segment length",
        ));
    }
    let declared = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
    if declared < 2 {
        return Err(Error::invalid(
            "AMV video encoder: invalid JPEG segment length",
        ));
    }
    let payload_len = declared - 2;
    let body_start = i + 2;
    if body_start + payload_len > jpeg.len() {
        return Err(Error::invalid(
            "AMV video encoder: JPEG segment length overruns buffer",
        ));
    }
    Ok((payload_len, body_start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    #[test]
    fn synthesise_rejects_non_soi() {
        let r = synthesise_jpeg(&[0x00, 0x00, 0x00, 0x00], 16, 16);
        assert!(r.is_err());
    }

    #[test]
    fn synthesise_rejects_too_short() {
        let r = synthesise_jpeg(&[0xFF, 0xD8, 0xFF], 16, 16);
        assert!(r.is_err());
    }

    #[test]
    fn synthesise_emits_soi_dqt_dht_sof_sos() {
        // Minimal AMV-shaped chunk: SOI + (zero entropy) + EOI.
        let payload = [0xFF, markers::SOI, 0xFF, markers::EOI];
        let out = synthesise_jpeg(&payload, 16, 16).unwrap();
        // Should start with SOI.
        assert_eq!(&out[0..2], &[0xFF, markers::SOI]);
        // Should contain DQT/DHT/SOF0/SOS markers somewhere.
        let mut i = 2usize;
        let mut have_dqt = false;
        let mut have_dht = false;
        let mut have_sof = false;
        let mut have_sos = false;
        while i + 1 < out.len() {
            if out[i] == 0xFF {
                match out[i + 1] {
                    m if m == markers::DQT => have_dqt = true,
                    m if m == markers::DHT => have_dht = true,
                    m if m == markers::SOF0 => have_sof = true,
                    m if m == markers::SOS => have_sos = true,
                    _ => {}
                }
            }
            i += 1;
        }
        assert!(have_dqt && have_dht && have_sof && have_sos);
    }

    #[test]
    fn flip_vertically_yuv420p() {
        // 4×4 Y plane, 2×2 Cb/Cr planes. Rows are 0,1,2,3 in luma.
        let mut vf = VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 4,
            height: 4,
            pts: None,
            time_base: TimeBase::new(1, 30),
            planes: vec![
                oxideav_core::frame::VideoPlane {
                    stride: 4,
                    data: vec![
                        0, 0, 0, 0, // row 0
                        1, 1, 1, 1, // row 1
                        2, 2, 2, 2, // row 2
                        3, 3, 3, 3, // row 3
                    ],
                },
                oxideav_core::frame::VideoPlane {
                    stride: 2,
                    data: vec![10, 10, 20, 20],
                },
                oxideav_core::frame::VideoPlane {
                    stride: 2,
                    data: vec![30, 30, 40, 40],
                },
            ],
        };
        flip_vertically(&mut vf);
        assert_eq!(
            vf.planes[0].data,
            vec![3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(vf.planes[1].data, vec![20, 20, 10, 10]);
        assert_eq!(vf.planes[2].data, vec![40, 40, 30, 30]);
    }
}
