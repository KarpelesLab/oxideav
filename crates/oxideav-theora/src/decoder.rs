//! Theora video decoder.
//!
//! Status: header parsing (identification + comment + setup) from extradata,
//! codec parameter population, frame-header classification, and full **intra
//! frame** decode. Inter frames still return `Error::Unsupported`.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat as CorePixelFormat, Rational,
    Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::bitreader::BitReader;
use crate::block::IntraFrameDecoder;
use crate::headers::{parse_headers_from_extradata, Headers, PixelFormat};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let headers = if params.extradata.is_empty() {
        None
    } else {
        Some(parse_headers_from_extradata(&params.extradata)?)
    };
    Ok(Box::new(TheoraDecoder {
        codec_id: params.codec_id.clone(),
        headers,
        ready_frames: VecDeque::new(),
        pending_pts: None,
        pending_tb: TimeBase::new(1, 90_000),
        eof: false,
    }))
}

pub struct TheoraDecoder {
    codec_id: CodecId,
    pub(crate) headers: Option<Headers>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
}

/// Theora "frame type": false = intra, true = inter (spec §7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    Intra,
    Inter,
}

/// Classification of a packet: the three headers (bit 0 = 1) or a data frame
/// (bit 0 = 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketKind {
    Identification,
    Comment,
    Setup,
    Frame(FrameType),
}

/// Inspect the leading bits of a Theora packet to identify its kind.
pub fn classify_packet(packet: &[u8]) -> Result<PacketKind> {
    if packet.is_empty() {
        return Err(Error::invalid("empty Theora packet"));
    }
    let b0 = packet[0];
    // Header packet: MSB (bit 0 in Theora numbering) = 1. The next 7 bits are
    // the header type code: 0x80 = ID, 0x81 = comment, 0x82 = setup.
    if b0 & 0x80 != 0 {
        return match b0 {
            0x80 => Ok(PacketKind::Identification),
            0x81 => Ok(PacketKind::Comment),
            0x82 => Ok(PacketKind::Setup),
            other => Err(Error::invalid(format!(
                "Theora: unknown header packet type 0x{other:02X}"
            ))),
        };
    }
    // Frame packet: bit 0 is 0. Bit 1 is FTYPE: 0 = intra, 1 = inter.
    let ftype_bit = (b0 & 0x40) != 0;
    Ok(PacketKind::Frame(if ftype_bit {
        FrameType::Inter
    } else {
        FrameType::Intra
    }))
}

impl TheoraDecoder {
    fn ensure_headers_from_packet(&mut self, packet: &[u8]) -> Result<bool> {
        match classify_packet(packet)? {
            PacketKind::Identification | PacketKind::Comment | PacketKind::Setup => Ok(true),
            PacketKind::Frame(_) => Ok(false),
        }
    }

    fn decode_frame(&mut self, packet: &[u8]) -> Result<()> {
        let Some(headers) = self.headers.as_ref() else {
            return Err(Error::invalid(
                "Theora frame packet before headers were parsed",
            ));
        };
        let kind = classify_packet(packet)?;
        match kind {
            PacketKind::Frame(FrameType::Inter) => {
                return Err(Error::unsupported("theora inter-frame decode: follow-up"));
            }
            PacketKind::Frame(FrameType::Intra) => {}
            _ => return Err(Error::invalid("Theora: expected frame packet")),
        }
        let mut br = BitReader::new(packet);
        // §7.1 frame header prelude (already consumed 2 bits by classify).
        let marker = br.read_bit()?;
        if marker {
            return Err(Error::invalid("Theora: frame marker non-zero"));
        }
        let ftype = br.read_bit()?;
        if ftype {
            // inter, already handled above; defensive.
            return Err(Error::unsupported("theora inter-frame decode: follow-up"));
        }
        let mut intra = IntraFrameDecoder::new(headers);
        intra.read_intra_frame_header(&mut br)?;
        intra.fill_bcoded_intra();
        // §7.4 macro-block modes: intra frame — all INTRA (implicit).
        // §7.6 qii.
        intra.read_qiis(&mut br)?;
        // §7.7.3 DCT coefficients.
        intra.decode_coefficients(&mut br)?;
        // §7.8.2 undo DC prediction.
        intra.undo_dc_prediction();
        // §7.9.4 reconstruction.
        let mut planes_buf = intra.reconstruct();
        // §7.10 loop filter.
        intra.loop_filter(&mut planes_buf);

        // Build VideoFrame. Crop to picture region: picw/pich/picx/picy
        // (the spec's picy is from the BOTTOM).
        let id = &headers.identification;
        let (cropped_planes, out_w, out_h) = crop_to_picture(id, &planes_buf);
        let core_fmt = pixel_format_to_core(id.pf);
        let planes = build_video_planes(&cropped_planes, out_w, out_h, id.pf);
        let frame = VideoFrame {
            format: core_fmt,
            width: out_w,
            height: out_h,
            pts: self.pending_pts,
            time_base: self.pending_tb,
            planes,
        };
        self.ready_frames.push_back(frame);
        Ok(())
    }
}

impl Decoder for TheoraDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        if packet.data.is_empty() {
            return Ok(());
        }
        // In-band header parsing: if the decoder was constructed without
        // extradata, fall through the 3 header packets before accepting data.
        if self.headers.is_none() {
            if self.ensure_headers_from_packet(&packet.data)? {
                // We cannot build `Headers` until we have all three; users
                // passing headers in-band are expected to use extradata
                // instead. Report the current limitation.
                return Err(Error::unsupported(
                    "Theora: in-band header parsing not implemented; pass Xiph-laced extradata",
                ));
            }
            return Err(Error::invalid(
                "Theora: frame packet received before headers",
            ));
        }
        self.decode_frame(&packet.data)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
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

/// Crop three full-frame planes (top-down) to the picture region.
/// Returns `(planes, picture_width, picture_height)`.
fn crop_to_picture(
    id: &crate::headers::Identification,
    full: &[Vec<u8>; 3],
) -> (Vec<Vec<u8>>, u32, u32) {
    let pw = id.picw;
    let ph = id.pich;
    let mut out = Vec::with_capacity(3);
    for pli in 0..3 {
        let (frame_w, frame_h) =
            crate::coded_order::plane_pixel_dims(id.fmbw as u32, id.fmbh as u32, id.pf, pli);
        // Chroma offsets & dimensions scale with subsampling.
        let sx = if pli == 0 { 0 } else { id.pf.chroma_shift_x() };
        let sy = if pli == 0 { 0 } else { id.pf.chroma_shift_y() };
        let picx = id.picx >> sx;
        let picy = id.picy >> sy;
        let out_w = pw >> sx;
        let out_h = ph >> sy;
        let mut plane = Vec::with_capacity((out_w * out_h) as usize);
        // picy is from the BOTTOM of the frame, but our buffer is top-down.
        // The picture's bottom row lives at (frame_h - 1 - picy), its top row
        // at (frame_h - 1 - (picy + out_h - 1)) = (frame_h - picy - out_h).
        let top_row_in_full = frame_h.saturating_sub(picy + out_h);
        for row in 0..out_h {
            let src_row = (top_row_in_full + row) as usize;
            let src_off = src_row * frame_w as usize + picx as usize;
            plane.extend_from_slice(&full[pli][src_off..src_off + out_w as usize]);
        }
        out.push(plane);
    }
    (out, pw, ph)
}

fn build_video_planes(cropped: &[Vec<u8>], pw: u32, ph: u32, pf: PixelFormat) -> Vec<VideoPlane> {
    let sx = pf.chroma_shift_x();
    let sy = pf.chroma_shift_y();
    let y_stride = pw as usize;
    let c_w = pw >> sx;
    let _c_h = ph >> sy;
    vec![
        VideoPlane {
            stride: y_stride,
            data: cropped[0].clone(),
        },
        VideoPlane {
            stride: c_w as usize,
            data: cropped[1].clone(),
        },
        VideoPlane {
            stride: c_w as usize,
            data: cropped[2].clone(),
        },
    ]
}

fn pixel_format_to_core(pf: PixelFormat) -> CorePixelFormat {
    match pf {
        PixelFormat::Yuv420 | PixelFormat::Reserved => CorePixelFormat::Yuv420P,
        PixelFormat::Yuv422 => CorePixelFormat::Yuv422P,
        PixelFormat::Yuv444 => CorePixelFormat::Yuv444P,
    }
}

/// Build a `CodecParameters` for a Theora stream given the identification
/// header.
pub fn codec_parameters_from_identification(
    id: &crate::headers::Identification,
) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(id.picw);
    params.height = Some(id.pich);
    params.pixel_format = Some(id.pf.to_core());
    if id.frd > 0 {
        params.frame_rate = Some(Rational::new(id.frn as i64, id.frd as i64));
    }
    if id.nombr > 0 {
        params.bit_rate = Some(id.nombr as u64);
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_headers() {
        assert_eq!(
            classify_packet(&[0x80]).unwrap(),
            PacketKind::Identification
        );
        assert_eq!(classify_packet(&[0x81]).unwrap(), PacketKind::Comment);
        assert_eq!(classify_packet(&[0x82]).unwrap(), PacketKind::Setup);
    }

    #[test]
    fn classify_frames() {
        // bit 0 clear → frame; bit 1 clear → intra.
        assert_eq!(
            classify_packet(&[0b0000_0000]).unwrap(),
            PacketKind::Frame(FrameType::Intra)
        );
        // bit 0 clear, bit 1 set → inter.
        assert_eq!(
            classify_packet(&[0b0100_0000]).unwrap(),
            PacketKind::Frame(FrameType::Inter)
        );
    }

    #[test]
    fn rejects_empty() {
        assert!(classify_packet(&[]).is_err());
    }
}
