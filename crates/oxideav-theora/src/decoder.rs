//! Theora video decoder.
//!
//! Current milestone: header parsing (identification + comment + setup) from
//! extradata, codec parameter population, frame-header classification, and
//! intra-frame *scaffold*. Actual I-frame block decoding is in flight — the
//! decoder returns `Error::Unsupported` at the point token decoding begins.
//! This is enough for probe/remux pipelines to work with Theora streams today.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, Result, TimeBase, VideoFrame,
};

use crate::bitreader::BitReader;
use crate::headers::{parse_headers_from_extradata, Headers};

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
        let Some(_headers) = self.headers.as_ref() else {
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
        // Parse the frame header so callers at least advance through the
        // stream without surprises. The actual block decode is a follow-up.
        let mut br = BitReader::new(packet);
        let _marker = br.read_bit()?; // must be 0
        let _ftype = br.read_bit()?;
        let _nqis_count = br.read_u32(6)?; // qis_count; drop
        let _reserved = br.read_u32(1)?;
        let _ = br; // avoid unused warning once we bail below.

        Err(Error::unsupported(
            "theora intra-frame block decode: follow-up",
        ))
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
