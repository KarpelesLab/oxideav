//! AV1 decoder shim.
//!
//! In this initial parse-only crate the decoder consumes packets and
//! exercises the OBU + sequence-header + frame-header parsers, but never
//! produces a video frame. Tile decode is the remaining work and dwarfs
//! everything else (~20 KLOC of CDF / transforms / intra+inter prediction
//! / loop restoration); see `tile_group::tile_decode_unsupported` for the
//! exact §refs.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::extradata::Av1CodecConfig;
use crate::frame_header::{parse_frame_header, FrameHeader, FrameType};
use crate::obu::{iter_obus, ObuType};
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::tile_decode::TileDecoder;
use crate::tile_group::tile_decode_unsupported;

/// Build the registry-side decoder factory.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Av1Decoder::new(params.clone())))
}

pub struct Av1Decoder {
    codec_id: CodecId,
    seq_header: Option<SequenceHeader>,
    last_frame_header: Option<FrameHeader>,
    last_error: Option<Error>,
    seen_frame: bool,
}

impl Av1Decoder {
    pub fn new(params: CodecParameters) -> Self {
        let mut me = Self {
            codec_id: params.codec_id.clone(),
            seq_header: None,
            last_frame_header: None,
            last_error: None,
            seen_frame: false,
        };
        // Bootstrap from extradata if present (av1C in MP4, codec private in
        // Matroska/WebM). Failures are recorded but not fatal at construction.
        if !params.extradata.is_empty() {
            match Av1CodecConfig::parse(&params.extradata) {
                Ok(cfg) => {
                    if let Some(sh) = cfg.seq_header {
                        me.seq_header = Some(sh);
                    }
                }
                Err(e) => {
                    me.last_error = Some(e);
                }
            }
        }
        me
    }

    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq_header.as_ref()
    }

    pub fn last_frame_header(&self) -> Option<&FrameHeader> {
        self.last_frame_header.as_ref()
    }

    /// Walk the OBU stream in `packet.data`, updating internal state. Returns
    /// the first error encountered (if any).
    fn ingest(&mut self, data: &[u8]) -> Result<()> {
        for obu in iter_obus(data) {
            let obu = obu?;
            match obu.header.obu_type {
                ObuType::TemporalDelimiter | ObuType::Padding => {
                    // Empty / ignored.
                }
                ObuType::SequenceHeader => {
                    self.seq_header = Some(parse_sequence_header(obu.payload)?);
                }
                ObuType::FrameHeader | ObuType::RedundantFrameHeader => {
                    let seq = self.seq_header.as_ref().ok_or_else(|| {
                        Error::invalid("av1: frame_header before sequence_header")
                    })?;
                    let fh = parse_frame_header(seq, obu.payload)?;
                    self.last_frame_header = Some(fh);
                    self.seen_frame = true;
                }
                ObuType::Frame => {
                    let seq = self
                        .seq_header
                        .as_ref()
                        .ok_or_else(|| Error::invalid("av1: frame_obu before sequence_header"))?;
                    // OBU_FRAME = frame_header_obu() + tile_group_obu(). We
                    // parse the header best-effort.
                    let fh = parse_frame_header(seq, obu.payload)?;
                    if !matches!(fh.frame_type, FrameType::Key | FrameType::Inter) {
                        // Other frame types are syntactically supported; no-op.
                    }
                    self.last_frame_header = Some(fh.clone());
                    self.seen_frame = true;
                    // Hand off to the tile-decode skeleton. Even though it
                    // can't produce pixels yet, it surfaces a much more
                    // precise §ref than `tile_decode_unsupported`. Pass an
                    // empty payload if we couldn't compute the split;
                    // `TileDecoder::new` will reject it.
                    let seq_clone = seq.clone();
                    let tile_payload = obu.payload;
                    match TileDecoder::new(&seq_clone, &fh, tile_payload)
                        .and_then(|mut td| td.decode())
                    {
                        Ok(_) => {}
                        Err(Error::Unsupported(s)) => return Err(Error::Unsupported(s)),
                        Err(e) => return Err(e),
                    }
                }
                ObuType::TileGroup => {
                    // Without a tile-payload boundary extracted from the
                    // enclosing OBU_FRAME it's not safe to drive the tile
                    // decoder here. Surface the historical message.
                    return Err(tile_decode_unsupported());
                }
                ObuType::Metadata | ObuType::TileList => {
                    // Metadata is informational; tile_list is for large-scale
                    // tile coding which we don't decode.
                }
                _ => {
                    // Reserved — ignore.
                }
            }
        }
        Ok(())
    }
}

impl Decoder for Av1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Drain ingest errors but don't surface them past the first frame —
        // many callers want to inspect headers even when we can't decode.
        match self.ingest(&packet.data) {
            Ok(()) => Ok(()),
            Err(Error::Unsupported(s)) => {
                // Headers parsed; tile body unsupported. Surface so the caller
                // knows there'll never be frames.
                Err(Error::Unsupported(s))
            }
            Err(e) => Err(e),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(tile_decode_unsupported())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
