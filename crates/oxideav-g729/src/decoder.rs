//! Top-level G.729 frame decoder — scaffold entry point.
//!
//! The final decoder will produce 80 `S16` samples (10 ms at 8 kHz) per
//! 10-byte packet. Today this module only wires `make_decoder` to the
//! codec registry; it returns `Unsupported` until LSP-to-LPC conversion,
//! adaptive/fixed codebook decode, gain decode, and the synthesis filter
//! are implemented.

use oxideav_codec::Decoder;
use oxideav_core::{CodecParameters, Error, Result};

use crate::bitreader::{parse_frame_params, FrameParams};

/// Build a boxed [`Decoder`] for a G.729 stream. Currently returns
/// [`Error::Unsupported`] — callers can still probe and demux G.729
/// streams via the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::unsupported(
        "G.729 decoder is a scaffold — LSP/LPC conversion, codebook decode, and synthesis pending",
    ))
}

/// Convenience wrapper that just dispatches to the bit-field parser; kept
/// here so downstream test code can access the frame-level entry point
/// without pulling in the submodule path.
pub fn parse_packet(packet: &[u8]) -> Result<FrameParams> {
    parse_frame_params(packet)
}
