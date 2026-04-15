//! Codec traits and registry.
//!
//! Format crates implement `Decoder` / `Encoder` and register themselves by
//! building `DecoderFactory` / `EncoderFactory` values. The central `oxideav`
//! aggregator pulls everything together into a single `Registry`.

pub mod registry;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, Result};

/// A packet-to-frame decoder.
pub trait Decoder: Send {
    fn codec_id(&self) -> &CodecId;

    /// Feed one compressed packet. May or may not produce a frame immediately —
    /// call `receive_frame` in a loop afterwards.
    fn send_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Pull the next decoded frame, if any. Returns `Error::NeedMore` when the
    /// decoder needs another packet.
    fn receive_frame(&mut self) -> Result<Frame>;

    /// Signal end-of-stream. After this, `receive_frame` will drain buffered
    /// frames and eventually return `Error::Eof`.
    fn flush(&mut self) -> Result<()>;
}

/// A frame-to-packet encoder.
pub trait Encoder: Send {
    fn codec_id(&self) -> &CodecId;

    /// Parameters describing this encoder's output stream (to feed into a muxer).
    fn output_params(&self) -> &CodecParameters;

    fn send_frame(&mut self, frame: &Frame) -> Result<()>;

    fn receive_packet(&mut self) -> Result<Packet>;

    fn flush(&mut self) -> Result<()>;
}

/// Factory that builds a decoder for a given codec parameter set.
pub type DecoderFactory = fn(params: &CodecParameters) -> Result<Box<dyn Decoder>>;

/// Factory that builds an encoder for a given codec parameter set.
pub type EncoderFactory = fn(params: &CodecParameters) -> Result<Box<dyn Encoder>>;

pub use registry::{CodecImplementation, CodecRegistry};
