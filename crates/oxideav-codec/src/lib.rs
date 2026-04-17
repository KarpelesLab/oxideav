//! Codec traits and registry.
//!
//! Format crates implement `Decoder` / `Encoder` and register themselves by
//! building `DecoderFactory` / `EncoderFactory` values. The central `oxideav`
//! aggregator pulls everything together into a single `Registry`.

pub mod registry;

use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, Packet, Result};

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

    /// Discard all carry-over state so the decoder can resume from a new
    /// bitstream position without producing stale output. Called by the
    /// player after a container seek.
    ///
    /// Unlike [`flush`](Self::flush) (which signals end-of-stream and
    /// drains buffered frames), `reset` is expected to:
    /// * drop every buffered input packet and pending output frame;
    /// * zero any per-stream filter / predictor / overlap memory so the
    ///   next `send_packet` decodes as if it were the first;
    /// * leave the codec id and stream parameters untouched.
    ///
    /// The default is a conservative "drain-then-forget": call
    /// [`flush`](Self::flush) and ignore any remaining frames. Stateful
    /// codecs (LPC predictors, backward-adaptive gain, IMDCT overlap,
    /// reference pictures, …) should override this to wipe their
    /// internal state explicitly — otherwise the first ~N output
    /// samples after a seek will be glitchy until the state re-adapts.
    fn reset(&mut self) -> Result<()> {
        self.flush()?;
        // Drain any remaining output frames so the next send_packet
        // starts clean. NeedMore / Eof both mean "no more frames"; any
        // other error is surfaced so the caller can see why.
        use oxideav_core::Error;
        loop {
            match self.receive_frame() {
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    }

    /// Advisory: announce the runtime environment (today: a thread budget
    /// for codec-internal parallelism). Called at most once, before the
    /// first `send_packet`. Default no-op; codecs that want to run
    /// slice-/GOP-/tile-parallel override this to capture the budget.
    /// Ignoring the hint is always safe — callers must still work with
    /// a decoder that runs serial.
    fn set_execution_context(&mut self, _ctx: &ExecutionContext) {}
}

/// A frame-to-packet encoder.
pub trait Encoder: Send {
    fn codec_id(&self) -> &CodecId;

    /// Parameters describing this encoder's output stream (to feed into a muxer).
    fn output_params(&self) -> &CodecParameters;

    fn send_frame(&mut self, frame: &Frame) -> Result<()>;

    fn receive_packet(&mut self) -> Result<Packet>;

    fn flush(&mut self) -> Result<()>;

    /// Advisory: announce the runtime environment. Same semantics as
    /// [`Decoder::set_execution_context`].
    fn set_execution_context(&mut self, _ctx: &ExecutionContext) {}
}

/// Factory that builds a decoder for a given codec parameter set.
pub type DecoderFactory = fn(params: &CodecParameters) -> Result<Box<dyn Decoder>>;

/// Factory that builds an encoder for a given codec parameter set.
pub type EncoderFactory = fn(params: &CodecParameters) -> Result<Box<dyn Encoder>>;

pub use registry::{CodecImplementation, CodecRegistry};
