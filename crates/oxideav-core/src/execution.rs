//! Runtime hints passed from the executor to codecs and filters.
//!
//! An [`ExecutionContext`] carries advisory information — today only a
//! thread budget — that codecs can use to tune their internal
//! parallelism. Codecs that don't care can ignore it; the default trait
//! method on [`Decoder`](../../oxideav_codec/trait.Decoder.html) /
//! [`Encoder`](../../oxideav_codec/trait.Encoder.html) is a no-op.

/// Advisory runtime information handed to a codec after construction.
///
/// The struct is deliberately tiny for now. New fields can be added
/// without breaking API consumers that already construct the value via
/// [`ExecutionContext::serial`] or [`ExecutionContext::with_threads`].
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    /// Advisory cap on how many threads a codec may use for its own
    /// internal parallelism (slice-parallel decode, GOP-parallel decode,
    /// etc.). Always `≥ 1`. `1` means "caller requests serial execution
    /// from this codec" — obey it unless you have a very good reason.
    pub threads: usize,
}

impl ExecutionContext {
    /// Ask the codec to run strictly single-threaded.
    pub const fn serial() -> Self {
        Self { threads: 1 }
    }

    /// Budget the codec to at most `threads` internal workers. Values
    /// below 1 are clamped up to 1.
    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::serial()
    }
}
