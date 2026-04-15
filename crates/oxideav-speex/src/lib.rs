//! Speex (CELP speech codec) — scaffold.
//!
//! What's landed:
//!   * MSB-first bit reader matching `libspeex/bits.c`.
//!   * Full 80-byte Speex-in-Ogg header parser (signature, version
//!     string, rate, mode, channels, VBR, frames-per-packet).
//!   * Mode enum (narrowband 8 kHz, wideband 16 kHz, ultra-wideband
//!     32 kHz) with sub-mode descriptor tables (NB 0..=8, HB 0..=4).
//!   * Named stubs for the LSP / pitch-gain / innovation codebook
//!     tables plus the sub-frame / LPC-order constants so follow-up
//!     work can fill in the numeric content without restructuring.
//!
//! What's *not* landed: the actual CELP synthesis — LSP-to-LPC
//! conversion, long-term (pitch) prediction, fixed-codebook lookup,
//! and the LPC synthesis filter. The decoder is registered so the
//! framework can probe/remux Speex-in-Ogg streams today; `make_decoder`
//! validates the header and then returns `Unsupported`.
//!
//! Reference: Xiph's Speex manual (<https://www.speex.org/docs/manual/speex-manual.pdf>)
//! and the BSD-licensed C implementation at <https://github.com/xiph/speex>.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod codec;
pub mod decoder;
pub mod header;
pub mod modes;
pub mod quant;

use oxideav_codec::CodecRegistry;

pub const CODEC_ID_STR: &str = "speex";

pub fn register(reg: &mut CodecRegistry) {
    codec::register(reg);
}
