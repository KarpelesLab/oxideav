//! Speex (CELP speech codec) — narrowband decoder + Ogg integration.
//!
//! Implements:
//!   * Bit-exact 80-byte Speex header parser (Speex-in-Ogg mapping).
//!   * MSB-first bit reader matching `libspeex/bits.c`.
//!   * Mode + sub-mode descriptors (NB 0..=8).
//!   * Float-mode CELP decoder for narrowband (NB) streams covering
//!     sub-modes 1..=7 (silence/vocoder, 5.95k, 8k, 11k, 15k, 18.2k,
//!     24.6k) and sub-mode 8 (3.95k vocoder + algebraic codebook).
//!   * Float-mode sub-band CELP decoder for wideband (16 kHz) streams
//!     — WB sub-modes 1..=4, QMF synthesis bank, high-band LSP & LPC
//!     synthesis, stochastic codebook + spectral-folding excitation.
//!     See [`wb_decoder`] and [`qmf`].
//!   * Ultra-wideband (32 kHz) is **not yet** implemented; UWB streams
//!     return `Error::Unsupported`. UWB layers a second SB-CELP stage
//!     on top of the WB decoder (the "low band" of the UWB layer is
//!     the WB decoder's full output), and the driver would have to
//!     stack two QMF syntheses — a substantial extension beyond the
//!     WB work done here.
//!
//! Tables (LSP, gain, fixed codebooks) are transcribed from the
//! BSD-licensed Xiph reference (`libspeex/{lsp_tables_nb,gain_table,
//! exc_*_table}.c`) — values only, no derived code.
//!
//! References:
//!   * <https://www.speex.org/docs/manual/speex-manual.pdf>
//!   * RFC 5574 — RTP payload format for Speex.
//!   * <https://github.com/xiph/speex>

#![allow(
    clippy::needless_range_loop,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::manual_range_contains
)]

pub mod bitreader;
pub mod codec;
pub mod decoder;
pub mod exc_tables;
pub mod gain_tables;
pub mod header;
pub mod hexc_tables;
pub mod lsp;
pub mod lsp_tables_nb;
pub mod lsp_tables_wb;
pub mod nb_decoder;
pub mod qmf;
pub mod submodes;
pub mod wb_decoder;
pub mod wb_submodes;

use oxideav_codec::CodecRegistry;

pub const CODEC_ID_STR: &str = "speex";

pub fn register(reg: &mut CodecRegistry) {
    codec::register(reg);
}
