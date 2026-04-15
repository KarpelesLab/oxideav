//! ITU-T G.729 (CS-ACELP, 8 kbit/s) decoder — scaffold.
//!
//! What's landed: crate layout, MSB-first bit reader for G.729 packets, a
//! parser for the 80-bit frame layout defined in ITU-T G.729 §3.6
//! (L0..L3 LSP indices, P1/P0 adaptive-codebook delay + parity,
//! C1/S1+C2/S2 fixed-codebook index/sign, GA1/GB1 + GA2/GB2 gains),
//! the 10th-order LPC predictor state (previous-frame LSF buffer for the
//! MA predictor), and the two LSP MA-predictor codebooks (L1 = 128 x 10
//! `q16.16` entries, L2/L3 = 32 x 5 `q16.16` entries) as `const` arrays.
//!
//! What's pending (follow-up): LSP-to-LPC conversion, adaptive-codebook
//! search decode, algebraic fixed-codebook decode, conjugate-structure
//! gain decode, short-term synthesis filter, and the adaptive postfilter.
//!
//! The decoder is registered so the framework can recognise G.729 streams
//! today; `make_decoder` currently returns `Unsupported`.
//!
//! Reference: ITU-T Recommendation G.729 (January 2007 edition) + Annex A.

// Scaffold-only — these lints come off as the decoder body lands.
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod codec;
pub mod decoder;
pub mod lpc;
pub mod lsp_tables;

use oxideav_codec::CodecRegistry;

pub const CODEC_ID_STR: &str = "g729";

/// Number of samples per G.729 frame (10 ms @ 8 kHz).
pub const FRAME_SAMPLES: usize = 80;

/// Number of samples per 5-ms subframe.
pub const SUBFRAME_SAMPLES: usize = 40;

/// Number of subframes per frame.
pub const SUBFRAMES_PER_FRAME: usize = 2;

/// LPC order (10th-order short-term predictor).
pub const LPC_ORDER: usize = 10;

/// Encoded frame size in bytes (80 bits = 10 bytes).
pub const FRAME_BYTES: usize = 10;

/// Sample rate (Hz).
pub const SAMPLE_RATE: u32 = 8_000;

/// Register G.729 with the codec registry.
pub fn register(reg: &mut CodecRegistry) {
    codec::register(reg);
}
