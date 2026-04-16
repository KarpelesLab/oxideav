//! CELT — the MDCT path of Opus (RFC 6716 §4.3) — full pipeline scaffold.
//!
//! What's landed end-to-end (CELT-only Opus packets now produce PCM):
//!
//! * Bit-exact RFC §4.1 range decoder (port of libopus `entdec.c`).
//! * §4.3 / Table 56 frame header symbol decoding (silence, post-filter,
//!   transient, intra).
//! * §4.3.2.1 coarse band-energy (`unquant_coarse_energy`).
//! * §4.3.2.2 fine band-energy (`unquant_fine_energy`).
//! * §4.3.2.3 final fine-energy pass (`unquant_energy_finalise`).
//! * §4.3.3 bit allocation (`rate::clt_compute_allocation`) — alloc table,
//!   trim, dynalloc boost, skip flags, intensity / dual stereo flags,
//!   coded-bands selection, fine-energy / PVQ split.
//! * §4.3.4 PVQ shape decoding (`bands::quant_all_bands`):
//!   - tf_decode, spreading flag, split-band recursion via theta,
//!   - mono / dual-stereo / intensity-stereo,
//!   - PVQ codeword decoder (`cwrs::decode_pulses`) — recurrence form,
//!   - exp_rotation / collapse-mask extraction.
//! * §4.3.5 anti-collapse processing (`bands::anti_collapse`).
//! * §4.3.6 denormalisation (`bands::denormalise_bands`).
//! * §4.3.7 inverse MDCT (`mdct::imdct_sub`) — pre-twiddle → length-N/4
//!   complex FFT (Bluestein for non-power-of-two sizes) → post-twiddle
//!   → mirror, plus window + overlap-add in the opus crate.
//! * §4.3.8 comb pitch post-filter (`post_filter::comb_filter`).
//!
//! Static tables (transcribed from libopus `static_modes_float.h`):
//! `EBAND_5MS`, `E_PROB_MODEL`, `PRED_COEF`/`BETA_COEF`/`BETA_INTRA`,
//! `BAND_ALLOCATION`, `LOG2_FRAC_TABLE`, `LOGN400`, `CACHE_INDEX50`,
//! `CACHE_BITS50`, `CACHE_CAPS50`, `E_MEANS`, `SPREAD_ICDF`, `TRIM_ICDF`,
//! `TF_SELECT_TABLE`, `COMB_FILTER_TAPS`.
//!
//! Known gaps (the pipeline runs and produces audio, but the output is
//! not yet bit-exact with libopus):
//!
//! * **§4.3.7 IMDCT**: uses Bluestein for the N/4 FFT instead of libopus'
//!   bespoke mixed-radix kiss_fft (15·8 split for the long block at LM=3).
//!   Produces audio with comparable RMS but the spectral peak does not
//!   yet match the encoder. A bit-exact port of `kiss_fft.c` against the
//!   stored `fft_state48000_960_*` tables is the next step.
//! * **§4.3.4 quant_all_bands**: the per-iteration `BandCtx` borrow trick
//!   keeps `RangeDecoder` separate from the partition state, but the norm-
//!   buffer / fold pointer arithmetic is not 1:1 with libopus and may pick
//!   the wrong fold source for the second-band edge case (see
//!   `special_hybrid_folding` in libopus, which we omit since CELT-only
//!   doesn't trigger it for `start=0`).
//! * **Anti-collapse seed**: we use the local `state.rng` as the LCG seed
//!   instead of plumbing the live range-coder `rng` through `quant_all_bands`
//!   per libopus.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::double_parens,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::excessive_precision,
    clippy::useless_vec,
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::manual_clamp,
    clippy::needless_late_init,
    clippy::if_same_then_else,
    clippy::nonminimal_bool,
    clippy::comparison_chain,
    clippy::single_match,
    clippy::needless_return,
    clippy::redundant_field_names,
    clippy::redundant_clone,
    clippy::let_and_return,
    clippy::manual_memcpy,
    clippy::ptr_arg,
    clippy::missing_safety_doc,
    clippy::wrong_self_convention,
    clippy::extra_unused_lifetimes,
    clippy::let_unit_value,
    clippy::needless_borrow,
    clippy::precedence,
    clippy::should_implement_trait,
    unused_mut,
    unused_variables,
    unused_assignments,
    clippy::assign_op_pattern,
    clippy::match_like_matches_macro,
    clippy::neg_multiply,
    clippy::int_plus_one
)]

pub mod bands;
pub mod cwrs;
pub mod header;
pub mod laplace;
pub mod mdct;
pub mod post_filter;
pub mod quant_bands;
pub mod range_decoder;
pub mod rate;
pub mod tables;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Error, Result};

pub const CODEC_ID_STR: &str = "celt";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("celt_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    // CELT-only standalone (i.e. without the Opus framing) isn't on the
    // CodecRegistry path today — Opus packets dispatch through `oxideav-opus`,
    // which holds the per-stream CELT state and runs the §4.3 pipeline.
    Err(Error::unsupported(
        "Standalone CELT decoder is not exposed; use the `opus` codec id",
    ))
}
