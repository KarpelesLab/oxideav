//! Pure-Rust VP8 video decoder + IVF container.
//!
//! Status:
//! * Boolean (arithmetic) decoder — RFC 6386 §7. Done; round-trip tested
//!   against a libvpx-style reference encoder.
//! * Frame tag + uncompressed key-frame chunk — RFC 6386 §9.1. Done.
//! * Frame header (segmentation / loop filter / quant / probs) — §9.2-§9.10.
//!   Done.
//! * Macroblock prediction modes — intra-16×16 + intra-4×4 + intra-8×8 chroma.
//!   All 14 modes implemented.
//! * Token / coefficient decoding — §13. Implemented as a flat unrolled
//!   variant matching libvpx's `GetCoeffs`.
//! * Inverse 4×4 DCT + 4×4 WHT — §14. Done; passes a forward-DCT round-trip
//!   test on a constant block.
//! * Loop filter — §15. Simple + normal modes wired up.
//! * I-frame decode (4:2:0 YUV) — produces correctly-shaped output. The
//!   no-neighbour case (top-left MB) is bit-exact against libvpx; uniform
//!   content streams decode at 100% pixel match. Multi-MB B_PRED-heavy
//!   content like `testsrc` is partially correct — there's an
//!   under-investigation issue in either context propagation between
//!   neighbouring B_PRED macroblocks or the post-IDCT pixel pipeline that
//!   degrades the per-frame pixel-match rate. Tracked in the integration
//!   test `tests/decode_keyframe.rs`.
//! * P-frame decode — structural pipeline in place: parses the inter
//!   header, decodes per-MB mode info (NEAREST/NEAR/ZERO/NEW/SPLIT),
//!   decodes MVs via the 19-entry per-component probability tree,
//!   manages LAST/GOLDEN/ALTREF reference slots with copy-to and
//!   refresh flags, and runs motion compensation via the 6-tap luma
//!   filter + bilinear chroma filter. Gray / static content round-trips
//!   bit-exactly through the keyframe; motion-heavy content currently
//!   suffers from the same B_PRED keyframe neighbour bug noted above
//!   (since P-frames reference the keyframe). A `find_near_mvs`
//!   approximation and sign-bias handling cover common cases; some
//!   corner cases (SPLIT_MV context) are simplified.
//! * IVF container — read-side demuxer with FourCC `VP80` probe.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::too_many_arguments)]
// VP8's bitstream/transform code reads more naturally with explicit shifts
// and bit ops; clippy's identity_op / manual_div_ceil / etc. lints flag a
// number of these as "could be simplified" but the explicit form is what
// the spec mirrors.
#![allow(clippy::identity_op)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_slice_fill)]
#![allow(clippy::let_and_return)]
#![allow(clippy::useless_asref)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::ptr_arg)]

pub mod bool_decoder;
pub mod bool_encoder;
pub mod decoder;
pub mod encoder;
pub mod fdct;
pub mod frame_header;
pub mod frame_tag;
pub mod inter;
pub mod intra;
pub mod ivf;
pub mod loopfilter;
pub mod mv;
pub mod tables;
pub mod tokens;
pub mod transform;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "vp8";

pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("vp8_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    reg.register_decoder_impl(cid.clone(), caps, make_decoder);
    let enc_caps = CodecCapabilities::video("vp8_sw_enc")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(16383, 16383);
    reg.register_encoder_impl(cid, enc_caps, make_encoder);
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    ivf::register(reg);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

/// Combined registration for callers that just want everything.
pub fn register(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

pub use decoder::{decode_frame, Vp8Decoder};
pub use frame_header::{parse_keyframe_header, FrameHeader};
pub use frame_tag::{parse_header, FrameTag, FrameType, KeyframeHeader, ParsedHeader};
