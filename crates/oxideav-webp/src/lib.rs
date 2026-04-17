//! Pure-Rust WebP image decoder.
//!
//! WebP is Google's image format. This crate handles every major flavour:
//!
//! * **Simple file format, lossy** — `RIFF WEBP` + `VP8 ` chunk holding a
//!   single VP8 keyframe. The bitstream is fed into `oxideav-vp8`; the
//!   resulting YUV 4:2:0 frame is converted to RGBA for output (consumers of
//!   still images universally expect RGB/RGBA).
//! * **Simple file format, lossless** — `RIFF WEBP` + `VP8L` chunk. The
//!   lossless bitstream (Huffman + LZ77 + color-cache + four transforms) is
//!   decoded from scratch in [`vp8l`] per the WebP Lossless specification.
//!   Output is native RGBA.
//! * **Extended file format** — `RIFF WEBP` + `VP8X` header + optional
//!   `ICCP` / `EXIF` / `XMP ` / `ANIM` / `ANMF` / `ALPH`. We decode the VP8X
//!   flags, stitch `ALPH` onto a `VP8 ` luma path (filtered raw or
//!   VP8L-compressed), and iterate `ANMF` chunks for animation. Unknown
//!   auxiliary chunks are skipped gracefully.
//! * **Animated WebP** — each `ANMF` sub-chunk emits one `VideoFrame` with a
//!   matching PTS/duration expressed in milliseconds. Frame disposal and
//!   blending modes are honoured against an internal RGBA canvas.
//!
//! VP8L lossless encoding is supported through [`encoder::make_encoder`]
//! — a minimal-but-correct pure-Rust encoder (length-limited Huffman +
//! 4 KB-window LZ77, no transforms). VP8 lossy encoding is still out of
//! scope.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_div_ceil)]

pub mod decoder;
pub mod demux;
pub mod encoder;
pub mod vp8l;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

/// Codec id string for the VP8L lossless still-image bitstream. Registered
/// so the codec registry reports it alongside other image codecs.
pub const CODEC_ID_VP8L: &str = "webp_vp8l";

/// Register every codec implementation this crate provides.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_VP8L);
    let caps = CodecCapabilities::video("webp_vp8l_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(16384, 16384);
    reg.register_both(cid, caps, make_vp8l_decoder, make_vp8l_encoder);
}

/// Register the WebP container demuxer + the `.webp` extension + its probe.
pub fn register_containers(reg: &mut ContainerRegistry) {
    demux::register(reg);
}

/// Combined registration for callers that want codecs + containers in one
/// call (matches the pattern used elsewhere in the workspace).
pub fn register(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

fn make_vp8l_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_vp8l_decoder(params)
}

fn make_vp8l_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

pub use decoder::{decode_webp, WebpFrame, WebpImage};
pub use vp8l::encode_vp8l_argb;
