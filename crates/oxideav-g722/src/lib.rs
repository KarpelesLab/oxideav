//! ITU-T G.722 wideband sub-band ADPCM codec (64 kbit/s mode).
//!
//! G.722 compresses 7 kHz-bandwidth speech / audio sampled at 16 kHz mono
//! into a nominal 64 kbit/s bitstream. The codec works by splitting the
//! input into two 4 kHz sub-bands via a 24-tap QMF, and running a
//! backward-adaptive ADPCM quantiser on each band independently:
//!
//! - Low-band (0..4 kHz): 6-bit quantiser with a 2-pole / 6-zero predictor.
//! - High-band (4..8 kHz): 2-bit quantiser with the same predictor topology.
//!
//! On encode the two bands are packed into an 8-bit sample-serial byte:
//! low-band in the high 6 bits, high-band in the low 2 bits. One byte
//! therefore describes two PCM samples worth of audio — hence 64 kbit/s
//! at 16 kHz (8 bits × 8000 packed pairs / sec).
//!
//! # Scope
//!
//! This crate ships a full decoder and encoder for the 64 kbit/s mode only.
//! The 56 and 48 kbit/s reduced-rate modes (which drop low-band bits) are
//! not yet implemented — callers that pass `CodecParameters::bit_rate =
//! Some(56_000)` or `Some(48_000)` get [`Error::Unsupported`] from the
//! factory.
//!
//! Also not yet handled:
//!
//! - The optional G.722 sync word / framing — this crate is transparent on
//!   byte boundaries, leaving framing to the surrounding container.
//! - Explicit reset-on-keyframe — the decoder runs continuously across
//!   packet boundaries, which matches how G.722 is typically carried in
//!   RTP / SIP.
//! - Bit-exact compatibility with the ITU-T Table 6 / 7 / 8 quantiser and
//!   log-scale adapter. The QMF coefficients and structure match
//!   SpanDSP/libg722 exactly (and pass a pure-QMF >30 dB roundtrip), but the
//!   ADPCM predictor + scale adapter use a simpler, self-consistent rule
//!   (`band_low.rs` / `band_high.rs`). Encoder and decoder share the same
//!   update rule so they stay in lock-step, giving a >35 dB PSNR round trip
//!   on a 1 kHz sine, but the on-wire bytes are not directly interchangeable
//!   with other G.722 implementations yet. See the module docs of
//!   [`band_low`] and [`band_high`] for details.
//!
//! All three are tracked for a later follow-up.
//!
//! Reference: ITU-T Recommendation G.722 (09/2012) and its Annex I public-domain
//! reference implementation, plus SpanDSP / libg722 (Steve Underwood, public
//! domain) for the QMF coefficient convention.

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod band_high;
pub mod band_low;
pub mod decoder;
pub mod encoder;
pub mod qmf;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "g722";

/// Register the G.722 decoder + encoder under the single codec id `"g722"`.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("g722_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(16_000);
    reg.register_both(
        CodecId::new(CODEC_ID_STR),
        caps,
        decoder::make_decoder,
        encoder::make_encoder,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecParameters, SampleFormat};

    fn params() -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        p.sample_rate = Some(16_000);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p
    }

    #[test]
    fn registers_both_directions() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(reg.has_decoder(&id));
        assert!(reg.has_encoder(&id));
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut p = params();
        p.sample_rate = Some(8_000);
        assert!(decoder::make_decoder(&p).is_err());
        assert!(encoder::make_encoder(&p).is_err());
    }

    #[test]
    fn rejects_stereo() {
        let mut p = params();
        p.channels = Some(2);
        assert!(decoder::make_decoder(&p).is_err());
        assert!(encoder::make_encoder(&p).is_err());
    }

    #[test]
    fn rejects_reduced_bit_rate() {
        let mut p = params();
        p.bit_rate = Some(56_000);
        assert!(matches!(
            decoder::make_decoder(&p),
            Err(oxideav_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            encoder::make_encoder(&p),
            Err(oxideav_core::Error::Unsupported(_))
        ));
        let mut p = params();
        p.bit_rate = Some(48_000);
        assert!(matches!(
            decoder::make_decoder(&p),
            Err(oxideav_core::Error::Unsupported(_))
        ));
    }

    #[test]
    fn accepts_64k_bit_rate_hint() {
        let mut p = params();
        p.bit_rate = Some(64_000);
        assert!(decoder::make_decoder(&p).is_ok());
        assert!(encoder::make_encoder(&p).is_ok());
    }
}
