//! GSM Full Rate (ETSI EN 300 961 / 06.10, RPE-LTP, 13 kbit/s) decoder.
//!
//! Scope: a full pure-Rust decoder — bit-unpacker, LAR/LTP/RPE decode,
//! short-term synthesis filter, and post-processing — wired into the
//! `oxideav_codec::Decoder` trait. Two framings are registered:
//!
//! - `gsm`    : the canonical ETSI 33-byte frame (4-bit 0xD magic + 260 bits).
//! - `gsm_ms` : Microsoft's 65-byte "WAV-49" packing of two standard frames.
//!
//! Each standard frame yields 160 S16 PCM samples at 8 kHz mono.
//!
//! The implementation follows the ETSI 06.10 spec and the public-domain
//! libgsm structure (Jutta Degener). Math is all fixed-point i16/i32
//! with saturating semantics — see `math.rs`.

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod decoder;
pub mod frame;
mod math;
pub mod synthesis;
pub mod tables;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "gsm";
pub const CODEC_ID_MS_STR: &str = "gsm_ms";

/// Register both the standard and Microsoft GSM framings.
pub fn register(reg: &mut CodecRegistry) {
    let caps_std = CodecCapabilities::audio("gsm_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(8_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps_std, decoder::make_decoder);

    let caps_ms = CodecCapabilities::audio("gsm_ms_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(8_000);
    reg.register_decoder_impl(
        CodecId::new(CODEC_ID_MS_STR),
        caps_ms,
        decoder::make_decoder,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecParameters, Packet, TimeBase};

    fn make_params(id: &str) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(id));
        p.sample_rate = Some(8_000);
        p.channels = Some(1);
        p
    }

    #[test]
    fn register_and_build_decoder() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_MS_STR)));
    }

    #[test]
    fn decode_zero_frame_produces_160_samples() {
        // 33-byte frame with magic 0xD and all remaining bits zero is
        // valid: LARc indices = 0 (signed -32/-16/...), subframes with
        // Nc=0 (which we clamp to the previous nrp; default 40), bc=mc=0,
        // xmaxc=0, all RPE pulses = 0. Output should be silence-ish.
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let mut dec = decoder::make_decoder(&make_params(CODEC_ID_STR)).unwrap();

        let mut payload = vec![0u8; frame::FRAME_SIZE];
        payload[0] = 0xD0;
        let pkt = Packet::new(0, TimeBase::new(1, 8000), payload);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        match frame {
            oxideav_core::Frame::Audio(af) => {
                assert_eq!(af.samples, 160);
                assert_eq!(af.channels, 1);
                assert_eq!(af.sample_rate, 8_000);
                assert_eq!(af.data.len(), 1);
                assert_eq!(af.data[0].len(), 160 * 2);
            }
            _ => panic!("expected audio frame"),
        }
    }

    #[test]
    fn ms_variant_emits_two_frames_per_packet() {
        let mut dec = decoder::make_decoder(&make_params(CODEC_ID_MS_STR)).unwrap();
        let mut payload = vec![0u8; frame::MS_FRAME_SIZE];
        // No magic in MS framing; zero payload is a valid (if silence-ish) pair.
        payload[0] = 0x00;
        let pkt = Packet::new(0, TimeBase::new(1, 8000), payload);
        dec.send_packet(&pkt).unwrap();
        let f1 = dec.receive_frame().unwrap();
        let f2 = dec.receive_frame().unwrap();
        match (f1, f2) {
            (oxideav_core::Frame::Audio(a), oxideav_core::Frame::Audio(b)) => {
                assert_eq!(a.samples, 160);
                assert_eq!(b.samples, 160);
            }
            _ => panic!("expected two audio frames"),
        }
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut p = make_params(CODEC_ID_STR);
        p.sample_rate = Some(16_000);
        assert!(decoder::make_decoder(&p).is_err());
    }

    #[test]
    fn rejects_stereo() {
        let mut p = make_params(CODEC_ID_STR);
        p.channels = Some(2);
        assert!(decoder::make_decoder(&p).is_err());
    }
}
