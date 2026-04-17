//! ITU-T G.723.1 dual-rate (6.3 / 5.3 kbit/s) speech codec.
//!
//! # Decoder
//!
//! What's landed: packet-layout bit reader (LSB-first, per Annex B),
//! frame-type discriminator (high-rate / low-rate / SID / erasure),
//! layout / codebook dimension tables, and a 10th-order LPC synthesis
//! filter. The decoder accepts real G.723.1 packets, validates the rate
//! discriminator against the payload length, and emits 240-sample 30 ms
//! silence frames at 8 kHz while the MP-MLQ / ACELP synthesis paths are
//! implemented.
//!
//! What's still stubbed: LSP-VQ codebook lookup + interpolation, adaptive /
//! fixed-codebook excitation reconstruction, MP-MLQ pulse decoding, gain
//! dequantisation, formant / pitch post-filter, and comfort-noise
//! generation for SID / erased frames.
//!
//! # Encoder
//!
//! The crate ships an **encoder for the 5.3 kbit/s ACELP rate only**. The
//! 6.3 kbit/s MP-MLQ rate returns [`Error::Unsupported`] when the caller
//! requests it via `CodecParameters::bit_rate = Some(6300)` — see
//! [`encoder::make_encoder`] and the [`encoder`] module docstring for the
//! full rationale. The ACELP encoder implements the full analysis pipeline
//! (LPC → LSP → open-loop pitch → closed-loop pitch refinement → 4-pulse
//! ACELP fixed-codebook search → joint gain quantisation → 158-bit
//! packing) but uses *simplified, locally-consistent* VQ codebooks rather
//! than the ITU-T Table 5 / Table 7 codebooks. Bitstreams produced here
//! therefore round-trip cleanly through the crate's own reference decoder
//! (see [`encoder::decode_acelp_local`]) but are **not** bit-compatible
//! with external G.723.1 implementations. Once the full ITU-T codebooks
//! are ported into the decoder, the encoder's VQ tables can be swapped in
//! a single patch without touching the analysis pipeline.
//!
//! Reference: ITU-T G.723.1 Recommendation (May 2006) and Annex B.

// Scaffold-only — symbols will be used once the full decoder body lands.
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items
)]

pub mod bitreader;
pub mod encoder;
pub mod header;
pub mod synthesis;
pub mod tables;

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, Error, Frame, Packet, Rational,
    Result, SampleFormat, TimeBase,
};

use crate::header::{parse_frame_type, FrameType};
use crate::tables::{FRAME_SIZE_SAMPLES, SAMPLE_RATE_HZ};

pub const CODEC_ID_STR: &str = "g723_1";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("g723_1_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(SAMPLE_RATE_HZ);
    reg.register_both(
        CodecId::new(CODEC_ID_STR),
        caps,
        make_decoder,
        encoder::make_encoder,
    );
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(G7231Decoder::new()))
}

/// Silence-emitting scaffold decoder. Accepts correctly-framed G.723.1
/// packets and produces 30 ms (240-sample) S16 mono frames of zeros. Once
/// the MP-MLQ / ACELP synthesis paths are implemented this decoder will
/// reconstruct the actual speech signal.
struct G7231Decoder {
    codec_id: CodecId,
    /// Buffered frames awaiting `receive_frame`.
    pending: std::collections::VecDeque<Frame>,
    /// Set when `flush` has been called and all buffered frames drained.
    drained: bool,
    /// Running sample count — used as PTS when a packet provides none.
    next_pts: i64,
    /// Time base for emitted frames (1/8000, matching the sample rate).
    time_base: TimeBase,
}

impl G7231Decoder {
    fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            pending: std::collections::VecDeque::new(),
            drained: false,
            next_pts: 0,
            time_base: TimeBase(Rational::new(1, SAMPLE_RATE_HZ as i64)),
        }
    }

    /// Build an S16 mono silence frame of `samples` samples, tagged with
    /// `pts` in the decoder's time base.
    fn silence_frame(&self, samples: u32, pts: Option<i64>) -> Frame {
        let bytes_len = samples as usize * SampleFormat::S16.bytes_per_sample();
        Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE_HZ,
            samples,
            pts,
            time_base: self.time_base,
            data: vec![vec![0u8; bytes_len]],
        })
    }
}

impl Decoder for G7231Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::invalid("G.723.1: empty packet"));
        }
        let frame_type = parse_frame_type(&packet.data)?;
        let expected = frame_type.frame_size();
        // Allow packets that are exactly the advertised size. Erasure frames
        // (Untransmitted) may carry just the discriminator byte or be fully
        // empty — accept either as long as there is at least 1 byte.
        match frame_type {
            // 0- or 1-byte packet is fine for Untransmitted; anything
            // longer is tolerated.
            FrameType::Untransmitted => {}
            _ if packet.data.len() < expected => {
                return Err(Error::invalid(format!(
                    "G.723.1: {} needs {} bytes, got {}",
                    frame_type.bit_rate_label(),
                    expected,
                    packet.data.len(),
                )));
            }
            _ => {}
        }

        let pts = packet.pts.or(Some(self.next_pts));
        self.next_pts = pts.unwrap_or(self.next_pts) + FRAME_SIZE_SAMPLES as i64;
        let frame = self.silence_frame(FRAME_SIZE_SAMPLES as u32, pts);
        self.pending.push_back(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(f);
        }
        if self.drained {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.drained = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::packet::PacketFlags;

    fn packet(data: Vec<u8>) -> Packet {
        Packet {
            stream_index: 0,
            time_base: TimeBase(Rational::new(1, SAMPLE_RATE_HZ as i64)),
            pts: None,
            dts: None,
            duration: None,
            flags: PacketFlags::default(),
            data,
        }
    }

    #[test]
    fn registers_in_registry() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(reg.has_encoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn decodes_high_rate_silence() {
        let mut dec = G7231Decoder::new();
        // Discriminator bits = 00 → high rate, 24 bytes.
        let pkt = packet(vec![0u8; 24]);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        assert_eq!(af.sample_rate, SAMPLE_RATE_HZ);
        assert_eq!(af.channels, 1);
        assert_eq!(af.samples, FRAME_SIZE_SAMPLES as u32);
        assert_eq!(af.format, SampleFormat::S16);
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), FRAME_SIZE_SAMPLES * 2);
        assert!(af.data[0].iter().all(|&b| b == 0));
    }

    #[test]
    fn decodes_low_rate_silence() {
        let mut dec = G7231Decoder::new();
        // Discriminator bits = 01 → low rate, 20 bytes.
        let mut data = vec![0u8; 20];
        data[0] = 0b01;
        let pkt = packet(data);
        dec.send_packet(&pkt).unwrap();
        let f = dec.receive_frame().unwrap();
        assert_eq!(f.time_base().as_rational().den, SAMPLE_RATE_HZ as i64);
    }

    #[test]
    fn rejects_short_high_rate_frame() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0u8; 10]); // high-rate needs 24
        assert!(dec.send_packet(&pkt).is_err());
    }

    #[test]
    fn accepts_untransmitted_single_byte() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0b11]);
        dec.send_packet(&pkt).unwrap();
        assert!(matches!(dec.receive_frame().unwrap(), Frame::Audio(_)));
    }

    #[test]
    fn flush_then_eof() {
        let mut dec = G7231Decoder::new();
        let pkt = packet(vec![0u8; 24]);
        dec.send_packet(&pkt).unwrap();
        dec.flush().unwrap();
        // One buffered frame, then EOF.
        assert!(dec.receive_frame().is_ok());
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn pts_increments_by_frame_size() {
        let mut dec = G7231Decoder::new();
        dec.send_packet(&packet(vec![0u8; 24])).unwrap();
        dec.send_packet(&packet(vec![0u8; 24])).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        assert_eq!(f0.pts(), Some(0));
        assert_eq!(f1.pts(), Some(FRAME_SIZE_SAMPLES as i64));
    }
}
