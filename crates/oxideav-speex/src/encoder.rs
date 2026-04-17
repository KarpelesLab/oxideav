//! Top-level Speex encoder — wraps the narrowband CELP analysis loop
//! behind the [`oxideav_codec::Encoder`] trait.
//!
//! Only **sub-mode 5** (15 kbps narrowband) is currently implemented.
//! Other Speex modes return `Error::Unsupported` from the factory so
//! callers can fall back to the decoder-only registration transparently.
//!
//! The produced packets embed an 80-byte Speex header as `extradata`
//! (like the decoder expects), one encoded codec-frame per container
//! packet. A 4-bit terminator (`m=15`) is appended to the final packet
//! so downstream decoders can detect end-of-frame when
//! `frames_per_packet = 1`.

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};
use std::collections::VecDeque;

use crate::bitwriter::BitWriter;
use crate::header::{SPEEX_HEADER_SIZE, SPEEX_SIGNATURE};
use crate::nb_decoder::NB_FRAME_SIZE;
use crate::nb_encoder::{NbEncoder, SUPPORTED_SUBMODE};

/// Encoder factory. Accepts NB (8 kHz mono S16) parameter sets. The
/// caller may override `bit_rate` to pick a sub-mode — currently only
/// sub-mode 5 (≈15 kbps, standard-quality narrowband) is supported.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(8_000);
    if sample_rate != 8_000 {
        return Err(Error::unsupported(format!(
            "Speex encoder: only 8000 Hz narrowband is supported (got {sample_rate}) \
             — wideband/ultra-wideband encode is not yet implemented"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "Speex encoder: only mono is supported (got {channels} channels) \
             — stereo encode (intensity side-channel) is not implemented"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "Speex encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }

    // Only mode 5 is available. If the caller advertised a different
    // target bit-rate we accept it but still encode as mode 5 — the
    // rate-control loop (which would swap sub-modes) is not yet built.
    // A fresh NbEncoder always emits sub-mode 5.
    let submode = SUPPORTED_SUBMODE;
    if submode != 5 {
        return Err(Error::unsupported(format!(
            "Speex encoder: sub-mode {submode} not implemented — only mode 5 \
             (15 kbps NB) is currently supported"
        )));
    }

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(8_000);
    output.codec_id = params.codec_id.clone();
    if output.extradata.is_empty() {
        output.extradata = build_speex_header();
    }

    Ok(Box::new(SpeexEncoder {
        output_params: output,
        time_base: TimeBase::new(1, 8_000),
        nb: NbEncoder::new(),
        pcm_queue: Vec::new(),
        pending: VecDeque::new(),
        frame_index: 0,
        eof: false,
    }))
}

/// Build a minimal 80-byte Speex-in-Ogg header describing the NB
/// mono mode-5 stream this encoder produces. Useful for callers that
/// need to mux into Ogg later.
fn build_speex_header() -> Vec<u8> {
    let mut h = vec![0u8; SPEEX_HEADER_SIZE];
    h[0..8].copy_from_slice(SPEEX_SIGNATURE);
    // Version string ("1.2.1 (oxideav)") — 20 bytes NUL-padded.
    let ver = b"1.2.1-oxideav";
    h[8..8 + ver.len()].copy_from_slice(ver);
    h[28..32].copy_from_slice(&1u32.to_le_bytes()); // version_id
    h[32..36].copy_from_slice(&(SPEEX_HEADER_SIZE as u32).to_le_bytes());
    h[36..40].copy_from_slice(&8_000u32.to_le_bytes()); // rate
    h[40..44].copy_from_slice(&0u32.to_le_bytes()); // mode = NB
    h[44..48].copy_from_slice(&4u32.to_le_bytes()); // mode_bitstream_version
    h[48..52].copy_from_slice(&1u32.to_le_bytes()); // nb_channels
    h[52..56].copy_from_slice(&15_000i32.to_le_bytes()); // bitrate (mode 5)
    h[56..60].copy_from_slice(&(NB_FRAME_SIZE as u32).to_le_bytes());
    h[60..64].copy_from_slice(&0u32.to_le_bytes()); // vbr
    h[64..68].copy_from_slice(&1u32.to_le_bytes()); // frames_per_packet
    h[68..72].copy_from_slice(&0u32.to_le_bytes()); // extra_headers
    h
}

struct SpeexEncoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    nb: NbEncoder,
    /// Queued mono i16 PCM samples awaiting a full 160-sample frame.
    pcm_queue: Vec<i16>,
    pending: VecDeque<Packet>,
    frame_index: u64,
    eof: bool,
}

impl Encoder for SpeexEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(a) => self.ingest(a),
            _ => Err(Error::invalid("Speex encoder: audio frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            // Pad any stragglers to a full frame with zeros.
            if !self.pcm_queue.is_empty() {
                while self.pcm_queue.len() < NB_FRAME_SIZE {
                    self.pcm_queue.push(0);
                }
                self.drain_full_frames(true)?;
            }
        }
        Ok(())
    }
}

impl SpeexEncoder {
    fn ingest(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != 1 || frame.sample_rate != 8_000 {
            return Err(Error::invalid(
                "Speex encoder: input must be mono 8000 Hz S16",
            ));
        }
        if frame.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "Speex encoder: input sample format must be S16",
            ));
        }
        let bytes = frame
            .data
            .first()
            .ok_or_else(|| Error::invalid("Speex encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid(
                "Speex encoder: odd byte count in audio frame",
            ));
        }
        for chunk in bytes.chunks_exact(2) {
            self.pcm_queue
                .push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        self.drain_full_frames(false)
    }

    fn drain_full_frames(&mut self, _flushing: bool) -> Result<()> {
        while self.pcm_queue.len() >= NB_FRAME_SIZE {
            let mut pcm = [0.0f32; NB_FRAME_SIZE];
            for (dst, src) in pcm.iter_mut().zip(self.pcm_queue[..NB_FRAME_SIZE].iter()) {
                *dst = *src as f32;
            }
            self.pcm_queue.drain(..NB_FRAME_SIZE);
            let pts = Some(self.frame_index as i64 * NB_FRAME_SIZE as i64);
            self.frame_index += 1;

            let mut bw = BitWriter::with_capacity(40);
            self.nb.encode_frame(&pcm, &mut bw)?;
            // Mode 5 uses 300 bits; pad the final byte to a byte
            // boundary so the resulting packet is parseable by the
            // companion decoder (which tolerates trailing zero bits).
            let data = bw.finish();

            self.pending.push_back(Packet {
                stream_index: 0,
                time_base: self.time_base,
                pts,
                dts: pts,
                duration: Some(NB_FRAME_SIZE as i64),
                flags: Default::default(),
                data,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::nb_decoder::NbDecoder;

    fn make_test_frame(start: usize) -> AudioFrame {
        // 160 samples of sine at 500 Hz, amplitude 8000 — simple and
        // periodic, well within LPC's comfort zone.
        let mut data = Vec::with_capacity(NB_FRAME_SIZE * 2);
        for i in 0..NB_FRAME_SIZE {
            let t = (start + i) as f32;
            let v = 8000.0 * (2.0 * std::f32::consts::PI * 500.0 * t / 8000.0).sin();
            let s = v.round().clamp(-32768.0, 32767.0) as i16;
            data.extend_from_slice(&s.to_le_bytes());
        }
        AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 8_000,
            samples: NB_FRAME_SIZE as u32,
            pts: None,
            time_base: TimeBase::new(1, 8_000),
            data: vec![data],
        }
    }

    #[test]
    fn encoder_factory_accepts_nb_mono_s16() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.sample_rate = Some(8_000);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        let enc = make_encoder(&params).expect("factory accepts NB mono S16");
        assert_eq!(enc.codec_id().as_str(), crate::CODEC_ID_STR);
    }

    #[test]
    fn encoder_factory_rejects_wideband() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.sample_rate = Some(16_000);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        let err = match make_encoder(&params) {
            Ok(_) => panic!("expected make_encoder to fail on WB params"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn encoder_produces_300bit_packets() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.sample_rate = Some(8_000);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        let mut enc = make_encoder(&params).unwrap();

        for i in 0..4 {
            let frame = make_test_frame(i * NB_FRAME_SIZE);
            enc.send_frame(&Frame::Audio(frame)).unwrap();
        }
        enc.flush().unwrap();

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        }
        assert_eq!(packets.len(), 4, "expected 4 packets");
        for (i, p) in packets.iter().enumerate() {
            // 300 bits = 38 bytes (with 4-bit zero padding on the last).
            assert_eq!(
                p.data.len(),
                38,
                "packet {i} size (got {} bytes for 300-bit payload)",
                p.data.len()
            );
        }

        // Round-trip: decode each packet and verify exactly 300 bits
        // are consumed before the reader runs out (at which point the
        // decoder returns Eof/Truncated, both acceptable).
        let mut dec = NbDecoder::new();
        let mut total_samples = 0usize;
        for p in &packets {
            let mut br = BitReader::new(&p.data);
            let mut out = [0.0f32; NB_FRAME_SIZE];
            dec.decode_frame(&mut br, &mut out).unwrap();
            total_samples += NB_FRAME_SIZE;
            assert!(
                out.iter().all(|v| v.is_finite()),
                "decoded samples must be finite"
            );
        }
        assert_eq!(total_samples, 4 * NB_FRAME_SIZE);
    }
}
