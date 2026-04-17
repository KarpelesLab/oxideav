//! Top-level Speex decoder — wires the NB CELP synthesis loop and the
//! WB sub-band CELP extension into the [`oxideav_codec::Decoder`] trait.
//!
//! Extracts the 80-byte Speex header from `CodecParameters::extradata`
//! (which the Ogg demuxer fills with the first Speex packet) and
//! validates it. NB streams produce `S16` mono audio frames at 8 kHz;
//! WB streams produce 16 kHz. UWB streams still return
//! `Error::Unsupported` (see [`crate::wb_decoder`] for the gap notes).
//!
//! Speex-in-Ogg packs `frames_per_packet` (default 1) codec frames into
//! one Ogg packet. The decoder loops over the bitstream until the
//! 4-bit terminator (`m=15`) is read or the bit buffer is exhausted.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::BitReader;
use crate::header::{SpeexHeader, SpeexMode};
use crate::nb_decoder::{NbDecoder, NB_FRAME_SIZE};
use crate::wb_decoder::{WbDecoder, WB_FULL_FRAME_SIZE};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "Speex decoder: missing extradata (expected Speex header packet)",
        ));
    }
    let header = SpeexHeader::parse(&params.extradata)?;

    if header.nb_channels > 2 {
        return Err(Error::unsupported(format!(
            "Speex decoder: {}-channel stream",
            header.nb_channels
        )));
    }

    match header.mode {
        SpeexMode::Narrowband => Ok(Box::new(NbDecoderImpl::new(
            params.codec_id.clone(),
            header,
        ))),
        SpeexMode::Wideband => Ok(Box::new(WbDecoderImpl::new(
            params.codec_id.clone(),
            header,
        ))),
        SpeexMode::UltraWideband => Err(Error::unsupported(
            "Speex decoder: ultra-wideband (32 kHz) decoder not yet implemented \
             — requires a second SB-CELP layer on top of the wideband decoder \
             and a stacked QMF synthesis (see libspeex/sb_celp.c sb_uwb_mode)",
        )),
    }
}

struct NbDecoderImpl {
    codec_id: CodecId,
    nb: NbDecoder,
    header: SpeexHeader,
    time_base: TimeBase,
    pending: Option<Packet>,
    eof: bool,
}

impl NbDecoderImpl {
    fn new(codec_id: CodecId, header: SpeexHeader) -> Self {
        let rate = if header.rate > 0 { header.rate } else { 8_000 };
        let time_base = TimeBase::new(1, rate as i64);
        Self {
            codec_id,
            nb: NbDecoder::new(),
            header,
            time_base,
            pending: None,
            eof: false,
        }
    }

    fn decode_packet(&mut self, pkt: &Packet) -> Result<Frame> {
        let mut br = BitReader::new(&pkt.data);
        let frames_per_packet = self.header.frames_per_packet.max(1) as usize;
        let channels = self.header.nb_channels.max(1) as usize;
        let total_samples = NB_FRAME_SIZE * frames_per_packet;

        // We only support mono right now; stereo Speex uses an interleaved
        // intensity-stereo side-channel which is part of `libspeex/stereo.c`.
        if channels != 1 {
            return Err(Error::unsupported(
                "Speex decoder: stereo (intensity-stereo side channel) not yet implemented \
                 — see libspeex/stereo.c",
            ));
        }

        let mut pcm = vec![0.0f32; total_samples];
        let mut produced = 0usize;
        for _ in 0..frames_per_packet {
            let mut frame_buf = [0.0f32; NB_FRAME_SIZE];
            match self.nb.decode_frame(&mut br, &mut frame_buf) {
                Ok(()) => {
                    pcm[produced..produced + NB_FRAME_SIZE].copy_from_slice(&frame_buf);
                    produced += NB_FRAME_SIZE;
                }
                Err(Error::Eof) => break, // 4-bit terminator (`m=15`)
                Err(e) => return Err(e),
            }
        }
        if produced == 0 {
            return Err(Error::invalid(
                "Speex decoder: no frames decoded from packet",
            ));
        }
        pcm.truncate(produced);

        // Convert float [-32768, 32767] (the reference scales output to
        // int16 range) to S16 little-endian interleaved bytes.
        let mut bytes = Vec::with_capacity(produced * 2);
        for v in &pcm {
            let i = v.round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&i.to_le_bytes());
        }

        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: channels as u16,
            sample_rate: if self.header.rate > 0 {
                self.header.rate
            } else {
                8_000
            },
            samples: produced as u32,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![bytes],
        }))
    }
}

impl Decoder for NbDecoderImpl {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "Speex decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        self.decode_packet(&pkt)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Wipe the NB CELP synthesis state — LPC filter memory, excitation
        // history, LSP/LSF history, pitch-gain history, innovation-gain
        // predictor, perceptual weighting filter memory, postfilter state.
        // All of these are carried across frames in `NbDecoder` and would
        // cause audible "ramp-up" artefacts after a seek if left intact.
        // Header / time_base are stream-level and untouched.
        self.nb = NbDecoder::new();
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

// =====================================================================
// Wideband (16 kHz) decoder driver — one `WbDecoder` frame per NB frame
// inside the packet. Output is `S16` 16 kHz mono.
// =====================================================================

struct WbDecoderImpl {
    codec_id: CodecId,
    wb: WbDecoder,
    header: SpeexHeader,
    time_base: TimeBase,
    pending: Option<Packet>,
    eof: bool,
}

impl WbDecoderImpl {
    fn new(codec_id: CodecId, header: SpeexHeader) -> Self {
        let rate = if header.rate > 0 { header.rate } else { 16_000 };
        let time_base = TimeBase::new(1, rate as i64);
        Self {
            codec_id,
            wb: WbDecoder::new(),
            header,
            time_base,
            pending: None,
            eof: false,
        }
    }

    fn decode_packet(&mut self, pkt: &Packet) -> Result<Frame> {
        let mut br = BitReader::new(&pkt.data);
        let frames_per_packet = self.header.frames_per_packet.max(1) as usize;
        let channels = self.header.nb_channels.max(1) as usize;

        if channels != 1 {
            return Err(Error::unsupported(
                "Speex decoder: stereo (intensity-stereo side channel) not yet implemented \
                 — see libspeex/stereo.c",
            ));
        }

        let mut pcm = Vec::with_capacity(WB_FULL_FRAME_SIZE * frames_per_packet);
        let mut produced = 0usize;
        for _ in 0..frames_per_packet {
            let mut frame_buf = [0.0f32; WB_FULL_FRAME_SIZE];
            match self.wb.decode_frame(&mut br, &mut frame_buf) {
                Ok(()) => {
                    pcm.extend_from_slice(&frame_buf);
                    produced += WB_FULL_FRAME_SIZE;
                }
                Err(Error::Eof) => break, // 4-bit terminator in low-band
                Err(e) => return Err(e),
            }
        }
        if produced == 0 {
            return Err(Error::invalid(
                "Speex decoder: no frames decoded from WB packet",
            ));
        }

        let mut bytes = Vec::with_capacity(produced * 2);
        for v in &pcm[..produced] {
            let i = v.round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&i.to_le_bytes());
        }

        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: channels as u16,
            sample_rate: if self.header.rate > 0 {
                self.header.rate
            } else {
                16_000
            },
            samples: produced as u32,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![bytes],
        }))
    }
}

impl Decoder for WbDecoderImpl {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "Speex decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        self.decode_packet(&pkt)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Wideband SB-CELP carries a full NB sub-decoder state plus the
        // high-band QMF analysis/synthesis memory and the high-band LPC
        // synthesis state. Rebuild a fresh `WbDecoder` so every carry-over
        // filter memory is zeroed.
        self.wb = WbDecoder::new();
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{SPEEX_HEADER_SIZE, SPEEX_SIGNATURE};

    fn good_extradata(mode: u32, rate: u32) -> Vec<u8> {
        let mut h = vec![0u8; SPEEX_HEADER_SIZE];
        h[0..8].copy_from_slice(SPEEX_SIGNATURE);
        h[28..32].copy_from_slice(&1u32.to_le_bytes());
        h[32..36].copy_from_slice(&80u32.to_le_bytes());
        h[36..40].copy_from_slice(&rate.to_le_bytes());
        h[40..44].copy_from_slice(&mode.to_le_bytes());
        h[48..52].copy_from_slice(&1u32.to_le_bytes()); // 1 channel
        h[52..56].copy_from_slice(&(-1i32).to_le_bytes());
        h[56..60].copy_from_slice(&160u32.to_le_bytes());
        h[64..68].copy_from_slice(&1u32.to_le_bytes());
        h
    }

    fn expect_err(params: &CodecParameters) -> Error {
        match make_decoder(params) {
            Ok(_) => panic!("expected make_decoder to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn empty_extradata_is_invalid() {
        let params = CodecParameters::audio(CodecId::new("speex"));
        assert!(matches!(expect_err(&params), Error::InvalidData(_)));
    }

    #[test]
    fn bad_signature_is_invalid() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = vec![0u8; SPEEX_HEADER_SIZE];
        assert!(matches!(expect_err(&params), Error::InvalidData(_)));
    }

    #[test]
    fn nb_header_yields_decoder() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = good_extradata(0, 8000);
        // Decoder factory should succeed for NB; actual frame decode
        // requires real packet data and is exercised by the integration
        // test in tests/decode_nb.rs.
        let dec = make_decoder(&params).expect("NB make_decoder");
        assert_eq!(dec.codec_id().as_str(), "speex");
    }

    #[test]
    fn wb_header_yields_decoder() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = good_extradata(1, 16000);
        let dec = make_decoder(&params).expect("WB make_decoder");
        assert_eq!(dec.codec_id().as_str(), "speex");
    }

    #[test]
    fn uwb_header_returns_unsupported() {
        let mut params = CodecParameters::audio(CodecId::new("speex"));
        params.extradata = good_extradata(2, 32000);
        assert!(matches!(expect_err(&params), Error::Unsupported(_)));
    }
}
