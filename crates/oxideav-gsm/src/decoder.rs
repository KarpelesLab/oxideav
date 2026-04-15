//! `oxideav_codec::Decoder` implementation for GSM 06.10 Full Rate.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::frame::{parse_frame, parse_ms_pair, FRAME_SIZE, MS_FRAME_SIZE};
use crate::synthesis::SynthesisState;

/// Codec IDs handled by this decoder.
pub const CODEC_ID_STANDARD: &str = "gsm";
pub const CODEC_ID_MS: &str = "gsm_ms";

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(8_000);
    if sample_rate != 8_000 {
        return Err(Error::unsupported(format!(
            "GSM 06.10 decoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "GSM 06.10 decoder: only mono is supported (got {channels} channels)"
        )));
    }
    let variant = match params.codec_id.as_str() {
        CODEC_ID_STANDARD => Variant::Standard,
        CODEC_ID_MS => Variant::Microsoft,
        other => {
            return Err(Error::unsupported(format!(
                "GSM decoder: unknown codec id {other:?}"
            )))
        }
    };
    Ok(Box::new(GsmDecoder {
        codec_id: params.codec_id.clone(),
        variant,
        state: SynthesisState::new(),
        pending: None,
        ms_second: None,
        eof: false,
        time_base: TimeBase::new(1, sample_rate as i64),
    }))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Variant {
    /// 33-byte payloads, each carries one 20 ms frame (160 samples).
    Standard,
    /// 65-byte payloads, each carries two 20 ms frames back-to-back.
    Microsoft,
}

struct GsmDecoder {
    codec_id: CodecId,
    variant: Variant,
    state: SynthesisState,
    /// Buffered input packet awaiting `receive_frame`.
    pending: Option<Packet>,
    /// Carry for MS framing: the second frame of the most-recently-received
    /// 65-byte packet (produced on the second `receive_frame` call).
    ms_second: Option<(crate::frame::GsmFrame, Option<i64>)>,
    eof: bool,
    time_base: TimeBase,
}

impl Decoder for GsmDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() || self.ms_second.is_some() {
            return Err(Error::other(
                "GSM decoder: receive_frame must drain previous packet first",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // Drain a buffered MS-second frame first.
        if let Some((gf, pts)) = self.ms_second.take() {
            let pcm = self.state.decode_frame(&gf);
            return Ok(pcm_to_audio_frame(&pcm, pts, self.time_base));
        }
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        match self.variant {
            Variant::Standard => {
                if pkt.data.len() != FRAME_SIZE {
                    return Err(Error::invalid(format!(
                        "GSM: expected {FRAME_SIZE}-byte packet, got {}",
                        pkt.data.len()
                    )));
                }
                let gf = parse_frame(&pkt.data)?;
                let pcm = self.state.decode_frame(&gf);
                Ok(pcm_to_audio_frame(&pcm, pkt.pts, self.time_base))
            }
            Variant::Microsoft => {
                if pkt.data.len() != MS_FRAME_SIZE {
                    return Err(Error::invalid(format!(
                        "GSM-MS: expected {MS_FRAME_SIZE}-byte packet, got {}",
                        pkt.data.len()
                    )));
                }
                let [g0, g1] = parse_ms_pair(&pkt.data)?;
                let pcm = self.state.decode_frame(&g0);
                // The second half gets the pts + 160 samples if both pts and
                // time_base were in 1/sr units. Otherwise we carry unknown.
                let pts2 = pkt.pts.map(|p| p + 160);
                self.ms_second = Some((g1, pts2));
                Ok(pcm_to_audio_frame(&pcm, pkt.pts, self.time_base))
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn pcm_to_audio_frame(samples: &[i16; 160], pts: Option<i64>, time_base: TimeBase) -> Frame {
    let mut bytes = Vec::with_capacity(160 * 2);
    for &s in samples.iter() {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: 8_000,
        samples: 160,
        pts,
        time_base,
        data: vec![bytes],
    })
}
