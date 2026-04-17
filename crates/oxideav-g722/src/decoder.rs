//! G.722 decoder (64 kbit/s mode).
//!
//! Accepts packets of arbitrary byte length; each byte is one G.722
//! sample-serial code (high-band in the low 2 bits, low-band in the high
//! 6 bits). Every code produces two 16-bit PCM samples at 16 kHz mono.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::band_high::HighBand;
use crate::band_low::LowBand;
use crate::qmf::QmfSynthesis;
use crate::CODEC_ID_STR;

/// Default sample rate for G.722 output (16 kHz wideband).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE_HZ);
    if sample_rate != SAMPLE_RATE_HZ {
        return Err(Error::unsupported(format!(
            "G.722: only {SAMPLE_RATE_HZ} Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.722: only mono is supported (got {channels} channels)"
        )));
    }
    // Reject reduced bit-rate requests.
    if let Some(br) = params.bit_rate {
        if br != 64_000 {
            return Err(Error::unsupported(format!(
                "G.722: only 64 kbit/s is supported; 56 and 48 kbit/s modes \
                 are not yet implemented (got {br})"
            )));
        }
    }
    Ok(Box::new(G722Decoder::new()))
}

pub struct G722Decoder {
    codec_id: CodecId,
    low: LowBand,
    high: HighBand,
    qmf: QmfSynthesis,
    pending: VecDeque<Frame>,
    drained: bool,
    next_pts: i64,
    time_base: TimeBase,
}

impl G722Decoder {
    pub fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            low: LowBand::new(),
            high: HighBand::new(),
            qmf: QmfSynthesis::new(),
            pending: VecDeque::new(),
            drained: false,
            next_pts: 0,
            time_base: TimeBase::new(1, SAMPLE_RATE_HZ as i64),
        }
    }
}

impl Default for G722Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for G722Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            // Empty packets are tolerated — produce no output.
            return Ok(());
        }
        // Each input byte → 2 output samples at 16 kHz.
        let n_samples = packet.data.len() * 2;
        let mut out_pcm = Vec::<i16>::with_capacity(n_samples);
        for &byte in &packet.data {
            // Low-band: high 6 bits. High-band: low 2 bits.
            let il = (byte >> 2) & 0x3F;
            let ih = byte & 0x03;
            let rl = self.low.decode(il);
            let rh = self.high.decode(ih);
            let (s0, s1) = self.qmf.process(rl as i16, rh as i16);
            out_pcm.push(s0);
            out_pcm.push(s1);
        }
        // Pack into little-endian byte buffer.
        let mut bytes = Vec::with_capacity(n_samples * 2);
        for s in &out_pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let pts = packet.pts.or(Some(self.next_pts));
        self.next_pts = pts.unwrap_or(self.next_pts) + n_samples as i64;

        self.pending.push_back(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE_HZ,
            samples: n_samples as u32,
            pts,
            time_base: self.time_base,
            data: vec![bytes],
        }));
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
