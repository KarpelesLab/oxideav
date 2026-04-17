//! G.722 encoder (64 kbit/s mode).
//!
//! Takes 16 kHz mono S16 input frames. Every two input samples feed through
//! the QMF analysis filter to produce one low/high band pair; each pair is
//! compressed into one 8-bit packed code (low-band 6 bits in the MSBs,
//! high-band 2 bits in the LSBs). The encoder emits one packet per
//! `send_frame` call carrying `samples / 2` bytes.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::band_high::HighBand;
use crate::band_low::LowBand;
use crate::decoder::SAMPLE_RATE_HZ;
use crate::qmf::QmfAnalysis;

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE_HZ);
    if sample_rate != SAMPLE_RATE_HZ {
        return Err(Error::unsupported(format!(
            "G.722 encoder: only {SAMPLE_RATE_HZ} Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "G.722 encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "G.722 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    if let Some(br) = params.bit_rate {
        if br != 64_000 {
            return Err(Error::unsupported(format!(
                "G.722 encoder: only 64 kbit/s is supported; 56 and 48 kbit/s \
                 modes are not yet implemented (got {br})"
            )));
        }
    }

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE_HZ);
    output.bit_rate = Some(64_000);

    Ok(Box::new(G722Encoder::new(output)))
}

pub struct G722Encoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    low: LowBand,
    high: HighBand,
    qmf: QmfAnalysis,
    /// Carry-over from odd-length input frames (needs pairs).
    odd_sample: Option<i16>,
    pending: VecDeque<Packet>,
    next_pts: i64,
}

impl G722Encoder {
    fn new(output_params: CodecParameters) -> Self {
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE_HZ as i64),
            low: LowBand::new(),
            high: HighBand::new(),
            qmf: QmfAnalysis::new(),
            odd_sample: None,
            pending: VecDeque::new(),
            next_pts: 0,
        }
    }

    /// Encode a slice of S16 samples (which must have even length) into
    /// one packed packet. Returns the resulting bytes.
    fn encode_pairs(&mut self, samples: &[i16]) -> Vec<u8> {
        debug_assert!(samples.len() % 2 == 0);
        let mut out = Vec::with_capacity(samples.len() / 2);
        for pair in samples.chunks_exact(2) {
            let (xlow, xhigh) = self.qmf.process(pair[0], pair[1]);
            let il = self.low.encode(xlow as i32);
            let ih = self.high.encode(xhigh as i32);
            // Bit layout: low-band (6 bits) in high nibble-block, high-band
            // (2 bits) in LSBs.
            let byte = ((il & 0x3F) << 2) | (ih & 0x03);
            out.push(byte);
        }
        out
    }
}

impl Encoder for G722Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let af = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::invalid("G.722 encoder: audio frames only")),
        };
        if af.channels != 1 {
            return Err(Error::invalid(format!(
                "G.722 encoder: input must be mono (got {} channels)",
                af.channels
            )));
        }
        if af.sample_rate != SAMPLE_RATE_HZ {
            return Err(Error::invalid(format!(
                "G.722 encoder: input must be 16000 Hz (got {} Hz)",
                af.sample_rate
            )));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "G.722 encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("G.722 encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("G.722 encoder: odd byte count"));
        }
        // Read samples from the interleaved S16 plane.
        let mut samples: Vec<i16> = Vec::with_capacity(bytes.len() / 2);
        if let Some(carry) = self.odd_sample.take() {
            samples.push(carry);
        }
        for chunk in bytes.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        // If we now have an odd count, keep the last one for the next call.
        if samples.len() % 2 == 1 {
            self.odd_sample = Some(samples.pop().unwrap());
        }
        if samples.is_empty() {
            return Ok(());
        }

        let n_in = samples.len();
        let encoded = self.encode_pairs(&samples);

        let pts = af.pts.or(Some(self.next_pts));
        self.next_pts = pts.unwrap_or(self.next_pts) + n_in as i64;

        let mut pkt = Packet::new(0, self.time_base, encoded);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.duration = Some(n_in as i64);
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(carry) = self.odd_sample.take() {
            // Pad with a zero so we can encode the final pair.
            let samples = [carry, 0];
            let encoded = self.encode_pairs(&samples);
            let pts = Some(self.next_pts);
            self.next_pts += 2;
            let mut pkt = Packet::new(0, self.time_base, encoded);
            pkt.pts = pts;
            pkt.dts = pts;
            pkt.duration = Some(2);
            pkt.flags.keyframe = true;
            self.pending.push_back(pkt);
        }
        Ok(())
    }
}
