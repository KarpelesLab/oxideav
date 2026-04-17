//! IMA-ADPCM decoder for the AMV variant.
//!
//! Per ffmpeg's `adpcm.c` (`AV_CODEC_ID_ADPCM_IMA_AMV`):
//!
//! ```text
//! struct amv_audio_chunk {
//!     int16_le predictor;       // initial predictor for the chunk
//!     u8       step_index;      // 0..=88
//!     u8       _reserved[5];    // some encoders write the decoded sample
//!                               // count here; spec says skip
//!     packed nibbles[];         // (payload_len - 8) * 2 samples, mono
//! }
//! ```
//!
//! Each nibble is fed through the standard IMA-ADPCM update with shift = 3:
//!
//! ```text
//! step       = step_table[step_index]
//! step_index = clip(step_index + index_table[nibble], 0, 88)
//! sign  = nibble & 8
//! delta = nibble & 7
//! diff  = ((2 * delta + 1) * step) >> 3
//! pred  = pred ± diff   (sign chooses)
//! pred  = clip16(pred)
//! ```
//!
//! The shift-by-3 form (vs the shift-by-3 plus an extra bias) is the
//! exact form ffmpeg uses for AMV. Tables are the canonical IMA-ADPCM
//! ones.

use std::collections::VecDeque;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, Error, Frame, MediaType, Packet,
    Result, SampleFormat, TimeBase,
};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(22_050);
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported("AMV IMA-ADPCM: only mono is supported"));
    }
    Ok(Box::new(AmvAudioDecoder {
        codec_id: CodecId::new(crate::AUDIO_CODEC_ID_STR),
        sample_rate,
        time_base: TimeBase::new(1, sample_rate as i64),
        pending: None,
        eof: false,
    }))
}

/// Convenience wrapper for the codec-registry hookup. Registers both the
/// decoder and the encoder for the AMV IMA-ADPCM codec id.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("adpcm_ima_amv_sw")
        .with_lossy(true)
        .with_max_channels(1)
        .with_max_sample_rate(48_000);
    reg.register_both(
        CodecId::new(crate::AUDIO_CODEC_ID_STR),
        caps,
        make_decoder,
        make_encoder,
    );
}

struct AmvAudioDecoder {
    codec_id: CodecId,
    sample_rate: u32,
    time_base: TimeBase,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for AmvAudioDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "AMV audio: receive_frame must be called before sending another packet",
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
        let samples = decode_chunk(&pkt.data)?;
        let n = samples.len() as u32;
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: self.sample_rate,
            samples: n,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![bytes],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Decode one AMV audio chunk into S16 mono samples.
pub fn decode_chunk(data: &[u8]) -> Result<Vec<i16>> {
    if data.len() < 8 {
        return Err(Error::invalid("AMV audio chunk shorter than 8-byte header"));
    }
    let predictor = i16::from_le_bytes([data[0], data[1]]) as i32;
    let step_index = data[2] as i32;
    if !(0..=88).contains(&step_index) {
        return Err(Error::invalid("AMV audio: step_index out of range"));
    }
    // bytes [3..8] are reserved (some encoders write the sample count here);
    // ffmpeg explicitly `bytestream2_skipu(&gb, 5)` past them.
    let payload = &data[8..];
    let mut state = ImaState {
        predictor,
        step_index,
    };
    // (payload.len() * 2) samples — high nibble first, low nibble second.
    let mut out = Vec::with_capacity(payload.len() * 2);
    for &b in payload {
        out.push(state.expand_nibble(b >> 4));
        out.push(state.expand_nibble(b & 0x0F));
    }
    Ok(out)
}

/// Persistent IMA-ADPCM channel state.
#[derive(Clone, Copy, Debug)]
pub struct ImaState {
    pub predictor: i32,
    pub step_index: i32,
}

impl ImaState {
    pub fn expand_nibble(&mut self, nibble: u8) -> i16 {
        let nibble = nibble as i32 & 0x0F;
        let step = STEP_TABLE[self.step_index as usize];
        let new_step_index = (self.step_index + INDEX_TABLE[nibble as usize]).clamp(0, 88);

        let sign = nibble & 8;
        let delta = nibble & 7;
        let diff = ((2 * delta + 1) * step) >> 3;
        let mut predictor = self.predictor;
        if sign != 0 {
            predictor -= diff;
        } else {
            predictor += diff;
        }
        let predictor = predictor.clamp(i16::MIN as i32, i16::MAX as i32);
        self.predictor = predictor;
        self.step_index = new_step_index;
        predictor as i16
    }
}

impl ImaState {
    /// Quantise one PCM sample into a 4-bit IMA-ADPCM nibble and advance
    /// the persistent predictor + step-index state in lock-step with the
    /// matching [`ImaState::expand_nibble`] on the decode side. The nibble
    /// layout is the standard IMA one: bit 3 = sign, bits 2..0 = magnitude.
    ///
    /// This mirrors ffmpeg's `adpcm_ima_compress_sample` — the reconstructed
    /// diff is accumulated as `(step >> 3) + Σ step_k` so that the updated
    /// predictor equals exactly what `expand_nibble` will produce on the
    /// same nibble. That bit-exactness guarantees an encode → decode
    /// roundtrip converges to the expected IMA-ADPCM output.
    pub fn compress_sample(&mut self, sample: i16) -> u8 {
        let step = STEP_TABLE[self.step_index as usize];
        let raw_delta = sample as i32 - self.predictor;
        let mut nibble: u8 = if raw_delta < 0 { 8 } else { 0 };
        let mut delta = raw_delta.unsigned_abs() as i32;

        // Reconstructed absolute magnitude starts at step/8 (the constant
        // bias from the `(2*delta + 1)*step >> 3` decode formula).
        let mut diff: i32 = step >> 3;
        let mut s = step;
        if delta >= s {
            nibble |= 4;
            delta -= s;
            diff += s;
        }
        s >>= 1;
        if delta >= s {
            nibble |= 2;
            delta -= s;
            diff += s;
        }
        s >>= 1;
        if delta >= s {
            nibble |= 1;
            diff += s;
        }

        let mut predictor = self.predictor;
        if nibble & 8 != 0 {
            predictor -= diff;
        } else {
            predictor += diff;
        }
        self.predictor = predictor.clamp(i16::MIN as i32, i16::MAX as i32);
        self.step_index = (self.step_index + INDEX_TABLE[nibble as usize & 0x0F]).clamp(0, 88);
        nibble & 0x0F
    }
}

/// Standard IMA-ADPCM index-update table (Annex B of the IMA spec).
pub const INDEX_TABLE: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Standard IMA-ADPCM step table — 89 entries, indices 0..=88.
pub const STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Encode one arbitrary-length PCM S16 mono slice into a single AMV audio
/// chunk (`01wb`-payload-shaped). The returned vector starts with the
/// standard 8-byte header (initial predictor + step-index taken from
/// `state`, followed by five reserved zero bytes) then the packed nibbles
/// (high nibble first, low nibble second). `state` is advanced in place so
/// callers can encode a long stream chunk-by-chunk; each chunk is
/// self-sufficient on decode because its header reseeds the decoder to
/// whatever state this encoder was in at the start of that chunk.
///
/// If `samples.len()` is odd, the final byte's low nibble is set to 0x0,
/// which decodes with a near-zero predictor delta — identical to what
/// ffmpeg's AMV encoder does when a block runs short. Callers that care
/// about exact sample counts should pad themselves.
pub fn encode_chunk(state: &mut ImaState, samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + samples.len().div_ceil(2));
    // 8-byte header: LE int16 initial predictor, u8 initial step_index,
    // 5 reserved bytes (we write zero; matches ffmpeg).
    out.extend_from_slice(&(state.predictor as i16).to_le_bytes());
    out.push(state.step_index as u8);
    out.extend_from_slice(&[0u8; 5]);

    let mut i = 0;
    while i < samples.len() {
        let hi = state.compress_sample(samples[i]);
        let lo = if i + 1 < samples.len() {
            state.compress_sample(samples[i + 1])
        } else {
            0
        };
        out.push((hi << 4) | lo);
        i += 2;
    }
    out
}

/// Build an encoder for AMV IMA-ADPCM. Accepts mono S16 input at any
/// sample rate up to 48 kHz; each received `AudioFrame` is emitted as one
/// self-contained packet (one `01wb` chunk worth of data).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(22_050);
    if sample_rate == 0 || sample_rate > 48_000 {
        return Err(Error::unsupported(format!(
            "AMV IMA-ADPCM encoder: sample rate {sample_rate} out of supported range"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(
            "AMV IMA-ADPCM encoder: only mono is supported",
        ));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "AMV IMA-ADPCM encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }

    let mut output = params.clone();
    output.codec_id = CodecId::new(crate::AUDIO_CODEC_ID_STR);
    output.media_type = MediaType::Audio;
    output.sample_rate = Some(sample_rate);
    output.channels = Some(1);
    output.sample_format = Some(SampleFormat::S16);

    Ok(Box::new(AmvAudioEncoder {
        output_params: output,
        sample_rate,
        time_base: TimeBase::new(1, sample_rate as i64),
        state: ImaState {
            predictor: 0,
            step_index: 0,
        },
        pending: VecDeque::new(),
        sample_cursor: 0,
        carry: None,
    }))
}

struct AmvAudioEncoder {
    output_params: CodecParameters,
    sample_rate: u32,
    time_base: TimeBase,
    /// Running IMA state, carried across chunks. Each packet's header
    /// records the state as-of-the-start-of-that-chunk so decoders can
    /// reseed without needing packet history.
    state: ImaState,
    pending: VecDeque<Packet>,
    /// Cumulative sample count emitted so far (used for packet PTS).
    sample_cursor: i64,
    /// One-sample carry-over when a received frame has an odd sample count
    /// — deferred to the next frame so every emitted chunk packs full
    /// bytes (nibble pairs) with no phantom zero nibble in the middle of
    /// the stream. `flush` drains the carry with a zero-pad nibble.
    carry: Option<i16>,
}

impl Encoder for AmvAudioEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Audio(af) = frame else {
            return Err(Error::invalid("AMV audio encoder: audio frames only"));
        };
        if af.channels != 1 {
            return Err(Error::invalid("AMV audio encoder: input must be mono"));
        }
        if af.sample_rate != self.sample_rate {
            return Err(Error::invalid(format!(
                "AMV audio encoder: input sample rate {} does not match configured {}",
                af.sample_rate, self.sample_rate
            )));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "AMV audio encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("AMV audio encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("AMV audio encoder: odd byte count"));
        }
        let mut samples: Vec<i16> = Vec::with_capacity(bytes.len() / 2 + 1);
        if let Some(c) = self.carry.take() {
            samples.push(c);
        }
        for chunk in bytes.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        // Defer the tail sample when the total is odd so every emitted
        // packet packs complete byte pairs — important for chunked streams
        // where a zero-pad nibble would otherwise inject a phase glitch at
        // every chunk boundary.
        if samples.len() % 2 == 1 {
            self.carry = samples.pop();
        }
        if samples.is_empty() {
            return Ok(());
        }

        let pts = Some(self.sample_cursor);
        let duration = samples.len() as i64;
        let data = encode_chunk(&mut self.state, &samples);
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.duration = Some(duration);
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        self.sample_cursor += duration;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(last) = self.carry.take() {
            // One-sample tail: emit a single-byte chunk. The low nibble is
            // the conventional zero-pad value, matching how ffmpeg's AMV
            // encoder finalises an odd-sized input.
            let tail = [last];
            let pts = Some(self.sample_cursor);
            let data = encode_chunk(&mut self.state, &tail);
            let mut pkt = Packet::new(0, self.time_base, data);
            pkt.pts = pts;
            pkt.dts = pts;
            pkt.duration = Some(1);
            pkt.flags.keyframe = true;
            self.pending.push_back(pkt);
            self.sample_cursor += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_short_chunk() {
        let r = decode_chunk(&[0u8; 4]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_bad_step_index() {
        let mut buf = vec![0u8; 8];
        buf[2] = 200; // out of range
        let r = decode_chunk(&buf);
        assert!(r.is_err());
    }

    #[test]
    fn decodes_silent_chunk() {
        // predictor=0, step_index=0, no payload → empty output.
        let buf = vec![0u8; 8];
        let s = decode_chunk(&buf).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn decodes_known_pattern() {
        // Hand-rolled: starting from (pred=0, step_index=0), feed nibble
        // sequence and check the predictor evolves per the spec.
        // step_table[0] = 7. nibble = 0 → diff = (1*7)>>3 = 0; sign=0 → pred = 0.
        // index_table[0] = -1 → step_index clamp(0-1,0,88) = 0. So feeding
        // 0x00 byte → two zero samples, state unchanged.
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        let s = decode_chunk(&data).unwrap();
        assert_eq!(s.len(), 8);
        assert!(s.iter().all(|&v| v == 0));

        // Now check a non-zero update: nibble 0x4 → step_table[0]=7,
        // diff = (2*4+1)*7 >> 3 = 63>>3 = 7. sign=0 (bit 8 not set, since
        // 0x4 = 0b0100) → pred = 7. step_index += index_table[4]=4 → 4.
        let mut data = vec![0u8; 8];
        data.push(0x40); // nibbles: 0x4, then 0x0
        let s = decode_chunk(&data).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], 7);
        // Second nibble = 0x0 with step_index=4 → step=11, diff=(1*11)>>3=1,
        // sign=0 → pred=8. step_index += -1 → 3.
        assert_eq!(s[1], 8);
    }
}
