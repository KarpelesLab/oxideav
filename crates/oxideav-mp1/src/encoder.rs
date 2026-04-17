//! Minimum-viable CBR MPEG-1 Audio Layer I encoder.
//!
//! # Scope
//! - MPEG-1 Layer I, 32 / 44.1 / 48 kHz, mono or plain stereo (no joint
//!   stereo, no CRC).
//! - One CBR bitrate per encoder instance, from the standard Layer I
//!   ladder (32..=448 kbps).
//! - Greedy, non-psychoacoustic bit allocation: subbands are iteratively
//!   awarded quantiser upgrades in decreasing order of "signal energy
//!   per extra-bit cost" until no more bits are available. Cost accounting
//!   is exact so the frame always fits.
//! - Scalefactors are extracted per-subband-per-channel from the peak of
//!   the 12 subband samples (one SF per subband per frame; Layer I has no
//!   SCFSI, unlike Layer II).
//!
//! Pipeline (mirror of the decoder):
//!   PCM → polyphase analysis → per-subband scalefactor extraction →
//!   bit allocation → sample quantisation → bit packing.
//!
//! # What is NOT implemented
//! - No psychoacoustic model. The bit allocator uses a crude "biggest
//!   SNR gain wins" heuristic driven by subband energy rather than a
//!   masked-to-noise ratio from a perceptual model. (Follow-up.)
//! - No joint stereo / intensity coding.
//! - No CRC-16.
//! - No free-format output.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::analysis::{analyze_frame, AnalysisState, SAMPLES_PER_FRAME};
use crate::bitalloc::{bits_per_sample, scale_table, SBLIMIT};
use crate::bitwriter::BitWriter;
use crate::CODEC_ID_STR;

/// Number of subband-sample blocks per Layer I frame (12 × 32 = 384 PCM).
const BLOCKS_PER_FRAME: usize = 12;

/// Build a Layer I CBR encoder for the requested parameters.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("MP1 encoder: missing channels"))?;
    if !(1..=2).contains(&channels) {
        return Err(Error::invalid("MP1 encoder: channels must be 1 or 2"));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("MP1 encoder: missing sample_rate"))?;
    match sample_rate {
        32_000 | 44_100 | 48_000 => {}
        _ => {
            return Err(Error::unsupported(format!(
                "MP1 encoder: unsupported sample rate {sample_rate} (need 32000/44100/48000)"
            )));
        }
    }

    let bitrate_kbps = params.bit_rate.map(|b| (b / 1000) as u32).unwrap_or(192);
    let br_index = bitrate_to_index(bitrate_kbps).ok_or_else(|| {
        Error::unsupported(format!(
            "MP1 encoder: unsupported bitrate {bitrate_kbps} kbps"
        ))
    })?;

    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "MP1 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }

    let sr_index = match sample_rate {
        44_100 => 0u8,
        48_000 => 1,
        32_000 => 2,
        _ => unreachable!(),
    };

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.codec_id = CodecId::new(CODEC_ID_STR);
    output.sample_format = Some(sample_format);
    output.channels = Some(channels);
    output.sample_rate = Some(sample_rate);
    output.bit_rate = Some((bitrate_kbps as u64) * 1000);

    Ok(Box::new(Mp1Encoder {
        output_params: output,
        channels,
        sample_rate,
        bitrate_kbps,
        sr_index,
        br_index,
        time_base: TimeBase::new(1, sample_rate as i64),
        analysis_state: [AnalysisState::new(), AnalysisState::new()],
        pcm_queue: vec![Vec::new(); channels as usize],
        pending_packets: VecDeque::new(),
        frame_index: 0,
        eof: false,
        cumulative_padded_bits: 0,
    }))
}

struct Mp1Encoder {
    output_params: CodecParameters,
    channels: u16,
    sample_rate: u32,
    bitrate_kbps: u32,
    sr_index: u8,
    br_index: u32,
    time_base: TimeBase,
    analysis_state: [AnalysisState; 2],
    pcm_queue: Vec<Vec<f32>>,
    pending_packets: VecDeque<Packet>,
    frame_index: u64,
    eof: bool,
    /// Fractional-byte CBR padding accumulator; see [`next_padding`].
    cumulative_padded_bits: u64,
}

impl Mp1Encoder {
    /// Frame size in bytes.
    ///
    /// Layer I uses 4-byte slots: `slots = 12 * bitrate / sample_rate + pad`,
    /// `bytes = slots * 4` (ISO 11172-3 §2.4.3.1).
    fn frame_bytes(&self, padding: bool) -> usize {
        let slots = 12 * self.bitrate_kbps * 1000 / self.sample_rate;
        let padded_slots = slots + if padding { 1 } else { 0 };
        (padded_slots * 4) as usize
    }

    /// Decide whether this frame should set the padding bit. For fractional
    /// slots per frame we accumulate remainders modulo `sample_rate` and pay
    /// off with one padding slot whenever the accumulator overflows. Layer
    /// I's slot is 32 bits wide, so the threshold is `sample_rate * 32`.
    fn next_padding(&mut self) -> bool {
        let num = 12_000u64 * self.bitrate_kbps as u64;
        let sr = self.sample_rate as u64;
        let rem = num - (num / sr) * sr;
        self.cumulative_padded_bits += rem;
        // One slot = 4 bytes = 32 bits. (`cumulative_padded_bits` tracks
        // fractional slot units scaled by sample_rate.)
        let pad = self.cumulative_padded_bits >= sr;
        if pad {
            self.cumulative_padded_bits -= sr;
        }
        pad
    }

    fn ingest(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != self.channels || frame.sample_rate != self.sample_rate {
            return Err(Error::invalid(
                "MP1 encoder: frame channel/sample-rate mismatch",
            ));
        }
        if frame.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "MP1 encoder: input frames must be S16 interleaved",
            ));
        }
        let data = frame
            .data
            .first()
            .ok_or_else(|| Error::invalid("MP1 encoder: empty frame"))?;
        let n_ch = self.channels as usize;
        let n_samples = data.len() / (2 * n_ch);
        for i in 0..n_samples {
            for ch in 0..n_ch {
                let off = (i * n_ch + ch) * 2;
                let s = i16::from_le_bytes([data[off], data[off + 1]]) as f32 / 32768.0;
                self.pcm_queue[ch].push(s);
            }
        }
        self.flush_ready_frames(false)
    }

    fn flush_ready_frames(&mut self, drain: bool) -> Result<()> {
        let n_ch = self.channels as usize;
        loop {
            let avail = self.pcm_queue[0].len();
            if avail < SAMPLES_PER_FRAME {
                if drain && avail > 0 {
                    for ch in 0..n_ch {
                        self.pcm_queue[ch].resize(SAMPLES_PER_FRAME, 0.0);
                    }
                } else {
                    return Ok(());
                }
            }
            let pkt = self.encode_one_frame()?;
            self.pending_packets.push_back(pkt);
            if drain && self.pcm_queue[0].iter().all(|&v| v == 0.0) {
                return Ok(());
            }
        }
    }

    fn encode_one_frame(&mut self) -> Result<Packet> {
        let n_ch = self.channels as usize;

        // Drain 384 samples/channel from the queue.
        let mut pcm_in: Vec<[f32; SAMPLES_PER_FRAME]> = vec![[0.0f32; SAMPLES_PER_FRAME]; n_ch];
        for ch in 0..n_ch {
            for i in 0..SAMPLES_PER_FRAME {
                pcm_in[ch][i] = self.pcm_queue[ch][i];
            }
            self.pcm_queue[ch].drain(..SAMPLES_PER_FRAME);
        }

        // --- 1. Analysis: 32 × 12 subband buffer per channel ---
        let mut sub: Vec<[[f32; 12]; 32]> = (0..n_ch).map(|_| [[0.0f32; 12]; 32]).collect();
        for ch in 0..n_ch {
            analyze_frame(&mut self.analysis_state[ch], &pcm_in[ch], &mut sub[ch]);
        }

        // --- 2. Scalefactor pick: one 6-bit SF index per subband per
        //     channel. Peak over all 12 samples in the subband. ---
        let mut scf_idx = vec![[0u8; 32]; n_ch];
        for ch in 0..n_ch {
            for sb in 0..SBLIMIT {
                let mut peak = 0.0f32;
                for i in 0..12 {
                    let v = sub[ch][sb][i].abs();
                    if v > peak {
                        peak = v;
                    }
                }
                scf_idx[ch][sb] = pick_scalefactor(peak);
            }
        }

        // --- 3. Subband energies for the allocator ---
        let mut energy = vec![[0.0f32; 32]; n_ch];
        for ch in 0..n_ch {
            for sb in 0..SBLIMIT {
                let mut e = 0.0f32;
                for i in 0..12 {
                    let v = sub[ch][sb][i];
                    e += v * v;
                }
                energy[ch][sb] = e / 12.0;
            }
        }

        // --- 4. Bit allocation ---
        // Frame size / budget accounting.
        let padding = self.next_padding();
        let frame_bytes = self.frame_bytes(padding);
        let frame_bits = frame_bytes * 8;

        // Fixed overhead: 32-bit header + bit-allocation indices
        // (4 bits × 32 subbands × channels). Scalefactors are 6 bits per
        // subband *whose allocation is non-zero* per channel, so they
        // enter the budget lazily as subbands get upgraded away from 0.
        let header_bits = 32u32;
        let bitalloc_bits = 4u32 * SBLIMIT as u32 * n_ch as u32;

        let sample_and_scf_budget: i64 =
            frame_bits as i64 - header_bits as i64 - bitalloc_bits as i64;
        if sample_and_scf_budget < 0 {
            return Err(Error::other("MP1 encoder: frame too small for header"));
        }

        let mut alloc = vec![[0u8; 32]; n_ch];
        let mut remaining: i64 = sample_and_scf_budget;

        // Noise floor: subbands with energy far below the frame-max
        // carry basically no signal, and spending bits on them just
        // adds quantisation noise that's audible above the tone. A
        // proper psychoacoustic model would handle this via masking;
        // without one, we suppress subbands more than 80 dB below the
        // loudest subband. (Follow-up: replace with ISO psymodel 1/2.)
        let peak_energy = energy
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(0.0f32, f32::max);
        let noise_floor = peak_energy * 1e-8; // ~-80 dB power

        // Greedy allocation: for each iteration, find the (ch, sb) pair
        // whose upgrade cost vs. energy gives the best energy-per-bit
        // return, and if we can afford it, apply it.
        loop {
            let mut best: Option<(usize, usize, u8, i64)> = None;
            let mut best_score = f32::NEG_INFINITY;

            for ch in 0..n_ch {
                for sb in 0..SBLIMIT {
                    let cur = alloc[ch][sb];
                    if cur >= 14 {
                        continue;
                    }
                    if energy[ch][sb] <= noise_floor {
                        continue;
                    }
                    let next = cur + 1;
                    let cost = upgrade_cost_bits(cur, next);
                    let score = energy[ch][sb] / (cost as f32).max(1.0);
                    if score > best_score && cost as i64 <= remaining {
                        best_score = score;
                        best = Some((ch, sb, next, cost as i64));
                    }
                }
            }

            match best {
                Some((ch, sb, next, cost)) => {
                    alloc[ch][sb] = next;
                    remaining -= cost;
                }
                None => break,
            }
        }

        // --- 5. Write frame ---
        let mut w = BitWriter::with_capacity(frame_bytes);

        // Header (32 bits).
        // syncword 0xFFF, ID=1 (MPEG-1), layer=11 (Layer I), protection=1
        // (no CRC), bitrate_index, sampling_frequency_index, padding bit,
        // private bit=0, mode (00=stereo, 11=mono), mode_extension=0,
        // copyright=0, original=0, emphasis=0.
        w.write_u32(0xFFF, 12);
        w.write_u32(1, 1); // ID = MPEG-1
        w.write_u32(0b11, 2); // Layer I
        w.write_u32(1, 1); // protection_bit = 1 (no CRC)
        w.write_u32(self.br_index, 4);
        w.write_u32(self.sr_index as u32, 2);
        w.write_u32(if padding { 1 } else { 0 }, 1);
        w.write_u32(0, 1); // private
        let mode_bits = if n_ch == 1 { 0b11u32 } else { 0b00u32 };
        w.write_u32(mode_bits, 2);
        w.write_u32(0, 2); // mode_extension
        w.write_u32(0, 1); // copyright
        w.write_u32(0, 1); // original
        w.write_u32(0, 2); // emphasis

        // --- 5a. Bit allocation (4 bits × subband × channel). Since we
        //     emit plain (non-joint) modes, every subband is below the
        //     bound — channel-paired order, same as the decoder reads.
        for sb in 0..SBLIMIT {
            for ch in 0..n_ch {
                w.write_u32(alloc[ch][sb] as u32, 4);
            }
        }

        // --- 5b. Scalefactors (6 bits per subband-per-channel with
        //     allocation != 0). Layer I has no SCFSI. Order: by subband,
        //     then by channel.
        for sb in 0..SBLIMIT {
            for ch in 0..n_ch {
                if alloc[ch][sb] != 0 {
                    w.write_u32(scf_idx[ch][sb] as u32, 6);
                }
            }
        }

        // --- 5c. Sample payload. 12 blocks, each block visits every
        //     subband × channel in order. One sample per subband per
        //     channel per block.
        let sc = scale_table();
        for block in 0..BLOCKS_PER_FRAME {
            for sb in 0..SBLIMIT {
                for ch in 0..n_ch {
                    let a = alloc[ch][sb];
                    if a == 0 {
                        continue;
                    }
                    let nb = bits_per_sample(a).unwrap();
                    let sf_mag = sc[scf_idx[ch][sb] as usize];
                    let raw = quantise_sample(sub[ch][sb][block], sf_mag, nb);
                    w.write_u32(raw, nb as u32);
                }
            }
        }

        // --- 5d. Pad to frame length. Fill with zero ancillary data.
        w.align_to_byte();
        let mut bytes = w.into_bytes();
        if bytes.len() > frame_bytes {
            bytes.truncate(frame_bytes);
        }
        if bytes.len() < frame_bytes {
            bytes.resize(frame_bytes, 0);
        }

        let pts = (self.frame_index as i64) * SAMPLES_PER_FRAME as i64;
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(SAMPLES_PER_FRAME as i64);
        pkt.flags.keyframe = true;
        self.frame_index += 1;
        Ok(pkt)
    }
}

impl Encoder for Mp1Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(a) => self.ingest(a),
            _ => Err(Error::invalid("MP1 encoder: audio frames only")),
        }
    }
    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending_packets.pop_front().ok_or(Error::NeedMore)
    }
    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.flush_ready_frames(true)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper functions.
// ---------------------------------------------------------------------------

/// Reverse-map a bitrate in kbps to its 4-bit header-field index (1..=14
/// for MPEG-1 Layer I). Returns `None` for unsupported values.
fn bitrate_to_index(kbps: u32) -> Option<u32> {
    const LUT: [u32; 14] = [
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ];
    LUT.iter()
        .position(|&v| v == kbps)
        .map(|idx| (idx + 1) as u32)
}

/// Given a signal peak magnitude, pick the smallest scalefactor index
/// whose magnitude `>= peak` (we want the tightest step size that still
/// covers the peak without clipping). Layer I SCALE[i] = 2 * 2^(-i/3),
/// so larger `i` means smaller magnitude.
///
/// Falls back to index 62 (smallest SF) for tiny peaks, and to index 0
/// (largest SF) for peaks that exceed the table range.
fn pick_scalefactor(peak: f32) -> u8 {
    if !peak.is_finite() || peak <= 0.0 {
        return 62;
    }
    let sc = scale_table();
    // We want the largest i such that SCALE[i] >= peak. SCALE is
    // monotonically decreasing, so scan from 0 and keep the last i that
    // satisfies the condition.
    let mut best = 0u8;
    for i in 0..63u8 {
        if sc[i as usize] >= peak {
            best = i;
        } else {
            break;
        }
    }
    best
}

/// Extra bits required when upgrading subband allocation from `cur` to
/// `next`. Accounts for 12 sample slots and a 6-bit scalefactor that is
/// only paid once when the allocation transitions 0 → non-zero.
fn upgrade_cost_bits(cur: u8, next: u8) -> u32 {
    debug_assert!(next > cur);
    let cur_nb = if cur == 0 { 0 } else { cur as u32 + 1 };
    let next_nb = next as u32 + 1;
    let sample_delta = 12 * (next_nb - cur_nb);
    if cur == 0 {
        // Pay scalefactor bits once.
        sample_delta + 6
    } else {
        sample_delta
    }
}

/// Quantise a normalised subband sample `s` (in [-1, 1]) for a subband
/// with scalefactor magnitude `sf_mag` and Layer I allocation `nb` bits.
///
/// Decoder computes: `out = (raw - 2^(nb-1) + 1) * 2/(2^nb - 1) * sf_mag`.
/// Inverting: `raw = round(s * (2^nb - 1) / (2 * sf_mag)) + 2^(nb-1) - 1`,
/// clamped to `0..=2^nb - 1`.
fn quantise_sample(s: f32, sf_mag: f32, nb: u8) -> u32 {
    let denom = (1u32 << nb) - 1; // 2^nb - 1
    let offset = (1i32 << (nb - 1)) - 1; // 2^(nb-1) - 1
    let max_code = denom;
    let sf = sf_mag.max(1e-30);
    let level = (s * (denom as f32) / (2.0 * sf)).round() as i32;
    // Decoder accepts level ∈ [-(offset), offset + 1] → raw ∈ [0, 2^nb-1].
    let raw = level + offset;
    raw.clamp(0, max_code as i32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitrate_index_lookup() {
        assert_eq!(bitrate_to_index(32), Some(1));
        assert_eq!(bitrate_to_index(192), Some(6));
        assert_eq!(bitrate_to_index(448), Some(14));
        assert_eq!(bitrate_to_index(999), None);
    }

    #[test]
    fn scalefactor_pick_reasonable() {
        // Peak 1.0 should land at an index whose magnitude >= 1.0.
        let i = pick_scalefactor(1.0);
        let sc = scale_table();
        let mag = sc[i as usize];
        assert!(mag >= 1.0, "mag={mag} i={i}");
        // Tiny peak → large index.
        let i = pick_scalefactor(1e-8);
        assert!(i >= 50, "got {i}");
    }

    #[test]
    fn quantise_roundtrip_basic() {
        // For nb=15 and SF index 3 (mag = 1.0), the decoder's dequant
        // factor is 2/(2^15 - 1). Quantising a sample near 0.5 should
        // give a raw code whose dequantised value is close.
        let sf_mag = 1.0f32;
        let s = 0.5f32;
        let raw = quantise_sample(s, sf_mag, 15);
        let denom = (1u32 << 15) - 1;
        let level = raw as i32 - (1i32 << 14) + 1;
        let recovered = level as f32 * 2.0 / denom as f32 * sf_mag;
        assert!((recovered - s).abs() < 1e-3, "recovered {recovered} vs {s}");
    }

    #[test]
    fn encoder_roundtrip_silence() {
        use crate::decoder::make_decoder;
        use oxideav_core::Frame as CoreFrame;

        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(44_100);
        params.sample_format = Some(SampleFormat::S16);
        params.bit_rate = Some(128_000);
        let mut enc = make_encoder(&params).unwrap();

        // Feed 3 frames of silence.
        let n = SAMPLES_PER_FRAME * 3;
        let mut data = Vec::new();
        for _ in 0..n {
            data.extend_from_slice(&0i16.to_le_bytes());
        }
        let frame = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 44_100,
            samples: n as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 44_100),
            data: vec![data],
        };
        enc.send_frame(&CoreFrame::Audio(frame)).unwrap();
        let mut packets: Vec<Packet> = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        enc.flush().unwrap();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        assert!(!packets.is_empty(), "no packets produced");

        // Decode back.
        let dparams = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&dparams).unwrap();
        let mut decoded = 0u32;
        for p in &packets {
            dec.send_packet(p).unwrap();
            if let Ok(CoreFrame::Audio(a)) = dec.receive_frame() {
                decoded += a.samples;
            }
        }
        assert!(
            decoded >= SAMPLES_PER_FRAME as u32,
            "decoded too few samples: {decoded}"
        );
    }
}
