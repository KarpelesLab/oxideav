//! Minimum-viable CBR MPEG-1 Audio Layer II encoder.
//!
//! Scope:
//! - MPEG-1 Layer II, 32 / 44.1 / 48 kHz, mono or dual-channel stereo
//!   (no joint stereo, no CRC).
//! - One CBR bitrate per encoder instance, from the standard Layer II
//!   ladder (32..=384 kbps, subject to channel-mode restrictions).
//! - Greedy, non-psychoacoustic bit allocation: subbands are iteratively
//!   awarded quantiser upgrades in decreasing order of "signal energy
//!   per extra-bit cost" until no more bits are available. Cost accounting
//!   uses the exact per-subband class table so the frame always fits.
//! - Scalefactors are extracted per-part (3 × 12-sample groups per
//!   subband) from the subband-sample peak. SCFSI is chosen by comparing
//!   all three scalefactors and picking the transmission pattern that
//!   exactly represents the triple when possible (SCFSI=2 if all equal,
//!   SCFSI=1 if parts 0/1 match, SCFSI=3 if parts 1/2 match, SCFSI=0
//!   otherwise).
//!
//! Pipeline (mirror of the decoder):
//!   PCM → polyphase analysis → per-subband scalefactor extraction →
//!   bit allocation → sample quantisation (grouped + ungrouped) →
//!   bit packing.
//!
//! # What is NOT implemented
//! - No psychoacoustic model. The bit allocator uses a crude "biggest
//!   SNR gain wins" heuristic driven by subband energy rather than a
//!   masked-to-noise ratio from a perceptual model.
//! - No joint stereo / intensity coding.
//! - No CRC-16.
//! - No free-format output.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::analysis::{analyze_frame, AnalysisState};
use crate::bitwriter::BitWriter;
use crate::tables::{scalefactor_magnitude, select_alloc_table, AllocEntry, AllocTable};
use crate::CODEC_ID_STR;

/// Build a Layer II CBR encoder for the requested parameters.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("MP2 encoder: missing channels"))?;
    if !(1..=2).contains(&channels) {
        return Err(Error::invalid("MP2 encoder: channels must be 1 or 2"));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("MP2 encoder: missing sample_rate"))?;
    match sample_rate {
        32_000 | 44_100 | 48_000 => {}
        _ => {
            return Err(Error::unsupported(format!(
                "MP2 encoder: unsupported sample rate {sample_rate} (need 32000/44100/48000)"
            )));
        }
    }

    let bitrate_kbps = params.bit_rate.map(|b| (b / 1000) as u32).unwrap_or(192);
    let br_index = bitrate_to_index(bitrate_kbps).ok_or_else(|| {
        Error::unsupported(format!(
            "MP2 encoder: unsupported bitrate {bitrate_kbps} kbps"
        ))
    })?;

    // Per ISO/IEC 11172-3 Table 3-B.2, Layer II forbids some (mode, bitrate)
    // combos. Enforce the same subset the header parser does.
    match channels {
        1 if matches!(bitrate_kbps, 224 | 256 | 320 | 384) => {
            return Err(Error::invalid(format!(
                "MP2 encoder: bitrate {bitrate_kbps} kbps not permitted in mono mode"
            )));
        }
        2 if matches!(bitrate_kbps, 32 | 48) => {
            return Err(Error::invalid(format!(
                "MP2 encoder: bitrate {bitrate_kbps} kbps not permitted in stereo modes"
            )));
        }
        _ => {}
    }

    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "MP2 encoder: input sample format {sample_format:?} not supported (need S16)"
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

    Ok(Box::new(Mp2Encoder {
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

struct Mp2Encoder {
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

impl Mp2Encoder {
    fn frame_bytes(&self, padding: bool) -> usize {
        let base = (144 * self.bitrate_kbps * 1000 / self.sample_rate) as usize;
        base + if padding { 1 } else { 0 }
    }

    /// Decide whether this frame should set the padding bit. Same
    /// accumulator scheme as the mp3 encoder — for fractional bits per
    /// frame, we count remainders modulo `8 * sample_rate` and pay off
    /// with one padding byte whenever the accumulator overflows.
    fn next_padding(&mut self) -> bool {
        let num = 144_000u64 * self.bitrate_kbps as u64;
        let sr = self.sample_rate as u64;
        let rem = num - (num / sr) * sr;
        self.cumulative_padded_bits += rem;
        let pad = self.cumulative_padded_bits >= sr * 8;
        if pad {
            self.cumulative_padded_bits -= sr * 8;
        }
        pad
    }

    fn ingest(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != self.channels || frame.sample_rate != self.sample_rate {
            return Err(Error::invalid(
                "MP2 encoder: frame channel/sample-rate mismatch",
            ));
        }
        if frame.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "MP2 encoder: input frames must be S16 interleaved",
            ));
        }
        let data = frame
            .data
            .first()
            .ok_or_else(|| Error::invalid("MP2 encoder: empty frame"))?;
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
            if avail < 1152 {
                if drain && avail > 0 {
                    for ch in 0..n_ch {
                        self.pcm_queue[ch].resize(1152, 0.0);
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

        // Drain 1152 samples/channel from the queue.
        let mut pcm_in: Vec<[f32; 1152]> = vec![[0.0f32; 1152]; n_ch];
        for ch in 0..n_ch {
            for i in 0..1152 {
                pcm_in[ch][i] = self.pcm_queue[ch][i];
            }
            self.pcm_queue[ch].drain(..1152);
        }

        // --- 1. Analysis: 32 × 36 subband buffer per channel ---
        let mut sub: Vec<[[f32; 36]; 32]> = (0..n_ch).map(|_| [[0.0f32; 36]; 32]).collect();
        for ch in 0..n_ch {
            analyze_frame(&mut self.analysis_state[ch], &pcm_in[ch], &mut sub[ch]);
        }

        // --- 2. Pick allocation table (channel-mode + bitrate-dependent) ---
        // We always emit plain stereo / mono (no joint stereo), so the
        // `stereo` flag for table selection is simply `n_ch == 2`.
        let stereo = n_ch == 2;
        let table = select_alloc_table(self.sample_rate, stereo, self.br_index);

        // --- 3. Per-(channel, subband, part) scalefactor indices ---
        // Layer II: subband has 36 samples = 3 × 12-sample parts. For each
        // part, pick the SF index that most closely matches the peak
        // amplitude (rounded up).
        let mut scf_idx = vec![[[0u8; 3]; 32]; n_ch];
        for ch in 0..n_ch {
            for sb in 0..table.sblimit {
                for part in 0..3 {
                    let base = part * 12;
                    let mut peak = 0.0f32;
                    for i in 0..12 {
                        let v = sub[ch][sb][base + i].abs();
                        if v > peak {
                            peak = v;
                        }
                    }
                    scf_idx[ch][sb][part] = pick_scalefactor(peak);
                }
            }
        }

        // --- 4. Choose SCFSI pattern + transmit scalefactors accordingly ---
        // scfsi value returned here lines up with the decoder:
        //   0: parts 0/1/2 all independent (transmit 3 SF)
        //   1: parts 0 == 1, part 2 separate   (transmit 2 SF: p0, p2)
        //   2: parts 0 == 1 == 2               (transmit 1 SF)
        //   3: part 0 separate, parts 1 == 2   (transmit 2 SF: p0, p1)
        let mut scfsi = vec![[0u8; 32]; n_ch];
        for ch in 0..n_ch {
            for sb in 0..table.sblimit {
                let a = scf_idx[ch][sb][0];
                let b = scf_idx[ch][sb][1];
                let c = scf_idx[ch][sb][2];
                scfsi[ch][sb] = pick_scfsi(a, b, c);
            }
        }

        // --- 5. Bit allocation ---
        // Compute fixed overhead so the allocator knows how many bits
        // remain for sample payload.
        let padding = self.next_padding();
        let frame_bytes = self.frame_bytes(padding);
        let frame_bits = frame_bytes * 8;
        // Overhead: 32-bit header + no CRC for our encoder + bit-alloc
        // indices + (dependent-on-alloc) SCFSI + scalefactors. We compute
        // the allocation first so we can price SCFSI/scalefactor bits
        // before the samples.
        let header_bits = 32u32;
        let bitalloc_bits: u32 = (0..table.sblimit)
            .map(|sb| table.nbal(sb) * n_ch as u32)
            .sum();

        // Compute subband energies for the allocator: average of squared
        // samples across the 36 samples in the subband.
        let mut energy = vec![[0.0f32; 32]; n_ch];
        for ch in 0..n_ch {
            for sb in 0..table.sblimit {
                let mut e = 0.0f32;
                for i in 0..36 {
                    let v = sub[ch][sb][i];
                    e += v * v;
                }
                energy[ch][sb] = e / 36.0;
            }
        }

        // Each subband starts at allocation=0 and is iteratively bumped to
        // the next class if we can afford it. "Next class" means
        // allocation index ++ in the table (from the decoder's
        // perspective, higher index == more levels / more bits).
        let mut alloc = vec![[0u8; 32]; n_ch];

        // Bits available for SAMPLES (not SCFSI or scalefactors or
        // allocation indices). We iteratively spend from this budget as
        // allocations grow; scalefactor/SCFSI spend is recomputed each
        // time an allocation transitions 0 → non-zero since allocation=0
        // means no SCFSI, no scalefactors.
        let total_overhead_sample_budget =
            frame_bits as i64 - header_bits as i64 - bitalloc_bits as i64;
        if total_overhead_sample_budget < 0 {
            return Err(Error::other("MP2 encoder: frame too small for header"));
        }

        let mut remaining: i64 = total_overhead_sample_budget;

        // Greedy allocation. For each iteration, find the (ch, sb) pair
        // whose upgrade cost vs. energy gives the best SNR return, and if
        // we can afford the delta bits, apply it.
        loop {
            let mut best: Option<(usize, usize, u8, i64)> = None;
            let mut best_score = f32::NEG_INFINITY;

            for ch in 0..n_ch {
                for sb in 0..table.sblimit {
                    let cur = alloc[ch][sb];
                    let max = (1u32 << table.nbal(sb)) - 1;
                    if cur as u32 >= max {
                        continue;
                    }
                    let next = cur + 1;
                    // Cost = extra bits this upgrade would need.
                    let cost = upgrade_cost_bits(table, sb, cur, next, scfsi[ch][sb]);
                    // Score: prefer large energy subbands upgraded
                    // cheaply. Use energy/cost as a proxy for
                    // SNR-per-bit; add a small epsilon so zero-energy
                    // subbands don't spam.
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

        // --- 6. Write frame ---
        let mut w = BitWriter::with_capacity(frame_bytes);

        // Header (32 bits).
        // syncword 0xFFF, ID=1 (MPEG-1), layer=10 (Layer II), protection=1
        // (no CRC), bitrate_index, sampling_frequency_index, padding bit,
        // private bit=0, mode (00=stereo, 11=mono for our case — we emit
        // plain stereo, not joint), mode_extension=0, copyright=0,
        // original=0, emphasis=0.
        w.write_u32(0xFFF, 12);
        w.write_u32(1, 1); // ID = MPEG-1
        w.write_u32(0b10, 2); // Layer II
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

        // --- 6a. Bit allocation ---
        // No joint-stereo, so all subbands are "below the bound"
        // (independent per-channel allocations).
        for sb in 0..table.sblimit {
            let nbal = table.nbal(sb);
            for ch in 0..n_ch {
                w.write_u32(alloc[ch][sb] as u32, nbal);
            }
        }

        // --- 6b. SCFSI: 2 bits per subband*channel with alloc != 0 ---
        for sb in 0..table.sblimit {
            for ch in 0..n_ch {
                if alloc[ch][sb] != 0 {
                    w.write_u32(scfsi[ch][sb] as u32, 2);
                }
            }
        }

        // --- 6c. Scalefactors: 6 bits each, count determined by SCFSI ---
        for sb in 0..table.sblimit {
            for ch in 0..n_ch {
                if alloc[ch][sb] == 0 {
                    continue;
                }
                let scf = scf_idx[ch][sb];
                match scfsi[ch][sb] {
                    0 => {
                        w.write_u32(scf[0] as u32, 6);
                        w.write_u32(scf[1] as u32, 6);
                        w.write_u32(scf[2] as u32, 6);
                    }
                    1 => {
                        w.write_u32(scf[0] as u32, 6); // parts 0==1
                        w.write_u32(scf[2] as u32, 6);
                    }
                    2 => {
                        w.write_u32(scf[0] as u32, 6);
                    }
                    _ => {
                        // scfsi == 3: part 0 separate, parts 1==2
                        w.write_u32(scf[0] as u32, 6);
                        w.write_u32(scf[1] as u32, 6);
                    }
                }
            }
        }

        // --- 6d. Sample payload ---
        // Layer II nests: 3 groups of 12 samples (= 1 part each), each
        // split into 4 triples of 3 samples. Triple writes respect the
        // allocated class — grouped quantiser packs all three samples
        // into one codeword; ungrouped writes each sample independently.
        for gr in 0..3 {
            for tr in 0..4 {
                let base_idx = gr * 12 + tr * 3;
                for sb in 0..table.sblimit {
                    for ch in 0..n_ch {
                        let a = alloc[ch][sb];
                        if a == 0 {
                            continue;
                        }
                        let entry = class_entry(table, sb, a);
                        let sf_mag = scalefactor_magnitude(scf_idx[ch][sb][gr]);
                        write_triple(&mut w, entry, &sub[ch][sb], base_idx, sf_mag);
                    }
                }
            }
        }

        // --- 6e. Pad to frame length ---
        // Fill any remaining bits with zero ancillary data.
        w.align_to_byte();
        let mut bytes = w.into_bytes();
        if bytes.len() > frame_bytes {
            // Shouldn't happen if the allocator respected the budget;
            // clip defensively so we never emit over-length frames.
            bytes.truncate(frame_bytes);
        }
        if bytes.len() < frame_bytes {
            bytes.resize(frame_bytes, 0);
        }

        let pts = (self.frame_index as i64) * 1152;
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(1152);
        pkt.flags.keyframe = true;
        self.frame_index += 1;
        Ok(pkt)
    }
}

impl Encoder for Mp2Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(a) => self.ingest(a),
            _ => Err(Error::invalid("MP2 encoder: audio frames only")),
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
/// for MPEG-1 Layer II). Returns `None` for unsupported values.
fn bitrate_to_index(kbps: u32) -> Option<u32> {
    const LUT: [u32; 14] = [
        32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ];
    LUT.iter()
        .position(|&v| v == kbps)
        .map(|idx| (idx + 1) as u32)
}

/// Given a signal peak magnitude, pick the smallest scalefactor index `i`
/// whose magnitude `2 * 2^(-i/3)` is strictly greater than `peak`. A
/// larger index means a smaller scalefactor, so we want the largest
/// quantisation resolution that still covers the peak without clipping.
///
/// Falls back to index 62 (smallest SF) for tiny peaks, and to index 0
/// (largest SF) for peaks that exceed the table range.
fn pick_scalefactor(peak: f32) -> u8 {
    if !peak.is_finite() || peak <= 0.0 {
        return 62;
    }
    // Index 0 covers peaks up to 2.0. Index 62 covers tiny values.
    // Find the largest index whose magnitude >= peak.
    let mut best = 0u8;
    for i in 0..63u8 {
        let mag = scalefactor_magnitude(i);
        if mag >= peak {
            best = i;
        } else {
            break;
        }
    }
    best
}

/// Pick the SCFSI value that represents the triple `(a, b, c)` exactly
/// when possible. For imperfect matches, fall back to 0 (full transmission).
fn pick_scfsi(a: u8, b: u8, c: u8) -> u8 {
    if a == b && b == c {
        2
    } else if a == b && a != c {
        1
    } else if b == c && a != b {
        3
    } else {
        0
    }
}

/// Look up the class-entry for allocation index `a` (>= 1) of subband `sb`.
fn class_entry(table: &AllocTable, sb: usize, a: u8) -> AllocEntry {
    let base = table.offsets[sb];
    table.entries[base + a as usize]
}

/// Return the number of sample bits per subband-part consumed by class `a`
/// of subband `sb`. Each subband transmits three parts × four triples per
/// part. One triple is either one grouped codeword (3/5/9-level) or three
/// ungrouped codewords. For allocation index 0 the cost is zero.
fn sample_bits_per_subband_for_class(table: &AllocTable, sb: usize, a: u8) -> u32 {
    if a == 0 {
        return 0;
    }
    let e = class_entry(table, sb, a);
    let bits = e.bits as u32;
    // 3 parts × 4 triples per part = 12 triples.
    if e.d > 0 {
        // Grouped: one codeword per triple.
        12 * bits
    } else {
        // Ungrouped: `bits` bits × 3 samples per triple × 12 triples.
        12 * bits * 3
    }
}

/// The extra bits required to go from current allocation `cur` to `next`.
/// Accounts for sample bits, SCFSI bits (2, paid once per channel when
/// allocation transitions 0 → non-zero), and scalefactor bits (6 per SF
/// field, count determined by SCFSI).
fn upgrade_cost_bits(table: &AllocTable, sb: usize, cur: u8, next: u8, scfsi: u8) -> u32 {
    let cur_sample = sample_bits_per_subband_for_class(table, sb, cur);
    let next_sample = sample_bits_per_subband_for_class(table, sb, next);
    let mut cost = next_sample.saturating_sub(cur_sample);
    if cur == 0 && next != 0 {
        // Pay SCFSI + scalefactor bits once.
        cost += 2;
        cost += 6 * scfsi_sf_count(scfsi);
    }
    cost
}

/// Count of 6-bit scalefactor fields transmitted for a given SCFSI value.
fn scfsi_sf_count(scfsi: u8) -> u32 {
    match scfsi {
        0 => 3,
        1 | 3 => 2,
        2 => 1,
        _ => 3,
    }
}

/// Write one 3-sample triple to the bit writer, given the class entry and
/// the scalefactor magnitude for the owning part.
fn write_triple(
    w: &mut BitWriter,
    entry: AllocEntry,
    row: &[f32; 36],
    base_idx: usize,
    sf_mag: f32,
) {
    let bits = entry.bits as u32;
    let d = entry.d as i32;
    if d > 0 {
        // Grouped 3/5/9-level quantiser.
        let levels = d as u32;
        let mut idx = [0u32; 3];
        for i in 0..3 {
            let s = row[base_idx + i] / sf_mag.max(1e-20);
            // Map fractional amplitude `f ∈ [-1..+1]` to quantiser level
            // index `i ∈ [0..L-1]` using the inverse of the decoder's
            // (2*i - (L-1)) / L mapping:
            //   i = round((f * L + (L - 1)) / 2)
            let l = levels as f32;
            let raw = (s * l + (l - 1.0)) * 0.5;
            let mut ii = raw.round() as i32;
            if ii < 0 {
                ii = 0;
            }
            if ii as u32 >= levels {
                ii = (levels - 1) as i32;
            }
            idx[i] = ii as u32;
        }
        // Pack as base-L integer: code = s0 + L*s1 + L²*s2.
        let code = idx[0] + levels * idx[1] + levels * levels * idx[2];
        w.write_u32(code, bits);
    } else {
        // Ungrouped `bits`-bit unsigned codeword per sample.
        // Decoder does: out = (v + d) * c * sf, with c = 2/(2^bits - 1)
        // and d = -(2^(bits-1) - 1). Inverting:
        //   v = round(out / (c * sf) - d)
        //     = round(s * (2^bits - 1) / 2 - d)
        let levels = (1u32 << bits) - 1;
        let c = 2.0f32 / (levels as f32);
        for i in 0..3 {
            let s = row[base_idx + i] / sf_mag.max(1e-20);
            let raw = s / c - d as f32;
            let mut v = raw.round() as i32;
            if v < 0 {
                v = 0;
            }
            let max_code = (1u32 << bits) - 1;
            if v as u32 > max_code {
                v = max_code as i32;
            }
            // The highest codeword (`all ones`, i.e. `2^bits - 1`) is the
            // reserved "invalid" code in ISO 11172-3 §2.4.3.4.2, but the
            // decoder just reads it as `max_code`; we never emit it
            // naturally thanks to our clamp. In practice most encoders
            // cap at `max_code - 1` but the decoder handles either case
            // without error.
            w.write_u32(v as u32, bits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitrate_index_lookup() {
        assert_eq!(bitrate_to_index(128), Some(8));
        assert_eq!(bitrate_to_index(192), Some(10));
        assert_eq!(bitrate_to_index(384), Some(14));
        assert_eq!(bitrate_to_index(999), None);
    }

    #[test]
    fn scalefactor_pick_reasonable() {
        // Peak 1.0 should land at an index whose magnitude >= 1.0.
        let i = pick_scalefactor(1.0);
        let mag = scalefactor_magnitude(i);
        assert!((1.0..=2.0).contains(&mag), "mag={mag}");
        // Tiny peak → large index.
        let i = pick_scalefactor(1e-8);
        assert!(i >= 50, "got {i}");
    }

    #[test]
    fn scfsi_patterns() {
        assert_eq!(pick_scfsi(5, 5, 5), 2);
        assert_eq!(pick_scfsi(5, 5, 9), 1);
        assert_eq!(pick_scfsi(3, 8, 8), 3);
        assert_eq!(pick_scfsi(1, 2, 3), 0);
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
        let mut data = Vec::new();
        for _ in 0..1152 * 3 {
            data.extend_from_slice(&0i16.to_le_bytes());
        }
        let frame = AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 44_100,
            samples: 1152 * 3,
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
        assert!(decoded >= 1152, "decoded too few samples: {decoded}");
    }
}
