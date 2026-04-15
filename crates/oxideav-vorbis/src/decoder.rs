//! Vorbis decoder top-level: stitches the bitstream parsers, floor +
//! residue decoders, channel coupling, IMDCT and windowed overlap-add into
//! a [`oxideav_codec::Decoder`] implementation.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::BitReader;
use crate::floor::{decode_floor_packet, synth_floor1, Floor1Decoded};
use crate::identification::{parse_identification_header, Identification};
use crate::imdct::{imdct_naive, sin_window_sample};
use crate::residue::decode_residue;
use crate::setup::{parse_setup, Floor, Setup};

/// Build a Vorbis decoder from the codec parameters.
///
/// `params.extradata` must be the Xiph-laced 3-packet header blob (id +
/// comment + setup) — that's the format produced by both `oxideav-ogg`
/// and `oxideav-mkv` for Vorbis tracks.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let packets = split_xiph_lacing(&params.extradata)
        .ok_or_else(|| Error::invalid("Vorbis decoder: bad extradata lacing"))?;
    if packets.len() < 3 {
        return Err(Error::invalid(
            "Vorbis decoder: expected 3 header packets in extradata",
        ));
    }
    let id = parse_identification_header(&packets[0])?;
    // packets[1] is the comment header — content is metadata only, ignore.
    let setup = parse_setup(&packets[2], id.audio_channels)?;
    let blocksize_short = 1u32 << id.blocksize_0;
    let blocksize_long = 1u32 << id.blocksize_1;
    let time_base = TimeBase::new(1, id.audio_sample_rate as i64);

    Ok(Box::new(VorbisDecoder {
        codec_id: params.codec_id.clone(),
        id,
        setup,
        blocksize_short,
        blocksize_long,
        time_base,
        prev_tail: Vec::new(),
        pending: None,
        eof: false,
        emit_pts: 0,
    }))
}

struct VorbisDecoder {
    codec_id: CodecId,
    id: Identification,
    setup: Setup,
    blocksize_short: u32,
    blocksize_long: u32,
    time_base: TimeBase,
    /// Per-channel "right tail" samples saved from the previous packet's
    /// IMDCT output. This is a raw (unwindowed) slice of the previous
    /// block in the range `[right_win_start, right_win_end)` — exactly
    /// the region that overlaps with the current block's left window.
    ///
    /// Empty on the first decoded packet (no prior block to overlap with);
    /// that first packet emits zero PCM samples but populates this tail.
    prev_tail: Vec<Vec<f32>>,
    pending: Option<Packet>,
    eof: bool,
    /// Running pts of the next sample we'll emit.
    emit_pts: i64,
}

impl Decoder for VorbisDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "Vorbis decoder: receive_frame must be called before sending another packet",
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
        decode_one(self, &pkt)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn decode_one(d: &mut VorbisDecoder, packet: &Packet) -> Result<Frame> {
    let n_channels = d.id.audio_channels as usize;
    let mut br = BitReader::new(&packet.data);
    let trace = std::env::var_os("OXIDEAV_VORBIS_TRACE").is_some();

    // Audio packet header: type bit + mode index.
    let header_bit = br.read_bit()?;
    if header_bit {
        return Err(Error::invalid("Vorbis audio packet: type bit set"));
    }
    // Number of bits needed for mode index = ilog(mode_count - 1). When only
    // one mode exists, this is 0 bits.
    let mode_bits = if d.setup.modes.len() <= 1 {
        0
    } else {
        ilog(d.setup.modes.len() as u32 - 1)
    };
    let mode_index = br.read_u32(mode_bits)? as usize;
    if trace {
        eprintln!(
            "[vorbis] pkt bytes={} mode_bits={} mode={} bitpos={}",
            packet.data.len(),
            mode_bits,
            mode_index,
            br.bit_position()
        );
    }
    if mode_index >= d.setup.modes.len() {
        return Err(Error::invalid("Vorbis audio packet: invalid mode index"));
    }
    let mode = d.setup.modes[mode_index].clone();
    let block_long = mode.blockflag;
    let n = if block_long {
        d.blocksize_long as usize
    } else {
        d.blocksize_short as usize
    };
    let n_half = n / 2;

    let (prev_long_for_window, next_long_for_window) = if block_long {
        let prev = br.read_bit()?;
        let next = br.read_bit()?;
        (prev, next)
    } else {
        (false, false)
    };

    let mapping = d.setup.mappings[mode.mapping as usize].clone();
    if trace {
        eprintln!(
            "[vorbis] blocksize={} submaps={} coupling_steps={} bitpos={}",
            n,
            mapping.submaps,
            mapping.coupling.len(),
            br.bit_position()
        );
    }

    // Per-channel floor decode.
    let mut floors_decoded: Vec<Floor1Decoded> = Vec::with_capacity(n_channels);
    let mut no_residue = vec![false; n_channels];
    for ch in 0..n_channels {
        let submap = if mapping.submaps > 1 {
            mapping.mux[ch]
        } else {
            0
        };
        let floor_idx = mapping.submap_floor[submap as usize] as usize;
        let floor = &d.setup.floors[floor_idx];
        let dec = decode_floor_packet(floor, &d.setup.codebooks, &mut br)?;
        if trace {
            eprintln!(
                "[vorbis] floor ch{}: unused={} y_len={} bitpos={}",
                ch,
                dec.unused,
                dec.y.len(),
                br.bit_position()
            );
        }
        no_residue[ch] = dec.unused;
        floors_decoded.push(dec);
    }

    // Channel coupling fixup of `no_residue`: if EITHER channel of a coupled
    // pair has residue, both must be decoded.
    for &(mag, ang) in &mapping.coupling {
        let mi = mag as usize;
        let ai = ang as usize;
        if !no_residue[mi] || !no_residue[ai] {
            no_residue[mi] = false;
            no_residue[ai] = false;
        }
    }

    // Allocate per-channel spectral buffers (length = n/2).
    let mut spectrum: Vec<Vec<f32>> = (0..n_channels).map(|_| vec![0f32; n_half]).collect();

    // Per-submap residue decode. Channels in the same submap are decoded
    // together (so type-2 interleaving works).
    for sm in 0..mapping.submaps as usize {
        // Find channels assigned to this submap.
        let mut ch_list: Vec<usize> = Vec::new();
        for ch in 0..n_channels {
            let smi = if mapping.submaps > 1 {
                mapping.mux[ch] as usize
            } else {
                0
            };
            if smi == sm {
                ch_list.push(ch);
            }
        }
        if ch_list.is_empty() {
            continue;
        }
        let res_idx = mapping.submap_residue[sm] as usize;
        let residue = &d.setup.residues[res_idx];
        let mut sub_vectors: Vec<Vec<f32>> = ch_list.iter().map(|_| vec![0f32; n_half]).collect();
        let dnd: Vec<bool> = ch_list.iter().map(|&ch| no_residue[ch]).collect();
        decode_residue(
            residue,
            &d.setup.codebooks,
            n_half,
            &dnd,
            &mut sub_vectors,
            &mut br,
        )?;
        // Add the submap residue back into the per-channel spectrum.
        for (i, &ch) in ch_list.iter().enumerate() {
            for k in 0..n_half {
                spectrum[ch][k] += sub_vectors[i][k];
            }
        }
    }

    // Inverse channel coupling (Vorbis I §1.3.3). Apply in REVERSE order.
    for &(mag, ang) in mapping.coupling.iter().rev() {
        let mi = mag as usize;
        let ai = ang as usize;
        for k in 0..n_half {
            let m = spectrum[mi][k];
            let a = spectrum[ai][k];
            let (new_m, new_a) = if m > 0.0 {
                if a > 0.0 {
                    (m, m - a)
                } else {
                    (m + a, m)
                }
            } else if a > 0.0 {
                (m, m + a)
            } else {
                (m - a, m)
            };
            spectrum[mi][k] = new_m;
            spectrum[ai][k] = new_a;
        }
    }

    // Multiply spectrum by floor curve per channel.
    for ch in 0..n_channels {
        if no_residue[ch] {
            for v in spectrum[ch].iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let submap = if mapping.submaps > 1 {
            mapping.mux[ch]
        } else {
            0
        };
        let floor_idx = mapping.submap_floor[submap as usize] as usize;
        match &d.setup.floors[floor_idx] {
            Floor::Type1(f1) => {
                synth_floor1(f1, &floors_decoded[ch], n_half, &mut spectrum[ch])?;
            }
            Floor::Type0(_) => {
                return Err(Error::unsupported("Vorbis floor 0 decode not implemented"));
            }
        }
    }

    if trace {
        eprintln!(
            "[vorbis] end_of_decode bitpos={}/{}",
            br.bit_position(),
            packet.data.len() * 8
        );
    }

    // IMDCT per channel → time-domain length-n samples (UNWINDOWED). The
    // window is applied below, only in the overlap regions, per
    // Vorbis I §1.3.4.
    let mut td: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        let mut out = vec![0f32; n];
        imdct_naive(&spectrum[ch], &mut out);
        td.push(out);
    }

    // Compute the four asymmetric window boundaries (Vorbis I §4.3.1 /
    // lewton's reference). `bs0 = 1 << blocksize_0` is the SHORT blocksize.
    // For long blocks, the left transition is narrow (short-sized) when
    // `prev_long_for_window=false`, and similarly on the right.
    let bs0 = d.blocksize_short as usize;
    let (left_win_start, left_win_end) = if !block_long || prev_long_for_window {
        // Short curr: always symmetric with full half-window. Long curr with
        // prev=long: long rising window covering [0..n/2].
        (0usize, n / 2)
    } else {
        ((n - bs0) / 4, (n + bs0) / 4)
    };
    let (right_win_start, right_win_end) = if !block_long || next_long_for_window {
        (n / 2, n)
    } else {
        ((3 * n - bs0) / 4, (3 * n + bs0) / 4)
    };
    let left_overlap_n = left_win_end - left_win_start;
    let right_overlap_n = right_win_end - right_win_start;

    // Overlap-add with the previous packet's saved tail. Emit samples in
    // `[left_win_start, right_win_start)`. The prev tail length equals
    // `left_overlap_n` by construction (its size was chosen when saved).
    let mut output_samples = vec![Vec::<f32>::new(); n_channels];
    if !d.prev_tail.is_empty() {
        for ch in 0..n_channels {
            let prev = &d.prev_tail[ch];
            let curr = &mut td[ch];
            // Sanity: if prev was stored with a differently-sized overlap
            // region (e.g. a stream-format bug upstream), clamp.
            let plen = prev.len().min(left_overlap_n);
            // win_slope[i] = sin(pi/2 * sin^2(pi*(i+0.5)/(2*plen)))
            // rising on the current block; prev was stored unwindowed, so
            // we multiply it by the falling slope (= win_slope[plen-1-i]).
            for i in 0..plen {
                let rising = sin_window_sample(i, 2 * plen);
                let falling = sin_window_sample(plen - 1 - i, 2 * plen);
                let ci = left_win_start + i;
                curr[ci] = curr[ci] * rising + prev[i] * falling;
            }
            // Emit from left_win_start to right_win_start.
            let mut emit = Vec::with_capacity(right_win_start - left_win_start);
            emit.extend_from_slice(&curr[left_win_start..right_win_start]);
            output_samples[ch] = emit;
        }
    }

    // Save the "right tail" for next iteration: raw IMDCT values in
    // [right_win_start, right_win_end). These remain unwindowed; next
    // packet will apply the matching falling slope during its OLA step.
    let mut new_tail: Vec<Vec<f32>> = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        new_tail.push(td[ch][right_win_start..right_win_end].to_vec());
    }
    d.prev_tail = new_tail;
    let _ = right_overlap_n;

    // Pack interleaved S16 PCM.
    let n_samples = output_samples.first().map(|v| v.len()).unwrap_or(0) as u32;
    let pts = packet.pts.unwrap_or(d.emit_pts);
    d.emit_pts = pts + n_samples as i64;
    let mut interleaved = Vec::with_capacity(n_samples as usize * n_channels * 2);
    for i in 0..n_samples as usize {
        for ch in 0..n_channels {
            let s = output_samples[ch].get(i).copied().unwrap_or(0.0);
            let clamped = (s * 32768.0).clamp(-32768.0, 32767.0) as i16;
            interleaved.extend_from_slice(&clamped.to_le_bytes());
        }
    }

    Ok(Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: n_channels as u16,
        sample_rate: d.id.audio_sample_rate,
        samples: n_samples,
        pts: Some(pts),
        time_base: d.time_base,
        data: vec![interleaved],
    }))
}

/// Split a Xiph-laced 3-packet extradata blob into individual packet bytes.
fn split_xiph_lacing(blob: &[u8]) -> Option<Vec<Vec<u8>>> {
    if blob.is_empty() {
        return None;
    }
    let n_packets = blob[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n_packets);
    let mut i = 1usize;
    for _ in 0..n_packets - 1 {
        let mut s = 0usize;
        loop {
            if i >= blob.len() {
                return None;
            }
            let b = blob[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    let used: usize = sizes.iter().sum();
    if i + used > blob.len() {
        return None;
    }
    let last = blob.len() - i - used;
    sizes.push(last);
    let mut packets = Vec::with_capacity(n_packets);
    for sz in sizes {
        packets.push(blob[i..i + sz].to_vec());
        i += sz;
    }
    Some(packets)
}

fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}
