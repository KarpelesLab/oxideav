//! MP3 packet → AudioFrame decoder, wired into [`oxideav_codec::Decoder`].
//!
//! The decoder threads side info, scalefactors, Huffman, requantise,
//! antialias, IMDCT, and polyphase synthesis. It maintains a per-channel
//! IMDCT overlap state and a per-channel synthesis FIFO across frames,
//! plus a 4 KiB bit reservoir.
//!
//! **Limitations** (this session):
//! - MPEG-1 Layer III only. MPEG-2 LSF / MPEG-2.5 packets return
//!   `Error::Unsupported`.
//! - Intensity-stereo (mode_extension bit 0) is not implemented; pure-IS
//!   joint-stereo frames will sound off but won't fail to decode. M/S
//!   joint-stereo (the dominant case) is supported.
//! - No CRC verification.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::BitReader;
use crate::frame::{parse_frame_header, ChannelMode, MpegVersion};
use crate::huffman::{decode_count1, decode_pair};
use crate::imdct::{imdct_granule, ImdctState};
use crate::requantize::{antialias, ms_stereo, requantize_granule};
use crate::reservoir::Reservoir;
use crate::scalefactor::{decode_mpeg1 as decode_sf_mpeg1, ScaleFactors};
use crate::sfband::sfband_long;
use crate::sideinfo::SideInfo;
use crate::synthesis::{synthesize_granule, SynthesisState};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mp3Decoder {
        codec_id: params.codec_id.clone(),
        time_base: TimeBase::new(1, 48_000),
        pending: None,
        reservoir: Reservoir::new(),
        prev_sf: [[ScaleFactors::default(); 2]; 2],
        imdct_state: [ImdctState::new(), ImdctState::new()],
        synth_state: [SynthesisState::new(), SynthesisState::new()],
        eof: false,
    }))
}

// ScaleFactors needs Copy for the array init — it's small enough.
impl Copy for ScaleFactors {}

struct Mp3Decoder {
    codec_id: CodecId,
    time_base: TimeBase,
    pending: Option<Packet>,
    reservoir: Reservoir,
    /// prev_sf[gr][ch] — only [1][ch] matters for scfsi reuse.
    prev_sf: [[ScaleFactors; 2]; 2],
    imdct_state: [ImdctState; 2],
    synth_state: [SynthesisState; 2],
    eof: bool,
}

impl Decoder for Mp3Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "MP3 decoder: receive_frame must be called before sending another packet",
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
}

impl Mp3Decoder {
    fn decode_packet(&mut self, pkt: &Packet) -> Result<Frame> {
        let data = &pkt.data;
        let hdr = parse_frame_header(data)?;
        if hdr.version != MpegVersion::Mpeg1 {
            return Err(Error::unsupported(
                "MP3 decoder: MPEG-2 LSF / MPEG-2.5 not yet supported",
            ));
        }
        let channels = hdr.channels() as usize;
        let crc_bytes = if hdr.no_crc { 0 } else { 2 };
        let header_len = 4 + crc_bytes;
        let si_bytes = hdr.side_info_bytes();
        if data.len() < header_len + si_bytes {
            return Err(Error::NeedMore);
        }
        let si = SideInfo::parse_mpeg1(&hdr, &data[header_len..])?;

        // Update time base on first frame.
        self.time_base = TimeBase::new(1, hdr.sample_rate as i64);

        let main_data_start = header_len + si_bytes;
        let main_data = &data[main_data_start..];

        // Combine reservoir + this frame's main data.
        let prev_view: Vec<u8> = self
            .reservoir
            .view_from_lookback(si.main_data_begin)
            .ok_or_else(|| Error::invalid("MP3 decoder: bit reservoir underflow (cold start?)"))?
            .to_vec();
        let mut combined = prev_view;
        combined.extend_from_slice(main_data);

        // Decode all granules.
        let mut pcm = vec![[[0.0f32; 576]; 2]; 2]; // pcm[gr][ch][i]

        // Track the expected next-granule bit offset within `combined`;
        // this lets each granule start cleanly at part2_3_length boundary
        // even when the previous one over- or under-read by a few bits.
        let mut next_granule_bit: u64 = 0;
        for gr in 0..2 {
            for ch in 0..channels {
                let gc = si.granules[gr][ch];

                // Build a fresh BitReader for this granule that starts at
                // the precise expected bit offset within `combined`.
                let bit_off = next_granule_bit;
                let byte_off = (bit_off / 8) as usize;
                if byte_off >= combined.len() {
                    return Err(Error::invalid(
                        "MP3 decoder: ran out of main_data while decoding granule",
                    ));
                }
                let mut br = BitReader::new(&combined[byte_off..]);
                let skip = (bit_off % 8) as u32;
                if skip > 0 {
                    br.read_u32(skip)?;
                }
                let part_start = br.bit_position();
                let part_end_bit = part_start + gc.part2_3_length as u64;

                // Scalefactors first.
                let sf = decode_sf_mpeg1(&mut br, &gc, &si.scfsi[ch], gr, &self.prev_sf[gr][ch])?;
                self.prev_sf[gr][ch] = sf;

                // Huffman big-value pairs.
                let mut is_ = [0i32; 576];
                let mut idx = 0usize;
                let big = (gc.big_values * 2) as usize;

                // Compute region boundaries (long-block layout).
                let bounds = sfband_long(hdr.sample_rate);
                let r0_end = if gc.window_switching_flag && gc.block_type == 2 {
                    36 // shortcut for short blocks
                } else {
                    bounds[(gc.region0_count as usize + 1).min(22)] as usize
                };
                let r1_end = if gc.window_switching_flag && gc.block_type == 2 {
                    576
                } else {
                    bounds[(gc.region0_count as usize + gc.region1_count as usize + 2).min(22)]
                        as usize
                };

                // Big-values: read exactly `big_values` pairs as advertised
                // in the side info. The encoder allocates enough part2_3
                // bits for this many pairs.
                while idx < big.min(576) {
                    let table = if idx < r0_end {
                        gc.table_select[0]
                    } else if idx < r1_end {
                        gc.table_select[1]
                    } else {
                        gc.table_select[2]
                    };
                    if table == 0 {
                        // pair of zeros
                        is_[idx] = 0;
                        if idx + 1 < 576 {
                            is_[idx + 1] = 0;
                        }
                        idx += 2;
                        continue;
                    }
                    let (x, y) = decode_pair(&mut br, table)?;
                    is_[idx] = x;
                    if idx + 1 < 576 {
                        is_[idx + 1] = y;
                    }
                    idx += 2;
                }

                // Count1 region.
                while idx + 4 <= 576 && br.bit_position() < part_end_bit {
                    let (v, w, x, y) = decode_count1(&mut br, gc.count1table_select)?;
                    is_[idx] = v;
                    is_[idx + 1] = w;
                    is_[idx + 2] = x;
                    is_[idx + 3] = y;
                    idx += 4;
                }

                // Advance the granule cursor — small over- or under-reads
                // within the part2_3_length envelope are absorbed here.
                next_granule_bit = bit_off + gc.part2_3_length as u64;

                // Requantise.
                let mut xr = [0.0f32; 576];
                requantize_granule(&is_, &mut xr, &gc, &sf, hdr.sample_rate);

                // Antialias (long blocks only / mixed-block long part).
                antialias(&mut xr, &gc);

                // Stash for stereo processing.
                pcm[gr][ch] = xr;
            }

            // Stereo processing on the granule (after both channels).
            if channels == 2 && hdr.channel_mode == ChannelMode::JointStereo {
                let ms_on = (hdr.mode_extension & 0x2) != 0;
                if ms_on {
                    // Borrow split.
                    let (l, r) = pcm[gr].split_at_mut(1);
                    ms_stereo(&mut l[0], &mut r[0]);
                }
                // Intensity stereo not yet implemented — leave as-is (MS is
                // the dominant joint-stereo mode; pure-IS-only frames will
                // sound off).
            }
        }

        // IMDCT + polyphase synthesis per granule per channel.
        let total_samples = 1152u32; // MPEG-1
        let bytes_per_sample = SampleFormat::S16.bytes_per_sample();
        let mut out_bytes =
            Vec::with_capacity(total_samples as usize * channels * bytes_per_sample);

        let mut pcm_per_gr = [[0.0f32; 576]; 2]; // [ch][i] for current granule
        for gr in 0..2 {
            for ch in 0..channels {
                let mut sb = [[0.0f32; 18]; 32];
                let gc = si.granules[gr][ch];
                imdct_granule(
                    &pcm[gr][ch],
                    &mut sb,
                    &mut self.imdct_state[ch],
                    if gc.window_switching_flag {
                        gc.block_type
                    } else {
                        0
                    },
                    gc.mixed_block_flag,
                );
                synthesize_granule(&mut self.synth_state[ch], &sb, &mut pcm_per_gr[ch]);
            }
            // Interleave samples.
            for i in 0..576 {
                for ch in 0..channels {
                    let f = pcm_per_gr[ch][i].clamp(-1.0, 1.0);
                    let s = (f * 32767.0) as i16;
                    out_bytes.extend_from_slice(&s.to_le_bytes());
                }
            }
        }

        // Append this frame's main_data to the reservoir for next frame.
        self.reservoir.append(main_data);

        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: channels as u16,
            sample_rate: hdr.sample_rate,
            samples: total_samples,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![out_bytes],
        }))
    }
}
