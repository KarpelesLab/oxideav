//! Top-level AAC-LC packet decoder. Wires the ADTS / ASC parser, the SCE /
//! CPE element decoders, and the IMDCT/overlap state into a single
//! `oxideav_codec::Decoder` impl.
//!
//! ISO/IEC 14496-3 §4.5.2.1 (raw_data_block) and §4.6.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::adts::parse_adts_header;
use crate::asc::parse_asc;
use crate::bitreader::BitReader;
use crate::ics::{
    decode_spectrum_long, decode_spectrum_short, parse_ics_info, parse_scalefactors,
    parse_section_data, IcsInfo, SectionData, INTENSITY_HCB, INTENSITY_HCB2, NOISE_HCB, SPEC_LEN,
    ZERO_HCB,
};
use crate::sfband::{SWB_LONG, SWB_SHORT};
use crate::syntax::{ElementType, WindowSequence, AOT_AAC_LC};
use crate::synth::{imdct_and_overlap, ChannelState, FRAME_LEN};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    // Figure out the stream config. Two paths:
    //   (a) extradata holds an AudioSpecificConfig (MP4 path).
    //   (b) ADTS — config will come from the first packet's ADTS header.
    let (sf_index, channels, object_type) = if !params.extradata.is_empty() {
        let asc = parse_asc(&params.extradata)?;
        if asc.sbr_present || asc.ps_present {
            return Err(Error::unsupported(
                "AAC: SBR/PS (HE-AAC v1/v2) not supported",
            ));
        }
        if asc.object_type != AOT_AAC_LC {
            return Err(Error::unsupported(
                "AAC: only AAC-LC profile (object_type=2) supported",
            ));
        }
        (
            sample_rate_to_index(asc.sampling_frequency).unwrap_or(asc.sampling_frequency_index),
            asc.channel_configuration,
            asc.object_type,
        )
    } else {
        // Will be filled in after seeing the first ADTS frame.
        (0xFF, 0, 0)
    };

    Ok(Box::new(AacDecoder {
        codec_id: params.codec_id.clone(),
        time_base: TimeBase::new(1, params.sample_rate.unwrap_or(44_100) as i64),
        pending: None,
        eof: false,
        sf_index,
        channels,
        object_type,
        chans: vec![ChannelState::new(); 2],
        configured: !params.extradata.is_empty(),
    }))
}

fn sample_rate_to_index(sr: u32) -> Option<u8> {
    use crate::syntax::SAMPLE_RATES;
    SAMPLE_RATES.iter().position(|&r| r == sr).map(|i| i as u8)
}

struct AacDecoder {
    codec_id: CodecId,
    time_base: TimeBase,
    pending: Option<Packet>,
    eof: bool,
    sf_index: u8,
    channels: u8,
    object_type: u8,
    chans: Vec<ChannelState>,
    configured: bool,
}

impl Decoder for AacDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "AAC decoder: receive_frame must be called before sending another packet",
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

impl AacDecoder {
    fn decode_packet(&mut self, pkt: &Packet) -> Result<Frame> {
        // Detect ADTS by syncword. Otherwise treat as raw_data_block (e.g. MP4).
        let data = &pkt.data;
        let (payload_offset, frame_end) =
            if data.len() >= 7 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0 {
                let hdr = parse_adts_header(data)?;
                if hdr.object_type != AOT_AAC_LC {
                    return Err(Error::unsupported(
                        "AAC: ADTS header advertises non-LC profile",
                    ));
                }
                self.sf_index = hdr.sampling_freq_index;
                self.channels = hdr.channel_configuration;
                self.object_type = hdr.object_type;
                self.configured = true;
                self.time_base = TimeBase::new(1, hdr.sample_rate().unwrap_or(44_100) as i64);
                (hdr.header_length(), hdr.frame_length.min(data.len()))
            } else {
                if !self.configured {
                    return Err(Error::invalid(
                        "AAC: first packet has no ADTS sync and no extradata config",
                    ));
                }
                (0, data.len())
            };
        let payload = &data[payload_offset..frame_end];

        // Decode raw_data_block.
        let mut br = BitReader::new(payload);
        let mut pcm: [Vec<f32>; 2] = [vec![0.0; FRAME_LEN], vec![0.0; FRAME_LEN]];
        let mut got_channels: usize = 0;

        loop {
            let id = br.read_u32(3)?;
            let kind = ElementType::from_id(id);
            match kind {
                ElementType::Sce => {
                    let _instance_tag = br.read_u32(4)?;
                    let mut spec = [0.0f32; SPEC_LEN];
                    let (info, sf, sec) = decode_ics(&mut br, self.sf_index, false)?;
                    fill_spectrum(&mut br, &info, &sec, &sf, &mut spec)?;
                    let mut channel_pcm = [0.0f32; FRAME_LEN];
                    imdct_and_overlap(
                        &spec,
                        info.window_sequence,
                        info.window_shape,
                        &mut self.chans[got_channels.min(1)],
                        &mut channel_pcm,
                    );
                    pcm[got_channels.min(1)].copy_from_slice(&channel_pcm);
                    got_channels += 1;
                }
                ElementType::Cpe => {
                    let _instance_tag = br.read_u32(4)?;
                    let common_window = br.read_bit()?;
                    if common_window {
                        // Shared ICS info, then ms_mask flags.
                        let info = parse_ics_info(&mut br, self.sf_index)?;
                        let ms_mask_present = br.read_u32(2)? as u8;
                        let max_sfb = info.max_sfb as usize;
                        let groups = info.num_window_groups as usize;
                        let mut ms_used = vec![false; groups * max_sfb];
                        match ms_mask_present {
                            0 => { /* ms not used */ }
                            1 => {
                                for i in 0..groups * max_sfb {
                                    ms_used[i] = br.read_bit()?;
                                }
                            }
                            2 => {
                                for i in 0..groups * max_sfb {
                                    ms_used[i] = true;
                                }
                            }
                            _ => return Err(Error::invalid("AAC: reserved ms_mask_present=3")),
                        }
                        let mut spec = [[0.0f32; SPEC_LEN]; 2];
                        let mut secs: [SectionData; 2] = Default::default();
                        let mut sfs: [Vec<i32>; 2] = Default::default();
                        let infos: [IcsInfo; 2] = [info.clone(), info.clone()];
                        for ch in 0..2 {
                            let gg = br.read_u32(8)? as u8;
                            let sec = parse_section_data(&mut br, &infos[ch])?;
                            let sf = parse_scalefactors(&mut br, &infos[ch], &sec, gg)?;
                            // Pulse, TNS, gain control all "absent" in AAC-LC mainstream.
                            let _pulse = br.read_bit()?;
                            if _pulse {
                                return Err(Error::unsupported(
                                    "AAC: pulse_data_present not implemented",
                                ));
                            }
                            let _tns = br.read_bit()?;
                            if _tns {
                                skip_tns_data(&mut br, &infos[ch])?;
                            }
                            let _gain_control = br.read_bit()?;
                            if _gain_control {
                                return Err(Error::unsupported(
                                    "AAC: gain_control_data_present in LC stream",
                                ));
                            }
                            secs[ch] = sec;
                            sfs[ch] = sf;
                            fill_spectrum(&mut br, &infos[ch], &secs[ch], &sfs[ch], &mut spec[ch])?;
                        }
                        // M/S stereo: replace L,R with (L+R)/sqrt(2), (L-R)/sqrt(2)?
                        // Per spec §4.6.13.3:
                        //   L = M + S; R = M - S  (no sqrt scaling — IS-only normalisation
                        //   uses sqrt(2), but MS as defined in 14496-3 is L=M+S, R=M-S).
                        apply_ms_stereo(&infos[0], &secs, &ms_used, &mut spec);
                        for ch in 0..2 {
                            let mut channel_pcm = [0.0f32; FRAME_LEN];
                            imdct_and_overlap(
                                &spec[ch],
                                infos[ch].window_sequence,
                                infos[ch].window_shape,
                                &mut self.chans[ch],
                                &mut channel_pcm,
                            );
                            pcm[ch].copy_from_slice(&channel_pcm);
                        }
                        got_channels = 2;
                    } else {
                        // Independent ICS for each channel.
                        let mut spec = [[0.0f32; SPEC_LEN]; 2];
                        let mut infos: [IcsInfo; 2] = Default::default();
                        for ch in 0..2 {
                            let (info, sf, sec) = decode_ics(&mut br, self.sf_index, true)?;
                            fill_spectrum(&mut br, &info, &sec, &sf, &mut spec[ch])?;
                            infos[ch] = info;
                        }
                        for ch in 0..2 {
                            let mut channel_pcm = [0.0f32; FRAME_LEN];
                            imdct_and_overlap(
                                &spec[ch],
                                infos[ch].window_sequence,
                                infos[ch].window_shape,
                                &mut self.chans[ch],
                                &mut channel_pcm,
                            );
                            pcm[ch].copy_from_slice(&channel_pcm);
                        }
                        got_channels = 2;
                    }
                }
                ElementType::Lfe => {
                    return Err(Error::unsupported("AAC: LFE element not implemented"));
                }
                ElementType::Cce => {
                    return Err(Error::unsupported("AAC: CCE element not implemented"));
                }
                ElementType::Dse => {
                    let _instance_tag = br.read_u32(4)?;
                    let data_byte_align = br.read_bit()?;
                    let mut count = br.read_u32(8)?;
                    if count == 255 {
                        count += br.read_u32(8)?;
                    }
                    if data_byte_align {
                        br.align_to_byte();
                    }
                    for _ in 0..count {
                        br.read_u32(8)?;
                    }
                }
                ElementType::Pce => {
                    return Err(Error::unsupported("AAC: PCE element not implemented"));
                }
                ElementType::Fil => {
                    let mut count = br.read_u32(4)?;
                    if count == 15 {
                        count += br.read_u32(8)? - 1;
                    }
                    // SBR extension payloads start with extension_type=0xD/0xE/0xF —
                    // we treat any non-empty extension as SBR refusal if it claims so.
                    if count > 0 {
                        // Peek extension_type (4 bits) without committing.
                        let peeked = br.peek_u32(4)?;
                        let is_sbr = peeked == 0xD || peeked == 0xE;
                        if is_sbr {
                            return Err(Error::unsupported(
                                "AAC: SBR extension payload — HE-AAC not supported",
                            ));
                        }
                        // Otherwise skip.
                        for _ in 0..count {
                            br.read_u32(8)?;
                        }
                    }
                }
                ElementType::End => break,
            }
        }

        // Convert PCM to interleaved S16.
        let channels_out = if got_channels == 0 {
            1
        } else {
            got_channels.min(self.channels.max(1) as usize)
        };
        let bytes_per_sample = SampleFormat::S16.bytes_per_sample();
        let mut out_bytes = Vec::with_capacity(FRAME_LEN * channels_out * bytes_per_sample);
        for n in 0..FRAME_LEN {
            for ch in 0..channels_out {
                let v = pcm[ch][n].clamp(-1.0, 1.0);
                let s = (v * 32767.0) as i16;
                out_bytes.extend_from_slice(&s.to_le_bytes());
            }
        }

        let sample_rate = crate::syntax::sample_rate(self.sf_index).unwrap_or(44_100);
        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: channels_out as u16,
            sample_rate,
            samples: FRAME_LEN as u32,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![out_bytes],
        }))
    }
}

/// Decode a single-channel ICS into (info, scalefactors, section_data).
/// Reads global_gain, ics_info, section_data, scalefactors, then advances
/// past pulse/TNS/gain-control flags. Spectrum decoding is left to caller.
fn decode_ics(
    br: &mut BitReader<'_>,
    sf_index: u8,
    is_in_cpe: bool,
) -> Result<(IcsInfo, Vec<i32>, SectionData)> {
    let global_gain = br.read_u32(8)? as u8;
    let info = parse_ics_info(br, sf_index)?;
    let sec = parse_section_data(br, &info)?;
    let sf = parse_scalefactors(br, &info, &sec, global_gain)?;
    let _pulse = br.read_bit()?;
    if _pulse {
        return Err(Error::unsupported(
            "AAC: pulse_data_present not implemented",
        ));
    }
    let _tns = br.read_bit()?;
    if _tns {
        skip_tns_data(br, &info)?;
    }
    let _gain_control = br.read_bit()?;
    if _gain_control {
        return Err(Error::unsupported("AAC: gain_control in LC stream"));
    }
    let _ = is_in_cpe;
    Ok((info, sf, sec))
}

fn fill_spectrum(
    br: &mut BitReader<'_>,
    info: &IcsInfo,
    sec: &SectionData,
    sf: &[i32],
    spec: &mut [f32; SPEC_LEN],
) -> Result<()> {
    if info.window_sequence == WindowSequence::EightShort {
        decode_spectrum_short(br, info, sec, sf, spec)
    } else {
        decode_spectrum_long(br, info, sec, sf, spec)
    }
}

/// Skip over the `tns_data` syntax element. We don't apply TNS yet but we
/// must consume the right number of bits to keep the stream aligned.
fn skip_tns_data(br: &mut BitReader<'_>, info: &IcsInfo) -> Result<()> {
    let n_filt_bits = if info.window_sequence == WindowSequence::EightShort {
        1
    } else {
        2
    };
    let length_bits = if info.window_sequence == WindowSequence::EightShort {
        4
    } else {
        6
    };
    let order_bits = if info.window_sequence == WindowSequence::EightShort {
        3
    } else {
        5
    };
    let n_windows = info.num_window_groups as usize;
    for _ in 0..n_windows {
        let n_filt = br.read_u32(n_filt_bits)? as usize;
        if n_filt > 0 {
            let coef_res = br.read_u32(1)? as u32;
            for _ in 0..n_filt {
                let _length = br.read_u32(length_bits)?;
                let order = br.read_u32(order_bits)? as usize;
                if order > 0 {
                    let _direction = br.read_bit()?;
                    let coef_compress = br.read_u32(1)? as u32;
                    let coef_bits = (3 + coef_res) - coef_compress;
                    for _ in 0..order {
                        br.read_u32(coef_bits)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Apply M/S stereo decoding in-place. Spec §4.6.13.
fn apply_ms_stereo(
    info: &IcsInfo,
    secs: &[SectionData; 2],
    ms_used: &[bool],
    spec: &mut [[f32; SPEC_LEN]; 2],
) {
    let max_sfb = info.max_sfb as usize;
    let groups = info.num_window_groups as usize;
    let starts = crate::ics::group_starts(info);
    let is_short = info.window_sequence.is_eight_short();
    let swb = if is_short {
        SWB_SHORT[info.sf_index as usize]
    } else {
        SWB_LONG[info.sf_index as usize]
    };

    for g in 0..groups {
        let group_len = info.window_group_length[g] as usize;
        for sfb in 0..max_sfb {
            if !ms_used[g * max_sfb + sfb] {
                continue;
            }
            let cb = secs[0].sfb_cb[g * max_sfb + sfb];
            // M/S only applies to non-zero, non-IS, non-noise bands.
            if cb == ZERO_HCB || cb == NOISE_HCB || cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
                continue;
            }
            let band_start = swb[sfb] as usize;
            let band_end = swb[sfb + 1] as usize;
            if is_short {
                let win_start_offset = starts[g] * 128;
                for w in 0..group_len {
                    for j in band_start..band_end {
                        let idx = win_start_offset + w * 128 + j;
                        let m = spec[0][idx];
                        let s = spec[1][idx];
                        spec[0][idx] = m + s;
                        spec[1][idx] = m - s;
                    }
                }
            } else {
                for j in band_start..band_end {
                    let m = spec[0][j];
                    let s = spec[1][j];
                    spec[0][j] = m + s;
                    spec[1][j] = m - s;
                }
            }
        }
    }
}
