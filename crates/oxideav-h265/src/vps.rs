//! HEVC Video Parameter Set parser (§7.3.2.1).
//!
//! Only the fields we currently inspect / propagate are retained; the rest
//! of the syntax is consumed by the bit reader so the parser keeps a
//! consistent position.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::ptl::{parse_profile_tier_level, ProfileTierLevel};

#[derive(Clone, Debug)]
pub struct VideoParameterSet {
    pub vps_video_parameter_set_id: u8,
    pub vps_base_layer_internal_flag: bool,
    pub vps_base_layer_available_flag: bool,
    pub vps_max_layers_minus1: u8,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: bool,
    pub profile_tier_level: ProfileTierLevel,
    pub vps_sub_layer_ordering_info_present_flag: bool,
    /// Per sublayer (length `vps_max_sub_layers_minus1 + 1`).
    pub vps_max_dec_pic_buffering_minus1: Vec<u32>,
    pub vps_max_num_reorder_pics: Vec<u32>,
    pub vps_max_latency_increase_plus1: Vec<u32>,
    pub vps_max_layer_id: u8,
    pub vps_num_layer_sets_minus1: u32,
}

/// Parse a VPS NAL RBSP payload (i.e. the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
pub fn parse_vps(rbsp: &[u8]) -> Result<VideoParameterSet> {
    let mut br = BitReader::new(rbsp);
    let vps_video_parameter_set_id = br.u(4)? as u8;
    let vps_base_layer_internal_flag = br.u1()? == 1;
    let vps_base_layer_available_flag = br.u1()? == 1;
    let vps_max_layers_minus1 = br.u(6)? as u8;
    let vps_max_sub_layers_minus1 = br.u(3)? as u8;
    if vps_max_sub_layers_minus1 > 6 {
        return Err(Error::invalid(
            "h265 VPS: vps_max_sub_layers_minus1 must be <= 6",
        ));
    }
    let vps_temporal_id_nesting_flag = br.u1()? == 1;
    // vps_reserved_0xffff_16bits
    let reserved = br.u(16)?;
    if reserved != 0xFFFF {
        return Err(Error::invalid(format!(
            "h265 VPS: reserved 16 bits != 0xFFFF (got 0x{reserved:04X})"
        )));
    }
    let profile_tier_level = parse_profile_tier_level(&mut br, true, vps_max_sub_layers_minus1)?;

    let vps_sub_layer_ordering_info_present_flag = br.u1()? == 1;
    let n = (vps_max_sub_layers_minus1 + 1) as usize;
    let lo = if vps_sub_layer_ordering_info_present_flag {
        0
    } else {
        vps_max_sub_layers_minus1 as usize
    };
    let mut vps_max_dec_pic_buffering_minus1 = vec![0u32; n];
    let mut vps_max_num_reorder_pics = vec![0u32; n];
    let mut vps_max_latency_increase_plus1 = vec![0u32; n];
    for i in lo..n {
        vps_max_dec_pic_buffering_minus1[i] = br.ue()?;
        vps_max_num_reorder_pics[i] = br.ue()?;
        vps_max_latency_increase_plus1[i] = br.ue()?;
    }

    let vps_max_layer_id = br.u(6)? as u8;
    let vps_num_layer_sets_minus1 = br.ue()?;
    if vps_num_layer_sets_minus1 > 1023 {
        return Err(Error::invalid("h265 VPS: vps_num_layer_sets_minus1 > 1023"));
    }
    // We deliberately stop after num_layer_sets — the rest of the VPS
    // (layer_id_included flags, HRD parameters, vps_extension) is not
    // required by the v1 scaffold.

    Ok(VideoParameterSet {
        vps_video_parameter_set_id,
        vps_base_layer_internal_flag,
        vps_base_layer_available_flag,
        vps_max_layers_minus1,
        vps_max_sub_layers_minus1,
        vps_temporal_id_nesting_flag,
        profile_tier_level,
        vps_sub_layer_ordering_info_present_flag,
        vps_max_dec_pic_buffering_minus1,
        vps_max_num_reorder_pics,
        vps_max_latency_increase_plus1,
        vps_max_layer_id,
        vps_num_layer_sets_minus1,
    })
}
