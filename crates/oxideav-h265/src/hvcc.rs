//! HEVCDecoderConfigurationRecord parser (ISO/IEC 14496-15 §8.3.3).
//!
//! `hvcC` carries:
//! * a 22-byte fixed prefix (configurationVersion, profile/tier/level
//!   summary, constraint flags, chromaFormat, bit depths, etc.);
//! * `lengthSizeMinusOne` (lower 2 bits of byte 21) — bytes per length
//!   prefix in sample data;
//! * an array of arrays of NAL units, conventionally one each for VPS, SPS,
//!   PPS (and optionally prefix SEI).

use oxideav_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct HvcConfig {
    pub configuration_version: u8,
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    pub general_profile_idc: u8,
    pub general_profile_compatibility_flags: u32,
    pub general_constraint_indicator_flags: u64,
    pub general_level_idc: u8,
    pub min_spatial_segmentation_idc: u16,
    pub parallelism_type: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub avg_frame_rate: u16,
    pub constant_frame_rate: u8,
    pub num_temporal_layers: u8,
    pub temporal_id_nested: bool,
    pub length_size_minus_one: u8,
    /// Each entry: (nal_unit_type, list of NAL bytes). NAL bytes include
    /// the 2-byte NAL header and are still emulation-prevented.
    pub nal_arrays: Vec<HvcNalArray>,
}

#[derive(Clone, Debug)]
pub struct HvcNalArray {
    pub array_completeness: bool,
    pub nal_unit_type: u8,
    pub nals: Vec<Vec<u8>>,
}

/// Parse a raw `hvcC` body (the contents of the box, excluding the 8-byte
/// box header).
pub fn parse_hvcc(body: &[u8]) -> Result<HvcConfig> {
    if body.len() < 23 {
        return Err(Error::invalid(format!(
            "h265 hvcC: body too short ({} < 23)",
            body.len()
        )));
    }
    let configuration_version = body[0];
    if configuration_version != 1 {
        return Err(Error::invalid(format!(
            "h265 hvcC: unsupported configurationVersion {configuration_version}"
        )));
    }
    let b1 = body[1];
    let general_profile_space = (b1 >> 6) & 0x3;
    let general_tier_flag = (b1 & 0x20) != 0;
    let general_profile_idc = b1 & 0x1F;
    let general_profile_compatibility_flags =
        u32::from_be_bytes([body[2], body[3], body[4], body[5]]);
    // 6 bytes of constraint indicator
    let mut gci = [0u8; 8];
    gci[2..8].copy_from_slice(&body[6..12]);
    let general_constraint_indicator_flags = u64::from_be_bytes(gci);
    let general_level_idc = body[12];
    // 4 reserved bits + min_spatial_segmentation_idc(12)
    let min_spatial_segmentation_idc = u16::from_be_bytes([body[13], body[14]]) & 0x0FFF;
    // 6 reserved + parallelismType(2)
    let parallelism_type = body[15] & 0x03;
    // 6 reserved + chromaFormat(2)
    let chroma_format_idc = body[16] & 0x03;
    // 5 reserved + bitDepthLumaMinus8(3)
    let bit_depth_luma_minus8 = body[17] & 0x07;
    // 5 reserved + bitDepthChromaMinus8(3)
    let bit_depth_chroma_minus8 = body[18] & 0x07;
    let avg_frame_rate = u16::from_be_bytes([body[19], body[20]]);
    let b21 = body[21];
    let constant_frame_rate = (b21 >> 6) & 0x03;
    let num_temporal_layers = (b21 >> 3) & 0x07;
    let temporal_id_nested = (b21 & 0x04) != 0;
    let length_size_minus_one = b21 & 0x03;
    let num_of_arrays = body[22] as usize;

    let mut pos = 23;
    let mut nal_arrays = Vec::with_capacity(num_of_arrays);
    for _ in 0..num_of_arrays {
        if pos + 3 > body.len() {
            return Err(Error::invalid("h265 hvcC: NAL array header truncated"));
        }
        let hdr = body[pos];
        let array_completeness = (hdr & 0x80) != 0;
        let nal_unit_type = hdr & 0x3F;
        let num_nalus = u16::from_be_bytes([body[pos + 1], body[pos + 2]]) as usize;
        pos += 3;
        let mut nals = Vec::with_capacity(num_nalus);
        for _ in 0..num_nalus {
            if pos + 2 > body.len() {
                return Err(Error::invalid("h265 hvcC: NAL length truncated"));
            }
            let nal_len = u16::from_be_bytes([body[pos], body[pos + 1]]) as usize;
            pos += 2;
            if pos + nal_len > body.len() {
                return Err(Error::invalid("h265 hvcC: NAL body truncated"));
            }
            nals.push(body[pos..pos + nal_len].to_vec());
            pos += nal_len;
        }
        nal_arrays.push(HvcNalArray {
            array_completeness,
            nal_unit_type,
            nals,
        });
    }

    Ok(HvcConfig {
        configuration_version,
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags,
        general_constraint_indicator_flags,
        general_level_idc,
        min_spatial_segmentation_idc,
        parallelism_type,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        avg_frame_rate,
        constant_frame_rate,
        num_temporal_layers,
        temporal_id_nested,
        length_size_minus_one,
        nal_arrays,
    })
}
