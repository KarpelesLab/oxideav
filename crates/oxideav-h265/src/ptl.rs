//! `profile_tier_level()` parser (§7.3.3) — shared by VPS and SPS.

use oxideav_core::Result;

use crate::bitreader::BitReader;

/// One profile/tier/level sublayer entry.
#[derive(Clone, Debug, Default)]
pub struct ProfileTierLevel {
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    pub general_profile_idc: u8,
    pub general_profile_compatibility_flag: u32,
    pub general_progressive_source_flag: bool,
    pub general_interlaced_source_flag: bool,
    pub general_non_packed_constraint_flag: bool,
    pub general_frame_only_constraint_flag: bool,
    pub general_level_idc: u8,
    /// Per sublayer (only present up to `max_num_sub_layers_minus1`):
    pub sub_layer_profile_present_flag: Vec<bool>,
    pub sub_layer_level_present_flag: Vec<bool>,
    pub sub_layer_level_idc: Vec<u8>,
}

/// Parse `profile_tier_level(profilePresentFlag, maxNumSubLayersMinus1)`.
/// §7.3.3.
pub fn parse_profile_tier_level(
    br: &mut BitReader<'_>,
    profile_present_flag: bool,
    max_num_sub_layers_minus1: u8,
) -> Result<ProfileTierLevel> {
    let mut ptl = ProfileTierLevel::default();
    if profile_present_flag {
        ptl.general_profile_space = br.u(2)? as u8;
        ptl.general_tier_flag = br.u1()? == 1;
        ptl.general_profile_idc = br.u(5)? as u8;
        ptl.general_profile_compatibility_flag = br.u(32)?;
        ptl.general_progressive_source_flag = br.u1()? == 1;
        ptl.general_interlaced_source_flag = br.u1()? == 1;
        ptl.general_non_packed_constraint_flag = br.u1()? == 1;
        ptl.general_frame_only_constraint_flag = br.u1()? == 1;
        // Skip the 43 + 1 bits of constraint flags (§A.3.x). We don't decode
        // them in v1; we just need to advance the bit pointer.
        br.skip(43)?;
        // general_inbld_flag or reserved_zero_bit
        br.skip(1)?;
    }
    ptl.general_level_idc = br.u(8)? as u8;

    let n = max_num_sub_layers_minus1 as usize;
    if n == 0 {
        return Ok(ptl);
    }
    ptl.sub_layer_profile_present_flag = Vec::with_capacity(n);
    ptl.sub_layer_level_present_flag = Vec::with_capacity(n);
    for _ in 0..n {
        ptl.sub_layer_profile_present_flag.push(br.u1()? == 1);
        ptl.sub_layer_level_present_flag.push(br.u1()? == 1);
    }
    // Reserved 2-bit padding bits, repeated until 8 sublayers are reached.
    if n < 8 {
        for _ in n..8 {
            br.skip(2)?;
        }
    }
    ptl.sub_layer_level_idc = vec![0u8; n];
    for i in 0..n {
        if ptl.sub_layer_profile_present_flag[i] {
            // 2 + 1 + 5 + 32 + 4 + 43 + 1 = 88 bits of sublayer profile data.
            br.skip(2 + 1 + 5 + 32 + 4 + 43 + 1)?;
        }
        if ptl.sub_layer_level_present_flag[i] {
            ptl.sub_layer_level_idc[i] = br.u(8)? as u8;
        }
    }
    Ok(ptl)
}

/// Common HEVC profile name, where known. Returns `None` for unknown idcs.
pub fn profile_name(profile_idc: u8) -> Option<&'static str> {
    match profile_idc {
        1 => Some("Main"),
        2 => Some("Main 10"),
        3 => Some("Main Still Picture"),
        4 => Some("Format Range Extensions"),
        5 => Some("High Throughput"),
        6 => Some("Multiview Main"),
        7 => Some("Scalable Main"),
        8 => Some("3D Main"),
        9 => Some("Screen Content Coding"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_only_is_minimum() {
        // profile_present_flag = false, max_sub_layers_minus1 = 0, just 8 bits of level_idc.
        let data = [0x5Au8];
        let mut br = BitReader::new(&data);
        let ptl = parse_profile_tier_level(&mut br, false, 0).unwrap();
        assert_eq!(ptl.general_level_idc, 0x5A);
    }

    #[test]
    fn profile_name_main() {
        assert_eq!(profile_name(1), Some("Main"));
        assert_eq!(profile_name(99), None);
    }
}
