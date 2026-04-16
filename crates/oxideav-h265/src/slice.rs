//! HEVC slice segment header parser (§7.3.6).
//!
//! Only the leading "I always need this to identify the slice" portion is
//! exposed in v1. We parse enough to:
//!
//! * Tell whether this is the first slice in the picture and where in the
//!   CTB grid it starts.
//! * Identify the active PPS (and via it, the active SPS).
//! * Read the slice type and the picture-order-count LSB.
//!
//! Reference picture list modifications, weighted prediction tables, and
//! the post-header `slice_data()` are deferred — once we plumb CABAC, those
//! sit naturally on top.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::{NalHeader, NalUnitType};
use crate::pps::PicParameterSet;
use crate::sps::SeqParameterSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    B,
    P,
    I,
}

impl SliceType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            _ => Err(Error::invalid(format!(
                "h265 slice: invalid slice_type {v}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SliceSegmentHeader {
    pub first_slice_segment_in_pic_flag: bool,
    pub no_output_of_prior_pics_flag: bool,
    pub slice_pic_parameter_set_id: u32,
    pub dependent_slice_segment_flag: bool,
    pub slice_segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    pub colour_plane_id: u8,
    pub slice_pic_order_cnt_lsb: u32,
    pub short_term_ref_pic_set_sps_flag: bool,
    pub short_term_ref_pic_set_idx: u32,
}

/// Parse a slice segment header given the active PPS+SPS and the NAL header
/// that introduced this slice. The bitstream `rbsp` is the bytes after the
/// 2-byte NAL header, with emulation-prevention removed.
pub fn parse_slice_segment_header(
    rbsp: &[u8],
    nal: &NalHeader,
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<SliceSegmentHeader> {
    let mut br = BitReader::new(rbsp);
    let first_slice_segment_in_pic_flag = br.u1()? == 1;

    let mut no_output_of_prior_pics_flag = false;
    if nal.nal_unit_type.is_irap() {
        no_output_of_prior_pics_flag = br.u1()? == 1;
    }
    let slice_pic_parameter_set_id = br.ue()?;
    if slice_pic_parameter_set_id != pps.pps_pic_parameter_set_id {
        return Err(Error::invalid(format!(
            "h265 slice: PPS id mismatch (slice {slice_pic_parameter_set_id} vs active {})",
            pps.pps_pic_parameter_set_id
        )));
    }

    let mut dependent_slice_segment_flag = false;
    let mut slice_segment_address: u32 = 0;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            dependent_slice_segment_flag = br.u1()? == 1;
        }
        let num_ctbs = num_ctbs_in_pic(sps);
        let n = ceil_log2(num_ctbs);
        slice_segment_address = br.u(n)?;
    }

    let mut slice_type = SliceType::I;
    let mut pic_output_flag = true;
    let mut colour_plane_id = 0u8;
    let mut slice_pic_order_cnt_lsb = 0u32;
    let mut short_term_ref_pic_set_sps_flag = false;
    let mut short_term_ref_pic_set_idx = 0u32;

    if !dependent_slice_segment_flag {
        // Skip num_extra_slice_header_bits flags.
        for _ in 0..pps.num_extra_slice_header_bits {
            br.skip(1)?;
        }
        let st = br.ue()?;
        slice_type = SliceType::from_u32(st)?;
        if pps.output_flag_present_flag {
            pic_output_flag = br.u1()? == 1;
        }
        if sps.separate_colour_plane_flag {
            colour_plane_id = br.u(2)? as u8;
        }
        if !matches!(
            nal.nal_unit_type,
            NalUnitType::IdrWRadl | NalUnitType::IdrNLp
        ) {
            let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            slice_pic_order_cnt_lsb = br.u(lsb_bits)?;
            short_term_ref_pic_set_sps_flag = br.u1()? == 1;
            if short_term_ref_pic_set_sps_flag && sps.num_short_term_ref_pic_sets > 1 {
                let n = ceil_log2(sps.num_short_term_ref_pic_sets);
                short_term_ref_pic_set_idx = br.u(n)?;
            }
            // Inline-RPS (st_ref_pic_set in the slice header) and long-term
            // refs aren't needed for the v1 scaffold acceptance bar — we
            // stop here. The decode-side `Unsupported` error will fire long
            // before any of the deferred fields are consulted.
        }
    }
    Ok(SliceSegmentHeader {
        first_slice_segment_in_pic_flag,
        no_output_of_prior_pics_flag,
        slice_pic_parameter_set_id,
        dependent_slice_segment_flag,
        slice_segment_address,
        slice_type,
        pic_output_flag,
        colour_plane_id,
        slice_pic_order_cnt_lsb,
        short_term_ref_pic_set_sps_flag,
        short_term_ref_pic_set_idx,
    })
}

/// PicWidthInCtbsY * PicHeightInCtbsY (§7.4.7.1).
pub fn num_ctbs_in_pic(sps: &SeqParameterSet) -> u32 {
    let ctb = sps.ctb_size().max(1);
    let w = sps.pic_width_in_luma_samples.div_ceil(ctb);
    let h = sps.pic_height_in_luma_samples.div_ceil(ctb);
    w * h
}

/// Bits required to address `n` distinct values: `ceil(log2(n))`. Returns 0
/// for n ≤ 1.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    32 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_matches() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }
}
