//! HEVC Picture Parameter Set parser (§7.3.2.3).
//!
//! v1 scope: every flag the slice-header parser needs to know about plus
//! the tiles / loop-filter geometry. Anything past `pps_extension_flag`
//! is not used in the v1 scaffold.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

#[derive(Clone, Debug)]
pub struct PicParameterSet {
    pub pps_pic_parameter_set_id: u32,
    pub pps_seq_parameter_set_id: u32,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub init_qp_minus26: i32,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub pps_cb_qp_offset: i32,
    pub pps_cr_qp_offset: i32,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u32,
    pub slice_segment_header_extension_present_flag: bool,
}

/// Parse a PPS NAL RBSP payload (the bytes after the 2-byte NAL header,
/// already stripped of emulation-prevention bytes).
pub fn parse_pps(rbsp: &[u8]) -> Result<PicParameterSet> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.ue()?;
    let pps_seq_parameter_set_id = br.ue()?;
    let dependent_slice_segments_enabled_flag = br.u1()? == 1;
    let output_flag_present_flag = br.u1()? == 1;
    let num_extra_slice_header_bits = br.u(3)? as u8;
    let sign_data_hiding_enabled_flag = br.u1()? == 1;
    let cabac_init_present_flag = br.u1()? == 1;
    let num_ref_idx_l0_default_active_minus1 = br.ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.ue()?;
    let init_qp_minus26 = br.se()?;
    let constrained_intra_pred_flag = br.u1()? == 1;
    let transform_skip_enabled_flag = br.u1()? == 1;
    let cu_qp_delta_enabled_flag = br.u1()? == 1;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
        br.ue()?
    } else {
        0
    };
    let pps_cb_qp_offset = br.se()?;
    let pps_cr_qp_offset = br.se()?;
    let pps_slice_chroma_qp_offsets_present_flag = br.u1()? == 1;
    let weighted_pred_flag = br.u1()? == 1;
    let weighted_bipred_flag = br.u1()? == 1;
    let transquant_bypass_enabled_flag = br.u1()? == 1;
    let tiles_enabled_flag = br.u1()? == 1;
    let entropy_coding_sync_enabled_flag = br.u1()? == 1;

    if tiles_enabled_flag {
        let num_tile_columns_minus1 = br.ue()?;
        let num_tile_rows_minus1 = br.ue()?;
        if num_tile_columns_minus1 > 64 || num_tile_rows_minus1 > 64 {
            return Err(Error::invalid(format!(
                "h265 PPS: tile dims out of range ({num_tile_columns_minus1}+1 x {num_tile_rows_minus1}+1)"
            )));
        }
        let uniform_spacing_flag = br.u1()? == 1;
        if !uniform_spacing_flag {
            for _ in 0..num_tile_columns_minus1 {
                let _ = br.ue()?; // column_width_minus1
            }
            for _ in 0..num_tile_rows_minus1 {
                let _ = br.ue()?; // row_height_minus1
            }
        }
        let _loop_filter_across_tiles_enabled_flag = br.u1()? == 1;
    }
    let pps_loop_filter_across_slices_enabled_flag = br.u1()? == 1;

    let deblocking_filter_control_present_flag = br.u1()? == 1;
    let mut deblocking_filter_override_enabled_flag = false;
    let mut pps_deblocking_filter_disabled_flag = false;
    if deblocking_filter_control_present_flag {
        deblocking_filter_override_enabled_flag = br.u1()? == 1;
        pps_deblocking_filter_disabled_flag = br.u1()? == 1;
        if !pps_deblocking_filter_disabled_flag {
            let _ = br.se()?; // pps_beta_offset_div2
            let _ = br.se()?; // pps_tc_offset_div2
        }
    }
    let pps_scaling_list_data_present_flag = br.u1()? == 1;
    if pps_scaling_list_data_present_flag {
        // Same skipping logic as in SPS — kept inline so the two parsers
        // stay independent.
        skip_scaling_list_data(&mut br)?;
    }
    let lists_modification_present_flag = br.u1()? == 1;
    let log2_parallel_merge_level_minus2 = br.ue()?;
    let slice_segment_header_extension_present_flag = br.u1()? == 1;
    // We deliberately stop before pps_extension_flag — the slice header
    // does not depend on it for the v1 scaffold.

    Ok(PicParameterSet {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp_minus26,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
    })
}

fn skip_scaling_list_data(br: &mut BitReader<'_>) -> Result<()> {
    for size_id in 0..4u32 {
        let max_matrix_id = if size_id == 3 { 2 } else { 6 };
        let mut matrix_id = 0u32;
        while matrix_id < max_matrix_id {
            let scaling_list_pred_mode_flag = br.u1()? == 1;
            if !scaling_list_pred_mode_flag {
                let _ = br.ue()?;
            } else {
                let mut next_coef: i32 = 8;
                let coef_num = core::cmp::min(64u32, 1u32 << (4 + (size_id << 1)));
                if size_id > 1 {
                    let dc = br.se()?;
                    next_coef = dc + 8;
                }
                for _ in 0..coef_num {
                    let d = br.se()?;
                    next_coef = (next_coef + d + 256) & 0xFF;
                }
            }
            matrix_id += if size_id == 3 { 3 } else { 1 };
        }
    }
    Ok(())
}
