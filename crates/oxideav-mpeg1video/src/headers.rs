//! Sequence, GOP, picture and slice headers per ISO/IEC 11172-2 §2.4.2.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Decoded sequence header fields.
#[derive(Clone, Debug)]
pub struct SequenceHeader {
    pub horizontal_size: u32,
    pub vertical_size: u32,
    pub aspect_ratio_info: u8,
    pub frame_rate_code: u8,
    pub bit_rate: u32,
    pub vbv_buffer_size: u32,
    pub constrained_parameters_flag: bool,
    pub intra_quantiser: [u8; 64],
    pub non_intra_quantiser: [u8; 64],
}

/// Default intra quant matrix from Table 2-D.15 / Annex A.
pub const DEFAULT_INTRA_QUANT: [u8; 64] = [
    8, 16, 19, 22, 26, 27, 29, 34, 16, 16, 22, 24, 27, 29, 34, 37, 19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40, 22, 26, 27, 29, 32, 35, 40, 48, 26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69, 27, 29, 35, 38, 46, 56, 69, 83,
];

/// Default non-intra quant matrix — all 16s.
pub const DEFAULT_NON_INTRA_QUANT: [u8; 64] = [16; 64];

/// Zig-zag scan order used for DCT coefficient transmission. Position k in
/// the zigzag stream corresponds to natural-order index `ZIGZAG[k]`.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Parse the body of a sequence header (payload after the 0x000001B3 marker).
/// `br` must be positioned immediately after the start code.
pub fn parse_sequence_header(br: &mut BitReader<'_>) -> Result<SequenceHeader> {
    let horizontal_size = br.read_u32(12)?;
    let vertical_size = br.read_u32(12)?;
    let aspect_ratio_info = br.read_u32(4)? as u8;
    let frame_rate_code = br.read_u32(4)? as u8;
    let bit_rate = br.read_u32(18)?;
    let marker = br.read_u32(1)?;
    if marker != 1 {
        return Err(Error::invalid("sequence header: missing marker bit"));
    }
    let vbv_buffer_size = br.read_u32(10)?;
    let constrained_parameters_flag = br.read_u32(1)? == 1;

    let mut intra = DEFAULT_INTRA_QUANT;
    let mut non_intra = DEFAULT_NON_INTRA_QUANT;
    if br.read_u32(1)? == 1 {
        for i in 0..64 {
            intra[ZIGZAG[i]] = br.read_u32(8)? as u8;
        }
    }
    if br.read_u32(1)? == 1 {
        for i in 0..64 {
            non_intra[ZIGZAG[i]] = br.read_u32(8)? as u8;
        }
    }

    Ok(SequenceHeader {
        horizontal_size,
        vertical_size,
        aspect_ratio_info,
        frame_rate_code,
        bit_rate,
        vbv_buffer_size,
        constrained_parameters_flag,
        intra_quantiser: intra,
        non_intra_quantiser: non_intra,
    })
}

/// Decoded GOP header.
#[derive(Clone, Debug)]
pub struct GopHeader {
    pub drop_frame_flag: bool,
    pub time_code_hours: u8,
    pub time_code_minutes: u8,
    pub time_code_seconds: u8,
    pub time_code_pictures: u8,
    pub closed_gop: bool,
    pub broken_link: bool,
}

pub fn parse_gop_header(br: &mut BitReader<'_>) -> Result<GopHeader> {
    let drop_frame_flag = br.read_u32(1)? == 1;
    let h = br.read_u32(5)? as u8;
    let m = br.read_u32(6)? as u8;
    let _marker = br.read_u32(1)?;
    let s = br.read_u32(6)? as u8;
    let p = br.read_u32(6)? as u8;
    let closed_gop = br.read_u32(1)? == 1;
    let broken_link = br.read_u32(1)? == 1;
    Ok(GopHeader {
        drop_frame_flag,
        time_code_hours: h,
        time_code_minutes: m,
        time_code_seconds: s,
        time_code_pictures: p,
        closed_gop,
        broken_link,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureType {
    I = 1,
    P = 2,
    B = 3,
    D = 4,
}

#[derive(Clone, Debug)]
pub struct PictureHeader {
    pub temporal_reference: u16,
    pub picture_type: PictureType,
    pub vbv_delay: u16,
    /// For P and B pictures.
    pub full_pel_forward_vector: bool,
    pub forward_f_code: u8,
    /// For B pictures.
    pub full_pel_backward_vector: bool,
    pub backward_f_code: u8,
}

pub fn parse_picture_header(br: &mut BitReader<'_>) -> Result<PictureHeader> {
    let temporal_reference = br.read_u32(10)? as u16;
    let picture_type = match br.read_u32(3)? {
        1 => PictureType::I,
        2 => PictureType::P,
        3 => PictureType::B,
        4 => PictureType::D,
        other => {
            return Err(Error::invalid(format!(
                "picture_coding_type {other} out of range"
            )))
        }
    };
    let vbv_delay = br.read_u32(16)? as u16;

    let (mut fp_fwd, mut f_fwd, mut fp_bwd, mut f_bwd) = (false, 0u8, false, 0u8);
    if matches!(picture_type, PictureType::P | PictureType::B) {
        fp_fwd = br.read_u32(1)? == 1;
        f_fwd = br.read_u32(3)? as u8;
        if f_fwd == 0 {
            return Err(Error::invalid("forward_f_code = 0"));
        }
    }
    if picture_type == PictureType::B {
        fp_bwd = br.read_u32(1)? == 1;
        f_bwd = br.read_u32(3)? as u8;
        if f_bwd == 0 {
            return Err(Error::invalid("backward_f_code = 0"));
        }
    }

    // Extra information bytes.
    while br.read_u32(1)? == 1 {
        br.read_u32(8)?;
    }

    Ok(PictureHeader {
        temporal_reference,
        picture_type,
        vbv_delay,
        full_pel_forward_vector: fp_fwd,
        forward_f_code: f_fwd,
        full_pel_backward_vector: fp_bwd,
        backward_f_code: f_bwd,
    })
}

/// Map `frame_rate_code` (4 bits) to the standard frame-rate rationals per
/// Table 2-D.4.
pub fn frame_rate_for_code(code: u8) -> Option<(i64, i64)> {
    Some(match code {
        1 => (24000, 1001), // 23.976
        2 => (24, 1),
        3 => (25, 1),
        4 => (30000, 1001), // 29.97
        5 => (30, 1),
        6 => (50, 1),
        7 => (60000, 1001), // 59.94
        8 => (60, 1),
        _ => return None,
    })
}
