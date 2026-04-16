//! Video Object Plane header (ISO/IEC 14496-2 §6.2.5).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::headers::vol::{ShapeType, VideoObjectLayer};

/// `vop_coding_type` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VopCodingType {
    /// I-VOP — intra-only.
    I,
    /// P-VOP — predicted from the previous reference.
    P,
    /// B-VOP — bidirectional.
    B,
    /// S-VOP — sprite VOP.
    S,
}

/// Parsed VOP header fields relevant to this decoder.
#[derive(Clone, Debug)]
pub struct VideoObjectPlane {
    pub vop_coding_type: VopCodingType,
    pub modulo_time_base: u32,
    pub vop_time_increment: u32,
    pub vop_coded: bool,
    pub rounding_type: bool,
    pub intra_dc_vlc_thr: u8,
    pub vop_quant: u32,
    pub vop_fcode_forward: u8,
    pub vop_fcode_backward: u8,
    /// Rectangle of the VOP — currently we only decode full frames. If the
    /// bitstream uses non-zero horizontal/vertical spatial reference the
    /// caller still receives the values here.
    pub width: u32,
    pub height: u32,
}

/// Parse the VOP header that follows a 0x000001B6 start code.
///
/// `vol` provides the per-sequence context needed to decode variable-length
/// fields (`vop_time_increment_bits`, shape type, etc.).
pub fn parse_vop(br: &mut BitReader<'_>, vol: &VideoObjectLayer) -> Result<VideoObjectPlane> {
    let vop_coding_type = match br.read_u32(2)? {
        0 => VopCodingType::I,
        1 => VopCodingType::P,
        2 => VopCodingType::B,
        3 => VopCodingType::S,
        _ => unreachable!(),
    };

    // modulo_time_base — 1-bits terminated by a 0-bit.
    let mut modulo_time_base = 0u32;
    loop {
        let b = br.read_u1()?;
        if b == 0 {
            break;
        }
        modulo_time_base += 1;
        if modulo_time_base > 60 {
            return Err(Error::invalid("mpeg4 VOP: modulo_time_base runaway"));
        }
    }

    br.read_marker()?;
    let vop_time_increment = br.read_u32(vol.vop_time_increment_bits)?;
    br.read_marker()?;

    let vop_coded = br.read_u1()? == 1;
    if !vop_coded {
        return Ok(VideoObjectPlane {
            vop_coding_type,
            modulo_time_base,
            vop_time_increment,
            vop_coded,
            rounding_type: false,
            intra_dc_vlc_thr: 0,
            vop_quant: 0,
            vop_fcode_forward: 0,
            vop_fcode_backward: 0,
            width: vol.width,
            height: vol.height,
        });
    }

    // Only rectangular shape reaches this point (checked in VOL parser).
    debug_assert_eq!(vol.shape, ShapeType::Rectangular);

    // vop_rounding_type — present only for P-VOPs (§6.2.5).
    let rounding_type = if vop_coding_type == VopCodingType::P {
        br.read_u1()? == 1
    } else {
        false
    };

    // reduced_resolution_vop flag (rarely used; skip if enabled).
    if vol.reduced_resolution_vop_enable
        && matches!(vop_coding_type, VopCodingType::I | VopCodingType::P)
    {
        let _reduced = br.read_u1()?;
    }

    // intra_dc_vlc_thr — 3 bits.
    let intra_dc_vlc_thr = br.read_u32(3)? as u8;

    // vop_quant — quant_precision bits (default 5).
    let vop_quant = br.read_u32(vol.quant_precision as u32)?;
    if vop_quant == 0 {
        return Err(Error::invalid("mpeg4 VOP: vop_quant = 0"));
    }

    let mut fcode_fwd = 0u8;
    let mut fcode_bwd = 0u8;
    if matches!(vop_coding_type, VopCodingType::P | VopCodingType::S) {
        fcode_fwd = br.read_u32(3)? as u8;
    }
    if vop_coding_type == VopCodingType::B {
        fcode_fwd = br.read_u32(3)? as u8;
        fcode_bwd = br.read_u32(3)? as u8;
    }

    Ok(VideoObjectPlane {
        vop_coding_type,
        modulo_time_base,
        vop_time_increment,
        vop_coded,
        rounding_type,
        intra_dc_vlc_thr,
        vop_quant,
        vop_fcode_forward: fcode_fwd,
        vop_fcode_backward: fcode_bwd,
        width: vol.width,
        height: vol.height,
    })
}
