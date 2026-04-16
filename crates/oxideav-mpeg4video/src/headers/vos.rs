//! Visual Object Sequence + Visual Object headers (ISO/IEC 14496-2 §6.2.2).

use oxideav_core::Result;

use crate::bitreader::BitReader;

/// Parsed Visual Object Sequence header. The VOS payload is a single byte that
/// carries `profile_and_level_indication` — a numeric id covering both profile
/// and level per Table G-1 of the spec.
#[derive(Clone, Debug)]
pub struct VisualObjectSequence {
    pub profile_and_level_indication: u8,
}

/// Parse the payload that follows the VOS start code (0x000001B0). Reader
/// must be positioned just after the 4-byte marker.
pub fn parse_vos(br: &mut BitReader<'_>) -> Result<VisualObjectSequence> {
    let profile_and_level_indication = br.read_u32(8)? as u8;
    Ok(VisualObjectSequence {
        profile_and_level_indication,
    })
}

/// Type of the `visual_object_type` field in the VO header (Table 6-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoType {
    Video,
    Still,
    Mesh,
    Face,
    ThreeDMesh,
    Other(u8),
}

impl VoType {
    fn from(code: u8) -> Self {
        match code {
            1 => VoType::Video,
            2 => VoType::Still,
            3 => VoType::Mesh,
            4 => VoType::Face,
            5 => VoType::ThreeDMesh,
            other => VoType::Other(other),
        }
    }
}

/// Parsed Visual Object header.
#[derive(Clone, Debug)]
pub struct VisualObject {
    pub is_identifier_present: bool,
    pub verid: u8,
    pub priority: u8,
    pub visual_object_type: VoType,
    /// Video signal parameters (optional, only for video VOs).
    pub video_signal_present: bool,
    pub video_format: u8,
    pub video_range: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Parse the payload that follows a visual_object_start_code (0x000001B5).
pub fn parse_visual_object(br: &mut BitReader<'_>) -> Result<VisualObject> {
    let is_identifier_present = br.read_u1()? == 1;
    let mut verid = 1;
    let mut priority = 0;
    if is_identifier_present {
        verid = br.read_u32(4)? as u8;
        priority = br.read_u32(3)? as u8;
    }
    let vo_type_code = br.read_u32(4)? as u8;
    let visual_object_type = VoType::from(vo_type_code);

    // video_signal_type (optional, 1 bit flag, per §6.2.2.2 — only for Video/Still)
    let mut video_signal_present = false;
    let mut video_format = 5; // "unspecified"
    let mut video_range = false;
    let mut colour_primaries = 1;
    let mut transfer_characteristics = 1;
    let mut matrix_coefficients = 1;

    if matches!(visual_object_type, VoType::Video | VoType::Still) {
        let has_video_signal = br.read_u1()? == 1;
        if has_video_signal {
            video_signal_present = true;
            video_format = br.read_u32(3)? as u8;
            video_range = br.read_u1()? == 1;
            let colour_description = br.read_u1()? == 1;
            if colour_description {
                colour_primaries = br.read_u32(8)? as u8;
                transfer_characteristics = br.read_u32(8)? as u8;
                matrix_coefficients = br.read_u32(8)? as u8;
            }
        }
    }

    // Spec requires next_start_code() alignment afterwards; consumers drive
    // their own start-code scanning so we stop here without aligning to avoid
    // reading past the payload slice.

    let _ = priority; // informational only
    let _ = verid;

    Ok(VisualObject {
        is_identifier_present,
        verid,
        priority,
        visual_object_type,
        video_signal_present,
        video_format,
        video_range,
        colour_primaries,
        transfer_characteristics,
        matrix_coefficients,
    })
}

/// Classify `profile_and_level_indication` into a human-readable label for
/// diagnostic messages. Table G-1.
pub fn profile_level_description(pli: u8) -> &'static str {
    match pli {
        0x01..=0x03 => "Simple Profile",
        0x10..=0x12 => "Simple Scalable Profile",
        0x21..=0x25 => "Core Profile",
        0x32..=0x34 => "Main Profile",
        0xF0..=0xF5 => "Advanced Simple Profile",
        0xF7..=0xFA => "Advanced Real Time Simple Profile",
        0xFB..=0xFF => "Core Scalable Profile",
        _ => "unknown profile",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_vos() {
        // One-byte payload: profile_and_level = 0xF3 (ASP @ L3).
        let data = [0xF3];
        let mut br = BitReader::new(&data);
        let v = parse_vos(&mut br).unwrap();
        assert_eq!(v.profile_and_level_indication, 0xF3);
    }

    #[test]
    fn classifies_profiles() {
        assert_eq!(profile_level_description(0x01), "Simple Profile");
        assert_eq!(profile_level_description(0xF3), "Advanced Simple Profile");
    }
}
