//! Start-code constants and scanning helpers for MPEG-4 Part 2.
//!
//! ISO/IEC 14496-2 §6.3.1. Start codes are byte-aligned `0x000001XX`
//! markers, optionally preceded by any number of `0x00` stuffing bytes.
//! The terminal byte identifies the layer / function.

/// Video Object start codes: `0x00`..=`0x1F`.
pub const VIDEO_OBJECT_START_MIN: u8 = 0x00;
pub const VIDEO_OBJECT_START_MAX: u8 = 0x1F;

/// Video Object Layer start codes: `0x20`..=`0x2F`.
pub const VOL_START_MIN: u8 = 0x20;
pub const VOL_START_MAX: u8 = 0x2F;

/// Visual Object Sequence (VOS) start / end.
pub const VOS_START_CODE: u8 = 0xB0;
pub const VOS_END_CODE: u8 = 0xB1;

/// User-data start code.
pub const USER_DATA_START_CODE: u8 = 0xB2;

/// Group-of-VOP start code.
pub const GOV_START_CODE: u8 = 0xB3;

/// Video Session Error code.
pub const VIDEO_SESSION_ERROR_CODE: u8 = 0xB4;

/// Visual Object start code.
pub const VISUAL_OBJECT_START_CODE: u8 = 0xB5;

/// Video Object Plane (VOP) start code — one per frame.
pub const VOP_START_CODE: u8 = 0xB6;

pub fn is_video_object(code: u8) -> bool {
    (VIDEO_OBJECT_START_MIN..=VIDEO_OBJECT_START_MAX).contains(&code)
}

pub fn is_video_object_layer(code: u8) -> bool {
    (VOL_START_MIN..=VOL_START_MAX).contains(&code)
}

/// Scan forward from `pos` for the next `0x00 0x00 0x01 XX` start code.
/// Returns `(position_of_first_zero, marker_byte)`.
pub fn find_next_start_code(data: &[u8], mut pos: usize) -> Option<(usize, u8)> {
    while pos + 4 <= data.len() {
        if data[pos] == 0 {
            let mut p = pos;
            while p < data.len() && data[p] == 0 {
                p += 1;
            }
            if p - pos >= 2 && p < data.len() && data[p] == 0x01 && p + 1 < data.len() {
                return Some((p - 2, data[p + 1]));
            }
            pos = p.max(pos + 1);
            continue;
        }
        pos += 1;
    }
    None
}

/// Iterator over all `(position, marker)` start codes in `data`.
pub fn iter_start_codes(data: &[u8]) -> impl Iterator<Item = (usize, u8)> + '_ {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let (start, code) = find_next_start_code(data, pos)?;
        pos = start + 4;
        Some((start, code))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_typical_sequence() {
        // VOS, VO, VOL, VOP markers.
        let data = [
            0x00, 0x00, 0x01, 0xB0, 0x01, // VOS
            0x00, 0x00, 0x01, 0xB5, 0x89, // VO descriptor
            0x00, 0x00, 0x01, 0x00, // video_object
            0x00, 0x00, 0x01, 0x20, // VOL
            0x00, 0x00, 0x01, 0xB6, // VOP
        ];
        let codes: Vec<_> = iter_start_codes(&data).map(|(_, c)| c).collect();
        assert_eq!(codes, vec![0xB0, 0xB5, 0x00, 0x20, 0xB6]);
    }

    #[test]
    fn classifies_layers() {
        assert!(is_video_object(0x00));
        assert!(is_video_object(0x1F));
        assert!(!is_video_object(0x20));
        assert!(is_video_object_layer(0x20));
        assert!(is_video_object_layer(0x2F));
        assert!(!is_video_object_layer(0x30));
    }
}
