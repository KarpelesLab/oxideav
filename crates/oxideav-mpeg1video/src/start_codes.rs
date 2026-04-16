//! Start-code constants and scanning helpers for MPEG-1 video.
//!
//! ISO/IEC 11172-2 start codes are byte-aligned sequences of `0x000001XX`,
//! optionally preceded by any number of 0x00 stuffing bytes. The last byte
//! identifies the layer / marker.

pub const PICTURE_START_CODE: u8 = 0x00;
pub const SLICE_START_MIN: u8 = 0x01;
pub const SLICE_START_MAX: u8 = 0xAF;
pub const USER_DATA_START_CODE: u8 = 0xB2;
pub const SEQUENCE_HEADER_CODE: u8 = 0xB3;
pub const SEQUENCE_ERROR_CODE: u8 = 0xB4;
pub const EXTENSION_START_CODE: u8 = 0xB5;
pub const SEQUENCE_END_CODE: u8 = 0xB7;
pub const GROUP_START_CODE: u8 = 0xB8;

pub fn is_slice(code: u8) -> bool {
    (SLICE_START_MIN..=SLICE_START_MAX).contains(&code)
}

/// Scan forward from `pos` looking for the next `0x00 0x00 0x01 XX` marker.
/// Returns `(position_of_first_zero, marker_byte)`. The reader can then skip
/// `pos + 4` to move past the start code.
pub fn find_next_start_code(data: &[u8], mut pos: usize) -> Option<(usize, u8)> {
    while pos + 4 <= data.len() {
        // Scan for `0x00 0x00`.
        if data[pos] == 0 {
            // Any number of 0x00 bytes, then 0x01 XX.
            let mut p = pos;
            while p < data.len() && data[p] == 0 {
                p += 1;
            }
            // Need at least two zeros before the 0x01.
            if p - pos >= 2 && p < data.len() && data[p] == 0x01 && p + 1 < data.len() {
                return Some((p - 2, data[p + 1]));
            }
            // Advance past the zeros we scanned.
            pos = p.max(pos + 1);
            continue;
        }
        pos += 1;
    }
    None
}

/// Iterator yielding `(position, marker)` pairs for every start code in `data`.
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
    fn finds_start_codes() {
        // 0x000001b3 at pos 0, 0x000001b8 at pos 8.
        let data = [
            0x00, 0x00, 0x01, 0xB3, 0x12, 0x34, 0x56, 0x78, 0x00, 0x00, 0x01, 0xB8,
        ];
        let mut it = iter_start_codes(&data);
        assert_eq!(it.next(), Some((0, 0xB3)));
        assert_eq!(it.next(), Some((8, 0xB8)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn handles_extra_zero_padding() {
        // 0x00000001b3 — three zeros before 0x01 is legal.
        let data = [0x00, 0x00, 0x00, 0x01, 0xB3];
        let mut it = iter_start_codes(&data);
        assert_eq!(it.next(), Some((1, 0xB3)));
    }
}
