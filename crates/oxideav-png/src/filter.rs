//! PNG per-row filters + CRC32 (used by chunks).
//!
//! PNG applies a single filter type byte at the start of each decoded row
//! ("filter type byte"), followed by the filtered pixel bytes. The filter
//! operates byte-wise; `bpp` (bytes per pixel, rounded up to at least 1) is
//! the stride used when subtracting a "left" or "upper-left" neighbour.

use oxideav_core::{Error, Result};

/// PNG filter type byte values. See RFC 2083 §6.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterType {
    None = 0,
    Sub = 1,
    Up = 2,
    Average = 3,
    Paeth = 4,
}

impl FilterType {
    pub fn from_u8(b: u8) -> Result<Self> {
        Ok(match b {
            0 => Self::None,
            1 => Self::Sub,
            2 => Self::Up,
            3 => Self::Average,
            4 => Self::Paeth,
            _ => return Err(Error::invalid(format!("PNG: unknown filter type {b}"))),
        })
    }
}

/// Reverse the filter on one row, writing the reconstructed bytes back into
/// `row`. `prev_row` is the previous row's already-reconstructed bytes (may
/// be an all-zero slice for the first row). `bpp` is the byte-distance to
/// the "left" pixel — at least 1, as specified by RFC 2083.
pub fn unfilter_row(filter: FilterType, row: &mut [u8], prev_row: &[u8], bpp: usize) -> Result<()> {
    if prev_row.len() != row.len() {
        return Err(Error::invalid(
            "PNG unfilter: prev_row length must match row length",
        ));
    }
    match filter {
        FilterType::None => {}
        FilterType::Sub => {
            for i in bpp..row.len() {
                row[i] = row[i].wrapping_add(row[i - bpp]);
            }
        }
        FilterType::Up => {
            for i in 0..row.len() {
                row[i] = row[i].wrapping_add(prev_row[i]);
            }
        }
        FilterType::Average => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
                let up = prev_row[i] as u16;
                let avg = ((left + up) / 2) as u8;
                row[i] = row[i].wrapping_add(avg);
            }
        }
        FilterType::Paeth => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] as i16 } else { 0 };
                let up = prev_row[i] as i16;
                let up_left = if i >= bpp {
                    prev_row[i - bpp] as i16
                } else {
                    0
                };
                let p = paeth_predictor(left, up, up_left) as u8;
                row[i] = row[i].wrapping_add(p);
            }
        }
    }
    Ok(())
}

/// Filter one row. `row` holds the raw pixel bytes; output is written to
/// `out` (must be same length as `row`). `prev_row` is the previous row's
/// *raw* bytes (zeros for first row — per the spec).
pub fn filter_row(filter: FilterType, row: &[u8], prev_row: &[u8], bpp: usize, out: &mut [u8]) {
    debug_assert_eq!(row.len(), out.len());
    debug_assert_eq!(row.len(), prev_row.len());
    match filter {
        FilterType::None => {
            out.copy_from_slice(row);
        }
        FilterType::Sub => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] } else { 0 };
                out[i] = row[i].wrapping_sub(left);
            }
        }
        FilterType::Up => {
            for i in 0..row.len() {
                out[i] = row[i].wrapping_sub(prev_row[i]);
            }
        }
        FilterType::Average => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
                let up = prev_row[i] as u16;
                let avg = ((left + up) / 2) as u8;
                out[i] = row[i].wrapping_sub(avg);
            }
        }
        FilterType::Paeth => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] as i16 } else { 0 };
                let up = prev_row[i] as i16;
                let up_left = if i >= bpp {
                    prev_row[i - bpp] as i16
                } else {
                    0
                };
                let p = paeth_predictor(left, up, up_left) as u8;
                out[i] = row[i].wrapping_sub(p);
            }
        }
    }
}

fn paeth_predictor(a: i16, b: i16, c: i16) -> i16 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Pick a filter for `row` using the sum-of-absolute-deltas heuristic from
/// the PNG specification (§12.8). Tries all five filters and picks the one
/// whose filtered bytes have the lowest absolute sum (treating bytes as
/// signed i8).
pub fn choose_filter_heuristic(
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    scratch: &mut [u8],
) -> FilterType {
    let mut best = FilterType::None;
    let mut best_sum = u64::MAX;
    for f in [
        FilterType::None,
        FilterType::Sub,
        FilterType::Up,
        FilterType::Average,
        FilterType::Paeth,
    ] {
        filter_row(f, row, prev_row, bpp, scratch);
        // Sum of absolute signed bytes.
        let mut sum: u64 = 0;
        for &b in scratch.iter() {
            let s = b as i8;
            sum = sum.saturating_add(s.unsigned_abs() as u64);
        }
        if sum < best_sum {
            best_sum = sum;
            best = f;
        }
    }
    best
}

// --- CRC32 ---------------------------------------------------------------

use std::sync::OnceLock;

static CRC_TABLE: OnceLock<[u32; 256]> = OnceLock::new();

fn crc_table() -> &'static [u32; 256] {
    CRC_TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    c = 0xEDB8_8320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            t[n as usize] = c;
        }
        t
    })
}

/// PNG CRC32 (IEEE 802.3 polynomial, start with 0xFFFFFFFF, invert result).
pub fn crc32(bytes: &[u8]) -> u32 {
    let tbl = crc_table();
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        c = tbl[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Same but with `once_cell` avoided — pure-loop crc32 for tiny use cases.
pub fn crc32_loop(bytes: &[u8]) -> u32 {
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            let mask = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    c ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_table_matches_loop() {
        let a = crc32(b"IEND");
        let b = crc32_loop(b"IEND");
        assert_eq!(a, b);
        // Known value: CRC32 of "IEND" chunk type (empty payload) — well-known.
        assert_eq!(a, 0xAE42_6082);
    }

    #[test]
    fn filter_roundtrip_all_types() {
        let row = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let prev = [5u8; 8];
        let bpp = 1;
        for f in [
            FilterType::None,
            FilterType::Sub,
            FilterType::Up,
            FilterType::Average,
            FilterType::Paeth,
        ] {
            let mut filtered = [0u8; 8];
            filter_row(f, &row, &prev, bpp, &mut filtered);
            let mut back = filtered;
            unfilter_row(f, &mut back, &prev, bpp).unwrap();
            assert_eq!(back, row, "filter {f:?} roundtrip mismatch");
        }
    }
}
