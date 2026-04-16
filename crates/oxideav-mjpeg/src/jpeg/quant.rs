//! Quantisation tables: defaults + DQT parsing.
//!
//! The tables stored here are always in **natural (row-major) order**. JPEG
//! DQT segments encode values in zigzag order — `parse_dqt` unshuffles them
//! on the way in. The encoder shuffles them back on the way out.

use oxideav_core::{Error, Result};

use super::zigzag::ZIGZAG;

/// Annex K.1 standard luma quantisation table (for quality 50).
pub const DEFAULT_LUMA_Q50: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Annex K.2 standard chroma quantisation table (for quality 50).
pub const DEFAULT_CHROMA_Q50: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Scale a base Q=50 table by a libjpeg-style quality factor in `1..=100`.
/// Returns a table in natural order.
pub fn scale_for_quality(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut out = [0u16; 64];
    for i in 0..64 {
        // Keep within 8-bit DQT precision so tables fit in Pq=0.
        let v = (((base[i] as i32) * scale + 50) / 100).clamp(1, 255);
        out[i] = v as u16;
    }
    out
}

/// One DQT table entry. JPEG allows up to 4 tables (ids 0..3), and
/// `precision` is either 8-bit or 16-bit.
#[derive(Clone, Debug)]
pub struct QuantTable {
    /// Natural-order values.
    pub values: [u16; 64],
}

impl QuantTable {
    pub const fn zero() -> Self {
        Self { values: [0; 64] }
    }
}

/// Parse a DQT marker segment payload (everything after the 2-byte length).
///
/// Stores tables by id in `tables`.
pub fn parse_dqt(payload: &[u8], tables: &mut [Option<QuantTable>; 4]) -> Result<()> {
    let mut i = 0;
    while i < payload.len() {
        let pq_tq = payload[i];
        let precision = pq_tq >> 4;
        let tq = (pq_tq & 0x0F) as usize;
        if tq >= 4 {
            return Err(Error::invalid("DQT: table id > 3"));
        }
        i += 1;
        let n_bytes = if precision == 0 { 64 } else { 128 };
        if i + n_bytes > payload.len() {
            return Err(Error::invalid("DQT: truncated table"));
        }
        let mut nat = [0u16; 64];
        if precision == 0 {
            for k in 0..64 {
                nat[ZIGZAG[k]] = payload[i + k] as u16;
            }
        } else {
            for k in 0..64 {
                let hi = payload[i + k * 2] as u16;
                let lo = payload[i + k * 2 + 1] as u16;
                nat[ZIGZAG[k]] = (hi << 8) | lo;
            }
        }
        tables[tq] = Some(QuantTable { values: nat });
        i += n_bytes;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_scaling_bounds() {
        let high = scale_for_quality(&DEFAULT_LUMA_Q50, 100);
        let low = scale_for_quality(&DEFAULT_LUMA_Q50, 1);
        assert!(high.iter().all(|&v| v >= 1));
        assert!(low.iter().all(|&v| (1..=255).contains(&v)));
    }
}
