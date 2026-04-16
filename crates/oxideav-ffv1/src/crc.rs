//! CRC-32 IEEE (polynomial 0x04C11DB7) used by FFV1 for the configuration
//! record and optional slice-footer parity field.
//!
//! This is the standard MSB-first (non-reflected) CRC with initial value 0
//! and no post-XOR. FFV1 stores the 4-byte CRC in big-endian byte order
//! immediately after the payload, chosen such that the CRC of `payload ||
//! stored_crc` equals zero.

/// Pre-built CRC-32 lookup table (MSB-first polynomial 0x04C11DB7).
static TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let poly: u32 = 0x04C1_1DB7;
    let mut i: u32 = 0;
    while i < 256 {
        let mut c: u32 = i << 24;
        let mut k = 0;
        while k < 8 {
            c = if (c & 0x8000_0000) != 0 {
                (c << 1) ^ poly
            } else {
                c << 1
            };
            k += 1;
        }
        table[i as usize] = c;
        i += 1;
    }
    table
}

/// Compute the CRC-32 (IEEE, polynomial 0x04C11DB7, MSB-first, init=0) over
/// `data`.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &b in data {
        let idx = ((crc >> 24) as u8) ^ b;
        crc = (crc << 8) ^ TABLE[idx as usize];
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(crc32_ieee(&[]), 0);
    }

    #[test]
    fn known_vector_a() {
        // Non-reflected CRC-32 / polynomial 0x04C11DB7 / init=0 / no
        // post-XOR over "123456789" equals 0x89A1897F (per online
        // CRC-32/MPEG-2-with-init-0 calculators).
        assert_eq!(crc32_ieee(b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn append_crc_makes_total_zero() {
        // The property we rely on for the config record and slice footers:
        // if we append the CRC in big-endian, the CRC of the whole buffer
        // is zero.
        let payload = b"hello FFV1 world";
        let crc = crc32_ieee(payload);
        let mut full = payload.to_vec();
        full.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(crc32_ieee(&full), 0);
    }

    #[test]
    fn ffv1_extradata_vector_from_libavcodec() {
        // A 38-byte FFV1 v3 config record body produced by FFmpeg — the CRC
        // over the first 38 bytes is 0x17137C03, and the full 42-byte
        // record has CRC 0.
        let ext = hex_to_bytes(
            "562b84d19c052f413c6026e95c376f5d1b76979d3ac9c420431e8b9f5520512f4ef8a1683b9b17137c03",
        );
        assert_eq!(ext.len(), 42);
        assert_eq!(crc32_ieee(&ext[..38]), 0x1713_7C03);
        assert_eq!(crc32_ieee(&ext), 0);
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(s.len() / 2);
        let bytes = s.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            let hi = hexval(bytes[i]);
            let lo = hexval(bytes[i + 1]);
            out.push((hi << 4) | lo);
            i += 2;
        }
        out
    }

    fn hexval(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => 0,
        }
    }
}
