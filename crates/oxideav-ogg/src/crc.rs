//! CRC-32 used by the Ogg page checksum field.
//!
//! Polynomial 0x04C11DB7, non-reflected, initial value 0, no final XOR — the
//! checksum is computed over the page bytes with the checksum field itself
//! zeroed.

/// CRC-32/Ogg lookup table, generated at compile time.
const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04C1_1DB7
            } else {
                r << 1
            };
            j += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

/// Compute the CRC-32 over `bytes` using the Ogg parameters.
pub fn checksum(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &b in bytes {
        crc = (crc << 8) ^ TABLE[((crc >> 24) as u8 ^ b) as usize];
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(checksum(&[]), 0);
    }

    #[test]
    fn deterministic() {
        // Same input must always produce the same checksum.
        assert_eq!(checksum(b"hello world"), checksum(b"hello world"));
        assert_ne!(checksum(b"hello world"), checksum(b"hello WORLD"));
    }
}
