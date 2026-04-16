//! JPEG 8×8 zigzag scan order (ITU T.81 Figure A.6).
//!
//! Coefficient `k` in zigzag order lives at position `ZIGZAG[k]` in the
//! natural row-major 8×8 block. The inverse mapping (natural → zigzag) is
//! [`INV_ZIGZAG`].

pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse of [`ZIGZAG`] — maps a natural-order index (row*8 + col) to its
/// zigzag position.
pub const INV_ZIGZAG: [usize; 64] = {
    let mut out = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        out[ZIGZAG[i]] = i;
        i += 1;
    }
    out
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_permutation() {
        let mut seen = [false; 64];
        for &v in &ZIGZAG {
            assert!(!seen[v]);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn inv_zigzag_inverts() {
        for i in 0..64 {
            assert_eq!(INV_ZIGZAG[ZIGZAG[i]], i);
        }
    }
}
