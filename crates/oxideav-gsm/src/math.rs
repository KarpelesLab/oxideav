//! 16-/32-bit saturating arithmetic helpers used throughout 06.10.
//!
//! The GSM spec is defined in terms of fixed-point, saturating integer
//! operations. These helpers give them short names (`add`, `sub`, `mult_r`,
//! `shl`, `saturate_i32_to_i16`) that line up with the pseudocode in the
//! ETSI reference and the libgsm implementation.

pub const MIN_WORD: i16 = i16::MIN;
pub const MAX_WORD: i16 = i16::MAX;

#[inline]
pub fn saturate_i32_to_i16(v: i32) -> i16 {
    if v > MAX_WORD as i32 {
        MAX_WORD
    } else if v < MIN_WORD as i32 {
        MIN_WORD
    } else {
        v as i16
    }
}

/// 16-bit saturating add.
#[inline]
pub fn add(a: i16, b: i16) -> i16 {
    saturate_i32_to_i16(a as i32 + b as i32)
}

/// 16-bit saturating subtract.
#[inline]
pub fn sub(a: i16, b: i16) -> i16 {
    saturate_i32_to_i16(a as i32 - b as i32)
}

/// Multiply with rounding: `MULT_R(a, b) = (a*b + 0x4000) >> 15`.
#[inline]
pub fn mult_r(a: i16, b: i16) -> i16 {
    if a == MIN_WORD && b == MIN_WORD {
        MAX_WORD
    } else {
        let prod = (a as i32 * b as i32) + 0x4000;
        saturate_i32_to_i16(prod >> 15)
    }
}

/// Shift-left with saturation; `n` may be 0..=15.
#[inline]
pub fn shl(a: i16, n: u32) -> i16 {
    let v = (a as i32) << n;
    saturate_i32_to_i16(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_saturates() {
        assert_eq!(add(30_000, 30_000), MAX_WORD);
        assert_eq!(add(-30_000, -30_000), MIN_WORD);
        assert_eq!(add(100, -50), 50);
    }

    #[test]
    fn mult_r_rounding() {
        assert_eq!(mult_r(MIN_WORD, MIN_WORD), MAX_WORD);
        assert_eq!(mult_r(0x4000, 0x4000), 0x2000);
    }

    #[test]
    fn shl_saturates() {
        assert_eq!(shl(0x4000, 1), MAX_WORD);
        assert_eq!(shl(-0x4000, 2), MIN_WORD);
    }
}
