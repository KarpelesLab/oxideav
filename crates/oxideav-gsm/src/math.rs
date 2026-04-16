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

/// `GSM_MULT(a, b) = (a*b) >> 15` (no rounding). Saturates to 16-bit.
#[inline]
pub fn mult(a: i16, b: i16) -> i16 {
    if a == MIN_WORD && b == MIN_WORD {
        MAX_WORD
    } else {
        saturate_i32_to_i16((a as i32 * b as i32) >> 15)
    }
}

/// Absolute value with saturation (|MIN_WORD| → MAX_WORD).
#[inline]
pub fn abs16(a: i16) -> i16 {
    if a == MIN_WORD {
        MAX_WORD
    } else if a < 0 {
        -a
    } else {
        a
    }
}

/// Saturating 32-bit add.
#[inline]
pub fn l_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Saturating 32-bit subtract.
#[inline]
#[allow(dead_code)]
pub fn l_sub(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// Number of left-shifts needed to normalise a non-zero 32-bit signed value
/// to the [2^30, 2^31) magnitude range. For negative values it works on
/// `!a`. Returns 0..=31. Matches libgsm's `gsm_norm`.
#[inline]
pub fn norm32(a: i32) -> u32 {
    if a == 0 {
        return 0; // Undefined in spec; libgsm asserts, we return 0.
    }
    if a < 0 {
        if a <= -1_073_741_824 {
            return 0;
        }
        let x = !a as u32;
        x.leading_zeros().saturating_sub(1)
    } else {
        (a as u32).leading_zeros().saturating_sub(1)
    }
}

/// Integer division `div = floor(num * 2^15 / denum)` for `0 <= num <= denum`,
/// returning a 15-bit result. Matches libgsm's `gsm_div`.
#[inline]
pub fn div16(num: i16, denum: i16) -> i16 {
    debug_assert!(num >= 0);
    debug_assert!(denum >= num);
    if num == 0 {
        return 0;
    }
    let mut l_num = num as i32;
    let l_den = denum as i32;
    let mut div: i16 = 0;
    for _ in 0..15 {
        div <<= 1;
        l_num <<= 1;
        if l_num >= l_den {
            l_num -= l_den;
            div += 1;
        }
    }
    div
}

/// Arithmetic shift-right with rounding semantics matching C's `>>` on
/// two's-complement; used as a readable stand-in for libgsm's `SASR(a, n)`.
#[inline]
pub fn sasr_i16(a: i16, n: u32) -> i16 {
    if n >= 16 {
        if a < 0 {
            -1
        } else {
            0
        }
    } else {
        a >> n
    }
}

#[inline]
pub fn sasr_i32(a: i32, n: u32) -> i32 {
    if n >= 32 {
        if a < 0 {
            -1
        } else {
            0
        }
    } else {
        a >> n
    }
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
