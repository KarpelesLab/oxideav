//! Forward 4×4 DCT + forward 4×4 Walsh-Hadamard — encoder companions to
//! [`crate::transform`]. RFC 6386 §14 describes the inverse; the forward
//! shapes below match libvpx's `vp8_short_fdct4x4` / `vp8_short_walsh4x4`
//! reference so that `fdct4x4 → idct4x4` recovers the input (up to the
//! spec-defined 1/8 residual scaling).
//!
//! Both transforms take a 16-element row-major i32 block and return a
//! 16-element i32 coefficient block (not yet quantised).

/// Forward 4×4 DCT (RFC 6386 §14 / libvpx reference).
///
/// Column then row. Output units match the decoder's `idct4x4`: running
/// `idct4x4(fdct4x4(x))` recovers `x` up to a factor-of-2 rounding (per
/// the VP8 spec which reserves 1 bit of headroom for the residual).
pub fn fdct4x4(input: &[i32; 16]) -> [i32; 16] {
    let mut work = [0i32; 16];
    for col in 0..4 {
        let s0 = input[col];
        let s1 = input[col + 4];
        let s2 = input[col + 8];
        let s3 = input[col + 12];
        let a = (s0 + s3) << 3;
        let b = (s1 + s2) << 3;
        let c = (s1 - s2) << 3;
        let d = (s0 - s3) << 3;
        work[col] = a + b;
        work[col + 8] = a - b;
        work[col + 4] = (c * 2217 + d * 5352 + 14500) >> 12;
        work[col + 12] = (d * 2217 - c * 5352 + 7500) >> 12;
    }
    let mut out = [0i32; 16];
    for row in 0..4 {
        let off = row * 4;
        let s0 = work[off];
        let s1 = work[off + 1];
        let s2 = work[off + 2];
        let s3 = work[off + 3];
        let a = s0 + s3;
        let b = s1 + s2;
        let c = s1 - s2;
        let d = s0 - s3;
        out[off] = (a + b + 7) >> 4;
        out[off + 2] = (a - b + 7) >> 4;
        out[off + 1] = ((c * 2217 + d * 5352 + 12000) >> 16) + (if d != 0 { 1 } else { 0 });
        out[off + 3] = (d * 2217 - c * 5352 + 51000) >> 16;
    }
    out
}

/// Forward 4×4 Walsh-Hadamard (RFC 6386 §14.3 / libvpx reference). Takes
/// the 16 DC coefficients from an intra-16×16 MB's luma sub-blocks and
/// returns the Y2 block coefficients that `iwht4x4` will invert.
pub fn fwht4x4(input: &[i32; 16]) -> [i32; 16] {
    // libvpx `vp8_short_walsh4x4_c`:
    //   pass 1 (rows): a1 = ((ip[0] + ip[2]) << 2); d1 = ((ip[1] + ip[3]) << 2);
    //                  c1 = ((ip[1] - ip[3]) << 2); b1 = ((ip[0] - ip[2]) << 2);
    //                  op[0] = a1 + d1 + (a1 != 0 ? 1 : 0);
    //                  op[1] = b1 + c1;
    //                  op[2] = b1 - c1;
    //                  op[3] = a1 - d1;
    //   pass 2 (cols): a2 = a1 + d1; d2 = a1 - d1; c2 = b1 - c1; b2 = b1 + c1;
    //                  op[0] = (a2 + 3) >> 3 + (a2 < 0 ? ... );  // see below
    // The iwht4x4 in `transform.rs` expects coefficients such that
    //   out = ((sum_of_transformed_with_signs + 3) >> 3)
    // Match that by this symmetric forward transform.
    let mut work = [0i32; 16];
    // Pass 1 — rows.
    for row in 0..4 {
        let off = row * 4;
        let a1 = (input[off] + input[off + 2]) << 2;
        let d1 = (input[off + 1] + input[off + 3]) << 2;
        let c1 = (input[off + 1] - input[off + 3]) << 2;
        let b1 = (input[off] - input[off + 2]) << 2;
        work[off] = a1 + d1 + if a1 != 0 { 1 } else { 0 };
        work[off + 1] = b1 + c1;
        work[off + 2] = b1 - c1;
        work[off + 3] = a1 - d1;
    }
    // Pass 2 — columns.
    let mut out = [0i32; 16];
    for col in 0..4 {
        let a1 = work[col] + work[col + 8];
        let d1 = work[col + 4] + work[col + 12];
        let c1 = work[col + 4] - work[col + 12];
        let b1 = work[col] - work[col + 8];
        let mut a2 = a1 + d1;
        let mut b2 = b1 + c1;
        let mut c2 = b1 - c1;
        let mut d2 = a1 - d1;
        if a2 < 0 {
            a2 += 1;
        }
        if b2 < 0 {
            b2 += 1;
        }
        if c2 < 0 {
            c2 += 1;
        }
        if d2 < 0 {
            d2 += 1;
        }
        out[col] = (a2 + 3) >> 3;
        out[col + 4] = (b2 + 3) >> 3;
        out[col + 8] = (c2 + 3) >> 3;
        out[col + 12] = (d2 + 3) >> 3;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{idct4x4, iwht4x4};

    #[test]
    fn fdct_then_idct_recovers_constant() {
        let x = [5i32; 16];
        let coeffs = fdct4x4(&x);
        let mut c16 = [0i16; 16];
        for i in 0..16 {
            c16[i] = coeffs[i] as i16;
        }
        let out = idct4x4(&c16);
        for &v in &out {
            assert!(
                (v as i32 - 5).abs() <= 1,
                "expected ~5, got {v}, coeffs={:?}",
                coeffs
            );
        }
    }

    #[test]
    fn fdct_then_idct_recovers_step() {
        let mut x = [0i32; 16];
        for i in 0..16 {
            x[i] = i as i32;
        }
        let coeffs = fdct4x4(&x);
        let mut c16 = [0i16; 16];
        for i in 0..16 {
            c16[i] = coeffs[i] as i16;
        }
        let out = idct4x4(&c16);
        // The fdct/idct pair should give ±1 accuracy for reasonably
        // small inputs — not exact due to VP8's integer rounding.
        let mut max = 0;
        for i in 0..16 {
            let d = (out[i] as i32 - x[i]).abs();
            if d > max {
                max = d;
            }
        }
        assert!(max <= 2, "round-trip error too large: {max}");
    }

    #[test]
    fn fwht_then_iwht_recovers() {
        let mut x = [0i32; 16];
        for i in 0..16 {
            x[i] = (i as i32) * 4 - 16;
        }
        let coeffs = fwht4x4(&x);
        let mut c16 = [0i16; 16];
        for i in 0..16 {
            c16[i] = coeffs[i] as i16;
        }
        let out = iwht4x4(&c16);
        let mut max = 0;
        for i in 0..16 {
            let d = (out[i] as i32 - x[i]).abs();
            if d > max {
                max = d;
            }
        }
        assert!(max <= 2, "WHT round-trip error too large: {max}");
    }
}
