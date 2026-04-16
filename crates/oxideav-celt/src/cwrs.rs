//! PVQ codebook decoder (libopus `cwrs.c`, RFC 6716 §4.3.4.5).
//!
//! Decodes the integer-codeword PVQ pulse vector for a band of size `n` with
//! `k` pulses. We use the SMALL_FOOTPRINT recurrence-based decoder (no large
//! lookup table), which is asymptotically O(N*K) but fits in pure Rust code
//! and never needs `unsafe`.

use crate::range_decoder::RangeDecoder;

/// Compute the next row/column of any recurrence that obeys
/// `u[i][j] = u[i-1][j] + u[i][j-1] + u[i-1][j-1]`.
fn unext(u: &mut [u32], len: usize, mut ui0: u32) {
    let mut j = 1;
    while j < len {
        let ui1 = u[j].wrapping_add(u[j - 1]).wrapping_add(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
    }
    u[j - 1] = ui0;
}

/// Compute the previous row/column of the same recurrence.
fn uprev(u: &mut [u32], n: usize, mut ui0: u32) {
    let mut j = 1;
    while j < n {
        let ui1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
    }
    u[j - 1] = ui0;
}

/// Compute V(_n,_k) and fill `_u[0.._k+1]` with U(_n, 0.._k+1).
fn ncwrs_urow(n: u32, k: u32, u: &mut [u32]) -> u32 {
    let len = (k + 2) as usize;
    debug_assert!(len >= 3);
    debug_assert!(n >= 2);
    debug_assert!(k > 0);
    u[0] = 0;
    u[1] = 1;
    let mut kk = 2;
    while kk < len {
        u[kk] = ((kk as u32) << 1) - 1;
        kk += 1;
    }
    for _ in 2..n {
        unext(&mut u[1..], (k + 1) as usize, 1);
    }
    u[k as usize].wrapping_add(u[(k + 1) as usize])
}

/// Decode `n*k` PVQ index into pulse vector `y`. Returns `yy = sum(y[i]^2)`.
fn cwrsi(n: usize, mut k: u32, mut idx: u32, y: &mut [i32], u: &mut [u32]) -> i32 {
    let mut yy: i32 = 0;
    let mut j = 0;
    while j < n {
        let mut p = u[(k + 1) as usize];
        let s = if idx >= p {
            idx = idx.wrapping_sub(p);
            -1i32
        } else {
            0i32
        };
        let mut yj = k as i32;
        p = u[k as usize];
        while p > idx {
            k -= 1;
            p = u[k as usize];
        }
        idx = idx.wrapping_sub(p);
        yj -= k as i32;
        let val = (yj + s) ^ s;
        y[j] = val;
        yy = yy.wrapping_add(val * val);
        uprev(u, (k + 2) as usize, 0);
        j += 1;
    }
    yy
}

/// Decode pulses from the range coder. Returns `||y||²`.
pub fn decode_pulses(y: &mut [i32], n: usize, k: u32, rc: &mut RangeDecoder<'_>) -> i32 {
    debug_assert!(k > 0);
    debug_assert!(n >= 2);
    let mut u = vec![0u32; (k + 2) as usize];
    let total = ncwrs_urow(n as u32, k, &mut u);
    let i = if total > 1 { rc.decode_uint(total) } else { 0 };
    cwrsi(n, k, i, y, &mut u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncwrs_urow_n2() {
        let mut u = vec![0u32; 6];
        let v = ncwrs_urow(2, 4, &mut u);
        // V(2,K) = 4*K = 16
        assert_eq!(v, 16);
    }

    #[test]
    fn decode_pulses_n2_k1_smoke() {
        // V(2,1)=4, so codeword is 2 bits.
        let mut buf = [0x80, 0x00, 0x00, 0x00];
        let mut rc = RangeDecoder::new(&buf[..]);
        let mut y = [0i32; 2];
        let yy = decode_pulses(&mut y, 2, 1, &mut rc);
        assert_eq!(yy, 1);
        assert_eq!(y.iter().map(|x| x.abs()).sum::<i32>(), 1);
        let _ = buf;
    }
}
