//! CELT pitch post-filter / comb filter (RFC 6716 §4.3.8, libopus
//! `comb_filter` in celt.c).
//!
//! Applied to the IMDCT output. Input `x` must contain at least
//! `max(T0, T1) + 2` samples of history before index 0; output is written
//! into `y[0..N]`. `T0/g0/tapset0` are the previous frame's filter, and the
//! `overlap` window cross-fades from the old to the new `T1/g1/tapset1`.

use crate::tables::COMB_FILTER_TAPS;

const Q15ONE: f32 = 1.0;

#[allow(clippy::too_many_arguments)]
pub fn comb_filter(
    y: &mut [f32],
    x: &[f32],         // must be indexable for negative offsets via x_history
    x_history: &[f32], // last `max_period+2` samples from previous frame, ending right before x[0]
    t0: i32,
    t1: i32,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: &[f32],
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        if !std::ptr::eq(x, y as &[f32]) {
            y[..n].copy_from_slice(&x[..n]);
        }
        return;
    }
    let t0 = t0.max(crate::tables::COMB_FILTER_MINPERIOD as i32) as usize;
    let t1 = t1.max(crate::tables::COMB_FILTER_MINPERIOD as i32) as usize;
    let g00 = g0 * COMB_FILTER_TAPS[tapset0][0];
    let g01 = g0 * COMB_FILTER_TAPS[tapset0][1];
    let g02 = g0 * COMB_FILTER_TAPS[tapset0][2];
    let g10 = g1 * COMB_FILTER_TAPS[tapset1][0];
    let g11 = g1 * COMB_FILTER_TAPS[tapset1][1];
    let g12 = g1 * COMB_FILTER_TAPS[tapset1][2];
    let same = g0 == g1 && t0 == t1 && tapset0 == tapset1;
    let overlap = if same { 0 } else { overlap };

    // Helper to read x[i - offset] with negative i falling back to x_history.
    let h = x_history;
    let read = |i: isize| -> f32 {
        if i >= 0 {
            x[i as usize]
        } else {
            // i is negative; look back into history
            let idx = h.len() as isize + i;
            if idx >= 0 {
                h[idx as usize]
            } else {
                0.0
            }
        }
    };

    for i in 0..overlap {
        let f = window[i] * window[i];
        let one_minus_f = Q15ONE - f;
        let i = i as isize;
        let val = read(i)
            + (one_minus_f * g00) * read(i - t0 as isize)
            + (one_minus_f * g01) * (read(i - t0 as isize + 1) + read(i - t0 as isize - 1))
            + (one_minus_f * g02) * (read(i - t0 as isize + 2) + read(i - t0 as isize - 2))
            + (f * g10) * read(i - t1 as isize)
            + (f * g11) * (read(i - t1 as isize + 1) + read(i - t1 as isize - 1))
            + (f * g12) * (read(i - t1 as isize + 2) + read(i - t1 as isize - 2));
        y[i as usize] = val;
    }

    if g1 == 0.0 {
        if overlap < n {
            y[overlap..n].copy_from_slice(&x[overlap..n]);
        }
        return;
    }
    for i in overlap..n {
        let i_s = i as isize;
        let val = read(i_s)
            + g10 * read(i_s - t1 as isize)
            + g11 * (read(i_s - t1 as isize + 1) + read(i_s - t1 as isize - 1))
            + g12 * (read(i_s - t1 as isize + 2) + read(i_s - t1 as isize - 2));
        y[i] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_filter_zero_gain_is_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        let hist = vec![0.0; 32];
        let win = vec![1.0; 0];
        comb_filter(&mut y, &x, &hist, 16, 16, 4, 0.0, 0.0, 0, 0, &win, 0);
        assert_eq!(y, x);
    }
}
