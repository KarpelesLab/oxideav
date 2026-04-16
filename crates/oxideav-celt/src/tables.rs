//! Static CELT tables from RFC 6716 §4.3 (informative reference: libopus
//! `static_modes_float.h`, `quant_bands.c`, `rate.c`).
//!
//! All tables are pure constants transcribed verbatim from the libopus
//! reference (data, not code logic). The CELT decoder needs:
//!
//! * `EBAND_5MS` — band edges (RFC 6716 §4.3 Table 55).
//! * `E_PROB_MODEL` — Laplace parameters per (LM, intra, band) for coarse
//!   energy decoding (RFC §4.3.2.1).
//! * `PRED_COEF` / `BETA_COEF` / `BETA_INTRA` — inter/intra prediction
//!   coefficients in Q15.
//! * `BAND_ALLOCATION` — base bit budget per band per quality level
//!   (RFC §4.3.3 / Table 57).
//! * `LOG2_FRAC_TABLE` — fractional log2 lookup (rate.c).
//! * `CACHE_INDEX50` / `CACHE_BITS50` / `CACHE_CAPS50` — PVQ pulse-count
//!   thresholds and per-band caps (RFC §4.3.3).

/// CELT band edges in units of MDCT bins for a 5-ms (LM=0) frame at 48 kHz.
/// Each entry is the *start* of the band; the next entry is the start of the
/// next band, so band `i` spans `[eband_5ms[i], eband_5ms[i+1])`.
///
/// At LM>0 (longer frames) the same table is multiplied by `1 << LM` to
/// expand the per-band MDCT-bin count.
///
/// Source: libopus `static_modes_float.h` (eband5ms[]).
pub const EBAND_5MS: [u16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Number of bands actually decoded for each [bandwidth][end_band_table_idx]
/// combination. CELT-only modes use the full table up to the bandwidth limit.
///
/// Maps `OpusBandwidth` → upper-band index used at the decoder. The lower
/// edge `start` is always 0 for CELT-only frames (Hybrid uses 17).
pub fn end_band_for_bandwidth_celt(cutoff_hz: u32) -> usize {
    // Per RFC 6716 §4.3 + libopus `mode.c` `compute_ebands`:
    //   NB: 13 bands  (≤ 4 kHz)
    //   WB: 17 bands  (≤ 8 kHz)
    //   SWB: 19 bands (≤ 12 kHz)
    //   FB: 21 bands  (≤ 20 kHz)
    match cutoff_hz {
        0..=4_000 => 13,
        4_001..=8_000 => 17,
        8_001..=12_000 => 19,
        _ => 21,
    }
}

/// log2(frame_samples_48k / 120) — the "LM" shift used throughout RFC §4.3.
pub fn lm_for_frame_samples(frame_samples_48k: u32) -> u32 {
    match frame_samples_48k {
        120 => 0,
        240 => 1,
        480 => 2,
        960 => 3,
        _ => 0,
    }
}

/// Number of CELT bands (always 21 for the standard mode).
pub const NB_EBANDS: usize = 21;

/// Inter-frame prediction coefficients for coarse energy (Q15), one per LM.
/// libopus `pred_coef`.
pub const PRED_COEF_Q15: [i16; 4] = [29440, 26112, 21248, 16384];

/// Inter-frame prediction beta coefficients (Q15), one per LM.
/// libopus `beta_coef`.
pub const BETA_COEF_Q15: [i16; 4] = [30147, 22282, 12124, 6554];

/// Intra-frame prediction beta coefficient (Q15). libopus `beta_intra`.
pub const BETA_INTRA_Q15: i16 = 4915;

/// Floating-point versions for ergonomic use in non-fixed-point code.
pub const PRED_COEF_F32: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
pub const BETA_COEF_F32: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
pub const BETA_INTRA_F32: f32 = 4915.0 / 32768.0;

/// Laplace probability-model parameters per (LM, intra, band-pair).
/// 4 frame sizes × 2 prediction types × 21 (prob, decay) pairs.
/// libopus `e_prob_model` (quant_bands.c).
#[rustfmt::skip]
pub const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    /* 120-sample frames (LM=0) */
    [
        /* Inter */
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128,
            64, 128, 92, 78, 92, 79, 92, 78, 90, 79, 116, 41, 115, 40,
            114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10, 177, 11,
        ],
        /* Intra */
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132,
            55, 132, 61, 114, 70, 96, 74, 88, 75, 88, 87, 74, 89, 66,
            91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43, 78, 50,
        ],
    ],
    /* 240-sample frames (LM=1) */
    [
        /* Inter */
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74,
            93, 74, 109, 40, 114, 36, 117, 34, 117, 34, 143, 17, 145, 18,
            146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177, 9,
        ],
        /* Intra */
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91,
            73, 91, 78, 89, 86, 80, 92, 66, 93, 64, 102, 59, 103, 60,
            104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77, 45,
        ],
    ],
    /* 480-sample frames (LM=2) */
    [
        /* Inter */
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38,
            112, 38, 124, 26, 132, 27, 136, 19, 140, 20, 155, 14, 159, 16,
            158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9, 159, 10,
        ],
        /* Intra */
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73,
            87, 72, 92, 75, 98, 72, 105, 58, 107, 54, 115, 52, 114, 55,
            112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77, 42,
        ],
    ],
    /* 960-sample frames (LM=3) */
    [
        /* Inter */
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36,
            119, 33, 127, 33, 134, 34, 139, 21, 147, 23, 152, 20, 158, 25,
            154, 26, 166, 21, 173, 16, 184, 13, 184, 10, 150, 13, 139, 15,
        ],
        /* Intra */
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72,
            96, 67, 101, 73, 107, 72, 113, 55, 118, 52, 125, 52, 118, 52,
            117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97, 33, 77, 40,
        ],
    ],
];

/// ICDF for the small-energy (≤1-bit budget) fallback in coarse energy
/// decoding. libopus `small_energy_icdf`.
pub const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Bit allocation table (`band_allocation` in libopus `modes.c`).
/// 11 quality levels × 21 bands. Units are 1/32 bit/sample.
pub const BITALLOC_SIZE: usize = 11;
#[rustfmt::skip]
pub const BAND_ALLOCATION: [u8; BITALLOC_SIZE * 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0,
    110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0, 0, 0, 0,
    118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4, 0, 0, 0, 0,
    126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12, 1, 0, 0,
    134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10, 1,
    144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1,
    162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1,
    172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45, 20,
    200, 200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153, 148, 129, 104,
];

/// Fractional log2 lookup (libopus `LOG2_FRAC_TABLE`). Values are log2(1+i/8)
/// in 1/8th-bit units, indexed 0..24.
pub const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// log2(N) per band in Q4 (libopus `logN400`).
#[rustfmt::skip]
pub const LOGN400: [i16; NB_EBANDS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];

/// PVQ pulse-cache index (libopus `cache_index50`). Indexed
/// `(LM+1)*nbEBands + i`. Negative entries mean N is too small for this band.
#[rustfmt::skip]
pub const CACHE_INDEX50: [i16; 105] = [
    -1, -1, -1, -1, -1, -1, -1, -1,   0,   0,   0,   0,  41,  41,  41,
     82,  82, 123, 164, 200, 222,   0,   0,   0,   0,   0,   0,   0,   0,  41,
     41,  41,  41, 123, 123, 123, 164, 164, 240, 266, 283, 295,  41,  41,  41,
     41,  41,  41,  41,  41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305,
    318, 328, 336, 123, 123, 123, 123, 123, 123, 123, 123, 240, 240, 240, 240,
    305, 305, 305, 318, 318, 343, 351, 358, 364, 240, 240, 240, 240, 240, 240,
    240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376, 382, 387,
];

/// PVQ pulse-cache bits table (libopus `cache_bits50`). `cache[0]` at any
/// index `CACHE_INDEX50[(LM+1)*nbEBands+i]` gives `cache.size`, then the
/// next `cache.size` bytes give the bit cost in 1/8 bit units for q=1..size.
#[rustfmt::skip]
pub const CACHE_BITS50: [u8; 392] = [
    40,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 40, 15, 23, 28,
    31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50,
    51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63, 65,
    66, 67, 68, 69, 70, 71, 71, 40, 20, 33, 41, 48, 53, 57, 61,
    64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92,
    94, 96, 98,101,103,105,107,108,110,112,114,117,119,121,123,
   124,126,128, 40, 23, 39, 51, 60, 67, 73, 79, 83, 87, 91, 94,
    97,100,102,105,107,111,115,118,121,124,126,129,131,135,139,
   142,145,148,150,153,155,159,163,166,169,172,174,177,179, 35,
    28, 49, 65, 78, 89, 99,107,114,120,126,132,136,141,145,149,
   153,159,165,171,176,180,185,189,192,199,205,211,216,220,225,
   229,232,239,245,251, 21, 33, 58, 79, 97,112,125,137,148,157,
   166,174,182,189,195,201,207,217,227,235,243,251, 17, 35, 63,
    86,106,123,139,152,165,177,187,197,206,214,222,230,237,250,
    25, 31, 55, 75, 91,105,117,128,138,146,154,161,168,174,180,
   185,190,200,208,215,222,229,235,240,245,255, 16, 36, 65, 89,
   110,128,144,159,173,185,196,207,217,226,234,242,250, 11, 41,
    74,103,128,151,172,191,209,225,241,255,  9, 43, 79,110,138,
   163,186,207,227,246, 12, 39, 71, 99,123,144,164,182,198,214,
   228,241,253,  9, 44, 81,113,142,168,192,214,235,255,  7, 49,
    90,127,160,191,220,247,  6, 51, 95,134,170,203,234,  7, 47,
    87,123,155,184,212,237,  6, 52, 97,137,174,208,240,  5, 57,
   106,151,192,231,  5, 59,111,158,202,243,  5, 55,103,147,187,
   224,  5, 60,113,161,206,248,  4, 65,122,175,224,  4, 67,127,
   182,234,
];

/// PVQ caps (libopus `cache_caps50`). Indexed `(2*LM+C-1)*nbEBands + i`.
#[rustfmt::skip]
pub const CACHE_CAPS50: [u8; 168] = [
    224,224,224,224,224,224,224,224,160,160,160,160,185,185,185,
    178,178,168,134, 61, 37,224,224,224,224,224,224,224,224,240,
    240,240,240,207,207,207,198,198,183,144, 66, 40,160,160,160,
    160,160,160,160,160,185,185,185,185,193,193,193,183,183,172,
    138, 64, 38,240,240,240,240,240,240,240,240,207,207,207,207,
    204,204,204,193,193,180,143, 66, 40,185,185,185,185,185,185,
    185,185,193,193,193,193,193,193,193,183,183,172,138, 65, 39,
    207,207,207,207,207,207,207,207,204,204,204,204,201,201,201,
    188,188,176,141, 66, 40,193,193,193,193,193,193,193,193,193,
    193,193,193,194,194,194,184,184,173,139, 65, 39,204,204,204,
    204,204,204,204,204,201,201,201,201,198,198,198,187,187,175,
    140, 66, 40,
];

/// Mean-energy offsets per band for denormalisation (libopus `eMeans` float).
pub const E_MEANS: [f32; 25] = [
    6.437500, 6.250000, 5.750000, 5.312500, 5.062500, 4.812500, 4.500000, 4.375000, 4.875000,
    4.687500, 4.562500, 4.437500, 4.875000, 4.625000, 4.312500, 4.500000, 4.375000, 4.625000,
    4.750000, 4.437500, 3.750000, 3.750000, 3.750000, 3.750000, 3.750000,
];

/// Spread decision ICDF (libopus `spread_icdf`).
pub const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// Allocation trim ICDF (libopus `trim_icdf`).
pub const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Tapset ICDF (libopus `tapset_icdf`).
pub const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// CELT spread mode constants.
pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// `tf_select_table` (libopus celt.c). Indexed `[LM][4*isTransient + 2*tf_select + per_band]`.
pub const TF_SELECT_TABLE: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// CELT comb-filter tap sets (libopus celt.c `init_caps`/`comb_filter`).
pub const COMB_FILTER_TAPS: [[f32; 3]; 3] = [
    [0.306_640_6, 0.218_750_0, 0.0],
    [0.460_937_5, 0.246_093_75, 0.0],
    [0.798_828_1, 0.108_398_44, 0.091_796_88],
];

/// CELT minimum comb-filter period (libopus `COMBFILTER_MINPERIOD`).
pub const COMB_FILTER_MINPERIOD: u32 = 15;

/// `MAX_FINE_BITS` from libopus rate.h.
pub const MAX_FINE_BITS: i32 = 8;

/// `FINE_OFFSET` from libopus rate.h.
pub const FINE_OFFSET: i32 = 21;

/// `QTHETA_OFFSET` from libopus rate.h.
pub const QTHETA_OFFSET: i32 = 4;

/// `QTHETA_OFFSET_TWOPHASE` from libopus rate.h.
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// `ALLOC_STEPS` from libopus rate.h.
pub const ALLOC_STEPS: i32 = 6;

/// 60-degree raised-cosine cosine table (libopus `bitexact_cos`).
/// Returns cos(itheta * pi/16384) in Q15. We implement this via formula.
pub fn bitexact_cos(x: i16) -> i16 {
    let x = x as i32;
    let tmp = (32_768 + (x * x)) >> 16;
    let mut x2 = 32_767 - tmp;
    x2 += (x2 * (-7_651 + ((x2 * (8_277 + ((-626 * x2) >> 15))) >> 15))) >> 15;
    (1 + x2) as i16
}

/// `bitexact_log2tan` (libopus mathops.h) — returns log2(tan(theta)) approx.
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = (icos as u32).leading_zeros() as i32 - 17;
    let ls = (isin as u32).leading_zeros() as i32 - 17;
    let icos = icos << lc;
    let isin = isin << ls;
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2_597) + 7_932)
        - frac_mul16(icos, frac_mul16(icos, -2_597) + 7_932)
}

#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16_384 + (a * b)) >> 15
}

/// Number of pulses for a given quantized index `q` (libopus rate.h
/// `get_pulses`). Translates encoded q to pulse count K.
#[inline]
pub fn get_pulses(q: i32) -> i32 {
    if q < 8 {
        q
    } else {
        (8 + (q & 7)) << ((q >> 3) - 1)
    }
}

/// Per-band PVQ pulse cap lookup. `cap[i] = (cache_caps50[(2*LM+C-1)*nbEBands+i]+64)*C*N >> 2`.
pub fn init_caps(lm: usize, c: usize) -> [i32; NB_EBANDS] {
    let mut cap = [0i32; NB_EBANDS];
    let row = (2 * lm + c - 1) * NB_EBANDS;
    let m = 1u32 << lm;
    for i in 0..NB_EBANDS {
        let n = (EBAND_5MS[i + 1] as u32 - EBAND_5MS[i] as u32) * m;
        cap[i] = ((CACHE_CAPS50[row + i] as i32 + 64) * c as i32 * n as i32) >> 2;
    }
    cap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eband_5ms_is_monotonic_and_ends_at_100() {
        for w in EBAND_5MS.windows(2) {
            assert!(w[0] < w[1], "EBAND_5MS not strictly increasing");
        }
        assert_eq!(*EBAND_5MS.last().unwrap(), 100);
    }

    #[test]
    fn lm_is_log2_frame_size_over_120() {
        assert_eq!(lm_for_frame_samples(120), 0);
        assert_eq!(lm_for_frame_samples(240), 1);
        assert_eq!(lm_for_frame_samples(480), 2);
        assert_eq!(lm_for_frame_samples(960), 3);
    }

    #[test]
    fn end_band_increases_with_bandwidth() {
        assert!(end_band_for_bandwidth_celt(4_000) < end_band_for_bandwidth_celt(8_000));
        assert!(end_band_for_bandwidth_celt(8_000) < end_band_for_bandwidth_celt(12_000));
        assert!(end_band_for_bandwidth_celt(12_000) < end_band_for_bandwidth_celt(20_000));
        assert_eq!(end_band_for_bandwidth_celt(20_000), 21);
    }
}
