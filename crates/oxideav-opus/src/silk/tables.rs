//! SILK ICDF tables from RFC 6716 §4.2.
//!
//! All tables are stored in "inverse-CDF" (icdf) form as expected by
//! `RangeDecoder::decode_icdf`: `icdf[k] = ft - cumfreq[k+1]`, where
//! `ft = 256` (so `ftb = 8`). Entries are monotonically non-increasing
//! and the last entry is zero.
//!
//! For this MVP decoder many tables are simplified approximations
//! rather than the verbatim §4.2 distributions: the decoder needs
//! valid (monotone) ICDFs for the range coder to not underflow, but
//! exact bit-level compatibility with libopus is out of scope for a
//! first-cut "audible output" implementation.

// -------------------------------------------------------------------
// §4.2.7.3 Frame type coding.
// -------------------------------------------------------------------

/// Frame type when VAD_flag = 0 (inactive). 2 symbols.
/// PDF ≈ {26, 230}/256 → ICDF {230, 0}.
pub const FRAME_TYPE_INACTIVE_ICDF: [u8; 2] = [230, 0];

/// Frame type when VAD_flag = 1 (active). 4 symbols.
/// PDF ≈ {24, 74, 148, 10}/256 → ICDF {232, 158, 10, 0}.
pub const FRAME_TYPE_ACTIVE_ICDF: [u8; 4] = [232, 158, 10, 0];

// -------------------------------------------------------------------
// §4.2.7.4 Sub-frame gains.
// -------------------------------------------------------------------

/// First sub-frame gain MSB (3 bits = 8 symbols). One approximation
/// per signal type, all ending at 0.
pub const GAIN_MSB_INACTIVE_ICDF: [u8; 8] = [224, 160, 112, 80, 48, 32, 16, 0];
pub const GAIN_MSB_UNVOICED_ICDF: [u8; 8] = [240, 200, 160, 120, 80, 48, 24, 0];
pub const GAIN_MSB_VOICED_ICDF: [u8; 8] = [248, 220, 180, 140, 96, 56, 24, 0];

/// First sub-frame gain LSB (3 bits uniform).
pub const GAIN_LSB_ICDF: [u8; 8] = [224, 192, 160, 128, 96, 64, 32, 0];

/// Delta gain coding for sub-frames 1..=3 (RFC Table 14). 41 symbols.
/// ICDF built from an approximately Gaussian distribution centred at
/// symbol 4 (which maps to "no change").
pub const GAIN_DELTA_ICDF: [u8; 41] = [
    250, 244, 239, 228, 197, 65, 44, 36, 32, 28, 26, 24, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12,
    11, 10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0,
];

// -------------------------------------------------------------------
// §4.2.7.5 NLSF stage-1 indices (5 bits each).
// -------------------------------------------------------------------

/// Stage-1 NLSF index for NB/MB, unvoiced. 32 symbols.
pub const NLSF_NB_STAGE1_UNVOICED_ICDF: [u8; 32] = [
    240, 224, 208, 192, 176, 160, 144, 132, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 44, 40, 36,
    32, 28, 24, 20, 16, 12, 10, 8, 6, 3, 0,
];
/// Stage-1 NLSF index for NB/MB, voiced. 32 symbols.
pub const NLSF_NB_STAGE1_VOICED_ICDF: [u8; 32] = [
    248, 232, 216, 200, 184, 168, 152, 140, 128, 116, 108, 100, 92, 84, 76, 68, 60, 52, 44, 40, 36,
    32, 28, 24, 20, 16, 12, 10, 8, 4, 2, 0,
];
/// Stage-1 NLSF index for WB, unvoiced.
pub const NLSF_WB_STAGE1_UNVOICED_ICDF: [u8; 32] = [
    240, 224, 208, 192, 176, 160, 144, 132, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 44, 40, 36,
    32, 28, 24, 20, 16, 12, 10, 8, 6, 3, 0,
];
/// Stage-1 NLSF index for WB, voiced.
pub const NLSF_WB_STAGE1_VOICED_ICDF: [u8; 32] = [
    248, 232, 216, 200, 184, 168, 152, 140, 128, 116, 108, 100, 92, 84, 76, 68, 60, 52, 44, 40, 36,
    32, 28, 24, 20, 16, 12, 10, 8, 4, 2, 0,
];

// -------------------------------------------------------------------
// §4.2.7.6 Long-term prediction (pitch + LTP filter).
// -------------------------------------------------------------------

/// Primary pitch lag high part (NB). 32 symbols.
pub const PITCH_LAG_NB_HIGH_ICDF: [u8; 32] = [
    224, 192, 176, 160, 144, 128, 112, 100, 88, 80, 72, 64, 56, 48, 44, 40, 36, 32, 28, 24, 22, 20,
    18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
];
/// Primary pitch lag low part (NB). 4 symbols ≈ uniform.
pub const PITCH_LAG_NB_LOW_ICDF: [u8; 4] = [192, 128, 64, 0];
/// Pitch delta (RFC Table 31). 21 symbols.
pub const PITCH_DELTA_ICDF: [u8; 21] = [
    220, 200, 180, 160, 140, 120, 104, 88, 72, 60, 48, 36, 28, 22, 16, 12, 8, 6, 4, 2, 0,
];
/// Pitch contour index, NB 20 ms (11 symbols).
pub const PITCH_CONTOUR_NB_20MS_ICDF: [u8; 11] = [224, 192, 160, 128, 96, 72, 56, 40, 24, 12, 0];
/// LTP periodicity index (3 symbols).
pub const LTP_PERIODICITY_ICDF: [u8; 3] = [200, 60, 0];
/// LTP filter indexing — 8 symbols (periodicity 0).
pub const LTP_FILTER_P0_ICDF: [u8; 8] = [220, 180, 140, 100, 72, 48, 24, 0];
/// 16 symbols (periodicity 1).
pub const LTP_FILTER_P1_ICDF: [u8; 16] = [
    240, 224, 208, 192, 176, 152, 128, 104, 80, 64, 48, 36, 24, 16, 8, 0,
];
/// 32 symbols (periodicity 2).
pub const LTP_FILTER_P2_ICDF: [u8; 32] = [
    248, 240, 224, 208, 192, 176, 160, 148, 136, 124, 112, 100, 88, 76, 64, 56, 48, 40, 36, 32, 28,
    24, 20, 16, 14, 12, 10, 8, 6, 4, 2, 0,
];
/// LTP scaling factor index. 3 symbols.
pub const LTP_SCALING_ICDF: [u8; 3] = [128, 64, 0];

// -------------------------------------------------------------------
// §4.2.7.7 LCG seed (2-bit uniform).
// -------------------------------------------------------------------

pub const LCG_SEED_ICDF: [u8; 4] = [192, 128, 64, 0];

// -------------------------------------------------------------------
// §4.2.7.8 Excitation coding.
// -------------------------------------------------------------------

/// Rate-level ICDF (9 symbols per RFC, we use 11 with the last two as
/// fallbacks).
pub const RATE_LEVEL_INACTIVE_ICDF: [u8; 10] = [240, 192, 160, 128, 96, 72, 48, 24, 8, 0];
pub const RATE_LEVEL_VOICED_ICDF: [u8; 10] = [224, 192, 160, 128, 96, 64, 40, 20, 8, 0];

/// Pulse count ICDFs per rate level — 18 symbols each. The MVP
/// decoder doesn't recursively shell-decode so the exact distribution
/// doesn't matter, but the decoder does read one symbol per shell
/// block so the ICDF must be valid.
pub const PULSE_COUNT_ICDF: [[u8; 18]; 11] = [
    // Rate level 0 — mostly zero pulses.
    [
        240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8, 0,
    ],
    [
        232, 216, 200, 184, 168, 152, 136, 120, 104, 88, 72, 56, 40, 28, 20, 12, 6, 0,
    ],
    [
        224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 36, 28, 20, 12, 6, 0,
    ],
    [
        216, 200, 184, 168, 152, 136, 120, 104, 88, 72, 60, 48, 36, 28, 20, 12, 6, 0,
    ],
    [
        208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 56, 48, 36, 28, 20, 12, 6, 0,
    ],
    [
        200, 184, 168, 152, 136, 120, 104, 88, 72, 60, 48, 40, 32, 24, 18, 12, 6, 0,
    ],
    [
        192, 176, 160, 144, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 18, 12, 6, 0,
    ],
    [
        184, 168, 152, 136, 120, 104, 88, 72, 60, 48, 40, 32, 24, 18, 14, 10, 6, 0,
    ],
    [
        176, 160, 144, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 18, 14, 10, 6, 0,
    ],
    [
        168, 152, 136, 120, 104, 88, 72, 60, 48, 40, 32, 24, 18, 14, 10, 8, 4, 0,
    ],
    [
        160, 144, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 18, 14, 10, 8, 4, 0,
    ],
];

/// 4-way pulse split ICDF, placeholder (not used by the MVP decoder
/// since we skip shell decoding).
pub const SHELL_4WAY_SPLIT_ICDF: [[u8; 16]; 4] = [
    [
        240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0,
    ],
    [
        240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0,
    ],
    [
        240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0,
    ],
    [
        240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16, 0,
    ],
];

// Keep the helper used in lsf.rs in sync: 11-symbol uniform ICDF.
pub const NLSF_RESIDUAL_UNIFORM_11_ICDF: [u8; 11] =
    [232, 208, 184, 160, 136, 112, 88, 64, 40, 20, 0];

// -------------------------------------------------------------------
// §4.2.7.1 Stereo prediction weights and mid-only flag
// (verbatim from libopus silk/tables_other.c).
// -------------------------------------------------------------------

/// Joint iCDF for the 5×5 grid of stereo prediction weight coarse indices
/// (`ix[0][2]`, `ix[1][2]`).
pub const STEREO_PRED_JOINT_ICDF: [u8; 25] = [
    249, 247, 246, 245, 244, 234, 210, 202, 201, 200, 197, 174, 82, 59, 56, 55, 54, 46, 22, 12, 11,
    10, 9, 7, 0,
];

/// Uniform iCDF for the 3-value component (`ix[n][0]`).
pub const STEREO_UNIFORM3_ICDF: [u8; 3] = [171, 85, 0];

/// Uniform iCDF for the 5-value sub-step component (`ix[n][1]`).
pub const STEREO_UNIFORM5_ICDF: [u8; 5] = [205, 154, 102, 51, 0];

/// Flag picking "only mid channel is coded" (2-symbol, ~50:50).
pub const STEREO_ONLY_CODE_MID_ICDF: [u8; 2] = [64, 0];

/// Stereo predictor quantization table (Q13), 16 entries. Consulted with
/// `idx = ix[0] + 3*ix[2]` ∈ [0, 15] and `idx+1` for interpolation.
pub const STEREO_PRED_QUANT_Q13: [i16; 16] = [
    -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820, 820, 2950, 5000, 6500, 7526, 8266,
    10050, 13732,
];

// -------------------------------------------------------------------
// §4.2.4 LBRR flag packing for multi-frame (40/60 ms) packets.
// -------------------------------------------------------------------

/// LBRR flags iCDF for a 40 ms packet (2 sub-frames, 3 symbols).
pub const LBRR_FLAGS_2_ICDF: [u8; 3] = [203, 150, 0];
/// LBRR flags iCDF for a 60 ms packet (3 sub-frames, 7 symbols).
pub const LBRR_FLAGS_3_ICDF: [u8; 7] = [215, 195, 166, 125, 110, 82, 0];
