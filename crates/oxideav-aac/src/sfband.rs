//! Scalefactor band offset tables — ISO/IEC 14496-3 §4.5.4.
//!
//! For each sample-rate-index there is a long-window (1024) and a short-
//! window (128) offset table. Each entry is the *starting* spectral
//! coefficient index of the band; the last entry equals the IMDCT length.
//!
//! Tables verified against ISO/IEC 14496-3:2009 Tables 4.110-4.128.

/// Long-window swb offsets per sampling-frequency-index (0..=12).
pub const SWB_LONG: [&[u16]; 13] = [
    SWB_LONG_96,
    SWB_LONG_96,
    SWB_LONG_64,
    SWB_LONG_48,
    SWB_LONG_48,
    SWB_LONG_32,
    SWB_LONG_24,
    SWB_LONG_24,
    SWB_LONG_16,
    SWB_LONG_16,
    SWB_LONG_16,
    SWB_LONG_8,
    SWB_LONG_8,
];

/// Short-window swb offsets per sampling-frequency-index.
pub const SWB_SHORT: [&[u16]; 13] = [
    SWB_SHORT_96,
    SWB_SHORT_96,
    SWB_SHORT_96,
    SWB_SHORT_48,
    SWB_SHORT_48,
    SWB_SHORT_48,
    SWB_SHORT_24,
    SWB_SHORT_24,
    SWB_SHORT_16,
    SWB_SHORT_16,
    SWB_SHORT_16,
    SWB_SHORT_8,
    SWB_SHORT_8,
];

const SWB_LONG_96: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

const SWB_LONG_64: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

const SWB_LONG_48: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

const SWB_LONG_32: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

const SWB_LONG_24: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

const SWB_LONG_16: &[u16] = &[
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

const SWB_LONG_8: &[u16] = &[
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

const SWB_SHORT_96: &[u16] = &[0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

const SWB_SHORT_48: &[u16] = &[0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

const SWB_SHORT_24: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

const SWB_SHORT_16: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

const SWB_SHORT_8: &[u16] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

/// Number of scalefactor bands for the long window.
pub fn num_swb_long(sf_index: u8) -> usize {
    SWB_LONG[sf_index as usize].len() - 1
}

/// Number of scalefactor bands for the short window.
pub fn num_swb_short(sf_index: u8) -> usize {
    SWB_SHORT[sf_index as usize].len() - 1
}
