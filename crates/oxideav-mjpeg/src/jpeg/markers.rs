//! JPEG marker byte constants (ITU T.81 Annex B).
//!
//! Each marker is a two-byte sequence `0xFF, 0x??`. Only the second byte
//! is declared here. The first byte is always `0xFF`; a run of `0xFF`
//! bytes before the payload byte is legal as fill.

/// Start of Image.
pub const SOI: u8 = 0xD8;
/// End of Image.
pub const EOI: u8 = 0xD9;
/// Start of Scan.
pub const SOS: u8 = 0xDA;
/// Define Quantization Table(s).
pub const DQT: u8 = 0xDB;
/// Define Huffman Table(s).
pub const DHT: u8 = 0xC4;
/// Define Restart Interval.
pub const DRI: u8 = 0xDD;
/// COMment segment.
pub const COM: u8 = 0xFE;

/// Baseline DCT sequential.
pub const SOF0: u8 = 0xC0;
/// Extended sequential DCT.
pub const SOF1: u8 = 0xC1;
/// Progressive DCT (unsupported).
pub const SOF2: u8 = 0xC2;
/// Lossless sequential (unsupported).
pub const SOF3: u8 = 0xC3;

/// Application-specific markers APP0..APP15 (0xE0..0xEF).
pub const APP0: u8 = 0xE0;
pub const APP15: u8 = 0xEF;

/// Restart markers RST0..RST7 (0xD0..0xD7).
pub const RST0: u8 = 0xD0;
pub const RST7: u8 = 0xD7;

/// Returns true if `b` is an `APPn` marker byte.
pub fn is_app(b: u8) -> bool {
    (APP0..=APP15).contains(&b)
}

/// Returns true if `b` is a restart (RSTn) marker byte.
pub fn is_rst(b: u8) -> bool {
    (RST0..=RST7).contains(&b)
}

/// Returns true if `b` is any SOFn (Start Of Frame) marker.
/// Excludes DHT (0xC4) and JPG (0xC8).
pub fn is_sof(b: u8) -> bool {
    matches!(
        b,
        0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
    )
}
