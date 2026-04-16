//! BITMAPINFOHEADER / WAVEFORMATEX parse + emit helpers.
//!
//! These are the two standard Windows structs AVI uses in `strf` chunks:
//! - Video streams embed a `BITMAPINFOHEADER` (optionally followed by codec
//!   extradata).
//! - Audio streams embed a `WAVEFORMATEX` (or the legacy 14-byte
//!   `WAVEFORMAT`/`PCMWAVEFORMAT`) followed by `cbSize` bytes of extradata.

use oxideav_core::{Error, Result};

/// Decoded BITMAPINFOHEADER + trailing extradata.
#[derive(Clone, Debug)]
pub struct BitmapInfoHeader {
    pub width: u32,
    pub height: u32,
    pub planes: u16,
    pub bit_count: u16,
    pub compression: [u8; 4],
    pub size_image: u32,
    pub x_pels_per_meter: i32,
    pub y_pels_per_meter: i32,
    pub clr_used: u32,
    pub clr_important: u32,
    /// Extradata following the 40-byte header (palette/codec private data).
    pub extradata: Vec<u8>,
}

/// Parse a BITMAPINFOHEADER (+ trailing extradata) from a `strf` payload.
///
/// The header is 40 bytes; anything after is treated as codec extradata.
/// `biCompression` is returned as-is; integer IDs like BI_RGB (0) appear as
/// `[0,0,0,0]`.
pub fn parse_bitmap_info_header(buf: &[u8]) -> Result<BitmapInfoHeader> {
    if buf.len() < 40 {
        return Err(Error::invalid("AVI: BITMAPINFOHEADER too short"));
    }
    let _bi_size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let width = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as i32 as u32;
    let height_signed = i32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    // height can be negative to indicate top-down orientation; take absolute.
    let height = height_signed.unsigned_abs();
    let planes = u16::from_le_bytes([buf[12], buf[13]]);
    let bit_count = u16::from_le_bytes([buf[14], buf[15]]);
    let mut compression = [0u8; 4];
    compression.copy_from_slice(&buf[16..20]);
    let size_image = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let x_pels_per_meter = i32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
    let y_pels_per_meter = i32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);
    let clr_used = u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]);
    let clr_important = u32::from_le_bytes([buf[36], buf[37], buf[38], buf[39]]);
    // Optional extradata. The 40-byte BITMAPINFOHEADER is followed by any
    // codec-private bytes the encoder chose to attach. `biSize` may either
    // equal 40 (fixed header) or include the extension bytes — either way
    // anything past byte 40 belongs to extradata.
    let extradata = if buf.len() > 40 {
        buf[40..].to_vec()
    } else {
        Vec::new()
    };
    Ok(BitmapInfoHeader {
        width,
        height,
        planes,
        bit_count,
        compression,
        size_image,
        x_pels_per_meter,
        y_pels_per_meter,
        clr_used,
        clr_important,
        extradata,
    })
}

/// Emit a 40-byte BITMAPINFOHEADER followed by optional extradata.
pub fn write_bitmap_info_header(
    width: u32,
    height: u32,
    compression: [u8; 4],
    bit_count: u16,
    extradata: &[u8],
) -> Vec<u8> {
    let bi_size = 40u32 + extradata.len() as u32;
    let mut out = Vec::with_capacity(bi_size as usize);
    out.extend_from_slice(&bi_size.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // planes = 1
    out.extend_from_slice(&bit_count.to_le_bytes());
    out.extend_from_slice(&compression);
    // size_image = width * height * bit_count / 8 (rough; 0 acceptable for
    // non-BI_RGB). Use 0 to let the decoder derive it.
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // x_pels_per_meter
    out.extend_from_slice(&0i32.to_le_bytes()); // y_pels_per_meter
    out.extend_from_slice(&0u32.to_le_bytes()); // clr_used
    out.extend_from_slice(&0u32.to_le_bytes()); // clr_important
    out.extend_from_slice(extradata);
    out
}

/// Decoded WAVEFORMATEX + extradata.
#[derive(Clone, Debug)]
pub struct WaveFormatEx {
    pub format_tag: u16,
    pub channels: u16,
    pub samples_per_sec: u32,
    pub avg_bytes_per_sec: u32,
    pub block_align: u16,
    pub bits_per_sample: u16,
    pub extradata: Vec<u8>,
}

/// Parse a WAVEFORMATEX from a `strf` payload.
///
/// The legacy 14-byte WAVEFORMAT (no `wBitsPerSample`) and the 16-byte
/// PCMWAVEFORMAT (no `cbSize`) are both accepted — missing fields default to
/// zero.
pub fn parse_waveformatex(buf: &[u8]) -> Result<WaveFormatEx> {
    if buf.len() < 14 {
        return Err(Error::invalid("AVI: WAVEFORMAT(EX) too short"));
    }
    let format_tag = u16::from_le_bytes([buf[0], buf[1]]);
    let channels = u16::from_le_bytes([buf[2], buf[3]]);
    let samples_per_sec = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let avg_bytes_per_sec = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let block_align = u16::from_le_bytes([buf[12], buf[13]]);
    let bits_per_sample = if buf.len() >= 16 {
        u16::from_le_bytes([buf[14], buf[15]])
    } else {
        0
    };
    let extradata = if buf.len() >= 18 {
        let cb_size = u16::from_le_bytes([buf[16], buf[17]]) as usize;
        if 18 + cb_size <= buf.len() {
            buf[18..18 + cb_size].to_vec()
        } else {
            buf[18..].to_vec()
        }
    } else {
        Vec::new()
    };
    Ok(WaveFormatEx {
        format_tag,
        channels,
        samples_per_sec,
        avg_bytes_per_sec,
        block_align,
        bits_per_sample,
        extradata,
    })
}

/// Emit a WAVEFORMATEX (always 18 bytes + extradata).
#[allow(clippy::too_many_arguments)]
pub fn write_waveformatex(
    format_tag: u16,
    channels: u16,
    samples_per_sec: u32,
    avg_bytes_per_sec: u32,
    block_align: u16,
    bits_per_sample: u16,
    extradata: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(18 + extradata.len());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&samples_per_sec.to_le_bytes());
    out.extend_from_slice(&avg_bytes_per_sec.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(&(extradata.len() as u16).to_le_bytes());
    out.extend_from_slice(extradata);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmih_roundtrip() {
        let bytes = write_bitmap_info_header(320, 240, *b"MJPG", 24, &[0xAA, 0xBB]);
        assert_eq!(bytes.len(), 42);
        let h = parse_bitmap_info_header(&bytes).unwrap();
        assert_eq!(h.width, 320);
        assert_eq!(h.height, 240);
        assert_eq!(&h.compression, b"MJPG");
        assert_eq!(h.bit_count, 24);
        assert_eq!(h.extradata, vec![0xAA, 0xBB]);
    }

    #[test]
    fn wfx_roundtrip() {
        let bytes = write_waveformatex(1, 2, 48_000, 192_000, 4, 16, &[]);
        assert_eq!(bytes.len(), 18);
        let h = parse_waveformatex(&bytes).unwrap();
        assert_eq!(h.format_tag, 1);
        assert_eq!(h.channels, 2);
        assert_eq!(h.samples_per_sec, 48_000);
        assert_eq!(h.block_align, 4);
        assert_eq!(h.bits_per_sample, 16);
    }

    #[test]
    fn wfx_short_legacy() {
        // Legacy 14-byte WAVEFORMAT without bps.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0001u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        let h = parse_waveformatex(&bytes).unwrap();
        assert_eq!(h.bits_per_sample, 0);
        assert!(h.extradata.is_empty());
    }
}
