//! ProRes frame / picture headers.
//!
//! This module models a simplified, progressive-only, single-picture
//! ProRes container that stays faithful to the shape of SMPTE RDD 36
//! without requiring byte-exact compatibility with the spec's
//! trailing-bits or reserved-field encoding. Fields match the spec
//! ones we care about; the quant-matrix bytes are emitted literally so
//! the decoder can recover the two per-plane matrices without any
//! profile lookup.
//!
//! ## Frame header (fixed 152 bytes)
//!
//! | offset | size | field                           |
//! |------- |----- |---------------------------------|
//! | 0      | 4    | frame_size (BE u32, incl. hdr)  |
//! | 4      | 4    | magic = b"icpf"                 |
//! | 8      | 2    | frame_hdr_size (BE u16 = 152)   |
//! | 10     | 2    | reserved = 0                    |
//! | 12     | 4    | creator id = b"oxav"            |
//! | 16     | 2    | width (BE u16)                  |
//! | 18     | 2    | height (BE u16)                 |
//! | 20     | 1    | chroma_fmt = 2 (4:2:2)          |
//! | 21     | 1    | flags (bit0 = load_luma_qmat,   |
//! |        |      |        bit1 = load_chroma_qmat) |
//! | 22     | 1    | profile_code (0=Proxy, 1=LT,    |
//! |        |      |               2=Standard)       |
//! | 23     | 1    | reserved = 0                    |
//! | 24     | 64   | luma_qmat (natural order)       |
//! | 88     | 64   | chroma_qmat (natural order)     |
//!
//! ## Picture header (fixed 8 bytes)
//!
//! | offset | size | field                           |
//! |------- |----- |---------------------------------|
//! | 0      | 1    | picture_hdr_size = 8            |
//! | 1      | 4    | picture_size (BE u32, payload   |
//! |        |      |   = slice-table + slice bytes)  |
//! | 5      | 2    | slice_count (BE u16)            |
//! | 7      | 1    | log2_desired_slice_mb_width (=3)|
//!
//! Followed by a slice-size table: `slice_count * u16` BE sizes, then
//! the concatenated slice payloads.

use oxideav_core::{Error, Result};

pub const MAGIC: &[u8; 4] = b"icpf";
pub const CREATOR: &[u8; 4] = b"oxav";
pub const FRAME_HDR_SIZE: usize = 152;
pub const PICTURE_HDR_SIZE: usize = 8;

pub const CHROMA_FMT_422: u8 = 2;

pub const FLAG_LOAD_LUMA_QMAT: u8 = 1 << 0;
pub const FLAG_LOAD_CHROMA_QMAT: u8 = 1 << 1;

/// Profile discriminator stored in the frame header.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Profile {
    Proxy = 0,
    Lt = 1,
    Standard = 2,
}

impl Profile {
    pub fn from_code(c: u8) -> Result<Self> {
        match c {
            0 => Ok(Self::Proxy),
            1 => Ok(Self::Lt),
            2 => Ok(Self::Standard),
            other => Err(Error::unsupported(format!(
                "prores: profile code {other} not supported (only 422 Proxy/LT/Standard)"
            ))),
        }
    }

    /// Macroblock-FourCC for this profile as would appear in a `.mov`
    /// `VisualSampleEntry`. Informational — the header stores a
    /// numeric code.
    pub fn fourcc(self) -> &'static [u8; 4] {
        match self {
            Profile::Proxy => b"apco",
            Profile::Lt => b"apcs",
            Profile::Standard => b"apcn",
        }
    }
}

/// Parsed frame header view.
pub struct FrameHeader {
    pub frame_size: u32,
    pub width: u16,
    pub height: u16,
    pub profile: Profile,
    pub luma_qmat: [u8; 64],
    pub chroma_qmat: [u8; 64],
}

pub fn write_frame_header(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    profile: Profile,
    luma_qmat: &[u8; 64],
    chroma_qmat: &[u8; 64],
    total_frame_size: u32,
) {
    out.extend_from_slice(&total_frame_size.to_be_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(FRAME_HDR_SIZE as u16).to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // reserved
    out.extend_from_slice(CREATOR);
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.push(CHROMA_FMT_422);
    out.push(FLAG_LOAD_LUMA_QMAT | FLAG_LOAD_CHROMA_QMAT);
    out.push(profile as u8);
    out.push(0); // reserved
    out.extend_from_slice(luma_qmat);
    out.extend_from_slice(chroma_qmat);
    debug_assert_eq!(out.len(), FRAME_HDR_SIZE);
}

pub fn parse_frame_header(data: &[u8]) -> Result<(FrameHeader, &[u8])> {
    if data.len() < FRAME_HDR_SIZE {
        return Err(Error::invalid("prores: frame header truncated"));
    }
    let frame_size = u32::from_be_bytes(data[0..4].try_into().unwrap());
    if &data[4..8] != MAGIC {
        return Err(Error::invalid("prores: frame magic mismatch (not 'icpf')"));
    }
    let hdr_size = u16::from_be_bytes(data[8..10].try_into().unwrap()) as usize;
    if hdr_size != FRAME_HDR_SIZE {
        return Err(Error::invalid("prores: unexpected frame header size"));
    }
    let width = u16::from_be_bytes(data[16..18].try_into().unwrap());
    let height = u16::from_be_bytes(data[18..20].try_into().unwrap());
    if data[20] != CHROMA_FMT_422 {
        return Err(Error::unsupported(
            "prores: only 4:2:2 chroma format is supported",
        ));
    }
    let flags = data[21];
    let profile = Profile::from_code(data[22])?;
    if flags & FLAG_LOAD_LUMA_QMAT == 0 || flags & FLAG_LOAD_CHROMA_QMAT == 0 {
        return Err(Error::invalid(
            "prores: frame header must carry both quant matrices",
        ));
    }
    let mut luma_qmat = [0u8; 64];
    let mut chroma_qmat = [0u8; 64];
    luma_qmat.copy_from_slice(&data[24..88]);
    chroma_qmat.copy_from_slice(&data[88..152]);
    Ok((
        FrameHeader {
            frame_size,
            width,
            height,
            profile,
            luma_qmat,
            chroma_qmat,
        },
        &data[FRAME_HDR_SIZE..],
    ))
}

pub struct PictureHeader {
    pub slice_count: u16,
    pub log2_slice_mb_width: u8,
}

pub fn write_picture_header(
    out: &mut Vec<u8>,
    picture_size: u32,
    slice_count: u16,
    log2_slice_mb_width: u8,
) {
    out.push(PICTURE_HDR_SIZE as u8);
    out.extend_from_slice(&picture_size.to_be_bytes());
    out.extend_from_slice(&slice_count.to_be_bytes());
    out.push(log2_slice_mb_width);
}

pub fn parse_picture_header(data: &[u8]) -> Result<(PictureHeader, &[u8])> {
    if data.len() < PICTURE_HDR_SIZE {
        return Err(Error::invalid("prores: picture header truncated"));
    }
    let hdr_size = data[0] as usize;
    if hdr_size != PICTURE_HDR_SIZE {
        return Err(Error::invalid("prores: unexpected picture header size"));
    }
    let _picture_size = u32::from_be_bytes(data[1..5].try_into().unwrap());
    let slice_count = u16::from_be_bytes(data[5..7].try_into().unwrap());
    let log2_slice_mb_width = data[7];
    Ok((
        PictureHeader {
            slice_count,
            log2_slice_mb_width,
        },
        &data[PICTURE_HDR_SIZE..],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_roundtrip() {
        let luma = [4u8; 64];
        let mut chroma = [0u8; 64];
        for (i, c) in chroma.iter_mut().enumerate() {
            *c = i as u8;
        }
        let mut out = Vec::new();
        write_frame_header(&mut out, 1920, 1080, Profile::Standard, &luma, &chroma, 200);
        assert_eq!(out.len(), FRAME_HDR_SIZE);
        let (hdr, rest) = parse_frame_header(&out).unwrap();
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
        assert_eq!(hdr.profile, Profile::Standard);
        assert_eq!(hdr.luma_qmat, luma);
        assert_eq!(hdr.chroma_qmat, chroma);
        assert!(rest.is_empty());
    }

    #[test]
    fn picture_header_roundtrip() {
        let mut out = Vec::new();
        write_picture_header(&mut out, 4321, 12, 3);
        assert_eq!(out.len(), PICTURE_HDR_SIZE);
        let (hdr, rest) = parse_picture_header(&out).unwrap();
        assert_eq!(hdr.slice_count, 12);
        assert_eq!(hdr.log2_slice_mb_width, 3);
        assert!(rest.is_empty());
    }
}
