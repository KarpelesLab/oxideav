//! Theora Identification header (packet 0) parser. Spec §6.2.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Pixel format a.k.a. chroma subsampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 4:2:0 — chroma at half width and height.
    Yuv420 = 0,
    /// Reserved.
    Reserved = 1,
    /// 4:2:2 — chroma at half width, full height.
    Yuv422 = 2,
    /// 4:4:4 — chroma at full resolution.
    Yuv444 = 3,
}

impl PixelFormat {
    pub fn from_code(c: u32) -> Result<Self> {
        Ok(match c {
            0 => Self::Yuv420,
            2 => Self::Yuv422,
            3 => Self::Yuv444,
            _ => return Err(Error::invalid(format!("reserved pixel format code {c}"))),
        })
    }

    pub fn chroma_shift_x(&self) -> u32 {
        match self {
            Self::Yuv420 | Self::Yuv422 => 1,
            Self::Yuv444 => 0,
            Self::Reserved => 0,
        }
    }

    pub fn chroma_shift_y(&self) -> u32 {
        match self {
            Self::Yuv420 => 1,
            Self::Yuv422 | Self::Yuv444 => 0,
            Self::Reserved => 0,
        }
    }

    pub fn to_core(&self) -> oxideav_core::PixelFormat {
        match self {
            Self::Yuv420 | Self::Reserved => oxideav_core::PixelFormat::Yuv420P,
            Self::Yuv422 => oxideav_core::PixelFormat::Yuv422P,
            Self::Yuv444 => oxideav_core::PixelFormat::Yuv444P,
        }
    }
}

/// Parsed contents of the Identification header.
#[derive(Clone, Debug)]
pub struct Identification {
    pub vmaj: u8,
    pub vmin: u8,
    pub vrev: u8,
    /// Width of coded frame in macroblocks (16-pixel units).
    pub fmbw: u16,
    /// Height of coded frame in macroblocks.
    pub fmbh: u16,
    /// Picture (displayed) width in pixels.
    pub picw: u32,
    /// Picture height in pixels.
    pub pich: u32,
    /// Picture X offset in pixels from frame origin.
    pub picx: u32,
    /// Picture Y offset in pixels from frame origin.
    pub picy: u32,
    pub frn: u32,
    pub frd: u32,
    pub parn: u32,
    pub pard: u32,
    pub cs: u8,
    pub nombr: u32,
    pub qual: u8,
    pub kfgshift: u8,
    pub pf: PixelFormat,
}

impl Identification {
    /// Coded frame width in pixels (always a multiple of 16).
    pub fn frame_width(&self) -> u32 {
        self.fmbw as u32 * 16
    }

    /// Coded frame height in pixels (always a multiple of 16).
    pub fn frame_height(&self) -> u32 {
        self.fmbh as u32 * 16
    }
}

pub fn parse_identification_header(packet: &[u8]) -> Result<Identification> {
    if packet.len() < 7 {
        return Err(Error::invalid("Theora identification header too short"));
    }
    if packet[0] != 0x80 || &packet[1..7] != b"theora" {
        return Err(Error::invalid("not a Theora identification header"));
    }
    let mut br = BitReader::new(&packet[7..]);
    let vmaj = br.read_u32(8)? as u8;
    let vmin = br.read_u32(8)? as u8;
    let vrev = br.read_u32(8)? as u8;
    if vmaj != 3 {
        return Err(Error::unsupported(format!(
            "Theora major version {vmaj} (expected 3)"
        )));
    }
    let fmbw = br.read_u32(16)? as u16;
    let fmbh = br.read_u32(16)? as u16;
    let picw = br.read_u32(24)?;
    let pich = br.read_u32(24)?;
    let picx = br.read_u32(8)?;
    let picy = br.read_u32(8)?;
    let frn = br.read_u32(32)?;
    let frd = br.read_u32(32)?;
    let parn = br.read_u32(24)?;
    let pard = br.read_u32(24)?;
    let cs = br.read_u32(8)? as u8;
    let nombr = br.read_u32(24)?;
    let qual = br.read_u32(6)? as u8;
    let kfgshift = br.read_u32(5)? as u8;
    let pf_code = br.read_u32(2)?;
    let _reserved = br.read_u32(3)?;

    let pf = PixelFormat::from_code(pf_code)?;

    if fmbw == 0 || fmbh == 0 {
        return Err(Error::invalid("Theora: zero frame size"));
    }

    Ok(Identification {
        vmaj,
        vmin,
        vrev,
        fmbw,
        fmbh,
        picw,
        pich,
        picx,
        picy,
        frn,
        frd,
        parn,
        pard,
        cs,
        nombr,
        qual,
        kfgshift,
        pf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short() {
        assert!(parse_identification_header(b"").is_err());
        assert!(parse_identification_header(&[0x80]).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let p = [0x00u8, b't', b'h', b'e', b'o', b'r', b'a', 3, 2, 1];
        assert!(parse_identification_header(&p).is_err());
    }
}
