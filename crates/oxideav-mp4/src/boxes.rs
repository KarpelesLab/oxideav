//! ISO Base Media File Format box header reader (ISO/IEC 14496-12).
//!
//! A box has a 4-byte big-endian size field followed by a 4-byte FourCC type.
//! - `size == 1` → the *next* 8 bytes are a 64-bit large size.
//! - `size == 0` → box extends to end of file.
//!
//! All multi-byte integers in MP4 are big-endian.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{Error, Result};

/// Decoded box header.
#[derive(Clone, Copy, Debug)]
pub struct BoxHeader {
    /// FourCC type, as a u32 with the 4 ASCII characters in big-endian order.
    pub fourcc: [u8; 4],
    /// Total box size in bytes (including the header itself). `None` means
    /// "rest of file" (input size==0).
    pub total_size: Option<u64>,
    /// Bytes consumed by the header (8 or 16).
    pub header_len: u64,
}

impl BoxHeader {
    pub fn type_str(&self) -> &str {
        std::str::from_utf8(&self.fourcc).unwrap_or("????")
    }

    pub fn payload_size(&self) -> Option<u64> {
        self.total_size.map(|t| t - self.header_len)
    }
}

pub fn read_box_header<R: Read + ?Sized>(r: &mut R) -> Result<Option<BoxHeader>> {
    let mut hdr = [0u8; 8];
    let mut got = 0;
    while got < 8 {
        match r.read(&mut hdr[got..]) {
            Ok(0) => {
                if got == 0 {
                    return Ok(None);
                } else {
                    return Err(Error::invalid("MP4: truncated box header"));
                }
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let size32 = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    let mut fourcc = [0u8; 4];
    fourcc.copy_from_slice(&hdr[4..8]);
    let (total_size, header_len) = match size32 {
        0 => (None, 8u64),
        1 => {
            let mut ext = [0u8; 8];
            r.read_exact(&mut ext)?;
            let large = u64::from_be_bytes(ext);
            (Some(large), 16u64)
        }
        n => (Some(n as u64), 8u64),
    };
    Ok(Some(BoxHeader {
        fourcc,
        total_size,
        header_len,
    }))
}

/// Read the full payload of a box as bytes. Fails if the box size is unknown.
pub fn read_box_body<R: Read + ?Sized>(r: &mut R, h: &BoxHeader) -> Result<Vec<u8>> {
    let payload = h
        .payload_size()
        .ok_or_else(|| Error::invalid("MP4: cannot read open-ended box body"))?;
    let mut buf = vec![0u8; payload as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Skip the payload of a box in a seekable reader.
pub fn skip_box_body<R: Seek + ?Sized>(r: &mut R, h: &BoxHeader) -> Result<()> {
    if let Some(payload) = h.payload_size() {
        if payload > 0 {
            r.seek(SeekFrom::Current(payload as i64))?;
        }
    } else {
        // "rest of file" — seek to end.
        r.seek(SeekFrom::End(0))?;
    }
    Ok(())
}

/// Convert a 4-char literal into a FourCC byte array.
pub const fn fourcc(s: &str) -> [u8; 4] {
    let b = s.as_bytes();
    [b[0], b[1], b[2], b[3]]
}

// Common box types.
pub const FTYP: [u8; 4] = fourcc("ftyp");
pub const MOOV: [u8; 4] = fourcc("moov");
pub const MVHD: [u8; 4] = fourcc("mvhd");
pub const TRAK: [u8; 4] = fourcc("trak");
pub const TKHD: [u8; 4] = fourcc("tkhd");
pub const EDTS: [u8; 4] = fourcc("edts");
pub const MDIA: [u8; 4] = fourcc("mdia");
pub const MDHD: [u8; 4] = fourcc("mdhd");
pub const HDLR: [u8; 4] = fourcc("hdlr");
pub const MINF: [u8; 4] = fourcc("minf");
pub const DINF: [u8; 4] = fourcc("dinf");
pub const STBL: [u8; 4] = fourcc("stbl");
pub const STSD: [u8; 4] = fourcc("stsd");
pub const STTS: [u8; 4] = fourcc("stts");
pub const STSS: [u8; 4] = fourcc("stss");
pub const STSC: [u8; 4] = fourcc("stsc");
pub const STSZ: [u8; 4] = fourcc("stsz");
pub const STZ2: [u8; 4] = fourcc("stz2");
pub const STCO: [u8; 4] = fourcc("stco");
pub const CO64: [u8; 4] = fourcc("co64");
pub const MDAT: [u8; 4] = fourcc("mdat");
pub const FREE: [u8; 4] = fourcc("free");
pub const SKIP: [u8; 4] = fourcc("skip");
pub const UDTA: [u8; 4] = fourcc("udta");
pub const META: [u8; 4] = fourcc("meta");
pub const ILST: [u8; 4] = fourcc("ilst");
pub const DATA: [u8; 4] = fourcc("data");

// Handler types.
pub const HANDLER_SOUN: [u8; 4] = fourcc("soun");
pub const HANDLER_VIDE: [u8; 4] = fourcc("vide");
