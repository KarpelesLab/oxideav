//! IFF 85 chunk reader (EA IFF 85, 1985).
//!
//! Each chunk is `[4-byte FourCC id][4-byte BE size][size bytes of data]`
//! with a single pad byte following if `size` is odd. Group chunks (`FORM`,
//! `LIST`, `CAT `) reserve the first 4 bytes of their data for a "form
//! type" and then contain nested chunks.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{Error, Result};

/// FourCC constants for the three group chunk types.
pub const GROUP_FORM: [u8; 4] = *b"FORM";
pub const GROUP_LIST: [u8; 4] = *b"LIST";
pub const GROUP_CAT: [u8; 4] = *b"CAT ";

/// Header of a single IFF chunk.
#[derive(Clone, Copy, Debug)]
pub struct ChunkHeader {
    pub id: [u8; 4],
    pub size: u32,
}

impl ChunkHeader {
    pub fn id_str(&self) -> &str {
        std::str::from_utf8(&self.id).unwrap_or("????")
    }

    /// Number of bytes to advance past the chunk body including any pad byte.
    pub fn padded_size(&self) -> u64 {
        (self.size as u64) + (self.size & 1) as u64
    }

    pub fn is_group(&self) -> bool {
        matches!(self.id, GROUP_FORM | GROUP_LIST | GROUP_CAT)
    }
}

/// Read a single chunk header, returning `Ok(None)` at clean EOF.
pub fn read_chunk_header<R: Read + ?Sized>(r: &mut R) -> Result<Option<ChunkHeader>> {
    let mut buf = [0u8; 8];
    let mut got = 0;
    while got < 8 {
        match r.read(&mut buf[got..]) {
            Ok(0) => {
                return if got == 0 {
                    Ok(None)
                } else {
                    Err(Error::invalid("IFF: truncated chunk header"))
                };
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let id = [buf[0], buf[1], buf[2], buf[3]];
    let size = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Ok(Some(ChunkHeader { id, size }))
}

/// Read the 4-byte form-type identifier at the start of a group chunk's body.
pub fn read_form_type<R: Read + ?Sized>(r: &mut R) -> Result<[u8; 4]> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// Read the entire body of a chunk (excluding the pad byte).
pub fn read_body<R: Read + ?Sized>(r: &mut R, header: &ChunkHeader) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; header.size as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Skip a chunk's body and its pad byte.
pub fn skip_chunk_body<R: Seek + ?Sized>(r: &mut R, header: &ChunkHeader) -> Result<()> {
    let n = header.padded_size();
    if n > 0 {
        r.seek(SeekFrom::Current(n as i64))?;
    }
    Ok(())
}

/// Skip the pad byte after a fully-consumed chunk body (if `size` is odd).
pub fn skip_pad<R: Seek + ?Sized>(r: &mut R, header: &ChunkHeader) -> Result<()> {
    if header.size & 1 == 1 {
        r.seek(SeekFrom::Current(1))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn padded_size_even_odd() {
        let a = ChunkHeader {
            id: *b"BODY",
            size: 10,
        };
        assert_eq!(a.padded_size(), 10);
        let b = ChunkHeader {
            id: *b"NAME",
            size: 9,
        };
        assert_eq!(b.padded_size(), 10);
    }

    #[test]
    fn read_chunk_header_parses_bytes() {
        let bytes = [b'V', b'H', b'D', b'R', 0, 0, 0, 20];
        let mut cur = Cursor::new(&bytes[..]);
        let h = read_chunk_header(&mut cur).unwrap().unwrap();
        assert_eq!(&h.id, b"VHDR");
        assert_eq!(h.size, 20);
    }
}
