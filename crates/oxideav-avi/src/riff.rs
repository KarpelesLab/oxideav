//! RIFF chunk primitives.
//!
//! RIFF is a generic little-endian container format originally defined by
//! Microsoft and IBM. A RIFF file consists of nested chunks, each with an
//! 8-byte header:
//!
//! ```text
//! [4-byte FourCC id][4-byte LE size][size bytes of data]
//! ```
//!
//! Chunks whose `size` is odd are padded with a single pad byte so the next
//! chunk starts on an even boundary. The special list-chunk types `RIFF` and
//! `LIST` start their body with an extra 4-byte FourCC "form type" identifying
//! the list contents, and contain nested chunks after it.
//!
//! AVI uses only `RIFF` (top-level) and `LIST` (nested containers like `hdrl`,
//! `strl`, `movi`, `INFO`). AVI data is exclusively little-endian.
//!
//! Note that IFF 85 (Electronic Arts), which shares the conceptual shape, uses
//! big-endian sizes and `FORM`/`LIST`/`CAT ` groups instead. RIFF and IFF are
//! *not* the same and must not share a primitives module.

use std::io::{Read, Seek, SeekFrom, Write};

use oxideav_core::{Error, Result};

/// FourCC of a top-level RIFF chunk.
pub const RIFF: [u8; 4] = *b"RIFF";
/// FourCC of a nested list chunk.
pub const LIST: [u8; 4] = *b"LIST";
/// Form-type of an AVI file.
pub const AVI_FORM: [u8; 4] = *b"AVI ";

/// Header of a single RIFF chunk.
#[derive(Clone, Copy, Debug)]
pub struct ChunkHeader {
    pub id: [u8; 4],
    pub size: u32,
}

impl ChunkHeader {
    pub fn is_list(&self) -> bool {
        matches!(self.id, RIFF | LIST)
    }

    /// Number of bytes the body + pad byte consume.
    pub fn padded_size(&self) -> u64 {
        (self.size as u64) + (self.size & 1) as u64
    }
}

/// Read a single chunk header; `Ok(None)` at clean EOF.
pub fn read_chunk_header<R: Read + ?Sized>(r: &mut R) -> Result<Option<ChunkHeader>> {
    let mut buf = [0u8; 8];
    let mut got = 0;
    while got < 8 {
        match r.read(&mut buf[got..]) {
            Ok(0) => {
                return if got == 0 {
                    Ok(None)
                } else {
                    Err(Error::invalid("AVI: truncated chunk header"))
                };
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let id = [buf[0], buf[1], buf[2], buf[3]];
    let size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Ok(Some(ChunkHeader { id, size }))
}

/// Read the 4-byte form-type of a list chunk (`RIFF`/`LIST`).
pub fn read_form_type<R: Read + ?Sized>(r: &mut R) -> Result<[u8; 4]> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// Read a chunk body of exactly `size` bytes (no pad byte).
pub fn read_body<R: Read + ?Sized>(r: &mut R, size: u32) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; size as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Skip a chunk's body and any trailing pad byte.
pub fn skip_chunk<R: Seek + ?Sized>(r: &mut R, header: &ChunkHeader) -> Result<()> {
    let n = header.padded_size();
    if n > 0 {
        r.seek(SeekFrom::Current(n as i64))?;
    }
    Ok(())
}

/// Skip only the pad byte after a consumed body.
pub fn skip_pad<R: Seek + ?Sized>(r: &mut R, size: u32) -> Result<()> {
    if size & 1 == 1 {
        r.seek(SeekFrom::Current(1))?;
    }
    Ok(())
}

/// Write an 8-byte chunk header with the given id and size.
pub fn write_chunk_header<W: Write + ?Sized>(w: &mut W, id: &[u8; 4], size: u32) -> Result<()> {
    w.write_all(id)?;
    w.write_all(&size.to_le_bytes())?;
    Ok(())
}

/// Write a complete chunk (header + body), inserting a pad byte if `body.len()`
/// is odd. Size field is clamped at `u32::MAX`.
pub fn write_chunk<W: Write + ?Sized>(w: &mut W, id: &[u8; 4], body: &[u8]) -> Result<()> {
    if body.len() > u32::MAX as usize {
        return Err(Error::invalid("AVI: chunk body too large for 32-bit size"));
    }
    write_chunk_header(w, id, body.len() as u32)?;
    w.write_all(body)?;
    if body.len() & 1 == 1 {
        w.write_all(&[0])?;
    }
    Ok(())
}

/// Write a list chunk: `RIFF` or `LIST` header + 4-byte form type + body.
/// Pad byte appended if body length is odd.
pub fn write_list_chunk<W: Write + ?Sized>(
    w: &mut W,
    list_id: &[u8; 4],
    form_type: &[u8; 4],
    body: &[u8],
) -> Result<()> {
    let total = 4u64 + body.len() as u64;
    if total > u32::MAX as u64 {
        return Err(Error::invalid("AVI: list chunk body too large"));
    }
    write_chunk_header(w, list_id, total as u32)?;
    w.write_all(form_type)?;
    w.write_all(body)?;
    if body.len() & 1 == 1 {
        w.write_all(&[0])?;
    }
    Ok(())
}

/// Begin a streaming chunk, leaving the size field as a 0 placeholder. Returns
/// the absolute offset of the size field so the caller can patch it later.
pub fn begin_chunk<W: Write + Seek + ?Sized>(w: &mut W, id: &[u8; 4]) -> Result<u64> {
    w.write_all(id)?;
    let size_off = w.stream_position()?;
    w.write_all(&[0u8; 4])?;
    Ok(size_off)
}

/// Begin a streaming list chunk (`RIFF` or `LIST`) with the given form type,
/// leaving the size field as a 0 placeholder. Returns the offset of the size
/// field.
pub fn begin_list<W: Write + Seek + ?Sized>(
    w: &mut W,
    list_id: &[u8; 4],
    form_type: &[u8; 4],
) -> Result<u64> {
    let size_off = begin_chunk(w, list_id)?;
    w.write_all(form_type)?;
    Ok(size_off)
}

/// Patch a previously-reserved chunk size field to the correct value.
///
/// `size_off` is the offset returned by `begin_chunk`/`begin_list`. `cur_pos`
/// is the current writer position (i.e. just past the body of the chunk).
/// After patching, the writer is restored to `cur_pos`. If the body length is
/// odd a single pad byte is appended *before* patching (the pad itself is not
/// counted in the chunk size per the RIFF spec).
pub fn finish_chunk<W: Write + Seek + ?Sized>(w: &mut W, size_off: u64) -> Result<()> {
    let end_pos = w.stream_position()?;
    // body_size = everything written after the 4-byte size field.
    let body_size = end_pos
        .checked_sub(size_off + 4)
        .ok_or_else(|| Error::other("AVI: invalid chunk cursor"))?;
    if body_size > u32::MAX as u64 {
        return Err(Error::invalid("AVI: chunk grew past 32-bit size limit"));
    }
    // Pad byte if needed (and restore end_pos accordingly).
    let needs_pad = body_size & 1 == 1;
    if needs_pad {
        w.write_all(&[0])?;
    }
    let after_pad = if needs_pad { end_pos + 1 } else { end_pos };
    w.seek(SeekFrom::Start(size_off))?;
    w.write_all(&(body_size as u32).to_le_bytes())?;
    w.seek(SeekFrom::Start(after_pad))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_roundtrip() {
        let mut buf = Vec::new();
        write_chunk_header(&mut buf, b"TEST", 17).unwrap();
        assert_eq!(&buf, &[b'T', b'E', b'S', b'T', 17, 0, 0, 0]);
        let mut cur = Cursor::new(&buf[..]);
        let h = read_chunk_header(&mut cur).unwrap().unwrap();
        assert_eq!(&h.id, b"TEST");
        assert_eq!(h.size, 17);
    }

    #[test]
    fn chunk_pads_odd_body() {
        let mut buf = Vec::new();
        write_chunk(&mut buf, b"DATA", &[1, 2, 3]).unwrap();
        assert_eq!(buf.len(), 8 + 3 + 1);
        assert_eq!(buf.last(), Some(&0));
    }

    #[test]
    fn streaming_chunk_patches_size() {
        let mut v = Vec::new();
        {
            let mut w = Cursor::new(&mut v);
            let off = begin_chunk(&mut w, b"dmy ").unwrap();
            w.write_all(&[0xAA, 0xBB, 0xCC]).unwrap();
            finish_chunk(&mut w, off).unwrap();
        }
        // Expect: "dmy " + size=3 LE + body + pad
        assert_eq!(&v[..4], b"dmy ");
        assert_eq!(&v[4..8], &3u32.to_le_bytes());
        assert_eq!(&v[8..11], &[0xAA, 0xBB, 0xCC]);
        assert_eq!(v.len(), 12); // +pad
    }

    #[test]
    fn list_chunk_roundtrip() {
        let mut buf = Vec::new();
        write_list_chunk(&mut buf, b"RIFF", b"AVI ", b"hello").unwrap();
        let mut cur = Cursor::new(&buf[..]);
        let h = read_chunk_header(&mut cur).unwrap().unwrap();
        assert_eq!(&h.id, b"RIFF");
        assert_eq!(h.size as usize, 4 + 5);
        let form = read_form_type(&mut cur).unwrap();
        assert_eq!(&form, b"AVI ");
    }
}
