//! EBML (Extensible Binary Meta Language) primitives.
//!
//! Reference: <https://www.rfc-editor.org/rfc/rfc8794.html>
//!
//! - **Variable-length integers (VINTs)** encode an unsigned value with a
//!   leading-zeros prefix that signals the byte width. The first byte's MSB
//!   determines width: `1xxxxxxx` is 1 byte, `01xxxxxx ...` is 2 bytes, etc.,
//!   up to 8 bytes (`00000001 ...`). Element IDs use the same encoding but
//!   keep the marker bits as part of the ID; element sizes strip the marker.
//! - **Element**: a `(id, size, data)` triple. Master elements contain other
//!   elements concatenated.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{Error, Result};

/// "Unknown size" marker used in streamable Segment headers (all payload bits 1).
pub const VINT_UNKNOWN_SIZE: u64 = u64::MAX;

/// Read an EBML VINT from `r`. Returns the parsed value and the number of
/// bytes consumed. If `keep_marker` is true, the leading 1 bit of the size
/// prefix is preserved (used for element IDs); otherwise it's stripped.
pub fn read_vint(r: &mut dyn Read, keep_marker: bool) -> Result<(u64, usize)> {
    let mut first = [0u8; 1];
    r.read_exact(&mut first)?;
    let b0 = first[0];
    if b0 == 0 {
        return Err(Error::invalid("EBML VINT: invalid leading byte 0x00"));
    }
    let len = (b0.leading_zeros() + 1) as usize;
    if len > 8 {
        return Err(Error::invalid("EBML VINT: width > 8 bytes"));
    }
    let mut value: u64 = if keep_marker {
        b0 as u64
    } else {
        (b0 & ((1u8 << (8 - len)) - 1)) as u64
    };
    let mut buf = [0u8; 8];
    let extra = len - 1;
    if extra > 0 {
        r.read_exact(&mut buf[..extra])?;
        for i in 0..extra {
            value = (value << 8) | (buf[i] as u64);
        }
    }
    // Detect the "unknown size" sentinel: all-payload-ones value.
    if !keep_marker && len <= 8 {
        let payload_bits = (8 - len) as u32 + 8 * extra as u32;
        let all_ones = if payload_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << payload_bits) - 1
        };
        if value == all_ones {
            return Ok((VINT_UNKNOWN_SIZE, len));
        }
    }
    Ok((value, len))
}

/// Encode `value` as a VINT, choosing the smallest valid width if `min_width`
/// is 0, or padding to at least `min_width` bytes otherwise. Returns the
/// encoded bytes.
pub fn write_vint(value: u64, min_width: u8) -> Vec<u8> {
    if value == VINT_UNKNOWN_SIZE {
        // 0xFF encodes "unknown size" in 1 byte.
        return vec![0xFF];
    }
    let mut width = min_width.max(1);
    loop {
        let payload_bits = (8 - width as u32) + 8 * (width as u32 - 1);
        let all_ones = if payload_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << payload_bits) - 1
        };
        // Reject the all-ones case (that's the unknown-size sentinel).
        if value < all_ones {
            break;
        }
        width += 1;
        if width > 8 {
            panic!("EBML VINT value too large to encode");
        }
    }
    let mut out = vec![0u8; width as usize];
    // Set marker bit at top of byte 0.
    out[0] = 1u8 << (8 - width);
    let mut v = value;
    for i in (0..width as usize).rev() {
        out[i] |= (v & 0xFF) as u8;
        v >>= 8;
    }
    out
}

/// Encode an element ID with its marker preserved (IDs are stored with the
/// marker bit included). Width is inferred from the ID's high byte.
pub fn write_element_id(id: u32) -> Vec<u8> {
    // ID layout: 1, 2, 3, or 4 bytes. The width equals the position of the
    // top set bit divided by 8 + 1, but specifically determined by leading zeros.
    let bytes = if id < 0x100 {
        1
    } else if id < 0x10000 {
        2
    } else if id < 0x1000000 {
        3
    } else {
        4
    };
    let mut out = Vec::with_capacity(bytes);
    for i in (0..bytes).rev() {
        out.push(((id >> (i * 8)) & 0xFF) as u8);
    }
    out
}

/// Header of an EBML element, fully read.
#[derive(Clone, Debug)]
pub struct ElementHeader {
    pub id: u32,
    /// Payload size; `VINT_UNKNOWN_SIZE` means "until parent ends".
    pub size: u64,
    /// Total bytes consumed for the header (id + size).
    pub header_len: usize,
}

/// Read an element header — VINT id (with marker) followed by VINT size (no marker).
pub fn read_element_header(r: &mut dyn Read) -> Result<ElementHeader> {
    let (id, id_len) = read_vint(r, true)?;
    if id > u32::MAX as u64 {
        return Err(Error::invalid("EBML: element id exceeds 32 bits"));
    }
    let (size, size_len) = read_vint(r, false)?;
    Ok(ElementHeader {
        id: id as u32,
        size,
        header_len: id_len + size_len,
    })
}

/// Read `n` bytes as a big-endian unsigned integer (1..=8 bytes).
pub fn read_uint(r: &mut dyn Read, n: usize) -> Result<u64> {
    if n > 8 {
        return Err(Error::invalid("EBML uint > 8 bytes"));
    }
    if n == 0 {
        return Ok(0);
    }
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf[..n])?;
    let mut v = 0u64;
    for i in 0..n {
        v = (v << 8) | (buf[i] as u64);
    }
    Ok(v)
}

pub fn read_int(r: &mut dyn Read, n: usize) -> Result<i64> {
    if n == 0 {
        return Ok(0);
    }
    let raw = read_uint(r, n)?;
    let shift = 64 - 8 * n as u32;
    Ok(((raw << shift) as i64) >> shift)
}

pub fn read_float(r: &mut dyn Read, n: usize) -> Result<f64> {
    match n {
        0 => Ok(0.0),
        4 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(f32::from_be_bytes(buf) as f64)
        }
        8 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(f64::from_be_bytes(buf))
        }
        _ => Err(Error::invalid(format!("EBML float must be 4 or 8 bytes (got {n})"))),
    }
}

pub fn read_string(r: &mut dyn Read, n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    // Trim trailing NULs (common in MKV strings).
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).map_err(|e| Error::invalid(format!("EBML string not UTF-8: {e}")))
}

pub fn read_bytes(r: &mut dyn Read, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Skip `n` bytes from a seekable reader.
pub fn skip<R: Seek + ?Sized>(r: &mut R, n: u64) -> Result<()> {
    if n > 0 {
        r.seek(SeekFrom::Current(n as i64))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn vint_round_trip_small() {
        for v in [0u64, 1, 126, 127, 128, 255, 16_000, 1_000_000, 1_234_567_890].iter() {
            let bytes = write_vint(*v, 0);
            let mut c = Cursor::new(&bytes);
            let (got, len) = read_vint(&mut c, false).unwrap();
            assert_eq!(got, *v, "v={v}");
            assert_eq!(len, bytes.len());
        }
    }

    #[test]
    fn vint_known_widths() {
        // 1-byte:  v=0   → 0x80
        assert_eq!(write_vint(0, 0), vec![0x80]);
        // 1-byte: v=126 → 0xFE  (127 is the all-ones unknown-size sentinel)
        assert_eq!(write_vint(126, 0), vec![0xFE]);
        // 2-byte: v=127 → 0x40 0x7F  (must spill into 2 bytes to avoid sentinel)
        assert_eq!(write_vint(127, 0), vec![0x40, 0x7F]);
    }

    #[test]
    fn id_round_trip() {
        // EBML root ID = 0x1A45DFA3 → 4 bytes preserving marker.
        let bytes = write_element_id(0x1A45DFA3);
        assert_eq!(bytes, vec![0x1A, 0x45, 0xDF, 0xA3]);
        let mut c = Cursor::new(&bytes);
        let (got, len) = read_vint(&mut c, true).unwrap();
        assert_eq!(got as u32, 0x1A45DFA3);
        assert_eq!(len, 4);
    }

    #[test]
    fn unknown_size_sentinel() {
        let mut c = Cursor::new(&[0xFFu8]);
        let (v, _) = read_vint(&mut c, false).unwrap();
        assert_eq!(v, VINT_UNKNOWN_SIZE);
    }
}
