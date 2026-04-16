//! Linear-scan VLC decoder shared by the Annex B tables.
//!
//! Mirrors the structure used in `oxideav-mpeg1video::vlc` — the tables are
//! small enough that a walk per symbol is perfectly fast for textbook-grade
//! decode and keeps each entry obvious to audit against the spec.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// One entry in a VLC table. `code` occupies the low `bits` bits MSB-first.
#[derive(Clone, Copy, Debug)]
pub struct VlcEntry<T: Copy> {
    pub code: u32,
    pub bits: u8,
    pub value: T,
}

impl<T: Copy> VlcEntry<T> {
    pub const fn new(bits: u8, code: u32, value: T) -> Self {
        Self { code, bits, value }
    }
}

/// Decode one symbol using linear scan over `table`.
pub fn decode<T: Copy>(br: &mut BitReader<'_>, table: &[VlcEntry<T>]) -> Result<T> {
    let max_bits = table.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    if max_bits == 0 {
        return Err(Error::invalid("mpeg4 vlc: empty table"));
    }
    let remaining = br.bits_remaining() as u32;
    let peek_bits = max_bits.min(remaining);
    if peek_bits == 0 {
        return Err(Error::invalid("mpeg4 vlc: no bits available"));
    }
    let peeked = br.peek_u32(peek_bits)?;
    let peeked_full = peeked << (max_bits - peek_bits);
    for e in table {
        if (e.bits as u32) > peek_bits {
            continue;
        }
        let shift = max_bits - e.bits as u32;
        let prefix = peeked_full >> shift;
        if prefix == e.code {
            br.consume(e.bits as u32)?;
            return Ok(e.value);
        }
    }
    Err(Error::invalid("mpeg4 vlc: no matching codeword"))
}
