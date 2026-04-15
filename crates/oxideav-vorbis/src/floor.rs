//! Floor 1 packet decoding and curve synthesis.
//!
//! Reference: Vorbis I §7.2.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::codebook::Codebook;
use crate::setup::{Floor, Floor1};

/// Bit width for the first two amplitude values (Y[0], Y[1]) per floor1
/// multiplier setting (Vorbis I §6.2.3 table).
fn amp_bits_for_multiplier(multiplier: u8) -> u32 {
    match multiplier {
        1 => 8, // range = 256
        2 => 7, // range = 128
        3 => 7, // range = 86 (still 7 bits)
        4 => 6, // range = 64
        _ => 8,
    }
}

/// Per-multiplier "step2 flag = unused" range as defined in Vorbis I
/// (multiplier values map to a fixed dB step size).
pub fn floor1_db_step(multiplier: u8) -> f32 {
    match multiplier {
        1 => 256.0,
        2 => 128.0,
        3 => 86.0,
        4 => 64.0,
        _ => 256.0,
    }
}

/// Decoded floor 1 amplitude vector + "is unused" flag.
#[derive(Clone, Debug)]
pub struct Floor1Decoded {
    pub unused: bool,
    /// Amplitude (Y) values, one per X-list entry. Empty if `unused`.
    pub y: Vec<i32>,
}

pub fn decode_floor1_packet(
    floor: &Floor1,
    codebooks: &[Codebook],
    br: &mut BitReader<'_>,
) -> Result<Floor1Decoded> {
    let nonzero = br.read_bit()?;
    if !nonzero {
        return Ok(Floor1Decoded {
            unused: true,
            y: Vec::new(),
        });
    }
    let amp_bits = amp_bits_for_multiplier(floor.multiplier);
    let mut y: Vec<i32> = Vec::with_capacity(floor.xlist.len());
    y.push(br.read_u32(amp_bits)? as i32);
    y.push(br.read_u32(amp_bits)? as i32);
    let mut offset = 2usize;
    for &class_idx in &floor.partition_class_list {
        let c = class_idx as usize;
        let cdim = floor.class_dimensions[c] as usize;
        let cbits = floor.class_subclasses[c] as u32;
        let csub = 1u32 << cbits;
        let cval = if cbits > 0 {
            let cb = &codebooks[floor.class_masterbook[c] as usize];
            cb.decode_scalar(br)?
        } else {
            0
        };
        for _j in 0..cdim {
            let book_index = floor.class_subbook[c][(cval & (csub - 1)) as usize];
            let v = if book_index >= 0 {
                let cb = &codebooks[book_index as usize];
                cb.decode_scalar(br)? as i32
            } else {
                0
            };
            y.push(v);
            offset += 1;
        }
        let _ = cval >> cbits; // shift consumed bits per spec; value not reused
    }
    if offset != floor.xlist.len() {
        return Err(Error::invalid(format!(
            "Vorbis floor1 decoded {} amplitudes, expected {}",
            offset,
            floor.xlist.len()
        )));
    }
    Ok(Floor1Decoded { unused: false, y })
}

/// Public entry point: decode a floor packet given its setup type.
pub fn decode_floor_packet(
    floor: &Floor,
    codebooks: &[Codebook],
    br: &mut BitReader<'_>,
) -> Result<Floor1Decoded> {
    match floor {
        Floor::Type1(f) => decode_floor1_packet(f, codebooks, br),
        Floor::Type0(_) => Err(Error::unsupported(
            "Vorbis floor 0 (LSP) decoding not implemented",
        )),
    }
}
