//! Vorbis setup header (packet 3) parser.
//!
//! Reference: Vorbis I §4.2.4.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::codebook::{parse_codebook, Codebook};

#[derive(Clone, Debug)]
pub struct Setup {
    pub codebooks: Vec<Codebook>,
    pub floors: Vec<Floor>,
    pub residues: Vec<Residue>,
    pub mappings: Vec<Mapping>,
    pub modes: Vec<Mode>,
}

/// Parse the entire setup header from the given packet bytes (must start with
/// the 0x05 + "vorbis" magic).
pub fn parse_setup(packet: &[u8], audio_channels: u8) -> Result<Setup> {
    if packet.len() < 7 || packet[0] != 0x05 || &packet[1..7] != b"vorbis" {
        return Err(Error::invalid("Vorbis setup header missing magic"));
    }
    let mut br = BitReader::new(&packet[7..]);

    // Codebooks.
    let codebook_count = (br.read_u32(8)? + 1) as usize;
    let mut codebooks = Vec::with_capacity(codebook_count);
    for _ in 0..codebook_count {
        codebooks.push(parse_codebook(&mut br)?);
    }

    // Time domain transforms (legacy — must be all zero).
    let time_count = (br.read_u32(6)? + 1) as usize;
    for _ in 0..time_count {
        let v = br.read_u32(16)?;
        if v != 0 {
            return Err(Error::invalid(format!(
                "Vorbis: nonzero time-domain placeholder {v}"
            )));
        }
    }

    // Floors.
    let floor_count = (br.read_u32(6)? + 1) as usize;
    let mut floors = Vec::with_capacity(floor_count);
    for _ in 0..floor_count {
        let kind = br.read_u32(16)? as u16;
        match kind {
            0 => floors.push(parse_floor0(&mut br)?),
            1 => floors.push(parse_floor1(&mut br, codebooks.len())?),
            other => {
                return Err(Error::unsupported(format!(
                    "Vorbis: unsupported floor type {other}"
                )));
            }
        }
    }

    // Residues.
    let residue_count = (br.read_u32(6)? + 1) as usize;
    let mut residues = Vec::with_capacity(residue_count);
    for _ in 0..residue_count {
        let kind = br.read_u32(16)? as u16;
        if kind > 2 {
            return Err(Error::invalid(format!(
                "Vorbis: invalid residue type {kind}"
            )));
        }
        residues.push(parse_residue(&mut br, kind, codebooks.len())?);
    }

    // Mappings.
    let mapping_count = (br.read_u32(6)? + 1) as usize;
    let mut mappings = Vec::with_capacity(mapping_count);
    for _ in 0..mapping_count {
        let kind = br.read_u32(16)? as u16;
        if kind != 0 {
            return Err(Error::unsupported(format!(
                "Vorbis: unsupported mapping type {kind}"
            )));
        }
        mappings.push(parse_mapping(
            &mut br,
            audio_channels,
            floors.len(),
            residues.len(),
        )?);
    }

    // Modes.
    let mode_count = (br.read_u32(6)? + 1) as usize;
    let mut modes = Vec::with_capacity(mode_count);
    for _ in 0..mode_count {
        modes.push(parse_mode(&mut br, mappings.len())?);
    }

    let framing = br.read_bit()?;
    if !framing {
        return Err(Error::invalid("Vorbis setup header framing bit unset"));
    }

    Ok(Setup {
        codebooks,
        floors,
        residues,
        mappings,
        modes,
    })
}

// --- Floor ----------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Floor {
    Type0(Floor0),
    Type1(Floor1),
}

#[derive(Clone, Debug)]
pub struct Floor0 {
    pub order: u8,
    pub rate: u16,
    pub bark_map_size: u16,
    pub amplitude_bits: u8,
    pub amplitude_offset: u8,
    pub number_of_books: u8,
    pub book_list: Vec<u8>,
}

fn parse_floor0(br: &mut BitReader<'_>) -> Result<Floor> {
    let order = br.read_u32(8)? as u8;
    let rate = br.read_u32(16)? as u16;
    let bark_map_size = br.read_u32(16)? as u16;
    let amplitude_bits = br.read_u32(6)? as u8;
    let amplitude_offset = br.read_u32(8)? as u8;
    let number_of_books = (br.read_u32(4)? + 1) as u8;
    let mut book_list = Vec::with_capacity(number_of_books as usize);
    for _ in 0..number_of_books {
        book_list.push(br.read_u32(8)? as u8);
    }
    Ok(Floor::Type0(Floor0 {
        order,
        rate,
        bark_map_size,
        amplitude_bits,
        amplitude_offset,
        number_of_books,
        book_list,
    }))
}

#[derive(Clone, Debug, Default)]
pub struct Floor1 {
    pub partition_class_list: Vec<u8>,
    pub class_dimensions: Vec<u8>,
    pub class_subclasses: Vec<u8>,
    pub class_masterbook: Vec<u8>,
    pub class_subbook: Vec<Vec<i16>>,
    pub multiplier: u8,
    pub rangebits: u8,
    pub xlist: Vec<u32>,
}

fn parse_floor1(br: &mut BitReader<'_>, n_codebooks: usize) -> Result<Floor> {
    let partitions = br.read_u32(5)? as usize;
    let mut partition_class_list = Vec::with_capacity(partitions);
    let mut max_class: i16 = -1;
    for _ in 0..partitions {
        let c = br.read_u32(4)? as u8;
        if c as i16 > max_class {
            max_class = c as i16;
        }
        partition_class_list.push(c);
    }
    let n_classes = (max_class + 1) as usize;
    let mut class_dimensions = vec![0u8; n_classes];
    let mut class_subclasses = vec![0u8; n_classes];
    let mut class_masterbook = vec![0u8; n_classes];
    let mut class_subbook: Vec<Vec<i16>> = vec![Vec::new(); n_classes];
    for c in 0..n_classes {
        class_dimensions[c] = (br.read_u32(3)? + 1) as u8;
        class_subclasses[c] = br.read_u32(2)? as u8;
        if class_subclasses[c] != 0 {
            let mb = br.read_u32(8)? as u8;
            if mb as usize >= n_codebooks {
                return Err(Error::invalid("Vorbis floor1: master codebook out of range"));
            }
            class_masterbook[c] = mb;
        }
        let n_sub = 1u32 << class_subclasses[c];
        let mut subbook = Vec::with_capacity(n_sub as usize);
        for _ in 0..n_sub {
            let val = br.read_u32(8)? as i16 - 1;
            if val >= 0 && val as usize >= n_codebooks {
                return Err(Error::invalid("Vorbis floor1: sub-codebook out of range"));
            }
            subbook.push(val);
        }
        class_subbook[c] = subbook;
    }
    let multiplier = (br.read_u32(2)? + 1) as u8;
    let rangebits = br.read_u32(4)? as u8;
    // X-list: 2 implicit values (0 and 2^rangebits) plus per-partition entries.
    let mut xlist: Vec<u32> = Vec::new();
    xlist.push(0);
    xlist.push(1u32 << rangebits);
    for &c in &partition_class_list {
        let dim = class_dimensions[c as usize];
        for _ in 0..dim {
            xlist.push(br.read_u32(rangebits as u32)?);
        }
    }
    Ok(Floor::Type1(Floor1 {
        partition_class_list,
        class_dimensions,
        class_subclasses,
        class_masterbook,
        class_subbook,
        multiplier,
        rangebits,
        xlist,
    }))
}

// --- Residue --------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Residue {
    pub kind: u16,
    pub begin: u32,
    pub end: u32,
    pub partition_size: u32,
    pub classifications: u8,
    pub classbook: u8,
    pub cascade: Vec<u8>,
    pub books: Vec<[i16; 8]>,
}

fn parse_residue(br: &mut BitReader<'_>, kind: u16, n_codebooks: usize) -> Result<Residue> {
    let begin = br.read_u32(24)?;
    let end = br.read_u32(24)?;
    let partition_size = br.read_u32(24)? + 1;
    let classifications = (br.read_u32(6)? + 1) as u8;
    let classbook = br.read_u32(8)? as u8;
    if (classbook as usize) >= n_codebooks {
        return Err(Error::invalid("Vorbis residue: classbook out of range"));
    }
    let mut cascade = vec![0u8; classifications as usize];
    for c in 0..classifications as usize {
        let mut high_bits = 0u8;
        let low_bits = br.read_u32(3)? as u8;
        let bitflag = br.read_bit()?;
        if bitflag {
            high_bits = br.read_u32(5)? as u8;
        }
        cascade[c] = (high_bits << 3) | low_bits;
    }
    let mut books: Vec<[i16; 8]> = vec![[-1; 8]; classifications as usize];
    for c in 0..classifications as usize {
        for j in 0..8 {
            if (cascade[c] & (1 << j)) != 0 {
                let book = br.read_u32(8)? as i16 - 1;
                // Some encoders leave the book as -1 ("no book"); we keep that.
                if book >= 0 && (book as usize) >= n_codebooks {
                    return Err(Error::invalid("Vorbis residue book out of range"));
                }
                books[c][j] = book;
            }
        }
    }
    Ok(Residue {
        kind,
        begin,
        end,
        partition_size,
        classifications,
        classbook,
        cascade,
        books,
    })
}

// --- Mapping --------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Mapping {
    pub submaps: u8,
    /// Channel-coupling steps: each (magnitude_channel, angle_channel).
    pub coupling: Vec<(u8, u8)>,
    pub mux: Vec<u8>,
    pub submap_floor: Vec<u8>,
    pub submap_residue: Vec<u8>,
}

fn parse_mapping(
    br: &mut BitReader<'_>,
    audio_channels: u8,
    n_floors: usize,
    n_residues: usize,
) -> Result<Mapping> {
    let submaps = if br.read_bit()? {
        (br.read_u32(4)? + 1) as u8
    } else {
        1
    };
    let coupling_steps = if br.read_bit()? {
        (br.read_u32(8)? + 1) as usize
    } else {
        0
    };
    let coupling_field_bits = ilog((audio_channels as i32 - 1).max(0) as u32);
    let mut coupling = Vec::with_capacity(coupling_steps);
    for _ in 0..coupling_steps {
        let mag = br.read_u32(coupling_field_bits)? as u8;
        let ang = br.read_u32(coupling_field_bits)? as u8;
        if mag == ang || mag >= audio_channels || ang >= audio_channels {
            return Err(Error::invalid("Vorbis mapping: invalid coupling pair"));
        }
        coupling.push((mag, ang));
    }
    let _reserved = br.read_u32(2)?;
    let mut mux = vec![0u8; audio_channels as usize];
    if submaps > 1 {
        for i in 0..audio_channels as usize {
            mux[i] = br.read_u32(4)? as u8;
            if mux[i] >= submaps {
                return Err(Error::invalid("Vorbis mapping: mux out of range"));
            }
        }
    }
    let mut submap_floor = Vec::with_capacity(submaps as usize);
    let mut submap_residue = Vec::with_capacity(submaps as usize);
    for _ in 0..submaps {
        let _discard = br.read_u32(8)?;
        let f = br.read_u32(8)? as u8;
        let r = br.read_u32(8)? as u8;
        if (f as usize) >= n_floors {
            return Err(Error::invalid("Vorbis mapping: floor out of range"));
        }
        if (r as usize) >= n_residues {
            return Err(Error::invalid("Vorbis mapping: residue out of range"));
        }
        submap_floor.push(f);
        submap_residue.push(r);
    }
    Ok(Mapping {
        submaps,
        coupling,
        mux,
        submap_floor,
        submap_residue,
    })
}

// --- Mode -----------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Mode {
    pub blockflag: bool,
    pub windowtype: u16,
    pub transformtype: u16,
    pub mapping: u8,
}

fn parse_mode(br: &mut BitReader<'_>, n_mappings: usize) -> Result<Mode> {
    let blockflag = br.read_bit()?;
    let windowtype = br.read_u32(16)? as u16;
    let transformtype = br.read_u32(16)? as u16;
    let mapping = br.read_u32(8)? as u8;
    if windowtype != 0 {
        return Err(Error::invalid("Vorbis mode: nonzero windowtype"));
    }
    if transformtype != 0 {
        return Err(Error::invalid("Vorbis mode: nonzero transformtype"));
    }
    if (mapping as usize) >= n_mappings {
        return Err(Error::invalid("Vorbis mode: mapping out of range"));
    }
    Ok(Mode {
        blockflag,
        windowtype,
        transformtype,
        mapping,
    })
}

fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}
