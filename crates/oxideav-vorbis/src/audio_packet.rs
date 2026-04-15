//! Audio packet decoding skeleton.
//!
//! The current scope is **partial**: this module wires up the audio packet
//! header parsing (mode lookup, blocksize selection, window flags) and
//! per-channel floor-1 amplitude extraction. It does **not** yet perform
//! residue decoding, channel coupling, IMDCT, windowing, or overlap-add —
//! those are the next chunk of the Vorbis decoder.
//!
//! A full reference implementation is Vorbis I §4.3.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::floor::{decode_floor_packet, Floor1Decoded};
use crate::identification::Identification;
use crate::setup::Setup;

/// Result of partial audio-packet decoding: header metadata + per-channel
/// floor decodes. Callers that just want to validate setup-header correctness
/// can stop here.
#[derive(Clone, Debug)]
pub struct PartialAudioPacket {
    pub mode_index: u8,
    pub blockflag: bool,
    pub previous_window_flag: bool,
    pub next_window_flag: bool,
    pub blocksize: u32,
    /// Per-channel floor decode (or `Floor1Decoded { unused: true, ... }` for
    /// channels whose floor packet says the channel is silent this block).
    pub floors: Vec<Floor1Decoded>,
}

pub fn decode_audio_packet_partial(
    packet: &[u8],
    id: &Identification,
    setup: &Setup,
    blocksize_short: u32,
    blocksize_long: u32,
) -> Result<PartialAudioPacket> {
    let mut br = BitReader::new(packet);
    let header_bit = br.read_bit()?;
    if header_bit {
        return Err(Error::invalid("Vorbis audio packet: type bit set"));
    }
    let mode_bits = ilog((setup.modes.len() as u32 - 1).max(1));
    let mode_index = br.read_u32(mode_bits)? as u8;
    if (mode_index as usize) >= setup.modes.len() {
        return Err(Error::invalid("Vorbis audio packet: invalid mode index"));
    }
    let mode = &setup.modes[mode_index as usize];
    let blocksize = if mode.blockflag {
        blocksize_long
    } else {
        blocksize_short
    };
    let (prev_flag, next_flag) = if mode.blockflag {
        (br.read_bit()?, br.read_bit()?)
    } else {
        (false, false)
    };

    let mapping = &setup.mappings[mode.mapping as usize];
    let n_channels = id.audio_channels as usize;
    let mut floors = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        let submap = if mapping.submaps > 1 {
            mapping.mux[ch]
        } else {
            0
        };
        let floor_idx = mapping.submap_floor[submap as usize] as usize;
        let floor = &setup.floors[floor_idx];
        let decoded = decode_floor_packet(floor, &setup.codebooks, &mut br)?;
        floors.push(decoded);
    }

    Ok(PartialAudioPacket {
        mode_index,
        blockflag: mode.blockflag,
        previous_window_flag: prev_flag,
        next_window_flag: next_flag,
        blocksize,
        floors,
    })
}

fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}
