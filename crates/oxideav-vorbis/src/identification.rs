//! Vorbis Identification header parsing (Vorbis I, §4.2.2).

use oxideav_core::{Error, Result};

/// Parsed contents of the first packet of a Vorbis logical bitstream.
#[derive(Clone, Debug)]
pub struct Identification {
    pub vorbis_version: u32,
    pub audio_channels: u8,
    pub audio_sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    pub blocksize_0: u8,
    pub blocksize_1: u8,
}

pub fn parse_identification_header(packet: &[u8]) -> Result<Identification> {
    if packet.len() < 30 {
        return Err(Error::invalid("Vorbis identification header too short"));
    }
    if packet[0] != 0x01 || &packet[1..7] != b"vorbis" {
        return Err(Error::invalid("not a Vorbis identification header"));
    }
    let vorbis_version =
        u32::from_le_bytes(packet[7..11].try_into().expect("4 bytes"));
    if vorbis_version != 0 {
        return Err(Error::unsupported(format!(
            "unsupported Vorbis version {vorbis_version}"
        )));
    }
    let audio_channels = packet[11];
    let audio_sample_rate =
        u32::from_le_bytes(packet[12..16].try_into().expect("4 bytes"));
    let bitrate_maximum =
        i32::from_le_bytes(packet[16..20].try_into().expect("4 bytes"));
    let bitrate_nominal =
        i32::from_le_bytes(packet[20..24].try_into().expect("4 bytes"));
    let bitrate_minimum =
        i32::from_le_bytes(packet[24..28].try_into().expect("4 bytes"));
    let blocksizes = packet[28];
    let blocksize_0 = blocksizes & 0x0F;
    let blocksize_1 = (blocksizes >> 4) & 0x0F;
    if packet[29] & 0x01 == 0 {
        return Err(Error::invalid("Vorbis ID header framing bit unset"));
    }
    if audio_channels == 0 || audio_sample_rate == 0 {
        return Err(Error::invalid("Vorbis ID header has zero channels or rate"));
    }
    Ok(Identification {
        vorbis_version,
        audio_channels,
        audio_sample_rate,
        bitrate_maximum,
        bitrate_nominal,
        bitrate_minimum,
        blocksize_0,
        blocksize_1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short() {
        assert!(parse_identification_header(b"").is_err());
        assert!(parse_identification_header(&[0x01]).is_err());
    }

    #[test]
    fn parses_minimal_valid_header() {
        let mut p = vec![0u8; 30];
        p[0] = 0x01;
        p[1..7].copy_from_slice(b"vorbis");
        // version = 0 (already)
        p[11] = 2; // channels
        p[12..16].copy_from_slice(&44_100u32.to_le_bytes());
        p[20..24].copy_from_slice(&192_000i32.to_le_bytes());
        p[28] = (8u8 << 4) | 6; // blocksize_1=8, blocksize_0=6
        p[29] = 1; // framing
        let id = parse_identification_header(&p).unwrap();
        assert_eq!(id.audio_channels, 2);
        assert_eq!(id.audio_sample_rate, 44_100);
        assert_eq!(id.bitrate_nominal, 192_000);
        assert_eq!(id.blocksize_0, 6);
        assert_eq!(id.blocksize_1, 8);
    }
}
