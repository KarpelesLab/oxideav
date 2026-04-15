//! GSM 06.10 frame unpacking.
//!
//! A standard GSM Full Rate frame carries 260 bits (packed into 33 bytes
//! with a 4-bit "magic" prefix of `0xD` at the very top of byte 0):
//!
//! ```text
//! 4-bit  magic (0xD) + 8 LAR coefficients:
//!   LAR1 6  LAR2 6  LAR3 5  LAR4 5  LAR5 4  LAR6 4  LAR7 3  LAR8 3
//! 4 sub-frames (40 samples each), per sub-frame:
//!   Nc  7   (LTP lag, 40..=120)
//!   bc  2   (LTP gain index)
//!   Mc  2   (RPE grid position)
//!   xmaxc 6 (block amplitude)
//!   13 × 3-bit RPE pulses
//! ```
//!
//! The Microsoft framing (`gsm_ms`) packs two standard frames (520 bits)
//! into 65 bytes and omits the 4-bit magic. We strip the magic for the
//! standard variant and treat each MS half as a standalone GSM frame.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// One RPE-LTP sub-frame: every GSM frame carries four of these.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubFrame {
    /// LTP lag, 40..=120 (7 bits unsigned in the bitstream).
    pub nc: u8,
    /// LTP gain index, 0..=3.
    pub bc: u8,
    /// RPE grid position, 0..=3.
    pub mc: u8,
    /// Block amplitude index, 0..=63.
    pub xmaxc: u8,
    /// 13 × 3-bit RPE pulses.
    pub xmc: [u8; 13],
}

/// One GSM Full Rate speech frame.
#[derive(Clone, Debug, Default)]
pub struct GsmFrame {
    /// 8 LAR coefficients in bit-packed form.
    pub larc: [u8; 8],
    pub sub: [SubFrame; 4],
}

/// Size of a standard GSM 06.10 framed payload (with 4-bit 0xD magic).
pub const FRAME_SIZE: usize = 33;
/// Size of an MS-framed GSM payload (two standard frames packed to 65 bytes).
pub const MS_FRAME_SIZE: usize = 65;

/// Parse a standard framing GSM payload (33 bytes starting with the 0xD magic).
pub fn parse_frame(data: &[u8]) -> Result<GsmFrame> {
    if data.len() != FRAME_SIZE {
        return Err(Error::invalid(format!(
            "GSM frame: expected {FRAME_SIZE} bytes, got {}",
            data.len()
        )));
    }
    let magic = (data[0] >> 4) & 0x0F;
    if magic != 0xD {
        return Err(Error::invalid(format!(
            "GSM frame: bad magic {magic:#x} (want 0xD)"
        )));
    }
    let mut br = BitReader::new(data);
    let _ = br.read_u16(4)?; // magic
    parse_bits(&mut br)
}

/// Parse an MS-framed payload (65 bytes = two packed standard frames, no magic).
pub fn parse_ms_pair(data: &[u8]) -> Result<[GsmFrame; 2]> {
    if data.len() != MS_FRAME_SIZE {
        return Err(Error::invalid(format!(
            "GSM-MS frame: expected {MS_FRAME_SIZE} bytes, got {}",
            data.len()
        )));
    }
    // The MS variant packs two 260-bit frames back-to-back. We iterate the
    // same 260-bit schedule twice, with the second starting at bit 260.
    let mut br = BitReader::new(data);
    let f0 = parse_bits(&mut br)?;
    let f1 = parse_bits(&mut br)?;
    Ok([f0, f1])
}

fn parse_bits(br: &mut BitReader<'_>) -> Result<GsmFrame> {
    // LAR bit widths per the spec:
    const LAR_BITS: [u32; 8] = [6, 6, 5, 5, 4, 4, 3, 3];
    let mut larc = [0u8; 8];
    for i in 0..8 {
        larc[i] = br.read_u16(LAR_BITS[i])? as u8;
    }
    let mut sub = [SubFrame::default(); 4];
    for s in &mut sub {
        s.nc = br.read_u16(7)? as u8;
        s.bc = br.read_u16(2)? as u8;
        s.mc = br.read_u16(2)? as u8;
        s.xmaxc = br.read_u16(6)? as u8;
        for p in &mut s.xmc {
            *p = br.read_u16(3)? as u8;
        }
    }
    Ok(GsmFrame { larc, sub })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitWriter;

    fn encode_frame(frame: &GsmFrame, include_magic: bool) -> Vec<u8> {
        let mut w = BitWriter::new();
        if include_magic {
            w.write(0xD, 4);
        }
        const LAR_BITS: [u32; 8] = [6, 6, 5, 5, 4, 4, 3, 3];
        for i in 0..8 {
            w.write(frame.larc[i] as u16, LAR_BITS[i]);
        }
        for s in &frame.sub {
            w.write(s.nc as u16, 7);
            w.write(s.bc as u16, 2);
            w.write(s.mc as u16, 2);
            w.write(s.xmaxc as u16, 6);
            for p in &s.xmc {
                w.write(*p as u16, 3);
            }
        }
        // Pad to byte boundary.
        while w.data.len() < FRAME_SIZE {
            w.data.push(0);
        }
        w.data
    }

    #[test]
    fn roundtrip_standard_framing() {
        let mut f = GsmFrame {
            larc: [0x2A, 0x15, 0x1F, 0x00, 0x0F, 0x0A, 0x05, 0x07],
            ..Default::default()
        };
        for (i, s) in f.sub.iter_mut().enumerate() {
            s.nc = (40 + i as u8 * 10) & 0x7F;
            s.bc = (i as u8) & 3;
            s.mc = ((i + 1) as u8) & 3;
            s.xmaxc = (20 + i as u8) & 0x3F;
            for (j, p) in s.xmc.iter_mut().enumerate() {
                *p = ((j + i) as u8) & 7;
            }
        }
        let bytes = encode_frame(&f, true);
        assert_eq!(bytes.len(), FRAME_SIZE);
        let parsed = parse_frame(&bytes).unwrap();
        assert_eq!(parsed.larc, f.larc);
        for i in 0..4 {
            assert_eq!(parsed.sub[i].nc, f.sub[i].nc);
            assert_eq!(parsed.sub[i].bc, f.sub[i].bc);
            assert_eq!(parsed.sub[i].mc, f.sub[i].mc);
            assert_eq!(parsed.sub[i].xmaxc, f.sub[i].xmaxc);
            assert_eq!(parsed.sub[i].xmc, f.sub[i].xmc);
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0; FRAME_SIZE];
        bytes[0] = 0x00;
        assert!(parse_frame(&bytes).is_err());
    }

    #[test]
    fn rejects_wrong_size() {
        assert!(parse_frame(&[0xD0; 10]).is_err());
        assert!(parse_ms_pair(&[0; 64]).is_err());
    }

    #[test]
    fn ms_pair_parses_two_frames() {
        // Build two frames back-to-back without magic, MSB-packed into 65 bytes.
        let mut f0 = GsmFrame {
            larc: [0x3F, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        f0.sub[0].nc = 120;
        let mut f1 = GsmFrame::default();
        f1.sub[3].xmaxc = 0x2A;
        let mut w = BitWriter::new();
        const LAR_BITS: [u32; 8] = [6, 6, 5, 5, 4, 4, 3, 3];
        for frame in [&f0, &f1] {
            for i in 0..8 {
                w.write(frame.larc[i] as u16, LAR_BITS[i]);
            }
            for s in &frame.sub {
                w.write(s.nc as u16, 7);
                w.write(s.bc as u16, 2);
                w.write(s.mc as u16, 2);
                w.write(s.xmaxc as u16, 6);
                for p in &s.xmc {
                    w.write(*p as u16, 3);
                }
            }
        }
        while w.data.len() < MS_FRAME_SIZE {
            w.data.push(0);
        }
        let [g0, g1] = parse_ms_pair(&w.data).unwrap();
        assert_eq!(g0.larc[0], 0x3F);
        assert_eq!(g1.larc[0], 0x00);
        assert_eq!(g0.sub[0].nc, 120);
        assert_eq!(g1.sub[3].xmaxc, 0x2A);
    }
}
