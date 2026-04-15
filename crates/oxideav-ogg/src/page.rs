//! Ogg page parsing and serialization (RFC 3533, §6).

use oxideav_core::{Error, Result};

use crate::crc;

/// Capture pattern at the start of every Ogg page.
pub const CAPTURE_PATTERN: [u8; 4] = *b"OggS";

/// Header type flag bits (page header byte 5).
pub mod flags {
    pub const CONTINUED: u8 = 0x01;
    pub const FIRST_PAGE: u8 = 0x02;
    pub const LAST_PAGE: u8 = 0x04;
}

/// One Ogg page after parsing.
#[derive(Clone, Debug)]
pub struct Page {
    /// Header type flags — combination of the constants in [`flags`].
    pub flags: u8,
    /// Granule position. Codec-defined; -1 means "no packets finish on this page".
    pub granule_position: i64,
    /// Logical bitstream identifier.
    pub serial: u32,
    /// Page sequence number within this logical bitstream (starts at 0).
    pub seq_no: u32,
    /// Lacing values — one byte per segment (0..=255).
    pub lacing: Vec<u8>,
    /// Concatenated segment bytes (length = sum of `lacing` values).
    pub data: Vec<u8>,
}

impl Page {
    pub fn is_continued(&self) -> bool {
        self.flags & flags::CONTINUED != 0
    }

    pub fn is_first(&self) -> bool {
        self.flags & flags::FIRST_PAGE != 0
    }

    pub fn is_last(&self) -> bool {
        self.flags & flags::LAST_PAGE != 0
    }

    /// Iterate over packet boundaries within this page.
    ///
    /// Returns `(payload_range, terminated)`: `payload_range` is a slice of
    /// `data`, and `terminated` is true when the packet's last segment is
    /// shorter than 255 bytes (meaning the packet ends inside this page).
    /// When false, the packet continues into the next page.
    pub fn packet_segments(&self) -> Vec<PacketSegment> {
        let mut out = Vec::new();
        let mut start_seg = 0usize;
        let mut data_off = 0usize;
        let mut packet_data_off = data_off;
        let mut packet_len = 0usize;
        for (i, &lv) in self.lacing.iter().enumerate() {
            packet_len += lv as usize;
            data_off += lv as usize;
            if lv < 255 {
                out.push(PacketSegment {
                    seg_range: start_seg..i + 1,
                    data: packet_data_off..packet_data_off + packet_len,
                    terminated: true,
                });
                start_seg = i + 1;
                packet_data_off = data_off;
                packet_len = 0;
            }
        }
        // Trailing partial packet (no terminator) — continues into next page.
        if start_seg < self.lacing.len() {
            out.push(PacketSegment {
                seg_range: start_seg..self.lacing.len(),
                data: packet_data_off..packet_data_off + packet_len,
                terminated: false,
            });
        }
        out
    }

    /// Serialize this page to bytes, computing the CRC.
    pub fn to_bytes(&self) -> Vec<u8> {
        assert!(self.lacing.len() <= 255, "Ogg page may carry at most 255 segments");
        let total_data: usize = self.lacing.iter().map(|&v| v as usize).sum();
        assert_eq!(self.data.len(), total_data, "page data length must match lacing sum");

        let mut buf = Vec::with_capacity(27 + self.lacing.len() + self.data.len());
        buf.extend_from_slice(&CAPTURE_PATTERN);
        buf.push(0); // version
        buf.push(self.flags);
        buf.extend_from_slice(&self.granule_position.to_le_bytes());
        buf.extend_from_slice(&self.serial.to_le_bytes());
        buf.extend_from_slice(&self.seq_no.to_le_bytes());
        let crc_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // checksum placeholder
        buf.push(self.lacing.len() as u8);
        buf.extend_from_slice(&self.lacing);
        buf.extend_from_slice(&self.data);

        let crc = crc::checksum(&buf);
        buf[crc_offset..crc_offset + 4].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Parse a single page from the start of `bytes`.
    ///
    /// Returns the parsed page and the number of bytes consumed. Validates
    /// the CRC; returns `Error::InvalidData` on mismatch.
    pub fn parse(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < 27 {
            return Err(Error::NeedMore);
        }
        if &bytes[0..4] != CAPTURE_PATTERN {
            return Err(Error::invalid("Ogg page missing 'OggS' capture pattern"));
        }
        if bytes[4] != 0 {
            return Err(Error::unsupported(format!(
                "unsupported Ogg page version {}",
                bytes[4]
            )));
        }
        let flags = bytes[5];
        let granule_position =
            i64::from_le_bytes(bytes[6..14].try_into().expect("8 bytes"));
        let serial = u32::from_le_bytes(bytes[14..18].try_into().expect("4 bytes"));
        let seq_no = u32::from_le_bytes(bytes[18..22].try_into().expect("4 bytes"));
        let claimed_crc = u32::from_le_bytes(bytes[22..26].try_into().expect("4 bytes"));
        let n_segs = bytes[26] as usize;
        let header_len = 27 + n_segs;
        if bytes.len() < header_len {
            return Err(Error::NeedMore);
        }
        let lacing = bytes[27..header_len].to_vec();
        let data_len: usize = lacing.iter().map(|&v| v as usize).sum();
        let total = header_len + data_len;
        if bytes.len() < total {
            return Err(Error::NeedMore);
        }
        let data = bytes[header_len..total].to_vec();

        // Validate CRC.
        let mut to_check = bytes[..total].to_vec();
        to_check[22..26].fill(0);
        let computed = crc::checksum(&to_check);
        if computed != claimed_crc {
            return Err(Error::InvalidData(format!(
                "Ogg page CRC mismatch (got {:08x}, expected {:08x})",
                computed, claimed_crc
            )));
        }

        Ok((
            Page {
                flags,
                granule_position,
                serial,
                seq_no,
                lacing,
                data,
            },
            total,
        ))
    }
}

/// One packet (or partial packet) carried inside a page.
#[derive(Clone, Debug)]
pub struct PacketSegment {
    pub seg_range: std::ops::Range<usize>,
    pub data: std::ops::Range<usize>,
    /// True if the packet ends inside this page; false if it continues into
    /// the next page.
    pub terminated: bool,
}

/// Build the lacing table for a packet of `len` bytes.
pub fn lace(len: usize) -> Vec<u8> {
    if len == 0 {
        return vec![0];
    }
    let full = len / 255;
    let rem = len % 255;
    let mut v = vec![255u8; full];
    if rem > 0 || full == 0 {
        v.push(rem as u8);
    } else {
        // Length is an exact multiple of 255 — append a zero terminator so the
        // last segment is recognized as a packet end.
        v.push(0);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lace_short() {
        assert_eq!(lace(0), vec![0]);
        assert_eq!(lace(1), vec![1]);
        assert_eq!(lace(254), vec![254]);
        assert_eq!(lace(255), vec![255, 0]);
        assert_eq!(lace(256), vec![255, 1]);
        assert_eq!(lace(510), vec![255, 255, 0]);
    }

    #[test]
    fn round_trip() {
        let pkt = (0u8..=200).collect::<Vec<u8>>();
        let lacing = lace(pkt.len());
        let p = Page {
            flags: flags::FIRST_PAGE,
            granule_position: 12345,
            serial: 0xdeadbeef,
            seq_no: 0,
            lacing,
            data: pkt.clone(),
        };
        let bytes = p.to_bytes();
        let (parsed, n) = Page::parse(&bytes).unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(parsed.granule_position, 12345);
        assert_eq!(parsed.serial, 0xdeadbeef);
        assert_eq!(parsed.flags & flags::FIRST_PAGE, flags::FIRST_PAGE);
        assert_eq!(parsed.data, pkt);
        let segs = parsed.packet_segments();
        assert_eq!(segs.len(), 1);
        assert!(segs[0].terminated);
    }

    #[test]
    fn multi_segment_packet() {
        // 600-byte packet → segments [255, 255, 90]
        let pkt: Vec<u8> = (0..600).map(|i| (i & 0xff) as u8).collect();
        let lacing = lace(pkt.len());
        assert_eq!(lacing, vec![255, 255, 90]);
        let p = Page {
            flags: 0,
            granule_position: 0,
            serial: 1,
            seq_no: 5,
            lacing,
            data: pkt.clone(),
        };
        let bytes = p.to_bytes();
        let (parsed, _) = Page::parse(&bytes).unwrap();
        let segs = parsed.packet_segments();
        assert_eq!(segs.len(), 1);
        assert!(segs[0].terminated);
        let pdata = &parsed.data[segs[0].data.clone()];
        assert_eq!(pdata, pkt);
    }
}
