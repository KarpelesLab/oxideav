//! JPEG marker-segment walker.
//!
//! Advances through a JPEG byte stream, returning each marker segment plus
//! the entropy-coded scan data between SOS and the next marker.

use oxideav_core::{Error, Result};

use super::markers;

/// One parsed marker segment.
#[derive(Debug)]
pub struct MarkerSegment<'a> {
    /// Marker byte (without the leading 0xFF).
    pub marker: u8,
    /// Payload bytes (excludes the 2-byte length and the marker itself).
    pub payload: &'a [u8],
}

/// Start-of-frame (SOF0) descriptor.
#[derive(Clone, Debug)]
pub struct SofInfo {
    pub precision: u8,
    pub height: u16,
    pub width: u16,
    pub components: Vec<SofComponent>,
}

#[derive(Clone, Copy, Debug)]
pub struct SofComponent {
    pub id: u8,
    pub h_factor: u8,
    pub v_factor: u8,
    pub qt_id: u8,
}

pub fn parse_sof(payload: &[u8]) -> Result<SofInfo> {
    if payload.len() < 6 {
        return Err(Error::invalid("SOF: too short"));
    }
    let precision = payload[0];
    let height = u16::from_be_bytes([payload[1], payload[2]]);
    let width = u16::from_be_bytes([payload[3], payload[4]]);
    let nf = payload[5] as usize;
    if payload.len() < 6 + nf * 3 {
        return Err(Error::invalid("SOF: component list truncated"));
    }
    let mut components = Vec::with_capacity(nf);
    for i in 0..nf {
        let off = 6 + i * 3;
        let id = payload[off];
        let hv = payload[off + 1];
        let qt = payload[off + 2];
        components.push(SofComponent {
            id,
            h_factor: hv >> 4,
            v_factor: hv & 0x0F,
            qt_id: qt,
        });
    }
    Ok(SofInfo {
        precision,
        height,
        width,
        components,
    })
}

/// Scan-header (SOS) info.
#[derive(Clone, Debug)]
pub struct SosInfo {
    pub components: Vec<SosComponent>,
    pub ss: u8,
    pub se: u8,
    pub ah: u8,
    pub al: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct SosComponent {
    pub id: u8,
    pub dc_table: u8,
    pub ac_table: u8,
}

pub fn parse_sos(payload: &[u8]) -> Result<SosInfo> {
    if payload.is_empty() {
        return Err(Error::invalid("SOS: empty"));
    }
    let ns = payload[0] as usize;
    if payload.len() < 1 + ns * 2 + 3 {
        return Err(Error::invalid("SOS: truncated"));
    }
    let mut comps = Vec::with_capacity(ns);
    for i in 0..ns {
        let off = 1 + i * 2;
        let id = payload[off];
        let ta_td = payload[off + 1];
        comps.push(SosComponent {
            id,
            dc_table: ta_td >> 4,
            ac_table: ta_td & 0x0F,
        });
    }
    let off = 1 + ns * 2;
    let ss = payload[off];
    let se = payload[off + 1];
    let ah_al = payload[off + 2];
    Ok(SosInfo {
        components: comps,
        ss,
        se,
        ah: ah_al >> 4,
        al: ah_al & 0x0F,
    })
}

/// DRI payload is a 16-bit big-endian restart interval count (in MCUs).
pub fn parse_dri(payload: &[u8]) -> Result<u16> {
    if payload.len() < 2 {
        return Err(Error::invalid("DRI: too short"));
    }
    Ok(u16::from_be_bytes([payload[0], payload[1]]))
}

/// Walker over JPEG markers.
///
/// SOI/EOI/RST* have no payload or length field. Every other marker from
/// 0xC0..0xFE (excluding SOI/EOI/RST*) is followed by a big-endian 16-bit
/// length (inclusive of those two bytes themselves) and a payload.
pub struct MarkerWalker<'a> {
    buf: &'a [u8],
    pub pos: usize,
}

impl<'a> MarkerWalker<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Seek forward to the next `0xFF` marker byte, consuming fill bytes.
    /// Returns the marker byte (without 0xFF) or None at EOF.
    pub fn next_marker(&mut self) -> Result<Option<u8>> {
        while self.pos < self.buf.len() {
            if self.buf[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            // Skip a run of 0xFF fill bytes.
            while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                return Ok(None);
            }
            let m = self.buf[self.pos];
            self.pos += 1;
            if m == 0x00 {
                // Stuffed zero — not a real marker. Keep looking.
                continue;
            }
            return Ok(Some(m));
        }
        Ok(None)
    }

    /// Read the length-prefixed payload of the marker we most recently
    /// returned from `next_marker`.
    pub fn read_segment_payload(&mut self) -> Result<&'a [u8]> {
        if self.pos + 2 > self.buf.len() {
            return Err(Error::invalid("marker segment: truncated length"));
        }
        let len = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]) as usize;
        if len < 2 {
            return Err(Error::invalid("marker segment: length < 2"));
        }
        if self.pos + len > self.buf.len() {
            return Err(Error::invalid("marker segment: payload truncated"));
        }
        let p = &self.buf[self.pos + 2..self.pos + len];
        self.pos += len;
        Ok(p)
    }

    /// Consume the entropy-coded scan immediately after an SOS segment. Ends
    /// at the byte just before the next real marker (excluding 0xFF00 stuffs
    /// and RST* markers, which belong to the scan).
    pub fn read_scan_data(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        while self.pos < self.buf.len() {
            if self.buf[self.pos] == 0xFF {
                // Lookahead.
                if self.pos + 1 >= self.buf.len() {
                    return Err(Error::invalid("scan: truncated at 0xFF"));
                }
                let nxt = self.buf[self.pos + 1];
                if nxt == 0x00 {
                    // 0xFF 0x00 → literal 0xFF, part of scan.
                    self.pos += 2;
                    continue;
                }
                if markers::is_rst(nxt) {
                    // Restart markers stay inside the scan.
                    self.pos += 2;
                    continue;
                }
                // Real marker. End of scan.
                return Ok(&self.buf[start..self.pos]);
            }
            self.pos += 1;
        }
        Ok(&self.buf[start..self.pos])
    }
}
