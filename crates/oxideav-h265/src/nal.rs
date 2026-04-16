//! HEVC NAL unit framing.
//!
//! Two byte-stream formats are commonly seen:
//!
//! * **Annex B** (ITU-T H.265 Annex B / `.hevc` / `.h265` files, MPEG-TS):
//!   each NAL unit is preceded by a 3- or 4-byte start code prefix
//!   (`0x000001` or `0x00000001`).
//! * **Length-prefixed** (HVCC / MP4 `hvc1` / `hev1`): each NAL unit begins
//!   with an N-byte big-endian length field, where N = `length_size_minus_one
//!   + 1` from the HEVCDecoderConfigurationRecord (typically 4).
//!
//! The 2-byte NAL header (§7.3.1.2) packs:
//!
//! ```text
//!   forbidden_zero_bit       u(1)   — must be 0
//!   nal_unit_type            u(6)
//!   nuh_layer_id             u(6)
//!   nuh_temporal_id_plus1    u(3)   — must be != 0
//! ```
//!
//! After framing, callers should strip emulation-prevention bytes via
//! `extract_rbsp` before applying the bit reader.

use oxideav_core::{Error, Result};

/// HEVC NAL unit type codes (§7.4.2.2). Only the names we touch in v1.
/// Use `as_u8` / `from_u8` for the numeric encoding.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NalUnitType {
    TrailN,
    TrailR,
    TsaN,
    TsaR,
    StsaN,
    StsaR,
    RadlN,
    RadlR,
    RaslN,
    RaslR,
    BlaWLp,
    BlaWRadl,
    BlaNLp,
    IdrWRadl,
    IdrNLp,
    CraNut,
    RsvIrapVcl22,
    RsvIrapVcl23,
    Vps,
    Sps,
    Pps,
    AccessUnitDelimiter,
    EndOfSeq,
    EndOfBitstream,
    FillerData,
    PrefixSeiNut,
    SuffixSeiNut,
    Other(u8),
}

impl NalUnitType {
    pub fn from_u8(v: u8) -> Self {
        use NalUnitType::*;
        match v {
            0 => TrailN,
            1 => TrailR,
            2 => TsaN,
            3 => TsaR,
            4 => StsaN,
            5 => StsaR,
            6 => RadlN,
            7 => RadlR,
            8 => RaslN,
            9 => RaslR,
            16 => BlaWLp,
            17 => BlaWRadl,
            18 => BlaNLp,
            19 => IdrWRadl,
            20 => IdrNLp,
            21 => CraNut,
            22 => RsvIrapVcl22,
            23 => RsvIrapVcl23,
            32 => Vps,
            33 => Sps,
            34 => Pps,
            35 => AccessUnitDelimiter,
            36 => EndOfSeq,
            37 => EndOfBitstream,
            38 => FillerData,
            39 => PrefixSeiNut,
            40 => SuffixSeiNut,
            other => Other(other),
        }
    }

    pub fn as_u8(self) -> u8 {
        use NalUnitType::*;
        match self {
            TrailN => 0,
            TrailR => 1,
            TsaN => 2,
            TsaR => 3,
            StsaN => 4,
            StsaR => 5,
            RadlN => 6,
            RadlR => 7,
            RaslN => 8,
            RaslR => 9,
            BlaWLp => 16,
            BlaWRadl => 17,
            BlaNLp => 18,
            IdrWRadl => 19,
            IdrNLp => 20,
            CraNut => 21,
            RsvIrapVcl22 => 22,
            RsvIrapVcl23 => 23,
            Vps => 32,
            Sps => 33,
            Pps => 34,
            AccessUnitDelimiter => 35,
            EndOfSeq => 36,
            EndOfBitstream => 37,
            FillerData => 38,
            PrefixSeiNut => 39,
            SuffixSeiNut => 40,
            Other(v) => v,
        }
    }

    /// Whether this NAL holds slice data (a "VCL" NAL).
    pub fn is_vcl(self) -> bool {
        matches!(self.as_u8(), 0..=31)
    }

    /// Whether this is one of the IRAP (intra random access point) types
    /// (§3.IRAP picture).
    pub fn is_irap(self) -> bool {
        matches!(self.as_u8(), 16..=23)
    }
}

/// Parsed 2-byte HEVC NAL header.
#[derive(Clone, Copy, Debug)]
pub struct NalHeader {
    pub nal_unit_type: NalUnitType,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
}

impl NalHeader {
    /// Parse the 2-byte NAL header. The caller must pass at least 2 bytes —
    /// the bytes immediately after the start code (or after the length
    /// prefix in HVCC mode).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(Error::invalid("h265: NAL header < 2 bytes"));
        }
        let b0 = bytes[0];
        let b1 = bytes[1];
        if b0 & 0x80 != 0 {
            return Err(Error::invalid(
                "h265: NAL forbidden_zero_bit must be 0 (corrupt or non-HEVC)",
            ));
        }
        let nal_unit_type = (b0 >> 1) & 0x3F;
        let nuh_layer_id = ((b0 & 0x01) << 5) | ((b1 >> 3) & 0x1F);
        let tid_plus1 = b1 & 0x07;
        if tid_plus1 == 0 {
            return Err(Error::invalid(
                "h265: nuh_temporal_id_plus1 must be > 0 (§7.4.2.2)",
            ));
        }
        Ok(Self {
            nal_unit_type: NalUnitType::from_u8(nal_unit_type),
            nuh_layer_id,
            nuh_temporal_id_plus1: tid_plus1,
        })
    }

    /// `TemporalId` = `nuh_temporal_id_plus1` − 1.
    pub fn temporal_id(self) -> u8 {
        self.nuh_temporal_id_plus1 - 1
    }
}

/// One NAL unit located in a buffer (zero-copy slice).
#[derive(Clone, Copy, Debug)]
pub struct NalRef<'a> {
    pub header: NalHeader,
    /// Body bytes including the 2-byte NAL header (still emulation-prevented).
    pub raw: &'a [u8],
}

impl<'a> NalRef<'a> {
    /// Bytes after the 2-byte NAL header — still emulation-prevented;
    /// callers normally want `extract_rbsp(self.payload())`.
    pub fn payload(&self) -> &'a [u8] {
        &self.raw[2..]
    }
}

/// Iterate over Annex B NAL units. Start codes are `0x000001` (3 bytes) or
/// `0x00000001` (4 bytes). Trailing zero stuffing is tolerated.
pub fn iter_annex_b(data: &[u8]) -> AnnexBIter<'_> {
    AnnexBIter { data, pos: 0 }
}

pub struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = NalRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (sc_off, sc_len) = find_start_code(self.data, self.pos)?;
        let body_start = sc_off + sc_len;
        // Find the next start code (or EOF) to bound the body.
        let body_end = match find_start_code(self.data, body_start) {
            Some((next_off, _)) => trim_trailing_zeros(&self.data[..next_off], body_start),
            None => trim_trailing_zeros(self.data, body_start),
        };
        self.pos = body_end;
        if body_end <= body_start || body_end - body_start < 2 {
            // Empty or too-short NAL — keep scanning so callers don't lock up.
            return self.next();
        }
        let raw = &self.data[body_start..body_end];
        // Skip silently on a malformed header — the iterator should be tolerant.
        let header = NalHeader::parse(raw).ok()?;
        Some(NalRef { header, raw })
    }
}

/// Trim trailing `0x00` bytes from `slice[..end]` down to (but not below)
/// `min_end`. Annex B may pad NAL bodies with zero stuffing.
fn trim_trailing_zeros(slice: &[u8], min_end: usize) -> usize {
    let mut end = slice.len();
    while end > min_end && slice[end - 1] == 0 {
        end -= 1;
    }
    end
}

/// Search forward from `from` for the start of the next start-code prefix
/// (`0x000001` or `0x00000001`). Returns `(offset_of_first_zero, prefix_len)`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            // Walk over an arbitrary number of zero stuffing bytes.
            let mut j = i + 2;
            while j < data.len() && data[j] == 0 {
                j += 1;
            }
            if j < data.len() && data[j] == 0x01 {
                let prefix_len = j - i + 1;
                return Some((i, prefix_len));
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    None
}

/// Iterate length-prefixed NAL units (HVCC / MP4 sample data). `length_size`
/// is the number of bytes used to encode each length field — typically 4.
pub fn iter_length_prefixed(data: &[u8], length_size: u8) -> Result<Vec<NalRef<'_>>> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(Error::invalid(format!(
            "h265: invalid length_size_minus_one (must be 0, 1 or 3): got {length_size}"
        )));
    }
    let n = length_size as usize;
    let mut out = Vec::new();
    let mut i = 0;
    while i + n <= data.len() {
        let mut len: usize = 0;
        for k in 0..n {
            len = (len << 8) | data[i + k] as usize;
        }
        i += n;
        if len < 2 || i + len > data.len() {
            return Err(Error::invalid(format!(
                "h265: length-prefixed NAL out of bounds (len={len}, remaining={})",
                data.len() - i
            )));
        }
        let raw = &data[i..i + len];
        let header = NalHeader::parse(raw)?;
        out.push(NalRef { header, raw });
        i += len;
    }
    if i != data.len() {
        return Err(Error::invalid(format!(
            "h265: trailing {} bytes after length-prefixed NAL stream",
            data.len() - i
        )));
    }
    Ok(out)
}

/// Strip HEVC emulation-prevention bytes. Per §7.4.1.1, any sequence
/// `0x00 0x00 0x03` inside NAL data is decoded by removing the `0x03`.
pub fn extract_rbsp(nal_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal_payload.len());
    let mut i = 0;
    while i < nal_payload.len() {
        if i + 2 < nal_payload.len()
            && nal_payload[i] == 0
            && nal_payload[i + 1] == 0
            && nal_payload[i + 2] == 0x03
        {
            out.push(0);
            out.push(0);
            i += 3;
            continue;
        }
        out.push(nal_payload[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_vps() {
        // VPS in our fixture starts with 0x40 0x01: nal_unit_type = 32, layer=0, tid+1=1.
        let h = NalHeader::parse(&[0x40, 0x01]).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Vps);
        assert_eq!(h.nuh_layer_id, 0);
        assert_eq!(h.nuh_temporal_id_plus1, 1);
        assert_eq!(h.temporal_id(), 0);
    }

    #[test]
    fn parse_header_sps_pps() {
        let h = NalHeader::parse(&[0x42, 0x01]).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Sps);
        let h = NalHeader::parse(&[0x44, 0x01]).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Pps);
    }

    #[test]
    fn forbidden_bit_is_rejected() {
        assert!(NalHeader::parse(&[0xC0, 0x01]).is_err());
    }

    #[test]
    fn temporal_id_zero_is_rejected() {
        assert!(NalHeader::parse(&[0x40, 0x00]).is_err());
    }

    #[test]
    fn rbsp_strip() {
        let input = [0x00u8, 0x00, 0x03, 0xAB, 0x00, 0x00, 0x03, 0xCD];
        let out = extract_rbsp(&input);
        assert_eq!(out, vec![0x00, 0x00, 0xAB, 0x00, 0x00, 0xCD]);
    }

    #[test]
    fn annex_b_iter_finds_three_nals() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xAA, 0x00, 0x00, 0x01, 0x42, 0x01, 0xBB, 0xCC,
            0x00, 0x00, 0x01, 0x44, 0x01, 0xDD,
        ];
        let nals: Vec<_> = iter_annex_b(&data).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Pps);
    }

    #[test]
    fn length_prefixed_roundtrip() {
        let data = [
            0x00u8, 0x00, 0x00, 0x03, 0x40, 0x01, 0xAA, // first NAL, len=3
            0x00, 0x00, 0x00, 0x02, 0x42, 0x01, // second NAL, len=2
        ];
        let nals = iter_length_prefixed(&data, 4).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Vps);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Sps);
    }
}
