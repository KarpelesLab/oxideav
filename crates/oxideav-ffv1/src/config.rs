//! FFV1 configuration record (RFC 9043 §4.2).
//!
//! The configuration record is a range-coded block describing stream-level
//! parameters (version, colorspace, chroma subsampling, slice grid, quant
//! tables, etc.) followed by a 32-bit CRC. FFV1 stores it in the container's
//! extradata. The encoder generates one; the decoder parses it.
//!
//! We implement the minimum shape needed for 8-bit YUV 4:2:0 / 4:4:4 version
//! 3 files: coder_type=1 (range, default states), intra=1, ec=0 (CRC of the
//! config record itself still emitted so FFmpeg accepts the record).

use oxideav_core::{Error, Result};

use crate::crc::crc32_ieee;
use crate::range_coder::{RangeDecoder, RangeEncoder};

/// Parsed FFV1 configuration record — a superset of what this codec supports,
/// so we can round-trip foreign producers' records and fail cleanly.
#[derive(Clone, Debug, Default)]
pub struct ConfigRecord {
    pub version: u32,
    pub micro_version: u32,
    pub coder_type: u32,
    pub colorspace_type: u32, // 0 = YCbCr, 1 = RGB (JPEG 2000 RCT)
    pub bits_per_raw_sample: u32,
    pub chroma_planes: bool,
    pub log2_h_chroma_subsample: u32,
    pub log2_v_chroma_subsample: u32,
    pub extra_plane: bool, // alpha
    pub num_h_slices: u32,
    pub num_v_slices: u32,
    pub quant_table_set_count: u32,
    pub ec: u32,
    pub intra: u32,
}

impl ConfigRecord {
    /// Construct a fresh config record for our simplest supported shape: 8-bit
    /// YCbCr, 4:2:0 or 4:4:4, one slice, default range coder states, intra.
    pub fn new_simple(yuv444: bool) -> Self {
        Self {
            version: 3,
            micro_version: 4,
            coder_type: 1, // range coder with default state transition
            colorspace_type: 0,
            bits_per_raw_sample: 8,
            chroma_planes: true,
            log2_h_chroma_subsample: if yuv444 { 0 } else { 1 },
            log2_v_chroma_subsample: if yuv444 { 0 } else { 1 },
            extra_plane: false,
            num_h_slices: 1,
            num_v_slices: 1,
            quant_table_set_count: 1,
            ec: 0,
            intra: 1,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut enc = RangeEncoder::new();
        // FFmpeg shares a single 32-byte state buffer across all symbol and
        // rac calls in the extradata — `put_rac(state, ...)` uses state[0]
        // (the same byte as `put_symbol`'s zero-bit marker). That subtle
        // detail matters for bit-for-bit compatibility.
        let mut state = [128u8; 32];
        enc.put_symbol_u(&mut state, self.version);
        if self.version >= 3 {
            enc.put_symbol_u(&mut state, self.micro_version);
        }
        enc.put_symbol_u(&mut state, self.coder_type);
        // state_transition_delta omitted (coder_type == 1).
        enc.put_symbol_u(&mut state, self.colorspace_type);
        if self.version >= 1 {
            enc.put_symbol_u(&mut state, self.bits_per_raw_sample);
        }
        enc.put_rac(&mut state[0], self.chroma_planes);
        enc.put_symbol_u(&mut state, self.log2_h_chroma_subsample);
        enc.put_symbol_u(&mut state, self.log2_v_chroma_subsample);
        enc.put_rac(&mut state[0], self.extra_plane);
        enc.put_symbol_u(&mut state, self.num_h_slices.saturating_sub(1));
        enc.put_symbol_u(&mut state, self.num_v_slices.saturating_sub(1));
        enc.put_symbol_u(&mut state, self.quant_table_set_count);
        // Emit the configured number of quantisation-table sets. Each set
        // has 5 sub-tables; we always emit FFmpeg's default sub-tables.
        for _ in 0..self.quant_table_set_count {
            emit_default_quant_table_set(&mut enc);
        }
        // initial_state_delta == false for each set (no custom states).
        for _ in 0..self.quant_table_set_count {
            enc.put_rac(&mut state[0], false);
        }
        if self.version >= 3 {
            enc.put_symbol_u(&mut state, self.ec);
            enc.put_symbol_u(&mut state, self.intra);
        }

        let mut bytes = enc.finish_for_extradata();
        // Append the CRC-32 (IEEE polynomial 0x04C11DB7) of the preceding
        // bytes in big-endian order. Chosen so that the CRC of the whole
        // record (including these 4 bytes) is zero — FFmpeg validates this.
        let crc = crc32_ieee(&bytes);
        bytes.extend_from_slice(&crc.to_be_bytes());
        bytes
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(Error::invalid("FFV1 config record too short"));
        }
        // Verify the trailing 32-bit CRC-32 parity: CRC of the whole record
        // (body + stored CRC bytes) must be zero. Reject on mismatch, which
        // is what FFmpeg does for a v3 extradata.
        if crc32_ieee(data) != 0 {
            return Err(Error::invalid("FFV1 config record CRC mismatch"));
        }
        let body = &data[..data.len() - 4];
        let mut dec = RangeDecoder::new(body);
        let mut state = [128u8; 32];

        let version = dec.get_symbol_u(&mut state);
        if version != 3 {
            return Err(Error::unsupported(format!("FFV1 version {version}")));
        }
        let micro_version = dec.get_symbol_u(&mut state);
        // `ac` / `coder_type`: 0 = Golomb-Rice, 1 = range-coder default-tab,
        // 2 = range-coder custom-tab.
        let coder_type = dec.get_symbol_u(&mut state);
        if coder_type == 2 {
            return Err(Error::unsupported(
                "FFV1 custom state transition tables not supported",
            ));
        }
        if coder_type != 1 {
            return Err(Error::unsupported(format!(
                "FFV1 Golomb-Rice coder (coder_type={coder_type})"
            )));
        }
        let colorspace_type = dec.get_symbol_u(&mut state);
        if colorspace_type != 0 {
            return Err(Error::unsupported("FFV1 RGB colorspace"));
        }
        let bits_per_raw_sample = dec.get_symbol_u(&mut state);
        if bits_per_raw_sample == 0 {
            // FFmpeg quirk: version-3 extradata encodes 8-bit as 0.
        } else if bits_per_raw_sample != 8 {
            return Err(Error::unsupported(format!(
                "FFV1 {bits_per_raw_sample}-bit samples"
            )));
        }
        let chroma_planes = dec.get_rac(&mut state[0]);
        let log2_h_chroma_subsample = dec.get_symbol_u(&mut state);
        let log2_v_chroma_subsample = dec.get_symbol_u(&mut state);
        let extra_plane = dec.get_rac(&mut state[0]);
        if extra_plane {
            return Err(Error::unsupported("FFV1 alpha plane"));
        }
        let num_h_slices = dec.get_symbol_u(&mut state) + 1;
        let num_v_slices = dec.get_symbol_u(&mut state) + 1;
        let quant_table_set_count = dec.get_symbol_u(&mut state);
        if quant_table_set_count == 0 || quant_table_set_count > 8 {
            return Err(Error::invalid("FFV1 bad quant_table_set_count"));
        }
        // Skip the quant-table sets; we use fixed FFmpeg defaults internally
        // and do not materialise alternate tables.
        for _ in 0..quant_table_set_count {
            skip_quant_table_set(&mut dec)?;
        }
        // One `states_coded` bit per quant_table_set.
        for _ in 0..quant_table_set_count {
            let states_coded = dec.get_rac(&mut state[0]);
            if states_coded {
                return Err(Error::unsupported("FFV1 initial_state_delta"));
            }
        }
        let ec = dec.get_symbol_u(&mut state);
        let intra = dec.get_symbol_u(&mut state);
        let bits_per_raw_sample = if bits_per_raw_sample == 0 {
            8
        } else {
            bits_per_raw_sample
        };

        Ok(Self {
            version,
            micro_version,
            coder_type,
            colorspace_type,
            bits_per_raw_sample,
            chroma_planes,
            log2_h_chroma_subsample,
            log2_v_chroma_subsample,
            extra_plane,
            num_h_slices,
            num_v_slices,
            quant_table_set_count,
            ec,
            intra,
        })
    }

    pub fn is_yuv420(&self) -> bool {
        self.chroma_planes && self.log2_h_chroma_subsample == 1 && self.log2_v_chroma_subsample == 1
    }

    pub fn is_yuv444(&self) -> bool {
        self.chroma_planes && self.log2_h_chroma_subsample == 0 && self.log2_v_chroma_subsample == 0
    }
}

/// Emit one quantisation-table set containing the FFmpeg default
/// (`ffv1_context=true`). Five tables, 256 entries each. RFC encodes each
/// *half* of each table as a run-length of equal values, using the
/// range-coded unsigned symbol encoding.
fn emit_default_quant_table_set(enc: &mut RangeEncoder) {
    let set = crate::state::default_quant_tables();
    for tbl in &set {
        // The first half is entries [0..128) but encoded starting at index 0
        // by reading runs. Each iteration: scan how many consecutive entries
        // from position `i` have the same value as `i`, emit `run - 1`, then
        // advance `i` by `run`.
        let mut i: usize = 0;
        let mut state = [128u8; 32];
        while i < 128 {
            let v = tbl[i];
            let mut j = i + 1;
            while j < 128 && tbl[j] == v {
                j += 1;
            }
            let run = (j - i) as u32;
            enc.put_symbol_u(&mut state, run - 1);
            i = j;
        }
    }
}

/// Skip over one quantisation-table set in a range-coded config record.
fn skip_quant_table_set(dec: &mut RangeDecoder<'_>) -> Result<()> {
    for _ in 0..5 {
        let mut state = [128u8; 32];
        let mut pos: u32 = 0;
        while pos < 128 {
            let run = dec.get_symbol_u(&mut state) + 1;
            pos += run;
        }
        if pos != 128 {
            return Err(Error::invalid("FFV1 quant table run overshoot"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple_420() {
        let c = ConfigRecord::new_simple(false);
        let bytes = c.encode();
        let parsed = ConfigRecord::parse(&bytes).expect("parse");
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.coder_type, 1);
        assert_eq!(parsed.bits_per_raw_sample, 8);
        assert!(parsed.is_yuv420());
        assert_eq!(parsed.num_h_slices, 1);
        assert_eq!(parsed.num_v_slices, 1);
        assert_eq!(parsed.quant_table_set_count, 1);
    }

    #[test]
    fn roundtrip_simple_444() {
        let c = ConfigRecord::new_simple(true);
        let bytes = c.encode();
        let parsed = ConfigRecord::parse(&bytes).expect("parse");
        assert!(parsed.is_yuv444());
    }
}
