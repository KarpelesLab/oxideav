//! CABAC arithmetic decoding engine — ITU-T H.264 (07/2019) §9.3.1.2 / §9.3.3.2.
//!
//! Implements the binary arithmetic decoder driving CABAC:
//!
//! * `decode_bin` — context-based decoding (§9.3.3.2.1) with `pStateIdx` /
//!   `valMPS` updates via Tables 9-44 (transIdxMPS) and 9-45 (transIdxLPS).
//! * `decode_bypass` — equiprobable decoding (§9.3.3.2.2).
//! * `decode_terminate` — end-of-slice detection (§9.3.3.2.4).
//!
//! The engine's arithmetic state is represented by `codIRange` (always in
//! `[256, 510]` post-renormalization) and `codIOffset`. Initialization
//! (§9.3.1.2) aligns to the next byte boundary, sets `codIRange = 0x01FE`,
//! and loads 9 bits into `codIOffset`.
//!
//! Table 9-35 (rangeTabLPS) is indexed by `(pStateIdx, (codIRange >> 6) & 3)`.
//! Tables 9-44 / 9-45 map the current `pStateIdx` to the next state after an
//! MPS / LPS decision respectively.

use oxideav_core::{Error, Result};

// NOTE: Agent B (cabac/context.rs) will replace this with the real
// implementation. Keep this stub compiling so engine.rs tests can run.
/// Minimal CABAC context model stand-in (§9.3.1.1). Agent B will provide
/// the full context initialisation logic; for the arithmetic engine we
/// only require the mutable `pStateIdx` and `valMPS`.
pub struct CabacContext {
    pub p_state_idx: u8,
    pub val_mps: u8,
}

impl CabacContext {
    /// Build a context with an explicit `(pStateIdx, valMPS)`. Useful for
    /// tests and placeholder callers.
    pub const fn new(p_state_idx: u8, val_mps: u8) -> Self {
        Self {
            p_state_idx,
            val_mps,
        }
    }
}

/// Table 9-35 — rangeTabLPS[pStateIdx][(codIRange >> 6) & 3].
///
/// Copied verbatim from ITU-T H.264 (07/2019). Do not reorder.
pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// Table 9-44 — transIdxMPS[pStateIdx]. Next state after decoding an MPS bin.
pub const TRANS_IDX_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// Table 9-45 — transIdxLPS[pStateIdx]. Next state after decoding an LPS bin.
pub const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Binary arithmetic decoder state (§9.3.1.2).
pub struct CabacDecoder<'a> {
    bytes: &'a [u8],
    bit_pos: usize,    // absolute bit position into bytes
    cod_i_range: u32,  // §9.3.1.2 codIRange (init 0x01FE at byte-alignment)
    cod_i_offset: u32, // §9.3.1.2 codIOffset
}

impl<'a> CabacDecoder<'a> {
    /// Build a CABAC decoder starting at the next byte boundary at or after
    /// `start_bit` into `bytes`. Performs the §9.3.1.2 init:
    /// `codIRange = 0x01FE`, `codIOffset = read_bits(9)`.
    pub fn new(bytes: &'a [u8], start_bit: usize) -> Result<Self> {
        // §9.3.1.2: "Before decoding the first macroblock of a slice, the
        // initialisation of the decoding process is specified as follows:
        //   codIRange = 0x01FE;
        //   codIOffset = read_bits( 9 );"
        // `read_bits` here reads from the byte-aligned position after any
        // cabac_alignment_one_bit padding — callers pass `start_bit` pointing
        // at (or before) the byte-aligned entry point, and we align up.
        let bit_pos = start_bit.div_ceil(8) * 8;
        let mut dec = Self {
            bytes,
            bit_pos,
            cod_i_range: 0x01FE,
            cod_i_offset: 0,
        };
        dec.cod_i_offset = dec.read_bits(9)?;
        Ok(dec)
    }

    /// Current byte position (for slice-data-end alignment after
    /// `decode_terminate` returns 1). Rounded down from the internal bit
    /// cursor.
    pub fn byte_position(&self) -> usize {
        self.bit_pos / 8
    }

    /// Read `n` (1..=9) bits MSB-first from `self.bytes` at `self.bit_pos`.
    fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 9, "CABAC engine only needs up to 9-bit reads");
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Ok(v)
    }

    #[inline]
    fn read_bit(&mut self) -> Result<u32> {
        let byte_idx = self.bit_pos >> 3;
        if byte_idx >= self.bytes.len() {
            return Err(Error::invalid("cabac: read past end of stream"));
        }
        let bit_in_byte = 7 - (self.bit_pos & 7) as u32;
        let v = ((self.bytes[byte_idx] >> bit_in_byte) & 1) as u32;
        self.bit_pos += 1;
        Ok(v)
    }

    /// §9.3.3.2.1 — Context-based binary decoding. Updates `ctx.p_state_idx`
    /// / `ctx.val_mps` per Tables 9-44 / 9-45.
    pub fn decode_bin(&mut self, ctx: &mut CabacContext) -> Result<u8> {
        // Step 1 — rangeTabLPS lookup.
        let rcp_idx = ((self.cod_i_range >> 6) & 3) as usize;
        let p_state = ctx.p_state_idx as usize;
        let r_lps = RANGE_TAB_LPS[p_state][rcp_idx] as u32;
        let mut cod_i_range = self.cod_i_range - r_lps;

        let bin_val;
        if self.cod_i_offset >= cod_i_range {
            // LPS path.
            bin_val = 1 ^ ctx.val_mps;
            self.cod_i_offset -= cod_i_range;
            cod_i_range = r_lps;

            // State transition — LPS.
            if ctx.p_state_idx == 0 {
                // pStateIdx == 0 means the MPS/LPS meaning swaps on an LPS.
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[p_state];
        } else {
            // MPS path.
            bin_val = ctx.val_mps;
            ctx.p_state_idx = TRANS_IDX_MPS[p_state];
        }

        self.cod_i_range = cod_i_range;
        self.renormalize()?;
        Ok(bin_val)
    }

    /// §9.3.3.2.2 — Bypass (equiprobable) binary decoding.
    pub fn decode_bypass(&mut self) -> Result<u8> {
        // §9.3.3.2.2: codIOffset = (codIOffset << 1) | read_bits(1);
        //   if codIOffset >= codIRange: binVal = 1; codIOffset -= codIRange;
        //   else: binVal = 0;
        self.cod_i_offset = (self.cod_i_offset << 1) | self.read_bit()?;
        if self.cod_i_offset >= self.cod_i_range {
            self.cod_i_offset -= self.cod_i_range;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// §9.3.3.2.4 — Terminate (end-of-slice). Returns 1 when the arithmetic
    /// decoder has terminated; the caller should then align to the next byte
    /// boundary via `byte_position`.
    pub fn decode_terminate(&mut self) -> Result<u8> {
        // §9.3.3.2.4: codIRange -= 2;
        //   if codIOffset >= codIRange: binVal = 1 (terminate, no renorm)
        //   else: binVal = 0; renormalize.
        self.cod_i_range -= 2;
        if self.cod_i_offset >= self.cod_i_range {
            Ok(1)
        } else {
            self.renormalize()?;
            Ok(0)
        }
    }

    /// §9.3.1.2 Table 9-47 — while `codIRange < 256`, shift both state
    /// variables left by 1 and pull a fresh bit into `codIOffset` LSB.
    #[inline]
    fn renormalize(&mut self) -> Result<()> {
        while self.cod_i_range < 256 {
            self.cod_i_range <<= 1;
            self.cod_i_offset = (self.cod_i_offset << 1) | self.read_bit()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a byte buffer that encodes a given sequence of bits (MSB-first)
    /// followed by enough `0xFF` padding to cover both the 9-bit init read
    /// and any renormalization reads the engine may perform.
    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let total_bytes = bits.len().div_ceil(8);
        let mut out = vec![0u8; total_bytes];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        // Trailing padding so the engine can read ahead freely.
        out.extend(std::iter::repeat_n(0xFF, 8));
        out
    }

    #[test]
    fn decoder_bypass_returns_raw_bits() {
        // Round-trip check: build a bitstream whose decoder-visible bypass
        // output matches a reference simulation of §9.3.3.2.2.
        //
        // With codIRange fixed at 510 during bypass-only use, feeding bit `b`:
        //   new_offset = (offset << 1) | b;
        //   if new_offset >= 510: out = 1, offset = new_offset - 510;
        //   else:                 out = 0, offset = new_offset;
        //
        // We feed a carefully chosen raw bit stream and assert decode output
        // matches an independent simulation driven by that same stream.
        let raw_init = [1u8, 0, 1, 0, 1, 0, 1, 0, 1]; // 9-bit init = 0b101010101 = 341
        let raw_bypass = [1u8, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1];
        let mut bits = raw_init.to_vec();
        bits.extend_from_slice(&raw_bypass);
        let buf = pack_bits(&bits);
        let mut dec = CabacDecoder::new(&buf, 0).unwrap();

        // Reference simulator.
        let mut ref_offset: u32 = raw_init.iter().fold(0u32, |acc, &b| (acc << 1) | b as u32);
        assert_eq!(dec.cod_i_offset, ref_offset);
        let ref_range: u32 = 510;

        let mut expected = Vec::with_capacity(raw_bypass.len());
        for &b in &raw_bypass {
            let new_offset = (ref_offset << 1) | b as u32;
            if new_offset >= ref_range {
                ref_offset = new_offset - ref_range;
                expected.push(1u8);
            } else {
                ref_offset = new_offset;
                expected.push(0u8);
            }
        }

        for &exp in &expected {
            let got = dec.decode_bypass().unwrap();
            assert_eq!(got, exp, "bypass output mismatch vs §9.3.3.2.2 simulation");
        }
        assert_eq!(dec.cod_i_offset, ref_offset);
        assert_eq!(dec.cod_i_range, ref_range);

        // Sanity: bypass did consume exactly `raw_bypass.len()` bits past init.
        assert_eq!(dec.bit_pos, 9 + raw_bypass.len());
    }

    #[test]
    fn decoder_bin_transitions_state() {
        // Sanity-check Tables 9-44 / 9-45:
        //   pStateIdx == 0 is the special case where an LPS flips valMPS.
        //   pStateIdx == 63 is the high-confidence terminal state.
        assert_eq!(TRANS_IDX_MPS[0], 1);
        assert_eq!(TRANS_IDX_MPS[62], 62);
        assert_eq!(TRANS_IDX_MPS[63], 63);
        assert_eq!(TRANS_IDX_LPS[0], 0);
        assert_eq!(TRANS_IDX_LPS[63], 63);

        // Case A — force the MPS path.
        //
        // With pStateIdx=0, valMPS=0, codIRange=510 (init), rangeIdx = (510>>6)&3
        // = 7 & 3 = 3, so rangeTabLPS[0][3] = 240. codIRange after the LPS
        // subtract is 510 - 240 = 270. If codIOffset < 270 we take the MPS
        // branch. Set all init+renorm bits to 0 => codIOffset = 0 < 270.
        let buf = pack_bits(&[0u8; 32]);
        let mut dec = CabacDecoder::new(&buf, 0).unwrap();
        let mut ctx = CabacContext::new(0, 0);
        let bin = dec.decode_bin(&mut ctx).unwrap();
        assert_eq!(bin, 0, "expected MPS (=valMPS=0)");
        assert_eq!(ctx.p_state_idx, TRANS_IDX_MPS[0]);
        assert_eq!(ctx.val_mps, 0);

        // Case B — force the LPS path from pStateIdx=0 (which must flip valMPS).
        //
        // To make codIOffset >= 270 during init, we need the top 9 bits to
        // encode a number in [270, 511]. 0b100001110 = 270 works.
        let mut init = [0u8; 9];
        // 270 = 0b100001110
        let v: u32 = 270;
        for i in 0..9 {
            init[i] = ((v >> (8 - i)) & 1) as u8;
        }
        let mut bits: Vec<u8> = init.to_vec();
        // Plenty of padding bits for renorm (LPS => codIRange=240, still <256
        // after the subtract-and-assign; renorm will loop at least once).
        bits.extend(std::iter::repeat_n(0u8, 32));
        let buf = pack_bits(&bits);
        let mut dec = CabacDecoder::new(&buf, 0).unwrap();
        assert_eq!(dec.cod_i_offset, 270);
        let mut ctx = CabacContext::new(0, 0);
        let bin = dec.decode_bin(&mut ctx).unwrap();
        assert_eq!(bin, 1, "expected LPS bin = 1 ^ valMPS = 1");
        // pStateIdx was 0 -> valMPS must flip per §9.3.3.2.1.
        assert_eq!(ctx.val_mps, 1);
        assert_eq!(ctx.p_state_idx, TRANS_IDX_LPS[0]);
    }

    #[test]
    fn decoder_terminate_round_trip() {
        // §9.3.3.2.4: decode_terminate subtracts 2 from codIRange and checks
        // codIOffset >= new codIRange for the terminate flag.
        //
        // With fresh init (codIRange = 510, codIOffset = X):
        //   new codIRange = 508. Terminate <=> codIOffset >= 508.
        //
        // First verify a non-terminate: offset = 0 => not terminated.
        {
            let buf = pack_bits(&[0u8; 32]);
            let mut dec = CabacDecoder::new(&buf, 0).unwrap();
            assert_eq!(dec.decode_terminate().unwrap(), 0);
            // After non-terminate the engine must renormalize; codIRange was
            // 508 (>=256) so no renorm loop runs.
            assert_eq!(dec.cod_i_range, 508);
        }

        // Now a terminate: we need codIOffset >= 508 during init. Pick 508.
        {
            let v: u32 = 508; // 0b111111100
            let mut init = [0u8; 9];
            for i in 0..9 {
                init[i] = ((v >> (8 - i)) & 1) as u8;
            }
            let mut bits: Vec<u8> = init.to_vec();
            bits.extend(std::iter::repeat_n(0u8, 16));
            let buf = pack_bits(&bits);
            let mut dec = CabacDecoder::new(&buf, 0).unwrap();
            assert_eq!(dec.cod_i_offset, 508);
            // After init, bit cursor is at bit 9 -> byte_position = 1.
            assert_eq!(dec.byte_position(), 1);
            assert_eq!(dec.decode_terminate().unwrap(), 1);
            // No renormalization on terminate path — byte position unchanged.
            assert_eq!(dec.byte_position(), 1);
        }
    }

    #[test]
    fn range_tab_lps_shape() {
        // Spec constraint: rangeTabLPS entries are all in [2, 240] and the
        // last row is all 2s (§9.3.3.2.1 note — high-confidence state).
        for (i, row) in RANGE_TAB_LPS.iter().enumerate() {
            for &v in row {
                assert!((2..=240).contains(&v), "row {} out of range: {}", i, v);
            }
        }
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
    }
}
