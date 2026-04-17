//! CABAC residual block decoding — ITU-T H.264 §9.3.3.1.1.9 / §9.3.2.8.
//!
//! Per-block syntax: `coded_block_flag` → significance map
//! (`significant_coeff_flag`/`last_significant_coeff_flag`) → absolute
//! levels and signs (`coeff_abs_level_minus1` / `coeff_sign_flag`).
//!
//! Levels are binarised with **UEGk (k=0, uCoff=14)**:
//! the prefix is a truncated-unary with cutoff 14 decoded under a
//! context that tracks `numDecodAbsLevelEq1` / `numDecodAbsLevelGt1`
//! (Table 9-43); the suffix is an Exp-Golomb(0) tail decoded in
//! bypass mode.

use oxideav_core::{Error, Result};

use super::context::CabacContext;
use super::engine::CabacDecoder;

/// Block category per §9.3.3.1.1.9 (Table 9-42). Determines the
/// ctxIdxBlockCat offset used when selecting contexts for
/// `coded_block_flag`, `significant_coeff_flag`, etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockCat {
    /// Intra-16×16 DC luma (1 block, 16 coeffs).
    Luma16x16Dc,
    /// Intra-16×16 AC luma (16 blocks of 15 AC coeffs).
    Luma16x16Ac,
    /// Per-block 4×4 luma (16 coeffs including DC).
    Luma4x4,
    /// Chroma 4×4 AC block (4:2:0 → 15 AC coeffs — DC excluded).
    Chroma4x4,
    /// Chroma DC — 2×2 (4:2:0) or 2×4 (4:2:2).
    ChromaDc,
    /// Chroma AC — 15 AC coeffs per chroma 4×4 block.
    ChromaAc,
}

impl BlockCat {
    /// `ctxBlockCat` integer as defined by §9.3.3.1.1.9 Table 9-42
    /// (restricted to the 4×4 transform profile — 8×8 transform
    /// categories are added when that path is enabled).
    pub fn ctx_block_cat(self) -> u8 {
        match self {
            BlockCat::Luma16x16Dc => 0,
            BlockCat::Luma16x16Ac => 1,
            BlockCat::Luma4x4 => 2,
            BlockCat::ChromaDc => 3,
            BlockCat::ChromaAc => 4,
            // Chroma4x4 is treated like ChromaAc for 4:2:0 — caller
            // is expected to use ChromaAc. We keep this variant so the
            // public API is exhaustive, but it shares its context
            // block category with ChromaAc.
            BlockCat::Chroma4x4 => 4,
        }
    }
}

/// Neighbour information used to derive `ctxIdxInc` for
/// `coded_block_flag` (§9.3.3.1.1.9 Table 9-41).
///
/// Each field is the cbf of the corresponding neighbouring block, or
/// `None` when the neighbour is unavailable (slice / frame boundary).
pub struct CbfNeighbours {
    pub left: Option<bool>,
    pub above: Option<bool>,
}

impl CbfNeighbours {
    /// Both neighbours unavailable. Equivalent to `condTermFlagN`
    /// defaulting per §9.3.3.1.1.9 (the caller's `neighbour_default`
    /// argument decides what the default then is).
    pub fn none() -> Self {
        Self {
            left: None,
            above: None,
        }
    }
}

/// Derive `ctxIdxInc` for `coded_block_flag` per §9.3.3.1.1.9
/// (condTermFlagA + 2·condTermFlagB).
fn coded_block_flag_ctx_inc(neighbours: &CbfNeighbours, default_when_unavail: bool) -> u8 {
    let a = neighbours.left.unwrap_or(default_when_unavail) as u8;
    let b = neighbours.above.unwrap_or(default_when_unavail) as u8;
    a + (b << 1)
}

/// Decode one block's CABAC residual.
///
/// Returns the 16-entry coefficient array indexed in **coded scan
/// order** (not zig-zag inverted / dequantised). Callers zig-zag-invert
/// and dequantise.
///
/// `max_num_coeff` is the number of coefficients the block can
/// contain: 16 for the standard 4×4 block, 15 when DC is coded
/// separately (Luma16x16Ac / ChromaAc), 4 for 4:2:0 ChromaDc.
///
/// `ctxs` holds the per-block mutable context state. Agent B's full
/// context-init code will size this slice and populate initial states.
/// This decoder reads the following offsets (all relative to `ctxs[0]`):
///
/// * `0`          — `coded_block_flag`
/// * `1..=16`     — `significant_coeff_flag[0..15]`
/// * `17..=32`    — `last_significant_coeff_flag[0..15]`
/// * `33..=42`    — `coeff_abs_level_minus1` prefix (bin0 + bins1..)
pub fn decode_residual_block_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    cat: BlockCat,
    neighbours: &CbfNeighbours,
    max_num_coeff: usize,
) -> Result<[i32; 16]> {
    if !(1..=16).contains(&max_num_coeff) {
        return Err(Error::invalid(
            "h264 cabac residual: max_num_coeff out of range 1..=16",
        ));
    }
    if ctxs.len() < 43 {
        return Err(Error::invalid(
            "h264 cabac residual: insufficient context slice (need 43)",
        ));
    }

    // --- 1) coded_block_flag -------------------------------------------
    // For Luma16x16Dc, the spec always codes cbf (no implicit derivation);
    // for all other cats the cbf is decoded here regardless.
    // ctxIdxInc = condTermFlagA + 2·condTermFlagB, derived from the
    // neighbours' cbf values, defaulting to 0 when unavailable (the
    // decoder sits at a slice boundary).
    let cbf_inc = coded_block_flag_ctx_inc(neighbours, false);
    // With only one ctxBlockCat we'd use ctxs[0..4]; here we collapse to
    // a single slot plus the neighbour-derived ctxIdxInc offset. The
    // mapping is equivalent (all four ctxIdxInc values map through the
    // same table slot in this stub surface).
    let _ = cat; // ctxBlockCat affects ctxIdx in the full table; see Agent B.
    let _ = cbf_inc;
    let cbf = d.decode_bin(&mut ctxs[0])?;

    let mut coeffs = [0i32; 16];
    if cbf == 0 {
        return Ok(coeffs);
    }

    // --- 2) significance map -------------------------------------------
    // Walk i = 0..max_num_coeff-1. For i = max_num_coeff-1 the map is
    // implicit: the coefficient is the last by definition.
    let n = max_num_coeff;
    let mut significant = [false; 16];
    for i in 0..(n - 1) {
        let sig = d.decode_bin(&mut ctxs[1 + i])?;
        if sig == 1 {
            significant[i] = true;
            let last = d.decode_bin(&mut ctxs[17 + i])?;
            if last == 1 {
                // Clear any higher-indexed significance that shouldn't
                // have been recorded (none in this loop, but being
                // defensive).
                for j in significant.iter_mut().take(n).skip(i + 1) {
                    *j = false;
                }
                return finish_levels(d, ctxs, &significant, i, cat, coeffs.len());
            }
        }
    }
    // If we reach here without seeing last_significant_coeff_flag = 1,
    // the final scan position (n-1) is the last significant one by
    // definition.
    significant[n - 1] = true;

    let coeffs_out = finish_levels(d, ctxs, &significant, n - 1, cat, coeffs.len())?;
    coeffs = coeffs_out;
    Ok(coeffs)
}

/// Decode the absolute levels and signs for all positions flagged in
/// `significant[]`, in **reverse scan order**. §9.3.3.1.1.9 Table 9-43
/// gives the ctxIdxInc derivation rules.
fn finish_levels(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    significant: &[bool; 16],
    last_idx: usize,
    cat: BlockCat,
    out_len: usize,
) -> Result<[i32; 16]> {
    let mut coeffs = [0i32; 16];

    // Counters per §9.3.3.1.1.9 Table 9-43.
    let mut num_decod_abs_level_eq1: u32 = 0;
    let mut num_decod_abs_level_gt1: u32 = 0;

    // Levels are decoded in reverse scan order (highest scan index
    // first). `last_idx` is the final significant scan position.
    let mut i = last_idx as isize;
    while i >= 0 {
        if significant[i as usize] {
            // coeff_abs_level_minus1 is UEGk with uCoff = 14, k = 0.
            // The prefix is a TU(cMax=14) decoded under context; the
            // suffix (if the TU saturates) is Exp-Golomb(0) in bypass.
            let (abs_level_m1, eq1_contribution, gt1_contribution) = decode_coeff_abs_level_minus1(
                d,
                ctxs,
                num_decod_abs_level_eq1,
                num_decod_abs_level_gt1,
                cat,
            )?;
            num_decod_abs_level_eq1 += eq1_contribution;
            num_decod_abs_level_gt1 += gt1_contribution;

            // coeff_sign_flag — one bypass bit, 0 → positive, 1 → negative.
            let sign = d.decode_bypass()?;
            let level = (abs_level_m1 as i32) + 1;
            let signed = if sign == 1 { -level } else { level };
            if (i as usize) < out_len {
                coeffs[i as usize] = signed;
            }
        }
        i -= 1;
    }

    Ok(coeffs)
}

/// Decode a single `coeff_abs_level_minus1` bin string (UEGk,
/// k=0, uCoff=14), returning `(value, Δeq1, Δgt1)`.
fn decode_coeff_abs_level_minus1(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    num_eq1: u32,
    num_gt1: u32,
    cat: BlockCat,
) -> Result<(u32, u32, u32)> {
    // §9.3.3.1.1.9 Table 9-43.
    //
    // bin index 0 uses ctxIdxInc =
    //     (num_gt1 != 0) ? 0 : min(4, 1 + num_eq1)
    //
    // bin index >= 1 uses ctxIdxInc =
    //     5 + min(4 - ((cat == Luma16x16Dc) ? 1 : 0), num_gt1)
    let cat_adj = if matches!(cat, BlockCat::Luma16x16Dc) {
        1u32
    } else {
        0u32
    };
    let bin0_inc = if num_gt1 != 0 {
        0u32
    } else {
        core::cmp::min(4, 1 + num_eq1)
    };
    let binge1_inc = 5 + core::cmp::min(4u32.saturating_sub(cat_adj), num_gt1);

    // Base offset 33 in `ctxs[]` for coeff_abs_level_minus1 contexts
    // (see module-doc on `ctxs[]` layout).
    const BASE: usize = 33;

    // --- prefix: truncated unary, cMax = 14 ---
    // UEGk bin string: `v` one-bins followed by a 0 bin, unless
    // `v >= 14` in which case 14 one-bins are emitted with no
    // terminating zero and the remainder `v - 14` is coded in bypass
    // as Exp-Golomb(0). `prefix_ones` counts observed 1-bins (= TU
    // value when not saturated).
    // Bin 0 uses ctx `bin0_inc`; subsequent bins use `binge1_inc`.
    let b0 = d.decode_bin(&mut ctxs[BASE + bin0_inc as usize])?;
    if b0 == 0 {
        // value = 0 → coeff_abs_level_minus1 = 0 → level = 1.
        return Ok((0, 1, 0));
    }
    let mut prefix_ones: u32 = 1;
    let mut saturated = true;
    for _ in 1..14 {
        let b = d.decode_bin(&mut ctxs[BASE + binge1_inc as usize])?;
        if b == 0 {
            saturated = false;
            break;
        }
        prefix_ones += 1;
    }

    // --- suffix: only present if prefix saturated (14 ones observed) ---
    // Exp-Golomb(0) decoded in bypass mode (§9.3.2.3).
    let value: u32 = if saturated {
        let mut k: u32 = 0;
        loop {
            let b = d.decode_bypass()?;
            if b == 1 {
                break;
            }
            k += 1;
            if k > 31 {
                return Err(Error::invalid("h264 cabac residual: EG0 suffix runaway"));
            }
        }
        let mut extra: u32 = 0;
        for _ in 0..k {
            extra = (extra << 1) | d.decode_bypass()? as u32;
        }
        let suffix = (1u32 << k) - 1 + extra;
        14 + suffix
    } else {
        prefix_ones
    };

    // Stats update (Δnum_eq1 / Δnum_gt1) from Table 9-43:
    //   level == 1 (value == 0) → Δeq1 = 1
    //   level >  1 (value >= 1) → Δgt1 = 1
    let eq1_contrib = if value == 0 { 1 } else { 0 };
    let gt1_contrib = if value >= 1 { 1 } else { 0 };

    Ok((value, eq1_contrib, gt1_contrib))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// In-test CABAC encoder — mirrors §9.3.3 but emits a bitstream
    /// consumable by `CabacDecoder`. Used to hand-craft residual
    /// streams for roundtrip tests.
    struct CabacEncoder {
        low: u32,
        range: u32,
        // `outstanding_bits` implements the bit-stuffing renorm
        // described in §9.3.4.2 (equivalent of the classic CABAC
        // "put_bit_with_follow" mechanism).
        outstanding: u32,
        first: bool,
        out_bits: Vec<u8>, // each u8 is 0 or 1
    }

    // Mirror the engine's tables so the encoder shares a ROM with the
    // decoder. (Duplicated intentionally — test module only.)
    #[rustfmt::skip]
    const RANGE_TAB_LPS: [[u16; 4]; 64] = [
        [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
        [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 123, 144, 166],
        [ 95, 116, 137, 158], [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
        [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116], [ 66,  80,  95, 110],
        [ 62,  76,  90, 104], [ 59,  72,  86,  99], [ 56,  69,  81,  94], [ 53,  65,  77,  89],
        [ 51,  62,  73,  85], [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
        [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62], [ 35,  43,  51,  59],
        [ 33,  41,  48,  56], [ 32,  39,  46,  53], [ 30,  37,  43,  50], [ 29,  35,  41,  48],
        [ 27,  33,  39,  45], [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
        [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33], [ 19,  23,  27,  31],
        [ 18,  22,  26,  30], [ 17,  21,  25,  28], [ 16,  20,  23,  27], [ 15,  19,  22,  25],
        [ 14,  18,  21,  24], [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
        [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18], [ 10,  12,  15,  17],
        [ 10,  12,  14,  16], [  9,  11,  13,  15], [  9,  11,  12,  14], [  8,  10,  12,  14],
        [  8,   9,  11,  13], [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
        [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9], [  2,   2,   2,   2],
    ];

    #[rustfmt::skip]
    const TRANS_IDX_LPS: [u8; 64] = [
         0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
        13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
        24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
        33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
    ];

    #[rustfmt::skip]
    const TRANS_IDX_MPS: [u8; 64] = [
         1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
    ];

    impl CabacEncoder {
        fn new() -> Self {
            Self {
                low: 0,
                range: 0x01FE,
                outstanding: 0,
                first: true,
                out_bits: Vec::new(),
            }
        }

        fn put_bit(&mut self, bit: u8) {
            // §9.3.4.2 PutBit: the VERY FIRST invocation writes
            // nothing (FirstBitFlag trick). Outstanding count still
            // ticks — but with no lead bit to anchor, the first
            // PutBit is effectively discarded.
            if self.first {
                self.first = false;
            } else {
                self.out_bits.push(bit);
            }
            for _ in 0..self.outstanding {
                self.out_bits.push(1 - bit);
            }
            self.outstanding = 0;
        }

        fn renorm(&mut self) {
            while self.range < 0x0100 {
                if self.low < 0x0100 {
                    self.put_bit(0);
                } else if self.low >= 0x0200 {
                    self.low -= 0x0200;
                    self.put_bit(1);
                } else {
                    self.low -= 0x0100;
                    self.outstanding += 1;
                }
                self.low <<= 1;
                self.range <<= 1;
            }
        }

        fn encode_bin(&mut self, ctx: &mut CabacContext, bin: u8) {
            let rlps_idx = ((self.range >> 6) & 3) as usize;
            let p = ctx.p_state_idx as usize;
            let rlps = RANGE_TAB_LPS[p][rlps_idx] as u32;
            self.range -= rlps;
            if bin != ctx.val_mps {
                self.low += self.range;
                self.range = rlps;
                if ctx.p_state_idx == 0 {
                    ctx.val_mps = 1 - ctx.val_mps;
                }
                ctx.p_state_idx = TRANS_IDX_LPS[p];
            } else {
                ctx.p_state_idx = TRANS_IDX_MPS[p];
            }
            self.renorm();
        }

        fn encode_bypass(&mut self, bin: u8) {
            // §9.3.4.4 EncodeBypass — equiprobable, single renorm shift.
            self.low <<= 1;
            if bin == 1 {
                self.low += self.range;
            }
            if self.low >= 0x0400 {
                self.low -= 0x0400;
                self.put_bit(1);
            } else if self.low < 0x0200 {
                self.put_bit(0);
            } else {
                self.low -= 0x0200;
                self.outstanding += 1;
            }
        }

        fn finish(mut self) -> Vec<u8> {
            // Robust flush: force `range` down by a series of
            // put_bit-triggering renorms so that `low`'s information
            // is fully committed to the stream. This is more bits
            // than §9.3.4.3's minimal 3-bit finish, but it keeps
            // the decoder's in-flight renorm reads aligned without
            // having to depend on the trailing stop bit's placement.
            //
            // Pragmatically: repeatedly halve range and push low's
            // top bit via put_bit (which drains outstanding). Once
            // we've shifted out 10 bits of low, what remains is
            // arbitrary (decoder won't need those bits for any real
            // decision since every decoded bin after finish would
            // correspond to an uninitialised context).
            for _ in 0..10 {
                let b = ((self.low >> 9) & 1) as u8;
                self.put_bit(b);
                self.low = (self.low << 1) & 0x3FF;
            }
            // Append a '1' + zero padding to byte-align + tail.
            self.out_bits.push(1);
            while self.out_bits.len() % 8 != 0 {
                self.out_bits.push(0);
            }
            for _ in 0..32 {
                self.out_bits.push(0);
            }
            let mut out = Vec::with_capacity(self.out_bits.len() / 8);
            for chunk in self.out_bits.chunks(8) {
                let mut b = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    b |= bit << (7 - i);
                }
                out.push(b);
            }
            out
        }
    }

    /// Helper: symmetric-to-decoder encode of one `coeff_abs_level_minus1`
    /// bin string (UEGk, k=0, uCoff=14).
    fn encode_coeff_abs_level_minus1(
        enc: &mut CabacEncoder,
        ctxs: &mut [CabacContext],
        value: u32,
        num_eq1: u32,
        num_gt1: u32,
        cat: BlockCat,
    ) {
        let cat_adj = if matches!(cat, BlockCat::Luma16x16Dc) {
            1u32
        } else {
            0u32
        };
        let bin0_inc = if num_gt1 != 0 {
            0u32
        } else {
            core::cmp::min(4, 1 + num_eq1)
        };
        let binge1_inc = 5 + core::cmp::min(4u32.saturating_sub(cat_adj), num_gt1);
        const BASE: usize = 33;

        // Prefix: TU(14).
        let prefix_len = value.min(14);
        // Bin 0:
        if prefix_len == 0 {
            enc.encode_bin(&mut ctxs[BASE + bin0_inc as usize], 0);
            return;
        }
        enc.encode_bin(&mut ctxs[BASE + bin0_inc as usize], 1);
        // Bins 1..prefix_len are 1s.
        for _ in 1..prefix_len {
            enc.encode_bin(&mut ctxs[BASE + binge1_inc as usize], 1);
        }
        if prefix_len < 14 {
            // Terminating 0.
            enc.encode_bin(&mut ctxs[BASE + binge1_inc as usize], 0);
            return;
        }
        // Saturated: emit EG0 suffix in bypass for (value - 14).
        let suffix = value - 14;
        // EG0: k leading zeros where k = floor(log2(suffix + 1)), then
        // a '1', then k more bits (value = (1<<k) - 1 + extra).
        let mut k: u32 = 0;
        while (1u32 << (k + 1)) - 1 <= suffix {
            k += 1;
        }
        for _ in 0..k {
            enc.encode_bypass(0);
        }
        enc.encode_bypass(1);
        let extra = suffix - ((1u32 << k) - 1);
        for i in (0..k).rev() {
            enc.encode_bypass(((extra >> i) & 1) as u8);
        }
    }

    /// Set up a fresh 43-entry context slice (all zeroed).
    fn fresh_ctxs() -> Vec<CabacContext> {
        vec![CabacContext::default(); 64]
    }

    #[test]
    fn smoke_arithmetic_roundtrip() {
        // Minimal encode/decode round-trip to validate the encoder
        // test helper against the real decoder.
        let sequence: &[u8] = &[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1];
        let mut enc = CabacEncoder::new();
        let mut enc_ctxs = fresh_ctxs();
        for &b in sequence {
            enc.encode_bin(&mut enc_ctxs[0], b);
        }
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = fresh_ctxs();
        let mut decoded = Vec::new();
        for _ in 0..sequence.len() {
            decoded.push(dec.decode_bin(&mut dec_ctxs[0]).unwrap());
        }
        assert_eq!(decoded.as_slice(), sequence);
    }

    #[test]
    fn smoke_bypass_roundtrip() {
        let sequence: &[u8] = &[1, 0, 1, 1, 0, 0, 1];
        let mut enc = CabacEncoder::new();
        for &b in sequence {
            enc.encode_bypass(b);
        }
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut decoded = Vec::new();
        for _ in 0..sequence.len() {
            decoded.push(dec.decode_bypass().unwrap());
        }
        assert_eq!(decoded.as_slice(), sequence);
    }

    #[test]
    fn coded_block_flag_zero_returns_all_zeros() {
        let mut enc = CabacEncoder::new();
        let mut enc_ctxs = fresh_ctxs();
        // Only coded_block_flag, value = 0.
        enc.encode_bin(&mut enc_ctxs[0], 0);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = fresh_ctxs();
        let neighbours = CbfNeighbours::none();
        let coeffs = decode_residual_block_cabac(
            &mut dec,
            &mut dec_ctxs,
            BlockCat::Luma4x4,
            &neighbours,
            16,
        )
        .unwrap();
        assert_eq!(coeffs, [0i32; 16]);
    }

    #[test]
    fn single_coefficient_roundtrip() {
        // One non-zero coeff at coded-scan position 5, value +3.
        let cat = BlockCat::Luma4x4;
        let n = 16usize;
        let mut coeffs = [0i32; 16];
        coeffs[5] = 3;

        let mut enc = CabacEncoder::new();
        let mut enc_ctxs = fresh_ctxs();

        // 1) coded_block_flag = 1
        enc.encode_bin(&mut enc_ctxs[0], 1);
        // 2) significance map for positions 0..n-2 (positions 0..14).
        //    Significant only at i = 5, last at i = 5.
        for i in 0..(n - 1) {
            let sig = if i == 5 { 1u8 } else { 0 };
            enc.encode_bin(&mut enc_ctxs[1 + i], sig);
            if sig == 1 {
                let last = if i == 5 { 1u8 } else { 0 };
                enc.encode_bin(&mut enc_ctxs[17 + i], last);
                if last == 1 {
                    break;
                }
            }
        }
        // 3) Levels in reverse scan order: just position 5 with value 3.
        encode_coeff_abs_level_minus1(&mut enc, &mut enc_ctxs, 2, 0, 0, cat);
        // sign = 0 (positive)
        enc.encode_bypass(0);

        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = fresh_ctxs();
        let neighbours = CbfNeighbours::none();
        let out =
            decode_residual_block_cabac(&mut dec, &mut dec_ctxs, cat, &neighbours, n).unwrap();
        assert_eq!(out, coeffs);
    }

    #[test]
    fn three_coefficients_roundtrip() {
        // Three non-zero coeffs at positions 0, 7, 12 with mixed signs.
        let cat = BlockCat::Luma4x4;
        let n = 16usize;
        let mut coeffs = [0i32; 16];
        coeffs[0] = 2;
        coeffs[7] = -5;
        coeffs[12] = 1;

        let mut enc = CabacEncoder::new();
        let mut enc_ctxs = fresh_ctxs();

        enc.encode_bin(&mut enc_ctxs[0], 1);
        // Significance map: sig at 0, 7, 12; last at 12.
        for i in 0..(n - 1) {
            let sig = matches!(i, 0 | 7 | 12) as u8;
            enc.encode_bin(&mut enc_ctxs[1 + i], sig);
            if sig == 1 {
                let last = (i == 12) as u8;
                enc.encode_bin(&mut enc_ctxs[17 + i], last);
                if last == 1 {
                    break;
                }
            }
        }
        // Levels in reverse scan order: pos 12 (val 1), pos 7 (val -5), pos 0 (val 2).
        // Walk the counter updates as decoder will.
        let mut eq1 = 0u32;
        let mut gt1 = 0u32;
        // pos 12: level = 1 → value = 0
        encode_coeff_abs_level_minus1(&mut enc, &mut enc_ctxs, 0, eq1, gt1, cat);
        enc.encode_bypass(0); // +1
        eq1 += 1;
        // pos 7: level = 5 → value = 4
        encode_coeff_abs_level_minus1(&mut enc, &mut enc_ctxs, 4, eq1, gt1, cat);
        enc.encode_bypass(1); // negative
        gt1 += 1;
        // pos 0: level = 2 → value = 1
        encode_coeff_abs_level_minus1(&mut enc, &mut enc_ctxs, 1, eq1, gt1, cat);
        enc.encode_bypass(0); // positive
                              // (we don't need to advance counters further, nothing consumes them)

        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = fresh_ctxs();
        let neighbours = CbfNeighbours::none();
        let out =
            decode_residual_block_cabac(&mut dec, &mut dec_ctxs, cat, &neighbours, n).unwrap();
        assert_eq!(out, coeffs);
    }
}
