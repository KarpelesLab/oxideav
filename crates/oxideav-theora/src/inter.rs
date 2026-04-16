//! Theora inter-frame decode.
//!
//! Implements the inter-frame specific bitstream sections of the Theora
//! I.2 specification:
//!
//! * §7.3.2/§7.3.3 super-block partition flags + per-block coded flags
//! * §7.3.4 macro-block coding-mode decode (8 modes, MSCHEME tables)
//! * §7.3.5 motion vector decode (MVMODE 0/1, LAST/LAST2 tracking)
//! * §7.5 motion compensation predictor (half-pel filter)
//! * §7.7.4 token decode shares the intra path, but wires up the inter
//!   Huffman group via `HTI_L`/`HTI_C`
//! * §7.8.2 inter-DC prediction with per-reference last_dc tracking
//! * §7.9.4 inter block reconstruction = motion-compensated predictor +
//!   IDCT residual
//!
//! Notes:
//! * 4-MV mode is implemented; chroma MV averaging uses
//!   `OC_DIV_ROUND_POW2`-equivalent rounding (Table 7.34).
//! * Filtering across uncoded edges (§7.10.4) is supported by `block.rs`.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::block::{read_long_run_bitstring, read_short_run_bitstring};
use crate::coded_order::FrameLayout;
use crate::headers::PixelFormat;

/// Reference-frame index (RFI). 0 = intra (no reference), 1 = previous
/// reference (LAST), 2 = golden frame.
pub type Rfi = u8;

/// Theora macro-block coding modes (spec §7.4 / Table 7.18).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    InterNoMv = 0,
    Intra = 1,
    InterMv = 2,
    InterMvLast = 3,
    InterMvLast2 = 4,
    InterGoldenNoMv = 5,
    InterGoldenMv = 6,
    InterMvFour = 7,
}

impl Mode {
    pub fn from_index(i: u8) -> Result<Self> {
        Ok(match i {
            0 => Mode::InterNoMv,
            1 => Mode::Intra,
            2 => Mode::InterMv,
            3 => Mode::InterMvLast,
            4 => Mode::InterMvLast2,
            5 => Mode::InterGoldenNoMv,
            6 => Mode::InterGoldenMv,
            7 => Mode::InterMvFour,
            _ => return Err(Error::invalid("Theora: bad MB coding mode")),
        })
    }

    /// Reference frame for this mode (Table 7.46).
    /// 0 = intra (no reference), 1 = last (previous reference), 2 = golden.
    pub fn rfi(self) -> Rfi {
        match self {
            Mode::Intra => 0,
            Mode::InterGoldenNoMv | Mode::InterGoldenMv => 2,
            _ => 1,
        }
    }

    /// True if this mode uses a motion vector.
    pub fn has_mv(self) -> bool {
        matches!(
            self,
            Mode::InterMv
                | Mode::InterMvLast
                | Mode::InterMvLast2
                | Mode::InterGoldenMv
                | Mode::InterMvFour
        )
    }

    /// True if four motion vectors (one per luma block) are decoded.
    pub fn is_four_mv(self) -> bool {
        matches!(self, Mode::InterMvFour)
    }
}

/// Mode alphabets. Spec Table 7.19 (`MSCHEME` 0..6 are explicit alphabets;
/// 7 means "raw" — just read 3 bits per macro-block). Index 0 = MSCHEME 0
/// (placeholder; MSCHEME=0 reads its own alphabet). Indices 1..6 are the
/// fixed alphabets from the spec table.
///
/// Mode numbering (Table 7.18):
///   0 = INTER_NOMV         3 = INTER_MV_LAST     6 = INTER_GOLDEN_MV
///   1 = INTRA              4 = INTER_MV_LAST2    7 = INTER_MV_FOUR
///   2 = INTER_MV           5 = INTER_GOLDEN_NOMV
pub const MODE_ALPHABETS: [[Mode; 8]; 7] = [
    // MSCHEME 0 — placeholder; bitstream supplies alphabet.
    [
        Mode::InterNoMv,
        Mode::Intra,
        Mode::InterMv,
        Mode::InterMvLast,
        Mode::InterMvLast2,
        Mode::InterGoldenNoMv,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
    // MSCHEME 1 column: 3,4,2,0,1,5,6,7
    [
        Mode::InterMvLast,     // 3
        Mode::InterMvLast2,    // 4
        Mode::InterMv,         // 2
        Mode::InterNoMv,       // 0
        Mode::Intra,           // 1
        Mode::InterGoldenNoMv, // 5
        Mode::InterGoldenMv,   // 6
        Mode::InterMvFour,     // 7
    ],
    // MSCHEME 2: 3,4,0,2,1,5,6,7
    [
        Mode::InterMvLast,
        Mode::InterMvLast2,
        Mode::InterNoMv,
        Mode::InterMv,
        Mode::Intra,
        Mode::InterGoldenNoMv,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
    // MSCHEME 3: 3,2,4,0,1,5,6,7
    [
        Mode::InterMvLast,
        Mode::InterMv,
        Mode::InterMvLast2,
        Mode::InterNoMv,
        Mode::Intra,
        Mode::InterGoldenNoMv,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
    // MSCHEME 4: 3,2,0,4,1,5,6,7
    [
        Mode::InterMvLast,
        Mode::InterMv,
        Mode::InterNoMv,
        Mode::InterMvLast2,
        Mode::Intra,
        Mode::InterGoldenNoMv,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
    // MSCHEME 5: 0,3,4,2,1,5,6,7
    [
        Mode::InterNoMv,
        Mode::InterMvLast,
        Mode::InterMvLast2,
        Mode::InterMv,
        Mode::Intra,
        Mode::InterGoldenNoMv,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
    // MSCHEME 6: 0,5,3,4,2,1,6,7
    [
        Mode::InterNoMv,
        Mode::InterGoldenNoMv,
        Mode::InterMvLast,
        Mode::InterMvLast2,
        Mode::InterMv,
        Mode::Intra,
        Mode::InterGoldenMv,
        Mode::InterMvFour,
    ],
];

/// Decode a single mode codeword (spec §7.3.4). The codeword is a
/// variable-length prefix: `0,10,110,1110,11110,111110,1111110,1111111`
/// (lengths 1..7) selecting a "rank" 0..7 in the alphabet.
pub fn decode_mode_rank(br: &mut BitReader<'_>) -> Result<u8> {
    for r in 0..7u8 {
        if !br.read_bit()? {
            return Ok(r);
        }
    }
    // The seven leading 1-bits select rank 7 (no extra bit).
    Ok(7)
}

/// Decode one motion-vector component (spec §7.3.5, Table 7.23 for MVMODE=0
/// VLC encoding). Codeword tree:
///   000      ->   0
///   001      ->  +1
///   010      ->  -1
///   0110     ->  +2
///   0111     ->  -2
///   1000     ->  +3
///   1001     ->  -3
///   101 xxx  ->  ±4..±7  (xxx = 3 bits: top 2 = mag-4, bottom = sign)
///   110 xxxx ->  ±8..±15 (xxxx = 4 bits: top 3 = mag-8, bottom = sign)
///   1110 xxxx ->  ±16..±23 (xxxx = 4 bits: top 3 = mag-16, bottom = sign)
///   1111 xxxx ->  ±24..±31 (xxxx = 4 bits: top 3 = mag-24, bottom = sign)
pub fn decode_mv_component_vlc(br: &mut BitReader<'_>) -> Result<i32> {
    let b0 = br.read_bit()?;
    let b1 = br.read_bit()?;
    let b2 = br.read_bit()?;
    if !b0 && !b1 {
        // 00x
        return Ok(if b2 { 1 } else { 0 });
    }
    if !b0 && b1 && !b2 {
        // 010
        return Ok(-1);
    }
    if !b0 && b1 && b2 {
        // 011x -> ±2
        let s = br.read_bit()?;
        return Ok(if s { -2 } else { 2 });
    }
    if b0 && !b1 && !b2 {
        // 100x -> ±3
        let s = br.read_bit()?;
        return Ok(if s { -3 } else { 3 });
    }
    if b0 && !b1 && b2 {
        // 101 + 3 bits = 6 total
        let off = br.read_u32(3)?;
        let sign = (off & 1) != 0;
        let mag = 4 + (off >> 1) as i32;
        return Ok(if sign { -mag } else { mag });
    }
    if b0 && b1 && !b2 {
        // 110 + 4 bits = 7 total
        let off = br.read_u32(4)?;
        let sign = (off & 1) != 0;
        let mag = 8 + (off >> 1) as i32;
        return Ok(if sign { -mag } else { mag });
    }
    // b0 b1 b2 = 1 1 1
    let b3 = br.read_bit()?;
    if !b3 {
        // 1110 + 4 bits = 8 total
        let off = br.read_u32(4)?;
        let sign = (off & 1) != 0;
        let mag = 16 + (off >> 1) as i32;
        return Ok(if sign { -mag } else { mag });
    }
    // 1111 + 4 bits
    let off = br.read_u32(4)?;
    let sign = (off & 1) != 0;
    let mag = 24 + (off >> 1) as i32;
    Ok(if sign { -mag } else { mag })
}

/// Decode one motion-vector component using MVMODE=1 (fixed-length 6 bits:
/// magnitude (5) + sign (1)). Spec §7.3.5.2 Table 7.31.
pub fn decode_mv_component_raw(br: &mut BitReader<'_>) -> Result<i32> {
    let mag = br.read_u32(5)? as i32;
    let sign = br.read_bit()?;
    Ok(if sign { -mag } else { mag })
}

/// Read the SBPMs (super-block partition modes) for inter frames, plus the
/// per-block BCODED flags. On return `bcoded` is filled per the spec.
///
/// Inter-frame algorithm (§7.3.2-§7.3.3):
///   1. NSBS-bit long-run "SB_PARTIAL[]" — 1 = partially coded (mixed),
///      0 = uniform (all blocks share one coded flag).
///   2. For uniform SBs (count NUNIFORM), an NUNIFORM-bit long-run
///      "SB_FULL[]" giving the single coded value (1=coded, 0=uncoded) for
///      every block in that SB.
///   3. For partial SBs, an NPART_BLOCKS-bit long-run with the per-block
///      BCODED value.
pub fn read_inter_bcoded(
    br: &mut BitReader<'_>,
    layout: &FrameLayout,
    bcoded: &mut [bool],
) -> Result<()> {
    let nsbs = total_sbs(layout) as usize;
    if nsbs == 0 {
        return Ok(());
    }
    // Pass 1: SB_PARTIAL — one bit per super-block.
    let sb_partial = read_long_run_bitstring(br, nsbs)?;
    // Pass 2: per uniform SB, decode its single coded bit.
    let n_uniform = sb_partial.iter().filter(|&&b| !b).count();
    let sb_full = if n_uniform > 0 {
        read_long_run_bitstring(br, n_uniform)?
    } else {
        Vec::new()
    };
    // Pass 3: walk SBs in coded order, assign block flags.
    let mut sb_iter = SbCodedIter::new(layout);
    let mut uniform_idx = 0usize;
    let mut partial_blocks: Vec<usize> = Vec::new();
    let mut bcoded_init: Vec<Option<bool>> = vec![None; bcoded.len()];
    while let Some(sb) = sb_iter.next(layout) {
        let is_partial = sb_partial[sb.sb_idx];
        if !is_partial {
            let val = sb_full[uniform_idx];
            uniform_idx += 1;
            for &bi in &sb.blocks {
                bcoded_init[bi] = Some(val);
            }
        } else {
            for &bi in &sb.blocks {
                partial_blocks.push(bi);
            }
        }
    }
    if !partial_blocks.is_empty() {
        let part_bits = read_short_run_bitstring(br, partial_blocks.len())?;
        for (bi, b) in partial_blocks.iter().zip(part_bits.iter()) {
            bcoded_init[*bi] = Some(*b);
        }
    }
    for (out, init) in bcoded.iter_mut().zip(bcoded_init.iter()) {
        *out = init.unwrap_or(false);
    }
    Ok(())
}

/// Total number of super-blocks in a frame across all three planes.
pub fn total_sbs(layout: &FrameLayout) -> u32 {
    let mut t = 0u32;
    for pli in 0..3 {
        let plane = &layout.planes[pli];
        let sbw = plane.nbw.div_ceil(4);
        let sbh = plane.nbh.div_ceil(4);
        t += sbw * sbh;
    }
    t
}

/// Iterator over super-blocks in coded order, yielding for each the list of
/// global block indices that fall inside the SB.
struct SbCodedIter {
    pli: usize,
    sbx: u32,
    sby: u32,
    sb_idx: usize,
}

struct SbBlocks {
    sb_idx: usize,
    blocks: Vec<usize>,
}

impl SbCodedIter {
    fn new(_layout: &FrameLayout) -> Self {
        Self {
            pli: 0,
            sbx: 0,
            sby: 0,
            sb_idx: 0,
        }
    }

    fn next(&mut self, layout: &FrameLayout) -> Option<SbBlocks> {
        loop {
            if self.pli >= 3 {
                return None;
            }
            let plane = &layout.planes[self.pli];
            let sbw = plane.nbw.div_ceil(4);
            let sbh = plane.nbh.div_ceil(4);
            if self.sbx >= sbw {
                self.sbx = 0;
                self.sby += 1;
            }
            if self.sby >= sbh {
                self.sby = 0;
                self.pli += 1;
                continue;
            }
            // Collect blocks within this super-block (Hilbert order).
            let mut blocks = Vec::with_capacity(16);
            for &(dx, dy) in &crate::coded_order::HILBERT_XY {
                let bx = self.sbx * 4 + dx as u32;
                let by = self.sby * 4 + dy as u32;
                if bx < plane.nbw && by < plane.nbh {
                    blocks.push(layout.global_coded(self.pli, bx, by) as usize);
                }
            }
            let result = SbBlocks {
                sb_idx: self.sb_idx,
                blocks,
            };
            self.sb_idx += 1;
            self.sbx += 1;
            return Some(result);
        }
    }
}

/// Result of reading a mode-alphabet header.
pub enum ModeScheme {
    /// MSCHEME 0..6: codeword decodes a rank into this alphabet.
    Alphabet([Mode; 8]),
    /// MSCHEME 7: every MB sends its 3-bit mode directly.
    Raw,
}

/// Read MSCHEME (3 bits) and any associated alphabet payload.
pub fn read_mode_alphabet(br: &mut BitReader<'_>) -> Result<ModeScheme> {
    let mscheme = br.read_u32(3)? as usize;
    if mscheme == 7 {
        return Ok(ModeScheme::Raw);
    }
    if (1..=6).contains(&mscheme) {
        return Ok(ModeScheme::Alphabet(MODE_ALPHABETS[mscheme]));
    }
    // mscheme == 0: For each MODE in 0..8, read 3-bit mi; assign MALPHABET[mi] = MODE.
    let mut alphabet = [Mode::InterNoMv; 8];
    for mode in 0..8u8 {
        let mi = br.read_u32(3)? as usize;
        if mi >= 8 {
            return Err(Error::invalid("Theora: mscheme=0 alphabet index >= 8"));
        }
        alphabet[mi] = Mode::from_index(mode)?;
    }
    Ok(ModeScheme::Alphabet(alphabet))
}

/// Convenience for "raw" MSCHEME=7: a per-MB 3-bit code is the mode index.
pub fn read_raw_mode(br: &mut BitReader<'_>) -> Result<Mode> {
    let m = br.read_u32(3)? as u8;
    Mode::from_index(m)
}

/// Sub-sampling axis flag for motion compensation.
/// `false` = full-resolution axis (luma always; chroma X for 4:4:4 etc.)
/// `true` = sub-sampled axis (chroma X for 4:2:0 / 4:2:2; chroma Y for 4:2:0)
pub type SubsampleFlag = bool;

/// Apply Theora motion compensation per spec §7.9.1.2 / §7.9.1.3.
///
/// `src` is the reference plane (full frame) in TOP-DOWN row-major layout
/// (width pw, height ph). `(bx_top, by_top)` is the destination block's
/// top-left corner in top-down pixel coordinates within the reference.
/// `mvx_half`, `mvy_half_bottomleft` are the luma MV components in half-pel
/// units (the same numerical values used for luma); for chroma, the same
/// values are interpreted at chroma resolution: a sub-sampled axis treats
/// the MV as quarter-pel of the chroma plane (i.e. divides by 2).
///
/// `subsample_x`, `subsample_y` indicate whether the corresponding axis is
/// sub-sampled in this plane.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate(
    src: &[u8],
    pw: i32,
    ph: i32,
    bx_top: i32,
    by_top: i32,
    mvx_half: i32,
    mvy_half_bottomleft: i32,
    subsample_x: SubsampleFlag,
    subsample_y: SubsampleFlag,
    out: &mut [i32; 64],
) {
    // Spec stores MV with Y positive going UP (bottom-left frame coords).
    // The reference buffer is laid out top-down, so we negate Y here to
    // convert: motion of +1 (up) means reading row y-1 of the top-down buffer.
    let mvy = -mvy_half_bottomleft;
    let mvx = mvx_half;

    // Decompose MV into integer pixel offset (truncate toward 0) and a
    // fractional indicator. Sub-sampled axes reduce precision: a quarter-pel
    // MV has integer part `mv >> 2` and fractional part = (mv & 2) >> 1, etc.
    let (int_x, half_x_dir) = decompose_mv(mvx, subsample_x);
    let (int_y, half_y_dir) = decompose_mv(mvy, subsample_y);
    let half = half_x_dir != 0 || half_y_dir != 0;

    for ry in 0..8i32 {
        for rx in 0..8i32 {
            let dx = bx_top + rx;
            let dy = by_top + ry;
            let x1 = (dx + int_x).clamp(0, pw - 1);
            let y1 = (dy + int_y).clamp(0, ph - 1);
            let p1 = src[(y1 * pw + x1) as usize] as i32;
            let v = if half {
                let x2 = (dx + int_x + half_x_dir).clamp(0, pw - 1);
                let y2 = (dy + int_y + half_y_dir).clamp(0, ph - 1);
                let p2 = src[(y2 * pw + x2) as usize] as i32;
                (p1 + p2) >> 1
            } else {
                p1
            };
            out[(ry as usize) * 8 + rx as usize] = v;
        }
    }
}

/// Decompose a half-pel MV component into (integer_offset, second_dir).
/// `subsample` true means the axis is at half resolution: the component
/// represents quarter-pel of THIS plane, integer offset = mv truncated /2,
/// fractional = (mv & 2) ? sign(mv) : 0.
fn decompose_mv(mv: i32, subsample: bool) -> (i32, i32) {
    if !subsample {
        // Luma half-pel: int = mv / 2 (toward 0), half = sign(mv) if mv odd.
        let int_part = if mv >= 0 { mv >> 1 } else { -((-mv) >> 1) };
        let dir = if mv & 1 != 0 { mv.signum() } else { 0 };
        (int_part, dir)
    } else {
        // Quarter-pel for chroma: int = mv / 4 (toward 0).
        // Half-pel-of-chroma occurs when the bit-1 (i.e. the second LSB) is
        // set (corresponding to chroma half-pel).
        let int_part = if mv >= 0 { mv >> 2 } else { -((-mv) >> 2) };
        // Quarter-pel grid step: chroma half-pel when (|mv| & 2) != 0.
        let abs_mv = mv.abs();
        let dir = if abs_mv & 2 != 0 { mv.signum() } else { 0 };
        (int_part, dir)
    }
}

/// Compute chroma half-pel MV from an integer luma MV (full-pel),
/// per spec Table 7.34 / §7.5.1. Returns (chroma_int, chroma_half).
/// The luma MV is in full-pel units; chroma MV in `pf`-dependent half-pel.
pub fn chroma_mv_split(luma: i32, pf: PixelFormat, axis_y: bool) -> (i32, i32) {
    // For Yuv420: both axes get /2.
    // For Yuv422: only x gets /2.
    // For Yuv444: no scaling.
    let scale_half = match pf {
        PixelFormat::Yuv420 => true,
        PixelFormat::Yuv422 => !axis_y,
        PixelFormat::Yuv444 | PixelFormat::Reserved => false,
    };
    if !scale_half {
        return (luma, 0);
    }
    // Round half-toward-zero to land on a half-pel grid (Table 7.34).
    // Map values: -1 → -0.5, -2 → -1, -3 → -1.5, etc.
    // We compute integer + half part.
    let int_part = luma.div_euclid(2);
    let half_part = luma.rem_euclid(2);
    (int_part, half_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_rfi_mapping() {
        assert_eq!(Mode::Intra.rfi(), 0);
        assert_eq!(Mode::InterNoMv.rfi(), 1);
        assert_eq!(Mode::InterMvFour.rfi(), 1);
        assert_eq!(Mode::InterGoldenNoMv.rfi(), 2);
        assert_eq!(Mode::InterGoldenMv.rfi(), 2);
    }

    #[test]
    fn mode_has_mv() {
        assert!(!Mode::InterNoMv.has_mv());
        assert!(!Mode::InterGoldenNoMv.has_mv());
        assert!(!Mode::Intra.has_mv());
        assert!(Mode::InterMv.has_mv());
        assert!(Mode::InterMvFour.has_mv());
    }

    #[test]
    fn mc_integer_copy() {
        let mut src = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                src[y * 16 + x] = (x + y * 16) as u8;
            }
        }
        let mut out = [0i32; 64];
        // mv=(0,0), no subsampling → integer copy at (4,4).
        motion_compensate(&src, 16, 16, 4, 4, 0, 0, false, false, &mut out);
        // Should be src[(4..12, 4..12)] — i.e. starts at value 4 + 4*16 = 68.
        assert_eq!(out[0], 68);
        assert_eq!(out[7], 75);
        assert_eq!(out[8], 84);
    }

    #[test]
    fn mc_half_pel_horizontal() {
        let mut src = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                src[y * 16 + x] = x as u8 * 10;
            }
        }
        let mut out = [0i32; 64];
        // mv=(2, 0) (= 1 luma pel right) → integer offset 1, no fractional.
        motion_compensate(&src, 16, 16, 4, 4, 2, 0, false, false, &mut out);
        // Should be src col 5 at row 4: value 50.
        assert_eq!(out[0], 50);

        // mv=(1, 0) (= 0.5 luma pel right) → integer 0, half +1.
        // Average of cols 4 and 5: (40 + 50) / 2 = 45.
        motion_compensate(&src, 16, 16, 4, 4, 1, 0, false, false, &mut out);
        assert_eq!(out[0], 45);
    }
}
