//! Macroblock-level decoding scaffold for MPEG-4 Part 2 VOPs.
//!
//! This module is a placeholder for the follow-up I-VOP MB decode. The
//! current scaffold exposes:
//! * `AcDcPredState` — the neighbour-DC/AC state tracked across MBs during
//!   I-VOP decode (§7.4.3). Populated by the block decoder as it reconstructs
//!   each block.
//!
//! The actual block-by-block macroblock decoder is not yet implemented; the
//! decoder façade reports `Error::Unsupported("mpeg4 I-VOP MB decode:
//! follow-up")` when a caller attempts to decode a VOP body.

/// Per-block neighbour state used for AC/DC prediction. One slot per 8×8
/// block in the picture (6 blocks per macroblock: Y0..Y3, Cb, Cr). We store
/// the decoded DC coefficient and the first row / column of AC coefficients
/// for use by the next MB's prediction step.
#[derive(Clone, Debug, Default)]
pub struct BlockNeighbour {
    pub dc: i32,
    /// First AC row (natural-order positions 1..=7).
    pub ac_top_row: [i32; 7],
    /// First AC column (natural-order positions 8, 16, 24, 32, 40, 48, 56).
    pub ac_left_col: [i32; 7],
    /// Quantiser used for this block (needed to scale prediction when the
    /// neighbour was quantised at a different Q — §7.4.3.1).
    pub quant: u8,
    /// Whether this block was coded intra (prediction is only valid across
    /// intra blocks).
    pub is_intra: bool,
}

/// State maintained while decoding an I-VOP: one neighbour slot per block
/// position in the picture. Indexed by `block_mb_y * (mb_width * 2) +
/// block_mb_x * 2` for luma quadrants etc. — the layout will be formalised
/// in the follow-up I-VOP implementation.
#[derive(Debug, Default)]
pub struct AcDcPredState {
    pub blocks: Vec<BlockNeighbour>,
}

impl AcDcPredState {
    /// Allocate state for a VOL of `mb_width` × `mb_height` macroblocks.
    /// Each MB contributes 4 luma blocks and 2 chroma blocks, laid out as
    /// three independent grids (Y has 2x2 per MB, Cb / Cr have 1x1 per MB).
    pub fn new(mb_width: usize, mb_height: usize) -> Self {
        let luma_blocks = (mb_width * 2) * (mb_height * 2);
        let chroma_blocks = mb_width * mb_height;
        let total = luma_blocks + 2 * chroma_blocks;
        Self {
            blocks: vec![BlockNeighbour::default(); total],
        }
    }
}
