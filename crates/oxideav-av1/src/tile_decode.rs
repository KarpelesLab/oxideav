//! AV1 tile-decode skeleton — §5.11.1 + §7.3.
//!
//! AV1 tile decode is a large endeavour. A fully-conforming decoder must
//! handle: CDFs + range coding (§9), partition quadtree + 10 partition
//! shapes (§5.11.4 + §6.10.4), per-block skip / transform-size / transform-
//! type / intra-mode / coefficient decoding (§5.11.5 – §5.11.39), intra
//! prediction (§7.11.2), dequantisation (§7.12), inverse transform
//! (§7.7), reconstruction + clipping, optional post-processing (loop
//! filter §7.14, CDEF §7.15, loop restoration §7.17, super-resolution
//! §7.16, film grain §7.20).
//!
//! This module is the **skeleton**. It runs as far as the symbol decoder
//! initialisation and the partition decision for the first superblock,
//! then returns a precise `Error::Unsupported` pointing at the next
//! unimplemented step. Each boundary error names the exact spec clause so
//! future work can pick up exactly where we stop.
//!
//! The output target is `Yuv420P`. Pixel reconstruction for real frames
//! will require shipping the remaining dozen or so features — this file
//! makes the framework observable enough to unit-test as pieces land.

use oxideav_core::{Error, Result};

use crate::frame_header::FrameHeader;
use crate::sequence_header::SequenceHeader;
use crate::symbol::SymbolDecoder;

/// Per-superblock partition shapes — §6.10.4 Table 9-3.
///
/// Only the simplest (`None`) is implemented. Others return
/// `Error::Unsupported`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
    HorzA = 4,
    HorzB = 5,
    VertA = 6,
    VertB = 7,
    Horz4 = 8,
    Vert4 = 9,
}

impl Partition {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::None,
            1 => Self::Horz,
            2 => Self::Vert,
            3 => Self::Split,
            4 => Self::HorzA,
            5 => Self::HorzB,
            6 => Self::VertA,
            7 => Self::VertB,
            8 => Self::Horz4,
            9 => Self::Vert4,
            _ => return Err(Error::invalid(format!("av1 partition: invalid {v}"))),
        })
    }
}

/// A single-tile decode context. In AV1 this would be `decode_tile()` from
/// §5.11.1 — in this skeleton we only initialise the symbol decoder, log
/// superblock iteration, and surface a precise `Unsupported` at the first
/// piece of syntax we can't consume.
pub struct TileDecoder<'a> {
    pub seq: &'a SequenceHeader,
    pub frame: &'a FrameHeader,
    pub symbol: SymbolDecoder<'a>,
    pub sb_size_log2: u32,
}

impl<'a> TileDecoder<'a> {
    /// Begin decoding a tile whose compressed payload is `tile_data`.
    pub fn new(seq: &'a SequenceHeader, frame: &'a FrameHeader, tile_data: &'a [u8]) -> Result<Self> {
        let symbol = SymbolDecoder::new(tile_data)?;
        let sb_size_log2 = if seq.use_128x128_superblock { 7 } else { 6 };
        Ok(Self {
            seq,
            frame,
            symbol,
            sb_size_log2,
        })
    }

    /// Iterate superblocks over the frame grid. Returns the number of
    /// superblocks visited before hitting an `Unsupported` boundary.
    pub fn decode(&mut self) -> Result<DecodedFrame> {
        let sb_size = 1u32 << self.sb_size_log2;
        let sbs_x = (self.frame.frame_width + sb_size - 1) / sb_size;
        let sbs_y = (self.frame.frame_height + sb_size - 1) / sb_size;
        if sbs_x == 0 || sbs_y == 0 {
            return Err(Error::invalid(
                "av1 tile_decode: zero-sized frame — impossible per §5.9",
            ));
        }
        // Walking the first superblock is the most informative boundary: it
        // forces us to decode a partition symbol, which in turn requires a
        // default CDF. We surface a precise Unsupported here.
        Err(Error::unsupported(format!(
            "av1 tile_decode: default CDF tables (§9.4.1 / §9.4.2) not populated — \
             required for partition decode at ({sbs_x}×{sbs_y}) {sb_size}×{sb_size} \
             superblocks. Skeleton ready (symbol decoder initialised, frame grid \
             computed); next milestone: build partition / intra-mode default CDFs"
        )))
    }
}

/// Result container for a successfully-decoded frame. Not yet produced by
/// `TileDecoder::decode()` — declared so downstream code can reference the
/// shape.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
}

impl DecodedFrame {
    /// Allocate a `Yuv420P` frame filled with mid-grey. Used as a sanity
    /// fallback — real decode will replace pixels superblock-by-superblock.
    pub fn mid_grey(width: u32, height: u32) -> Self {
        let y_stride = width as usize;
        let uv_w = (width as usize + 1) / 2;
        let uv_h = (height as usize + 1) / 2;
        Self {
            width,
            height,
            y: vec![128; y_stride * height as usize],
            u: vec![128; uv_w * uv_h],
            v: vec![128; uv_w * uv_h],
            y_stride,
            uv_stride: uv_w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_grey_frame_has_correct_dimensions() {
        let f = DecodedFrame::mid_grey(64, 64);
        assert_eq!(f.y.len(), 64 * 64);
        assert_eq!(f.u.len(), 32 * 32);
        assert_eq!(f.v.len(), 32 * 32);
        assert!(f.y.iter().all(|&p| p == 128));
    }

    #[test]
    fn partition_from_u32_accepts_valid() {
        for v in 0u32..=9 {
            Partition::from_u32(v).unwrap();
        }
        assert!(Partition::from_u32(10).is_err());
    }
}
