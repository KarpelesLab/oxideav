//! VP9 tile / partition / block decode — skeleton.
//!
//! Reference: VP9 spec §6.4 (`decode_tiles`), §6.4.1 (`decode_tile`),
//! §6.4.2 (`decode_partition`), §6.4.3 (`decode_block`), §7.4 (block-level
//! semantic process), §8.5 (intra prediction), §8.6 (inter prediction),
//! §8.7 (reconstruction), §8.8 (loop filter).
//!
//! Status: this module wires up the machinery — bool decoder per tile,
//! tile-grid walk, superblock counting — and stops at the first piece of
//! tile syntax we can't yet consume (§6.4.3 partition decode). Each
//! boundary raises an `Error::Unsupported` with a precise §ref so future
//! work can pick up exactly where we stop.
//!
//! The intra primitives and inverse transforms landed alongside this are
//! available as standalone primitives via `crate::intra` and
//! `crate::transform`, ready to be wired in once partition / mode /
//! coefficient decoding lands.
//!
//! Structural parallel to `oxideav_av1::tile_decode` — the two crates
//! mirror each other on purpose.

use oxideav_core::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::compressed_header::CompressedHeader;
use crate::headers::UncompressedHeader;

/// VP9 superblock size is always 64×64 (§3).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// Per-superblock partition decisions — §6.4.2 Table 9-5.
///
/// The `PARTITION_NONE` / `PARTITION_HORZ` / `PARTITION_VERT` /
/// `PARTITION_SPLIT` enumeration is reused at every level of the partition
/// quadtree (§6.4.2).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
}

impl Partition {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::None,
            1 => Self::Horz,
            2 => Self::Vert,
            3 => Self::Split,
            _ => return Err(Error::invalid(format!("vp9 partition: invalid {v}"))),
        })
    }
}

/// Tile-grid geometry derived from the uncompressed header's tile_info
/// field (§6.2.6).
#[derive(Clone, Copy, Debug)]
pub struct TileGrid {
    pub tile_cols: u32,
    pub tile_rows: u32,
    /// Total frame width in 8-pixel mi units (`MiCols` in spec parlance).
    pub mi_cols: u32,
    /// Total frame height in 8-pixel mi units (`MiRows`).
    pub mi_rows: u32,
    /// Total superblocks in the frame.
    pub sbs_x: u32,
    pub sbs_y: u32,
}

impl TileGrid {
    pub fn from_header(hdr: &UncompressedHeader) -> Self {
        let tile_cols = 1u32 << hdr.tile_info.log2_tile_cols as u32;
        let tile_rows = 1u32 << hdr.tile_info.log2_tile_rows as u32;
        // Spec §7.2 — MiCols = ALIGN(width, 8) / 8, MiRows = ALIGN(height, 8) / 8.
        let mi_cols = (hdr.width + 7) / 8;
        let mi_rows = (hdr.height + 7) / 8;
        let sbs_x = (hdr.width + SUPERBLOCK_SIZE - 1) / SUPERBLOCK_SIZE;
        let sbs_y = (hdr.height + SUPERBLOCK_SIZE - 1) / SUPERBLOCK_SIZE;
        Self {
            tile_cols,
            tile_rows,
            mi_cols,
            mi_rows,
            sbs_x,
            sbs_y,
        }
    }
}

/// A single-tile decode context. In VP9 this would be the `decode_tile()`
/// procedure of §6.4.1 — in this skeleton we only initialise the bool
/// decoder, log superblock iteration, and surface a precise `Unsupported`
/// at the first piece of syntax we can't consume.
pub struct TileDecoder<'a> {
    pub hdr: &'a UncompressedHeader,
    pub ch: &'a CompressedHeader,
    pub bool_dec: BoolDecoder<'a>,
    /// Tile's column index within the tile grid.
    pub tile_col: u32,
    /// Tile's row index within the tile grid.
    pub tile_row: u32,
}

impl<'a> TileDecoder<'a> {
    /// Begin decoding a tile whose compressed payload is `tile_data`.
    pub fn new(
        hdr: &'a UncompressedHeader,
        ch: &'a CompressedHeader,
        tile_data: &'a [u8],
        tile_col: u32,
        tile_row: u32,
    ) -> Result<Self> {
        let bool_dec = BoolDecoder::new(tile_data)?;
        Ok(Self {
            hdr,
            ch,
            bool_dec,
            tile_col,
            tile_row,
        })
    }

    /// Iterate the tile's superblocks and stop at the first piece of tile
    /// syntax we don't yet handle (partition decode, §6.4.3).
    pub fn decode(&mut self) -> Result<()> {
        // Walking the first superblock is the most informative boundary: it
        // forces us to decode a partition symbol, which in turn requires the
        // default partition probability tables (§10.5).
        Err(Error::unsupported(format!(
            "vp9 tile_decode: partition syntax §6.4.3 not implemented \
             (tile={},{}); range decode + intra primitives (DC/V/H) + \
             iDCT 4×4/8×8 are available as primitives — next milestone: \
             default partition probs (§10.5) + decode_partition (§6.4.2)",
            self.tile_col, self.tile_row,
        )))
    }
}

/// Walk the tile / partition / block tree per §6.4. The compressed header
/// must already have been parsed by the caller. Currently this computes
/// the tile grid, initialises a `TileDecoder` for the first tile, and
/// surfaces a precise `Unsupported` pointing at the next unimplemented
/// clause.
pub fn decode_tiles(
    tile_payload: &[u8],
    hdr: &UncompressedHeader,
    ch: &CompressedHeader,
) -> Result<()> {
    let grid = TileGrid::from_header(hdr);
    if grid.sbs_x == 0 || grid.sbs_y == 0 {
        return Err(Error::invalid(
            "vp9 decode_tiles: zero-sized frame — impossible per §6.2.2",
        ));
    }
    if tile_payload.is_empty() {
        return Err(Error::invalid(
            "vp9 decode_tiles: tile payload empty — §6.4",
        ));
    }
    // For now we only try to enter the first tile; the tile-size prefix
    // parsing for subsequent tiles is simple (§6.4) but irrelevant until
    // the first tile actually decodes. Stop at the first partition symbol.
    let mut td = TileDecoder::new(hdr, ch, tile_payload, 0, 0)?;
    td.decode()
}

/// Recurse into a single 64×64 superblock per §6.4.2 — stub. Exposed so
/// unit tests / higher layers can poke at the entry point without standing
/// up the full `TileDecoder`.
pub fn decode_partition(
    _bd: &mut BoolDecoder<'_>,
    _row: u32,
    _col: u32,
    _sb_size: u32,
) -> Result<()> {
    Err(Error::unsupported(
        "vp9 §6.4.2 decode_partition: partition quadtree not implemented \
         (needs default partition probability tables, §10.5)",
    ))
}

/// Decode one block per §6.4.3 — stub.
pub fn decode_block(_bd: &mut BoolDecoder<'_>, _row: u32, _col: u32, _bsize: u32) -> Result<()> {
    Err(Error::unsupported(
        "vp9 §6.4.3 decode_block: block decode (residual + prediction + \
         reconstruction) not implemented",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::{
        ColorConfig, ColorSpace, FrameType, LoopFilterParams, QuantizationParams,
        SegmentationParams, TileInfo, UncompressedHeader,
    };

    fn synth_header(width: u32, height: u32) -> UncompressedHeader {
        UncompressedHeader {
            profile: 0,
            show_existing_frame: false,
            existing_frame_to_show: 0,
            frame_type: FrameType::Key,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            reset_frame_context: 0,
            color_config: ColorConfig {
                bit_depth: 8,
                color_space: ColorSpace::Bt709,
                color_range: false,
                subsampling_x: true,
                subsampling_y: true,
            },
            width,
            height,
            render_width: None,
            render_height: None,
            refresh_frame_flags: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interpolation_filter: 0,
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams::default(),
            quantization: QuantizationParams::default(),
            segmentation: SegmentationParams::default(),
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            header_size: 0,
            uncompressed_header_size: 0,
        }
    }

    #[test]
    fn tile_grid_64x64_is_one_superblock() {
        let h = synth_header(64, 64);
        let g = TileGrid::from_header(&h);
        assert_eq!(g.tile_cols, 1);
        assert_eq!(g.tile_rows, 1);
        assert_eq!(g.sbs_x, 1);
        assert_eq!(g.sbs_y, 1);
        assert_eq!(g.mi_cols, 8);
        assert_eq!(g.mi_rows, 8);
    }

    #[test]
    fn tile_grid_128x96_rounds_up() {
        let h = synth_header(128, 96);
        let g = TileGrid::from_header(&h);
        assert_eq!(g.sbs_x, 2);
        assert_eq!(g.sbs_y, 2);
    }

    #[test]
    fn decode_tiles_surfaces_partition_unsupported() {
        let h = synth_header(64, 64);
        let ch = CompressedHeader::default();
        // A byte buffer that the bool decoder can initialise from.
        let payload = [0xAB, 0x00, 0xCD, 0xEF];
        match decode_tiles(&payload, &h, &ch) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains("§6.4.3"), "msg should cite §6.4.3: {s}");
                assert!(s.contains("partition"), "msg should mention partition: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn partition_from_u32_accepts_valid() {
        for v in 0u32..=3 {
            Partition::from_u32(v).unwrap();
        }
        assert!(Partition::from_u32(4).is_err());
    }
}
