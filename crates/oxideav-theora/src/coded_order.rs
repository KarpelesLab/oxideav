//! Coded-order block index mapping.
//!
//! Theora accesses blocks in "coded order": super-blocks raster-ordered, then
//! within each super-block the four macro-blocks follow a Hilbert curve, and
//! within each macro-block area (2x2 blocks in the luma plane) or 4x4 block
//! area (per super-block for any plane) the blocks follow the 16-element
//! Hilbert curve shown in spec Figure 2.4. Indices continue across color
//! planes (luma, Cb, Cr).
//!
//! Importantly, Theora uses a bottom-left origin (spec §2.1). So "row 0" is
//! the bottom of the plane, and coordinates increase upward.
//!
//! This module provides helpers to compute, per plane, the tables:
//!
//! * `coded_to_raster[ci]` — for each coded index within the plane, the
//!   `(bx, by)` 8×8-block coordinate in the plane (by is from the bottom).
//! * `raster_to_coded[by * nbw + bx]` — inverse mapping.
//!
//! For DC prediction we need raster traversal (left/down/right-down neighbour
//! lookups). For token decode we need coded-order traversal. We provide both.

use crate::headers::PixelFormat;

/// Hilbert curve order of the 16 blocks inside a super-block (spec Figure 2.4,
/// expressed with bottom-left origin). Entry `HILBERT_XY[i] = (x, y)` gives
/// the 0..=3 column and row of the block at coded-position `i` within a super
/// block that spans 4x4 blocks.
pub const HILBERT_XY: [(u8, u8); 16] = [
    // From spec Figure 2.4 (bottom-left origin). Reading the numbers that
    // form the Hilbert order: indices 0..15 in the diagram correspond to:
    //   0:(0,0)  1:(1,0)  2:(1,1)  3:(0,1)
    //   4:(0,2)  5:(0,3)  6:(1,3)  7:(1,2)
    //   8:(2,2)  9:(2,3) 10:(3,3) 11:(3,2)
    //  12:(3,1) 13:(2,1) 14:(2,0) 15:(3,0)
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (1, 2),
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 2),
    (3, 1),
    (2, 1),
    (2, 0),
    (3, 0),
];

/// Size (width, height) of one plane in 8x8 blocks.
pub fn plane_block_dims(fmbw: u32, fmbh: u32, pf: PixelFormat, pli: usize) -> (u32, u32) {
    if pli == 0 {
        (fmbw * 2, fmbh * 2)
    } else {
        match pf {
            PixelFormat::Yuv420 => (fmbw, fmbh),
            PixelFormat::Yuv422 => (fmbw, fmbh * 2),
            PixelFormat::Yuv444 | PixelFormat::Reserved => (fmbw * 2, fmbh * 2),
        }
    }
}

/// Size (width, height) of one plane in pixels.
pub fn plane_pixel_dims(fmbw: u32, fmbh: u32, pf: PixelFormat, pli: usize) -> (u32, u32) {
    let (bw, bh) = plane_block_dims(fmbw, fmbh, pf, pli);
    (bw * 8, bh * 8)
}

/// Size (width, height) of one plane in super-blocks (4x4 block units). Rounded
/// up — partial super-blocks at the top/right are allowed.
pub fn plane_sb_dims(fmbw: u32, fmbh: u32, pf: PixelFormat, pli: usize) -> (u32, u32) {
    let (bw, bh) = plane_block_dims(fmbw, fmbh, pf, pli);
    (bw.div_ceil(4), bh.div_ceil(4))
}

/// Per-plane layout tables.
#[derive(Clone, Debug)]
pub struct PlaneLayout {
    /// Plane width in 8×8 blocks.
    pub nbw: u32,
    /// Plane height in 8×8 blocks.
    pub nbh: u32,
    /// Mapping coded_index → raster index (`bx + by * nbw`). Length = nbw*nbh.
    pub coded_to_raster: Vec<u32>,
    /// Mapping raster_index → coded_index (in-plane). Length = nbw*nbh.
    pub raster_to_coded: Vec<u32>,
}

impl PlaneLayout {
    pub fn new(fmbw: u32, fmbh: u32, pf: PixelFormat, pli: usize) -> Self {
        let (nbw, nbh) = plane_block_dims(fmbw, fmbh, pf, pli);
        let (sbw, sbh) = plane_sb_dims(fmbw, fmbh, pf, pli);
        let total = (nbw * nbh) as usize;
        let mut coded_to_raster = Vec::with_capacity(total);

        // Traverse super-blocks in raster order (bottom-left origin), then
        // blocks within each super-block in Hilbert order.
        for sby in 0..sbh {
            for sbx in 0..sbw {
                for &(dx, dy) in &HILBERT_XY {
                    let bx = sbx * 4 + dx as u32;
                    let by = sby * 4 + dy as u32;
                    if bx < nbw && by < nbh {
                        let raster = by * nbw + bx;
                        coded_to_raster.push(raster);
                    }
                }
            }
        }
        assert_eq!(coded_to_raster.len(), total);

        let mut raster_to_coded = vec![0u32; total];
        for (ci, &raster) in coded_to_raster.iter().enumerate() {
            raster_to_coded[raster as usize] = ci as u32;
        }

        Self {
            nbw,
            nbh,
            coded_to_raster,
            raster_to_coded,
        }
    }

    /// Get (bx, by) from coded-order index within this plane.
    pub fn coded_xy(&self, ci: u32) -> (u32, u32) {
        let r = self.coded_to_raster[ci as usize];
        (r % self.nbw, r / self.nbw)
    }
}

/// Full frame block layout across all three planes, with global coded indices.
#[derive(Clone, Debug)]
pub struct FrameLayout {
    pub planes: [PlaneLayout; 3],
    /// Starting global coded-index for each plane.
    pub plane_offsets: [u32; 3],
    /// Total number of blocks across all planes.
    pub nbs: u32,
    pub fmbw: u32,
    pub fmbh: u32,
    pub pf: PixelFormat,
}

impl FrameLayout {
    pub fn new(fmbw: u32, fmbh: u32, pf: PixelFormat) -> Self {
        let planes = [
            PlaneLayout::new(fmbw, fmbh, pf, 0),
            PlaneLayout::new(fmbw, fmbh, pf, 1),
            PlaneLayout::new(fmbw, fmbh, pf, 2),
        ];
        let n0 = planes[0].nbw * planes[0].nbh;
        let n1 = planes[1].nbw * planes[1].nbh;
        let n2 = planes[2].nbw * planes[2].nbh;
        let plane_offsets = [0, n0, n0 + n1];
        let nbs = n0 + n1 + n2;
        Self {
            planes,
            plane_offsets,
            nbs,
            fmbw,
            fmbh,
            pf,
        }
    }

    /// Identify which plane a global coded-order index belongs to.
    pub fn plane_of(&self, bi: u32) -> usize {
        if bi < self.plane_offsets[1] {
            0
        } else if bi < self.plane_offsets[2] {
            1
        } else {
            2
        }
    }

    /// Convert global coded-order index to (plane, bx, by).
    pub fn global_xy(&self, bi: u32) -> (usize, u32, u32) {
        let pli = self.plane_of(bi);
        let in_plane = bi - self.plane_offsets[pli];
        let (bx, by) = self.planes[pli].coded_xy(in_plane);
        (pli, bx, by)
    }

    /// Convert (plane, bx, by) to global coded-order index.
    pub fn global_coded(&self, pli: usize, bx: u32, by: u32) -> u32 {
        let plane = &self.planes[pli];
        let raster = by * plane.nbw + bx;
        self.plane_offsets[pli] + plane.raster_to_coded[raster as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hilbert_is_permutation() {
        let mut seen = [false; 16];
        for &(x, y) in &HILBERT_XY {
            let idx = (y as usize) * 4 + x as usize;
            assert!(!seen[idx], "duplicate in Hilbert order");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn layout_total_matches_dimensions() {
        // 64x48 Yuv420P: luma 8x6, chroma 4x3 each.
        // fmbw=4, fmbh=3.
        let f = FrameLayout::new(4, 3, PixelFormat::Yuv420);
        assert_eq!(f.planes[0].nbw, 8);
        assert_eq!(f.planes[0].nbh, 6);
        assert_eq!(f.planes[1].nbw, 4);
        assert_eq!(f.planes[1].nbh, 3);
        assert_eq!(f.planes[2].nbw, 4);
        assert_eq!(f.planes[2].nbh, 3);
        assert_eq!(f.nbs, 48 + 12 + 12);
        // Round-trip a few coordinates.
        for bi in 0..f.nbs {
            let (pli, bx, by) = f.global_xy(bi);
            assert_eq!(f.global_coded(pli, bx, by), bi);
        }
    }

    #[test]
    fn coded_order_matches_spec_example() {
        // Spec §2.3: 240x48 luma plane -> 30 blocks wide, 6 blocks tall.
        // First super-block (sbx=0, sby=0, covers blocks bx 0..3, by 0..3):
        // coded indices 0..15 should map to:
        //   y=3: bx 0:5, bx 1:6, bx 2:9, bx 3:10
        //   y=2: bx 0:4, bx 1:7, bx 2:8, bx 3:11
        //   y=1: bx 0:3, bx 1:2, bx 2:13, bx 3:12
        //   y=0: bx 0:0, bx 1:1, bx 2:14, bx 3:15
        let f = FrameLayout::new(15, 3, PixelFormat::Yuv420);
        // block at (bx=0, by=0) in luma should have coded index 0.
        assert_eq!(f.global_coded(0, 0, 0), 0);
        assert_eq!(f.global_coded(0, 1, 0), 1);
        assert_eq!(f.global_coded(0, 0, 1), 3);
        assert_eq!(f.global_coded(0, 1, 1), 2);
        assert_eq!(f.global_coded(0, 0, 2), 4);
        assert_eq!(f.global_coded(0, 0, 3), 5);
        assert_eq!(f.global_coded(0, 1, 3), 6);
        assert_eq!(f.global_coded(0, 3, 0), 15);
        assert_eq!(f.global_coded(0, 3, 3), 10);
    }
}
