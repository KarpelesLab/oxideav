//! Palette types and quantisation strategies.
//!
//! A [`Palette`] is a flat list of RGBA colours (up to 256 entries).
//! [`generate_palette`] is the public entry point for building one from
//! a batch of source [`VideoFrame`](oxideav_core::VideoFrame)s. Two
//! strategies are supported at v1:
//!
//! - [`PaletteStrategy::Uniform`] — fixed 3-3-2 cube, 256 entries.
//! - [`PaletteStrategy::MedianCut`] — Heckbert's 1982 median-cut
//!   scheme. Starts with one box containing every sampled colour and
//!   repeatedly splits the box with the largest range on the widest
//!   axis until `max_colors` boxes are left.
//!
//! [`PaletteStrategy::Octree`] is reserved and returns `Unsupported`
//! for v1.

use oxideav_core::{Error, PixelFormat, Result, VideoFrame};

/// An indexed-colour palette.
#[derive(Clone, Debug, Default)]
pub struct Palette {
    /// RGBA colour entries. Typically ≤ 256.
    pub colors: Vec<[u8; 4]>,
}

/// Palette-generation strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteStrategy {
    MedianCut,
    /// Reserved for v2 — returns `Unsupported` in v1.
    Octree,
    /// 3-3-2 uniform cube (or nearest fit to `max_colors`).
    Uniform,
}

/// Options governing [`generate_palette`].
#[derive(Clone, Debug)]
pub struct PaletteGenOptions {
    pub strategy: PaletteStrategy,
    /// Maximum palette entries. A value of 0 is treated as 1.
    pub max_colors: u8,
    /// If set, the resulting palette will include an entry with the
    /// given index reserved for transparency (alpha = 0).
    pub transparency: Option<u8>,
}

impl Default for PaletteGenOptions {
    fn default() -> Self {
        Self {
            strategy: PaletteStrategy::MedianCut,
            max_colors: 255,
            transparency: None,
        }
    }
}

/// Build a palette from a batch of frames. Every frame must be
/// `Rgb24` or `Rgba` — use [`crate::convert`] to stage through one
/// of those first.
pub fn generate_palette(frames: &[&VideoFrame], opts: &PaletteGenOptions) -> Result<Palette> {
    if frames.is_empty() {
        return Err(Error::invalid("generate_palette: no frames"));
    }
    let pixels = collect_pixels(frames)?;
    let max = opts.max_colors.max(1) as usize;

    let mut colors = match opts.strategy {
        PaletteStrategy::MedianCut => median_cut(&pixels, max),
        PaletteStrategy::Uniform => uniform_palette(max),
        PaletteStrategy::Octree => {
            return Err(Error::unsupported(
                "palette: octree strategy not implemented",
            ))
        }
    };

    if let Some(idx) = opts.transparency {
        let i = idx as usize;
        if i < colors.len() {
            colors[i] = [0, 0, 0, 0];
        } else {
            colors.push([0, 0, 0, 0]);
        }
    }

    Ok(Palette { colors })
}

/// Gather tightly packed (R, G, B, A) pixels from each frame, dropping
/// stride padding.
fn collect_pixels(frames: &[&VideoFrame]) -> Result<Vec<[u8; 4]>> {
    let mut out = Vec::new();
    for frame in frames {
        let w = frame.width as usize;
        let h = frame.height as usize;
        match frame.format {
            PixelFormat::Rgb24 => {
                let plane = &frame.planes[0];
                for row in 0..h {
                    let off = row * plane.stride;
                    for col in 0..w {
                        out.push([
                            plane.data[off + col * 3],
                            plane.data[off + col * 3 + 1],
                            plane.data[off + col * 3 + 2],
                            255,
                        ]);
                    }
                }
            }
            PixelFormat::Rgba => {
                let plane = &frame.planes[0];
                for row in 0..h {
                    let off = row * plane.stride;
                    for col in 0..w {
                        out.push([
                            plane.data[off + col * 4],
                            plane.data[off + col * 4 + 1],
                            plane.data[off + col * 4 + 2],
                            plane.data[off + col * 4 + 3],
                        ]);
                    }
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "generate_palette: frames must be Rgb24 or Rgba, got {other:?}"
                )));
            }
        }
    }
    Ok(out)
}

/// Heckbert median-cut quantisation on an opaque RGB triple set (alpha
/// is preserved from the first sampled pixel in each leaf box).
fn median_cut(pixels: &[[u8; 4]], max_colors: usize) -> Vec<[u8; 4]> {
    if pixels.is_empty() {
        return Vec::new();
    }

    // Start with one box holding every sampled colour.
    let mut boxes: Vec<Box3> = vec![Box3::from(pixels)];
    while boxes.len() < max_colors {
        // Pick the box with the largest single-axis range.
        let idx = match boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .max_by_key(|(_, b)| b.max_range())
        {
            Some((i, _)) => i,
            None => break, // every remaining box is already a single colour
        };
        let taken = boxes.swap_remove(idx);
        let (a, b) = taken.split();
        boxes.push(a);
        boxes.push(b);
    }

    // Output each box as the average of its colours.
    boxes.iter().map(|b| b.average()).collect()
}

struct Box3 {
    colors: Vec<[u8; 4]>,
}

impl Box3 {
    fn from(p: &[[u8; 4]]) -> Self {
        Self { colors: p.to_vec() }
    }

    fn max_range(&self) -> i32 {
        let (rmin, rmax) = self.range(0);
        let (gmin, gmax) = self.range(1);
        let (bmin, bmax) = self.range(2);
        (rmax - rmin).max(gmax - gmin).max(bmax - bmin)
    }

    fn range(&self, c: usize) -> (i32, i32) {
        let mut lo = 255i32;
        let mut hi = 0i32;
        for p in &self.colors {
            let v = p[c] as i32;
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    }

    fn widest_axis(&self) -> usize {
        let mut best = 0usize;
        let mut best_range = -1i32;
        for c in 0..3 {
            let (lo, hi) = self.range(c);
            if hi - lo > best_range {
                best_range = hi - lo;
                best = c;
            }
        }
        best
    }

    fn split(self) -> (Self, Self) {
        let axis = self.widest_axis();
        let mut colors = self.colors;
        colors.sort_unstable_by_key(|p| p[axis]);
        let mid = colors.len() / 2;
        let b = colors.split_off(mid);
        (Self { colors }, Self { colors: b })
    }

    fn average(&self) -> [u8; 4] {
        if self.colors.is_empty() {
            return [0, 0, 0, 255];
        }
        let mut sum = [0u64; 4];
        for p in &self.colors {
            for c in 0..4 {
                sum[c] += p[c] as u64;
            }
        }
        let n = self.colors.len() as u64;
        [
            (sum[0] / n) as u8,
            (sum[1] / n) as u8,
            (sum[2] / n) as u8,
            (sum[3] / n) as u8,
        ]
    }
}

/// Uniform 3-3-2 RGB cube (or truncated to `max` entries).
fn uniform_palette(max: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(256);
    for r in 0..8u8 {
        for g in 0..8u8 {
            for b in 0..4u8 {
                // Spread the 3 or 2 bits evenly over 0..=255.
                let rr = (r as u32 * 255 / 7) as u8;
                let gg = (g as u32 * 255 / 7) as u8;
                let bb = (b as u32 * 255 / 3) as u8;
                out.push([rr, gg, bb, 255]);
                if out.len() >= max {
                    return out;
                }
            }
        }
    }
    out
}
