//! VP8L lossless decoder.
//!
//! Implemented per the "WebP Lossless Bitstream Specification" (Google,
//! current as of 2026). The pipeline:
//!
//! 1. Read the 5-byte signature header (`0x2f` + 14-bit width-1 +
//!    14-bit height-1 + alpha_is_used + 3-bit version).
//! 2. Optionally read one or more **transforms** — predictor, colour,
//!    subtract-green, and colour-indexing. Transforms run in reverse on
//!    decode (first transform encountered is the last applied).
//! 3. Read the main image stream: meta Huffman table (entropy image +
//!    per-tile Huffman group selection), colour-cache hash table, then a
//!    prefix-coded stream of
//!    * 0..255: literal green
//!    * 256..279: LZ77 length code (distance follows)
//!    * 280..: colour-cache index
//!
//!    plus Huffman trees for red, blue, alpha, and distance.
//!
//! This module deliberately mirrors the spec's naming to keep cross-
//! reference cheap. Control flow is laid out as three phases —
//! `read_header`, `decode_pixels`, `apply_transforms` — so tests can
//! exercise each independently.

pub mod bit_reader;
pub mod encoder;
pub mod huffman;
pub mod transform;

pub use encoder::encode_vp8l_argb;

use oxideav_core::{Error, Result};

use bit_reader::BitReader;
use huffman::{HuffmanCode, HuffmanTree};
use transform::Transform;

/// The signature byte identifying the start of a VP8L stream.
pub const VP8L_SIGNATURE: u8 = 0x2f;

/// Maximum number of transforms (spec caps the chain at 4).
const MAX_TRANSFORMS: usize = 4;

/// Number of symbols in each Huffman alphabet. See spec §5.
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_DISTANCE_CODES: usize = 40;
// Base green alphabet: 256 literals + 24 length codes. Colour-cache codes
// (0..cache_size) are appended at runtime.
const GREEN_BASE_CODES: usize = NUM_LITERAL_CODES + NUM_LENGTH_CODES;

/// Decoded VP8L image. `pixels` is ARGB-native, one u32 per pixel in raster
/// order (row-major, top-to-bottom). Each pixel stores
/// `(a<<24) | (r<<16) | (g<<8) | b` in memory order matching the spec.
#[derive(Clone, Debug)]
pub struct Vp8lImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
    pub has_alpha: bool,
}

impl Vp8lImage {
    /// Write out the image as packed RGBA bytes (4 bytes/pixel, R,G,B,A)
    /// in row order.
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            let a = ((p >> 24) & 0xff) as u8;
            let r = ((p >> 16) & 0xff) as u8;
            let g = ((p >> 8) & 0xff) as u8;
            let b = (p & 0xff) as u8;
            out.extend_from_slice(&[r, g, b, a]);
        }
        out
    }
}

/// Decode a complete VP8L bitstream.
pub fn decode(data: &[u8]) -> Result<Vp8lImage> {
    let mut br = BitReader::new(data);
    let sig = br.read_bits(8)? as u8;
    if sig != VP8L_SIGNATURE {
        return Err(Error::invalid(format!("VP8L: bad signature 0x{sig:02x}")));
    }
    let width = br.read_bits(14)? + 1;
    let height = br.read_bits(14)? + 1;
    let alpha_is_used = br.read_bits(1)? != 0;
    let version = br.read_bits(3)?;
    if version != 0 {
        return Err(Error::invalid(format!(
            "VP8L: unsupported version {version}"
        )));
    }

    // ── Transforms ───────────────────────────────────────────────────
    // Transforms are read front-to-back, applied back-to-front. Each
    // transform mutates the logical width of the image stream that
    // follows it (colour-indexing packs pixels; everything else is
    // width-neutral).
    let mut transforms: Vec<Transform> = Vec::new();
    let mut xsize = width;
    while br.read_bits(1)? != 0 {
        if transforms.len() >= MAX_TRANSFORMS {
            return Err(Error::invalid("VP8L: too many transforms"));
        }
        let t = Transform::read(&mut br, xsize, height)?;
        xsize = t.image_width_or_default(xsize);
        transforms.push(t);
    }

    // ── Main image stream ────────────────────────────────────────────
    let pixels = decode_image_stream(&mut br, xsize, height, true)?;

    // ── Apply transforms in reverse ──────────────────────────────────
    let mut current = pixels;
    let mut current_w = xsize;
    for t in transforms.iter().rev() {
        let new_w = t.output_width(current_w);
        current = t.apply(&current, current_w, height)?;
        current_w = new_w;
    }
    debug_assert_eq!(current_w, width);
    debug_assert_eq!(current.len() as u32, width * height);

    Ok(Vp8lImage {
        width,
        height,
        pixels: current,
        has_alpha: alpha_is_used,
    })
}

/// Read a meta-Huffman-coded image stream of `width × height` pixels.
///
/// `main_image` = true for the outermost call (can carry a colour cache
/// and meta-huffman-image); transforms that internally call back to
/// `decode_image_stream` pass `false` to get the simpler single-group
/// variant.
pub(crate) fn decode_image_stream(
    br: &mut BitReader<'_>,
    width: u32,
    height: u32,
    main_image: bool,
) -> Result<Vec<u32>> {
    // Colour cache.
    let color_cache_bits = if br.read_bits(1)? != 0 {
        let bits = br.read_bits(4)?;
        if !(1..=11).contains(&bits) {
            return Err(Error::invalid(format!(
                "VP8L: invalid color cache bits {bits}"
            )));
        }
        Some(bits)
    } else {
        None
    };
    let cache_size = color_cache_bits.map(|b| 1u32 << b).unwrap_or(0);

    // Meta-Huffman: only carried by the main image.
    let (meta_image, meta_bits, num_groups) = if main_image && br.read_bits(1)? != 0 {
        let bits = br.read_bits(3)? + 2;
        let meta_w = ((width as i64 + (1 << bits) - 1) >> bits) as u32;
        let meta_h = ((height as i64 + (1 << bits) - 1) >> bits) as u32;
        let meta_pixels = decode_image_stream(br, meta_w, meta_h, false)?;
        let mut ng = 0u32;
        for px in &meta_pixels {
            let id = (px >> 8) & 0xffff;
            if id + 1 > ng {
                ng = id + 1;
            }
        }
        (
            Some(MetaImage {
                pixels: meta_pixels,
                width: meta_w,
            }),
            bits,
            ng.max(1) as usize,
        )
    } else {
        (None, 0, 1)
    };

    // Per-group Huffman trees.
    let mut groups: Vec<HuffmanGroup> = Vec::with_capacity(num_groups);
    for _ in 0..num_groups {
        let g = HuffmanGroup::read(br, cache_size)?;
        groups.push(g);
    }

    // Pixel decode loop.
    let pixel_count = (width as usize) * (height as usize);
    let mut pixels: Vec<u32> = Vec::with_capacity(pixel_count);
    let mut cache: Vec<u32> = if cache_size == 0 {
        Vec::new()
    } else {
        vec![0u32; cache_size as usize]
    };
    let cache_bits = color_cache_bits.unwrap_or(0);
    let mut x: u32 = 0;
    let mut y: u32 = 0;
    while pixels.len() < pixel_count {
        let group_idx = if let Some(mi) = &meta_image {
            let mx = x >> meta_bits;
            let my = y >> meta_bits;
            let idx = (my * mi.width + mx) as usize;
            let px = mi.pixels[idx];
            ((px >> 8) & 0xffff) as usize
        } else {
            0
        };
        let g = &groups[group_idx];
        let code = g.decode_green(br)?;
        if code < NUM_LITERAL_CODES as u32 {
            // Literal green — decode R, B, A separately.
            let green = code & 0xff;
            let red = g.decode_red(br)? & 0xff;
            let blue = g.decode_blue(br)? & 0xff;
            let alpha = g.decode_alpha(br)? & 0xff;
            let argb = (alpha << 24) | (red << 16) | (green << 8) | blue;
            pixels.push(argb);
            cache_add(&mut cache, cache_bits, argb);
            advance_xy(&mut x, &mut y, width);
        } else if code < GREEN_BASE_CODES as u32 {
            // LZ77 backward reference.
            let len_code = code - NUM_LITERAL_CODES as u32;
            let length = decode_length_or_distance(br, len_code)? as usize;
            let dist_code = g.decode_distance(br)?;
            let distance_raw = decode_length_or_distance(br, dist_code)? as usize;
            let distance = map_plane_distance(distance_raw, width as usize);
            if distance == 0 {
                return Err(Error::invalid("VP8L: LZ77 distance = 0"));
            }
            if distance > pixels.len() {
                return Err(Error::invalid("VP8L: LZ77 distance past stream start"));
            }
            for _ in 0..length {
                if pixels.len() >= pixel_count {
                    break;
                }
                let src = pixels[pixels.len() - distance];
                pixels.push(src);
                cache_add(&mut cache, cache_bits, src);
                advance_xy(&mut x, &mut y, width);
            }
        } else {
            // Colour-cache reference.
            if cache_size == 0 {
                return Err(Error::invalid("VP8L: cache code without cache"));
            }
            let idx = code as usize - GREEN_BASE_CODES;
            if idx >= cache.len() {
                return Err(Error::invalid("VP8L: cache index out of range"));
            }
            let argb = cache[idx];
            pixels.push(argb);
            advance_xy(&mut x, &mut y, width);
        }
    }

    Ok(pixels)
}

/// Decode a "length or distance" symbol per spec §5.2.2. For symbols 0..3
/// the value is just `symbol + 1`; larger symbols expand via extra bits.
fn decode_length_or_distance(br: &mut BitReader<'_>, symbol: u32) -> Result<u32> {
    if symbol < 4 {
        Ok(symbol + 1)
    } else {
        let extra_bits = (symbol - 2) >> 1;
        let offset = (2 + (symbol & 1)) << extra_bits;
        let extra = br.read_bits(extra_bits as u8)?;
        Ok(offset + extra + 1)
    }
}

/// Plane-distance codes 1..120 map to an (xi, yi) neighbour offset per the
/// spec's Table 2; everything above 120 subtracts 120 for the raw
/// distance. The returned value is the flat "number of pixels behind"
/// count used for the LZ77 backwards copy.
fn map_plane_distance(code: usize, width: usize) -> usize {
    if code == 0 {
        return 0;
    }
    if code > 120 {
        return code - 120;
    }
    let (xi, yi) = PLANE_DIST[code - 1];
    let d = (yi as isize) * (width as isize) + (xi as isize);
    if d < 1 {
        1
    } else {
        d as usize
    }
}

/// Lookup table for short-distance plane codes (code-1 → (xi, yi)). Taken
/// from the VP8L spec §5.2.2, re-derived from the formula
/// `(xi, yi) = (dx-8, dy)` over the first 120 diamond-ordered neighbours.
#[rustfmt::skip]
const PLANE_DIST: [(i8, i8); 120] = [
    (0,1), (1,0), (1,1), (-1,1), (0,2), (2,0), (1,2), (-1,2),
    (2,1), (-2,1), (2,2), (-2,2), (0,3), (3,0), (1,3), (-1,3),
    (3,1), (-3,1), (2,3), (-2,3), (3,2), (-3,2), (0,4), (4,0),
    (1,4), (-1,4), (4,1), (-4,1), (3,3), (-3,3), (2,4), (-2,4),
    (4,2), (-4,2), (0,5), (3,4), (-3,4), (4,3), (-4,3), (5,0),
    (1,5), (-1,5), (5,1), (-5,1), (2,5), (-2,5), (5,2), (-5,2),
    (4,4), (-4,4), (3,5), (-3,5), (5,3), (-5,3), (0,6), (6,0),
    (1,6), (-1,6), (6,1), (-6,1), (2,6), (-2,6), (6,2), (-6,2),
    (4,5), (-4,5), (5,4), (-5,4), (3,6), (-3,6), (6,3), (-6,3),
    (0,7), (7,0), (1,7), (-1,7), (5,5), (-5,5), (7,1), (-7,1),
    (4,6), (-4,6), (6,4), (-6,4), (2,7), (-2,7), (7,2), (-7,2),
    (3,7), (-3,7), (7,3), (-7,3), (5,6), (-5,6), (6,5), (-6,5),
    (8,0), (4,7), (-4,7), (7,4), (-7,4), (8,1), (8,2), (6,6),
    (-6,6), (8,3), (5,7), (-5,7), (7,5), (-7,5), (8,4), (6,7),
    (-6,7), (7,6), (-7,6), (8,5), (7,7), (-7,7), (8,6), (8,7),
];

fn cache_add(cache: &mut [u32], cache_bits: u32, argb: u32) {
    if cache.is_empty() {
        return;
    }
    // Spec hash: (0x1e35a7bd * argb) >> (32 - cache_bits)
    let idx = 0x1e35_a7bd_u32.wrapping_mul(argb) >> (32 - cache_bits);
    let idx = idx as usize;
    if idx < cache.len() {
        cache[idx] = argb;
    }
}

fn advance_xy(x: &mut u32, y: &mut u32, width: u32) {
    *x += 1;
    if *x >= width {
        *x = 0;
        *y += 1;
    }
}

struct MetaImage {
    pixels: Vec<u32>,
    width: u32,
}

/// A single "Huffman group" — five trees covering green/length/cache,
/// red, blue, alpha, and distance symbols.
pub struct HuffmanGroup {
    green: HuffmanTree,
    red: HuffmanTree,
    blue: HuffmanTree,
    alpha: HuffmanTree,
    distance: HuffmanTree,
}

impl HuffmanGroup {
    fn read(br: &mut BitReader<'_>, cache_size: u32) -> Result<Self> {
        let green_alpha = GREEN_BASE_CODES + cache_size as usize;
        let green = HuffmanTree::read(br, green_alpha)?;
        let red = HuffmanTree::read(br, NUM_LITERAL_CODES)?;
        let blue = HuffmanTree::read(br, NUM_LITERAL_CODES)?;
        let alpha = HuffmanTree::read(br, NUM_LITERAL_CODES)?;
        let distance = HuffmanTree::read(br, NUM_DISTANCE_CODES)?;
        Ok(Self {
            green,
            red,
            blue,
            alpha,
            distance,
        })
    }

    fn decode_green(&self, br: &mut BitReader<'_>) -> Result<u32> {
        self.green.decode(br).map(|c: HuffmanCode| c as u32)
    }
    fn decode_red(&self, br: &mut BitReader<'_>) -> Result<u32> {
        self.red.decode(br).map(|c: HuffmanCode| c as u32)
    }
    fn decode_blue(&self, br: &mut BitReader<'_>) -> Result<u32> {
        self.blue.decode(br).map(|c: HuffmanCode| c as u32)
    }
    fn decode_alpha(&self, br: &mut BitReader<'_>) -> Result<u32> {
        self.alpha.decode(br).map(|c: HuffmanCode| c as u32)
    }
    fn decode_distance(&self, br: &mut BitReader<'_>) -> Result<u32> {
        self.distance.decode(br).map(|c: HuffmanCode| c as u32)
    }
}
