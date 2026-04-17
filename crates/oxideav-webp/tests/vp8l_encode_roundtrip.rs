//! Encode → decode roundtrip tests for the pure-Rust VP8L encoder.
//!
//! The encoder emits a bare VP8L bitstream (no RIFF wrapper). The
//! existing crate-internal decoder (`vp8l::decode`) takes the same bare
//! bitstream, so we keep the tests self-contained — no `cwebp`/`libwebp`
//! fallbacks required. Each test asserts byte-identical RGBA output
//! after the round trip, since VP8L is lossless.

use oxideav_webp::encode_vp8l_argb;
use oxideav_webp::vp8l;

/// Pack an R,G,B,A byte slice into ARGB u32 pixels (the layout the VP8L
/// encoder expects).
fn rgba_bytes_to_argb_pixels(rgba: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(rgba.len() / 4);
    for p in rgba.chunks_exact(4) {
        let r = p[0] as u32;
        let g = p[1] as u32;
        let b = p[2] as u32;
        let a = p[3] as u32;
        out.push((a << 24) | (r << 16) | (g << 8) | b);
    }
    out
}

/// Encode an RGBA buffer, round-trip it through the VP8L decoder, and
/// assert the emitted RGBA bytes match the input exactly.
fn roundtrip(width: u32, height: u32, rgba: &[u8], has_alpha: bool) {
    assert_eq!(
        rgba.len(),
        (width as usize) * (height as usize) * 4,
        "test rgba buffer has wrong length"
    );
    let pixels = rgba_bytes_to_argb_pixels(rgba);
    let bitstream = encode_vp8l_argb(width, height, &pixels, has_alpha).expect("encode succeeds");
    let decoded = vp8l::decode(&bitstream).expect("decode succeeds");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    let out_rgba = decoded.to_rgba();
    assert_eq!(out_rgba.len(), rgba.len(), "decoded RGBA length mismatch");
    if out_rgba != rgba {
        // Give a more useful message than "slices differ" on failure.
        let mut diff_count = 0usize;
        let mut first_diff = None;
        for (i, (a, b)) in rgba.iter().zip(out_rgba.iter()).enumerate() {
            if a != b {
                diff_count += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, *a, *b));
                }
            }
        }
        panic!("VP8L roundtrip differs at {diff_count} byte(s); first diff @ {first_diff:?}");
    }
}

#[test]
fn vp8l_encode_gradient_128x128() {
    // Build a 128×128 RGBA gradient. Use non-aligned coefficients so the
    // encoder can't accidentally collapse everything to a single colour.
    let w = 128u32;
    let h = 128u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = (x * 2) as u8; // R: horizontal gradient
            rgba[i + 1] = (y * 2) as u8; // G: vertical gradient
            rgba[i + 2] = ((x + y) * 2) as u8; // B: diagonal
            rgba[i + 3] = 0xff; // A: opaque
        }
    }
    roundtrip(w, h, &rgba, false);
}

#[test]
fn vp8l_encode_solid_colour() {
    // 32×32 solid pink — exercises the encoder's degenerate-histogram
    // paths (only one green/red/blue/alpha value ever shows up).
    let w = 32u32;
    let h = 32u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for i in (0..rgba.len()).step_by(4) {
        rgba[i] = 0xFF;
        rgba[i + 1] = 0x66;
        rgba[i + 2] = 0x99;
        rgba[i + 3] = 0xFF;
    }
    roundtrip(w, h, &rgba, false);
}

#[test]
fn vp8l_encode_random_64x64() {
    // Deterministic pseudo-random RGBA — avoids `rand` dep. Uses a
    // tiny xorshift32 PRNG seeded from a constant.
    let w = 64u32;
    let h = 64u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0xC0DE_F00D;
    for b in rgba.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *b = (s & 0xff) as u8;
    }
    // Round-trip; `has_alpha=true` because random bytes will include
    // alpha != 0xff — we want the header flag to reflect reality.
    roundtrip(w, h, &rgba, true);
}

#[test]
fn vp8l_encode_tiny_1x1() {
    // Smallest valid image: 1×1 single pixel. Edge case for the header
    // width/height encoding + the single-symbol Huffman path.
    let rgba = [0x12u8, 0x34, 0x56, 0x78];
    roundtrip(1, 1, &rgba, true);
}

#[test]
fn vp8l_encode_two_pixel_wide() {
    // 2×1 image — LZ77 min-match is 3 pixels, so this is pure literals.
    let rgba = [0x01u8, 0x02, 0x03, 0xff, 0x04, 0x05, 0x06, 0xff];
    roundtrip(2, 1, &rgba, false);
}
