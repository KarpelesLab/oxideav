//! YCbCr ↔ RGB conversion, JPEG / JFIF colourspace (full-range BT.601).
//!
//! Formulas per JFIF 1.02:
//! ```text
//! Y  =   0.299 R + 0.587 G + 0.114 B
//! Cb = - 0.168736 R - 0.331264 G + 0.5 B + 128
//! Cr =   0.5 R - 0.418688 G - 0.081312 B + 128
//!
//! R = Y                   + 1.402   (Cr - 128)
//! G = Y - 0.344136 (Cb-128) - 0.714136 (Cr - 128)
//! B = Y + 1.772    (Cb-128)
//! ```

/// Convert a single RGB pixel (0..255 each) to 8-bit YCbCr.
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = -0.168_736 * rf - 0.331_264 * gf + 0.5 * bf + 128.0;
    let cr = 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf + 128.0;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

/// Convert a single YCbCr pixel back to 8-bit RGB.
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let cbf = cb as f32 - 128.0;
    let crf = cr as f32 - 128.0;
    let r = yf + 1.402 * crf;
    let g = yf - 0.344_136 * cbf - 0.714_136 * crf;
    let b = yf + 1.772 * cbf;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn red_roundtrip() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
        // BT.601 / JFIF: Y=76, Cb=85, Cr=255.
        assert!((y as i32 - 76).abs() <= 1);
        assert!((cb as i32 - 85).abs() <= 1);
        assert!((cr as i32 - 255).abs() <= 1);
        let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
        assert!(r >= 250);
        assert!(g <= 4);
        assert!(b <= 4);
    }

    #[test]
    fn gray_is_grey() {
        let (y, cb, cr) = rgb_to_ycbcr(128, 128, 128);
        assert_eq!(y, 128);
        assert!((cb as i32 - 128).abs() <= 1);
        assert!((cr as i32 - 128).abs() <= 1);
    }
}
