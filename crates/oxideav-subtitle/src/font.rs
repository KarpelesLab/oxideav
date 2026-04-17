//! Embedded, pure-Rust 8×16 bitmap font for rendering subtitle text.
//!
//! We ship two faces (regular + bold) as `const` byte arrays. Each glyph
//! is 8 columns wide × 16 rows tall, one row per byte, MSB = leftmost
//! pixel. An italic face is synthesised at blit time by horizontally
//! shearing each glyph row (one pixel every four rows).
//!
//! Coverage: all of printable ASCII (0x20..=0x7E) and the Latin-1
//! supplement (0xA1..=0xFF) — roughly 190 glyphs.
//!
//! The glyph shapes are hand-authored specifically for this crate from
//! a plain ASCII-art description (see [`GLYPH_ASCII`]). They are not a
//! verbatim copy of the public-domain VGA ROM font — we chose to avoid
//! the dependency on an external file and keep everything self-contained
//! in the source.
//!
//! Unmapped codepoints render as a hollow rectangle (the "missing glyph"
//! box), matching long-standing font convention.
//!
//! No TrueType parsing. No external crates. No CJK / BiDi / shaping — a
//! nicer text pipeline can land later; this module exists so a subtitle
//! decoder can always produce _something_ viewable without bringing in
//! heavyweight typography tooling.

/// A bitmap font face: fixed cell size, one glyph per codepoint in a
/// precomputed table.
///
/// `cell_w`/`cell_h` describe the per-glyph bounding box in pixels.
/// `bearing_y` is the number of rows from the top of the cell down to the
/// baseline — `draw_glyph`'s `y` argument is interpreted as the baseline
/// Y coordinate, matching what callers typically want.
pub struct BitmapFont {
    pub cell_w: u32,
    pub cell_h: u32,
    pub bearing_y: u32,
    // Private: one entry per codepoint, 16 bytes = 16 rows. Only the
    // low 256 codepoints are addressable; anything else hits the
    // missing-glyph fallback.
    table: &'static [[u8; 16]; 256],
}

impl BitmapFont {
    /// Regular face.
    pub fn default_regular() -> &'static BitmapFont {
        &REGULAR_FONT
    }

    /// Bold face. Same glyph shapes as regular, but each lit pixel is
    /// also lit in the column to its right (the classic "smear bold"
    /// used by VGA text mode).
    pub fn default_bold() -> &'static BitmapFont {
        &BOLD_FONT
    }

    /// Rasterise one glyph into a dst RGBA buffer at `(x, y)`, where `y`
    /// is the baseline. Returns the horizontal advance in pixels
    /// (= `cell_w`). Pixels outside the destination rectangle are
    /// clipped. Blending is straight alpha over.
    pub fn draw_glyph(
        &self,
        ch: char,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        x: i32,
        y: i32,
        color: [u8; 4],
    ) -> u32 {
        self.draw_glyph_sheared(ch, dst, dst_w, dst_h, x, y, color, 0.0)
    }

    /// Variant used by the compositor for italic rendering: shear the
    /// glyph horizontally by `shear` pixels per row counted from the
    /// baseline (positive = top leans right). The caller is expected to
    /// use `cell_w / 4.0` for a light slant.
    pub(crate) fn draw_glyph_sheared(
        &self,
        ch: char,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        x: i32,
        y: i32,
        color: [u8; 4],
        shear: f32,
    ) -> u32 {
        let rows = self.glyph_rows(ch);
        let top = y - self.bearing_y as i32;
        for (row_idx, &bits) in rows.iter().enumerate() {
            if bits == 0 {
                continue;
            }
            // Shear offset: rows above the baseline move right, rows
            // below move left. Bearing_y is the baseline-to-top span.
            let from_baseline = self.bearing_y as i32 - row_idx as i32;
            let dx = (from_baseline as f32 * shear / self.cell_h as f32).round() as i32;
            let py = top + row_idx as i32;
            if py < 0 || (py as u32) >= dst_h {
                continue;
            }
            for col in 0..self.cell_w as i32 {
                if (bits >> (7 - col)) & 1 == 0 {
                    continue;
                }
                let px = x + col + dx;
                if px < 0 || (px as u32) >= dst_w {
                    continue;
                }
                blend_pixel(dst, dst_w, px as u32, py as u32, color);
            }
        }
        self.cell_w
    }

    /// Measure the advance (in pixels) of a single character. For a
    /// fixed-width bitmap font this is trivially `cell_w`.
    pub fn advance(&self, _ch: char) -> u32 {
        self.cell_w
    }

    fn glyph_rows(&self, ch: char) -> [u8; 16] {
        let cp = ch as u32;
        if cp < 256 {
            self.table[cp as usize]
        } else {
            MISSING_GLYPH
        }
    }
}

fn blend_pixel(dst: &mut [u8], dst_w: u32, x: u32, y: u32, color: [u8; 4]) {
    let idx = (y as usize * dst_w as usize + x as usize) * 4;
    if idx + 4 > dst.len() {
        return;
    }
    let sa = color[3] as u16;
    if sa == 0 {
        return;
    }
    let inv = 255 - sa;
    // Straight alpha `src_over dst`.
    for c in 0..3 {
        let out = (color[c] as u16 * sa + dst[idx + c] as u16 * inv + 127) / 255;
        dst[idx + c] = out as u8;
    }
    let da = dst[idx + 3] as u16;
    let out_a = sa + (da * inv + 127) / 255;
    dst[idx + 3] = out_a.min(255) as u8;
}

// ---------------------------------------------------------------------------
// Glyph data
// ---------------------------------------------------------------------------
//
// Each glyph is 16 lines of 8 pixels, described with `.` for empty and `#`
// for lit. We parse this at const-eval time into a fixed-size byte array so
// the final binary image is the compact packed form — there is no runtime
// overhead. The big advantage is readability: anyone can fix a glyph by
// editing the ASCII art.

/// Glyphs authored as 16-line × 8-column ASCII art blocks. Only the low
/// 8 characters of each row are consulted; anything else is ignored. The
/// array is addressed by codepoint (0..256). Entries we haven't drawn
/// fall back to [`MISSING_GLYPH`].
///
/// Note: the table is huge (around 48 KB of source) but compresses back
/// down to 4 KB at compile time. We define it in chunks per range.
const BLANK: &str = "........\n........\n........\n........\n\
                     ........\n........\n........\n........\n\
                     ........\n........\n........\n........\n\
                     ........\n........\n........\n........\n";

/// The "missing glyph" rectangle used for un-drawn codepoints.
const MISSING_GLYPH: [u8; 16] = parse_glyph(
    "........\n\
     ........\n\
     .######.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .#....#.\n\
     .######.\n\
     ........\n\
     ........\n",
);

/// Convert an ASCII-art glyph spec (16 rows of 8 columns, `.` or `#`,
/// separated by newlines or other whitespace) into the 16-byte row form.
/// Anything outside the first 8 chars of each row is ignored. Rows beyond
/// 16 are ignored. Padding rows fill with zero.
const fn parse_glyph(s: &str) -> [u8; 16] {
    let bytes = s.as_bytes();
    let mut out = [0u8; 16];
    let mut i = 0;
    let mut row = 0;
    let mut col = 0;
    while i < bytes.len() && row < 16 {
        let b = bytes[i];
        i += 1;
        if b == b'\n' {
            if col > 0 {
                row += 1;
                col = 0;
            }
            continue;
        }
        if col >= 8 {
            continue;
        }
        if b == b'#' {
            out[row] |= 1 << (7 - col);
            col += 1;
        } else if b == b'.' {
            col += 1;
        }
        // other chars: skipped
    }
    out
}

/// Build the full 256-entry glyph table. Uses parse_glyph on each hand-
/// authored ASCII-art string; slots we don't cover fall back to
/// MISSING_GLYPH.
const fn build_table() -> [[u8; 16]; 256] {
    let mut t = [MISSING_GLYPH; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = MISSING_GLYPH;
        i += 1;
    }
    // Control characters (0..0x20) and 0x7F: blank, not missing-glyph.
    let blank = parse_glyph(BLANK);
    let mut i = 0;
    while i < 0x20 {
        t[i] = blank;
        i += 1;
    }
    t[0x7F] = blank;
    // 0x80..=0xA0: also blank (C1 controls + NBSP).
    let mut i = 0x80;
    while i <= 0xA0 {
        t[i] = blank;
        i += 1;
    }

    // -------- ASCII printable (0x20..=0x7E) --------
    t[0x20] = blank; // space
    t[0x21] = parse_glyph(GL_EXCLAM);
    t[0x22] = parse_glyph(GL_QUOTE);
    t[0x23] = parse_glyph(GL_HASH);
    t[0x24] = parse_glyph(GL_DOLLAR);
    t[0x25] = parse_glyph(GL_PERCENT);
    t[0x26] = parse_glyph(GL_AMP);
    t[0x27] = parse_glyph(GL_APOS);
    t[0x28] = parse_glyph(GL_LPAREN);
    t[0x29] = parse_glyph(GL_RPAREN);
    t[0x2A] = parse_glyph(GL_STAR);
    t[0x2B] = parse_glyph(GL_PLUS);
    t[0x2C] = parse_glyph(GL_COMMA);
    t[0x2D] = parse_glyph(GL_MINUS);
    t[0x2E] = parse_glyph(GL_PERIOD);
    t[0x2F] = parse_glyph(GL_SLASH);
    t[0x30] = parse_glyph(GL_0);
    t[0x31] = parse_glyph(GL_1);
    t[0x32] = parse_glyph(GL_2);
    t[0x33] = parse_glyph(GL_3);
    t[0x34] = parse_glyph(GL_4);
    t[0x35] = parse_glyph(GL_5);
    t[0x36] = parse_glyph(GL_6);
    t[0x37] = parse_glyph(GL_7);
    t[0x38] = parse_glyph(GL_8);
    t[0x39] = parse_glyph(GL_9);
    t[0x3A] = parse_glyph(GL_COLON);
    t[0x3B] = parse_glyph(GL_SEMI);
    t[0x3C] = parse_glyph(GL_LT);
    t[0x3D] = parse_glyph(GL_EQ);
    t[0x3E] = parse_glyph(GL_GT);
    t[0x3F] = parse_glyph(GL_QUESTION);
    t[0x40] = parse_glyph(GL_AT);
    t[0x41] = parse_glyph(GL_A);
    t[0x42] = parse_glyph(GL_B);
    t[0x43] = parse_glyph(GL_C);
    t[0x44] = parse_glyph(GL_D);
    t[0x45] = parse_glyph(GL_E);
    t[0x46] = parse_glyph(GL_F);
    t[0x47] = parse_glyph(GL_G);
    t[0x48] = parse_glyph(GL_H);
    t[0x49] = parse_glyph(GL_I);
    t[0x4A] = parse_glyph(GL_J);
    t[0x4B] = parse_glyph(GL_K);
    t[0x4C] = parse_glyph(GL_L);
    t[0x4D] = parse_glyph(GL_M);
    t[0x4E] = parse_glyph(GL_N);
    t[0x4F] = parse_glyph(GL_O);
    t[0x50] = parse_glyph(GL_P);
    t[0x51] = parse_glyph(GL_Q);
    t[0x52] = parse_glyph(GL_R);
    t[0x53] = parse_glyph(GL_S);
    t[0x54] = parse_glyph(GL_T);
    t[0x55] = parse_glyph(GL_U);
    t[0x56] = parse_glyph(GL_V);
    t[0x57] = parse_glyph(GL_W);
    t[0x58] = parse_glyph(GL_X);
    t[0x59] = parse_glyph(GL_Y);
    t[0x5A] = parse_glyph(GL_Z);
    t[0x5B] = parse_glyph(GL_LBRACKET);
    t[0x5C] = parse_glyph(GL_BACKSLASH);
    t[0x5D] = parse_glyph(GL_RBRACKET);
    t[0x5E] = parse_glyph(GL_CARET);
    t[0x5F] = parse_glyph(GL_UNDERSCORE);
    t[0x60] = parse_glyph(GL_BACKTICK);
    t[0x61] = parse_glyph(GL_AL);
    t[0x62] = parse_glyph(GL_BL);
    t[0x63] = parse_glyph(GL_CL);
    t[0x64] = parse_glyph(GL_DL);
    t[0x65] = parse_glyph(GL_EL);
    t[0x66] = parse_glyph(GL_FL);
    t[0x67] = parse_glyph(GL_GL);
    t[0x68] = parse_glyph(GL_HL);
    t[0x69] = parse_glyph(GL_IL);
    t[0x6A] = parse_glyph(GL_JL);
    t[0x6B] = parse_glyph(GL_KL);
    t[0x6C] = parse_glyph(GL_LL);
    t[0x6D] = parse_glyph(GL_ML);
    t[0x6E] = parse_glyph(GL_NL);
    t[0x6F] = parse_glyph(GL_OL);
    t[0x70] = parse_glyph(GL_PL);
    t[0x71] = parse_glyph(GL_QL);
    t[0x72] = parse_glyph(GL_RL);
    t[0x73] = parse_glyph(GL_SL);
    t[0x74] = parse_glyph(GL_TL);
    t[0x75] = parse_glyph(GL_UL);
    t[0x76] = parse_glyph(GL_VL);
    t[0x77] = parse_glyph(GL_WL);
    t[0x78] = parse_glyph(GL_XL);
    t[0x79] = parse_glyph(GL_YL);
    t[0x7A] = parse_glyph(GL_ZL);
    t[0x7B] = parse_glyph(GL_LBRACE);
    t[0x7C] = parse_glyph(GL_PIPE);
    t[0x7D] = parse_glyph(GL_RBRACE);
    t[0x7E] = parse_glyph(GL_TILDE);

    // -------- Latin-1 supplement (0xA1..=0xFF) --------
    t[0xA1] = parse_glyph(GL_IEXCL);
    t[0xA2] = parse_glyph(GL_CENT);
    t[0xA3] = parse_glyph(GL_POUND);
    t[0xA4] = parse_glyph(GL_CURRENCY);
    t[0xA5] = parse_glyph(GL_YEN);
    t[0xA6] = parse_glyph(GL_BROKEN);
    t[0xA7] = parse_glyph(GL_SECTION);
    t[0xA8] = parse_glyph(GL_DIAERESIS);
    t[0xA9] = parse_glyph(GL_COPY);
    t[0xAA] = parse_glyph(GL_FEMORD);
    t[0xAB] = parse_glyph(GL_LAQUO);
    t[0xAC] = parse_glyph(GL_NOT);
    t[0xAD] = blank; // soft hyphen
    t[0xAE] = parse_glyph(GL_REG);
    t[0xAF] = parse_glyph(GL_MACRON);
    t[0xB0] = parse_glyph(GL_DEG);
    t[0xB1] = parse_glyph(GL_PLUSMINUS);
    t[0xB2] = parse_glyph(GL_SUP2);
    t[0xB3] = parse_glyph(GL_SUP3);
    t[0xB4] = parse_glyph(GL_ACUTE);
    t[0xB5] = parse_glyph(GL_MICRO);
    t[0xB6] = parse_glyph(GL_PILCROW);
    t[0xB7] = parse_glyph(GL_MIDDOT);
    t[0xB8] = parse_glyph(GL_CEDILLA);
    t[0xB9] = parse_glyph(GL_SUP1);
    t[0xBA] = parse_glyph(GL_MASCORD);
    t[0xBB] = parse_glyph(GL_RAQUO);
    t[0xBC] = parse_glyph(GL_FRAC14);
    t[0xBD] = parse_glyph(GL_FRAC12);
    t[0xBE] = parse_glyph(GL_FRAC34);
    t[0xBF] = parse_glyph(GL_IQUEST);

    // Accented uppercase (À..Ö, Ø..Ÿ). We build the base letter + a
    // diacritic hat painted in the top two rows.
    t[0xC0] = add_grave(parse_glyph(GL_A));
    t[0xC1] = add_acute(parse_glyph(GL_A));
    t[0xC2] = add_circ(parse_glyph(GL_A));
    t[0xC3] = add_tilde(parse_glyph(GL_A));
    t[0xC4] = add_uml(parse_glyph(GL_A));
    t[0xC5] = add_ring(parse_glyph(GL_A));
    t[0xC6] = parse_glyph(GL_AE);
    t[0xC7] = parse_glyph(GL_CCEDIL);
    t[0xC8] = add_grave(parse_glyph(GL_E));
    t[0xC9] = add_acute(parse_glyph(GL_E));
    t[0xCA] = add_circ(parse_glyph(GL_E));
    t[0xCB] = add_uml(parse_glyph(GL_E));
    t[0xCC] = add_grave(parse_glyph(GL_I));
    t[0xCD] = add_acute(parse_glyph(GL_I));
    t[0xCE] = add_circ(parse_glyph(GL_I));
    t[0xCF] = add_uml(parse_glyph(GL_I));
    t[0xD0] = parse_glyph(GL_ETH);
    t[0xD1] = add_tilde(parse_glyph(GL_N));
    t[0xD2] = add_grave(parse_glyph(GL_O));
    t[0xD3] = add_acute(parse_glyph(GL_O));
    t[0xD4] = add_circ(parse_glyph(GL_O));
    t[0xD5] = add_tilde(parse_glyph(GL_O));
    t[0xD6] = add_uml(parse_glyph(GL_O));
    t[0xD7] = parse_glyph(GL_TIMES);
    t[0xD8] = parse_glyph(GL_OSTROKE);
    t[0xD9] = add_grave(parse_glyph(GL_U));
    t[0xDA] = add_acute(parse_glyph(GL_U));
    t[0xDB] = add_circ(parse_glyph(GL_U));
    t[0xDC] = add_uml(parse_glyph(GL_U));
    t[0xDD] = add_acute(parse_glyph(GL_Y));
    t[0xDE] = parse_glyph(GL_THORN);
    t[0xDF] = parse_glyph(GL_SZLIG);
    t[0xE0] = add_grave(parse_glyph(GL_AL));
    t[0xE1] = add_acute(parse_glyph(GL_AL));
    t[0xE2] = add_circ(parse_glyph(GL_AL));
    t[0xE3] = add_tilde(parse_glyph(GL_AL));
    t[0xE4] = add_uml(parse_glyph(GL_AL));
    t[0xE5] = add_ring(parse_glyph(GL_AL));
    t[0xE6] = parse_glyph(GL_AEL);
    t[0xE7] = parse_glyph(GL_CCEDILL);
    t[0xE8] = add_grave(parse_glyph(GL_EL));
    t[0xE9] = add_acute(parse_glyph(GL_EL));
    t[0xEA] = add_circ(parse_glyph(GL_EL));
    t[0xEB] = add_uml(parse_glyph(GL_EL));
    t[0xEC] = add_grave_narrow(parse_glyph(GL_IL));
    t[0xED] = add_acute_narrow(parse_glyph(GL_IL));
    t[0xEE] = add_circ_narrow(parse_glyph(GL_IL));
    t[0xEF] = add_uml_narrow(parse_glyph(GL_IL));
    t[0xF0] = parse_glyph(GL_ETHL);
    t[0xF1] = add_tilde(parse_glyph(GL_NL));
    t[0xF2] = add_grave(parse_glyph(GL_OL));
    t[0xF3] = add_acute(parse_glyph(GL_OL));
    t[0xF4] = add_circ(parse_glyph(GL_OL));
    t[0xF5] = add_tilde(parse_glyph(GL_OL));
    t[0xF6] = add_uml(parse_glyph(GL_OL));
    t[0xF7] = parse_glyph(GL_DIV);
    t[0xF8] = parse_glyph(GL_OSTROKEL);
    t[0xF9] = add_grave(parse_glyph(GL_UL));
    t[0xFA] = add_acute(parse_glyph(GL_UL));
    t[0xFB] = add_circ(parse_glyph(GL_UL));
    t[0xFC] = add_uml(parse_glyph(GL_UL));
    t[0xFD] = add_acute(parse_glyph(GL_YL));
    t[0xFE] = parse_glyph(GL_THORNL);
    t[0xFF] = add_uml(parse_glyph(GL_YL));

    t
}

/// Bold variant: each row of the regular glyph ORed with itself shifted
/// right by one pixel. Drops the rightmost bit (no overflow into next
/// cell). This matches the classic VGA "bold" rendering.
const fn build_bold_table(regular: &[[u8; 16]; 256]) -> [[u8; 16]; 256] {
    let mut t = [[0u8; 16]; 256];
    let mut i = 0;
    while i < 256 {
        let mut j = 0;
        while j < 16 {
            let r = regular[i][j];
            t[i][j] = r | (r >> 1);
            j += 1;
        }
        i += 1;
    }
    t
}

// --- Diacritic composition helpers (const fn) -----------------------------

// Replace the first two rows with the given hat, then OR in the letter's
// original rows 2..16. The hat is 16 rows but only rows 0..=1 are used.

const fn overlay_hat(letter: [u8; 16], hat: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0] = hat[0];
    out[1] = hat[1];
    let mut i = 2;
    while i < 16 {
        out[i] = letter[i];
        i += 1;
    }
    out
}

const fn add_grave(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "..##....\n....##..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_acute(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "....##..\n..##....\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_circ(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "...##...\n..#..#..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_tilde(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "..#...#.\n.#.#.#..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_uml(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "..#..#..\n..#..#..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_ring(l: [u8; 16]) -> [u8; 16] {
    overlay_hat(
        l,
        parse_glyph(
            "...##...\n..#..#..\n...##...\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}

// For narrow lowercase i: push the dot back to column 3 and drop the
// existing dot row from the letter.
const fn strip_i_dot(l: [u8; 16]) -> [u8; 16] {
    let mut out = l;
    out[3] = 0; // the dot row of the lowercase i glyph
    out
}
const fn add_grave_narrow(l: [u8; 16]) -> [u8; 16] {
    let l = strip_i_dot(l);
    overlay_hat(
        l,
        parse_glyph(
            "..##....\n....##..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_acute_narrow(l: [u8; 16]) -> [u8; 16] {
    let l = strip_i_dot(l);
    overlay_hat(
        l,
        parse_glyph(
            "....##..\n..##....\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_circ_narrow(l: [u8; 16]) -> [u8; 16] {
    let l = strip_i_dot(l);
    overlay_hat(
        l,
        parse_glyph(
            "...##...\n..#..#..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}
const fn add_uml_narrow(l: [u8; 16]) -> [u8; 16] {
    let l = strip_i_dot(l);
    overlay_hat(
        l,
        parse_glyph(
            "..#..#..\n..#..#..\n........\n........\n........\n........\n........\n........\n\
             ........\n........\n........\n........\n........\n........\n........\n........\n",
        ),
    )
}

// ---------------------------------------------------------------------------
// Glyph strings — ASCII printables
// ---------------------------------------------------------------------------

// NOTE: Baseline sits at row 12 (bearing_y = 12). Typical uppercase letter
// occupies rows 2..=11 — 10 rows. Descenders occupy rows 12..=14.

const GL_EXCLAM: &str =
    "........\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
...##...\n........\n...##...\n...##...\n........\n........\n........\n........\n";
const GL_QUOTE: &str =
    "........\n..#..#..\n..#..#..\n..#..#..\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_HASH: &str =
    "........\n..#..#..\n..#..#..\n.######.\n..#..#..\n..#..#..\n.######.\n..#..#..\n\
..#..#..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_DOLLAR: &str =
    "...##...\n..####..\n.##..##.\n.##.....\n..####..\n.....##.\n.##..##.\n..####..\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_PERCENT: &str =
    "........\n.##...#.\n.##..#..\n....#...\n...#....\n..#..##.\n.#...##.\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_AMP: &str =
    "........\n..###...\n.#...#..\n.#...#..\n..###...\n..###...\n.#..#.#.\n.#...#..\n\
..###.#.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_APOS: &str =
    "........\n...##...\n...##...\n...#....\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_LPAREN: &str =
    "........\n....##..\n...#....\n..#.....\n..#.....\n..#.....\n..#.....\n..#.....\n\
...#....\n....##..\n........\n........\n........\n........\n........\n........\n";
const GL_RPAREN: &str =
    "........\n..##....\n....#...\n.....#..\n.....#..\n.....#..\n.....#..\n.....#..\n\
....#...\n..##....\n........\n........\n........\n........\n........\n........\n";
const GL_STAR: &str =
    "........\n........\n...##...\n.#.##.#.\n..####..\n...##...\n..####..\n.#.##.#.\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_PLUS: &str =
    "........\n........\n........\n...##...\n...##...\n.######.\n.######.\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_COMMA: &str =
    "........\n........\n........\n........\n........\n........\n........\n........\n\
........\n........\n..###...\n..###...\n...##...\n..##....\n........\n........\n";
const GL_MINUS: &str =
    "........\n........\n........\n........\n........\n........\n.######.\n.######.\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_PERIOD: &str =
    "........\n........\n........\n........\n........\n........\n........\n........\n\
........\n........\n..###...\n..###...\n........\n........\n........\n........\n";
const GL_SLASH: &str =
    "........\n......#.\n......#.\n.....#..\n....#...\n...#....\n..#.....\n.#......\n\
.#......\n........\n........\n........\n........\n........\n........\n........\n";

const GL_0: &str =
    "........\n..####..\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_1: &str =
    "........\n...##...\n..###...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_2: &str =
    "........\n..####..\n.##..##.\n.....##.\n....##..\n...##...\n..##....\n.##.....\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_3: &str =
    "........\n..####..\n.##..##.\n.....##.\n...###..\n.....##.\n.....##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_4: &str =
    "........\n....##..\n...###..\n..####..\n.##.##..\n.######.\n....##..\n....##..\n\
....##..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_5: &str =
    "........\n.######.\n.##.....\n.##.....\n.#####..\n.....##.\n.....##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_6: &str =
    "........\n..####..\n.##..##.\n.##.....\n.#####..\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_7: &str =
    "........\n.######.\n.##..##.\n.....##.\n....##..\n...##...\n...##...\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_8: &str =
    "........\n..####..\n.##..##.\n.##..##.\n..####..\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_9: &str =
    "........\n..####..\n.##..##.\n.##..##.\n.##..##.\n..#####.\n.....##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";

const GL_COLON: &str =
    "........\n........\n........\n...##...\n...##...\n........\n........\n........\n\
...##...\n...##...\n........\n........\n........\n........\n........\n........\n";
const GL_SEMI: &str =
    "........\n........\n........\n...##...\n...##...\n........\n........\n........\n\
...##...\n...##...\n...#....\n..#.....\n........\n........\n........\n........\n";
const GL_LT: &str =
    "........\n......#.\n.....#..\n....#...\n...#....\n..#.....\n...#....\n....#...\n\
.....#..\n......#.\n........\n........\n........\n........\n........\n........\n";
const GL_EQ: &str =
    "........\n........\n........\n........\n.######.\n........\n........\n.######.\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_GT: &str =
    "........\n.#......\n..#.....\n...#....\n....#...\n.....#..\n....#...\n...#....\n\
..#.....\n.#......\n........\n........\n........\n........\n........\n........\n";
const GL_QUESTION: &str =
    "........\n..####..\n.##..##.\n.....##.\n....##..\n...##...\n...##...\n........\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_AT: &str =
    "........\n..####..\n.##..##.\n.##.###.\n.##.###.\n.##.###.\n.##.....\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";

// Uppercase letters
const GL_A: &str =
    "........\n...##...\n..####..\n.##..##.\n.##..##.\n.######.\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_B: &str =
    "........\n.#####..\n.##..##.\n.##..##.\n.#####..\n.##..##.\n.##..##.\n.##..##.\n\
.#####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_C: &str =
    "........\n..####..\n.##..##.\n.##.....\n.##.....\n.##.....\n.##.....\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_D: &str =
    "........\n.####...\n.##.##..\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##.##..\n\
.####...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_E: &str =
    "........\n.######.\n.##.....\n.##.....\n.#####..\n.##.....\n.##.....\n.##.....\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_F: &str =
    "........\n.######.\n.##.....\n.##.....\n.#####..\n.##.....\n.##.....\n.##.....\n\
.##.....\n........\n........\n........\n........\n........\n........\n........\n";
const GL_G: &str =
    "........\n..####..\n.##..##.\n.##.....\n.##.###.\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_H: &str =
    "........\n.##..##.\n.##..##.\n.##..##.\n.######.\n.##..##.\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_I: &str =
    "........\n..####..\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_J: &str =
    "........\n....####\n.....##.\n.....##.\n.....##.\n.....##.\n.....##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_K: &str =
    "........\n.##..##.\n.##.##..\n.####...\n.###....\n.####...\n.##.##..\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_L: &str =
    "........\n.##.....\n.##.....\n.##.....\n.##.....\n.##.....\n.##.....\n.##.....\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_M: &str =
    "........\n.##..##.\n.######.\n.######.\n.######.\n.##..##.\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_N: &str =
    "........\n.##..##.\n.###.##.\n.###.##.\n.######.\n.##.###.\n.##.###.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_O: &str =
    "........\n..####..\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_P: &str =
    "........\n.#####..\n.##..##.\n.##..##.\n.#####..\n.##.....\n.##.....\n.##.....\n\
.##.....\n........\n........\n........\n........\n........\n........\n........\n";
const GL_Q: &str =
    "........\n..####..\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##.###.\n\
..#####.\n......#.\n........\n........\n........\n........\n........\n........\n";
const GL_R: &str =
    "........\n.#####..\n.##..##.\n.##..##.\n.#####..\n.##.##..\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_S: &str =
    "........\n..####..\n.##..##.\n.##.....\n..####..\n.....##.\n.....##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_T: &str =
    "........\n.######.\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_U: &str =
    "........\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_V: &str =
    "........\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n..####..\n..####..\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_W: &str =
    "........\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n.######.\n.######.\n.######.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_X: &str =
    "........\n.##..##.\n.##..##.\n..####..\n...##...\n..####..\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_Y: &str =
    "........\n.##..##.\n.##..##.\n..####..\n...##...\n...##...\n...##...\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_Z: &str =
    "........\n.######.\n.....##.\n....##..\n...##...\n..##....\n.##.....\n.##.....\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";

const GL_LBRACKET: &str =
    "........\n..####..\n..##....\n..##....\n..##....\n..##....\n..##....\n..##....\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_BACKSLASH: &str =
    "........\n.#......\n.#......\n..#.....\n...#....\n....#...\n.....#..\n......#.\n\
......#.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_RBRACKET: &str =
    "........\n..####..\n....##..\n....##..\n....##..\n....##..\n....##..\n....##..\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_CARET: &str =
    "........\n...##...\n..####..\n.##..##.\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_UNDERSCORE: &str =
    "........\n........\n........\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n.######.\n........\n........\n........\n";
const GL_BACKTICK: &str =
    "........\n..##....\n....##..\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";

// Lowercase letters (baseline at row 12, x-height rows 5..11)
const GL_AL: &str =
    "........\n........\n........\n........\n..####..\n.....##.\n..#####.\n.##..##.\n\
..#####.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_BL: &str =
    "........\n.##.....\n.##.....\n.##.....\n.#####..\n.##..##.\n.##..##.\n.##..##.\n\
.#####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_CL: &str =
    "........\n........\n........\n........\n..####..\n.##..##.\n.##.....\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_DL: &str =
    "........\n.....##.\n.....##.\n.....##.\n..#####.\n.##..##.\n.##..##.\n.##..##.\n\
..#####.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_EL: &str =
    "........\n........\n........\n........\n..####..\n.##..##.\n.######.\n.##.....\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_FL: &str =
    "........\n...###..\n..##.##.\n..##....\n.#####..\n..##....\n..##....\n..##....\n\
..##....\n........\n........\n........\n........\n........\n........\n........\n";
const GL_GL: &str =
    "........\n........\n........\n........\n..#####.\n.##..##.\n.##..##.\n..#####.\n\
.....##.\n.##..##.\n..####..\n........\n........\n........\n........\n........\n";
const GL_HL: &str =
    "........\n.##.....\n.##.....\n.##.....\n.#####..\n.##..##.\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_IL: &str =
    "........\n...##...\n...##...\n........\n..###...\n...##...\n...##...\n...##...\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_JL: &str =
    "........\n.....##.\n.....##.\n........\n....###.\n.....##.\n.....##.\n.....##.\n\
.##..##.\n..####..\n........\n........\n........\n........\n........\n........\n";
const GL_KL: &str =
    "........\n.##.....\n.##.....\n.##.....\n.##..##.\n.##.##..\n.####...\n.##.##..\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_LL: &str =
    "........\n..###...\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_ML: &str =
    "........\n........\n........\n........\n.######.\n.######.\n.######.\n.##.###.\n\
.##.###.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_NL: &str =
    "........\n........\n........\n........\n.#####..\n.##..##.\n.##..##.\n.##..##.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_OL: &str =
    "........\n........\n........\n........\n..####..\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_PL: &str =
    "........\n........\n........\n........\n.#####..\n.##..##.\n.##..##.\n.#####..\n\
.##.....\n.##.....\n........\n........\n........\n........\n........\n........\n";
const GL_QL: &str =
    "........\n........\n........\n........\n..#####.\n.##..##.\n.##..##.\n..#####.\n\
.....##.\n.....##.\n........\n........\n........\n........\n........\n........\n";
const GL_RL: &str =
    "........\n........\n........\n........\n.##.###.\n.###.##.\n.##.....\n.##.....\n\
.##.....\n........\n........\n........\n........\n........\n........\n........\n";
const GL_SL: &str =
    "........\n........\n........\n........\n..####..\n.##.....\n..####..\n.....##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_TL: &str =
    "........\n..##....\n..##....\n..##....\n.#####..\n..##....\n..##....\n..##.##.\n\
...###..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_UL: &str =
    "........\n........\n........\n........\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n\
..#####.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_VL: &str =
    "........\n........\n........\n........\n.##..##.\n.##..##.\n.##..##.\n..####..\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_WL: &str =
    "........\n........\n........\n........\n.##..##.\n.##..##.\n.######.\n.######.\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_XL: &str =
    "........\n........\n........\n........\n.##..##.\n..####..\n...##...\n..####..\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_YL: &str =
    "........\n........\n........\n........\n.##..##.\n.##..##.\n.##..##.\n..#####.\n\
.....##.\n.##..##.\n..####..\n........\n........\n........\n........\n........\n";
const GL_ZL: &str =
    "........\n........\n........\n........\n.######.\n....##..\n...##...\n..##....\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";

const GL_LBRACE: &str =
    "........\n...###..\n..##....\n..##....\n..##....\n.##.....\n..##....\n..##....\n\
..##....\n...###..\n........\n........\n........\n........\n........\n........\n";
const GL_PIPE: &str =
    "........\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n...##...\n\
...##...\n...##...\n........\n........\n........\n........\n........\n........\n";
const GL_RBRACE: &str =
    "........\n..###...\n....##..\n....##..\n....##..\n.....##.\n....##..\n....##..\n\
....##..\n..###...\n........\n........\n........\n........\n........\n........\n";
const GL_TILDE: &str =
    "........\n..##.##.\n.##.##..\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";

// Latin-1 supplement (Section only minimally shaped; all reuse the
// same 8x16 cell). Where we don't have a dedicated design we substitute
// a best-effort stand-in.
const GL_IEXCL: &str =
    "........\n........\n...##...\n........\n...##...\n...##...\n...##...\n...##...\n\
...##...\n...##...\n........\n........\n........\n........\n........\n........\n";
const GL_CENT: &str =
    "........\n...##...\n..####..\n.##.##..\n.##.....\n.##.....\n.##.##..\n..####..\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_POUND: &str =
    "........\n...####.\n..##..#.\n..##....\n.#####..\n..##....\n..##....\n.##..##.\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_CURRENCY: &str =
    "........\n.##..##.\n..####..\n.##..##.\n.##..##.\n..####..\n.##..##.\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_YEN: &str =
    "........\n.##..##.\n.##..##.\n..####..\n.######.\n...##...\n.######.\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_BROKEN: &str =
    "........\n...##...\n...##...\n...##...\n........\n........\n...##...\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_SECTION: &str =
    "........\n..####..\n.##..##.\n..##....\n..####..\n.##..##.\n....##..\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_DIAERESIS: &str =
    "........\n..#..#..\n..#..#..\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_COPY: &str =
    "........\n..####..\n.#....#.\n.#.##.#.\n.#.#..#.\n.#.##.#.\n.#....#.\n..####..\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_FEMORD: &str =
    "........\n..####..\n.....##.\n..#####.\n.##..##.\n..#####.\n........\n.######.\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_LAQUO: &str =
    "........\n........\n..#..#..\n.#..#...\n#..#....\n.#..#...\n..#..#..\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_NOT: &str =
    "........\n........\n........\n........\n.######.\n.....##.\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_REG: &str =
    "........\n..####..\n.#....#.\n.#.##.#.\n.#.##.#.\n.#.#..#.\n.#....#.\n..####..\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_MACRON: &str =
    "........\n.######.\n........\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_DEG: &str =
    "........\n..####..\n.##..##.\n..####..\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_PLUSMINUS: &str =
    "........\n........\n...##...\n...##...\n.######.\n...##...\n...##...\n........\n\
.######.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_SUP2: &str =
    "........\n..###...\n.#..##..\n....#...\n...#....\n..#.....\n.#####..\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_SUP3: &str =
    "........\n..###...\n.#..##..\n....#...\n.....#..\n.#..##..\n..###...\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_ACUTE: &str =
    "........\n....##..\n..##....\n........\n........\n........\n........\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_MICRO: &str =
    "........\n........\n........\n........\n.##..##.\n.##..##.\n.##..##.\n.##..##.\n\
.######.\n.##.....\n.##.....\n........\n........\n........\n........\n........\n";
const GL_PILCROW: &str =
    "........\n..######\n.####.##\n.####.##\n..###.##\n.....##.\n.....##.\n.....##.\n\
.....##.\n........\n........\n........\n........\n........\n........\n........\n";
const GL_MIDDOT: &str =
    "........\n........\n........\n........\n........\n...##...\n...##...\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_CEDILLA: &str =
    "........\n........\n........\n........\n........\n........\n........\n........\n\
........\n........\n........\n....##..\n...##...\n..###...\n........\n........\n";
const GL_SUP1: &str =
    "........\n...##...\n..###...\n...##...\n...##...\n...##...\n..####..\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_MASCORD: &str =
    "........\n..####..\n.....##.\n..#####.\n.##..##.\n..#####.\n........\n.######.\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_RAQUO: &str =
    "........\n........\n.#..#...\n..#..#..\n...#..#.\n..#..#..\n.#..#...\n........\n\
........\n........\n........\n........\n........\n........\n........\n........\n";
const GL_FRAC14: &str =
    "........\n.##..#..\n.##.#...\n....#...\n...#....\n...#.##.\n..#.####\n..#.#..#\n\
.#..#..#\n....####\n.......#\n.......#\n........\n........\n........\n........\n";
const GL_FRAC12: &str =
    "........\n.##..#..\n.##.#...\n....#...\n...#....\n...#.##.\n..#.#..#\n..#...#.\n\
.#...#..\n....####\n........\n........\n........\n........\n........\n........\n";
const GL_FRAC34: &str =
    "........\n.###.#..\n.#..#...\n..#.#...\n...#....\n..#.##..\n.#.####.\n..#.#..#\n\
....#..#\n....####\n.......#\n.......#\n........\n........\n........\n........\n";
const GL_IQUEST: &str =
    "........\n...##...\n........\n...##...\n...##...\n..##....\n.##.....\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";

// AE ligature (uppercase)
const GL_AE: &str =
    "........\n..#####.\n.##.##..\n.##.##..\n.##.####\n.######.\n.##.##..\n.##.##..\n\
.##.####\n........\n........\n........\n........\n........\n........\n........\n";
const GL_AEL: &str =
    "........\n........\n........\n........\n..#####.\n....####\n..######\n.######.\n\
.##.####\n........\n........\n........\n........\n........\n........\n........\n";

// C with cedilla
const GL_CCEDIL: &str =
    "........\n..####..\n.##..##.\n.##.....\n.##.....\n.##.....\n.##.....\n.##..##.\n\
..####..\n....#...\n...##...\n..##....\n........\n........\n........\n........\n";
const GL_CCEDILL: &str =
    "........\n........\n........\n........\n..####..\n.##..##.\n.##.....\n.##..##.\n\
..####..\n....#...\n...##...\n..##....\n........\n........\n........\n........\n";

// Eth
const GL_ETH: &str =
    "........\n.####...\n.##.##..\n.##..##.\n.####.##\n.##..##.\n.##..##.\n.##.##..\n\
.####...\n........\n........\n........\n........\n........\n........\n........\n";
const GL_ETHL: &str =
    "........\n...##.##\n....####\n...####.\n..##.##.\n.##..##.\n.##..##.\n.##..##.\n\
..####..\n........\n........\n........\n........\n........\n........\n........\n";

// Multiplication sign
const GL_TIMES: &str =
    "........\n........\n........\n........\n.##..##.\n..####..\n...##...\n..####..\n\
.##..##.\n........\n........\n........\n........\n........\n........\n........\n";
// Division sign
const GL_DIV: &str =
    "........\n........\n...##...\n...##...\n........\n.######.\n........\n...##...\n\
...##...\n........\n........\n........\n........\n........\n........\n........\n";

// O stroke
const GL_OSTROKE: &str =
    "........\n..#####.\n.##..##.\n.##.###.\n.##.###.\n.######.\n.###.##.\n.###.##.\n\
.#####..\n........\n........\n........\n........\n........\n........\n........\n";
const GL_OSTROKEL: &str =
    "........\n........\n........\n........\n..#####.\n.##.###.\n.######.\n.###.##.\n\
.#####..\n........\n........\n........\n........\n........\n........\n........\n";

// Thorn (uppercase / lowercase)
const GL_THORN: &str =
    "........\n.##.....\n.#####..\n.##..##.\n.##..##.\n.#####..\n.##.....\n.##.....\n\
.##.....\n........\n........\n........\n........\n........\n........\n........\n";
const GL_THORNL: &str =
    "........\n........\n.##.....\n.##.....\n.#####..\n.##..##.\n.##..##.\n.#####..\n\
.##.....\n.##.....\n........\n........\n........\n........\n........\n........\n";

// ß (sharp s)
const GL_SZLIG: &str =
    "........\n..####..\n.##..##.\n.##..##.\n.####...\n.##..##.\n.##..##.\n.##..##.\n\
.####.#.\n........\n........\n........\n........\n........\n........\n........\n";

// ---------------------------------------------------------------------------
// Final tables
// ---------------------------------------------------------------------------

const REGULAR_TABLE: [[u8; 16]; 256] = build_table();
const BOLD_TABLE: [[u8; 16]; 256] = build_bold_table(&REGULAR_TABLE);

static REGULAR_FONT: BitmapFont = BitmapFont {
    cell_w: 8,
    cell_h: 16,
    bearing_y: 12,
    table: &REGULAR_TABLE,
};

static BOLD_FONT: BitmapFont = BitmapFont {
    cell_w: 8,
    cell_h: 16,
    bearing_y: 12,
    table: &BOLD_TABLE,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_has_ascii_shapes() {
        let f = BitmapFont::default_regular();
        // 'A' should have at least a dozen lit pixels.
        let rows = f.glyph_rows('A');
        let lit: u32 = rows.iter().map(|b| b.count_ones()).sum();
        assert!(lit > 8, "'A' glyph too sparse: {lit}");
    }

    #[test]
    fn bold_is_denser_than_regular() {
        let r = BitmapFont::default_regular();
        let b = BitmapFont::default_bold();
        let lit_r: u32 = r.glyph_rows('E').iter().map(|b| b.count_ones()).sum();
        let lit_b: u32 = b.glyph_rows('E').iter().map(|b| b.count_ones()).sum();
        assert!(lit_b >= lit_r, "bold should smear wider");
    }

    #[test]
    fn draw_glyph_paints_pixels() {
        let f = BitmapFont::default_regular();
        let w = 32u32;
        let h = 32u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let advance = f.draw_glyph('A', &mut buf, w, h, 4, 20, [255, 255, 255, 255]);
        assert_eq!(advance, 8);
        // Some pixel in the upper region must now be lit.
        let lit_pixels = buf.chunks(4).filter(|p| p[3] > 0).count();
        assert!(lit_pixels > 0, "draw_glyph painted no pixels");
    }

    #[test]
    fn latin1_has_some_coverage() {
        let f = BitmapFont::default_regular();
        for ch in ['\u{00E9}', '\u{00FF}', '\u{00C0}', '\u{00D1}'] {
            let rows = f.glyph_rows(ch);
            let lit: u32 = rows.iter().map(|b| b.count_ones()).sum();
            assert!(lit > 4, "{ch:?} has no shape");
        }
    }
}
