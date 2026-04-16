//! Slice-level encode/decode for FFV1 version 3.
//!
//! A slice is the independently-decodable unit. Each slice is laid out per
//! RFC 9043 §4.5:
//!
//! ```text
//! slice_data      (range-coded: slice_header + plane samples)
//! slice_size      (3 bytes, big-endian; size of slice_data)
//! error_status    (1 byte, only present when ec != 0 in the config record)
//! slice_crc_parity (4 bytes, big-endian; only when ec != 0)
//! ```
//!
//! The CRC-32 (polynomial 0x04C11DB7, MSB-first, init=0) is computed so that
//! a CRC over the entire slice (including the 4-byte CRC field itself) is
//! zero. Our encoder always emits a single slice covering the whole frame;
//! the decoder accepts any number of slices arranged on a regular grid per
//! the `num_h_slices × num_v_slices` in the config record.
//!
//! For self-roundtrip robustness we default to `ec = 0` (no CRC in slice
//! footers); a conforming decoder — including FFmpeg — must handle both
//! forms because the flag lives in the config record.

use oxideav_core::{Error, Result};

use crate::crc::crc32_ieee;
use crate::predictor::predict;
use crate::range_coder::{RangeDecoder, RangeEncoder};
use crate::state::{compute_context, context_count, default_quant_tables, PlaneState, QuantTables};

/// Geometry for a single plane within a slice.
#[derive(Clone, Copy, Debug)]
pub struct PlaneGeom {
    pub width: u32,
    pub height: u32,
}

/// Encode a single plane's worth of 8-bit samples to range-coded residuals.
/// `samples` is a row-major buffer of exactly `width * height` bytes (no
/// stride); the caller should arrange that. `state` holds the per-context
/// range-coder states, one entry per context index.
pub fn encode_plane(
    enc: &mut RangeEncoder,
    samples: &[u8],
    width: u32,
    height: u32,
    tables: &QuantTables,
    state: &mut PlaneState,
) {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(samples.len(), w * h, "encode_plane: dimensions mismatch");

    for y in 0..h {
        for x in 0..w {
            let s = samples[y * w + x] as i32;
            let (big_l, l, t, tl, big_t, tr) = neighbours(samples, w, h, x, y);
            let mut ctx = compute_context(tables, big_l, l, t, tl, big_t, tr);
            let sign_flip = ctx < 0;
            if sign_flip {
                ctx = -ctx;
            }
            let pred = predict(l, t, tl);
            // Wrap residual into signed 8-bit domain (n LSBs of difference,
            // interpreted as signed).
            let diff = s - pred;
            let wrapped: i32 = (diff << 24) >> 24; // sign-extend low 8 bits
            let residual = if sign_flip { -wrapped } else { wrapped };
            let state_row = &mut state.states[ctx as usize];
            enc.put_symbol(state_row, residual, true);
        }
    }
}

/// Decode a single plane's worth of 8-bit samples. Mirrors `encode_plane`.
pub fn decode_plane(
    dec: &mut RangeDecoder<'_>,
    samples: &mut [u8],
    width: u32,
    height: u32,
    tables: &QuantTables,
    state: &mut PlaneState,
) -> Result<()> {
    let w = width as usize;
    let h = height as usize;
    if samples.len() != w * h {
        return Err(Error::invalid("decode_plane: buffer length mismatch"));
    }
    for y in 0..h {
        for x in 0..w {
            let (big_l, l, t, tl, big_t, tr) = neighbours(samples, w, h, x, y);
            let mut ctx = compute_context(tables, big_l, l, t, tl, big_t, tr);
            let sign_flip = ctx < 0;
            if sign_flip {
                ctx = -ctx;
            }
            let pred = predict(l, t, tl);
            let state_row = &mut state.states[ctx as usize];
            let mut residual = dec.get_symbol(state_row, true);
            if sign_flip {
                residual = -residual;
            }
            // Reconstruct: pred + residual, wrapped into 8-bit. Since the
            // encoder already wrapped the diff to signed 8-bit, adding back
            // may need to be masked.
            let recon = ((pred + residual) as u32) & 0xFF;
            samples[y * w + x] = recon as u8;
        }
    }
    Ok(())
}

/// Fetch the FFV1 six-tap neighbourhood (LL, L, T, LT, TT, TR) for sample
/// `(x, y)`. This mirrors FFmpeg's `get_context` + `predict` helpers, which
/// run against a sample buffer padded with zero rows above the slice and
/// the following per-row edge conventions:
///
/// * Just before encoding row `y`, the encoder sets
///   `sample[y][-1] = sample[y - 1][0]` — i.e. the value to the left of the
///   first pixel in this row equals the first pixel of the previous row
///   (or zero above the first row, since the initial buffer was memset to
///   zero).
/// * Similarly `sample[y][w] = sample[y][w - 1]` (right edge).
/// * Rows above the slice (y < 0) are all zeros.
///
/// So the neighbour lookups at image boundaries are:
///
/// | position   | L                    | T        | LT                   | TR                    | LL                                   | TT                    |
/// |------------|----------------------|----------|----------------------|-----------------------|--------------------------------------|-----------------------|
/// | (0, 0)     | 0                    | 0        | 0                    | 0                     | 0                                    | 0                     |
/// | (0, y>0)   | `samples[y-1][0]`    | `T`      | 0 (set by edge fix)  | `samples[y-1][1]`     | 0                                    | `samples[y-2][0]`     |
/// | (1, 0)     | `samples[0][0]`      | 0        | 0                    | 0                     | 0                                    | 0                     |
/// | (x>1, 0)   | `samples[0][x-1]`    | 0        | 0                    | 0                     | `samples[0][x-2]`                    | 0                     |
/// | (w-1, y)   | `samples[y][w-2]`    | `samples[y-1][w-1]` | `samples[y-1][w-2]` | `samples[y-1][w-1]` (right edge trick) | `samples[y][w-3]`                    | `samples[y-2][w-1]`   |
#[inline]
fn neighbours(
    samples: &[u8],
    w: usize,
    _h: usize,
    x: usize,
    y: usize,
) -> (i32, i32, i32, i32, i32, i32) {
    // --- Previous row (or its zero sentinel) ---------------------------
    let prev_row_exists = y >= 1;
    let prev_row_base = if prev_row_exists { (y - 1) * w } else { 0 };
    let prev_row_sample = |col: isize| -> i32 {
        if !prev_row_exists {
            return 0;
        }
        if col < 0 {
            // `sample[y-1][-1]` was itself set to `sample[y-2][0]` during
            // the previous iteration — or zero if y-2 is outside the image.
            if y >= 2 {
                return samples[(y - 2) * w] as i32;
            }
            return 0;
        }
        if (col as usize) >= w {
            // Right-edge trick: `sample[y-1][w] == sample[y-1][w-1]`.
            return samples[prev_row_base + w - 1] as i32;
        }
        samples[prev_row_base + col as usize] as i32
    };

    // --- Current row --------------------------------------------------
    // `sample[y][-1]` = first pixel of the *previous* row (or zero).
    let cur_row_sample = |col: isize| -> i32 {
        if col < 0 {
            // -1 is the only negative index we'll ever ask for here.
            return if y >= 1 {
                samples[prev_row_base] as i32
            } else {
                0
            };
        }
        samples[y * w + col as usize] as i32
    };

    let l = cur_row_sample(x as isize - 1);
    // `LL` at x=0 resolves to `sample[y][-2]`, which the encoder leaves at
    // its memset-zero sentinel (only `-1` is patched by the edge trick).
    let big_l = if x >= 2 {
        samples[y * w + x - 2] as i32
    } else {
        0
    };

    let t = prev_row_sample(x as isize);
    let tl = prev_row_sample(x as isize - 1);
    let tr = prev_row_sample(x as isize + 1);

    let big_t = if y >= 2 {
        samples[(y - 2) * w + x] as i32
    } else {
        0
    };

    (big_l, l, t, tl, big_t, tr)
}

// -------------------------------------------------------------------------
// Slice header
// -------------------------------------------------------------------------

/// Range-coded slice header fields (RFC 9043 §4.4, coder_type=1).
#[derive(Clone, Debug)]
pub struct SliceHeader {
    /// X position in units of slice cells (`num_h_slices` cells total).
    pub slice_x: u32,
    /// Y position in units of slice cells.
    pub slice_y: u32,
    /// Width (number of cells occupied) minus 1.
    pub slice_width_minus1: u32,
    /// Height (number of cells occupied) minus 1.
    pub slice_height_minus1: u32,
    /// Per-plane quant_table_set index. For our simple profile this is [0;3].
    pub qt_idx: [u32; 3],
    pub picture_structure: u32,
    pub sar_num: u32,
    pub sar_den: u32,
}

impl SliceHeader {
    /// Build a header for a slice that occupies cell `(x, y)` alone in the
    /// num_h × num_v grid.
    pub fn single_cell(x: u32, y: u32) -> Self {
        Self {
            slice_x: x,
            slice_y: y,
            slice_width_minus1: 0,
            slice_height_minus1: 0,
            qt_idx: [0; 3],
            picture_structure: 0,
            sar_num: 0,
            sar_den: 0,
        }
    }

    pub fn encode(&self, enc: &mut RangeEncoder, num_planes: usize) {
        let mut st = [128u8; 32];
        enc.put_symbol_u(&mut st, self.slice_x);
        enc.put_symbol_u(&mut st, self.slice_y);
        enc.put_symbol_u(&mut st, self.slice_width_minus1);
        enc.put_symbol_u(&mut st, self.slice_height_minus1);
        for i in 0..num_planes {
            enc.put_symbol_u(&mut st, self.qt_idx[i]);
        }
        enc.put_symbol_u(&mut st, self.picture_structure);
        enc.put_symbol_u(&mut st, self.sar_num);
        enc.put_symbol_u(&mut st, self.sar_den);
    }

    pub fn parse(dec: &mut RangeDecoder<'_>, num_planes: usize) -> Result<Self> {
        let mut st = [128u8; 32];
        let slice_x = dec.get_symbol_u(&mut st);
        let slice_y = dec.get_symbol_u(&mut st);
        let slice_width_minus1 = dec.get_symbol_u(&mut st);
        let slice_height_minus1 = dec.get_symbol_u(&mut st);
        let mut qt_idx = [0u32; 3];
        for i in 0..num_planes.min(3) {
            qt_idx[i] = dec.get_symbol_u(&mut st);
        }
        let picture_structure = dec.get_symbol_u(&mut st);
        let sar_num = dec.get_symbol_u(&mut st);
        let sar_den = dec.get_symbol_u(&mut st);
        Ok(Self {
            slice_x,
            slice_y,
            slice_width_minus1,
            slice_height_minus1,
            qt_idx,
            picture_structure,
            sar_num,
            sar_den,
        })
    }
}

// -------------------------------------------------------------------------
// Whole-frame encode for our single-slice profile
// -------------------------------------------------------------------------

/// A lightweight view of one slice's per-plane pixel data. Each plane is a
/// contiguous row-major buffer of exactly `width * height` bytes.
pub struct SlicePlanes<'a> {
    pub y: &'a [u8],
    pub u: Option<&'a [u8]>,
    pub v: Option<&'a [u8]>,
    pub y_geom: PlaneGeom,
    pub c_geom: PlaneGeom,
}

/// Encode one FFV1 frame as a single slice covering the whole frame: leading
/// keyframe bit, slice header, planes, and a 3- or 8-byte footer depending
/// on `ec`. Returns the whole-packet bytes.
///
/// When `ec == true`, the footer carries the 1-byte `error_status` and 4-byte
/// big-endian CRC-32 parity required for FFmpeg compatibility with `-slicecrc
/// 1`. When `ec == false`, only the 3-byte `slice_size` field is appended.
pub fn encode_single_slice_frame(planes: &SlicePlanes<'_>, ec: bool) -> Vec<u8> {
    let tables = default_quant_tables();
    let ctx_count = context_count(&tables);

    let mut enc = RangeEncoder::new();
    // Leading keyframe bit on the first slice's range coder (our only one).
    let mut keystate = 128u8;
    enc.put_rac(&mut keystate, true);

    let num_planes = if planes.u.is_some() && planes.v.is_some() {
        3
    } else {
        1
    };
    let hdr = SliceHeader::single_cell(0, 0);
    hdr.encode(&mut enc, num_planes);

    // FFmpeg's YUV encoder uses two `PlaneContext`s: Y in `plane[0]`, U+V
    // both in `plane[1]`. The state for plane[1] is *not* reset between U
    // and V — V picks up where U left off. We mirror that here so our
    // bitstream is byte-identical to what a v3 decoder expects.
    let mut y_state = PlaneState::new(ctx_count);
    encode_plane(
        &mut enc,
        planes.y,
        planes.y_geom.width,
        planes.y_geom.height,
        &tables,
        &mut y_state,
    );
    if let (Some(u), Some(v)) = (planes.u, planes.v) {
        let mut chroma_state = PlaneState::new(ctx_count);
        encode_plane(
            &mut enc,
            u,
            planes.c_geom.width,
            planes.c_geom.height,
            &tables,
            &mut chroma_state,
        );
        encode_plane(
            &mut enc,
            v,
            planes.c_geom.width,
            planes.c_geom.height,
            &tables,
            &mut chroma_state,
        );
    }
    // Flush the range coder using FFV1's slice termination (put_rac 129 + 2
    // renormalisations). Yields the `slice_data` byte sequence.
    let slice_data = enc.finish_for_slice();

    // Append the footer.
    let mut out = slice_data;
    let data_len = out.len() as u32;
    out.push(((data_len >> 16) & 0xFF) as u8);
    out.push(((data_len >> 8) & 0xFF) as u8);
    out.push((data_len & 0xFF) as u8);
    if ec {
        out.push(0); // error_status
                     // CRC-32 IEEE of everything in the slice up to (but not including) the
                     // 4 bytes we're about to write. Stored big-endian so the CRC over
                     // the whole slice ends up being zero.
        let crc = crc32_ieee(&out);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    out
}

/// Decode one full FFV1 packet (one frame) that may contain multiple slices
/// laid out in a regular `num_h_slices × num_v_slices` grid.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame(
    data: &[u8],
    frame_width: u32,
    frame_height: u32,
    num_h_slices: u32,
    num_v_slices: u32,
    has_chroma: bool,
    log2_h_sub: u32,
    log2_v_sub: u32,
    ec: bool,
) -> Result<DecodedFrame> {
    if data.is_empty() {
        return Err(Error::invalid("FFV1 decode: empty packet"));
    }

    // Walk the packet from the back, reading each slice's trailer to find
    // its start byte. Collect `(start, size)` pairs in order.
    let trailer_len = if ec { 3 + 1 + 4 } else { 3 };
    let mut boundaries: Vec<(usize, usize)> = Vec::new(); // (start, slice_data_len)
    let mut tail = data.len();
    while tail > 0 {
        if tail < trailer_len {
            return Err(Error::invalid("FFV1 decode: truncated slice trailer"));
        }
        // slice_size lies at `[tail - trailer_len .. tail - trailer_len + 3]`.
        let sz_off = tail - trailer_len;
        let slice_size = (u32::from(data[sz_off]) << 16)
            | (u32::from(data[sz_off + 1]) << 8)
            | u32::from(data[sz_off + 2]);
        let slice_size = slice_size as usize;
        let total = slice_size + trailer_len;
        if total > tail {
            return Err(Error::invalid("FFV1 decode: slice_size overruns buffer"));
        }
        let start = tail - total;
        // Optionally verify the per-slice CRC.
        if ec {
            let crc_end = tail;
            let whole = &data[start..crc_end];
            if crc32_ieee(whole) != 0 {
                return Err(Error::invalid("FFV1 decode: slice CRC mismatch"));
            }
        }
        boundaries.push((start, slice_size));
        tail = start;
    }
    boundaries.reverse();

    let tables = default_quant_tables();
    let ctx_count = context_count(&tables);
    let num_planes = if has_chroma { 3 } else { 1 };

    let wu = frame_width as usize;
    let hu = frame_height as usize;
    let chroma_w = if has_chroma {
        frame_width.div_ceil(1 << log2_h_sub)
    } else {
        0
    };
    let chroma_h = if has_chroma {
        frame_height.div_ceil(1 << log2_v_sub)
    } else {
        0
    };
    let cwu = chroma_w as usize;
    let chu = chroma_h as usize;

    let mut y_buf = vec![0u8; wu * hu];
    let mut u_buf = if has_chroma {
        vec![0u8; cwu * chu]
    } else {
        Vec::new()
    };
    let mut v_buf = if has_chroma {
        vec![0u8; cwu * chu]
    } else {
        Vec::new()
    };

    for (slice_idx, (start, size)) in boundaries.iter().enumerate() {
        let slice_bytes = &data[*start..*start + *size];
        let mut dec = RangeDecoder::new(slice_bytes);
        if slice_idx == 0 {
            // The leading keyframe bit lives in the first slice's range coder.
            let mut keystate = 128u8;
            let _keyframe = dec.get_rac(&mut keystate);
        }
        let hdr = SliceHeader::parse(&mut dec, num_planes)?;

        // Resolve the slice's pixel region from its grid coordinates.
        let sx = hdr.slice_x;
        let sy = hdr.slice_y;
        let sw_cells = hdr.slice_width_minus1 + 1;
        let sh_cells = hdr.slice_height_minus1 + 1;
        if sx + sw_cells > num_h_slices || sy + sh_cells > num_v_slices {
            return Err(Error::invalid(
                "FFV1 decode: slice grid coordinates out of range",
            ));
        }
        let x0 = (sx * frame_width / num_h_slices) as usize;
        let x1 = ((sx + sw_cells) * frame_width / num_h_slices) as usize;
        let y0 = (sy * frame_height / num_v_slices) as usize;
        let y1 = ((sy + sh_cells) * frame_height / num_v_slices) as usize;
        let slice_w = x1 - x0;
        let slice_h = y1 - y0;
        if slice_w == 0 || slice_h == 0 {
            return Err(Error::invalid("FFV1 decode: empty slice region"));
        }

        // Y uses its own `PlaneContext` (FFmpeg's `plane[0]`), U and V
        // share a second one (`plane[1]`). U's final state carries through
        // into V — we deliberately don't reset between them.
        let mut y_state = PlaneState::new(ctx_count);
        let mut y_tile = vec![0u8; slice_w * slice_h];
        decode_plane(
            &mut dec,
            &mut y_tile,
            slice_w as u32,
            slice_h as u32,
            &tables,
            &mut y_state,
        )?;
        for row in 0..slice_h {
            let src_row = &y_tile[row * slice_w..row * slice_w + slice_w];
            let dst_off = (y0 + row) * wu + x0;
            y_buf[dst_off..dst_off + slice_w].copy_from_slice(src_row);
        }
        if has_chroma {
            let cx0 = x0 >> log2_h_sub;
            let cy0 = y0 >> log2_v_sub;
            let cslice_w = slice_w.div_ceil(1 << log2_h_sub);
            let cslice_h = slice_h.div_ceil(1 << log2_v_sub);
            let mut chroma_state = PlaneState::new(ctx_count);
            let mut u_tile = vec![0u8; cslice_w * cslice_h];
            decode_plane(
                &mut dec,
                &mut u_tile,
                cslice_w as u32,
                cslice_h as u32,
                &tables,
                &mut chroma_state,
            )?;
            let mut v_tile = vec![0u8; cslice_w * cslice_h];
            decode_plane(
                &mut dec,
                &mut v_tile,
                cslice_w as u32,
                cslice_h as u32,
                &tables,
                &mut chroma_state,
            )?;
            for row in 0..cslice_h {
                let dst_off = (cy0 + row) * cwu + cx0;
                let src = &u_tile[row * cslice_w..row * cslice_w + cslice_w];
                u_buf[dst_off..dst_off + cslice_w].copy_from_slice(src);
                let src = &v_tile[row * cslice_w..row * cslice_w + cslice_w];
                v_buf[dst_off..dst_off + cslice_w].copy_from_slice(src);
            }
        }
    }

    Ok(DecodedFrame {
        y: y_buf,
        u: if has_chroma { Some(u_buf) } else { None },
        v: if has_chroma { Some(v_buf) } else { None },
        y_geom: PlaneGeom {
            width: frame_width,
            height: frame_height,
        },
        c_geom: PlaneGeom {
            width: chroma_w,
            height: chroma_h,
        },
    })
}

/// Layout of a decoded frame (all slices joined).
pub struct DecodedFrame {
    pub y: Vec<u8>,
    pub u: Option<Vec<u8>>,
    pub v: Option<Vec<u8>>,
    pub y_geom: PlaneGeom,
    pub c_geom: PlaneGeom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn run_roundtrip(
        y: &[u8],
        u: Option<&[u8]>,
        v: Option<&[u8]>,
        w: u32,
        h: u32,
        cw: u32,
        ch: u32,
        ec: bool,
    ) {
        let planes = SlicePlanes {
            y,
            u,
            v,
            y_geom: PlaneGeom {
                width: w,
                height: h,
            },
            c_geom: PlaneGeom {
                width: cw,
                height: ch,
            },
        };
        let bytes = encode_single_slice_frame(&planes, ec);
        let has_chroma = u.is_some();
        let (l2h, l2v) = if has_chroma && cw * 2 == w && ch * 2 == h {
            (1, 1)
        } else {
            (0, 0)
        };
        let decoded = decode_frame(&bytes, w, h, 1, 1, has_chroma, l2h, l2v, ec).expect("decode");
        assert_eq!(decoded.y, y);
        if let Some(u) = u {
            assert_eq!(decoded.u.as_deref(), Some(u));
        }
        if let Some(v) = v {
            assert_eq!(decoded.v.as_deref(), Some(v));
        }
    }

    #[test]
    fn single_plane_roundtrip_flat() {
        let src = vec![128u8; 16 * 16];
        run_roundtrip(&src, None, None, 16, 16, 0, 0, false);
    }

    #[test]
    fn single_plane_roundtrip_gradient() {
        let w = 8u32;
        let h = 8u32;
        let mut src = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                src.push(((x * 32 + y * 4) & 0xFF) as u8);
            }
        }
        run_roundtrip(&src, None, None, w, h, 0, 0, false);
    }

    #[test]
    fn three_plane_roundtrip_420() {
        let w = 16u32;
        let h = 16u32;
        let cw = w / 2;
        let ch = h / 2;
        let y_src: Vec<u8> = (0..w * h).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let u_src: Vec<u8> = (0..cw * ch).map(|i| ((i * 13 + 64) & 0xFF) as u8).collect();
        let v_src: Vec<u8> = (0..cw * ch)
            .map(|i| ((i * 19 + 128) & 0xFF) as u8)
            .collect();
        run_roundtrip(&y_src, Some(&u_src), Some(&v_src), w, h, cw, ch, false);
    }

    #[test]
    fn three_plane_roundtrip_420_with_crc() {
        let w = 16u32;
        let h = 16u32;
        let cw = w / 2;
        let ch = h / 2;
        let y_src: Vec<u8> = (0..w * h).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let u_src: Vec<u8> = (0..cw * ch).map(|i| ((i * 13 + 64) & 0xFF) as u8).collect();
        let v_src: Vec<u8> = (0..cw * ch)
            .map(|i| ((i * 19 + 128) & 0xFF) as u8)
            .collect();
        run_roundtrip(&y_src, Some(&u_src), Some(&v_src), w, h, cw, ch, true);
    }

    #[test]
    fn three_plane_roundtrip_444_random() {
        let w = 32u32;
        let h = 24u32;
        let mut rng = 0xdead_beefu32;
        let mut rand_byte = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 16) as u8
        };
        let y_src: Vec<u8> = (0..w * h).map(|_| rand_byte()).collect();
        let u_src: Vec<u8> = (0..w * h).map(|_| rand_byte()).collect();
        let v_src: Vec<u8> = (0..w * h).map(|_| rand_byte()).collect();
        run_roundtrip(&y_src, Some(&u_src), Some(&v_src), w, h, w, h, false);
    }
}
