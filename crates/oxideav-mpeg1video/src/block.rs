//! Block-level decoding: DC + AC coefficient parsing, dequantisation, IDCT.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::dct::idct8x8;
use crate::headers::ZIGZAG;
use crate::tables::dct_coeffs::{self, DctSym};
use crate::tables::dct_dc;
use crate::vlc;

/// Sign-extend an `size`-bit unsigned DC-differential value to i32.
fn extend_dc(value: u32, size: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let vt = 1u32 << (size - 1);
    if value < vt {
        (value as i32) - ((1i32 << size) - 1)
    } else {
        value as i32
    }
}

/// Decode one intra macroblock block. `is_chroma` picks the DC size table.
/// `prev_dc_pel` carries the running "DC value in pel-space" (i.e. already
/// multiplied by 8 compared to the DC size differential) for the component
/// across blocks of the same slice.
pub fn decode_intra_block(
    br: &mut BitReader<'_>,
    is_chroma: bool,
    prev_dc_pel: &mut i32,
    quant_scale: u8,
    intra_quant: &[u8; 64],
    out_samples: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    // 1. DC differential.
    let dc_tbl = if is_chroma {
        dct_dc::chroma()
    } else {
        dct_dc::luma()
    };
    let dc_size = vlc::decode(br, dc_tbl)?;
    let dc_diff = if dc_size == 0 {
        0
    } else {
        let bits = br.read_u32(dc_size as u32)?;
        extend_dc(bits, dc_size as u32)
    };
    // Per §2.4.4.1, `dct_dc_*_past` stores the reconstructed DC coefficient
    // in pel-space (already multiplied by 8). Reset value at slice start is
    // 1024. The reconstructed DC for this block is:
    //   dct_recon[0][0] = dct_dc_differential * 8 + dct_dc_past
    // and then `dct_dc_past = dct_recon[0][0]`.
    let dc_rec = prev_dc_pel.wrapping_add(dc_diff * 8);
    *prev_dc_pel = dc_rec;

    // 2. Zig-zag AC coefficients using Table B-14.
    let mut coeffs = [0i32; 64];
    coeffs[0] = dc_rec;

    let ac_tbl = dct_coeffs::table();
    let mut k: usize = 1;
    loop {
        if k >= 64 {
            break;
        }
        let sym = vlc::decode(br, ac_tbl)?;
        let (run, level) = match sym {
            DctSym::Eob => break,
            DctSym::EobOrFirstOne => break,
            DctSym::RunLevel { run, level_abs } => {
                let sign = br.read_u32(1)?;
                let mut lv = level_abs as i32;
                if sign == 1 {
                    lv = -lv;
                }
                (run as usize, lv)
            }
            DctSym::Escape => {
                let run = br.read_u32(6)? as usize;
                // Short form: 8-bit signed level.
                let first = br.read_u32(8)? as i32;
                let level = if first == 0 {
                    // Long-escape form (MPEG-1): another 8 bits give an
                    // unsigned positive level ∈ 128..=255.
                    let l = br.read_u32(8)? as i32;
                    if l < 128 {
                        return Err(Error::invalid("dct escape: long form level < 128"));
                    }
                    l
                } else if first == 128 {
                    // Long-escape form negative: following 8 bits form
                    // level ∈ -256..=-129.
                    let l = br.read_u32(8)? as i32;
                    if l > 128 {
                        return Err(Error::invalid("dct escape: long form neg level > 128"));
                    }
                    l - 256
                } else if first >= 128 {
                    first - 256
                } else {
                    first
                };
                (run, level)
            }
        };
        k += run;
        if k >= 64 {
            return Err(Error::invalid("intra block: AC run past end"));
        }
        // Intra dequantisation per §2.4.4.1:
        //   coeff' = (2 * level * quantizer_scale * W[i]) / 16
        // followed by "mismatch control" (make odd) and saturation to ±2047.
        let qf = intra_quant[ZIGZAG[k]] as i32;
        let mut rec = (2 * level * quant_scale as i32 * qf) / 16;
        if rec & 1 == 0 && rec != 0 {
            rec = if rec > 0 { rec - 1 } else { rec + 1 };
        }
        rec = rec.clamp(-2048, 2047);
        coeffs[ZIGZAG[k]] = rec;
        k += 1;
    }

    // 3. IDCT.
    let mut fblock = [0.0f32; 64];
    for i in 0..64 {
        fblock[i] = coeffs[i] as f32;
    }
    idct8x8(&mut fblock);

    // Write back, clamped to [0,255].
    for j in 0..8 {
        for i in 0..8 {
            let v = fblock[j * 8 + i];
            let px = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v.round() as u8
            };
            out_samples[j * dst_stride + i] = px;
        }
    }
    Ok(())
}
