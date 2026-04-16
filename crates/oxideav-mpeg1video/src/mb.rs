//! Macroblock / slice-level decoding.
//!
//! Only I-pictures are fully implemented. P/B are stubbed — they parse the
//! macroblock type but do not perform motion compensation yet.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::block::decode_intra_block;
use crate::headers::{PictureType, SequenceHeader};
use crate::picture::PictureBuffer;
use crate::tables::{mb_type, mba};
use crate::vlc;

/// Decode one slice. A slice begins with a slice_start_code (the caller has
/// already consumed the start code byte) and extends until the next start
/// code. `slice_start_code` is the MB row number (1..=175 mapped to row 0..=174).
pub fn decode_slice(
    br: &mut BitReader<'_>,
    slice_start_code: u8,
    seq: &SequenceHeader,
    picture_type: PictureType,
    pic: &mut PictureBuffer,
    dc_pred: &mut [i32; 3],
) -> Result<()> {
    // Quantiser_scale (5 bits).
    let mut quant_scale = br.read_u32(5)? as u8;
    if quant_scale == 0 {
        quant_scale = 1;
    }

    // extra_bit_slice.
    while br.read_u32(1)? == 1 {
        br.read_u32(8)?;
    }

    // DC predictors (`dct_dc_*_past`) are stored in pel-space per §2.4.4.1.
    // At slice start they reset to 1024 (i.e. a mid-grey DC coefficient).
    dc_pred[0] = 1024;
    dc_pred[1] = 1024;
    dc_pred[2] = 1024;

    let mb_row = slice_start_code as i32 - 1;
    if mb_row < 0 || (mb_row as usize) >= pic.mb_height {
        return Err(Error::invalid("slice: MB row out of range"));
    }
    let mb_width = pic.mb_width as i32;
    let mut mb_addr: i32 = mb_row * mb_width - 1;

    loop {
        // Read MB address increment — may be a sequence of stuffing codes
        // followed by an escape + actual increment.
        let mut incr: u32 = 0;
        loop {
            let sym = vlc::decode(br, mba::table())?;
            if sym == mba::STUFFING {
                continue;
            }
            if sym == mba::ESCAPE {
                incr += 33;
                continue;
            }
            incr += sym as u32;
            break;
        }
        mb_addr += incr as i32;
        if mb_addr >= (mb_row + 1) * mb_width {
            return Err(Error::invalid("slice: MB address past end of row"));
        }

        let mb_x = (mb_addr % mb_width) as usize;
        let mb_y = (mb_addr / mb_width) as usize;

        // macroblock_type per picture.
        let mb_type_flags = match picture_type {
            PictureType::I => vlc::decode(br, mb_type::I_TABLE)?,
            PictureType::P => vlc::decode(br, mb_type::P_TABLE)?,
            PictureType::B => vlc::decode(br, mb_type::B_TABLE)?,
            PictureType::D => {
                return Err(Error::unsupported("D-picture not supported"));
            }
        };

        if !mb_type_flags.intra {
            // Reset DC predictors per §2.4.4.1 when a non-intra MB is seen.
            dc_pred[0] = 128;
            dc_pred[1] = 128;
            dc_pred[2] = 128;
            return Err(Error::unsupported(
                "mpeg1video: non-intra macroblocks (P/B) not implemented",
            ));
        }

        if mb_type_flags.quant {
            let qs = br.read_u32(5)? as u8;
            if qs != 0 {
                quant_scale = qs;
            }
        }

        // Decode 6 blocks (Y0, Y1, Y2, Y3, Cb, Cr).
        for b in 0..6 {
            let (is_chroma, comp_idx, dst_x0, dst_y0, stride_ptr) = match b {
                0 => (false, 0, mb_x * 16, mb_y * 16, pic.y_stride),
                1 => (false, 0, mb_x * 16 + 8, mb_y * 16, pic.y_stride),
                2 => (false, 0, mb_x * 16, mb_y * 16 + 8, pic.y_stride),
                3 => (false, 0, mb_x * 16 + 8, mb_y * 16 + 8, pic.y_stride),
                4 => (true, 1, mb_x * 8, mb_y * 8, pic.c_stride),
                5 => (true, 2, mb_x * 8, mb_y * 8, pic.c_stride),
                _ => unreachable!(),
            };
            let buf = match comp_idx {
                0 => &mut pic.y[..],
                1 => &mut pic.cb[..],
                _ => &mut pic.cr[..],
            };
            let sub = &mut buf[dst_y0 * stride_ptr + dst_x0..];
            decode_intra_block(
                br,
                is_chroma,
                &mut dc_pred[comp_idx],
                quant_scale,
                &seq.intra_quantiser,
                sub,
                stride_ptr,
            )?;
        }

        // Peek ahead: if next 23 bits are zero (start of next start code) we
        // stop. Otherwise loop for the next MB.
        // Simpler: check remaining bits in the reader (byte_pos vs slice end
        // is managed by the caller supplying a bounded slice).
        if br.bits_remaining() < 24 {
            break;
        }
        // Check if upcoming bits are 23 zero bits (next_start_code).
        let peek = br.peek_u32(23)?;
        if peek == 0 {
            break;
        }
    }

    Ok(())
}
