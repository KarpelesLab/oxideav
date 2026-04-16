//! Video Object Layer header (ISO/IEC 14496-2 §6.2.3).
//!
//! The VOL carries the per-sequence picture geometry, frame rate, quantisation
//! type selector and shape information. Populating `CodecParameters` requires a
//! successfully parsed VOL.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `aspect_ratio_info` — Table 6-12.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AspectRatioInfo {
    Square,
    Par12_11,
    Par10_11,
    Par16_11,
    Par40_33,
    Extended { par_width: u8, par_height: u8 },
    Reserved(u8),
}

/// `video_object_layer_shape` — Table 6-14.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType {
    Rectangular = 0,
    Binary = 1,
    BinaryOnly = 2,
    GrayScale = 3,
}

/// `chroma_format` — Table 6-15.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    Yuv420,
    Reserved(u8),
}

/// Parsed Video Object Layer header. Only the subset of fields this decoder
/// consumes or exposes is stored — unparsed optional sub-blocks (scalability,
/// sprite VOPs, complexity-estimation) are read-then-discarded.
#[derive(Clone, Debug)]
pub struct VideoObjectLayer {
    pub random_accessible_vol: bool,
    pub video_object_type_indication: u8,
    pub is_object_layer_identifier: bool,
    pub verid: u8,
    pub priority: u8,
    pub aspect_ratio_info: AspectRatioInfo,
    /// VOL control parameters — bitrate / VBV etc. Not exposed.
    pub vol_control_parameters: bool,
    pub chroma_format: ChromaFormat,
    pub low_delay: bool,
    pub vbv_parameters_present: bool,
    pub shape: ShapeType,
    /// `vop_time_increment_resolution` — ticks per second, used to reconstruct
    /// VOP timestamps. Also serves as the rational numerator for frame_rate.
    pub vop_time_increment_resolution: u32,
    /// Number of bits used by `vop_time_increment` in each VOP header — the
    /// smallest integer that can hold (resolution - 1).
    pub vop_time_increment_bits: u32,
    /// If true the VOL has a fixed inter-VOP time increment (used as the
    /// denominator for frame_rate).
    pub fixed_vop_rate: bool,
    pub fixed_vop_time_increment: u32,
    /// Coded picture width (in luma pels).
    pub width: u32,
    /// Coded picture height (in luma pels).
    pub height: u32,
    pub interlaced: bool,
    pub obmc_disable: bool,
    pub sprite_enable: u8,
    pub not_8_bit: bool,
    pub quant_precision: u8,
    pub bits_per_pixel: u8,
    /// Quantisation type: false = H.263 quant, true = MPEG-4 (matrix) quant.
    pub mpeg_quant: bool,
    pub intra_quant_matrix: Option<[u8; 64]>,
    pub non_intra_quant_matrix: Option<[u8; 64]>,
    pub quarter_sample: bool,
    pub complexity_estimation_disable: bool,
    pub resync_marker_disable: bool,
    pub data_partitioned: bool,
    pub reversible_vlc: bool,
    pub newpred_enable: bool,
    pub reduced_resolution_vop_enable: bool,
    pub scalability: bool,
}

/// Default MPEG-4 intra quant matrix — Table 7-1.
pub const DEFAULT_INTRA_QUANT_MATRIX: [u8; 64] = [
    8, 17, 18, 19, 21, 23, 25, 27, 17, 18, 19, 21, 23, 25, 27, 28, 20, 21, 22, 23, 24, 26, 28, 30,
    21, 22, 23, 24, 26, 28, 30, 32, 22, 23, 24, 26, 28, 30, 32, 35, 23, 24, 26, 28, 30, 32, 35, 38,
    25, 26, 28, 30, 32, 35, 38, 41, 27, 28, 30, 32, 35, 38, 41, 45,
];

/// Default MPEG-4 non-intra quant matrix — Table 7-2.
pub const DEFAULT_NON_INTRA_QUANT_MATRIX: [u8; 64] = [
    16, 17, 18, 19, 20, 21, 22, 23, 17, 18, 19, 20, 21, 22, 23, 24, 18, 19, 20, 21, 22, 23, 24, 25,
    19, 20, 21, 22, 23, 24, 26, 27, 20, 21, 22, 23, 25, 26, 27, 28, 21, 22, 23, 24, 26, 27, 28, 30,
    22, 23, 24, 26, 27, 28, 30, 31, 23, 24, 25, 27, 28, 30, 31, 33,
];

/// Zig-zag scan order used for block coefficient transmission.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

fn bits_needed(max_value: u32) -> u32 {
    // Number of bits required to represent `max_value`. Minimum 1 bit per spec.
    if max_value == 0 {
        return 1;
    }
    32 - max_value.leading_zeros()
}

fn parse_aspect_ratio(br: &mut BitReader<'_>) -> Result<AspectRatioInfo> {
    let code = br.read_u32(4)? as u8;
    Ok(match code {
        1 => AspectRatioInfo::Square,
        2 => AspectRatioInfo::Par12_11,
        3 => AspectRatioInfo::Par10_11,
        4 => AspectRatioInfo::Par16_11,
        5 => AspectRatioInfo::Par40_33,
        0xF => {
            let par_width = br.read_u32(8)? as u8;
            let par_height = br.read_u32(8)? as u8;
            AspectRatioInfo::Extended {
                par_width,
                par_height,
            }
        }
        other => AspectRatioInfo::Reserved(other),
    })
}

/// Parse a VOL header payload. `br` must be positioned just past the
/// `0x000001[20..2F]` start code.
///
/// This implementation covers rectangular-shape ASP bitstreams — the dominant
/// case for XVID/DivX/FMP4. Scalable / binary-shape / sprite-only paths return
/// `Error::Unsupported`.
pub fn parse_vol(br: &mut BitReader<'_>) -> Result<VideoObjectLayer> {
    let random_accessible_vol = br.read_u1()? == 1;
    let video_object_type_indication = br.read_u32(8)? as u8;

    let is_object_layer_identifier = br.read_u1()? == 1;
    let mut verid = 1u8;
    let mut priority = 0u8;
    if is_object_layer_identifier {
        verid = br.read_u32(4)? as u8;
        priority = br.read_u32(3)? as u8;
    }

    let aspect_ratio_info = parse_aspect_ratio(br)?;

    let vol_control_parameters = br.read_u1()? == 1;
    let mut chroma_format = ChromaFormat::Yuv420;
    let mut low_delay = false;
    let mut vbv_parameters_present = false;
    if vol_control_parameters {
        let cf = br.read_u32(2)? as u8;
        chroma_format = match cf {
            1 => ChromaFormat::Yuv420,
            other => ChromaFormat::Reserved(other),
        };
        low_delay = br.read_u1()? == 1;
        let vbv_parameters = br.read_u1()? == 1;
        if vbv_parameters {
            vbv_parameters_present = true;
            // first_half_bit_rate (15) + marker + latter_half_bit_rate (15) + marker
            br.skip(15)?;
            br.read_marker()?;
            br.skip(15)?;
            br.read_marker()?;
            // first_half_vbv_buffer_size (15) + marker + latter (3)
            br.skip(15)?;
            br.read_marker()?;
            br.skip(3)?;
            // first_half_vbv_occupancy (11) + marker + latter (15) + marker
            br.skip(11)?;
            br.read_marker()?;
            br.skip(15)?;
            br.read_marker()?;
        }
    }

    let shape_code = br.read_u32(2)? as u8;
    let shape = match shape_code {
        0 => ShapeType::Rectangular,
        1 => ShapeType::Binary,
        2 => ShapeType::BinaryOnly,
        3 => ShapeType::GrayScale,
        _ => unreachable!(),
    };
    if shape != ShapeType::Rectangular {
        return Err(Error::unsupported(
            "mpeg4 non-rectangular shape VOL: follow-up",
        ));
    }

    br.read_marker()?; // marker
    let vop_time_increment_resolution = br.read_u32(16)?;
    br.read_marker()?; // marker
    if vop_time_increment_resolution == 0 {
        return Err(Error::invalid(
            "mpeg4 VOL: vop_time_increment_resolution = 0",
        ));
    }
    let vop_time_increment_bits = bits_needed(vop_time_increment_resolution - 1).max(1);

    let fixed_vop_rate = br.read_u1()? == 1;
    let mut fixed_vop_time_increment = 1u32;
    if fixed_vop_rate {
        fixed_vop_time_increment = br.read_u32(vop_time_increment_bits)?;
        if fixed_vop_time_increment == 0 {
            fixed_vop_time_increment = 1;
        }
    }

    // Only for Rectangular shape: width/height.
    br.read_marker()?;
    let width = br.read_u32(13)?;
    br.read_marker()?;
    let height = br.read_u32(13)?;
    br.read_marker()?;

    let interlaced = br.read_u1()? == 1;
    let obmc_disable = br.read_u1()? == 1;

    // sprite_enable — 1 bit in verid<=1, 2 bits in verid>=2 per spec corrigendum.
    let sprite_enable = if verid == 1 {
        br.read_u32(1)? as u8
    } else {
        br.read_u32(2)? as u8
    };
    if sprite_enable != 0 {
        return Err(Error::unsupported(
            "mpeg4 sprite / GMC VOPs: follow-up (out of scope)",
        ));
    }

    let not_8_bit = br.read_u1()? == 1;
    let (quant_precision, bits_per_pixel) = if not_8_bit {
        (br.read_u32(4)? as u8, br.read_u32(4)? as u8)
    } else {
        (5u8, 8u8)
    };

    let mpeg_quant = br.read_u1()? == 1;
    let mut intra_q: Option<[u8; 64]> = None;
    let mut non_intra_q: Option<[u8; 64]> = None;
    if mpeg_quant {
        if br.read_u1()? == 1 {
            intra_q = Some(read_quant_matrix(br)?);
        }
        if br.read_u1()? == 1 {
            non_intra_q = Some(read_quant_matrix(br)?);
        }
    }

    // quarter_sample was added in verid>=2 (MPEG-4 Part 2 / 2000 corrigendum).
    // verid==1 streams never carry this bit.
    let quarter_sample = if verid != 1 {
        let v = br.read_u1()? == 1;
        if v {
            return Err(Error::unsupported(
                "mpeg4 quarter-pel motion: follow-up (out of scope)",
            ));
        }
        v
    } else {
        false
    };

    let complexity_estimation_disable = br.read_u1()? == 1;
    if !complexity_estimation_disable {
        return Err(Error::unsupported(
            "mpeg4 complexity_estimation_header: follow-up",
        ));
    }

    let resync_marker_disable = br.read_u1()? == 1;
    let data_partitioned = br.read_u1()? == 1;
    let mut reversible_vlc = false;
    if data_partitioned {
        reversible_vlc = br.read_u1()? == 1;
    }

    // newpred_enable and reduced_resolution_vop_enable were added in verid>=2.
    let (newpred_enable, reduced_resolution_vop_enable) = if verid != 1 {
        let np = br.read_u1()? == 1;
        if np {
            return Err(Error::unsupported("mpeg4 newpred_enable: follow-up"));
        }
        let rr = br.read_u1()? == 1;
        (np, rr)
    } else {
        (false, false)
    };

    let scalability = br.read_u1()? == 1;
    if scalability {
        return Err(Error::unsupported("mpeg4 scalable VOL: follow-up"));
    }

    Ok(VideoObjectLayer {
        random_accessible_vol,
        video_object_type_indication,
        is_object_layer_identifier,
        verid,
        priority,
        aspect_ratio_info,
        vol_control_parameters,
        chroma_format,
        low_delay,
        vbv_parameters_present,
        shape,
        vop_time_increment_resolution,
        vop_time_increment_bits,
        fixed_vop_rate,
        fixed_vop_time_increment,
        width,
        height,
        interlaced,
        obmc_disable,
        sprite_enable,
        not_8_bit,
        quant_precision,
        bits_per_pixel,
        mpeg_quant,
        intra_quant_matrix: intra_q,
        non_intra_quant_matrix: non_intra_q,
        quarter_sample,
        complexity_estimation_disable,
        resync_marker_disable,
        data_partitioned,
        reversible_vlc,
        newpred_enable,
        reduced_resolution_vop_enable,
        scalability,
    })
}

/// Parse a custom quant matrix (§6.2.3). 1..=64 signed-8-bit values,
/// terminated by a zero value; remaining entries default.
fn read_quant_matrix(br: &mut BitReader<'_>) -> Result<[u8; 64]> {
    let mut m = [0u8; 64];
    let mut last_nonzero = 8u8;
    for i in 0..64 {
        let v = br.read_u32(8)? as u8;
        if v == 0 {
            // Fill remainder with the last non-zero value per §6.2.3.
            for j in i..64 {
                m[ZIGZAG[j]] = last_nonzero;
            }
            return Ok(m);
        }
        m[ZIGZAG[i]] = v;
        last_nonzero = v;
    }
    Ok(m)
}

impl VideoObjectLayer {
    /// Number of macroblocks horizontally (16 pels each), rounded up.
    pub fn mb_width(&self) -> u32 {
        self.width.div_ceil(16)
    }
    /// Number of macroblocks vertically (16 pels each), rounded up.
    pub fn mb_height(&self) -> u32 {
        self.height.div_ceil(16)
    }

    /// Rational frame rate as (num, den) in seconds. Returns (resolution, 1)
    /// if no fixed rate is advertised — this is the best the VOL header gives
    /// us in that case.
    pub fn frame_rate(&self) -> (i64, i64) {
        if self.fixed_vop_rate && self.fixed_vop_time_increment != 0 {
            (
                self.vop_time_increment_resolution as i64,
                self.fixed_vop_time_increment as i64,
            )
        } else {
            (self.vop_time_increment_resolution as i64, 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_needed_basic() {
        assert_eq!(bits_needed(0), 1);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(9), 4);
        assert_eq!(bits_needed(15), 4);
        assert_eq!(bits_needed(16), 5);
    }
}
