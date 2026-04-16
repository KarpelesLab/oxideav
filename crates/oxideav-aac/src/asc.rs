//! AudioSpecificConfig (ASC) parser — ISO/IEC 14496-3 §1.6.2.1.
//!
//! ASC is the typical setup blob for in-MP4 AAC streams. It packs the
//! object type, sample rate index (with optional 24-bit escape), and
//! channel configuration. SBR / PS extensions append further fields; we
//! detect them and surface them so the caller can decide whether to fail.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::syntax::{sample_rate, AOT_PS, AOT_SBR};

#[derive(Clone, Debug)]
pub struct AudioSpecificConfig {
    pub object_type: u8,
    pub sampling_frequency_index: u8,
    pub sampling_frequency: u32,
    pub channel_configuration: u8,
    /// If SBR-extended, the sampling frequency *after* SBR upsampling.
    pub ext_sampling_frequency: Option<u32>,
    pub sbr_present: bool,
    pub ps_present: bool,
}

fn read_audio_object_type(br: &mut BitReader<'_>) -> Result<u8> {
    let mut aot = br.read_u32(5)? as u8;
    if aot == 31 {
        aot = (32 + br.read_u32(6)?) as u8;
    }
    Ok(aot)
}

fn read_sampling_frequency(br: &mut BitReader<'_>) -> Result<(u8, u32)> {
    let idx = br.read_u32(4)? as u8;
    let freq = if idx == 0xF {
        br.read_u32(24)?
    } else {
        sample_rate(idx).ok_or_else(|| Error::invalid("ASC: reserved SF index"))?
    };
    Ok((idx, freq))
}

/// Parse an AudioSpecificConfig blob.
pub fn parse_asc(data: &[u8]) -> Result<AudioSpecificConfig> {
    let mut br = BitReader::new(data);

    let mut object_type = read_audio_object_type(&mut br)?;
    let (mut sampling_frequency_index, mut sampling_frequency) = read_sampling_frequency(&mut br)?;
    let channel_configuration = br.read_u32(4)? as u8;

    let mut sbr_present = false;
    let mut ps_present = false;
    let mut ext_sampling_frequency = None;

    // Explicit SBR/PS — object_type 5 (SBR) or 29 (PS) advertises an extension
    // before the GASpecificConfig.
    if object_type == AOT_SBR || object_type == AOT_PS {
        sbr_present = true;
        if object_type == AOT_PS {
            ps_present = true;
        }
        let (_ext_idx, ext_sf) = read_sampling_frequency(&mut br)?;
        ext_sampling_frequency = Some(ext_sf);
        // The actual underlying object type follows.
        object_type = read_audio_object_type(&mut br)?;
        // For SBR-only streams the underlying is usually AAC-LC (2). PS
        // (29) implies stereo from a mono base channel.
        // Note we don't carry forward the index, just the rate — that's
        // enough for downstream IMDCT sizing.
        let _ = sampling_frequency_index;
        let _ = sampling_frequency;
        sampling_frequency_index = 0xF;
        sampling_frequency = ext_sf;
    }

    Ok(AudioSpecificConfig {
        object_type,
        sampling_frequency_index,
        sampling_frequency,
        channel_configuration,
        ext_sampling_frequency,
        sbr_present,
        ps_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::AOT_AAC_LC;

    #[test]
    fn lc_44100_stereo() {
        // 5 bits aot=2, 4 bits idx=4 (44100), 4 bits ch=2, then 3 bits gaSpecific
        // = 00010 0100 0010 000 = 0001 0010 0001 0000 -> 0x12, 0x10
        let bytes = [0x12u8, 0x10];
        let asc = parse_asc(&bytes).unwrap();
        assert_eq!(asc.object_type, AOT_AAC_LC);
        assert_eq!(asc.sampling_frequency, 44_100);
        assert_eq!(asc.channel_configuration, 2);
        assert!(!asc.sbr_present);
    }
}
