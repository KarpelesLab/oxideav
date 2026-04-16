//! AVI ↔ oxideav codec id mapping.
//!
//! The demuxer reads the 4-byte `biCompression` FourCC for video streams and
//! the 16-bit `wFormatTag` for audio streams, and maps them to the stable
//! oxideav `codec_id` strings used by the rest of the pipeline.
//!
//! The muxer goes the other way: given a `CodecParameters`, it returns the
//! payload bytes to place in a `strf` chunk (a `BITMAPINFOHEADER` for video
//! or a `WAVEFORMATEX` for audio) plus the 4-byte chunk suffix used to tag
//! its packets inside the `movi` list (`dc` for video, `wb` for audio).
//!
//! This is the one and only codec-aware module in the crate.

use oxideav_core::{CodecId, CodecParameters, Error, MediaType, Result};

use crate::stream_format::{write_bitmap_info_header, write_waveformatex};

/// FourCC → codec_id for video streams. Uppercase keys to normalise the
/// case-insensitive FourCCs some encoders emit (e.g. `mjpg`/`MJPG`/`MJpg`).
pub fn video_codec_id(fourcc: &[u8; 4]) -> CodecId {
    let upper = uppercase4(fourcc);
    let name = match &upper {
        b"MJPG" => "mjpeg",
        b"FFV1" => "ffv1",
        // BI_RGB (uncompressed): biCompression=0x00000000. FourCC is all zeros.
        [0, 0, 0, 0] => "rgb24",
        b"DIB " => "rgb24",
        b"RGB " => "rgb24",
        other => {
            let s = std::str::from_utf8(other).unwrap_or("????");
            return CodecId::new(format!("avi:{s}"));
        }
    };
    CodecId::new(name)
}

/// WAVEFORMATEX wFormatTag → codec_id.
pub fn audio_codec_id(format_tag: u16) -> CodecId {
    let name = match format_tag {
        0x0001 => "pcm_s16le",
        0x0003 => "pcm_f32le",
        0x0050 => "mp2",
        0x0055 => "mp3",
        0x00FF => "aac",
        0x2000 => "ac3",
        0x706D => "aac",
        0xF1AC => "flac",
        other => return CodecId::new(format!("avi:tag_{other:04x}")),
    };
    CodecId::new(name)
}

/// Result of building a stream-format chunk for the muxer.
pub(crate) struct StrfEntry {
    /// Two-ASCII-digit FourCC suffix used for packet chunks in `movi`: `dc`
    /// for compressed video, `wb` for audio, `db` for uncompressed video.
    pub chunk_suffix: [u8; 2],
    /// 4-byte `fccHandler` field for the `strh` chunk.
    pub handler_fourcc: [u8; 4],
    /// Full `strf` payload (BITMAPINFOHEADER or WAVEFORMATEX).
    pub strf: Vec<u8>,
    /// ffmpeg-compatible four-char stream-type tag (`vids`/`auds`) for strh.
    pub strh_type: [u8; 4],
    /// Sample size hint for `strh.dwSampleSize` — 0 means "variable" (VBR).
    pub sample_size: u32,
    /// Scale / rate pair for `strh.dwScale / dwRate` (rate/scale = samples
    /// per second). For video we use frame_rate; for audio sample_rate/1.
    pub scale: u32,
    pub rate: u32,
}

/// Build the `strf` chunk + `strh` metadata for the given stream. Errors with
/// `Unsupported` if the codec has no AVI packaging in our table.
pub(crate) fn build_strf(params: &CodecParameters) -> Result<StrfEntry> {
    match params.codec_id.as_str() {
        "mjpeg" => mjpeg_entry(params),
        "ffv1" => ffv1_entry(params),
        "pcm_s16le" => pcm_s16le_entry(params),
        other => Err(Error::unsupported(format!(
            "avi muxer: no packaging for codec {other}"
        ))),
    }
}

fn mjpeg_entry(params: &CodecParameters) -> Result<StrfEntry> {
    if params.media_type != MediaType::Video {
        return Err(Error::invalid("avi muxer: mjpeg must be video"));
    }
    let width = params
        .width
        .ok_or_else(|| Error::invalid("avi muxer: mjpeg requires width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("avi muxer: mjpeg requires height"))?;
    let strf = write_bitmap_info_header(width, height, *b"MJPG", 24, &params.extradata);
    let (scale, rate) = video_scale_rate(params);
    Ok(StrfEntry {
        chunk_suffix: *b"dc",
        handler_fourcc: *b"MJPG",
        strf,
        strh_type: *b"vids",
        sample_size: 0,
        scale,
        rate,
    })
}

fn ffv1_entry(params: &CodecParameters) -> Result<StrfEntry> {
    if params.media_type != MediaType::Video {
        return Err(Error::invalid("avi muxer: ffv1 must be video"));
    }
    let width = params
        .width
        .ok_or_else(|| Error::invalid("avi muxer: ffv1 requires width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("avi muxer: ffv1 requires height"))?;
    let strf = write_bitmap_info_header(width, height, *b"FFV1", 24, &params.extradata);
    let (scale, rate) = video_scale_rate(params);
    Ok(StrfEntry {
        chunk_suffix: *b"dc",
        handler_fourcc: *b"FFV1",
        strf,
        strh_type: *b"vids",
        sample_size: 0,
        scale,
        rate,
    })
}

fn pcm_s16le_entry(params: &CodecParameters) -> Result<StrfEntry> {
    if params.media_type != MediaType::Audio {
        return Err(Error::invalid("avi muxer: pcm_s16le must be audio"));
    }
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("avi muxer: pcm requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("avi muxer: pcm requires sample_rate"))?;
    let bits_per_sample: u16 = 16;
    let block_align = channels * (bits_per_sample / 8);
    let avg_bytes_per_sec = sample_rate * block_align as u32;
    let strf = write_waveformatex(
        0x0001,
        channels,
        sample_rate,
        avg_bytes_per_sec,
        block_align,
        bits_per_sample,
        &[],
    );
    // AVI stores PCM with dwSampleSize = block_align so chunks can hold any
    // integer number of frames. Scale/rate tracks sample_rate directly.
    Ok(StrfEntry {
        chunk_suffix: *b"wb",
        handler_fourcc: *b"\0\0\0\0",
        strf,
        strh_type: *b"auds",
        sample_size: block_align as u32,
        scale: 1,
        rate: sample_rate,
    })
}

fn video_scale_rate(params: &CodecParameters) -> (u32, u32) {
    // dwRate / dwScale = frames per second.
    if let Some(fr) = params.frame_rate {
        let num = fr.num.max(1) as u32;
        let den = fr.den.max(1) as u32;
        return (den, num);
    }
    (1, 25) // default 25 fps if unknown
}

fn uppercase4(s: &[u8; 4]) -> [u8; 4] {
    let mut out = *s;
    for b in out.iter_mut() {
        if b.is_ascii_lowercase() {
            *b -= 32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_mapping() {
        assert_eq!(video_codec_id(b"MJPG").as_str(), "mjpeg");
        assert_eq!(video_codec_id(b"mjpg").as_str(), "mjpeg");
        assert_eq!(video_codec_id(b"FFV1").as_str(), "ffv1");
        assert_eq!(video_codec_id(&[0, 0, 0, 0]).as_str(), "rgb24");
    }

    #[test]
    fn audio_mapping() {
        assert_eq!(audio_codec_id(0x0001).as_str(), "pcm_s16le");
        assert_eq!(audio_codec_id(0x0055).as_str(), "mp3");
    }

    #[test]
    fn unsupported_codec() {
        let p = CodecParameters::audio(CodecId::new("opus"));
        assert!(build_strf(&p).is_err());
    }
}
