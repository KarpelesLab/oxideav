//! Map between Matroska codec ID strings and oxideav [`CodecId`].
//!
//! Reference: <https://www.matroska.org/technical/codec_specs.html>

use oxideav_core::CodecId;

/// Best-effort mapping from a Matroska codec id string (e.g. `"A_FLAC"`) to
/// the oxideav codec id we use internally.
pub fn from_matroska(s: &str) -> CodecId {
    let id = match s {
        "A_FLAC" => "flac",
        "A_OPUS" => "opus",
        "A_VORBIS" => "vorbis",
        "A_PCM/INT/LIT" => "pcm_s16le",
        "A_PCM/INT/BIG" => "pcm_s16be",
        "A_PCM/FLOAT/IEEE" => "pcm_f32le",
        "A_AAC" | "A_AAC/MPEG4/LC" | "A_AAC/MPEG2/LC" => "aac",
        "A_MPEG/L3" => "mp3",
        "A_AC3" => "ac3",
        "A_EAC3" => "eac3",
        "V_VP8" => "vp8",
        "V_VP9" => "vp9",
        "V_AV1" => "av1",
        "V_MPEG4/ISO/AVC" => "h264",
        "V_MPEGH/ISO/HEVC" => "h265",
        other => return CodecId::new(format!("mkv:{other}")),
    };
    CodecId::new(id)
}

/// Inverse of `from_matroska` for codecs we support writing. Returns `None`
/// for codecs without a Matroska mapping we know.
pub fn to_matroska(id: &CodecId) -> Option<&'static str> {
    Some(match id.as_str() {
        "flac" => "A_FLAC",
        "opus" => "A_OPUS",
        "vorbis" => "A_VORBIS",
        "pcm_s16le" => "A_PCM/INT/LIT",
        "pcm_s16be" => "A_PCM/INT/BIG",
        "pcm_f32le" => "A_PCM/FLOAT/IEEE",
        "aac" => "A_AAC",
        "mp3" => "A_MPEG/L3",
        "ac3" => "A_AC3",
        "eac3" => "A_EAC3",
        "vp8" => "V_VP8",
        "vp9" => "V_VP9",
        "av1" => "V_AV1",
        "h264" => "V_MPEG4/ISO/AVC",
        "h265" => "V_MPEGH/ISO/HEVC",
        _ => return None,
    })
}
