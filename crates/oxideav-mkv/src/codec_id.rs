//! Map between Matroska codec ID strings and oxideav [`CodecId`].
//!
//! Reference: <https://www.matroska.org/technical/codec_specs.html>

use oxideav_core::CodecId;

/// Matroska CodecID strings that WebM permits.
///
/// WebM is a restricted subset of Matroska: the WebM specification
/// (<https://www.webmproject.org/docs/container/>) only allows VP8, VP9 and
/// AV1 for video, and Vorbis and Opus for audio. Anything else must be
/// rejected by a WebM muxer — writing e.g. H.264 into a DocType="webm"
/// file produces an invalid WebM even though the container bytes are valid
/// Matroska.
pub const ALLOWED_WEBM_CODECS: &[&str] = &[
    // Video.
    "V_VP8", "V_VP9", "V_AV1", // Audio.
    "A_VORBIS", "A_OPUS",
];

/// Return true if `matroska_codec_id` (e.g. `"A_OPUS"`) is permitted inside
/// a WebM container.
pub fn is_webm_matroska_codec(matroska_codec_id: &str) -> bool {
    ALLOWED_WEBM_CODECS.contains(&matroska_codec_id)
}

/// Return true if the oxideav-internal [`CodecId`] corresponds to a codec
/// that WebM permits. Unknown / unmapped codec ids return false.
pub fn is_webm_codec(id: &CodecId) -> bool {
    matches!(id.as_str(), "vp8" | "vp9" | "av1" | "vorbis" | "opus")
}

/// Best-effort mapping from a Matroska codec id string (e.g. `"A_FLAC"`) to
/// the oxideav codec id we use internally.
///
/// `codec_private` is consulted for `V_MS/VFW/FOURCC` tracks because the
/// BITMAPINFOHEADER's `biCompression` field carries the actual codec. For
/// other codec ids it is ignored.
pub fn from_matroska(s: &str, codec_private: &[u8]) -> CodecId {
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
        "V_FFV1" => "ffv1",
        "V_THEORA" => "theora",
        "V_MS/VFW/FOURCC" => return from_bitmapinfoheader(codec_private),
        other => return CodecId::new(format!("mkv:{other}")),
    };
    CodecId::new(id)
}

/// Extract the codec id from a BITMAPINFOHEADER `CodecPrivate` blob. The
/// fourcc lives at bytes 16..20 (biCompression). Unrecognised fourcc falls
/// back to `mkv:BI/<fourcc>`.
fn from_bitmapinfoheader(cp: &[u8]) -> CodecId {
    if cp.len() < 20 {
        return CodecId::new("mkv:BI/<truncated>");
    }
    let fourcc = &cp[16..20];
    let fourcc_str = std::str::from_utf8(fourcc).unwrap_or("????");
    match fourcc_str {
        "FFV1" => CodecId::new("ffv1"),
        other => CodecId::new(format!("mkv:BI/{other}")),
    }
}

/// If `codec_private` is a BITMAPINFOHEADER, return the inner codec-specific
/// extradata (everything after the 40-byte header). Otherwise returns the
/// slice unchanged.
pub fn strip_bitmapinfoheader(codec_id: &str, codec_private: &[u8]) -> Vec<u8> {
    if codec_id == "V_MS/VFW/FOURCC" && codec_private.len() >= 40 {
        codec_private[40..].to_vec()
    } else {
        codec_private.to_vec()
    }
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
        "ffv1" => "V_FFV1",
        _ => return None,
    })
}
