//! Map between MP4 sample-entry FourCCs and oxideav codec IDs.

use oxideav_core::CodecId;

pub fn from_sample_entry(fourcc: &[u8; 4]) -> CodecId {
    let id = match fourcc {
        b"mp4a" => "aac",
        b"alac" => "alac",
        b"fLaC" | b"flac" => "flac",
        b"Opus" | b"opus" => "opus",
        b"avc1" | b"avc3" => "h264",
        b"hvc1" | b"hev1" => "h265",
        b"vp08" => "vp8",
        b"vp09" => "vp9",
        b"av01" => "av1",
        b"jpeg" | b"mjpa" | b"mjpb" => "mjpeg",
        // MP4 sample entry `mp4v` is carried for both MPEG-1 video (OTI 0x6A)
        // and MPEG-4 Part 2 / ASP (OTI 0x20). Part 2 is overwhelmingly more
        // common in MP4, so default to `mpeg4video` here. A finer mapping
        // based on the ESDS `object_type_indication` belongs at a higher
        // level — this is a best-effort shortcut.
        // TODO: wire OTI-based dispatch so OTI=0x6A resolves back to mpeg1video.
        b"mp4v" => "mpeg4video",
        b"lpcm" | b"sowt" | b"twos" => "pcm_s16le",
        other => {
            let s = std::str::from_utf8(other).unwrap_or("????");
            return CodecId::new(format!("mp4:{s}"));
        }
    };
    CodecId::new(id)
}
