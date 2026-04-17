//! Pure-Rust AMV decoder (container + video + audio).
//!
//! AMV is the file format used by cheap Chinese MP3/MP4 player devices:
//! a hard-coded, deliberately-broken subset of RIFF/AVI carrying a stripped
//! Motion-JPEG video stream plus an IMA-ADPCM audio variant. ffmpeg
//! demuxes/decodes it via the AVI demuxer and the MJPEG decoder; we ship
//! a self-contained crate instead.
//!
//! ## Quirks (re-derived from the bitstream + ffmpeg's `amvenc.c`,
//! `mjpegdec.c`, and `adpcm.c`):
//!
//! * **Container**: `RIFF…AMV ` top-level form. Inside `LIST hdrl` an
//!   `amvh` chunk (56 bytes) carries width/height/frame-rate. Inside
//!   `LIST movi`, frame chunks alternate `00dc` (video) / `01wb` (audio).
//!   Chunk size fields are sometimes left as zero (the spec's "deliberately
//!   broken" fields), so chunk sizes drive packet boundaries — every length
//!   field carries the actual payload size for the chunk types we care
//!   about. There is **no 2-byte chunk padding** between chunks; AMV writes
//!   chunks back-to-back regardless of payload parity.
//!
//! * **Video frames**: each `00dc` payload is just `FFD8` (SOI) + entropy
//!   data + `FFD9` (EOI). The standard JPEG header segments (DQT, DHT, SOF0,
//!   SOS) are *omitted*. The decoder reconstitutes a real baseline JPEG
//!   internally (Annex K standard quant + Huffman tables, 4:2:0 luma
//!   sampling, the width/height taken from the container) and feeds it to
//!   the standard MJPEG decoder. Output frames are then **vertically
//!   flipped** — AMV stores the picture upside-down on disk.
//!
//! * **Audio frames**: a single-channel IMA-ADPCM variant. Each `01wb`
//!   chunk is one 8-byte header (LE int16 initial predictor, u8 initial
//!   step index, 5 reserved bytes some encoders use as a sample-count
//!   hint) followed by packed nibbles producing `(payload_len - 8) * 2`
//!   samples. The nibble expansion is the standard IMA-ADPCM update with
//!   the FFmpeg shift = 3 form (`(2*delta + 1) * step) >> 3`).

pub mod audio;
pub mod demux;
pub mod video;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

/// Codec id for AMV video — the modified-MJPEG bitstream.
pub const VIDEO_CODEC_ID_STR: &str = "amv";
/// Codec id for AMV audio — IMA-ADPCM, AMV variant.
pub const AUDIO_CODEC_ID_STR: &str = "adpcm_ima_amv";

pub fn register_codecs(reg: &mut CodecRegistry) {
    let v_caps = CodecCapabilities::video("amv_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(4096, 4096);
    reg.register_both(
        CodecId::new(VIDEO_CODEC_ID_STR),
        v_caps,
        video::make_decoder,
        video::make_encoder,
    );

    let a_caps = CodecCapabilities::audio("adpcm_ima_amv_sw")
        .with_lossy(true)
        .with_max_channels(1)
        .with_max_sample_rate(48_000);
    reg.register_both(
        CodecId::new(AUDIO_CODEC_ID_STR),
        a_caps,
        audio::make_decoder,
        audio::make_encoder,
    );
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    demux::register(reg);
}

/// Combined registration helper, mirroring `oxideav-vp8::register`.
pub fn register(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}
