//! Matroska element IDs we care about.
//!
//! Reference: <https://www.matroska.org/technical/elements.html>

#![allow(dead_code)]

// EBML root + EBML metadata.
pub const EBML_HEADER: u32 = 0x1A45DFA3;
pub const EBML_VERSION: u32 = 0x4286;
pub const EBML_READ_VERSION: u32 = 0x42F7;
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
pub const EBML_DOC_TYPE: u32 = 0x4282;
pub const EBML_DOC_TYPE_VERSION: u32 = 0x4287;
pub const EBML_DOC_TYPE_READ_VERSION: u32 = 0x4285;

// Top-level Segment.
pub const SEGMENT: u32 = 0x18538067;

// Within Segment.
pub const SEEK_HEAD: u32 = 0x114D9B74;
pub const INFO: u32 = 0x1549A966;
pub const TRACKS: u32 = 0x1654AE6B;
pub const CLUSTER: u32 = 0x1F43B675;
pub const CUES: u32 = 0x1C53BB6B;
pub const ATTACHMENTS: u32 = 0x1941A469;
pub const CHAPTERS: u32 = 0x1043A770;
pub const TAGS: u32 = 0x1254C367;
pub const VOID: u32 = 0xEC;
pub const CRC32: u32 = 0xBF;

// Info.
pub const TIMECODE_SCALE: u32 = 0x2AD7B1;
pub const DURATION: u32 = 0x4489;
pub const SEGMENT_UID: u32 = 0x73A4;
pub const MUXING_APP: u32 = 0x4D80;
pub const WRITING_APP: u32 = 0x5741;
pub const TITLE: u32 = 0x7BA9;
pub const DATE_UTC: u32 = 0x4461;

// Tags (Segment\Tags\Tag\SimpleTag).
pub const TAG: u32 = 0x7373;
pub const TARGETS: u32 = 0x63C0;
pub const SIMPLE_TAG: u32 = 0x67C8;
pub const TAG_NAME: u32 = 0x45A3;
pub const TAG_STRING: u32 = 0x4487;
pub const TAG_LANGUAGE: u32 = 0x447A;

// Tracks > TrackEntry.
pub const TRACK_ENTRY: u32 = 0xAE;
pub const TRACK_NUMBER: u32 = 0xD7;
pub const TRACK_UID: u32 = 0x73C5;
pub const TRACK_TYPE: u32 = 0x83;
pub const FLAG_ENABLED: u32 = 0xB9;
pub const FLAG_DEFAULT: u32 = 0x88;
pub const FLAG_LACING: u32 = 0x9C;
pub const NAME: u32 = 0x536E;
pub const LANGUAGE: u32 = 0x22B59C;
pub const CODEC_ID: u32 = 0x86;
pub const CODEC_PRIVATE: u32 = 0x63A2;
pub const CODEC_NAME: u32 = 0x258688;
pub const CODEC_DELAY: u32 = 0x56AA;
pub const SEEK_PRE_ROLL: u32 = 0x56BB;
pub const VIDEO: u32 = 0xE0;
pub const AUDIO: u32 = 0xE1;
pub const PIXEL_WIDTH: u32 = 0xB0;
pub const PIXEL_HEIGHT: u32 = 0xBA;
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
pub const CHANNELS: u32 = 0x9F;
pub const BIT_DEPTH: u32 = 0x6264;

// Cluster.
pub const TIMECODE: u32 = 0xE7;
pub const SIMPLE_BLOCK: u32 = 0xA3;
pub const BLOCK_GROUP: u32 = 0xA0;
pub const BLOCK: u32 = 0xA1;
pub const BLOCK_DURATION: u32 = 0x9B;
pub const REFERENCE_BLOCK: u32 = 0xFB;

// Cues (seek index).
pub const CUE_POINT: u32 = 0xBB;
pub const CUE_TIME: u32 = 0xB3;
pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
pub const CUE_TRACK: u32 = 0xF7;
pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
pub const CUE_RELATIVE_POSITION: u32 = 0xF0;
pub const CUE_DURATION: u32 = 0xB2;
pub const CUE_BLOCK_NUMBER: u32 = 0x5378;

// TrackType values.
pub const TRACK_TYPE_VIDEO: u64 = 1;
pub const TRACK_TYPE_AUDIO: u64 = 2;
pub const TRACK_TYPE_SUBTITLE: u64 = 17;
