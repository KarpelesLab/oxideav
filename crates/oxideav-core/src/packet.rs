//! Compressed-data packet passed between demuxer → decoder and encoder → muxer.

use crate::time::TimeBase;

/// Metadata flags on a packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacketFlags {
    /// Packet is (or starts) a keyframe / random-access point.
    pub keyframe: bool,
    /// Packet holds codec-level headers rather than media data.
    pub header: bool,
    /// Packet's data may be corrupt but decode should still be attempted.
    pub corrupt: bool,
    /// Packet should be discarded (e.g., decoder delay padding).
    pub discard: bool,
    /// Packet is the last in its source container's natural framing unit
    /// (Ogg page, MP4 chunk, MKV cluster, …). Container muxers may use this
    /// signal to recreate similar boundaries in their output. Decoders
    /// should ignore it.
    pub unit_boundary: bool,
}

/// A chunk of compressed (encoded) data belonging to one stream.
#[derive(Clone, Debug)]
pub struct Packet {
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Time base in which `pts` and `dts` are expressed.
    pub time_base: TimeBase,
    /// Presentation timestamp (display order). `None` if unknown.
    pub pts: Option<i64>,
    /// Decode timestamp (decode order). Often equal to `pts` for intra-only codecs.
    pub dts: Option<i64>,
    /// Packet duration in `time_base` units, or `None` if unknown.
    pub duration: Option<i64>,
    /// Flags describing this packet.
    pub flags: PacketFlags,
    /// Compressed payload.
    pub data: Vec<u8>,
}

impl Packet {
    pub fn new(stream_index: u32, time_base: TimeBase, data: Vec<u8>) -> Self {
        Self {
            stream_index,
            time_base,
            pts: None,
            dts: None,
            duration: None,
            flags: PacketFlags::default(),
            data,
        }
    }

    pub fn with_pts(mut self, pts: i64) -> Self {
        self.pts = Some(pts);
        self
    }

    pub fn with_dts(mut self, dts: i64) -> Self {
        self.dts = Some(dts);
        self
    }

    pub fn with_duration(mut self, d: i64) -> Self {
        self.duration = Some(d);
        self
    }

    pub fn with_keyframe(mut self, kf: bool) -> Self {
        self.flags.keyframe = kf;
        self
    }
}
