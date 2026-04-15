//! Sniff the codec carried by a logical Ogg bitstream from its first packet.
//!
//! Ogg is codec-agnostic: the first packet of every logical stream is a
//! codec-specific identification header that begins with a recognisable
//! signature. We use that to set [`CodecId`] in the demuxer's
//! [`StreamInfo`] without depending on any per-codec crate.

use oxideav_core::CodecId;

/// Identify the codec of a logical Ogg bitstream from its first packet.
pub fn detect(first_packet: &[u8]) -> CodecId {
    // Vorbis I, RFC 5215 §2.1: packet type 0x01, then "vorbis".
    if first_packet.len() >= 7
        && first_packet[0] == 0x01
        && &first_packet[1..7] == b"vorbis"
    {
        return CodecId::new("vorbis");
    }
    // Opus, RFC 7845 §5.1: "OpusHead".
    if first_packet.len() >= 8 && &first_packet[0..8] == b"OpusHead" {
        return CodecId::new("opus");
    }
    // FLAC-in-Ogg, https://xiph.org/flac/ogg_mapping.html: 0x7F + "FLAC".
    if first_packet.len() >= 5
        && first_packet[0] == 0x7F
        && &first_packet[1..5] == b"FLAC"
    {
        return CodecId::new("flac");
    }
    // Theora: 0x80 + "theora".
    if first_packet.len() >= 7
        && first_packet[0] == 0x80
        && &first_packet[1..7] == b"theora"
    {
        return CodecId::new("theora");
    }
    // Speex: "Speex   " (8 bytes including trailing spaces).
    if first_packet.len() >= 8 && &first_packet[0..8] == b"Speex   " {
        return CodecId::new("speex");
    }
    CodecId::new("unknown")
}

/// Number of header packets a codec expects before audio/video data.
///
/// Ogg streams typically begin with one or more setup packets that don't carry
/// timestamps. The demuxer skips past them when reporting packet PTS.
pub fn header_packet_count(id: &CodecId) -> usize {
    match id.as_str() {
        // Vorbis: identification, comment, setup.
        "vorbis" => 3,
        // Opus: head, tags.
        "opus" => 2,
        // FLAC-in-Ogg: 1 mapping packet + every metadata block (≥1 STREAMINFO).
        // We treat the mapping packet as the only "header" packet — STREAMINFO
        // and other metadata are also packets but each carries its own framing.
        // Conservative default: 1.
        "flac" => 1,
        // Theora and Speex have 3 and 2 header packets respectively.
        "theora" => 3,
        "speex" => 2,
        _ => 0,
    }
}
