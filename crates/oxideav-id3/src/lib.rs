// Per-frame parsing is a long dispatch; clippy prefers short fns but
// breaking this up only obfuscates the spec reference.
#![allow(clippy::needless_range_loop)]

//! ID3v1 and ID3v2 (2.2 / 2.3 / 2.4) tag parser.
//!
//! This crate is a *parser* — it consumes a byte buffer and exposes
//! structured metadata. It never writes tags back.
//!
//! The public surface is small:
//!
//! * [`parse_tag`] — take a `&[u8]` that starts with the 10-byte ID3v2
//!   header and return an [`Id3Tag`] plus the number of bytes consumed
//!   (so callers can seek past the tag and resume normal file reads).
//! * [`parse_id3v1`] — take the last 128 bytes of a file and, if they
//!   start with `TAG`, return the v1 tag.
//! * [`tag_size_at_head`] — peek at the first 10 bytes to work out the
//!   total on-disk tag size without parsing frames.
//! * [`to_key_value_pairs`] — normalise an [`Id3Tag`] into the
//!   Vorbis-comment-style `(key, value)` pairs the rest of the workspace
//!   uses (`title`, `artist`, `album`, `date`, ...).
//! * [`attached_pictures`] — pull the `APIC` / `PIC` frames out of a tag.
//!
//! Unsynchronisation (`0xFF 0x00` → `0xFF`) is reversed at the right
//! level for each version (whole-tag in 2.2/2.3, per-frame in 2.4), and
//! the v2.4 data-length indicator is honoured so tools that set it see
//! their real payload length.
//!
//! Frames this parser knows about structurally:
//!
//! * `T***` text frames (v2.3/2.4) and their v2.2 equivalents (3-char ids).
//! * `TXXX` user-defined text.
//! * `W***` URL frames and `WXXX` user-defined URL.
//! * `COMM` comments and `USLT` lyrics.
//! * `APIC` attached pictures (v2.3/2.4) and `PIC` (v2.2).
//!
//! Everything else lands in [`Id3Frame::Unknown`] with the raw payload
//! preserved so future code can extend recognition without reparsing.

use oxideav_core::{AttachedPicture, Error, PictureType, Result};

pub const ID3V2_HEADER_SIZE: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Id3Version {
    V2_2,
    V2_3,
    V2_4,
    V1,
}

#[derive(Clone, Debug)]
pub struct Id3Tag {
    pub version: Id3Version,
    pub frames: Vec<Id3Frame>,
}

#[derive(Clone, Debug)]
pub enum Id3Frame {
    /// Text-information frame (`T***` except `TXXX`), already decoded
    /// from whatever encoding was declared. Multi-value frames (v2.4
    /// uses NUL as the separator) split into several entries.
    Text { id: String, values: Vec<String> },
    /// `COMM` comment frame.
    Comment {
        lang: [u8; 3],
        description: String,
        text: String,
    },
    /// `USLT` unsynchronised lyrics frame — same on-the-wire shape as
    /// `COMM` so we use the same variant data.
    Lyrics {
        lang: [u8; 3],
        description: String,
        text: String,
    },
    /// `TXXX` user-defined text.
    UserText { description: String, value: String },
    /// `WXXX` user-defined URL.
    UserUrl { description: String, url: String },
    /// Non-user `W***` URL frame (`WCOM`, `WOAF`, ...). The URL is
    /// always ISO-8859-1 per spec.
    Url { id: String, url: String },
    /// `APIC` (v2.3/2.4) or `PIC` (v2.2) attached picture.
    Picture(AttachedPicture),
    /// Any frame whose id we don't parse structurally (SYLT, RGAD,
    /// PRIV, ...). The payload is preserved verbatim so callers or
    /// later versions can recognise it without needing to reparse.
    Unknown { id: String, raw: Vec<u8> },
}

/// Parse an ID3v2 tag from a buffer that starts with the 10-byte
/// header. On success, returns the [`Id3Tag`] and the total number of
/// bytes consumed from `buf` (header + body + optional footer) —
/// callers can seek by that many bytes to reach the next byte after
/// the tag.
pub fn parse_tag(buf: &[u8]) -> Result<(Id3Tag, usize)> {
    if buf.len() < ID3V2_HEADER_SIZE {
        return Err(Error::NeedMore);
    }
    if &buf[0..3] != b"ID3" {
        return Err(Error::invalid("not an ID3v2 tag"));
    }
    let major = buf[3];
    let _revision = buf[4];
    let flags = buf[5];
    let size = synchsafe_u32(buf[6], buf[7], buf[8], buf[9]) as usize;
    let total = ID3V2_HEADER_SIZE + size + if flags & 0x10 != 0 { 10 } else { 0 };
    if buf.len() < ID3V2_HEADER_SIZE + size {
        return Err(Error::NeedMore);
    }
    let body = &buf[ID3V2_HEADER_SIZE..ID3V2_HEADER_SIZE + size];
    let version = match major {
        2 => Id3Version::V2_2,
        3 => Id3Version::V2_3,
        4 => Id3Version::V2_4,
        other => {
            return Err(Error::unsupported(format!(
                "unknown ID3v2 major version {other}"
            )));
        }
    };

    // Whole-tag unsync is a v2.2/v2.3 mechanism. v2.4 moves it to a
    // per-frame flag, but some taggers still set the header bit; we
    // honour whichever is present.
    let unsync_whole_tag =
        (flags & 0x80) != 0 && matches!(version, Id3Version::V2_2 | Id3Version::V2_3);
    // v2.4 also has a whole-tag unsync flag but it's strictly
    // "advisory" — the spec says the tag *may* be unsynchronised, and
    // per-frame flags are authoritative. We still decode the whole
    // body when the flag is set in v2.4 so older or strict taggers
    // work; per-frame unsync on an already-reversed buffer is a no-op.
    let unsync_v24_body = (flags & 0x80) != 0 && matches!(version, Id3Version::V2_4);

    let body_owned;
    let mut body: &[u8] = if unsync_whole_tag || unsync_v24_body {
        body_owned = reverse_unsync(body);
        &body_owned
    } else {
        body
    };

    // Extended header: 6 bytes in v2.3 (size is non-synchsafe), 6+
    // bytes in v2.4 (first 4 bytes are synchsafe size INCLUDING those
    // 4 bytes). We just skip it — none of the fields affect frame
    // parsing for our purposes.
    if flags & 0x40 != 0 {
        body = skip_extended_header(version, body)?;
    }

    let frames = parse_frames(version, body);
    Ok((Id3Tag { version, frames }, total))
}

/// Peek at the first 10 bytes of a file. Returns `Some(total_size)` —
/// header + body + optional footer — when a valid ID3v2 tag starts
/// there, or `None` otherwise. Callers use this to size a read without
/// parsing frames yet.
pub fn tag_size_at_head(first10: &[u8]) -> Option<usize> {
    if first10.len() < 10 || &first10[0..3] != b"ID3" {
        return None;
    }
    let flags = first10[5];
    let size = synchsafe_u32(first10[6], first10[7], first10[8], first10[9]) as usize;
    let footer = if flags & 0x10 != 0 { 10 } else { 0 };
    Some(ID3V2_HEADER_SIZE + size + footer)
}

/// Parse an ID3v1 trailer. Returns `None` when the buffer doesn't
/// start with `TAG` or is shorter than 128 bytes.
pub fn parse_id3v1(trailer_128: &[u8]) -> Option<Id3Tag> {
    if trailer_128.len() < 128 || &trailer_128[0..3] != b"TAG" {
        return None;
    }
    let title = v1_string(&trailer_128[3..33]);
    let artist = v1_string(&trailer_128[33..63]);
    let album = v1_string(&trailer_128[63..93]);
    let year = v1_string(&trailer_128[93..97]);
    // ID3v1.1: if byte 125 is NUL and byte 126 is non-zero, the last
    // 2 bytes are a track number; otherwise the full 30 bytes are a
    // free-form comment.
    let (comment, track) = if trailer_128[125] == 0 && trailer_128[126] != 0 {
        (v1_string(&trailer_128[97..125]), Some(trailer_128[126]))
    } else {
        (v1_string(&trailer_128[97..127]), None)
    };
    let genre_byte = trailer_128[127];
    let genre = id3v1_genre(genre_byte).map(|s| s.to_string());

    let mut frames = Vec::new();
    if !title.is_empty() {
        frames.push(Id3Frame::Text {
            id: "TIT2".into(),
            values: vec![title],
        });
    }
    if !artist.is_empty() {
        frames.push(Id3Frame::Text {
            id: "TPE1".into(),
            values: vec![artist],
        });
    }
    if !album.is_empty() {
        frames.push(Id3Frame::Text {
            id: "TALB".into(),
            values: vec![album],
        });
    }
    if !year.is_empty() {
        frames.push(Id3Frame::Text {
            id: "TYER".into(),
            values: vec![year],
        });
    }
    if !comment.is_empty() {
        frames.push(Id3Frame::Comment {
            lang: *b"XXX",
            description: String::new(),
            text: comment,
        });
    }
    if let Some(t) = track {
        frames.push(Id3Frame::Text {
            id: "TRCK".into(),
            values: vec![t.to_string()],
        });
    }
    if let Some(g) = genre {
        frames.push(Id3Frame::Text {
            id: "TCON".into(),
            values: vec![g],
        });
    }

    Some(Id3Tag {
        version: Id3Version::V1,
        frames,
    })
}

/// Normalise an [`Id3Tag`] into flat `(key, value)` pairs using the
/// lowercase Vorbis-comment keys the rest of the workspace expects.
/// Known v2.3/v2.4 four-char ids map to their Vorbis equivalents;
/// v2.2 three-char ids map via the v2.2→v2.3 promotion table; unknown
/// ids pass through with their raw id lowercased.
pub fn to_key_value_pairs(tag: &Id3Tag) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for f in &tag.frames {
        match f {
            Id3Frame::Text { id, values } => {
                let key = text_frame_to_key(id);
                let value = values.join("/");
                if !value.is_empty() {
                    push_unique(&mut out, key, value);
                }
            }
            Id3Frame::Comment {
                description, text, ..
            } => {
                if !text.is_empty() {
                    let key = if description.is_empty() {
                        "comment".to_string()
                    } else {
                        format!("comment:{}", description.to_lowercase())
                    };
                    push_unique(&mut out, key, text.clone());
                }
            }
            Id3Frame::Lyrics {
                description, text, ..
            } => {
                if !text.is_empty() {
                    let key = if description.is_empty() {
                        "lyrics".to_string()
                    } else {
                        format!("lyrics:{}", description.to_lowercase())
                    };
                    push_unique(&mut out, key, text.clone());
                }
            }
            Id3Frame::UserText { description, value } => {
                if !value.is_empty() {
                    let key = if description.is_empty() {
                        "user_text".to_string()
                    } else {
                        description.to_lowercase()
                    };
                    push_unique(&mut out, key, value.clone());
                }
            }
            Id3Frame::UserUrl { description, url } => {
                if !url.is_empty() {
                    let key = if description.is_empty() {
                        "user_url".to_string()
                    } else {
                        format!("url:{}", description.to_lowercase())
                    };
                    push_unique(&mut out, key, url.clone());
                }
            }
            Id3Frame::Url { id, url } => {
                if !url.is_empty() {
                    push_unique(&mut out, format!("url:{}", id.to_lowercase()), url.clone());
                }
            }
            // Pictures are surfaced via attached_pictures(), not k/v.
            Id3Frame::Picture(_) => {}
            Id3Frame::Unknown { .. } => {}
        }
    }
    out
}

/// Extract only the attached pictures from a tag, cloned out as a
/// convenient Vec for callers that don't want to match on the enum.
pub fn attached_pictures(tag: &Id3Tag) -> Vec<AttachedPicture> {
    tag.frames
        .iter()
        .filter_map(|f| match f {
            Id3Frame::Picture(p) => Some(p.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn synchsafe_u32(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((a as u32 & 0x7F) << 21)
        | ((b as u32 & 0x7F) << 14)
        | ((c as u32 & 0x7F) << 7)
        | (d as u32 & 0x7F)
}

fn regular_u32(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((a as u32) << 24) | ((b as u32) << 16) | ((c as u32) << 8) | (d as u32)
}

fn regular_u24(a: u8, b: u8, c: u8) -> u32 {
    ((a as u32) << 16) | ((b as u32) << 8) | (c as u32)
}

/// Reverse the ID3 unsynchronisation encoding: every `0xFF 0x00`
/// sequence collapses back to a bare `0xFF`. Other bytes pass through
/// verbatim. This is a byte-for-byte, stream-safe operation.
fn reverse_unsync(buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len());
    let mut i = 0;
    while i < buf.len() {
        out.push(buf[i]);
        if buf[i] == 0xFF && i + 1 < buf.len() && buf[i + 1] == 0x00 {
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

fn skip_extended_header(version: Id3Version, body: &[u8]) -> Result<&[u8]> {
    match version {
        Id3Version::V2_3 => {
            if body.len() < 4 {
                return Err(Error::invalid("ID3v2.3 extended header truncated"));
            }
            let ext_size = regular_u32(body[0], body[1], body[2], body[3]) as usize;
            // v2.3: ext_size does NOT include itself, so skip 4 + ext_size.
            let total = 4 + ext_size;
            if total > body.len() {
                return Err(Error::invalid("ID3v2.3 extended header overflows tag"));
            }
            Ok(&body[total..])
        }
        Id3Version::V2_4 => {
            if body.len() < 4 {
                return Err(Error::invalid("ID3v2.4 extended header truncated"));
            }
            let ext_size = synchsafe_u32(body[0], body[1], body[2], body[3]) as usize;
            // v2.4: ext_size INCLUDES itself, so skip ext_size bytes total.
            if ext_size < 4 || ext_size > body.len() {
                return Err(Error::invalid("ID3v2.4 extended header size invalid"));
            }
            Ok(&body[ext_size..])
        }
        _ => Ok(body),
    }
}

fn parse_frames(version: Id3Version, body: &[u8]) -> Vec<Id3Frame> {
    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < body.len() {
        // A byte of 0x00 at the start of a frame id is the padding
        // sentinel — everything from here to end of body is zeros.
        if body[i] == 0 {
            break;
        }
        match parse_one_frame(version, &body[i..]) {
            Ok((frame, consumed)) => {
                frames.push(frame);
                i += consumed;
            }
            Err(_) => {
                // Give up on further frames but keep what we parsed. A
                // single corrupted frame is common in real-world files
                // (truncated tags, buggy taggers) and shouldn't nuke
                // the whole tag.
                break;
            }
        }
    }
    frames
}

fn parse_one_frame(version: Id3Version, buf: &[u8]) -> Result<(Id3Frame, usize)> {
    match version {
        Id3Version::V2_2 => parse_v22_frame(buf),
        Id3Version::V2_3 => parse_v23_frame(buf),
        Id3Version::V2_4 => parse_v24_frame(buf),
        Id3Version::V1 => Err(Error::invalid("parse_one_frame called on v1")),
    }
}

fn parse_v22_frame(buf: &[u8]) -> Result<(Id3Frame, usize)> {
    if buf.len() < 6 {
        return Err(Error::invalid("v2.2 frame header truncated"));
    }
    let id = std::str::from_utf8(&buf[0..3])
        .map_err(|_| Error::invalid("v2.2 frame id not ASCII"))?
        .to_string();
    let size = regular_u24(buf[3], buf[4], buf[5]) as usize;
    if 6 + size > buf.len() {
        return Err(Error::invalid("v2.2 frame overflows tag body"));
    }
    let payload = &buf[6..6 + size];
    let frame = dispatch_v22(&id, payload);
    Ok((frame, 6 + size))
}

fn parse_v23_frame(buf: &[u8]) -> Result<(Id3Frame, usize)> {
    if buf.len() < 10 {
        return Err(Error::invalid("v2.3 frame header truncated"));
    }
    let id = std::str::from_utf8(&buf[0..4])
        .map_err(|_| Error::invalid("v2.3 frame id not ASCII"))?
        .to_string();
    let size = regular_u32(buf[4], buf[5], buf[6], buf[7]) as usize;
    let _flags = u16::from_be_bytes([buf[8], buf[9]]);
    if 10 + size > buf.len() {
        return Err(Error::invalid("v2.3 frame overflows tag body"));
    }
    let payload = &buf[10..10 + size];
    let frame = dispatch_v23_v24(&id, payload);
    Ok((frame, 10 + size))
}

fn parse_v24_frame(buf: &[u8]) -> Result<(Id3Frame, usize)> {
    if buf.len() < 10 {
        return Err(Error::invalid("v2.4 frame header truncated"));
    }
    let id = std::str::from_utf8(&buf[0..4])
        .map_err(|_| Error::invalid("v2.4 frame id not ASCII"))?
        .to_string();
    let size = synchsafe_u32(buf[4], buf[5], buf[6], buf[7]) as usize;
    let flags = u16::from_be_bytes([buf[8], buf[9]]);
    if 10 + size > buf.len() {
        return Err(Error::invalid("v2.4 frame overflows tag body"));
    }
    // Format flags (low byte): bit 0x01 = data-length indicator,
    // bit 0x02 = unsync, bit 0x04 = encryption, bit 0x08 =
    // compression, bit 0x40 = grouping identity.
    let fmt_flags = (flags & 0xFF) as u8;
    let data_length_indicator = fmt_flags & 0x01 != 0;
    let frame_unsync = fmt_flags & 0x02 != 0;
    let encrypted = fmt_flags & 0x04 != 0;
    let compressed = fmt_flags & 0x08 != 0;
    let grouping = fmt_flags & 0x40 != 0;

    let mut payload = &buf[10..10 + size];
    if grouping && !payload.is_empty() {
        payload = &payload[1..];
    }
    if encrypted || compressed {
        // We don't carry keys/zlib, so just emit an Unknown frame so
        // callers can see it was present.
        return Ok((
            Id3Frame::Unknown {
                id,
                raw: payload.to_vec(),
            },
            10 + size,
        ));
    }
    // The data-length indicator is 4 synchsafe bytes giving the real
    // (post-decompression, post-unsync) size. We don't decompress so
    // we just skip past the indicator.
    if data_length_indicator {
        if payload.len() < 4 {
            return Err(Error::invalid("v2.4 frame data-length indicator truncated"));
        }
        payload = &payload[4..];
    }
    let unsync_owned;
    if frame_unsync {
        unsync_owned = reverse_unsync(payload);
        payload = &unsync_owned;
        // Rust can't see the borrow across the `unsync_owned` binding
        // without an extra let, so give it one.
        let _ = &unsync_owned;
    }
    let frame = dispatch_v23_v24(&id, payload);
    Ok((frame, 10 + size))
}

/// Dispatch a v2.3/v2.4 frame payload to the right parser based on
/// its 4-char id.
fn dispatch_v23_v24(id: &str, payload: &[u8]) -> Id3Frame {
    if id == "TXXX" {
        return parse_txxx(id, payload);
    }
    if id.starts_with('T') && id != "TXXX" {
        return parse_text_frame(id, payload);
    }
    if id == "WXXX" {
        return parse_wxxx(id, payload);
    }
    if id.starts_with('W') && id != "WXXX" {
        return parse_url_frame(id, payload);
    }
    match id {
        "COMM" => parse_comm_like(payload, false),
        "USLT" => parse_comm_like(payload, true),
        "APIC" => parse_apic(payload),
        _ => Id3Frame::Unknown {
            id: id.to_string(),
            raw: payload.to_vec(),
        },
    }
}

/// Dispatch a v2.2 (3-char id) frame payload. v2.2 ids are promoted
/// to their v2.3 four-char equivalents for caller-facing output so
/// `to_key_value_pairs` doesn't need to know about both.
fn dispatch_v22(id: &str, payload: &[u8]) -> Id3Frame {
    // Text frames — v2.2 uses 3-char ids that promote cleanly. We use
    // the dominant v2.3 equivalent.
    let promoted = v22_promote(id);
    if id == "TXX" {
        return parse_txxx(promoted, payload);
    }
    if id.starts_with('T') {
        return parse_text_frame(promoted, payload);
    }
    if id == "WXX" {
        return parse_wxxx(promoted, payload);
    }
    if id.starts_with('W') {
        return parse_url_frame(promoted, payload);
    }
    match id {
        "COM" => parse_comm_like(payload, false),
        "ULT" => parse_comm_like(payload, true),
        "PIC" => parse_pic(payload),
        _ => Id3Frame::Unknown {
            id: id.to_string(),
            raw: payload.to_vec(),
        },
    }
}

/// Promote v2.2 3-char ids to their v2.3 4-char equivalents. Entries
/// follow the ID3v2.2 → v2.3 conversion table. Unknown ids pass
/// through unchanged (they land in `Unknown` anyway).
fn v22_promote(id: &str) -> &str {
    match id {
        "TT1" => "TIT1",
        "TT2" => "TIT2",
        "TT3" => "TIT3",
        "TP1" => "TPE1",
        "TP2" => "TPE2",
        "TP3" => "TPE3",
        "TP4" => "TPE4",
        "TCM" => "TCOM",
        "TXT" => "TEXT",
        "TLA" => "TLAN",
        "TCO" => "TCON",
        "TAL" => "TALB",
        "TPA" => "TPOS",
        "TRK" => "TRCK",
        "TRC" => "TSRC",
        "TYE" => "TYER",
        "TDA" => "TDAT",
        "TIM" => "TIME",
        "TRD" => "TRDA",
        "TMT" => "TMED",
        "TFT" => "TFLT",
        "TBP" => "TBPM",
        "TCP" => "TCMP",
        "TCR" => "TCOP",
        "TPB" => "TPUB",
        "TEN" => "TENC",
        "TSS" => "TSSE",
        "TOF" => "TOFN",
        "TLE" => "TLEN",
        "TSI" => "TSIZ",
        "TDY" => "TDLY",
        "TKE" => "TKEY",
        "TOT" => "TOAL",
        "TOA" => "TOPE",
        "TOL" => "TOLY",
        "TOR" => "TORY",
        "TXX" => "TXXX",
        "WAF" => "WOAF",
        "WAR" => "WOAR",
        "WAS" => "WOAS",
        "WCM" => "WCOM",
        "WCP" => "WCOP",
        "WPB" => "WPUB",
        "WXX" => "WXXX",
        other => other,
    }
}

fn parse_text_frame(id: &str, payload: &[u8]) -> Id3Frame {
    if payload.is_empty() {
        return Id3Frame::Text {
            id: id.to_string(),
            values: Vec::new(),
        };
    }
    let enc = payload[0];
    let text = decode_text(enc, &payload[1..]);
    // v2.4 splits multi-value text frames on NUL; v2.2/v2.3 use '/'.
    // We split on NUL unconditionally; v2.2/v2.3 frames almost never
    // have embedded NULs in practice so this is safe.
    let values: Vec<String> = text
        .split('\u{0}')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Id3Frame::Text {
        id: id.to_string(),
        values,
    }
}

fn parse_txxx(id: &str, payload: &[u8]) -> Id3Frame {
    if payload.is_empty() {
        return Id3Frame::UserText {
            description: String::new(),
            value: String::new(),
        };
    }
    let enc = payload[0];
    let rest = &payload[1..];
    let (description, after) = split_once_nul(enc, rest);
    let value = decode_text(enc, after);
    // TXXX frames with an empty description may occur; keep id for
    // Unknown fallback if caller wants it.
    let _ = id;
    Id3Frame::UserText { description, value }
}

fn parse_wxxx(id: &str, payload: &[u8]) -> Id3Frame {
    if payload.is_empty() {
        return Id3Frame::UserUrl {
            description: String::new(),
            url: String::new(),
        };
    }
    let enc = payload[0];
    let rest = &payload[1..];
    let (description, after) = split_once_nul(enc, rest);
    // The URL itself is always ISO-8859-1 per spec.
    let url = latin1_trim(after);
    let _ = id;
    Id3Frame::UserUrl { description, url }
}

fn parse_url_frame(id: &str, payload: &[u8]) -> Id3Frame {
    // W*** URL frames: no encoding byte, payload is ISO-8859-1.
    let url = latin1_trim(payload);
    Id3Frame::Url {
        id: id.to_string(),
        url,
    }
}

fn parse_comm_like(payload: &[u8], lyrics: bool) -> Id3Frame {
    if payload.len() < 4 {
        let (d, t) = (String::new(), String::new());
        return if lyrics {
            Id3Frame::Lyrics {
                lang: [0; 3],
                description: d,
                text: t,
            }
        } else {
            Id3Frame::Comment {
                lang: [0; 3],
                description: d,
                text: t,
            }
        };
    }
    let enc = payload[0];
    let mut lang = [0u8; 3];
    lang.copy_from_slice(&payload[1..4]);
    let rest = &payload[4..];
    let (description, after) = split_once_nul(enc, rest);
    let text = decode_text(enc, after);
    if lyrics {
        Id3Frame::Lyrics {
            lang,
            description,
            text,
        }
    } else {
        Id3Frame::Comment {
            lang,
            description,
            text,
        }
    }
}

fn parse_apic(payload: &[u8]) -> Id3Frame {
    if payload.is_empty() {
        return Id3Frame::Unknown {
            id: "APIC".into(),
            raw: Vec::new(),
        };
    }
    let enc = payload[0];
    let rest = &payload[1..];
    // MIME type is null-terminated ISO-8859-1.
    let (mime_bytes, after_mime) = split_once_nul_bytes(rest);
    let mime_type = latin1_trim(mime_bytes);
    if after_mime.is_empty() {
        return Id3Frame::Unknown {
            id: "APIC".into(),
            raw: payload.to_vec(),
        };
    }
    let picture_type = PictureType::from_u8(after_mime[0]);
    let after_type = &after_mime[1..];
    let (description, data) = split_once_nul(enc, after_type);
    Id3Frame::Picture(AttachedPicture {
        mime_type,
        picture_type,
        description,
        data: data.to_vec(),
    })
}

fn parse_pic(payload: &[u8]) -> Id3Frame {
    // v2.2 PIC layout: 1 encoding byte + 3 image-format bytes (JPG /
    // PNG / ...) + 1 picture-type byte + description (NUL-term'd in
    // declared encoding) + binary data.
    if payload.len() < 5 {
        return Id3Frame::Unknown {
            id: "PIC".into(),
            raw: payload.to_vec(),
        };
    }
    let enc = payload[0];
    let fmt = &payload[1..4];
    let picture_type = PictureType::from_u8(payload[4]);
    let after = &payload[5..];
    let (description, data) = split_once_nul(enc, after);
    // Turn the 3-char image format into a MIME type so the rest of
    // the workspace doesn't have to special-case v2.2.
    let mime_type = match fmt {
        b"JPG" | b"jpg" => "image/jpeg".to_string(),
        b"PNG" | b"png" => "image/png".to_string(),
        other => format!(
            "image/{}",
            std::str::from_utf8(other)
                .unwrap_or("")
                .to_ascii_lowercase()
        ),
    };
    Id3Frame::Picture(AttachedPicture {
        mime_type,
        picture_type,
        description,
        data: data.to_vec(),
    })
}

/// Split `buf` on the first terminator for the given encoding,
/// returning (decoded_prefix, remainder_after_terminator). For 1-byte
/// encodings the terminator is `0x00`; for UTF-16 variants it is
/// `0x00 0x00` aligned on even offsets.
fn split_once_nul(enc: u8, buf: &[u8]) -> (String, &[u8]) {
    if enc == 1 || enc == 2 {
        // UTF-16: find a 2-byte NUL on an even boundary.
        let mut i = 0;
        while i + 1 < buf.len() {
            if buf[i] == 0 && buf[i + 1] == 0 {
                let prefix = decode_text(enc, &buf[..i]);
                return (prefix, &buf[i + 2..]);
            }
            i += 2;
        }
        (decode_text(enc, buf), &[])
    } else if let Some(pos) = buf.iter().position(|&b| b == 0) {
        let prefix = decode_text(enc, &buf[..pos]);
        (prefix, &buf[pos + 1..])
    } else {
        (decode_text(enc, buf), &[])
    }
}

/// Raw-bytes variant of [`split_once_nul`] that doesn't interpret the
/// declared encoding — used for MIME type fields which are always
/// ISO-8859-1 regardless of the frame-level encoding byte.
fn split_once_nul_bytes(buf: &[u8]) -> (&[u8], &[u8]) {
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        (&buf[..pos], &buf[pos + 1..])
    } else {
        (buf, &[])
    }
}

fn decode_text(enc: u8, buf: &[u8]) -> String {
    let s = match enc {
        0 => latin1_to_string(buf),
        1 => decode_utf16_bom(buf),
        2 => decode_utf16_be(buf),
        3 => String::from_utf8_lossy(buf).to_string(),
        _ => latin1_to_string(buf),
    };
    // Trim trailing NULs — many taggers pad strings with them.
    s.trim_end_matches('\u{0}').to_string()
}

fn latin1_to_string(buf: &[u8]) -> String {
    buf.iter().map(|&b| b as char).collect()
}

fn latin1_trim(buf: &[u8]) -> String {
    latin1_to_string(buf).trim_end_matches('\u{0}').to_string()
}

/// Decode a fixed-width ID3v1 text field. ID3v1 pads short strings
/// with NUL *or* spaces; we strip both from the trailing edge.
fn v1_string(buf: &[u8]) -> String {
    // Truncate at first NUL — anything after is padding.
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    latin1_to_string(&buf[..end])
        .trim_end_matches(' ')
        .to_string()
}

fn decode_utf16_bom(buf: &[u8]) -> String {
    if buf.len() < 2 {
        return String::new();
    }
    let (body, le) = match (buf[0], buf[1]) {
        (0xFF, 0xFE) => (&buf[2..], true),
        (0xFE, 0xFF) => (&buf[2..], false),
        _ => (buf, true), // Assume LE if missing a BOM.
    };
    decode_utf16_body(body, le)
}

fn decode_utf16_be(buf: &[u8]) -> String {
    decode_utf16_body(buf, false)
}

fn decode_utf16_body(buf: &[u8], little_endian: bool) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(buf.len() / 2);
    let mut i = 0;
    while i + 1 < buf.len() {
        let u = if little_endian {
            u16::from_le_bytes([buf[i], buf[i + 1]])
        } else {
            u16::from_be_bytes([buf[i], buf[i + 1]])
        };
        units.push(u);
        i += 2;
    }
    String::from_utf16_lossy(&units)
}

/// Map an ID3v2 text frame id to the Vorbis-style key the rest of
/// the workspace uses. Recognises the common frames and falls back
/// to a lowercased raw id for the rest.
fn text_frame_to_key(id: &str) -> String {
    match id {
        "TIT2" => "title",
        "TIT1" => "contentgroup",
        "TIT3" => "subtitle",
        "TPE1" => "artist",
        "TPE2" => "albumartist",
        "TPE3" => "conductor",
        "TPE4" => "remixer",
        "TALB" => "album",
        "TYER" => "date",
        "TDRC" => "date",
        "TDRL" => "releasedate",
        "TDOR" => "originaldate",
        "TCON" => "genre",
        "TRCK" => "track",
        "TPOS" => "disc",
        "TCOM" => "composer",
        "TEXT" => "lyricist",
        "TLAN" => "language",
        "TPUB" => "publisher",
        "TCOP" => "copyright",
        "TENC" => "encodedby",
        "TSSE" => "encoder",
        "TBPM" => "bpm",
        "TCMP" => "compilation",
        "TKEY" => "key",
        "TMED" => "media",
        "TOAL" => "originalalbum",
        "TOPE" => "originalartist",
        "TOLY" => "originallyricist",
        "TORY" => "originalyear",
        "TSRC" => "isrc",
        _ => {
            // Unknown T-frame: expose the raw id lowercased so callers
            // don't drop data silently.
            return id.to_ascii_lowercase();
        }
    }
    .to_string()
}

fn push_unique(out: &mut Vec<(String, String)>, key: String, value: String) {
    if !out.iter().any(|(k, v)| *k == key && *v == value) {
        out.push((key, value));
    }
}

/// Lookup table for ID3v1's genre byte. Covers Winamp's extended
/// ID3v1.1 set (0..191). Indexes beyond the table (or the sentinel
/// 0xFF = "no genre") return None.
fn id3v1_genre(b: u8) -> Option<&'static str> {
    const GENRES: &[&str] = &[
        "Blues",
        "Classic Rock",
        "Country",
        "Dance",
        "Disco",
        "Funk",
        "Grunge",
        "Hip-Hop",
        "Jazz",
        "Metal",
        "New Age",
        "Oldies",
        "Other",
        "Pop",
        "R&B",
        "Rap",
        "Reggae",
        "Rock",
        "Techno",
        "Industrial",
        "Alternative",
        "Ska",
        "Death Metal",
        "Pranks",
        "Soundtrack",
        "Euro-Techno",
        "Ambient",
        "Trip-Hop",
        "Vocal",
        "Jazz+Funk",
        "Fusion",
        "Trance",
        "Classical",
        "Instrumental",
        "Acid",
        "House",
        "Game",
        "Sound Clip",
        "Gospel",
        "Noise",
        "AlternRock",
        "Bass",
        "Soul",
        "Punk",
        "Space",
        "Meditative",
        "Instrumental Pop",
        "Instrumental Rock",
        "Ethnic",
        "Gothic",
        "Darkwave",
        "Techno-Industrial",
        "Electronic",
        "Pop-Folk",
        "Eurodance",
        "Dream",
        "Southern Rock",
        "Comedy",
        "Cult",
        "Gangsta",
        "Top 40",
        "Christian Rap",
        "Pop/Funk",
        "Jungle",
        "Native American",
        "Cabaret",
        "New Wave",
        "Psychadelic",
        "Rave",
        "Showtunes",
        "Trailer",
        "Lo-Fi",
        "Tribal",
        "Acid Punk",
        "Acid Jazz",
        "Polka",
        "Retro",
        "Musical",
        "Rock & Roll",
        "Hard Rock",
        "Folk",
        "Folk-Rock",
        "National Folk",
        "Swing",
        "Fast Fusion",
        "Bebob",
        "Latin",
        "Revival",
        "Celtic",
        "Bluegrass",
        "Avantgarde",
        "Gothic Rock",
        "Progressive Rock",
        "Psychedelic Rock",
        "Symphonic Rock",
        "Slow Rock",
        "Big Band",
        "Chorus",
        "Easy Listening",
        "Acoustic",
        "Humour",
        "Speech",
        "Chanson",
        "Opera",
        "Chamber Music",
        "Sonata",
        "Symphony",
        "Booty Bass",
        "Primus",
        "Porn Groove",
        "Satire",
        "Slow Jam",
        "Club",
        "Tango",
        "Samba",
        "Folklore",
        "Ballad",
        "Power Ballad",
        "Rhythmic Soul",
        "Freestyle",
        "Duet",
        "Punk Rock",
        "Drum Solo",
        "A capella",
        "Euro-House",
        "Dance Hall",
        "Goa",
        "Drum & Bass",
        "Club-House",
        "Hardcore",
        "Terror",
        "Indie",
        "BritPop",
        "Negerpunk",
        "Polsk Punk",
        "Beat",
        "Christian Gangsta Rap",
        "Heavy Metal",
        "Black Metal",
        "Crossover",
        "Contemporary Christian",
        "Christian Rock",
        "Merengue",
        "Salsa",
        "Thrash Metal",
        "Anime",
        "JPop",
        "Synthpop",
    ];
    GENRES.get(b as usize).copied()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a v2.3 tag carrying a TIT2 title and an APIC front cover.
    fn build_v23_tag_title_and_apic() -> Vec<u8> {
        // TIT2 frame: encoding=0 (latin1) + b"Song Title"
        let title = b"Song Title";
        let mut tit2 = Vec::new();
        tit2.extend_from_slice(b"TIT2");
        let t_size = (1 + title.len()) as u32;
        tit2.extend_from_slice(&t_size.to_be_bytes());
        tit2.extend_from_slice(&[0, 0]); // flags
        tit2.push(0); // encoding
        tit2.extend_from_slice(title);

        // APIC: enc=0 + mime "image/png\0" + picture_type=0x03 +
        // description "\0" + binary "PNGDATA".
        let mut apic = Vec::new();
        let mut apic_body = Vec::new();
        apic_body.push(0u8);
        apic_body.extend_from_slice(b"image/png\0");
        apic_body.push(0x03);
        apic_body.push(0);
        apic_body.extend_from_slice(b"PNGDATA");
        apic.extend_from_slice(b"APIC");
        apic.extend_from_slice(&(apic_body.len() as u32).to_be_bytes());
        apic.extend_from_slice(&[0, 0]);
        apic.extend_from_slice(&apic_body);

        let body = [tit2, apic].concat();
        let size = body.len();
        let mut tag = Vec::new();
        tag.extend_from_slice(b"ID3");
        tag.push(3); // major
        tag.push(0); // revision
        tag.push(0); // flags
                     // synchsafe size
        let s = size as u32;
        tag.push(((s >> 21) & 0x7F) as u8);
        tag.push(((s >> 14) & 0x7F) as u8);
        tag.push(((s >> 7) & 0x7F) as u8);
        tag.push((s & 0x7F) as u8);
        tag.extend_from_slice(&body);
        tag
    }

    #[test]
    fn parse_v23_title_and_apic() {
        let tag = build_v23_tag_title_and_apic();
        let (parsed, consumed) = parse_tag(&tag).unwrap();
        assert_eq!(consumed, tag.len());
        assert_eq!(parsed.version, Id3Version::V2_3);
        // Title frame
        let title = parsed.frames.iter().find_map(|f| match f {
            Id3Frame::Text { id, values } if id == "TIT2" => Some(values.clone()),
            _ => None,
        });
        assert_eq!(title.as_deref(), Some(&["Song Title".to_string()][..]));
        // APIC
        let pic = parsed.frames.iter().find_map(|f| match f {
            Id3Frame::Picture(p) => Some(p.clone()),
            _ => None,
        });
        let pic = pic.expect("APIC frame");
        assert_eq!(pic.mime_type, "image/png");
        assert_eq!(pic.picture_type, PictureType::FrontCover);
        assert_eq!(pic.data, b"PNGDATA");
    }

    #[test]
    fn to_kv_title_artist() {
        let mut tag = build_v23_tag_title_and_apic();
        // Append a TPE1 artist frame.
        let artist = b"An Artist";
        let tpe1_body_len = 1 + artist.len();
        let mut frame = Vec::new();
        frame.extend_from_slice(b"TPE1");
        frame.extend_from_slice(&(tpe1_body_len as u32).to_be_bytes());
        frame.extend_from_slice(&[0, 0]);
        frame.push(0);
        frame.extend_from_slice(artist);
        // Splice into the body
        let body_len_offset = 6;
        let old_size = synchsafe_u32(
            tag[body_len_offset],
            tag[body_len_offset + 1],
            tag[body_len_offset + 2],
            tag[body_len_offset + 3],
        ) as usize;
        tag.extend_from_slice(&frame);
        let new_size = (old_size + frame.len()) as u32;
        tag[body_len_offset] = ((new_size >> 21) & 0x7F) as u8;
        tag[body_len_offset + 1] = ((new_size >> 14) & 0x7F) as u8;
        tag[body_len_offset + 2] = ((new_size >> 7) & 0x7F) as u8;
        tag[body_len_offset + 3] = (new_size & 0x7F) as u8;
        let (parsed, _) = parse_tag(&tag).unwrap();
        let kv = to_key_value_pairs(&parsed);
        assert!(kv.contains(&("title".to_string(), "Song Title".to_string())));
        assert!(kv.contains(&("artist".to_string(), "An Artist".to_string())));
        let pics = attached_pictures(&parsed);
        assert_eq!(pics.len(), 1);
    }

    #[test]
    fn parse_v22_pic() {
        // v2.2 tag with one PIC frame.
        // PIC body: enc=0, fmt="JPG", type=0x03, description="\0", data="JPGDATA"
        let mut pic_body = Vec::new();
        pic_body.push(0u8);
        pic_body.extend_from_slice(b"JPG");
        pic_body.push(0x03);
        pic_body.push(0); // empty description terminator
        pic_body.extend_from_slice(b"JPGDATA");
        let mut frame = Vec::new();
        frame.extend_from_slice(b"PIC");
        let size = pic_body.len() as u32;
        frame.push(((size >> 16) & 0xFF) as u8);
        frame.push(((size >> 8) & 0xFF) as u8);
        frame.push((size & 0xFF) as u8);
        frame.extend_from_slice(&pic_body);

        let mut tag = Vec::new();
        tag.extend_from_slice(b"ID3");
        tag.push(2);
        tag.push(0);
        tag.push(0);
        let s = frame.len() as u32;
        tag.push(((s >> 21) & 0x7F) as u8);
        tag.push(((s >> 14) & 0x7F) as u8);
        tag.push(((s >> 7) & 0x7F) as u8);
        tag.push((s & 0x7F) as u8);
        tag.extend_from_slice(&frame);

        let (parsed, _) = parse_tag(&tag).unwrap();
        let pic = attached_pictures(&parsed);
        assert_eq!(pic.len(), 1);
        assert_eq!(pic[0].mime_type, "image/jpeg");
        assert_eq!(pic[0].picture_type, PictureType::FrontCover);
        assert_eq!(pic[0].data, b"JPGDATA");
    }

    #[test]
    fn parse_v1_trailer() {
        let mut trailer = vec![0u8; 128];
        trailer[0..3].copy_from_slice(b"TAG");
        let title = b"TinyTitle";
        trailer[3..3 + title.len()].copy_from_slice(title);
        let artist = b"TinyArtist";
        trailer[33..33 + artist.len()].copy_from_slice(artist);
        // v1.1 track number
        trailer[125] = 0;
        trailer[126] = 7;
        trailer[127] = 17; // genre = Rock
        let tag = parse_id3v1(&trailer).unwrap();
        let kv = to_key_value_pairs(&tag);
        assert!(kv.contains(&("title".to_string(), "TinyTitle".to_string())));
        assert!(kv.contains(&("artist".to_string(), "TinyArtist".to_string())));
        assert!(kv.contains(&("track".to_string(), "7".to_string())));
        assert!(kv.contains(&("genre".to_string(), "Rock".to_string())));
    }

    #[test]
    fn v24_per_frame_unsync_and_dli() {
        // Build a v2.4 tag with a single TIT2 frame that has the
        // data-length indicator + unsync flags set. The TIT2 payload
        // (encoding byte + text) contains an 0xFF that we escape.
        let enc_plus_text = [&[0u8][..], b"AB\xFFCD"].concat();
        // Unsynchronise: insert 0x00 after every 0xFF.
        let mut unsynced = Vec::new();
        for &b in &enc_plus_text {
            unsynced.push(b);
            if b == 0xFF {
                unsynced.push(0x00);
            }
        }
        // DLI prefix: 4 synchsafe bytes giving the *pre-unsync* length.
        let dli = (enc_plus_text.len() as u32).to_be_bytes();
        let mut synchsafe_dli = [0u8; 4];
        let v = enc_plus_text.len() as u32;
        synchsafe_dli[0] = ((v >> 21) & 0x7F) as u8;
        synchsafe_dli[1] = ((v >> 14) & 0x7F) as u8;
        synchsafe_dli[2] = ((v >> 7) & 0x7F) as u8;
        synchsafe_dli[3] = (v & 0x7F) as u8;
        let _ = dli;
        let frame_body = [&synchsafe_dli[..], &unsynced[..]].concat();
        let size = frame_body.len() as u32;
        let mut synchsafe_size = [0u8; 4];
        synchsafe_size[0] = ((size >> 21) & 0x7F) as u8;
        synchsafe_size[1] = ((size >> 14) & 0x7F) as u8;
        synchsafe_size[2] = ((size >> 7) & 0x7F) as u8;
        synchsafe_size[3] = (size & 0x7F) as u8;
        let mut frame = Vec::new();
        frame.extend_from_slice(b"TIT2");
        frame.extend_from_slice(&synchsafe_size);
        // Flags: format-flags low byte = 0x01 (DLI) | 0x02 (unsync) = 0x03
        frame.push(0); // status flags
        frame.push(0x03);
        frame.extend_from_slice(&frame_body);

        let mut tag = Vec::new();
        tag.extend_from_slice(b"ID3");
        tag.push(4);
        tag.push(0);
        tag.push(0); // no whole-tag unsync
        let tag_size = frame.len() as u32;
        tag.push(((tag_size >> 21) & 0x7F) as u8);
        tag.push(((tag_size >> 14) & 0x7F) as u8);
        tag.push(((tag_size >> 7) & 0x7F) as u8);
        tag.push((tag_size & 0x7F) as u8);
        tag.extend_from_slice(&frame);

        let (parsed, _) = parse_tag(&tag).unwrap();
        let got = parsed.frames.iter().find_map(|f| match f {
            Id3Frame::Text { id, values } if id == "TIT2" => Some(values.clone()),
            _ => None,
        });
        assert_eq!(got.as_deref(), Some(&["AB\u{FF}CD".to_string()][..]));
    }

    #[test]
    fn whole_tag_unsync_v23() {
        // TIT2 payload containing 0xFF 0x00 → needs one pass of
        // reverse_unsync at the tag level.
        let payload = [&[0u8][..], b"X\xFFY"].concat();
        let mut frame = Vec::new();
        frame.extend_from_slice(b"TIT2");
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&[0, 0]);
        frame.extend_from_slice(&payload);
        // Apply unsync: after every 0xFF insert 0x00.
        let mut unsynced = Vec::new();
        for &b in &frame {
            unsynced.push(b);
            if b == 0xFF {
                unsynced.push(0x00);
            }
        }
        let mut tag = Vec::new();
        tag.extend_from_slice(b"ID3");
        tag.push(3);
        tag.push(0);
        tag.push(0x80); // unsync flag
        let sz = unsynced.len() as u32;
        tag.push(((sz >> 21) & 0x7F) as u8);
        tag.push(((sz >> 14) & 0x7F) as u8);
        tag.push(((sz >> 7) & 0x7F) as u8);
        tag.push((sz & 0x7F) as u8);
        tag.extend_from_slice(&unsynced);
        let (parsed, _) = parse_tag(&tag).unwrap();
        let got = parsed.frames.iter().find_map(|f| match f {
            Id3Frame::Text { id, values } if id == "TIT2" => Some(values.clone()),
            _ => None,
        });
        assert_eq!(got.as_deref(), Some(&["X\u{FF}Y".to_string()][..]));
    }

    #[test]
    fn tag_size_at_head_basic() {
        let tag = build_v23_tag_title_and_apic();
        let size = tag_size_at_head(&tag[0..10]).unwrap();
        assert_eq!(size, tag.len());
    }
}
