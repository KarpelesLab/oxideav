//! Theora Comment header (packet 1). Spec §6.3.
//!
//! Identical layout to a Vorbis comment block but with a 0x81 + "theora" magic
//! prefix instead of 0x03 + "vorbis".

use oxideav_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct Comment {
    pub vendor: String,
    pub comments: Vec<(String, String)>,
}

pub fn parse_comment_header(packet: &[u8]) -> Result<Comment> {
    if packet.len() < 7 {
        return Err(Error::invalid("Theora comment header too short"));
    }
    if packet[0] != 0x81 || &packet[1..7] != b"theora" {
        return Err(Error::invalid("not a Theora comment header"));
    }
    let body = &packet[7..];
    let mut off = 0usize;

    fn read_le32(buf: &[u8], off: &mut usize) -> Result<u32> {
        if buf.len() < *off + 4 {
            return Err(Error::invalid("Theora comment: truncated length"));
        }
        let v = u32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
        *off += 4;
        Ok(v)
    }

    let vlen = read_le32(body, &mut off)? as usize;
    if body.len() < off + vlen {
        return Err(Error::invalid("Theora comment: vendor string truncated"));
    }
    let vendor = String::from_utf8_lossy(&body[off..off + vlen]).into_owned();
    off += vlen;

    let count = read_le32(body, &mut off)? as usize;
    let mut comments = Vec::with_capacity(count);
    for _ in 0..count {
        let clen = read_le32(body, &mut off)? as usize;
        if body.len() < off + clen {
            return Err(Error::invalid("Theora comment: user string truncated"));
        }
        let raw = String::from_utf8_lossy(&body[off..off + clen]).into_owned();
        off += clen;
        let (k, v) = match raw.find('=') {
            Some(p) => (raw[..p].to_owned(), raw[p + 1..].to_owned()),
            None => (raw, String::new()),
        };
        comments.push((k, v));
    }
    Ok(Comment { vendor, comments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short() {
        assert!(parse_comment_header(b"").is_err());
    }

    #[test]
    fn parses_minimal() {
        let mut p = vec![0x81u8];
        p.extend_from_slice(b"theora");
        p.extend_from_slice(&5u32.to_le_bytes());
        p.extend_from_slice(b"hello");
        p.extend_from_slice(&1u32.to_le_bytes());
        let comment = b"KEY=value";
        p.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        p.extend_from_slice(comment);
        let c = parse_comment_header(&p).unwrap();
        assert_eq!(c.vendor, "hello");
        assert_eq!(c.comments.len(), 1);
        assert_eq!(c.comments[0].0, "KEY");
        assert_eq!(c.comments[0].1, "value");
    }
}
