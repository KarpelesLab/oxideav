//! Theora header packet parsers.

pub mod comment;
pub mod identification;
pub mod setup;

pub use comment::{parse_comment_header, Comment};
pub use identification::{parse_identification_header, Identification, PixelFormat};
pub use setup::{parse_setup_header, Setup};

use oxideav_core::{Error, Result};

/// Parse Xiph-laced extradata (same layout used for Vorbis/Theora in MKV/MP4):
/// one count byte (N-1) + Xiph-style packet lengths + the N packets.
pub fn parse_xiph_extradata(extradata: &[u8]) -> Result<Vec<Vec<u8>>> {
    if extradata.is_empty() {
        return Err(Error::invalid("Theora: empty extradata"));
    }
    let n = extradata[0] as usize + 1;
    let mut off = 1usize;
    let mut sizes = Vec::with_capacity(n);
    for _ in 0..n - 1 {
        let mut sz = 0usize;
        loop {
            if off >= extradata.len() {
                return Err(Error::invalid("Theora: truncated Xiph lacing"));
            }
            let b = extradata[off];
            off += 1;
            sz += b as usize;
            if b != 255 {
                break;
            }
        }
        sizes.push(sz);
    }
    let mut packets = Vec::with_capacity(n);
    for sz in sizes {
        if extradata.len() < off + sz {
            return Err(Error::invalid("Theora: packet body truncated"));
        }
        packets.push(extradata[off..off + sz].to_vec());
        off += sz;
    }
    packets.push(extradata[off..].to_vec());
    Ok(packets)
}

/// Parse all three Theora headers from Xiph-laced extradata. Returns the
/// three parsed headers in order.
#[derive(Clone, Debug)]
pub struct Headers {
    pub identification: Identification,
    pub comment: Comment,
    pub setup: Setup,
}

pub fn parse_headers_from_extradata(extradata: &[u8]) -> Result<Headers> {
    let packets = parse_xiph_extradata(extradata)?;
    if packets.len() < 3 {
        return Err(Error::invalid(format!(
            "Theora: extradata yielded {} packets, expected 3",
            packets.len()
        )));
    }
    let identification = parse_identification_header(&packets[0])?;
    let comment = parse_comment_header(&packets[1])?;
    let setup = parse_setup_header(&packets[2])?;
    Ok(Headers {
        identification,
        comment,
        setup,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xiph_lacing_roundtrip() {
        // Mimic three packets of sizes [300, 20, 1000]. With n=3, the lacing
        // stores the first two sizes using 255-continued bytes.
        let p0 = vec![0xAAu8; 300];
        let p1 = vec![0xBBu8; 20];
        let p2 = vec![0xCCu8; 1000];
        let mut extra = vec![2u8]; // n - 1
                                   // size 300 = 255 + 45
        extra.push(255);
        extra.push(45);
        // size 20 = 20
        extra.push(20);
        extra.extend_from_slice(&p0);
        extra.extend_from_slice(&p1);
        extra.extend_from_slice(&p2);
        let out = parse_xiph_extradata(&extra).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 300);
        assert_eq!(out[1].len(), 20);
        assert_eq!(out[2].len(), 1000);
    }
}
