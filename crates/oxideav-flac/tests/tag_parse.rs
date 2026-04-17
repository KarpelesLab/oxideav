//! End-to-end FLAC metadata + picture surfaces test.
//!
//! Builds a minimal FLAC prefix: `fLaC` magic + STREAMINFO block +
//! VorbisComment block (title / artist) + PICTURE block. Opens
//! through the container registry and asserts that the demuxer's
//! `metadata()` and `attached_pictures()` expose the embedded data.
//!
//! No frames follow the metadata blocks — we only exercise the
//! demuxer's header-parsing path.

use std::io::Cursor;

use oxideav_container::ContainerRegistry;
use oxideav_core::PictureType;

/// Pack a FLAC metadata block header: last-flag (1 bit) + type (7 bits) +
/// 24-bit big-endian length.
fn block_header(is_last: bool, block_type: u8, length: usize) -> [u8; 4] {
    let b0 = if is_last { 0x80 } else { 0x00 } | (block_type & 0x7F);
    [
        b0,
        ((length >> 16) & 0xFF) as u8,
        ((length >> 8) & 0xFF) as u8,
        (length & 0xFF) as u8,
    ]
}

fn build_streaminfo() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&4096u16.to_be_bytes());
    block[2..4].copy_from_slice(&4096u16.to_be_bytes());
    // packed: sr(20)=44100, ch-1(3)=1 (stereo), bps-1(5)=15 (16-bit),
    // total_samples(36)=0 (unknown).
    let packed: u64 = (44_100u64 << 44) | (1u64 << 41) | (15u64 << 36);
    block[10..18].copy_from_slice(&packed.to_be_bytes());
    block
}

fn build_vorbis_comment() -> Vec<u8> {
    let vendor = b"oxideav-test";
    let comments: &[&[u8]] = &[b"TITLE=Song", b"ARTIST=Artist"];
    let mut out = Vec::new();
    out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor);
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in comments {
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
        out.extend_from_slice(c);
    }
    out
}

fn build_picture_block() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&3u32.to_be_bytes()); // FrontCover
    let mime = b"image/png";
    out.extend_from_slice(&(mime.len() as u32).to_be_bytes());
    out.extend_from_slice(mime);
    let desc = b"cover";
    out.extend_from_slice(&(desc.len() as u32).to_be_bytes());
    out.extend_from_slice(desc);
    out.extend_from_slice(&0u32.to_be_bytes()); // width
    out.extend_from_slice(&0u32.to_be_bytes()); // height
    out.extend_from_slice(&0u32.to_be_bytes()); // depth
    out.extend_from_slice(&0u32.to_be_bytes()); // colour count
    let data = b"PNGDATA";
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn flac_metadata_and_picture_surface_through() {
    let streaminfo = build_streaminfo();
    let vorbis = build_vorbis_comment();
    let picture = build_picture_block();

    let mut file = Vec::new();
    file.extend_from_slice(b"fLaC");
    file.extend_from_slice(&block_header(false, 0, streaminfo.len()));
    file.extend_from_slice(&streaminfo);
    file.extend_from_slice(&block_header(false, 4, vorbis.len()));
    file.extend_from_slice(&vorbis);
    file.extend_from_slice(&block_header(true, 6, picture.len()));
    file.extend_from_slice(&picture);

    let mut reg = ContainerRegistry::new();
    oxideav_flac::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg.open_demuxer("flac", cursor).expect("open flac");

    let md = demuxer.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "title" && v == "Song"),
        "title missing: {:?}",
        md
    );
    assert!(
        md.iter().any(|(k, v)| k == "artist" && v == "Artist"),
        "artist missing: {:?}",
        md
    );

    let pics = demuxer.attached_pictures();
    assert_eq!(pics.len(), 1, "expected 1 picture, got {}", pics.len());
    assert_eq!(pics[0].mime_type, "image/png");
    assert_eq!(pics[0].picture_type, PictureType::FrontCover);
    assert_eq!(pics[0].description, "cover");
    assert_eq!(pics[0].data, b"PNGDATA");
}

#[test]
fn flac_id3v2_prefix_surfaces_fallback_metadata() {
    // Some taggers slap an ID3v2 tag in front of fLaC. Verify the
    // demuxer parses it and exposes the tags, even though FLAC's own
    // Vorbis-comment block is the canonical spot.
    let mut tag = Vec::new();
    // TIT2 "FromId3"
    let payload = [&[0u8][..], b"FromId3"].concat();
    let mut tit2 = Vec::new();
    tit2.extend_from_slice(b"TIT2");
    tit2.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    tit2.extend_from_slice(&[0, 0]);
    tit2.extend_from_slice(&payload);
    let body = tit2;
    let s = body.len() as u32;
    tag.extend_from_slice(b"ID3");
    tag.push(3);
    tag.push(0);
    tag.push(0);
    tag.push(((s >> 21) & 0x7F) as u8);
    tag.push(((s >> 14) & 0x7F) as u8);
    tag.push(((s >> 7) & 0x7F) as u8);
    tag.push((s & 0x7F) as u8);
    tag.extend_from_slice(&body);

    let streaminfo = build_streaminfo();
    let mut file = Vec::new();
    file.extend_from_slice(&tag);
    file.extend_from_slice(b"fLaC");
    file.extend_from_slice(&block_header(true, 0, streaminfo.len()));
    file.extend_from_slice(&streaminfo);

    let mut reg = ContainerRegistry::new();
    oxideav_flac::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg.open_demuxer("flac", cursor).expect("open flac");
    let md = demuxer.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "title" && v == "FromId3"),
        "ID3v2 title missing: {:?}",
        md
    );
}
