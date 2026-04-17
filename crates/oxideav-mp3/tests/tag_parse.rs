//! End-to-end: an MP3 file with an ID3v2.3 tag (title + artist + APIC)
//! in front of a minimal MPEG-1 Layer III frame. Opens through the
//! container registry and asserts metadata + attached_pictures flow
//! through to the demuxer's API.
//!
//! The MPEG frame here is a synthesised 128 kbps / 44.1 kHz / mono
//! header + zeroed body — the decoder isn't exercised, just the
//! container's first-frame probe.

use std::io::Cursor;

use oxideav_container::ContainerRegistry;
use oxideav_core::PictureType;

/// Build a minimal ID3v2.3 tag carrying TIT2, TPE1, and an APIC.
fn build_id3v23_tag() -> Vec<u8> {
    // TIT2 "Song"
    let mut tit2 = Vec::new();
    let title_payload = [&[0u8][..], b"Song"].concat();
    tit2.extend_from_slice(b"TIT2");
    tit2.extend_from_slice(&(title_payload.len() as u32).to_be_bytes());
    tit2.extend_from_slice(&[0, 0]);
    tit2.extend_from_slice(&title_payload);

    // TPE1 "Artist"
    let mut tpe1 = Vec::new();
    let artist_payload = [&[0u8][..], b"Artist"].concat();
    tpe1.extend_from_slice(b"TPE1");
    tpe1.extend_from_slice(&(artist_payload.len() as u32).to_be_bytes());
    tpe1.extend_from_slice(&[0, 0]);
    tpe1.extend_from_slice(&artist_payload);

    // APIC: encoding=0, mime "image/png\0", type=0x03, desc "cover\0", data.
    let mut apic_payload = Vec::new();
    apic_payload.push(0u8);
    apic_payload.extend_from_slice(b"image/png\0");
    apic_payload.push(0x03);
    apic_payload.extend_from_slice(b"cover\0");
    apic_payload.extend_from_slice(b"PNGBYTES");
    let mut apic = Vec::new();
    apic.extend_from_slice(b"APIC");
    apic.extend_from_slice(&(apic_payload.len() as u32).to_be_bytes());
    apic.extend_from_slice(&[0, 0]);
    apic.extend_from_slice(&apic_payload);

    let body = [tit2, tpe1, apic].concat();
    let size = body.len() as u32;
    let mut tag = Vec::new();
    tag.extend_from_slice(b"ID3");
    tag.push(3);
    tag.push(0);
    tag.push(0);
    tag.push(((size >> 21) & 0x7F) as u8);
    tag.push(((size >> 14) & 0x7F) as u8);
    tag.push(((size >> 7) & 0x7F) as u8);
    tag.push((size & 0x7F) as u8);
    tag.extend_from_slice(&body);
    tag
}

/// Build a 128 kbps / 44.1 kHz / stereo MPEG-1 Layer III frame with a
/// zero-filled body. Frame length is exactly 417 bytes.
fn build_mp3_frame() -> Vec<u8> {
    let mut f = vec![0u8; 417];
    f[0] = 0xFF;
    f[1] = 0xFB;
    f[2] = 0x90;
    f[3] = 0x00;
    f
}

#[test]
fn mp3_metadata_and_pictures_flow_through() {
    let mut file = Vec::new();
    file.extend_from_slice(&build_id3v23_tag());
    file.extend_from_slice(&build_mp3_frame());
    file.extend_from_slice(&build_mp3_frame());

    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_container::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg.open_demuxer("mp3", cursor).expect("open mp3 demuxer");

    let md = demuxer.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "title" && v == "Song"),
        "title not in metadata: {:?}",
        md
    );
    assert!(
        md.iter().any(|(k, v)| k == "artist" && v == "Artist"),
        "artist not in metadata: {:?}",
        md
    );

    let pics = demuxer.attached_pictures();
    assert_eq!(pics.len(), 1);
    assert_eq!(pics[0].mime_type, "image/png");
    assert_eq!(pics[0].picture_type, PictureType::FrontCover);
    assert_eq!(pics[0].description, "cover");
    assert_eq!(pics[0].data, b"PNGBYTES");
}
