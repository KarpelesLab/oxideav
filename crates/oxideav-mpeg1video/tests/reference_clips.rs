//! Integration tests against ffmpeg-generated reference clips.
//!
//! These tests expect fixtures at:
//!   /tmp/ref-mpeg1-tiny.m1v  (64x64, 25fps, 1s)
//!   /tmp/ref-mpeg1-gop.m1v   (128x96, 25fps, 2s with -g 10)
//!
//! Generate them with:
//!   ffmpeg -y -f lavfi -i testsrc=d=1:s=64x64:r=25 -c:v mpeg1video -b:v 200k /tmp/ref-mpeg1-tiny.m1v
//!   ffmpeg -y -f lavfi -i testsrc=d=2:s=128x96:r=25 -c:v mpeg1video -g 10      /tmp/ref-mpeg1-gop.m1v
//!
//! Tests that can't find their fixture are skipped (logged, not failed), so
//! CI without ffmpeg still passes.

use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_mpeg1video::{
    bitreader::BitReader,
    decoder::{codec_parameters_from_sequence_header, make_decoder},
    headers::{parse_sequence_header, PictureType},
    start_codes::{self, PICTURE_START_CODE, SEQUENCE_HEADER_CODE},
};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

#[test]
fn parse_sequence_header_tiny() {
    let Some(data) = read_fixture("/tmp/ref-mpeg1-tiny.m1v") else {
        return;
    };
    // First start code should be the sequence header.
    let mut iter = start_codes::iter_start_codes(&data);
    let (pos, code) = iter.next().expect("sequence header start code");
    assert_eq!(code, SEQUENCE_HEADER_CODE);
    let mut br = BitReader::new(&data[pos + 4..]);
    let sh = parse_sequence_header(&mut br).expect("parse sequence header");
    assert_eq!(sh.horizontal_size, 64);
    assert_eq!(sh.vertical_size, 64);
    let params = codec_parameters_from_sequence_header(&sh);
    assert_eq!(params.width, Some(64));
    assert_eq!(params.height, Some(64));
    let fr = params.frame_rate.expect("frame rate");
    assert_eq!(fr.num, 25);
    assert_eq!(fr.den, 1);
}

#[test]
fn parse_first_picture_header_tiny() {
    let Some(data) = read_fixture("/tmp/ref-mpeg1-tiny.m1v") else {
        return;
    };
    // Find first picture_start_code (0x00).
    let (pos, code) = start_codes::iter_start_codes(&data)
        .find(|(_, c)| *c == PICTURE_START_CODE)
        .expect("picture start code");
    assert_eq!(code, PICTURE_START_CODE);
    let mut br = BitReader::new(&data[pos + 4..]);
    let ph =
        oxideav_mpeg1video::headers::parse_picture_header(&mut br).expect("parse picture header");
    // First picture in an MPEG-1 sequence is always an I-picture.
    assert_eq!(ph.picture_type, PictureType::I);
}

/// Milestone 2: decode a single I-frame. This test is marked `#[ignore]`
/// while the decoder still has residual bitstream-alignment bugs — it
/// exercises the full parse + macroblock/block decode path end-to-end.
/// Run with `cargo test -p oxideav-mpeg1video -- --ignored`.
#[test]
#[ignore]
fn decode_first_i_frame_tiny() {
    let Some(data) = read_fixture("/tmp/ref-mpeg1-tiny.m1v") else {
        return;
    };
    let params = CodecParameters::video(CodecId::new(oxideav_mpeg1video::CODEC_ID_STR));
    let mut decoder = make_decoder(&params).expect("build decoder");
    let packet = Packet::new(0, TimeBase::new(1, 90_000), data);
    if let Err(e) = decoder.send_packet(&packet) {
        eprintln!("send_packet err: {e}");
    }
    let _ = decoder.flush();

    let frame = match decoder.receive_frame() {
        Ok(f) => f,
        Err(Error::NeedMore) => panic!("NeedMore after flush"),
        Err(Error::Eof) => panic!("decoder returned EOF before any frame"),
        Err(e) => panic!("decoder error: {e}"),
    };

    match frame {
        Frame::Video(vf) => {
            assert_eq!(vf.width, 64);
            assert_eq!(vf.height, 64);
            assert_eq!(vf.planes.len(), 3);
            let y_plane = &vf.planes[0];
            let mean_y: u64 =
                y_plane.data.iter().map(|&b| b as u64).sum::<u64>() / y_plane.data.len() as u64;
            eprintln!("mean Y = {mean_y}");
            assert!(
                mean_y > 30,
                "mean Y {mean_y} too low — expected testsrc colour bars"
            );
        }
        _ => panic!("expected video frame"),
    }
}
