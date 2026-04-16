//! Integration tests against ffmpeg-generated MPEG-4 Part 2 reference clips.
//!
//! Fixtures expected at:
//!   /tmp/ref-mpeg4-iframes.avi  (64x64 @ 10 fps, 1s, every frame I)
//!   /tmp/ref-mpeg4-gop.avi      (128x96 @ 10 fps, 2s, GOP=10)
//!
//! Generated with:
//!   ffmpeg -y -f lavfi -i testsrc=d=1:s=64x64:r=10 -c:v mpeg4 -g 1 -b:v 500k \
//!       /tmp/ref-mpeg4-iframes.avi
//!   ffmpeg -y -f lavfi -i testsrc=d=2:s=128x96:r=10 -c:v mpeg4 -g 10 -b:v 800k \
//!       /tmp/ref-mpeg4-gop.avi
//!
//! Tests that can't find their fixture are skipped (logged, not failed) so
//! CI without ffmpeg still passes.

use std::path::Path;

use oxideav_mpeg4video::{
    bitreader::BitReader,
    decoder::codec_parameters_from_vol,
    headers::{
        vol::parse_vol,
        vop::{parse_vop, VopCodingType},
        vos::{parse_visual_object, parse_vos, profile_level_description},
    },
    start_codes::{self, VISUAL_OBJECT_START_CODE, VOL_START_MIN, VOP_START_CODE, VOS_START_CODE},
};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

/// Find the first occurrence of a start code matching `predicate`.
fn find_start_code<F: Fn(u8) -> bool>(data: &[u8], predicate: F) -> Option<(usize, u8)> {
    start_codes::iter_start_codes(data).find(|(_, c)| predicate(*c))
}

#[test]
fn parse_vos_vo_vol_iframes() {
    let Some(data) = read_fixture("/tmp/ref-mpeg4-iframes.avi") else {
        return;
    };
    // VOS.
    let (pos, code) = find_start_code(&data, |c| c == VOS_START_CODE).expect("VOS start code");
    assert_eq!(code, VOS_START_CODE);
    let mut br = BitReader::new(&data[pos + 4..pos + 5]);
    let vos = parse_vos(&mut br).expect("parse VOS");
    eprintln!(
        "profile_and_level = 0x{:02x} ({})",
        vos.profile_and_level_indication,
        profile_level_description(vos.profile_and_level_indication)
    );

    // Visual Object.
    let (pos, _) = find_start_code(&data, |c| c == VISUAL_OBJECT_START_CODE)
        .expect("visual_object start code");
    // Payload ends at the next start code.
    let next = start_codes::iter_start_codes(&data[pos + 4..])
        .next()
        .map(|(p, _)| pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[pos + 4..next]);
    let _vo = parse_visual_object(&mut br).expect("parse VO");

    // VOL.
    let (pos, _) =
        find_start_code(&data, start_codes::is_video_object_layer).expect("VOL start code");
    let next = start_codes::iter_start_codes(&data[pos + 4..])
        .next()
        .map(|(p, _)| pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[pos + 4..next]);
    let vol = parse_vol(&mut br).expect("parse VOL");
    assert_eq!(vol.width, 64, "VOL width");
    assert_eq!(vol.height, 64, "VOL height");

    // CodecParameters population.
    let params = codec_parameters_from_vol(&vol);
    assert_eq!(params.width, Some(64));
    assert_eq!(params.height, Some(64));
    let fr = params.frame_rate.expect("frame rate");
    // Frame rate is (resolution / fixed_vop_time_increment) if fixed, else
    // (resolution, 1). ffmpeg with -r 10 produces resolution 10 or 1000 and
    // fixed_vop_time_increment 1 or 100 — any ratio should reduce to 10/1.
    let ratio = fr.num as f64 / fr.den as f64;
    assert!(
        (ratio - 10.0).abs() < 0.5,
        "expected frame rate ~10 fps, got {}/{} = {}",
        fr.num,
        fr.den,
        ratio
    );
    // Sanity: bounds on VOL are what we built.
    assert_eq!(vol.mb_width(), 4);
    assert_eq!(vol.mb_height(), 4);
    let _ = VOL_START_MIN; // silence unused-import warning
}

#[test]
fn parse_first_vop_header_iframes() {
    let Some(data) = read_fixture("/tmp/ref-mpeg4-iframes.avi") else {
        return;
    };
    // Parse VOL first for the VOP parser's context.
    let (pos, _) =
        find_start_code(&data, start_codes::is_video_object_layer).expect("VOL start code");
    let next = start_codes::iter_start_codes(&data[pos + 4..])
        .next()
        .map(|(p, _)| pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[pos + 4..next]);
    let vol = parse_vol(&mut br).expect("parse VOL");

    // First VOP.
    let (pos, code) = find_start_code(&data, |c| c == VOP_START_CODE).expect("VOP start code");
    assert_eq!(code, VOP_START_CODE);
    let next = start_codes::iter_start_codes(&data[pos + 4..])
        .next()
        .map(|(p, _)| pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[pos + 4..next]);
    let vop = parse_vop(&mut br, &vol).expect("parse VOP");
    assert_eq!(
        vop.vop_coding_type,
        VopCodingType::I,
        "first VOP in an I-only stream is I"
    );
    assert!(vop.vop_coded, "first VOP coded");
    assert!(vop.vop_quant > 0, "VOP quant > 0");
    eprintln!(
        "VOP0: type={:?} quant={} time_increment={}",
        vop.vop_coding_type, vop.vop_quant, vop.vop_time_increment
    );
}

#[test]
fn parse_vol_gop_clip() {
    let Some(data) = read_fixture("/tmp/ref-mpeg4-gop.avi") else {
        return;
    };
    let (pos, _) =
        find_start_code(&data, start_codes::is_video_object_layer).expect("VOL start code");
    let next = start_codes::iter_start_codes(&data[pos + 4..])
        .next()
        .map(|(p, _)| pos + 4 + p)
        .unwrap_or(data.len());
    let mut br = BitReader::new(&data[pos + 4..next]);
    let vol = parse_vol(&mut br).expect("parse VOL");
    assert_eq!(vol.width, 128);
    assert_eq!(vol.height, 96);
}

/// VOP body decode is not landed yet — the full decoder rejects with
/// `Unsupported` so callers know a follow-up owes them I-VOP decode.
/// This test documents the intended behaviour.
#[test]
fn decoder_rejects_iframe_body_with_clear_message() {
    use oxideav_core::{CodecId, CodecParameters, Error, Packet, TimeBase};

    let Some(data) = read_fixture("/tmp/ref-mpeg4-iframes.avi") else {
        return;
    };
    let params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).expect("build decoder");
    let packet = Packet::new(0, TimeBase::new(1, 90_000), data);
    match dec.send_packet(&packet) {
        Err(Error::Unsupported(msg)) => {
            eprintln!("expected Unsupported: {msg}");
            assert!(
                msg.contains("I-VOP") || msg.contains("mpeg4"),
                "message should mention mpeg4 I-VOP: {msg}"
            );
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(()) => panic!("expected Unsupported while I-VOP decode is stubbed"),
    }
}
