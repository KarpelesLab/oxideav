//! Integration tests against ffmpeg / aomenc generated AV1 reference clips.
//!
//! Fixtures expected at:
//!   /tmp/av1.mp4   — 64x64 @ 24fps, 0.1s (3 frames), all keyframes, 1 tile.
//!   /tmp/av1.ivf   — same content, IVF container (raw OBUs concatenated
//!                    with IVF framing).
//!
//! Generated with:
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.1" \
//!     -f rawvideo -pix_fmt yuv420p /tmp/av1in.yuv
//!   aomenc --ivf -w 64 -h 64 --fps=24/1 --cpu-used=8 \
//!     --tile-columns=0 --tile-rows=0 \
//!     --kf-min-dist=1 --kf-max-dist=1 -o /tmp/av1.ivf /tmp/av1in.yuv
//!   ffmpeg -y -i /tmp/av1.ivf -c copy /tmp/av1.mp4
//!
//! Tests that can't find their fixture are skipped (logged, not failed) so
//! CI without ffmpeg + aomenc still passes.

use std::path::Path;

use oxideav_av1::{
    iter_obus, parse_frame_header, parse_sequence_header, Av1CodecConfig, Av1Decoder, ObuType,
};
use oxideav_core::{CodecId, CodecParameters, Error, Packet, TimeBase};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

/// Locate the av1C box body inside an MP4 file by walking the box tree.
fn mp4_find_box<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    fn rec<'b>(data: &'b [u8], path: &[&[u8; 4]], depth: usize) -> Option<&'b [u8]> {
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let sz = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            let ty = &data[pos + 4..pos + 8];
            let (header_len, total) = if sz == 1 && pos + 16 <= data.len() {
                let large = u64::from_be_bytes([
                    data[pos + 8],
                    data[pos + 9],
                    data[pos + 10],
                    data[pos + 11],
                    data[pos + 12],
                    data[pos + 13],
                    data[pos + 14],
                    data[pos + 15],
                ]) as usize;
                (16, large)
            } else if sz == 0 {
                (8, data.len() - pos)
            } else {
                (8, sz)
            };
            let body_start = pos + header_len;
            let body_end = pos + total;
            if body_end > data.len() {
                return None;
            }
            if ty == path[depth] {
                if depth + 1 == path.len() {
                    return Some(&data[body_start..body_end]);
                }
                let body = &data[body_start..body_end];
                let inner_start = match path[depth] {
                    b"stsd" => 8usize, // version+flags+entry_count
                    b"av01" | b"avc1" | b"avc3" | b"hvc1" | b"hev1" => 78usize, // VisualSampleEntry
                    _ => 0,
                };
                if body.len() > inner_start {
                    if let Some(found) = rec(&body[inner_start..], path, depth + 1) {
                        return Some(found);
                    }
                }
            }
            pos += total;
        }
        None
    }
    rec(data, path, 0)
}

/// Strip the IVF wrapper and return the concatenated OBU payloads of all
/// frames. IVF: 32-byte file header, then per-frame [4-byte LE size, 8-byte
/// LE pts, payload].
fn ivf_concat_obus(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out = Vec::new();
    let mut pos = header_len;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        if pos + size > data.len() {
            break;
        }
        out.extend_from_slice(&data[pos..pos + size]);
        pos += size;
    }
    Some(out)
}

#[test]
fn parse_av1c_from_mp4() {
    let Some(data) = read_fixture("/tmp/av1.mp4") else {
        return;
    };
    let av1c = mp4_find_box(
        &data,
        &[
            b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"av01", b"av1C",
        ],
    )
    .expect("av1C box in MP4");
    let cfg = Av1CodecConfig::parse(av1c).expect("parse av1C");
    assert_eq!(cfg.version, 1);
    assert_eq!(cfg.seq_profile, 0);
    let sh = cfg.seq_header.as_ref().expect("embedded seq header");
    assert_eq!(sh.max_frame_width, 64, "expected 64x64");
    assert_eq!(sh.max_frame_height, 64, "expected 64x64");
    assert_eq!(sh.color_config.bit_depth, 8);
    assert_eq!(sh.color_config.num_planes, 3);
    eprintln!(
        "av1C: profile={} level={} tier={} subx={} suby={}",
        cfg.seq_profile,
        cfg.seq_level_idx_0,
        cfg.seq_tier_0 as u8,
        cfg.chroma_subsampling_x as u8,
        cfg.chroma_subsampling_y as u8,
    );
}

#[test]
fn iterate_obus_in_ivf() {
    let Some(data) = read_fixture("/tmp/av1.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut counts = std::collections::HashMap::<&'static str, usize>::new();
    let mut sh = None;
    let mut had_temporal = false;
    let mut frame_seen = 0usize;
    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        *counts.entry(o.header.obu_type.name()).or_insert(0) += 1;
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::TemporalDelimiter => had_temporal = true,
            ObuType::Frame | ObuType::FrameHeader => {
                let s = sh.as_ref().expect("seq hdr before frame");
                let fh = parse_frame_header(s, o.payload).expect("frame header");
                eprintln!(
                    "frame {frame_seen}: type={:?} size={}x{} show={} primary_ref={}",
                    fh.frame_type,
                    fh.frame_width,
                    fh.frame_height,
                    fh.show_frame,
                    fh.primary_ref_frame,
                );
                frame_seen += 1;
            }
            _ => {}
        }
    }
    let sh = sh.expect("at least one sequence header");
    assert_eq!(sh.max_frame_width, 64);
    assert_eq!(sh.max_frame_height, 64);
    assert_eq!(sh.seq_profile, 0);
    assert!(had_temporal);
    assert!(frame_seen >= 1, "expected at least one parsed frame header");
    eprintln!("OBU counts: {counts:?}");
}

#[test]
fn decoder_extradata_picks_up_seq_header() {
    let Some(data) = read_fixture("/tmp/av1.mp4") else {
        return;
    };
    let av1c = mp4_find_box(
        &data,
        &[
            b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"av01", b"av1C",
        ],
    )
    .expect("av1C body");
    let mut params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    params.extradata = av1c.to_vec();
    params.width = Some(64);
    params.height = Some(64);
    let dec = Av1Decoder::new(params);
    let sh = dec.sequence_header().expect("seq header from extradata");
    assert_eq!(sh.max_frame_width, 64);
    assert_eq!(sh.max_frame_height, 64);
}

#[test]
fn decoder_returns_unsupported_for_tile_decode() {
    let Some(data) = read_fixture("/tmp/av1.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");

    let params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    let mut dec = oxideav_av1::make_decoder(&params).expect("build decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 24), obus);
    // The tile-decode skeleton now surfaces a more precise message —
    // partition CDFs (§9.4) — before the historical "§5.11 pending" text.
    // Accept either because the skeleton message is the near-term target
    // and may be improved as more sub-sections land.
    match dec.send_packet(&pkt) {
        Err(Error::Unsupported(s)) => {
            assert!(
                s.contains("§5.11") || s.contains("§9.4") || s.contains("§7"),
                "Unsupported message should reference an AV1 spec §, got: {s}"
            );
        }
        other => panic!("expected Unsupported(tile decode), got {other:?}"),
    }
    match dec.receive_frame() {
        Err(Error::Unsupported(s)) => assert!(
            s.contains("§5.11") || s.contains("§9.4") || s.contains("§7"),
            "msg: {s}"
        ),
        other => panic!("receive_frame should be Unsupported, got {other:?}"),
    }
}
