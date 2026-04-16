//! Integration tests against ffmpeg-generated HEVC clips.
//!
//! Fixtures expected at:
//!   /tmp/h265_iframe.mp4   (64x64 @ 24 fps, single I frame)
//!   /tmp/h265.es           (Annex B stream extracted from above)
//!
//! Generated with:
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.1" \
//!       -pix_fmt yuv420p -c:v libx265 -profile:v main -preset:v slow \
//!       -g 1 -x265-params log-level=error /tmp/h265_iframe.mp4
//!   ffmpeg -y -i /tmp/h265_iframe.mp4 -c:v copy -f hevc /tmp/h265.es
//!
//! Tests skip cleanly (logged, not failed) when fixtures are missing so CI
//! without ffmpeg keeps passing.

use std::path::Path;

use oxideav_h265::nal::{iter_annex_b, iter_length_prefixed, NalHeader, NalUnitType};
use oxideav_h265::pps::parse_pps;
use oxideav_h265::sps::parse_sps;
use oxideav_h265::vps::parse_vps;
use oxideav_h265::{decoder::HevcDecoder, hvcc::parse_hvcc, slice::parse_slice_segment_header};
use oxideav_h265::{nal::extract_rbsp, CODEC_ID_STR};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

/// Walk an MP4 box tree and find the body of the first box whose path
/// matches the supplied 4-char fourcc list. Best-effort; returns `None`
/// if any segment is missing.
fn mp4_find_box<'a>(data: &'a [u8], path: &[&str]) -> Option<&'a [u8]> {
    let mut buf: &'a [u8] = data;
    let mut idx = 0;
    while idx < path.len() {
        let target = path[idx];
        let target_b = target.as_bytes();
        let mut pos = 0;
        let mut found: Option<(&'a [u8], usize, usize)> = None;
        while pos + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            let kind = &buf[pos + 4..pos + 8];
            let (header_len, body_len) = if size == 1 {
                if pos + 16 > buf.len() {
                    return None;
                }
                let large = u64::from_be_bytes([
                    buf[pos + 8],
                    buf[pos + 9],
                    buf[pos + 10],
                    buf[pos + 11],
                    buf[pos + 12],
                    buf[pos + 13],
                    buf[pos + 14],
                    buf[pos + 15],
                ]) as usize;
                (16, large.saturating_sub(16))
            } else if size == 0 {
                (8, buf.len() - pos - 8)
            } else {
                (8, size.saturating_sub(8))
            };
            if pos + header_len + body_len > buf.len() {
                return None;
            }
            if kind == target_b {
                found = Some((
                    &buf[pos + header_len..pos + header_len + body_len],
                    header_len,
                    body_len,
                ));
                break;
            }
            pos += header_len + body_len;
        }
        let (next_buf, _h, _b) = found?;
        // Some boxes have leading non-box payload (like stsd has 8 bytes
        // of FullBox header + entry_count) — handle them on the way down.
        let cur_path_segment = path[idx];
        buf = match cur_path_segment {
            // moov/trak/mdia/minf/stbl/stsd: its body starts with 4 bytes of
            // version+flags, 4 bytes of entry_count, then the first entry.
            "stsd" => {
                if next_buf.len() < 8 {
                    return None;
                }
                &next_buf[8..]
            }
            _ => next_buf,
        };
        idx += 1;
    }
    Some(buf)
}

/// Find a child box body anywhere inside a sample-entry payload by walking
/// the trailing child-box list (sample entries have a fixed-size header
/// followed by child boxes).
fn find_in_sample_entry<'a>(
    entry_body: &'a [u8],
    header_len: usize,
    kind: &str,
) -> Option<&'a [u8]> {
    let kind_b = kind.as_bytes();
    let mut buf = &entry_body[header_len..];
    while buf.len() >= 8 {
        let size = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let k = &buf[4..8];
        if size < 8 || size > buf.len() {
            return None;
        }
        if k == kind_b {
            return Some(&buf[8..size]);
        }
        buf = &buf[size..];
    }
    None
}

#[test]
fn parse_vps_sps_pps_from_annex_b() {
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    let mut got_vps = false;
    let mut got_sps = false;
    let mut got_pps = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Vps => {
                let vps = parse_vps(&rbsp).expect("VPS parse");
                assert_eq!(vps.vps_video_parameter_set_id, 0);
                got_vps = true;
            }
            NalUnitType::Sps => {
                let sps = parse_sps(&rbsp).expect("SPS parse");
                assert_eq!(sps.pic_width_in_luma_samples, 64);
                assert_eq!(sps.pic_height_in_luma_samples, 64);
                assert_eq!(sps.chroma_format_idc, 1, "expected 4:2:0");
                assert_eq!(sps.bit_depth_y(), 8);
                assert_eq!(sps.bit_depth_c(), 8);
                // x265 may signal Main directly (profile_idc=1) or as
                // Format Range Extensions (profile_idc=4); both encode the
                // same 8-bit 4:2:0 main-tier stream.
                let pidc = sps.profile_tier_level.general_profile_idc;
                assert!(matches!(pidc, 1 | 2 | 4), "unexpected profile_idc={pidc}");
                got_sps = true;
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&rbsp).expect("PPS parse");
                assert_eq!(pps.pps_pic_parameter_set_id, 0);
                got_pps = true;
            }
            _ => {}
        }
    }
    assert!(got_vps, "no VPS found in Annex B fixture");
    assert!(got_sps, "no SPS found in Annex B fixture");
    assert!(got_pps, "no PPS found in Annex B fixture");
}

#[test]
fn parse_slice_header_from_annex_b() {
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    let mut sps_opt = None;
    let mut pps_opt = None;
    let mut found_slice = false;
    for nal in iter_annex_b(&data) {
        let rbsp = extract_rbsp(nal.payload());
        match nal.header.nal_unit_type {
            NalUnitType::Sps => {
                sps_opt = Some(parse_sps(&rbsp).expect("SPS"));
            }
            NalUnitType::Pps => {
                pps_opt = Some(parse_pps(&rbsp).expect("PPS"));
            }
            t if t.is_vcl() => {
                let sps = sps_opt.as_ref().expect("SPS before slice");
                let pps = pps_opt.as_ref().expect("PPS before slice");
                let hdr =
                    parse_slice_segment_header(&rbsp, &nal.header, sps, pps).expect("slice header");
                assert!(hdr.first_slice_segment_in_pic_flag, "first slice expected");
                assert_eq!(hdr.slice_pic_parameter_set_id, pps.pps_pic_parameter_set_id);
                // x265 marks the IDR frame as I slice (slice_type == 2).
                assert_eq!(hdr.slice_type, oxideav_h265::slice::SliceType::I);
                found_slice = true;
                break;
            }
            _ => {}
        }
    }
    assert!(found_slice, "no slice NAL in Annex B fixture");
}

#[test]
fn parse_hvcc_from_mp4_extradata() {
    let Some(data) = read_fixture("/tmp/h265_iframe.mp4") else {
        return;
    };
    // Walk down to the stsd's first sample entry, then find the hvcC
    // child box inside it. Sample entries have a 78-byte VisualSampleEntry
    // preamble before child boxes start.
    let stsd_first_entry = mp4_find_box(&data, &["moov", "trak", "mdia", "minf", "stbl", "stsd"])
        .expect("walk down to stsd's first entry");
    // The "first entry" is itself a box; skip its 8-byte box header, then
    // search inside for hvcC after the 78-byte VisualSampleEntry preamble.
    assert!(
        stsd_first_entry.len() >= 8,
        "stsd entry too short to read box header"
    );
    let entry_kind = &stsd_first_entry[4..8];
    assert!(
        entry_kind == b"hvc1" || entry_kind == b"hev1",
        "expected hvc1/hev1 sample entry, got {:?}",
        std::str::from_utf8(entry_kind).unwrap_or("???")
    );
    let entry_body = &stsd_first_entry[8..];
    let hvcc = find_in_sample_entry(entry_body, 78, "hvcC").expect("hvcC inside sample entry");
    let cfg = parse_hvcc(hvcc).expect("parse hvcC");
    assert_eq!(cfg.configuration_version, 1);
    assert_eq!(cfg.length_size_minus_one, 3, "default 4-byte length prefix");
    // x265 main profile — accept either profile_idc=1 (Main) or 4 (RExt)
    // with the Main compatibility bit set.
    assert!(
        cfg.general_profile_idc == 1 || cfg.general_profile_idc == 4,
        "unexpected hvcC profile_idc={}",
        cfg.general_profile_idc
    );

    // Run the decoder's extradata bootstrap and confirm it picks up VPS/SPS/PPS.
    let mut dec = HevcDecoder::new(oxideav_core::CodecId::new(CODEC_ID_STR));
    dec.consume_extradata(hvcc).expect("consume extradata");
    assert!(!dec.vps.is_empty(), "VPS table populated from hvcC");
    assert!(!dec.sps.is_empty(), "SPS table populated from hvcC");
    assert!(!dec.pps.is_empty(), "PPS table populated from hvcC");
    let sps = dec.sps.values().next().unwrap();
    assert_eq!(sps.pic_width_in_luma_samples, 64);
    assert_eq!(sps.pic_height_in_luma_samples, 64);
}

#[test]
fn nal_header_round_trip_on_fixture() {
    let Some(data) = read_fixture("/tmp/h265.es") else {
        return;
    };
    // Every NAL header in a clean fixture must round-trip from_u8/as_u8.
    for nal in iter_annex_b(&data) {
        let raw = nal.header;
        let v = raw.nal_unit_type.as_u8();
        let parsed = NalUnitType::from_u8(v);
        assert_eq!(parsed.as_u8(), v);
        assert!(raw.nuh_temporal_id_plus1 > 0);
    }
}

#[test]
fn length_prefixed_nal_iter_round_trips() {
    // Synthetic length-prefixed bitstream: VPS NAL(0x40,0x01,0xAA) prefixed
    // by a 4-byte length of 3.
    let data = [0x00u8, 0x00, 0x00, 0x03, 0x40, 0x01, 0xAA];
    let nals = iter_length_prefixed(&data, 4).expect("length-prefixed iter");
    assert_eq!(nals.len(), 1);
    let h: NalHeader = nals[0].header;
    assert_eq!(h.nal_unit_type, NalUnitType::Vps);
}
