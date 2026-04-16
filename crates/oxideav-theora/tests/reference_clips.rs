//! Integration tests against ffmpeg-generated Theora Ogg files.
//!
//! These tests read `/tmp/ref-theora-*.ogv` produced by this worktree's setup
//! step. If the files are missing, the tests print a skip message and return
//! without failing, so the suite still passes on machines that didn't run the
//! setup commands.

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, TimeBase};
use oxideav_theora::{
    classify_packet, codec_parameters_from_identification, make_decoder_for_tests,
    parse_headers_from_extradata, FrameType, PacketKind,
};

/// Walk a raw Ogg file byte-by-byte, emitting complete logical packets (after
/// Xiph-style lacing reassembly). Good enough for short files: we don't
/// bother with multi-stream demultiplexing — theora ffmpeg files have only a
/// video track.
fn collect_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i + 27 <= data.len() {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let n_segs = data[i + 26] as usize;
        if i + 27 + n_segs > data.len() {
            break;
        }
        let lacing = &data[i + 27..i + 27 + n_segs];
        let mut off = i + 27 + n_segs;
        for &lv in lacing {
            if off + lv as usize > data.len() {
                break;
            }
            buf.extend_from_slice(&data[off..off + lv as usize]);
            off += lv as usize;
            if lv < 255 {
                out.push(std::mem::take(&mut buf));
            }
        }
        i = off;
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn xiph_lace(packets: &[&[u8]]) -> Vec<u8> {
    let n = packets.len();
    let mut out = Vec::new();
    out.push((n - 1) as u8);
    for p in &packets[..n - 1] {
        let mut sz = p.len();
        while sz >= 255 {
            out.push(255);
            sz -= 255;
        }
        out.push(sz as u8);
    }
    for p in packets {
        out.extend_from_slice(p);
    }
    out
}

#[test]
fn parses_iframes_clip_headers() {
    let path = "/tmp/ref-theora-iframes-420.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    assert!(
        pkts.len() >= 3,
        "expected at least 3 packets, got {}",
        pkts.len()
    );
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let h = parse_headers_from_extradata(&extradata).expect("parse headers");
    assert_eq!(h.identification.picw, 64);
    assert_eq!(h.identification.pich, 48);
    assert_eq!(h.identification.pf, oxideav_theora::PixelFormat::Yuv420);
    // frame_rate = 10/1 by construction.
    assert_eq!(h.identification.frn, 10);
    assert_eq!(h.identification.frd, 1);
    // Setup header yielded 80 Huffman trees.
    assert_eq!(h.setup.huffs.len(), 80);
    // At least some trees should have more than one node.
    let non_trivial = h.setup.huffs.iter().filter(|t| t.nodes.len() > 1).count();
    assert!(non_trivial > 0, "all Huffman trees are 1-node?");
}

#[test]
fn codec_parameters_from_identification_populates_fields() {
    let path = "/tmp/ref-theora-iframes-420.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let h = parse_headers_from_extradata(&extradata).expect("parse headers");
    let params = codec_parameters_from_identification(&h.identification);
    assert_eq!(params.width, Some(64));
    assert_eq!(params.height, Some(48));
    assert_eq!(params.pixel_format, Some(PixelFormat::Yuv420P));
}

#[test]
fn classifies_frame_packets() {
    let path = "/tmp/ref-theora-gop.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    assert!(pkts.len() >= 5, "need headers + at least 2 frames");
    // Packet 3 is the first frame, must be keyframe (intra).
    let k3 = classify_packet(&pkts[3]).expect("classify pkt 3");
    assert_eq!(k3, PacketKind::Frame(FrameType::Intra));
    // Find at least one inter frame somewhere after.
    let any_inter = pkts.iter().skip(4).any(|p| {
        classify_packet(p)
            .map(|k| k == PacketKind::Frame(FrameType::Inter))
            .unwrap_or(false)
    });
    assert!(any_inter, "expected an inter frame in the GOP clip");
}

#[test]
fn intra_frame_decode_is_todo() {
    let path = "/tmp/ref-theora-gop.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder");
    let packet = Packet::new(0, TimeBase::new(1, 10), pkts[3].clone());
    match dec.send_packet(&packet) {
        Err(Error::Unsupported(msg)) => {
            assert!(
                msg.contains("intra-frame block decode") || msg.contains("inter-frame"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected Unsupported (intra-frame decode follow-up), got {other:?}"),
    }

    // The second data packet (index 4) should be an inter frame rejected with
    // the dedicated follow-up message.
    let packet_inter = Packet::new(0, TimeBase::new(1, 10), pkts[4].clone());
    let _ = dec.flush();
    // Re-create decoder to reset state.
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder again");
    match dec.send_packet(&packet_inter) {
        Err(Error::Unsupported(msg)) => {
            // Either "inter-frame decode" or the generic intra-block message
            // depending on which bit hits first.
            assert!(msg.contains("follow-up"), "unexpected message: {msg}");
        }
        Ok(()) => {
            // If we ever accept the packet silently, receive_frame should
            // still not produce a real frame yet.
            match dec.receive_frame() {
                Err(_) => {}
                Ok(Frame::Video(_)) => panic!("unexpected video frame from inter packet"),
                Ok(_) => panic!("unexpected non-video frame"),
            }
        }
        other => panic!("unexpected send_packet result: {other:?}"),
    }
}
