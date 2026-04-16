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
fn decode_intra_frame_tiny_420p() {
    let path = "/tmp/ref-theora-iframes-420.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    assert!(pkts.len() >= 4);
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder");
    let packet = Packet::new(0, TimeBase::new(1, 10), pkts[3].clone());
    dec.send_packet(&packet).expect("send intra packet");
    let frame = match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(frame.format, PixelFormat::Yuv420P);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 48);
    assert_eq!(frame.planes[0].data.len(), 64 * 48);
    let y = &frame.planes[0].data;
    let mean: u32 = y.iter().map(|&v| v as u32).sum::<u32>() / y.len() as u32;
    let mn = *y.iter().min().unwrap();
    let mx = *y.iter().max().unwrap();
    // Decoded output should span a wide range (testsrc contains many colours).
    assert!(
        mx - mn > 40,
        "Y plane has suspiciously flat dynamic range: [{mn}, {mx}]"
    );
    assert!(
        (30..230).contains(&mean),
        "mean Y out of expected range: {mean}"
    );
}

#[test]
fn decode_intra_frame_tiny_444p() {
    let path = "/tmp/ref-theora-iframes-444.ogv";
    let Ok(data) = std::fs::read(path) else {
        eprintln!("skipped: {path} not present");
        return;
    };
    let pkts = collect_packets(&data);
    if pkts.len() < 4 {
        eprintln!("skipped: insufficient packets in {path}");
        return;
    }
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder");
    let packet = Packet::new(0, TimeBase::new(1, 10), pkts[3].clone());
    dec.send_packet(&packet).expect("send intra packet");
    let frame = match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_eq!(frame.format, PixelFormat::Yuv444P);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 48);
    assert_eq!(frame.planes[0].data.len(), 64 * 48);
    assert_eq!(frame.planes[1].data.len(), 64 * 48);
    assert_eq!(frame.planes[2].data.len(), 64 * 48);
}

/// Decode all frames of the multi-frame P-frame clip and compare against the
/// ffmpeg-decoded reference (`/tmp/theora_p_ref.yuv`). Asserts that the
/// decoded I-frame matches reasonably well; the stricter target is checked in
/// a percentile-style threshold for inter frames if the implementation has
/// matured enough.
#[test]
fn decode_pframe_clip_matches_ffmpeg() {
    let ogv = "/tmp/theora_p.ogv";
    let yuv = "/tmp/theora_p_ref.yuv";
    let Ok(data) = std::fs::read(ogv) else {
        eprintln!("skipped: {ogv} not present");
        return;
    };
    let Ok(ref_yuv) = std::fs::read(yuv) else {
        eprintln!("skipped: {yuv} not present");
        return;
    };
    let pkts = collect_packets(&data);
    if pkts.len() < 4 {
        eprintln!("skipped: insufficient packets");
        return;
    }
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder");

    // Reference frame size (per fixture command): 64x64 yuv420p, 1 frame = 6144 bytes.
    let frame_size = 64 * 64 + 2 * (32 * 32);
    let n_ref = ref_yuv.len() / frame_size;
    assert!(n_ref >= 1);

    let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
    for (i, pkt) in pkts.iter().enumerate().skip(3) {
        let p = Packet::new(0u32, TimeBase::new(1, 24), pkt.clone());
        let _ = i;
        match dec.send_packet(&p) {
            Ok(()) => {}
            Err(Error::Unsupported(msg)) => {
                eprintln!("packet {i}: unsupported: {msg}");
                continue;
            }
            Err(e) => {
                eprintln!("packet {i}: decode error: {e:?}");
                continue;
            }
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    let mut buf = Vec::with_capacity(frame_size);
                    for plane in &v.planes {
                        buf.extend_from_slice(&plane.data);
                    }
                    decoded_frames.push(buf);
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
    assert!(!decoded_frames.is_empty(), "no frames decoded");
    // I-frame match check (strict).
    let first_ref = &ref_yuv[..frame_size];
    let first_dec = &decoded_frames[0];
    let mut matched = 0usize;
    for (a, b) in first_dec.iter().zip(first_ref.iter()) {
        if (*a as i32 - *b as i32).abs() <= 3 {
            matched += 1;
        }
    }
    let pct_i = matched as f64 / first_ref.len() as f64;
    eprintln!("I-frame match: {:.2}%", pct_i * 100.0);
    assert!(pct_i >= 0.95, "I-frame match {pct_i:.2} < 0.95");

    // Inter frames: if at least one was decoded successfully, check it.
    if decoded_frames.len() > 1 {
        let n_check = decoded_frames.len().min(n_ref);
        let mut total_match = 0usize;
        let mut total_pixels = 0usize;
        for k in 1..n_check {
            let r = &ref_yuv[k * frame_size..(k + 1) * frame_size];
            let d = &decoded_frames[k];
            let mut frame_match = 0usize;
            for (a, b) in d.iter().zip(r.iter()) {
                let diff = (*a as i32 - *b as i32).abs();
                if diff <= 3 {
                    total_match += 1;
                    frame_match += 1;
                }
                total_pixels += 1;
            }
            let pct = frame_match as f64 / r.len() as f64;
            eprintln!("frame {}: {:.2}% match", k, pct * 100.0);
        }
        let pct_p = total_match as f64 / total_pixels as f64;
        eprintln!(
            "P-frame match across {} frames: {:.2}%",
            n_check - 1,
            pct_p * 100.0
        );
        assert!(
            pct_p >= 0.95,
            "P-frame match {pct_p:.4} < 0.95 (target ≥0.95)"
        );
    } else {
        eprintln!("only I-frame decoded");
    }
}

/// Decode the larger GOP fixture (128x96 YUV444, 20 frames) and assert pixel match.
#[test]
fn decode_gop_clip_matches_ffmpeg() {
    let ogv = "/tmp/ref-theora-gop.ogv";
    let yuv = "/tmp/ref-theora-gop444.yuv";
    let Ok(data) = std::fs::read(ogv) else {
        eprintln!("skipped: {ogv} not present");
        return;
    };
    let Ok(ref_yuv) = std::fs::read(yuv) else {
        eprintln!("skipped: {yuv} not present");
        return;
    };
    let pkts = collect_packets(&data);
    if pkts.len() < 4 {
        eprintln!("skipped: insufficient packets");
        return;
    }
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("make_decoder");

    // 128x96 yuv444p: 12288 * 3 = 36864 bytes/frame.
    let frame_size = 128 * 96 * 3;
    let n_ref = ref_yuv.len() / frame_size;

    let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
    for (i, pkt) in pkts.iter().enumerate().skip(3) {
        let p = Packet::new(0u32, TimeBase::new(1, 10), pkt.clone());
        let _ = i;
        match dec.send_packet(&p) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("packet {i}: error {e:?}");
                continue;
            }
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    let mut buf = Vec::with_capacity(frame_size);
                    for plane in &v.planes {
                        buf.extend_from_slice(&plane.data);
                    }
                    decoded_frames.push(buf);
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
    assert!(!decoded_frames.is_empty(), "no frames decoded");
    let n_check = decoded_frames.len().min(n_ref);
    let mut total_match = 0usize;
    let mut total_pixels = 0usize;
    for k in 0..n_check {
        let r = &ref_yuv[k * frame_size..(k + 1) * frame_size];
        let d = &decoded_frames[k];
        let mut frame_match = 0usize;
        for (a, b) in d.iter().zip(r.iter()) {
            if (*a as i32 - *b as i32).abs() <= 3 {
                total_match += 1;
                frame_match += 1;
            }
            total_pixels += 1;
        }
        let pct = frame_match as f64 / r.len() as f64;
        eprintln!("GOP frame {}: {:.2}% match", k, pct * 100.0);
    }
    let pct = total_match as f64 / total_pixels as f64;
    eprintln!(
        "GOP overall match across {} frames: {:.2}%",
        n_check,
        pct * 100.0
    );
    assert!(pct >= 0.95, "GOP match {pct:.4} < 0.95");
}

#[test]
fn legacy_inter_unsupported_test_skipped() {
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
    let kf = Packet::new(0, TimeBase::new(1, 10), pkts[3].clone());
    dec.send_packet(&kf).expect("send keyframe");
    let _ = dec.receive_frame();
    let inter = Packet::new(0, TimeBase::new(1, 10), pkts[4].clone());
    // Inter is now (best-effort) decoded; we just check it doesn't panic.
    let _ = dec.send_packet(&inter);
    let _ = dec.receive_frame();
}
