//! Encoder integration tests: round-trip via own decoder + ffmpeg interop.

use std::process::Command;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_theora::{
    classify_packet, make_decoder_for_tests, make_encoder_for_tests, FrameType, PacketKind,
};

/// Build a test 64x64 yuv420p frame from the canonical input file generated
/// by ffmpeg's `testsrc` filter.
fn load_testsrc_frame() -> Option<VideoFrame> {
    let raw = std::fs::read("/tmp/theora_in.yuv").ok()?;
    if raw.len() < 64 * 64 + 2 * 32 * 32 {
        return None;
    }
    let y = raw[..64 * 64].to_vec();
    let u = raw[64 * 64..64 * 64 + 32 * 32].to_vec();
    let v = raw[64 * 64 + 32 * 32..64 * 64 + 2 * 32 * 32].to_vec();
    Some(VideoFrame {
        format: PixelFormat::Yuv420P,
        width: 64,
        height: 64,
        pts: Some(0),
        time_base: TimeBase::new(1, 24),
        planes: vec![
            VideoPlane {
                stride: 64,
                data: y,
            },
            VideoPlane {
                stride: 32,
                data: u,
            },
            VideoPlane {
                stride: 32,
                data: v,
            },
        ],
    })
}

fn pixel_match(decoded: &[u8], reference: &[u8], tol: i32) -> f64 {
    let mut matched = 0usize;
    for (a, b) in decoded.iter().zip(reference.iter()) {
        if (*a as i32 - *b as i32).abs() <= tol {
            matched += 1;
        }
    }
    matched as f64 / reference.len() as f64
}

fn build_encoder() -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    make_encoder_for_tests(&params).expect("encoder")
}

fn collect_encoded_packets(enc: &mut dyn Encoder, frame: &VideoFrame) -> Vec<Packet> {
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let mut out = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.push(p);
    }
    out
}

#[test]
fn encode_intra_frame_round_trip_via_own_decoder() {
    let Some(frame) = load_testsrc_frame() else {
        eprintln!("skipped: /tmp/theora_in.yuv missing");
        return;
    };
    let mut enc = build_encoder();
    let pkts = collect_encoded_packets(&mut *enc, &frame);
    assert!(
        pkts.len() >= 4,
        "expected 3 headers + 1 frame, got {} packets",
        pkts.len()
    );
    // First three packets must be headers.
    assert_eq!(
        classify_packet(&pkts[0].data).unwrap(),
        PacketKind::Identification
    );
    assert_eq!(classify_packet(&pkts[1].data).unwrap(), PacketKind::Comment);
    assert_eq!(classify_packet(&pkts[2].data).unwrap(), PacketKind::Setup);
    assert_eq!(
        classify_packet(&pkts[3].data).unwrap(),
        PacketKind::Frame(FrameType::Intra)
    );

    // Decode through our own decoder.
    let extradata = enc.output_params().extradata.clone();
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 24), pkts[3].data.clone());
    dec.send_packet(&pkt).expect("send packet");
    let decoded = match dec.receive_frame().expect("receive frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.format, PixelFormat::Yuv420P);

    // Compare per-plane against original.
    let mut all_decoded = Vec::with_capacity(64 * 64 + 2 * 32 * 32);
    for p in &decoded.planes {
        all_decoded.extend_from_slice(&p.data);
    }
    let mut all_orig = Vec::with_capacity(64 * 64 + 2 * 32 * 32);
    for p in &frame.planes {
        all_orig.extend_from_slice(&p.data);
    }
    let pct = pixel_match(&all_decoded, &all_orig, 12);
    let pct_strict = pixel_match(&all_decoded, &all_orig, 3);
    eprintln!(
        "round-trip pixel match: {:.2}% (tol=12), {:.2}% (tol=3); frame packet {} bytes",
        pct * 100.0,
        pct_strict * 100.0,
        pkts[3].data.len()
    );
    // 99% target with looser tolerance to allow lossy quantisation noise.
    assert!(pct >= 0.99, "round-trip match {pct:.4} < 0.99");
}

/// Mux our headers + frame into a minimal Ogg container and ask ffmpeg to
/// decode it. Compare the decoded YUV against the input.
#[test]
fn ffmpeg_can_decode_our_intra_frame() {
    let Some(frame) = load_testsrc_frame() else {
        eprintln!("skipped: /tmp/theora_in.yuv missing");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipped: ffmpeg not on PATH");
        return;
    }

    let mut enc = build_encoder();
    let pkts = collect_encoded_packets(&mut *enc, &frame);
    assert!(pkts.len() >= 4);

    // Compose a one-stream Ogg file. Three header packets each on their own
    // page; then the frame packet on a page. Use a single serialno.
    let serial = 0x1234_5678u32;
    let mut ogg = Vec::new();
    write_ogg_page(
        &mut ogg,
        serial,
        0, // page seq
        0, // granule
        &[&pkts[0].data],
        true,  // bos
        false, // eos
    );
    write_ogg_page(
        &mut ogg,
        serial,
        1,
        0,
        &[&pkts[1].data, &pkts[2].data],
        false,
        false,
    );
    // Frame packet: granulepos = 1 << kfgshift (kfgshift=6 in our id header)
    // for a keyframe.
    let granule: u64 = 1u64 << 6;
    write_ogg_page(&mut ogg, serial, 2, granule, &[&pkts[3].data], false, true);

    std::fs::write("/tmp/our_output.ogv", &ogg).expect("write ogg");
    let _ = std::fs::remove_file("/tmp/check.yuv");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            "/tmp/our_output.ogv",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "/tmp/check.yuv",
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed to decode our stream");

    let decoded = std::fs::read("/tmp/check.yuv").expect("read check.yuv");
    let mut orig = Vec::new();
    for p in &frame.planes {
        orig.extend_from_slice(&p.data);
    }
    assert_eq!(
        decoded.len(),
        orig.len(),
        "ffmpeg-decoded YUV size ({}) != input size ({})",
        decoded.len(),
        orig.len()
    );
    let pct = pixel_match(&decoded, &orig, 12);
    let pct_strict = pixel_match(&decoded, &orig, 3);
    let file_size = std::fs::metadata("/tmp/our_output.ogv")
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "ffmpeg interop pixel match: {:.2}% (tol=12), {:.2}% (tol=3); Ogg {} bytes, frame {} bytes",
        pct * 100.0,
        pct_strict * 100.0,
        file_size,
        pkts[3].data.len()
    );
    assert!(pct >= 0.95, "ffmpeg decode match {pct:.4} < 0.95");
}

// --- minimal Ogg page writer ---------------------------------------------

fn ogg_lacing(packets: &[&[u8]]) -> Vec<u8> {
    let mut lacing = Vec::new();
    for p in packets {
        let mut sz = p.len();
        loop {
            if sz >= 255 {
                lacing.push(255);
                sz -= 255;
                if sz == 0 {
                    // Even multiple of 255 needs a trailing 0 to terminate.
                    lacing.push(0);
                    break;
                }
            } else {
                lacing.push(sz as u8);
                break;
            }
        }
    }
    lacing
}

fn write_ogg_page(
    out: &mut Vec<u8>,
    serial: u32,
    page_seq: u32,
    granulepos: u64,
    packets: &[&[u8]],
    bos: bool,
    eos: bool,
) {
    let lacing = ogg_lacing(packets);
    assert!(
        lacing.len() <= 255,
        "page payload exceeds 255 segments (need to split)"
    );
    let mut header = Vec::with_capacity(27 + lacing.len());
    header.extend_from_slice(b"OggS");
    header.push(0); // version
    let mut flags = 0u8;
    if bos {
        flags |= 0x02;
    }
    if eos {
        flags |= 0x04;
    }
    header.push(flags);
    header.extend_from_slice(&granulepos.to_le_bytes());
    header.extend_from_slice(&serial.to_le_bytes());
    header.extend_from_slice(&page_seq.to_le_bytes());
    header.extend_from_slice(&[0u8; 4]); // CRC placeholder
    header.push(lacing.len() as u8);
    header.extend_from_slice(&lacing);
    let mut body = Vec::new();
    for p in packets {
        body.extend_from_slice(p);
    }
    let mut full = header.clone();
    full.extend_from_slice(&body);
    let crc = ogg_crc32(&full);
    full[22..26].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&full);
}

const OGG_CRC_POLY: u32 = 0x04C1_1DB7;

fn ogg_crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        let mut r = (i as u32) << 24;
        for _ in 0..8 {
            if r & 0x8000_0000 != 0 {
                r = (r << 1) ^ OGG_CRC_POLY;
            } else {
                r <<= 1;
            }
        }
        *slot = r;
    }
    t
}

fn ogg_crc32(data: &[u8]) -> u32 {
    let table = ogg_crc_table();
    let mut crc = 0u32;
    for &b in data {
        let idx = ((crc >> 24) as u8 ^ b) as usize;
        crc = (crc << 8) ^ table[idx];
    }
    crc
}
