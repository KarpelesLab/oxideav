//! Encoder integration tests: round-trip via own decoder + ffmpeg interop.

use std::process::Command;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_theora::{
    classify_packet, encoder::make_encoder_with_options, encoder::EncoderOptions,
    make_decoder_for_tests, make_encoder_for_tests, FrameType, PacketKind,
};

/// Build a test 64x64 yuv420p frame from the canonical input file generated
/// by ffmpeg's `testsrc` filter.
fn load_testsrc_frame() -> Option<VideoFrame> {
    let raw = std::fs::read(std::env::temp_dir().join("theora_in.yuv")).ok()?;
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

/// Build an encoder pinned to the legacy keyint=12 schedule (the small-clip
/// test below exercises that cadence explicitly).
fn build_encoder_keyint12() -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    let opts = EncoderOptions {
        keyint: 12,
        ..Default::default()
    };
    make_encoder_with_options(&params, opts).expect("encoder")
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
        eprintln!("skipped: theora_in.yuv missing from temp dir");
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
        eprintln!("skipped: theora_in.yuv missing from temp dir");
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

    let tmp = std::env::temp_dir();
    let ogv_path = tmp.join("our_output.ogv");
    let check_path = tmp.join("check.yuv");
    std::fs::write(&ogv_path, &ogg).expect("write ogg");
    let _ = std::fs::remove_file(&check_path);

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            ogv_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            check_path.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed to decode our stream");

    let decoded = std::fs::read(&check_path).expect("read check.yuv");
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
    let file_size = std::fs::metadata(&ogv_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "ffmpeg interop pixel match: {:.2}% (tol=12), {:.2}% (tol=3); Ogg {} bytes, frame {} bytes",
        pct * 100.0,
        pct_strict * 100.0,
        file_size,
        pkts[3].data.len()
    );
    assert!(pct >= 0.95, "ffmpeg decode match {pct:.4} < 0.95");
}

// --- P-frame round-trip --------------------------------------------------

/// Load all 24 frames of `theora_p_in.yuv` from the temp dir (raw 64x64 yuv420p, 24 frames).
fn load_pframe_input() -> Option<Vec<VideoFrame>> {
    let raw = std::fs::read(std::env::temp_dir().join("theora_p_in.yuv")).ok()?;
    let frame_size = 64 * 64 + 2 * 32 * 32;
    if raw.len() < frame_size {
        return None;
    }
    let n = raw.len() / frame_size;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * frame_size;
        let y = raw[off..off + 64 * 64].to_vec();
        let u = raw[off + 64 * 64..off + 64 * 64 + 32 * 32].to_vec();
        let v = raw[off + 64 * 64 + 32 * 32..off + frame_size].to_vec();
        out.push(VideoFrame {
            format: PixelFormat::Yuv420P,
            width: 64,
            height: 64,
            pts: Some(i as i64),
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
        });
    }
    Some(out)
}

#[test]
fn encode_pframe_clip_round_trip_via_own_decoder() {
    let Some(frames) = load_pframe_input() else {
        eprintln!("skipped: theora_p_in.yuv missing from temp dir");
        return;
    };
    assert!(!frames.is_empty());

    let mut enc = build_encoder_keyint12();
    let mut frame_pkts: Vec<Packet> = Vec::new();
    let mut header_pkts: Vec<Packet> = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        while let Ok(p) = enc.receive_packet() {
            if p.flags.header {
                header_pkts.push(p);
            } else {
                frame_pkts.push(p);
            }
        }
    }
    assert_eq!(header_pkts.len(), 3);
    assert_eq!(frame_pkts.len(), frames.len());

    // Verify packet kinds: frame 0 must be intra; rest follow keyint=12 schedule.
    for (i, p) in frame_pkts.iter().enumerate() {
        let k = classify_packet(&p.data).unwrap();
        if i % 12 == 0 {
            assert_eq!(
                k,
                PacketKind::Frame(FrameType::Intra),
                "frame {i} should be I"
            );
        } else {
            assert_eq!(
                k,
                PacketKind::Frame(FrameType::Inter),
                "frame {i} should be P"
            );
        }
    }

    // Decode through our own decoder.
    let extradata = enc.output_params().extradata.clone();
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = make_decoder_for_tests(&params).expect("decoder");

    let mut decoded_frames: Vec<VideoFrame> = Vec::new();
    for p in &frame_pkts {
        let pkt = Packet::new(0, TimeBase::new(1, 24), p.data.clone());
        dec.send_packet(&pkt).expect("send packet");
        while let Ok(f) = dec.receive_frame() {
            if let Frame::Video(v) = f {
                decoded_frames.push(v);
            }
        }
    }
    assert_eq!(decoded_frames.len(), frames.len());

    // Per-frame match.
    let mut total_match: usize = 0;
    let mut total_pixels: usize = 0;
    let mut total_bytes: usize = 0;
    for (i, (orig, dec)) in frames.iter().zip(decoded_frames.iter()).enumerate() {
        let mut o = Vec::new();
        let mut d = Vec::new();
        for p in &orig.planes {
            o.extend_from_slice(&p.data);
        }
        for p in &dec.planes {
            d.extend_from_slice(&p.data);
        }
        let mut m = 0usize;
        for (a, b) in d.iter().zip(o.iter()) {
            if (*a as i32 - *b as i32).abs() <= 16 {
                m += 1;
            }
        }
        total_match += m;
        total_pixels += o.len();
        total_bytes += frame_pkts[i].data.len();
        let pct = m as f64 / o.len() as f64;
        eprintln!(
            "round-trip frame {i}: {:.2}% match, {} bytes",
            pct * 100.0,
            frame_pkts[i].data.len()
        );
    }
    let overall = total_match as f64 / total_pixels as f64;
    eprintln!(
        "P-frame round-trip overall match (tol=16): {:.2}%; total bytes {}",
        overall * 100.0,
        total_bytes
    );
    assert!(overall >= 0.95, "overall match {overall:.4} < 0.95");
}

#[test]
fn ffmpeg_can_decode_our_pframe_clip() {
    let Some(frames) = load_pframe_input() else {
        eprintln!("skipped: theora_p_in.yuv missing from temp dir");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipped: ffmpeg not on PATH");
        return;
    }

    let mut enc = build_encoder();
    let mut frame_pkts: Vec<Packet> = Vec::new();
    let mut header_pkts: Vec<Packet> = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        while let Ok(p) = enc.receive_packet() {
            if p.flags.header {
                header_pkts.push(p);
            } else {
                frame_pkts.push(p);
            }
        }
    }

    // Pack into a minimal Ogg file. Each frame on its own page, sequential
    // page numbers and Theora-style granule positions.
    let serial = 0xDEADBEEFu32;
    let mut ogg = Vec::new();
    let mut seq = 0u32;
    write_ogg_page(
        &mut ogg,
        serial,
        seq,
        0,
        &[&header_pkts[0].data],
        true,
        false,
    );
    seq += 1;
    write_ogg_page(
        &mut ogg,
        serial,
        seq,
        0,
        &[&header_pkts[1].data, &header_pkts[2].data],
        false,
        false,
    );
    seq += 1;

    // Compute granulepos per Theora rules: granulepos =
    // (kf_index << kfgshift) + (current - kf_index). kfgshift=6.
    let kfgshift = 6u32;
    let mut last_kf_index: u64 = 0;
    for (i, p) in frame_pkts.iter().enumerate() {
        if p.flags.keyframe {
            last_kf_index = i as u64 + 1; // 1-based
        }
        let frames_since_kf = (i as u64 + 1) - last_kf_index;
        let granule = (last_kf_index << kfgshift) | frames_since_kf;
        let is_eos = i == frame_pkts.len() - 1;
        write_ogg_page(&mut ogg, serial, seq, granule, &[&p.data], false, is_eos);
        seq += 1;
    }

    let tmp = std::env::temp_dir();
    let ogv_path = tmp.join("our_pframe_output.ogv");
    let check_path = tmp.join("theora_p_check.yuv");
    std::fs::write(&ogv_path, &ogg).expect("write ogg");
    let _ = std::fs::remove_file(&check_path);

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            ogv_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            check_path.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode our P-frame stream"
    );

    let decoded = std::fs::read(&check_path).expect("read theora_p_check.yuv");
    let frame_size = 64 * 64 + 2 * 32 * 32;
    let n_dec = decoded.len() / frame_size;
    let n_check = n_dec.min(frames.len());
    let mut total_match = 0usize;
    let mut total_pixels = 0usize;
    for i in 0..n_check {
        let mut orig = Vec::new();
        for p in &frames[i].planes {
            orig.extend_from_slice(&p.data);
        }
        let dec = &decoded[i * frame_size..(i + 1) * frame_size];
        let mut m = 0usize;
        for (a, b) in dec.iter().zip(orig.iter()) {
            if (*a as i32 - *b as i32).abs() <= 16 {
                m += 1;
            }
        }
        total_match += m;
        total_pixels += orig.len();
        let pct = m as f64 / orig.len() as f64;
        eprintln!("ffmpeg-decoded frame {i}: {:.2}% match", pct * 100.0);
    }
    let overall = total_match as f64 / total_pixels as f64;
    let file_size = std::fs::metadata(&ogv_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "ffmpeg interop P-frame match: {:.2}% (tol=16); Ogg {} bytes",
        overall * 100.0,
        file_size
    );
    assert!(overall >= 0.90, "ffmpeg P-frame match {overall:.4} < 0.90");
}

// --- PSNR-based P-frame compression test ---------------------------------

/// Synthetic 30-frame "moving test pattern" clip: a Y gradient with a solid
/// rectangle that shifts right by 1 pixel per frame. Chroma is neutral. The
/// motion is purely horizontal translation so a decent motion estimator
/// finds near-zero residuals.
fn generate_moving_pattern_clip(n_frames: u32) -> Vec<VideoFrame> {
    let w: u32 = 64;
    let h: u32 = 64;
    let mut out = Vec::with_capacity(n_frames as usize);
    for f in 0..n_frames {
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h {
            for i in 0..w {
                // Diagonal gradient base.
                y[(j * w + i) as usize] = ((i + j) * 2).clamp(0, 255) as u8;
            }
        }
        // Solid white rectangle (16x16) moving horizontally.
        let rx = (4 + f) % (w - 16);
        let ry = 20u32;
        for j in 0..16u32 {
            for i in 0..16u32 {
                y[((ry + j) * w + (rx + i)) as usize] = 220;
            }
        }
        // Neutral chroma planes.
        let u = vec![128u8; (w / 2 * h / 2) as usize];
        let v = vec![128u8; (w / 2 * h / 2) as usize];
        out.push(VideoFrame {
            format: PixelFormat::Yuv420P,
            width: w,
            height: h,
            pts: Some(f as i64),
            time_base: TimeBase::new(1, 30),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: u,
                },
                VideoPlane {
                    stride: (w / 2) as usize,
                    data: v,
                },
            ],
        });
    }
    out
}

/// PSNR (dB) between two equally-sized 8-bit planes.
fn psnr_db(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        sse += d * d;
    }
    let mse = sse / a.len() as f64;
    if mse <= 0.0 {
        return 99.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn build_enc_with_opts(opts: oxideav_theora::encoder::EncoderOptions) -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.media_type = MediaType::Video;
    params.width = Some(64);
    params.height = Some(64);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    oxideav_theora::encoder::make_encoder_with_options(&params, opts).expect("encoder")
}

/// Encode `frames` with the given options, decode with our decoder, and
/// return (decoded_frames, per-frame packet sizes, per-frame kinds).
fn encode_decode_clip(
    frames: &[VideoFrame],
    opts: oxideav_theora::encoder::EncoderOptions,
) -> (Vec<VideoFrame>, Vec<usize>, Vec<FrameType>) {
    let mut enc = build_enc_with_opts(opts);
    let mut frame_pkts: Vec<Packet> = Vec::new();
    let mut header_pkts: Vec<Packet> = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        while let Ok(p) = enc.receive_packet() {
            if p.flags.header {
                header_pkts.push(p);
            } else {
                frame_pkts.push(p);
            }
        }
    }
    assert_eq!(header_pkts.len(), 3);
    assert_eq!(frame_pkts.len(), frames.len());
    let sizes: Vec<usize> = frame_pkts.iter().map(|p| p.data.len()).collect();
    let kinds: Vec<FrameType> = frame_pkts
        .iter()
        .map(|p| match classify_packet(&p.data).unwrap() {
            PacketKind::Frame(t) => t,
            _ => panic!("expected frame packet"),
        })
        .collect();

    let extradata = enc.output_params().extradata.clone();
    let mut p2 = CodecParameters::video(CodecId::new("theora"));
    p2.extradata = extradata;
    let mut dec = make_decoder_for_tests(&p2).expect("decoder");

    let mut decoded: Vec<VideoFrame> = Vec::new();
    for p in &frame_pkts {
        let pkt = Packet::new(0, TimeBase::new(1, 30), p.data.clone());
        dec.send_packet(&pkt).expect("send packet");
        while let Ok(f) = dec.receive_frame() {
            if let Frame::Video(v) = f {
                decoded.push(v);
            }
        }
    }
    (decoded, sizes, kinds)
}

#[test]
fn pframe_moving_pattern_psnr_and_bitrate() {
    let frames = generate_moving_pattern_clip(30);
    assert_eq!(frames.len(), 30);

    // Run 1: default GOP (keyint=30 → 1 I-frame, 29 P-frames).
    let opts_p = oxideav_theora::encoder::EncoderOptions {
        keyint: 30,
        ..Default::default()
    };
    let (decoded_p, sizes_p, kinds_p) = encode_decode_clip(&frames, opts_p);
    let total_p: usize = sizes_p.iter().sum();

    // Run 2: all-keyframe (keyint=1).
    let opts_i = oxideav_theora::encoder::EncoderOptions {
        keyint: 1,
        ..Default::default()
    };
    let (_decoded_i, sizes_i, kinds_i) = encode_decode_clip(&frames, opts_i);
    let total_i: usize = sizes_i.iter().sum();

    // Sanity: run 1 has exactly one intra (the first), run 2 has all intra.
    assert_eq!(kinds_p[0], FrameType::Intra);
    for k in &kinds_p[1..] {
        assert_eq!(*k, FrameType::Inter, "expected P-frames after first I");
    }
    for k in &kinds_i {
        assert_eq!(*k, FrameType::Intra);
    }

    // PSNR on P-frames only: decoded_p[1..] vs original luma planes.
    let mut psnr_sum = 0.0f64;
    let mut psnr_count = 0u32;
    let mut min_psnr = f64::INFINITY;
    for (i, (orig, dec)) in frames.iter().zip(decoded_p.iter()).enumerate().skip(1) {
        let p = psnr_db(&orig.planes[0].data, &dec.planes[0].data);
        psnr_sum += p;
        psnr_count += 1;
        min_psnr = min_psnr.min(p);
        eprintln!("P-frame {i}: PSNR(Y) = {p:.2} dB, {} bytes", sizes_p[i]);
    }
    let avg_psnr = psnr_sum / psnr_count as f64;
    eprintln!(
        "moving-pattern clip: avg P-frame Y-PSNR = {:.2} dB, min = {:.2} dB",
        avg_psnr, min_psnr
    );
    eprintln!(
        "bitrate compare: keyint=30 total {} bytes (I={}, P={}), keyint=1 total {} bytes ({}x)",
        total_p,
        sizes_p[0],
        total_p - sizes_p[0],
        total_i,
        total_i as f64 / total_p as f64
    );

    assert!(
        min_psnr > 28.0,
        "P-frame PSNR too low: min={min_psnr:.2} dB (target > 28 dB)"
    );
    assert!(
        total_p < total_i,
        "P-frame bitstream ({total_p} B) should be smaller than all-keyframe ({total_i} B)"
    );
}

#[test]
fn pframe_moving_pattern_ffmpeg_interop() {
    if std::process::Command::new("/usr/bin/ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("skipped: /usr/bin/ffmpeg not available");
        return;
    }
    let frames = generate_moving_pattern_clip(30);
    let opts = oxideav_theora::encoder::EncoderOptions {
        keyint: 30,
        ..Default::default()
    };
    let mut enc = build_enc_with_opts(opts);
    let mut frame_pkts: Vec<Packet> = Vec::new();
    let mut header_pkts: Vec<Packet> = Vec::new();
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
        while let Ok(p) = enc.receive_packet() {
            if p.flags.header {
                header_pkts.push(p);
            } else {
                frame_pkts.push(p);
            }
        }
    }

    // Mux to Ogg identical to the existing ffmpeg_can_decode_our_pframe_clip
    // test, but shorter.
    let serial = 0xAABB_CCDDu32;
    let mut ogg = Vec::new();
    let mut seq = 0u32;
    write_ogg_page(
        &mut ogg,
        serial,
        seq,
        0,
        &[&header_pkts[0].data],
        true,
        false,
    );
    seq += 1;
    write_ogg_page(
        &mut ogg,
        serial,
        seq,
        0,
        &[&header_pkts[1].data, &header_pkts[2].data],
        false,
        false,
    );
    seq += 1;

    let kfgshift = 6u32;
    let mut last_kf_index: u64 = 0;
    for (i, p) in frame_pkts.iter().enumerate() {
        if p.flags.keyframe {
            last_kf_index = i as u64 + 1;
        }
        let frames_since_kf = (i as u64 + 1) - last_kf_index;
        let granule = (last_kf_index << kfgshift) | frames_since_kf;
        let is_eos = i == frame_pkts.len() - 1;
        write_ogg_page(&mut ogg, serial, seq, granule, &[&p.data], false, is_eos);
        seq += 1;
    }

    let tmp = std::env::temp_dir();
    let ogv_path = tmp.join("oxideav_moving_pattern.ogv");
    let check_path = tmp.join("oxideav_moving_pattern_check.yuv");
    std::fs::write(&ogv_path, &ogg).expect("write ogv");
    let _ = std::fs::remove_file(&check_path);

    let status = std::process::Command::new("/usr/bin/ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            ogv_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            check_path.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed to decode moving pattern");

    let decoded = std::fs::read(&check_path).expect("read yuv");
    let frame_size = 64 * 64 + 2 * 32 * 32;
    // ffmpeg output must contain the same number of frames as we emitted.
    let n_frames_out = decoded.len() / frame_size;
    assert_eq!(
        n_frames_out,
        frames.len(),
        "ffmpeg decoded {n_frames_out} frames, expected {}",
        frames.len()
    );
    eprintln!(
        "ffmpeg interop: {} frames decoded from {} byte Ogg",
        n_frames_out,
        ogg.len()
    );
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
