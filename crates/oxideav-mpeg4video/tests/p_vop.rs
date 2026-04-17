//! P-VOP encoder integration tests.
//!
//! Exercises the GOP = 16 cadence: 1 I-VOP followed by 15 P-VOPs. Validates:
//! * Our decoder can round-trip the bitstream.
//! * PSNR of the round-trip is above the acceptance bar.
//! * Total bitrate is substantially less than an all-I encode.
//! * (Optional) ffmpeg's `mpeg4` decoder accepts our elementary stream.

use std::process::Command;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

/// Synthesise a deterministic 64×64 frame stream. The content is a moving
/// gradient that's motion-compensation friendly (mostly linear translation
/// between frames), which exercises the ME + half-pel path without needing
/// an external source.
fn make_frame(idx: u32, width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![128u8; cw * ch];
    let mut cr = vec![128u8; cw * ch];
    let shift = idx as i32; // 1 pel per frame horizontal shift
    let shift_v = (idx / 2) as i32;
    for row in 0..h {
        for col in 0..w {
            let x = col as i32 - shift;
            let yy = row as i32 - shift_v;
            // Gradient with some periodic bumps so we have non-trivial DCT energy.
            let base = ((x.rem_euclid(64) * 4) + (yy.rem_euclid(48) * 2)) as u8;
            let bump = ((x.rem_euclid(16) as u8).wrapping_mul(2))
                .wrapping_add((yy.rem_euclid(16) as u8).wrapping_mul(3));
            y[row * w + col] = base.wrapping_add(bump);
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            let x = col as i32 - shift / 2;
            let yy = row as i32 - shift_v / 2;
            cb[row * cw + col] =
                (128i32 + (x.rem_euclid(16)) - (yy.rem_euclid(16))).clamp(0, 255) as u8;
            cr[row * cw + col] =
                (128i32 + (yy.rem_euclid(16)) - (x.rem_euclid(16))).clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width,
        height,
        pts: Some(idx as i64),
        time_base: TimeBase::new(1, 24),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn flatten_frame(v: &VideoFrame) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &v.planes {
        out.extend_from_slice(&p.data);
    }
    out
}

fn build_encoder(width: u32, height: u32) -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    oxideav_mpeg4video::encoder::make_encoder(&params).expect("build encoder")
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq: u64 = 0;
    for i in 0..n {
        let d = a[i] as i32 - b[i] as i32;
        sum_sq += (d * d) as u64;
    }
    let mse = sum_sq as f64 / n as f64;
    if mse == 0.0 {
        return 100.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn command_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Encode 16 frames (1 I + 15 P) and decode them back through our own
/// decoder. Assert PSNR > 30 dB per frame and that the total byte count is
/// substantially less than an all-I encode.
#[test]
fn p_vop_round_trip_psnr_and_bitrate() {
    let (width, height) = (64u32, 64u32);
    let num_frames = 16u32;

    // Source frames.
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    // Encode with default (GOP = 16 → 1 I + 15 P).
    let mut enc = build_encoder(width, height);
    let mut packets: Vec<Packet> = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
    assert!(
        packets.len() >= num_frames as usize,
        "expected at least {} packets, got {}",
        num_frames,
        packets.len()
    );

    // First packet is a keyframe (I-VOP).
    assert!(packets[0].flags.keyframe, "first packet should be keyframe");
    // Subsequent packets (1..=15) should NOT be keyframes.
    for (i, pkt) in packets.iter().enumerate().take(num_frames as usize).skip(1) {
        assert!(
            !pkt.flags.keyframe,
            "packet {} incorrectly marked keyframe",
            i
        );
    }

    // Total size of P-VOPs must be well below the I-VOP's size (better
    // than 50% compression per the acceptance bar: all-I would be
    // `num_frames * i_size`; our stream is `i_size + sum_of_p_sizes`.
    // Check `sum_of_p_sizes < 0.5 * (num_frames-1) * i_size`.
    let i_size = packets[0].data.len();
    let p_size: usize = packets[1..num_frames as usize]
        .iter()
        .map(|p| p.data.len())
        .sum();
    let all_i_equivalent = i_size * (num_frames as usize - 1);
    eprintln!(
        "bitrate: I={} bytes; P_total={} bytes; all-I equivalent P total={}",
        i_size, p_size, all_i_equivalent
    );
    assert!(
        p_size < all_i_equivalent / 2,
        "P-VOP total {} bytes is not < 50% of all-I equivalent {} bytes",
        p_size,
        all_i_equivalent
    );

    // Concatenate into a single ES and decode with our decoder.
    let mut es = Vec::new();
    for pkt in &packets {
        es.extend_from_slice(&pkt.data);
    }
    let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&dec_params).expect("build decoder");
    let in_pkt = Packet::new(0, TimeBase::new(1, 24), es.clone());
    dec.send_packet(&in_pkt).expect("send_packet");
    dec.flush().expect("flush decoder");

    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(f)) => decoded.push(f),
            Ok(_) => panic!("non-video frame"),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e:?}"),
        }
    }
    assert_eq!(
        decoded.len(),
        num_frames as usize,
        "decoded {} frames, expected {}",
        decoded.len(),
        num_frames
    );

    // PSNR per frame.
    for (i, (src, dec_f)) in src_frames.iter().zip(decoded.iter()).enumerate() {
        let s = flatten_frame(src);
        let d = flatten_frame(dec_f);
        let p = psnr(&s, &d);
        eprintln!("frame {i}: PSNR = {p:.2} dB");
        assert!(p > 30.0, "frame {i} PSNR {p:.2} dB below 30 dB threshold");
    }

    // Also dump our decoder's output so the ffmpeg-interop test can compare
    // frame 0 pels between decoders.
    let mut our_yuv = Vec::new();
    for f in &decoded {
        our_yuv.extend_from_slice(&flatten_frame(f));
    }
    let tmp = std::env::temp_dir();
    std::fs::write(tmp.join("oxideav_pvop_ours_decoded.yuv"), &our_yuv).ok();
}

/// FFmpeg interop: encode with our encoder, decode with ffmpeg. Skip when
/// ffmpeg isn't on PATH (keeps CI happy).
#[test]
fn p_vop_ffmpeg_decode() {
    if !command_exists("ffmpeg") {
        eprintln!("ffmpeg missing — skipping P-VOP ffmpeg interop test");
        return;
    }
    let (width, height) = (64u32, 64u32);
    let num_frames = 16u32;
    let mut src_frames = Vec::with_capacity(num_frames as usize);
    for i in 0..num_frames {
        src_frames.push(make_frame(i, width, height));
    }

    let mut enc = build_encoder(width, height);
    let mut es = Vec::new();
    for f in &src_frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => es.extend_from_slice(&pkt.data),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }

    let tmp = std::env::temp_dir();
    let es_path = tmp.join("oxideav_pvop_ours.m4v");
    std::fs::write(&es_path, &es).expect("write m4v");
    let yuv_out = tmp.join("oxideav_pvop_ffmpeg.yuv");
    let _ = std::fs::remove_file(&yuv_out);
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "m4v",
            "-i",
            es_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            yuv_out.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed to decode our P-VOP stream");
    let ffmpeg_decoded = std::fs::read(&yuv_out).expect("read ffmpeg output");
    let per_frame_bytes = (width as usize * height as usize * 3) / 2;
    assert!(
        ffmpeg_decoded.len() >= per_frame_bytes,
        "ffmpeg output too small: {} bytes",
        ffmpeg_decoded.len()
    );

    // FFmpeg's I-VOP reconstruction must match source closely (no prior
    // drift). P-VOPs accumulate drift with ffmpeg's integer IDCT vs our
    // float IDCT, so we only check (a) the I-VOP frame is accurate and
    // (b) PSNR stays reasonable for the first couple of P-VOPs. Longer
    // GOPs will drift; a bit-exact integer-IDCT encoder path is listed
    // in the encoder's follow-up items (see `src/pvop.rs`).
    let src0 = flatten_frame(&src_frames[0]);
    let ff0 = &ffmpeg_decoded[0..per_frame_bytes];
    let p0 = psnr(&src0, ff0);
    eprintln!("ffmpeg decode frame 0 (I-VOP): PSNR = {p0:.2} dB");
    assert!(
        p0 > 30.0,
        "ffmpeg I-VOP PSNR {p0:.2} dB below 30 dB — bitstream is malformed"
    );
    // First P-VOP after the I.
    let src1 = flatten_frame(&src_frames[1]);
    let ff1 = &ffmpeg_decoded[per_frame_bytes..2 * per_frame_bytes];
    let p1 = psnr(&src1, ff1);
    eprintln!("ffmpeg decode frame 1 (first P-VOP): PSNR = {p1:.2} dB");
    assert!(
        p1 > 25.0,
        "ffmpeg first P-VOP PSNR {p1:.2} dB below 25 dB — bitstream or MV drift"
    );
    let _ = &es_path;
}
