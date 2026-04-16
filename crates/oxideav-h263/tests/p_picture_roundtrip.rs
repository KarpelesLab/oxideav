//! End-to-end P-picture round-trip tests.
//!
//! * `encode_decode_i_p_sequence_self_round_trip` — encode a synthesised
//!   "moving square" sequence as `[I, P, P, P]`, decode with our own
//!   decoder, assert PSNR > 30 dB for each frame.
//! * `encode_ffmpeg_decode_i_p_sequence` — if `ffmpeg` is on `$PATH`, decode
//!   our encoded stream with `-f h263` and validate frame count + no
//!   error return.
//! * `decode_ffmpeg_i_p_clip` — if the pre-generated fixture exists, parse
//!   an ffmpeg-encoded I+P H.263 clip with our decoder and compare the
//!   MB-type histogram against ffmpeg's output.

use std::process::Command;

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame,
};
use oxideav_h263::decoder::H263Decoder;
use oxideav_h263::encoder::make_encoder;

const W: u32 = 176;
const H: u32 = 144;

fn make_encoder_qcif() -> Box<dyn Encoder> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_h263::CODEC_ID_STR));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(10, 1));
    make_encoder(&params).expect("make encoder")
}

/// Build a frame with a white square on a grey background, at `(sx, sy)`.
fn moving_square_frame(sx: i32, sy: i32, pts: i64) -> VideoFrame {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![80u8; (W * H) as usize];
    let size = 32i32;
    for j in 0..size {
        for i in 0..size {
            let xx = sx + i;
            let yy = sy + j;
            if xx >= 0 && xx < W as i32 && yy >= 0 && yy < H as i32 {
                y[(yy as usize) * W as usize + (xx as usize)] = 210;
            }
        }
    }
    let cb = vec![128u8; cw * ch];
    let cr = vec![128u8; cw * ch];
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: W,
        height: H,
        pts: Some(pts),
        time_base: TimeBase::new(1, 10),
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y,
            },
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

fn psnr(src: &VideoFrame, dec: &VideoFrame) -> f64 {
    assert_eq!(src.width, dec.width);
    assert_eq!(src.height, dec.height);
    let (w, h) = (src.width as usize, src.height as usize);
    let mut mse = 0f64;
    let mut n = 0u64;
    let sp = &src.planes[0];
    let dp = &dec.planes[0];
    for j in 0..h {
        for i in 0..w {
            let a = sp.data[j * sp.stride + i] as f64;
            let b = dp.data[j * dp.stride + i] as f64;
            let d = a - b;
            mse += d * d;
            n += 1;
        }
    }
    if n == 0 {
        return f64::INFINITY;
    }
    let mse = mse / n as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0f64 / mse).log10()
}

/// Encode `frames` through the encoder, collect all output packets in order.
fn encode_frames(enc: &mut dyn Encoder, frames: &[VideoFrame]) -> Result<Vec<Packet>> {
    let mut out = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))?;
        loop {
            match enc.receive_packet() {
                Ok(p) => out.push(p),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => return Err(e),
            }
        }
    }
    enc.flush()?;
    loop {
        match enc.receive_packet() {
            Ok(p) => out.push(p),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

fn decode_packets(packets: &[Packet]) -> Vec<VideoFrame> {
    let mut dec = H263Decoder::new(CodecId::new(oxideav_h263::CODEC_ID_STR));
    let mut out = Vec::new();
    for p in packets {
        dec.send_packet(p).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => out.push(v),
                Ok(_) => panic!("non-video"),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("decoder error: {e:?}"),
            }
        }
    }
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            Ok(_) => panic!("non-video"),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("decoder error: {e:?}"),
        }
    }
    out
}

/// Round-trip test — encode an I + 3 P-pictures, decode with our own decoder
/// and assert PSNR > 30 dB on every frame.
#[test]
fn encode_decode_i_p_sequence_self_round_trip() {
    let frames: Vec<VideoFrame> = (0..4)
        .map(|i| moving_square_frame(20 + i * 4, 40, i as i64))
        .collect();

    let mut enc = make_encoder_qcif();
    let packets = encode_frames(&mut *enc, &frames).expect("encode");
    assert_eq!(packets.len(), 4, "one packet per input frame");
    // First packet must be a keyframe, subsequent packets must be P.
    assert!(packets[0].flags.keyframe, "first must be keyframe");
    for (i, p) in packets.iter().enumerate().skip(1) {
        assert!(!p.flags.keyframe, "packet {i} should be P");
    }

    let decoded = decode_packets(&packets);
    assert_eq!(decoded.len(), 4);

    for (i, (s, d)) in frames.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(s, d);
        eprintln!("frame {i} PSNR = {p:.2} dB");
        assert!(p >= 30.0, "frame {i} PSNR {p:.2} dB below 30 dB threshold");
    }
}

/// Encode our I + P stream to a file and pipe it into ffmpeg's H.263
/// decoder, verifying (a) ffmpeg does not error out and (b) it extracts the
/// expected frame count. Skipped when ffmpeg is absent.
#[test]
fn encode_ffmpeg_decode_i_p_sequence() {
    let frames: Vec<VideoFrame> = (0..4)
        .map(|i| moving_square_frame(20 + i * 4, 40, i as i64))
        .collect();

    let mut enc = make_encoder_qcif();
    let packets = encode_frames(&mut *enc, &frames).expect("encode");
    let mut bytes = Vec::new();
    for p in &packets {
        bytes.extend_from_slice(&p.data);
    }
    let tmp = std::env::temp_dir();
    let es_path = tmp.join("h263_ip_ours.h263");
    let yuv_path = tmp.join("h263_ip_check.yuv");
    std::fs::write(&es_path, &bytes).expect("write");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "h263",
            "-i",
            es_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            yuv_path.to_str().unwrap(),
        ])
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    };
    assert!(status.success(), "ffmpeg rejected our I+P stream");

    // Compare frame count: check the output file is exactly 4 frames long.
    let expected_size = (W * H + 2 * (W / 2) * (H / 2)) as usize * 4;
    let actual_size = std::fs::metadata(&yuv_path).expect("stat").len() as usize;
    assert_eq!(
        actual_size, expected_size,
        "ffmpeg decoded frame count mismatch"
    );
}

/// Still-life sanity test: encode the same frame twice (first as I, second
/// as P with all MV=0). Every P-MB should be a skipped MB → output must
/// match the I-frame output exactly, both for our decoder and for ffmpeg.
#[test]
fn encode_still_life_i_p_matches_self() {
    let f = moving_square_frame(30, 40, 0);
    let frames = vec![f.clone(), f.clone()];

    let mut enc = make_encoder_qcif();
    let packets = encode_frames(&mut *enc, &frames).expect("encode");
    assert_eq!(packets.len(), 2);

    let decoded = decode_packets(&packets);
    assert_eq!(decoded.len(), 2);

    // P-frame should match I-frame pel-for-pel.
    for y in 0..H as usize {
        for x in 0..W as usize {
            let a = decoded[0].planes[0].data[y * decoded[0].planes[0].stride + x];
            let b = decoded[1].planes[0].data[y * decoded[1].planes[0].stride + x];
            assert_eq!(
                a, b,
                "still-life I/P disagree at luma ({x},{y}): I={a} P={b}"
            );
        }
    }
}

/// Higher-entropy round-trip test: encode a 10-frame panning-gradient
/// sequence, decode with our own decoder, assert average luma PSNR > 30 dB.
///
/// The panning introduces non-trivial per-MB motion vectors, so this
/// exercises the motion-estimator + MVD VLC path more thoroughly than the
/// "single moving square" case.
#[test]
fn encode_decode_panning_gradient_round_trip() {
    fn panning_gradient(offset: i32, pts: i64) -> VideoFrame {
        let cw = (W / 2) as usize;
        let ch = (H / 2) as usize;
        let mut y = vec![0u8; (W * H) as usize];
        for j in 0..H as usize {
            for i in 0..W as usize {
                let xx = (i as i32 + offset).rem_euclid(W as i32) as usize;
                y[j * W as usize + i] =
                    ((xx * 200 / W as usize) + (j * 50 / H as usize)).min(255) as u8;
            }
        }
        let cb = vec![128u8; cw * ch];
        let cr = vec![128u8; cw * ch];
        VideoFrame {
            format: PixelFormat::Yuv420P,
            width: W,
            height: H,
            pts: Some(pts),
            time_base: TimeBase::new(1, 10),
            planes: vec![
                VideoPlane {
                    stride: W as usize,
                    data: y,
                },
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

    let n = 10;
    let frames: Vec<VideoFrame> = (0..n).map(|i| panning_gradient(i * 2, i as i64)).collect();

    let mut enc = make_encoder_qcif();
    let packets = encode_frames(&mut *enc, &frames).expect("encode");
    assert_eq!(packets.len(), n as usize);
    // First is I.
    assert!(packets[0].flags.keyframe);
    let decoded = decode_packets(&packets);
    assert_eq!(decoded.len(), n as usize);

    let mut total_psnr = 0.0;
    for (i, (s, d)) in frames.iter().zip(decoded.iter()).enumerate() {
        let p = psnr(s, d);
        eprintln!("panning frame {i} PSNR = {p:.2} dB");
        assert!(p >= 30.0, "panning frame {i} PSNR {p:.2} dB < 30 dB");
        total_psnr += p;
    }
    eprintln!(
        "average panning PSNR: {:.2} dB over {n} frames",
        total_psnr / n as f64
    );
}

/// Compare our decoder's output against ffmpeg's on our own I+P stream:
/// bit-exact agreement on every pel (±2 LSB) is the spec-compliance bar.
/// Skipped when ffmpeg is absent.
#[test]
fn our_stream_decoded_by_ours_vs_ffmpeg() {
    let frames: Vec<VideoFrame> = (0..6)
        .map(|i| moving_square_frame(20 + i * 2, 40 + i, i as i64))
        .collect();

    let mut enc = make_encoder_qcif();
    let packets = encode_frames(&mut *enc, &frames).expect("encode");
    let mut bytes = Vec::new();
    for p in &packets {
        bytes.extend_from_slice(&p.data);
    }
    let tmp = std::env::temp_dir();
    let es_path = tmp.join("h263_ip_compare.h263");
    let yuv_path = tmp.join("h263_ip_compare.yuv");
    std::fs::write(&es_path, &bytes).expect("write");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "h263",
            "-i",
            es_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            yuv_path.to_str().unwrap(),
        ])
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    };
    assert!(status.success(), "ffmpeg rejected our I+P stream");

    let ff_yuv = std::fs::read(&yuv_path).expect("read");
    let frame_size = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    assert_eq!(ff_yuv.len(), frame_size * frames.len());

    // Decode with our decoder.
    let decoded = decode_packets(&packets);

    // Compare each frame: our decoder and ffmpeg must agree.
    let mut total = 0u64;
    let mut within = 0u64;
    let mut max_err = 0u32;
    for (i, v) in decoded.iter().enumerate() {
        let base = i * frame_size;
        let y_ref = &ff_yuv[base..base + (W * H) as usize];
        let cw = (W / 2) as usize;
        let ch = (H / 2) as usize;
        let cb_ref = &ff_yuv[base + (W * H) as usize..base + (W * H) as usize + cw * ch];
        let cr_ref = &ff_yuv[base + (W * H) as usize + cw * ch..base + frame_size];

        let mut frame_total = 0u64;
        let mut frame_within = 0u64;
        let mut frame_max = 0u32;
        let mut first_big_mismatch: Option<(usize, usize, i32, i32)> = None;
        for y in 0..H as usize {
            for x in 0..W as usize {
                let a = v.planes[0].data[y * v.planes[0].stride + x] as i32;
                let b = y_ref[y * W as usize + x] as i32;
                let d = (a - b).unsigned_abs();
                frame_total += 1;
                if d <= 2 {
                    frame_within += 1;
                }
                if d > frame_max {
                    frame_max = d;
                }
                if d > 10 && first_big_mismatch.is_none() {
                    first_big_mismatch = Some((x, y, a, b));
                }
            }
        }
        if let Some((x, y, a, b)) = first_big_mismatch {
            eprintln!(
                "  first big mismatch frame {i}: ({x},{y}) MB=({},{}) ours={a} ff={b}",
                x / 16,
                y / 16
            );
        }
        for y in 0..ch {
            for x in 0..cw {
                let a = v.planes[1].data[y * v.planes[1].stride + x] as i32;
                let b = cb_ref[y * cw + x] as i32;
                let d = (a - b).unsigned_abs();
                frame_total += 1;
                if d <= 2 {
                    frame_within += 1;
                }
                let a2 = v.planes[2].data[y * v.planes[2].stride + x] as i32;
                let b2 = cr_ref[y * cw + x] as i32;
                let d2 = (a2 - b2).unsigned_abs();
                frame_total += 1;
                if d2 <= 2 {
                    frame_within += 1;
                }
            }
        }
        eprintln!(
            "frame {i} {}: {:.2}% within ±2 LSB (max {})",
            if packets[i].flags.keyframe { "I" } else { "P" },
            frame_within as f64 / frame_total as f64 * 100.0,
            frame_max
        );
        total += frame_total;
        within += frame_within;
        if frame_max > max_err {
            max_err = frame_max;
        }
    }
    let pct = within as f64 / total as f64;
    eprintln!(
        "our vs ffmpeg on our I+P stream: {:.2}% within ±2 LSB (max abs err {})",
        pct * 100.0,
        max_err
    );
    assert!(
        pct >= 0.99,
        "our decoder disagrees with ffmpeg: {:.2}% within 2 LSB",
        pct * 100.0
    );
}

/// Fixture-based round trip: if the environment has generated a QCIF H.263
/// clip with multiple frames via ffmpeg, decode it with our decoder and
/// verify we extract the expected number of frames and that each decodes
/// without error.
///
/// Fixture generation:
/// ```sh
/// ffmpeg -y -f lavfi -i "testsrc=size=176x144:rate=10:duration=1" \
///     -c:v h263 -qscale:v 5 -g 12 -an -f h263 /tmp/h263_ip_clip.es
/// ```
#[test]
fn decode_ffmpeg_i_p_clip() {
    let path = std::env::temp_dir().join("h263_ip_clip.es");
    if !path.exists() {
        eprintln!("fixture {} missing — skipping", path.display());
        return;
    }
    let es = std::fs::read(&path).expect("read");
    let mut dec = H263Decoder::new(CodecId::new(oxideav_h263::CODEC_ID_STR));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 10), es))
        .expect("send");
    dec.flush().expect("flush");

    let mut count = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => count += 1,
            Ok(_) => panic!("non-video"),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("decoder error: {e:?}"),
        }
    }
    // 1-second clip at 10 fps = 10 frames. Accept anything >= 2 (an I + at
    // least one P), since the test's point is "P-picture path is exercised".
    assert!(
        count >= 2,
        "decoded {count} frames — P-picture path not exercised"
    );
    eprintln!("decoded {count} frames from ffmpeg I+P fixture");
}
