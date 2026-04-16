//! End-to-end P-frame encoder round-trip test.
//!
//! Pipeline:
//!   raw 64×64 yuv420p (24 frames from `/tmp/mpeg1_p_in.yuv`)
//!     → our encoder (I + 23 P) → MPEG-1 elementary stream
//!     → our decoder → reconstructed yuv420p sequence
//!     → measure pixel match against the input
//!
//! The fixture is generated with:
//!   ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=1" \
//!          -f rawvideo -pix_fmt yuv420p /tmp/mpeg1_p_in.yuv
//!
//! Tests that can't find their fixture are skipped (logged, not failed).

use std::path::Path;

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational,
    TimeBase, VideoFrame,
};
use oxideav_mpeg1video::{
    decoder::make_decoder,
    encoder::{make_encoder, DEFAULT_QUANT_SCALE},
    CODEC_ID_STR,
};

const W: u32 = 64;
const H: u32 = 64;
const NFRAMES: usize = 24;
const FIXTURE: &str = "/tmp/mpeg1_p_in.yuv";

fn read_yuv_n_frames(path: &str, w: u32, h: u32, n: usize) -> Option<Vec<VideoFrame>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping P-frame test");
        return None;
    }
    let bytes = std::fs::read(path).expect("read fixture");
    let y_size = (w * h) as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let c_size = (cw * ch) as usize;
    let frame_size = y_size + 2 * c_size;
    let need = frame_size * n;
    if bytes.len() < need {
        panic!(
            "fixture too short: {} < {} ({} frames × {} bytes)",
            bytes.len(),
            need,
            n,
            frame_size
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * frame_size;
        let y = bytes[base..base + y_size].to_vec();
        let cb = bytes[base + y_size..base + y_size + c_size].to_vec();
        let cr = bytes[base + y_size + c_size..base + frame_size].to_vec();
        out.push(VideoFrame {
            format: PixelFormat::Yuv420P,
            width: w,
            height: h,
            pts: Some(i as i64),
            time_base: TimeBase::new(1, 24),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cb,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cr,
                },
            ],
        });
    }
    Some(out)
}

fn encode_frames(frames: &[VideoFrame]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(24, 1));
    params.bit_rate = Some(1_500_000);
    let mut enc = make_encoder(&params).expect("build encoder");
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
    }
    enc.flush().expect("flush");
    let mut data = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => data.extend_from_slice(&p.data),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }
    data
}

fn encode_frames_intra_only(frames: &[VideoFrame]) -> Vec<u8> {
    // Force I-only by encoding each frame in isolation (each `make_encoder`
    // session starts a new GOP at frame 0 — picture coding type = I).
    let mut out = Vec::new();
    for f in frames {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(W);
        params.height = Some(H);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(24, 1));
        params.bit_rate = Some(1_500_000);
        let mut enc = make_encoder(&params).expect("build encoder");
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        enc.flush().expect("flush");
        loop {
            match enc.receive_packet() {
                Ok(p) => out.extend_from_slice(&p.data),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("encoder error: {e}"),
            }
        }
    }
    out
}

fn decode_frames(bytes: &[u8]) -> Vec<VideoFrame> {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("build decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 24), bytes.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("decoder error: {e}"),
        }
    }
    frames
}

fn pixel_match(orig: &VideoFrame, recon: &VideoFrame, tolerance: i32) -> f64 {
    assert_eq!(orig.width, recon.width);
    assert_eq!(orig.height, recon.height);
    assert_eq!(orig.planes.len(), recon.planes.len());
    let mut total: u64 = 0;
    let mut matched: u64 = 0;
    for (o, r) in orig.planes.iter().zip(recon.planes.iter()) {
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            total += 1;
            if (*a as i32 - *b as i32).abs() <= tolerance {
                matched += 1;
            }
        }
    }
    matched as f64 / total as f64
}

fn mean_abs_diff(orig: &VideoFrame, recon: &VideoFrame) -> f64 {
    let mut total = 0u64;
    let mut count = 0u64;
    for (o, r) in orig.planes.iter().zip(recon.planes.iter()) {
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            total += (*a as i32 - *b as i32).unsigned_abs() as u64;
            count += 1;
        }
    }
    total as f64 / count as f64
}

#[test]
fn pframe_round_trip_64x64_24frames() {
    let Some(frames) = read_yuv_n_frames(FIXTURE, W, H, NFRAMES) else {
        return;
    };
    assert_eq!(frames.len(), NFRAMES);
    let bytes = encode_frames(&frames);
    eprintln!(
        "encoded {} bytes for {} frames (I + {} P) at qp={}",
        bytes.len(),
        NFRAMES,
        NFRAMES - 1,
        DEFAULT_QUANT_SCALE
    );
    assert!(!bytes.is_empty(), "encoder produced no bytes");
    assert_eq!(&bytes[..4], &[0x00, 0x00, 0x01, 0xB3]);

    let recon = decode_frames(&bytes);
    assert_eq!(recon.len(), NFRAMES, "decoder produced wrong frame count");

    let mut total_pct16: f64 = 0.0;
    let mut total_pct32: f64 = 0.0;
    let mut total_mad: f64 = 0.0;
    let mut min_pct16: f64 = 100.0;
    for (i, (orig, rec)) in frames.iter().zip(recon.iter()).enumerate() {
        let pct16 = pixel_match(orig, rec, 16) * 100.0;
        let pct32 = pixel_match(orig, rec, 32) * 100.0;
        let mad = mean_abs_diff(orig, rec);
        eprintln!("frame {i}: MAD={mad:.2}, pct(±16)={pct16:.2}%, pct(±32)={pct32:.2}%");
        total_pct16 += pct16;
        total_pct32 += pct32;
        total_mad += mad;
        if pct16 < min_pct16 {
            min_pct16 = pct16;
        }
    }
    let avg_pct16 = total_pct16 / NFRAMES as f64;
    let avg_pct32 = total_pct32 / NFRAMES as f64;
    let avg_mad = total_mad / NFRAMES as f64;
    eprintln!(
        "avg pct(±16)={avg_pct16:.2}%, avg pct(±32)={avg_pct32:.2}%, avg MAD={avg_mad:.2}, worst pct(±16)={min_pct16:.2}%"
    );

    assert!(
        avg_pct16 >= 95.0,
        "average ≤±16 match {avg_pct16:.2}% < 95%"
    );
}

#[test]
fn pframe_size_smaller_than_intra_only() {
    let Some(frames) = read_yuv_n_frames(FIXTURE, W, H, NFRAMES) else {
        return;
    };
    let p_bytes = encode_frames(&frames);
    let i_bytes = encode_frames_intra_only(&frames);
    eprintln!(
        "P-enabled: {} bytes; I-only: {} bytes; ratio = {:.2}x",
        p_bytes.len(),
        i_bytes.len(),
        i_bytes.len() as f64 / p_bytes.len() as f64
    );
    assert!(
        p_bytes.len() < i_bytes.len(),
        "P-frame coding should compress better than I-only ({} >= {})",
        p_bytes.len(),
        i_bytes.len()
    );
}

#[test]
fn ffmpeg_decodes_our_pframe_output() {
    let Some(frames) = read_yuv_n_frames(FIXTURE, W, H, NFRAMES) else {
        return;
    };
    let Some(_) = which("ffmpeg") else {
        eprintln!("ffmpeg not found — skipping ffmpeg interop test");
        return;
    };
    let bytes = encode_frames(&frames);
    let in_path = "/tmp/mpeg1_p_oxideav.m1v";
    let out_path = "/tmp/mpeg1_p_oxideav_decoded.yuv";
    std::fs::write(in_path, &bytes).expect("write encoded m1v");
    let _ = std::fs::remove_file(out_path);

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "mpegvideo",
            "-i",
            in_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("spawn ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode our P-frame output"
    );

    // ffmpeg writes all 24 frames concatenated.
    let ff = read_yuv_n_frames(out_path, W, H, NFRAMES).expect("read ffmpeg output");
    let mut total_pct16: f64 = 0.0;
    let mut min_pct16: f64 = 100.0;
    for (i, (orig, rec)) in frames.iter().zip(ff.iter()).enumerate() {
        let pct16 = pixel_match(orig, rec, 16) * 100.0;
        let mad = mean_abs_diff(orig, rec);
        eprintln!("ffmpeg frame {i}: MAD={mad:.2}, pct(±16)={pct16:.2}%");
        total_pct16 += pct16;
        if pct16 < min_pct16 {
            min_pct16 = pct16;
        }
    }
    let avg = total_pct16 / NFRAMES as f64;
    eprintln!("ffmpeg avg pct(±16)={avg:.2}%, worst {min_pct16:.2}%");
    assert!(avg >= 90.0, "ffmpeg avg ≤±16 match {avg:.2}% < 90%");
}

fn which(prog: &str) -> Option<String> {
    let p = std::process::Command::new("which")
        .arg(prog)
        .output()
        .ok()?;
    if p.status.success() {
        let s = String::from_utf8_lossy(&p.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    } else {
        None
    }
}
