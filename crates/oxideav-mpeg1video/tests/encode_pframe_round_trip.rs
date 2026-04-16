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

/// Compute PSNR in dB across all three planes of two YUV frames. Returns
/// f64::INFINITY for bit-identical frames.
fn psnr_db(orig: &VideoFrame, recon: &VideoFrame) -> f64 {
    let mut sq: f64 = 0.0;
    let mut count: u64 = 0;
    for (o, r) in orig.planes.iter().zip(recon.planes.iter()) {
        for (a, b) in o.data.iter().zip(r.data.iter()) {
            let d = *a as f64 - *b as f64;
            sq += d * d;
            count += 1;
        }
    }
    if count == 0 {
        return f64::INFINITY;
    }
    let mse = sq / count as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
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

/// Encode an I + 4 P sequence of a synthetic moving test pattern and
/// measure PSNR of the round-trip against the original YUV. Spec target per
/// task brief: PSNR > 30 dB.
#[test]
fn ibppp_psnr_over_30db() {
    // Synth a 64×64 scene with a horizontally-moving gradient (+1 px/frame).
    // This exercises the P-frame MC path directly — the encoder's motion
    // estimation should discover the +2 half-pel horizontal MV.
    let w = 64u32;
    let h = 64u32;
    let n = 5;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut frames = Vec::with_capacity(n);
    for t in 0..n {
        let mut y = vec![0u8; (w * h) as usize];
        let mut cb = vec![128u8; (cw * ch) as usize];
        let mut cr = vec![128u8; (cw * ch) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                // Diagonal gradient shifting by (t, t/2) pels per frame.
                let ix = i.wrapping_add(t) & 0x3F;
                let jy = j.wrapping_add(t / 2) & 0x3F;
                let v = ((ix * 4) ^ (jy * 4)) as u8;
                y[j * w as usize + i] = v;
            }
        }
        for j in 0..ch as usize {
            for i in 0..cw as usize {
                cb[j * cw as usize + i] = 128u8.wrapping_add(((i + t) & 0x1F) as u8);
                cr[j * cw as usize + i] = 128u8.wrapping_add(((j + t / 2) & 0x1F) as u8);
            }
        }
        frames.push(VideoFrame {
            format: PixelFormat::Yuv420P,
            width: w,
            height: h,
            pts: Some(t as i64),
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

    let bytes = encode_frames(&frames);
    assert!(!bytes.is_empty());
    let recon = decode_frames(&bytes);
    assert_eq!(recon.len(), n, "decoder produced wrong frame count");

    let mut total_psnr = 0.0;
    let mut min_psnr = f64::INFINITY;
    for (i, (orig, rec)) in frames.iter().zip(recon.iter()).enumerate() {
        let p = psnr_db(orig, rec);
        eprintln!("IBPPP frame {i}: PSNR={p:.2} dB");
        total_psnr += p;
        if p < min_psnr {
            min_psnr = p;
        }
    }
    let avg = total_psnr / n as f64;
    eprintln!("IBPPP avg PSNR={avg:.2} dB (min {min_psnr:.2} dB)");
    assert!(
        avg > 30.0,
        "average PSNR {avg:.2} dB ≤ 30 dB threshold per task brief"
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

/// ffmpeg interop: generate an MPEG-1 elementary stream with `-c:v mpeg1video
/// -bf 0 -g 30`, decode with our decoder, and compute PSNR against an ffmpeg
/// reference decode. Skip if ffmpeg isn't available.
#[test]
fn decode_ffmpeg_pframe_stream_psnr_over_30db() {
    let Some(_) = which("ffmpeg") else {
        eprintln!("ffmpeg not found — skipping ffmpeg-source P-frame PSNR test");
        return;
    };
    // Generate a short testsrc clip as the ffmpeg-encoded source stream.
    let stream_path = "/tmp/mpeg1_bf0_g30.m1v";
    let ff_decode_path = "/tmp/mpeg1_bf0_g30_ffdec.yuv";
    let _ = std::fs::remove_file(stream_path);
    let _ = std::fs::remove_file(ff_decode_path);

    let enc_status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=24:duration=1",
            "-c:v",
            "mpeg1video",
            "-bf",
            "0",
            "-g",
            "30",
            "-pix_fmt",
            "yuv420p",
            stream_path,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let enc_status = match enc_status {
        Ok(s) if s.success() => s,
        _ => {
            eprintln!("ffmpeg failed to generate source stream — skipping");
            return;
        }
    };
    let _ = enc_status;

    // Decode with ffmpeg to get a reference YUV.
    let ff_dec_status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "mpegvideo",
            "-i",
            stream_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            ff_decode_path,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let ff_dec_status = match ff_dec_status {
        Ok(s) if s.success() => s,
        _ => {
            eprintln!("ffmpeg failed to decode reference stream — skipping");
            return;
        }
    };
    let _ = ff_dec_status;

    let bytes = match std::fs::read(stream_path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("can't read generated stream — skipping");
            return;
        }
    };
    // Check frame count: testsrc at 24fps × 1s = 24 frames.
    let ff_frames = match read_yuv_n_frames(ff_decode_path, W, H, 24) {
        Some(f) => f,
        None => {
            eprintln!("can't read ffmpeg reference output — skipping");
            return;
        }
    };
    let our_frames = decode_frames(&bytes);
    assert!(
        !our_frames.is_empty(),
        "our decoder produced no frames from ffmpeg stream"
    );
    let pairs = our_frames.len().min(ff_frames.len());
    assert!(pairs > 0);
    let mut total = 0.0;
    let mut min_p = f64::INFINITY;
    for i in 0..pairs {
        let p = psnr_db(&ff_frames[i], &our_frames[i]);
        eprintln!("ffmpeg→ours frame {i}: PSNR={p:.2} dB");
        total += p;
        if p < min_p {
            min_p = p;
        }
    }
    let avg = total / pairs as f64;
    eprintln!("ffmpeg→ours avg PSNR={avg:.2} dB over {pairs} frames (min {min_p:.2} dB)");
    assert!(
        avg > 30.0,
        "ffmpeg interop avg PSNR {avg:.2} dB ≤ 30 dB threshold"
    );
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
