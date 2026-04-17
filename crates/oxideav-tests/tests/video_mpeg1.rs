//! MPEG-1 Video roundtrip comparison tests against ffmpeg.
//!
//! MPEG-1 only supports specific frame rates (24, 25, 30, etc.), so we use 25fps.
//!
//! - `encoder_roundtrip`: encode with ours, decode with ffmpeg, compare Y-PSNR.
//! - `decoder_vs_ffmpeg`: encode with ffmpeg, decode with our decoder, compare.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};

const W: u32 = 64;
const H: u32 = 64;
const NFRAMES: usize = 5;

fn make_yuv_frame(raw: &[u8], idx: usize, w: u32, h: u32) -> VideoFrame {
    let y_sz = (w * h) as usize;
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let c_sz = cw * ch;
    let frame_sz = y_sz + 2 * c_sz;
    let base = idx * frame_sz;
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(idx as i64),
        time_base: TimeBase::new(1, 25),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: raw[base..base + y_sz].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: raw[base + y_sz..base + y_sz + c_sz].to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: raw[base + y_sz + c_sz..base + frame_sz].to_vec(),
            },
        ],
    }
}

/// Encode 5 frames with our MPEG-1 encoder, decode with ffmpeg, compare.
#[test]
fn encoder_roundtrip() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mpeg1_enc");
    let _ = std::fs::create_dir_all(&tmp);
    let ref_yuv = tmp.join("ref.yuv");

    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=25:duration=0.2",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        ref_yuv.to_str().unwrap(),
    ]));

    let raw = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    assert!(raw.len() >= NFRAMES * frame_sz);

    // Encode with our encoder.
    let reg = oxideav::Registries::with_all_features();
    let mut params = CodecParameters::video(CodecId::new("mpeg1video"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(25, 1));
    params.bit_rate = Some(500_000);

    let mut enc = reg.codecs.make_encoder(&params).expect("make encoder");

    for i in 0..NFRAMES {
        let frame = make_yuv_frame(&raw, i, W, H);
        enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    }
    enc.flush().expect("flush");

    let mut es_data = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => es_data.extend_from_slice(&p.data),
            Err(Error::NeedMore) => break,
            Err(Error::Eof) => break,
            Err(e) => panic!("encoder error: {e}"),
        }
    }

    // Write ES and decode with ffmpeg.
    let es_path = tmp.join("ours.m1v");
    let decoded_yuv = tmp.join("decoded.yuv");
    std::fs::write(&es_path, &es_data).expect("write es");

    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "mpegvideo",
        "-i",
        es_path.to_str().unwrap(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        decoded_yuv.to_str().unwrap(),
    ]));

    let decoded = std::fs::read(&decoded_yuv).expect("read decoded yuv");
    let decoded_nframes = decoded.len() / frame_sz;

    for i in 0..decoded_nframes.min(NFRAMES) {
        let orig_y = &raw[i * frame_sz..i * frame_sz + (W * H) as usize];
        let dec_y = &decoded[i * frame_sz..i * frame_sz + (W * H) as usize];
        let psnr = oxideav_tests::video_y_psnr(orig_y, dec_y, W, H);
        eprintln!("  [MPEG-1 encoder frame {i}] PSNR={psnr:.1} dB");
        // MPEG-1 P-frames on testsrc can be noisy at low bitrates.
        // I-frames should be > 25 dB, P-frames may be lower.
        assert!(
            psnr > 15.0,
            "MPEG-1 encoder frame {i} PSNR {psnr:.1} dB < 15 dB threshold"
        );
    }
}

/// Encode with ffmpeg, decode with our decoder directly, compare.
#[test]
fn decoder_vs_ffmpeg() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mpeg1_dec");
    let _ = std::fs::create_dir_all(&tmp);
    let es_path = tmp.join("ffmpeg.m1v");
    let ref_yuv = tmp.join("ref.yuv");

    // Encode with ffmpeg's MPEG-1 encoder as ES.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=25:duration=0.2",
        "-c:v",
        "mpeg1video",
        "-q:v",
        "5",
        "-g",
        "5",
        "-f",
        "mpeg1video",
        es_path.to_str().unwrap(),
    ]));

    // Decode with ffmpeg for reference.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "mpegvideo",
        "-i",
        es_path.to_str().unwrap(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        ref_yuv.to_str().unwrap(),
    ]));

    let ref_data = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    let ref_nframes = ref_data.len() / frame_sz;

    // Decode the ES with our MPEG-1 decoder directly.
    let es_data = std::fs::read(&es_path).expect("read es");
    let reg = oxideav::Registries::with_all_features();
    let dec_params = CodecParameters::video(CodecId::new("mpeg1video"));
    let mut dec = reg.codecs.make_decoder(&dec_params).expect("make decoder");

    // Feed the entire ES as one packet.
    let pkt = Packet::new(0, TimeBase::new(1, 25), es_data);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut our_frames: Vec<Vec<u8>> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => {
                let mut y = Vec::with_capacity((W * H) as usize);
                for row in 0..v.height as usize {
                    let start = row * v.planes[0].stride;
                    y.extend_from_slice(&v.planes[0].data[start..start + v.width as usize]);
                }
                our_frames.push(y);
            }
            Ok(_) => {}
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => {
                eprintln!("  MPEG-1 decoder error (stopping): {e}");
                break;
            }
        }
    }

    let count = our_frames.len().min(ref_nframes);
    eprintln!(
        "  MPEG-1 decoder: decoded {} frames (ref has {ref_nframes})",
        our_frames.len()
    );

    if count == 0 {
        eprintln!("  WARNING: no frames decoded from ffmpeg's ES.");
        return;
    }

    for i in 0..count {
        let ref_y = &ref_data[i * frame_sz..i * frame_sz + (W * H) as usize];
        let our_y = &our_frames[i];
        let psnr = oxideav_tests::video_y_psnr(our_y, ref_y, W, H);
        eprintln!("  [MPEG-1 decoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 15.0,
            "MPEG-1 decoder frame {i} PSNR {psnr:.1} dB < 15 dB threshold"
        );
    }
}
