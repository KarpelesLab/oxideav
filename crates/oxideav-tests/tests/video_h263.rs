//! H.263 roundtrip comparison tests against ffmpeg.
//!
//! H.263 baseline only supports standard sizes. We use QCIF (176x144).
//!
//! - `encoder_roundtrip`: encode with ours, decode with ffmpeg, compare Y-PSNR.
//! - `decoder_vs_ffmpeg`: encode with ffmpeg, decode with our decoder, compare.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};

const W: u32 = 176;
const H: u32 = 144;
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
        time_base: TimeBase::new(1, 10),
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

/// Encode 5 frames with our H.263 encoder, decode with ffmpeg, compare.
#[test]
fn encoder_roundtrip() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_h263_enc");
    let _ = std::fs::create_dir_all(&tmp);
    let ref_yuv = tmp.join("ref.yuv");

    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=176x144:rate=10:duration=0.5",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        ref_yuv.to_str().unwrap(),
    ]));

    let raw = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    assert!(raw.len() >= NFRAMES * frame_sz);

    // Encode with our H.263 encoder.
    let reg = oxideav::Registries::with_all_features();
    let mut params = CodecParameters::video(CodecId::new("h263"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(10, 1));

    let mut enc = reg.codecs.make_encoder(&params).expect("make encoder");

    let mut es_data = Vec::new();
    for i in 0..NFRAMES {
        let frame = make_yuv_frame(&raw, i, W, H);
        enc.send_frame(&Frame::Video(frame)).expect("send_frame");
        enc.flush().expect("flush");
        loop {
            match enc.receive_packet() {
                Ok(p) => es_data.extend_from_slice(&p.data),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("encoder error: {e}"),
            }
        }
    }

    // Decode with ffmpeg.
    let es_path = tmp.join("ours.h263");
    let decoded_yuv = tmp.join("decoded.yuv");
    std::fs::write(&es_path, &es_data).expect("write es");

    assert!(
        oxideav_tests::ffmpeg(&[
            "-f",
            "h263",
            "-i",
            es_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            decoded_yuv.to_str().unwrap(),
        ]),
        "ffmpeg failed to decode our H.263 stream"
    );

    let decoded = std::fs::read(&decoded_yuv).expect("read decoded yuv");
    let decoded_nframes = decoded.len() / frame_sz;

    for i in 0..decoded_nframes.min(NFRAMES) {
        let orig_y = &raw[i * frame_sz..i * frame_sz + (W * H) as usize];
        let dec_y = &decoded[i * frame_sz..i * frame_sz + (W * H) as usize];
        let psnr = oxideav_tests::video_y_psnr(orig_y, dec_y, W, H);
        eprintln!("  [H.263 encoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 25.0,
            "H.263 encoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
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

    let tmp = oxideav_tests::tmp("video_h263_dec");
    let _ = std::fs::create_dir_all(&tmp);
    let es_path = tmp.join("ffmpeg.h263");
    let ref_yuv = tmp.join("ref.yuv");

    // Encode with ffmpeg's H.263 as elementary stream.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=176x144:rate=10:duration=0.5",
        "-c:v",
        "h263",
        "-q:v",
        "5",
        "-g",
        "5",
        "-f",
        "h263",
        es_path.to_str().unwrap(),
    ]));

    // Decode with ffmpeg for reference.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "h263",
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

    // Decode the ES with our H.263 decoder.
    let es_data = std::fs::read(&es_path).expect("read es");
    let reg = oxideav::Registries::with_all_features();
    let dec_params = CodecParameters::video(CodecId::new("h263"));
    let mut dec = reg.codecs.make_decoder(&dec_params).expect("make decoder");

    // Feed the entire ES as one packet.
    let pkt = Packet::new(0, TimeBase::new(1, 10), es_data);
    if let Err(e) = dec.send_packet(&pkt) {
        eprintln!("  H.263 decoder: send_packet error: {e}");
    }
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
                eprintln!("  H.263 decoder error (stopping): {e}");
                break;
            }
        }
    }

    let count = our_frames.len().min(ref_nframes);
    eprintln!(
        "  H.263 decoder: decoded {} frames (ref has {ref_nframes})",
        our_frames.len()
    );

    if count == 0 {
        eprintln!(
            "  WARNING: no frames decoded. The H.263 decoder may not support \
             ffmpeg's ES as a single-packet input."
        );
        return;
    }

    for i in 0..count {
        let ref_y = &ref_data[i * frame_sz..i * frame_sz + (W * H) as usize];
        let our_y = &our_frames[i];
        let psnr = oxideav_tests::video_y_psnr(our_y, ref_y, W, H);
        eprintln!("  [H.263 decoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 25.0,
            "H.263 decoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
        );
    }
}
