//! MPEG-4 Part 2 (Visual) roundtrip comparison tests against ffmpeg.
//!
//! - `encoder_roundtrip`: encode with ours, decode with ffmpeg, compare Y-PSNR.
//! - `decoder_vs_ffmpeg`: encode with ffmpeg, decode with both, compare Y-PSNR.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, PixelFormat, Rational, TimeBase, VideoFrame, VideoPlane,
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

/// Encode 5 frames with our MPEG-4 encoder, decode with ffmpeg, compare.
#[test]
fn encoder_roundtrip() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mpeg4_enc");
    let _ = std::fs::create_dir_all(&tmp);
    let ref_yuv = tmp.join("ref.yuv");

    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=10:duration=0.5",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        ref_yuv.to_str().unwrap(),
    ]));

    let raw = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    assert!(raw.len() >= NFRAMES * frame_sz);

    // Encode with our MPEG-4 encoder.
    let reg = oxideav::Registries::with_all_features();
    let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(10, 1));

    let mut enc = reg.codecs.make_encoder(&params).expect("make encoder");

    let mut es_data = Vec::new();
    for i in 0..NFRAMES {
        let frame = make_yuv_frame(&raw, i, W, H);
        enc.send_frame(&Frame::Video(frame)).expect("send_frame");
        loop {
            match enc.receive_packet() {
                Ok(p) => es_data.extend_from_slice(&p.data),
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("encoder error: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    while let Ok(p) = enc.receive_packet() {
        es_data.extend_from_slice(&p.data);
    }

    // Decode with ffmpeg.
    let es_path = tmp.join("ours.m4v");
    let decoded_yuv = tmp.join("decoded.yuv");
    std::fs::write(&es_path, &es_data).expect("write es");

    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "m4v",
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
        eprintln!("  [MPEG-4 encoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 25.0,
            "MPEG-4 encoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
        );
    }
}

/// Encode with ffmpeg, decode with both, compare Y-PSNR.
#[test]
fn decoder_vs_ffmpeg() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mpeg4_dec");
    let _ = std::fs::create_dir_all(&tmp);
    let avi_path = tmp.join("ffmpeg.avi");
    let ref_yuv = tmp.join("ref.yuv");

    // Encode with ffmpeg's MPEG-4 into AVI.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=10:duration=0.5",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-f",
        "avi",
        avi_path.to_str().unwrap(),
    ]));

    // Decode with ffmpeg for reference.
    assert!(oxideav_tests::ffmpeg(&[
        "-i",
        avi_path.to_str().unwrap(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        ref_yuv.to_str().unwrap(),
    ]));

    let ref_data = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    let ref_nframes = ref_data.len() / frame_sz;

    // Decode with our decoder.
    let reg = oxideav::Registries::with_all_features();
    let avi_data = std::fs::read(&avi_path).expect("read avi");
    let mut file: Box<dyn oxideav::container::ReadSeek> = Box::new(std::io::Cursor::new(avi_data));
    let format = reg
        .containers
        .probe_input(&mut *file, Some("avi"))
        .expect("probe");
    let mut dmx = reg
        .containers
        .open_demuxer(&format, file)
        .expect("open demuxer");

    let video_idx = dmx
        .streams()
        .iter()
        .position(|s| s.params.width.is_some())
        .expect("no video stream");
    let params = dmx.streams()[video_idx].params.clone();
    let mut dec = reg.codecs.make_decoder(&params).expect("make decoder");

    let mut our_frames: Vec<Vec<u8>> = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(pkt) => {
                if pkt.stream_index != video_idx as u32 {
                    continue;
                }
                dec.send_packet(&pkt).expect("send_packet");
                loop {
                    match dec.receive_frame() {
                        Ok(Frame::Video(v)) => {
                            let mut y = Vec::with_capacity((W * H) as usize);
                            for row in 0..v.height as usize {
                                let start = row * v.planes[0].stride;
                                y.extend_from_slice(
                                    &v.planes[0].data[start..start + v.width as usize],
                                );
                            }
                            our_frames.push(y);
                        }
                        Ok(_) => {}
                        Err(Error::NeedMore) => break,
                        Err(Error::Eof) => break,
                        Err(e) => panic!("decoder error: {e}"),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("demuxer error: {e}"),
        }
    }

    dec.flush().expect("flush");
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
            Err(_) => break,
        }
    }

    let count = our_frames.len().min(ref_nframes);
    eprintln!("  MPEG-4 decoder: decoded {count} frames (ref has {ref_nframes})");
    assert!(count > 0, "no frames decoded");

    for i in 0..count {
        let ref_y = &ref_data[i * frame_sz..i * frame_sz + (W * H) as usize];
        let our_y = &our_frames[i];
        let psnr = oxideav_tests::video_y_psnr(our_y, ref_y, W, H);
        eprintln!("  [MPEG-4 decoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 25.0,
            "MPEG-4 decoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
        );
    }
}
