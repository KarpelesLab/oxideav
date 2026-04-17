//! MJPEG roundtrip comparison tests against ffmpeg.
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

/// Encode 5 frames with our MJPEG encoder, decode with ffmpeg, compare PSNR.
#[test]
fn encoder_roundtrip() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mjpeg_enc");
    let _ = std::fs::create_dir_all(&tmp);
    let ref_yuv = tmp.join("ref.yuv");

    // Generate test pattern.
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

    // Encode each frame as a separate JPEG (MJPEG is per-frame).
    let reg = oxideav::Registries::with_all_features();
    let mut all_encoded = Vec::new();
    for i in 0..NFRAMES {
        let frame = make_yuv_frame(&raw, i, W, H);
        let mut params = CodecParameters::video(CodecId::new("mjpeg"));
        params.width = Some(W);
        params.height = Some(H);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        params.frame_rate = Some(Rational::new(10, 1));
        let mut enc = reg.codecs.make_encoder(&params).expect("make encoder");
        enc.send_frame(&Frame::Video(frame)).expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        all_encoded.push(pkt.data);
    }

    // Decode each JPEG frame individually with ffmpeg and compare.
    for (i, encoded_jpeg) in all_encoded.iter().enumerate() {
        let jpeg_path = tmp.join(format!("frame_{i}.jpg"));
        let frame_yuv = tmp.join(format!("frame_{i}.yuv"));
        std::fs::write(&jpeg_path, encoded_jpeg).expect("write jpeg");

        assert!(oxideav_tests::ffmpeg(&[
            "-f",
            "mjpeg",
            "-i",
            jpeg_path.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            frame_yuv.to_str().unwrap(),
        ]));

        let decoded = std::fs::read(&frame_yuv).expect("read decoded yuv");
        let orig_offset = i * frame_sz;
        let orig_y = &raw[orig_offset..orig_offset + (W * H) as usize];
        let decoded_y = &decoded[..(W * H) as usize];
        let psnr = oxideav_tests::video_y_psnr(orig_y, decoded_y, W, H);
        eprintln!(
            "  [MJPEG encoder frame {i}] PSNR={psnr:.1} dB  encoded={} bytes",
            encoded_jpeg.len()
        );
        assert!(
            psnr > 25.0,
            "MJPEG encoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
        );
    }
}

/// Encode with ffmpeg's MJPEG, decode with both ffmpeg and ours, compare.
#[test]
fn decoder_vs_ffmpeg() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_mjpeg_dec");
    let _ = std::fs::create_dir_all(&tmp);
    let avi_path = tmp.join("ffmpeg.avi");
    let ref_yuv = tmp.join("ref.yuv");

    // Generate test content and encode with ffmpeg's MJPEG into AVI.
    // Force yuv420p to avoid chroma subsampling modes our decoder doesn't support.
    assert!(oxideav_tests::ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=10:duration=0.5",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        "-f",
        "avi",
        avi_path.to_str().unwrap(),
    ]));

    // Decode with ffmpeg to get reference YUV.
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

    // Decode with our decoder via the registry.
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

    // Find the video stream.
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
                            // Extract Y plane.
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

    // Flush decoder.
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

    let decoded_count = our_frames.len().min(ref_nframes);
    eprintln!("  MJPEG decoder: decoded {decoded_count} frames (ref has {ref_nframes})");
    assert!(decoded_count > 0, "no frames decoded");

    for i in 0..decoded_count {
        let ref_y = &ref_data[i * frame_sz..i * frame_sz + (W * H) as usize];
        let our_y = &our_frames[i];
        let psnr = oxideav_tests::video_y_psnr(our_y, ref_y, W, H);
        eprintln!("  [MJPEG decoder frame {i}] PSNR={psnr:.1} dB");
        assert!(
            psnr > 25.0,
            "MJPEG decoder frame {i} PSNR {psnr:.1} dB < 25 dB threshold"
        );
    }
}
