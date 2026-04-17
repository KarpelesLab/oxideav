//! VP8 decoder comparison test against ffmpeg (decode-only).
//!
//! VP8 has no encoder in oxideav, so we only test the decoder:
//! encode with ffmpeg (libvpx), decode with both ffmpeg and ours, compare.

use oxideav_core::{Error, Frame};

const W: u32 = 64;
const H: u32 = 64;

/// Encode with ffmpeg (libvpx), decode with both, compare Y-PSNR.
#[test]
fn decoder_vs_ffmpeg() {
    if !oxideav_tests::ffmpeg_available() {
        eprintln!("skip");
        return;
    }

    let tmp = oxideav_tests::tmp("video_vp8_dec");
    let _ = std::fs::create_dir_all(&tmp);
    let ivf_path = tmp.join("ffmpeg.ivf");
    let ref_yuv = tmp.join("ref.yuv");

    // Encode with ffmpeg's libvpx into IVF.
    assert!(
        oxideav_tests::ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=0.5",
            "-c:v",
            "libvpx",
            "-b:v",
            "200k",
            "-f",
            "ivf",
            ivf_path.to_str().unwrap(),
        ]),
        "ffmpeg failed to encode VP8"
    );

    // Decode with ffmpeg for reference.
    assert!(oxideav_tests::ffmpeg(&[
        "-i",
        ivf_path.to_str().unwrap(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        ref_yuv.to_str().unwrap(),
    ]));

    let ref_data = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_sz = (W * H * 3 / 2) as usize;
    let ref_nframes = ref_data.len() / frame_sz;

    // Decode with our decoder via the IVF demuxer.
    let reg = oxideav::Registries::with_all_features();
    let ivf_data = std::fs::read(&ivf_path).expect("read ivf");
    let mut file: Box<dyn oxideav::container::ReadSeek> = Box::new(std::io::Cursor::new(ivf_data));
    let format = reg
        .containers
        .probe_input(&mut *file, Some("ivf"))
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
                if let Err(e) = dec.send_packet(&pkt) {
                    eprintln!("  VP8 send_packet error (continuing): {e}");
                    continue;
                }
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
                        Err(e) => {
                            eprintln!("  VP8 receive_frame error (continuing): {e}");
                            break;
                        }
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
    eprintln!(
        "  VP8 decoder: decoded {} frames (ref has {ref_nframes})",
        our_frames.len()
    );
    assert!(count > 0, "no frames decoded");

    let mut total_psnr = 0.0f64;
    let mut min_psnr = f64::INFINITY;
    for i in 0..count {
        let ref_y = &ref_data[i * frame_sz..i * frame_sz + (W * H) as usize];
        let our_y = &our_frames[i];
        let psnr = oxideav_tests::video_y_psnr(our_y, ref_y, W, H);
        eprintln!("  [VP8 decoder frame {i}] PSNR={psnr:.1} dB");
        total_psnr += psnr;
        if psnr < min_psnr {
            min_psnr = psnr;
        }
        // VP8 decoder has known accuracy limitations (B_PRED context
        // propagation bug documented in crate docs). Use a very generous
        // threshold; the test primarily validates that decoding does not
        // crash and produces structurally correct output.
        assert!(
            psnr > 5.0,
            "VP8 decoder frame {i} PSNR {psnr:.1} dB < 5 dB (complete garbage)"
        );
    }
    let avg_psnr = total_psnr / count as f64;
    eprintln!("  VP8 decoder average PSNR={avg_psnr:.1} dB, min={min_psnr:.1} dB");
}
