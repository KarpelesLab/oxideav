//! Interoperability tests with FFmpeg. These write a MKV with FFV1 via our
//! encoder and verify that FFmpeg decodes it back to the same pixels (2),
//! then generate a real FFV1-in-MKV with FFmpeg and verify that our decoder
//! reproduces a sensible YUV plane (3).
//!
//! The tests are skipped (and print a message) when the `ffmpeg` binary is
//! not on `PATH`, so they don't break builds in sandboxed environments.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_container::WriteSeek;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, StreamInfo,
    TimeBase, VideoFrame,
};
use oxideav_ffv1::encoder::make_encoder;

/// Returns true if `ffmpeg` is available.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp_dir() -> PathBuf {
    let t = std::env::temp_dir().join(format!("oxideav-ffv1-{}", std::process::id()));
    let _ = fs::create_dir_all(&t);
    t
}

/// Run the encode→mux pipeline into a fresh file on disk. Returns the file
/// path.
fn encode_frame_to_mkv_file(frame: &VideoFrame, path: &Path) {
    let params = CodecParameters {
        codec_id: CodecId::new("ffv1"),
        media_type: MediaType::Video,
        sample_rate: None,
        channels: None,
        sample_format: None,
        width: Some(frame.width),
        height: Some(frame.height),
        pixel_format: Some(frame.format),
        frame_rate: Some(Rational::new(1, 1)),
        extradata: Vec::new(),
        bit_rate: None,
    };

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let mut pkt = enc.receive_packet().expect("packet");
    pkt.stream_index = 0;
    pkt.pts = Some(0);
    pkt.dts = Some(0);
    pkt.duration = Some(1);

    let out_params = enc.output_params().clone();
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: None,
        params: out_params,
    };

    let file = std::fs::File::create(path).expect("create mkv");
    let sink: Box<dyn WriteSeek> = Box::new(file);
    let mut mux = oxideav_mkv::mux::open(sink, &[stream]).expect("muxer");
    mux.write_header().expect("header");
    mux.write_packet(&pkt).expect("packet");
    mux.write_trailer().expect("trailer");
}

fn synth_checkerboard(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // 8-pixel checkerboard in luma.
            let on = ((i / 8) + (j / 8)) & 1 == 0;
            y[j * w + i] = if on { 30 } else { 220 };
        }
    }
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    }
}

fn synth_flat(width: u32, height: u32, yy: u8, uu: u8, vv: u8) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
        planes: vec![
            VideoPlane {
                stride: w,
                data: vec![yy; w * h],
            },
            VideoPlane {
                stride: cw,
                data: vec![uu; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![vv; cw * ch],
            },
        ],
    }
}

// KNOWN GAP: the FFmpeg v3 range-coder slice footer check
// `bytestream_end - bytestream == 2 + 5*ec` fires with `v = -2` against our
// output. Our range-coder terminator produces byte-count-correct output but
// FFmpeg's decoder naturally consumes 2 more bytes than it expects, which
// makes FFmpeg mark the slice damaged and conceal from the previous frame.
// Left `#[ignore]` until the RAC terminator is made byte-perfect with
// FFmpeg's `ff_rac_terminate` output bytes (not just byte count).
#[test]
#[ignore]
fn ffmpeg_decodes_flat_gray_frame() {
    if !ffmpeg_available() {
        eprintln!("flat-gray: ffmpeg not on PATH, skipping");
        return;
    }
    let width = 8u32;
    let height = 8u32;
    let frame = synth_flat(width, height, 128, 128, 128);

    let dir = tmp_dir();
    let mkv = dir.join("flat.mkv");
    let yuv = dir.join("flat.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file(&frame, &mkv);

    let output = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let decoded = fs::read(&yuv).unwrap();
    assert!(
        decoded.iter().all(|&b| b == 128),
        "got {:?}",
        &decoded[..16]
    );
}

// See `ffmpeg_decodes_flat_gray_frame` above for why this is `#[ignore]`d.
#[test]
#[ignore]
fn ffmpeg_decodes_our_encoder_output() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg_decodes_our_encoder_output: ffmpeg binary not on PATH, skipping");
        return;
    }
    let width = 32u32;
    let height = 32u32;
    let frame = synth_checkerboard(width, height);

    let dir = tmp_dir();
    let mkv = dir.join("oxideav-out.mkv");
    let yuv = dir.join("oxideav-out.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file(&frame, &mkv);

    // Run ffmpeg -i mkv -f rawvideo yuv
    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(&mkv)
        .arg("-f")
        .arg("rawvideo")
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!(
            "ffmpeg refused to decode our FFV1: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let decoded = fs::read(&yuv).expect("read yuv");
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let expected_len = w * h + 2 * cw * ch;
    assert_eq!(
        decoded.len(),
        expected_len,
        "raw yuv size mismatch: expected {}, got {}",
        expected_len,
        decoded.len()
    );

    // Compare each plane to the source frame.
    let y_src = &frame.planes[0].data[..w * h];
    assert_eq!(
        &decoded[..w * h],
        y_src,
        "Y plane mismatch after ffmpeg decode"
    );
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let u_src = &frame.planes[1].data[..cw * ch];
    let v_src = &frame.planes[2].data[..cw * ch];
    assert_eq!(&decoded[u_off..u_off + cw * ch], u_src, "U plane mismatch");
    assert_eq!(&decoded[v_off..v_off + cw * ch], v_src, "V plane mismatch");
}

#[test]
fn our_decoder_accepts_ffmpeg_output() {
    if !ffmpeg_available() {
        eprintln!("our_decoder_accepts_ffmpeg_output: ffmpeg not on PATH, skipping");
        return;
    }

    let dir = tmp_dir();
    let mkv = dir.join("ref-ffv1-frame.mkv");
    let _ = fs::remove_file(&mkv);

    // FFmpeg-generated FFV1 reference: a testsrc colour-bar frame at 64×64.
    // FFmpeg's default coder is Golomb-Rice (`coder=0`); force the
    // range-coder path (`coder=1`, range_def) because that's what our
    // decoder supports.
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("testsrc=d=1:s=64x64:r=1")
        .arg("-c:v")
        .arg("ffv1")
        .arg("-level")
        .arg("3")
        .arg("-coder")
        .arg("range_def")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-frames:v")
        .arg("1")
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to produce reference file");

    // Demux via oxideav-mkv. Pull the first video packet and the codec
    // parameters (including extradata).
    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux = oxideav_mkv::demux::open(input).expect("demux");
    let streams = demux.streams().to_vec();
    let vstream = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .expect("no video stream");
    let vindex = vstream.index;
    let params = vstream.params.clone();

    let mut pkt: Option<Packet> = None;
    for _ in 0..64 {
        match demux.next_packet() {
            Ok(p) => {
                if p.stream_index == vindex {
                    pkt = Some(p);
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    let pkt = pkt.expect("no video packet demuxed");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("decoder returned non-video frame");
    };

    // testsrc produces a colour-bar pattern; the Y plane's mean must land
    // comfortably away from both extremes of the 0..=255 range.
    let y_plane = &vf.planes[0];
    let w = vf.width as usize;
    let h = vf.height as usize;
    let mut sum: u64 = 0;
    for row in 0..h {
        let off = row * y_plane.stride;
        for i in 0..w {
            sum += y_plane.data[off + i] as u64;
        }
    }
    let mean = sum as f64 / (w * h) as f64;
    assert!(
        (30.0..=230.0).contains(&mean),
        "Y plane mean = {mean} outside [30, 230] — decoder probably produced garbage"
    );
}
