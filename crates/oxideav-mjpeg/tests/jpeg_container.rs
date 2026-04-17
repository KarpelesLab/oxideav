//! Round-trip test for the still-image JPEG container.
//!
//! Encodes a synthetic 64x64 YUV420P frame through the MJPEG encoder, writes
//! the resulting packet to a buffer via the `jpeg` muxer, reads it back via
//! the `jpeg` demuxer, decodes the packet, and checks that the luma-plane
//! PSNR stays above the same 35 dB floor the regular MJPEG round-trip tests
//! use.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use oxideav_container::{ContainerRegistry, ProbeData};
use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational, TimeBase, VideoFrame};

fn make_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let y_stride = w as usize;
    let mut y = vec![0u8; y_stride * h as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            y[j * y_stride + i] = (((i + j) * 2) % 255) as u8;
        }
    }
    let cb_stride = cw as usize;
    let cr_stride = cw as usize;
    let mut cb = vec![0u8; cb_stride * ch as usize];
    let mut cr = vec![0u8; cr_stride * ch as usize];
    for j in 0..ch as usize {
        for i in 0..cw as usize {
            cb[j * cb_stride + i] = ((128 + (i as i32 - cw as i32 / 2)) as u8).clamp(0, 255);
            cr[j * cr_stride + i] = ((128 + (j as i32 - ch as i32 / 2)) as u8).clamp(0, 255);
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: y_stride,
                data: y,
            },
            VideoPlane {
                stride: cb_stride,
                data: cb,
            },
            VideoPlane {
                stride: cr_stride,
                data: cr,
            },
        ],
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: f64 = 0.0;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return 99.0;
    }
    let mse = sse / a.len() as f64;
    20.0 * (255.0_f64 / mse.sqrt()).log10()
}

#[test]
fn probe_recognises_jpeg_magic() {
    // Register containers into a fresh registry to exercise the public API.
    let mut containers = ContainerRegistry::new();
    oxideav_mjpeg::register_containers(&mut containers);

    // The probe fn is exposed directly for symmetry with other crates.
    let p_good = ProbeData {
        buf: &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10],
        ext: None,
    };
    assert_eq!(oxideav_mjpeg::container::probe(&p_good), 100);

    let p_bad = ProbeData {
        buf: &[0x00, 0x00, 0x00, 0x00],
        ext: None,
    };
    assert_eq!(oxideav_mjpeg::container::probe(&p_bad), 0);
}

#[test]
fn mux_then_demux_roundtrip() {
    let w = 64u32;
    let h = 64u32;

    // --- Encode a synthetic frame ------------------------------------------
    let frame = make_gradient_frame(w, h);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).expect("enc");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let encoded_pkt = enc.receive_packet().expect("recv");

    // --- Mux the packet into an in-memory buffer ---------------------------
    let mut containers = ContainerRegistry::new();
    oxideav_mjpeg::register_containers(&mut containers);

    // The Muxer API takes ownership of the output. Wrap an Arc<Mutex<Cursor>>
    // so the test can read the produced bytes back out after write_trailer().
    let shared = Arc::new(Mutex::new(Cursor::new(Vec::<u8>::new())));
    let writer = SharedWriter(Arc::clone(&shared));
    let stream = oxideav_core::StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: Some(1),
        start_time: Some(0),
        params: enc_params.clone(),
    };
    let mut muxer = containers
        .open_muxer("jpeg", Box::new(writer), &[stream.clone()])
        .expect("muxer");
    muxer.write_header().expect("write_header");
    muxer.write_packet(&encoded_pkt).expect("write_packet");
    muxer.write_trailer().expect("write_trailer");
    drop(muxer);

    let muxed: Vec<u8> = Arc::try_unwrap(shared)
        .expect("no lingering writer refs")
        .into_inner()
        .expect("lock")
        .into_inner();
    // The muxer is a JPEG pass-through, so the file bytes equal the packet
    // payload. Verify that explicitly — if the muxer ever grows a header/
    // trailer, this check will catch it.
    assert_eq!(muxed, encoded_pkt.data);

    // Sanity: file starts with SOI and ends with EOI.
    assert_eq!(muxed[0], 0xFF);
    assert_eq!(muxed[1], 0xD8);
    assert_eq!(muxed[muxed.len() - 2], 0xFF);
    assert_eq!(muxed[muxed.len() - 1], 0xD9);

    // Append trailing garbage to prove the demuxer trims it.
    let mut stored: Vec<u8> = muxed.clone();
    stored.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

    // --- Demux + decode ----------------------------------------------------
    let input_buf: Cursor<Vec<u8>> = Cursor::new(stored);
    let mut demuxer = containers
        .open_demuxer("jpeg", Box::new(input_buf))
        .expect("demuxer");
    let streams = demuxer.streams();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].params.codec_id.as_str(), "mjpeg");
    assert_eq!(streams[0].params.width, Some(w));
    assert_eq!(streams[0].params.height, Some(h));

    let pkt = demuxer.next_packet().expect("packet");
    // Second call must return EOF — single-frame container.
    match demuxer.next_packet() {
        Err(oxideav_core::Error::Eof) => {}
        other => panic!("expected Eof on second next_packet, got {:?}", other),
    }

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&pkt).expect("send");
    let out = dec.receive_frame().expect("decode");
    let Frame::Video(v) = out else {
        panic!("expected video frame")
    };

    assert_eq!(v.width, w);
    assert_eq!(v.height, h);
    assert_eq!(v.format, PixelFormat::Yuv420P);

    // PSNR on the Y plane (visible area).
    let sw = w as usize;
    let sh = h as usize;
    let mut original = Vec::with_capacity(sw * sh);
    let mut decoded = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            original.push(frame.planes[0].data[j * frame.planes[0].stride + i]);
            decoded.push(v.planes[0].data[j * v.planes[0].stride + i]);
        }
    }
    let psnr_y = psnr(&original, &decoded);
    eprintln!("jpeg container roundtrip PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

/// Write/Seek adapter backed by a shared in-memory `Cursor<Vec<u8>>`, used
/// so the test can reclaim the muxer's output after the muxer is dropped.
struct SharedWriter(Arc<Mutex<Cursor<Vec<u8>>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

impl Seek for SharedWriter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}
