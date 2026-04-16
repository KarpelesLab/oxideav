//! Synthetic encode→decode roundtrip tests.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
};

fn make_gradient_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (u32, u32) = match pix {
        PixelFormat::Yuv444P => (w, h),
        PixelFormat::Yuv422P => (w.div_ceil(2), h),
        PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
        _ => panic!("unsupported for this test"),
    };
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
        format: pix,
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

fn run_roundtrip(w: u32, h: u32, pix: PixelFormat) -> f64 {
    let frame = make_gradient_frame(w, h, pix);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).expect("enc");

    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    assert!(pkt.data.len() >= 10);
    assert_eq!(pkt.data[0], 0xFF);
    assert_eq!(pkt.data[1], 0xD8, "SOI");
    assert_eq!(pkt.data[pkt.data.len() - 2], 0xFF);
    assert_eq!(pkt.data[pkt.data.len() - 1], 0xD9, "EOI");

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");

    let in_pkt = Packet::new(0, TimeBase::new(1, 30), pkt.data.clone());
    dec.send_packet(&in_pkt).expect("send");
    let out = dec.receive_frame().expect("decode");
    let Frame::Video(v) = out else {
        panic!("expected video");
    };

    assert_eq!(v.width, w);
    assert_eq!(v.height, h);
    assert_eq!(v.format, pix);

    // Compute PSNR across Y plane (visible area) — it's the most perceptually
    // important and has the highest resolution.
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
    psnr(&original, &decoded)
}

#[test]
fn roundtrip_yuv420p_320x240_gradient() {
    let psnr_y = run_roundtrip(320, 240, PixelFormat::Yuv420P);
    eprintln!("yuv420p 320x240 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

#[test]
fn roundtrip_yuv422p_128x64() {
    let psnr_y = run_roundtrip(128, 64, PixelFormat::Yuv422P);
    eprintln!("yuv422p 128x64 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

#[test]
fn roundtrip_yuv444p_96x48() {
    let psnr_y = run_roundtrip(96, 48, PixelFormat::Yuv444P);
    eprintln!("yuv444p 96x48 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

/// Synthetic image designed to produce plenty of 0xFF bytes in the entropy
/// stream. All samples at the mid-point cancel to low-entropy DC — but the
/// very-high-quality tables used by Annex K make lots of long Huffman codes.
#[test]
fn byte_stuffing_roundtrip() {
    // Noisy random-ish texture stresses the entropy coder.
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; (w * h) as usize];
    for j in 0..h {
        for i in 0..w {
            // LCG-style pseudo-random.
            let mut x = (i
                .wrapping_mul(1103515245)
                .wrapping_add(j.wrapping_mul(12345)))
                & 0xFF;
            x ^= (i * j) & 0xFF;
            y[(j * w + i) as usize] = x as u8;
        }
    }
    let cb = vec![128u8; (cw * ch) as usize];
    let cr = vec![128u8; (cw * ch) as usize];
    let frame = VideoFrame {
        format: pix,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y.clone(),
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
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();

    // Verify the stream contains a stuffed 0xFF00 somewhere (i.e., byte
    // stuffing actually happened). Skip the header (before SOS).
    let sos_pos = pkt
        .data
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS found");
    let scan = &pkt.data[sos_pos + 2..pkt.data.len() - 2];
    let stuffs = scan
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0x00)
        .count();
    assert!(stuffs > 0, "expected at least one 0xFF00 stuff");

    // Decode and check dimensions.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    let in_pkt = Packet::new(0, TimeBase::new(1, 30), pkt.data);
    dec.send_packet(&in_pkt).unwrap();
    let out = dec.receive_frame().unwrap();
    let Frame::Video(v) = out else {
        panic!();
    };
    assert_eq!(v.width, w);
    assert_eq!(v.height, h);
}
