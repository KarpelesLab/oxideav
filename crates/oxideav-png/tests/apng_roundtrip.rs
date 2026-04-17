//! APNG encode/decode roundtrip: build a 3-frame animation, encode, parse
//! the APNG, decode each frame, and verify the frame count matches and each
//! frame's data is byte-identical to the input.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, Rational, TimeBase, VideoFrame, VideoPlane,
};

fn make_frame(idx: u8, w: u32, h: u32) -> VideoFrame {
    let bpp = 4usize;
    let mut data = vec![0u8; w as usize * h as usize * bpp];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let off = (y * w as usize + x) * bpp;
            // Each frame has a different colour pattern.
            data[off] = ((x + y) as u8).wrapping_add(idx.wrapping_mul(17));
            data[off + 1] = (x as u8).wrapping_mul(idx.wrapping_add(1));
            data[off + 2] = (y as u8).wrapping_mul(idx.wrapping_add(2));
            data[off + 3] = 255;
        }
    }
    VideoFrame {
        format: PixelFormat::Rgba,
        width: w,
        height: h,
        pts: Some(idx as i64),
        time_base: TimeBase::new(1, 100),
        planes: vec![VideoPlane {
            stride: w as usize * bpp,
            data,
        }],
    }
}

#[test]
fn apng_three_frames_roundtrip_byte_identical() {
    let w = 8u32;
    let h = 4u32;
    let mut params = CodecParameters::video(CodecId::new("png"));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Rgba);
    params.frame_rate = Some(Rational::new(10, 1));

    let mut enc = oxideav_png::encoder::make_encoder(&params).expect("make encoder");

    let frames = vec![
        make_frame(0, w, h),
        make_frame(1, w, h),
        make_frame(2, w, h),
    ];
    for f in &frames {
        enc.send_frame(&Frame::Video(f.clone())).expect("send");
    }
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("recv");

    // Parse the APNG.
    let info = oxideav_png::decoder::parse_apng(&pkt.data).expect("parse apng");
    assert_eq!(info.actl.num_frames, 3);
    assert_eq!(info.frames.len(), 3);

    let decoded = oxideav_png::decoder::decode_apng_frames(&info, TimeBase::new(1, 100))
        .expect("decode apng");
    assert_eq!(decoded.len(), 3);

    for (i, (orig, got)) in frames.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(got.width, w);
        assert_eq!(got.height, h);
        assert_eq!(
            got.planes[0].data, orig.planes[0].data,
            "frame {i} bytes differ"
        );
    }
}
