//! A synthetic PNG with a corrupted chunk CRC must produce an error rather
//! than be silently accepted.

use oxideav_core::Error;

#[test]
fn bad_chunk_crc_is_rejected() {
    // Build a minimal valid 2x2 Gray8 PNG via the encoder, then flip a byte
    // inside one of its chunk CRCs.
    use oxideav_core::{
        CodecId, CodecParameters, Frame, PixelFormat, TimeBase, VideoFrame, VideoPlane,
    };

    let mut params = CodecParameters::video(CodecId::new("png"));
    params.width = Some(2);
    params.height = Some(2);
    params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_png::encoder::make_encoder(&params).unwrap();
    enc.send_frame(&Frame::Video(VideoFrame {
        format: PixelFormat::Gray8,
        width: 2,
        height: 2,
        pts: Some(0),
        time_base: TimeBase::new(1, 100),
        planes: vec![VideoPlane {
            stride: 2,
            data: vec![10u8, 20, 30, 40],
        }],
    }))
    .unwrap();
    enc.flush().unwrap();
    let mut bytes = enc.receive_packet().unwrap().data;

    // Locate the IDAT chunk's CRC (last 4 bytes of its chunk, which is
    // the next-to-last chunk in the file — IEND is always last). Easier:
    // flip the IHDR CRC which sits at byte offset 8+4+4+13 = 29..33.
    // IHDR: 8 (magic) + 4 (len) + 4 (type) + 13 (data) = 29, then 4 bytes of CRC.
    bytes[30] ^= 0x01;

    let err =
        oxideav_png::decoder::decode_png_to_frame(&bytes, None, TimeBase::new(1, 100)).unwrap_err();
    match err {
        Error::InvalidData(msg) => assert!(msg.contains("CRC"), "expected CRC error, got: {msg}"),
        other => panic!("expected InvalidData, got: {other:?}"),
    }
}
