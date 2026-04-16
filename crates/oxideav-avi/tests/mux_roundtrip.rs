//! Muxer → Demuxer roundtrip tests. Deliberately independent from ffmpeg so
//! they can run in restricted CI environments.

use std::io::Read;

use oxideav_container::{ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, MediaType, Packet, PixelFormat, Rational, SampleFormat, StreamInfo,
    TimeBase,
};

fn pcm_stream() -> StreamInfo {
    let mut params = CodecParameters::audio(CodecId::new("pcm_s16le"));
    params.channels = Some(2);
    params.sample_rate = Some(48_000);
    params.sample_format = Some(SampleFormat::S16);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 48_000),
        duration: None,
        start_time: Some(0),
        params,
    }
}

fn make_pcm_payload(frames: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let l = (i as i16).wrapping_mul(7);
        let r = (i as i16).wrapping_mul(11);
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

#[test]
fn pcm_roundtrip_byte_exact() {
    let stream = pcm_stream();
    let frames_per_packet: usize = 1024;
    let total_packets = 4;
    let mut sent: Vec<Vec<u8>> = Vec::new();
    for i in 0..total_packets {
        sent.push(make_pcm_payload(frames_per_packet + i));
    }

    let tmp = std::env::temp_dir().join("oxideav-avi-pcm-roundtrip.avi");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_avi::muxer::open(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        for (i, payload) in sent.iter().enumerate() {
            let mut pkt = Packet::new(0, stream.time_base, payload.clone());
            pkt.pts = Some((i as i64) * frames_per_packet as i64);
            pkt.flags.keyframe = true;
            mux.write_packet(&pkt).unwrap();
        }
        mux.write_trailer().unwrap();
    }

    let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = oxideav_avi::demuxer::open(rs).unwrap();
    assert_eq!(dmx.format_name(), "avi");
    assert_eq!(dmx.streams().len(), 1);
    assert_eq!(dmx.streams()[0].params.codec_id, CodecId::new("pcm_s16le"));
    assert_eq!(dmx.streams()[0].params.channels, Some(2));
    assert_eq!(dmx.streams()[0].params.sample_rate, Some(48_000));

    let mut got: Vec<Vec<u8>> = Vec::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => got.push(p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e}"),
        }
    }
    assert_eq!(got.len(), sent.len());
    for (i, (g, s)) in got.iter().zip(sent.iter()).enumerate() {
        assert_eq!(g, s, "packet {i} byte mismatch");
    }
}

#[test]
fn mjpeg_roundtrip_via_avi() {
    use oxideav_core::{Frame, VideoFrame, VideoPlane};

    // Build a synthetic 64x64 Yuv420P frame.
    let w = 64u32;
    let h = 64u32;
    let chroma_w = (w / 2) as usize;
    let chroma_h = (h / 2) as usize;
    let y_plane: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
    let cb_plane: Vec<u8> = vec![128u8; chroma_w * chroma_h];
    let cr_plane: Vec<u8> = vec![128u8; chroma_w * chroma_h];

    let time_base = TimeBase::new(1, 25);
    let frame = Frame::Video(VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base,
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: cb_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: cr_plane,
            },
        ],
    });

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.media_type = MediaType::Video;
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Yuv420P);
    enc_params.frame_rate = Some(Rational::new(25, 1));

    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).unwrap();
    enc.send_frame(&frame).unwrap();
    let jpeg_bytes = match enc.receive_packet() {
        Ok(p) => p.data,
        Err(e) => panic!("mjpeg encoder produced no packet: {e:?}"),
    };
    assert_eq!(&jpeg_bytes[0..2], &[0xFF, 0xD8]);

    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: None,
        start_time: Some(0),
        params: enc_params.clone(),
    };
    let tmp = std::env::temp_dir().join("oxideav-avi-mjpeg-roundtrip.avi");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_avi::muxer::open(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        let mut pkt = Packet::new(0, time_base, jpeg_bytes.clone());
        pkt.pts = Some(0);
        pkt.flags.keyframe = true;
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }

    let rs: Box<dyn ReadSeek> = Box::new(std::fs::File::open(&tmp).unwrap());
    let mut dmx = oxideav_avi::demuxer::open(rs).unwrap();
    assert_eq!(dmx.streams()[0].params.codec_id.as_str(), "mjpeg");
    assert_eq!(dmx.streams()[0].params.width, Some(w));
    assert_eq!(dmx.streams()[0].params.height, Some(h));

    let out = dmx.next_packet().unwrap();
    assert_eq!(out.data, jpeg_bytes, "MJPEG bytes preserved across AVI");
    assert!(matches!(dmx.next_packet(), Err(oxideav_core::Error::Eof)));
}

#[test]
fn unsupported_codec_errors_at_open() {
    use std::io::Cursor;

    let mut params = CodecParameters::audio(CodecId::new("opus"));
    params.channels = Some(2);
    params.sample_rate = Some(48_000);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 48_000),
        duration: None,
        start_time: Some(0),
        params,
    };
    let cursor: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    match oxideav_avi::muxer::open(cursor, &[stream]) {
        Err(oxideav_core::Error::Unsupported(_)) => {}
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(_) => panic!("expected Unsupported"),
    }
}

#[test]
fn muxer_writes_idx1_chunk() {
    let stream = pcm_stream();
    let tmp = std::env::temp_dir().join("oxideav-avi-has-idx1.avi");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_avi::muxer::open(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        for i in 0..3 {
            let payload = make_pcm_payload(512);
            let mut pkt = Packet::new(0, stream.time_base, payload);
            pkt.pts = Some(i * 512);
            pkt.flags.keyframe = true;
            mux.write_packet(&pkt).unwrap();
        }
        mux.write_trailer().unwrap();
    }

    // Read the whole file and grep for the idx1 marker.
    let mut buf = Vec::new();
    std::fs::File::open(&tmp)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let pos = buf
        .windows(4)
        .position(|w| w == b"idx1")
        .expect("idx1 chunk present");
    // idx1 should come AFTER movi. Locate movi, ensure it precedes idx1.
    let movi_pos = buf
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi list present");
    assert!(
        pos > movi_pos,
        "idx1 must follow movi (movi@{movi_pos}, idx1@{pos})"
    );

    // Sanity-check that the idx1 body has the expected size: 3 entries * 16.
    let idx1_size_off = pos + 4;
    let size = u32::from_le_bytes([
        buf[idx1_size_off],
        buf[idx1_size_off + 1],
        buf[idx1_size_off + 2],
        buf[idx1_size_off + 3],
    ]);
    assert_eq!(size as usize, 3 * 16);
}
