//! Integration tests for the 8SVX muxer: build a FORM/8SVX file with the
//! registered muxer, read it back through the demuxer, and make sure the
//! PCM bytes and container metadata round-trip intact.

use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

use oxideav_container::{ContainerRegistry, Muxer, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, SampleFormat, StreamInfo, TimeBase,
};

use oxideav_iff::svx::SvxMuxer;

/// 200 ms of 8-bit signed sawtooth at 8 kHz mono = 1600 samples.
fn sawtooth_200ms_8khz() -> Vec<u8> {
    let sr = 8000u32;
    let total = (sr as u64 * 200 / 1000) as usize; // 1600
                                                   // Sawtooth: ramp -128..127 and wrap. Written as u8-encoded i8 bytes.
    (0..total)
        .map(|i| ((i as i32 * 5 - 128).rem_euclid(256) - 128) as i8 as u8)
        .collect()
}

fn build_stream(sr: u32) -> StreamInfo {
    let mut params = CodecParameters::audio(CodecId::new("pcm_s8"));
    params.media_type = MediaType::Audio;
    params.channels = Some(1);
    params.sample_rate = Some(sr);
    params.sample_format = Some(SampleFormat::S8);
    params.bit_rate = Some(8 * sr as u64);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, sr as i64),
        duration: None,
        start_time: Some(0),
        params,
    }
}

/// Build a registry populated with the iff crate's demuxer/muxer.
fn registry() -> ContainerRegistry {
    let mut reg = ContainerRegistry::new();
    oxideav_iff::register(&mut reg);
    reg
}

/// Returns a fresh path under `std::env::temp_dir()`. Tests in the
/// same process may run in parallel, so we use an atomic counter plus
/// the test's own name to keep writes disjoint.
fn tmp_path(tag: &str) -> std::path::PathBuf {
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxideav-iff-{tag}-{}-{n}.8svx", std::process::id()))
}

#[test]
fn mux_roundtrip_200ms_sawtooth() {
    let sr = 8000u32;
    let payload = sawtooth_200ms_8khz();
    assert_eq!(payload.len(), 1600);

    let stream = build_stream(sr);
    let reg = registry();
    let path = tmp_path("sawtooth");

    // Write via the registry-registered muxer.
    {
        let f = std::fs::File::create(&path).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = reg
            .open_muxer("iff_8svx", ws, std::slice::from_ref(&stream))
            .unwrap();
        assert_eq!(mux.format_name(), "iff_8svx");
        mux.write_header().unwrap();
        let pkt = Packet::new(0, stream.time_base, payload.clone());
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    // FORM/8SVX magic.
    assert_eq!(&bytes[0..4], b"FORM");
    assert_eq!(&bytes[8..12], b"8SVX");

    // Demux and compare the BODY bytes.
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes.clone()));
    let mut dmx = reg.open_demuxer("iff_8svx", rs).unwrap();
    assert_eq!(dmx.format_name(), "iff_8svx");
    let s = &dmx.streams()[0];
    assert_eq!(s.params.codec_id, CodecId::new("pcm_s8"));
    assert_eq!(s.params.sample_rate, Some(sr));
    assert_eq!(s.params.channels, Some(1));
    assert_eq!(s.params.sample_format, Some(SampleFormat::S8));
    assert_eq!(s.duration, Some(payload.len() as i64));

    let mut got = Vec::<u8>::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => got.extend_from_slice(&p.data),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(got, payload, "BODY bytes must round-trip verbatim");

    // Spot-check the VHDR fields by hand. VHDR lives at bytes 12..40.
    // (FORM header=8, "8SVX"=4, VHDR header=8, VHDR body=20.)
    let vhdr = &bytes[20..40];
    let one_shot = u32::from_be_bytes([vhdr[0], vhdr[1], vhdr[2], vhdr[3]]);
    assert_eq!(one_shot, payload.len() as u32, "oneShotHiSamples");
    let repeat = u32::from_be_bytes([vhdr[4], vhdr[5], vhdr[6], vhdr[7]]);
    assert_eq!(repeat, 0);
    let per_cycle = u32::from_be_bytes([vhdr[8], vhdr[9], vhdr[10], vhdr[11]]);
    assert_eq!(per_cycle, 0);
    let sps = u16::from_be_bytes([vhdr[12], vhdr[13]]);
    assert_eq!(sps, sr as u16);
    assert_eq!(vhdr[14], 1, "ctOctave");
    assert_eq!(vhdr[15], 0, "sCompression (none)");
    let vol = u32::from_be_bytes([vhdr[16], vhdr[17], vhdr[18], vhdr[19]]);
    assert_eq!(vol, 0x0001_0000, "volume 1.0 in 16.16 fixed-point");
}

#[test]
fn mux_roundtrip_with_name_chunk() {
    let sr = 8000u32;
    let payload = sawtooth_200ms_8khz();
    let stream = build_stream(sr);
    let metadata = vec![("title".to_string(), "test".to_string())];
    let path = tmp_path("name");

    {
        let f = std::fs::File::create(&path).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux =
            SvxMuxer::with_metadata(ws, std::slice::from_ref(&stream), &metadata).unwrap();
        mux.write_header().unwrap();
        mux.write_packet(&Packet::new(0, stream.time_base, payload.clone()))
            .unwrap();
        mux.write_trailer().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    // The NAME chunk must appear between VHDR and BODY.
    let name_pos = bytes
        .windows(4)
        .position(|w| w == b"NAME")
        .expect("NAME chunk");
    let body_pos = bytes
        .windows(4)
        .position(|w| w == b"BODY")
        .expect("BODY chunk");
    assert!(name_pos < body_pos, "NAME must precede BODY");

    let reg = registry();
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = reg.open_demuxer("iff_8svx", rs).unwrap();
    let md = dmx.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "title" && v == "test"),
        "title=\"test\" metadata must round-trip, got {:?}",
        md
    );

    // Body still decodes verbatim.
    let mut got = Vec::<u8>::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => got.extend_from_slice(&p.data),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(got, payload);
}

#[test]
fn mux_roundtrip_odd_length_adds_pad_byte() {
    // Odd-length BODY forces the IFF pad byte path; make sure the file
    // ends on an even boundary and the demuxer still returns exactly the
    // bytes we fed in (not the trailing pad).
    let sr = 8000u32;
    let payload: Vec<u8> = (0..1601u16).map(|i| (i as u8).wrapping_sub(128)).collect();
    assert_eq!(payload.len() % 2, 1);

    let stream = build_stream(sr);
    let path = tmp_path("odd");
    {
        let f = std::fs::File::create(&path).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = SvxMuxer::new(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        mux.write_packet(&Packet::new(0, stream.time_base, payload.clone()))
            .unwrap();
        mux.write_trailer().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(bytes.len() % 2, 0, "IFF files must end on an even boundary");

    let reg = registry();
    let rs: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    let mut dmx = reg.open_demuxer("iff_8svx", rs).unwrap();
    let mut got = Vec::<u8>::new();
    loop {
        match dmx.next_packet() {
            Ok(p) => got.extend_from_slice(&p.data),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(got, payload);
}

#[test]
fn muxer_rejects_wrong_codec() {
    let mut params = CodecParameters::audio(CodecId::new("pcm_s16le"));
    params.media_type = MediaType::Audio;
    params.channels = Some(1);
    params.sample_rate = Some(8000);
    params.sample_format = Some(SampleFormat::S16);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 8000),
        duration: None,
        start_time: Some(0),
        params,
    };
    let cur = Cursor::new(Vec::<u8>::new());
    let ws: Box<dyn WriteSeek> = Box::new(cur);
    match SvxMuxer::new(ws, std::slice::from_ref(&stream)) {
        Err(Error::Unsupported(_)) => {}
        Err(e) => panic!("expected Unsupported for non-pcm_s8 codec, got {e:?}"),
        Ok(_) => panic!("expected muxer construction to fail for non-pcm_s8"),
    }
}
