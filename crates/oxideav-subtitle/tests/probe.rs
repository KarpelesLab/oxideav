//! Container probe scoring.

use std::io::Cursor;

use oxideav_container::ContainerRegistry;
use oxideav_subtitle;

fn make_registry() -> ContainerRegistry {
    let mut reg = ContainerRegistry::new();
    oxideav_subtitle::register_containers(&mut reg);
    reg
}

#[test]
fn webvtt_probes_top() {
    let reg = make_registry();
    let mut c = Cursor::new(b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n".to_vec());
    let name = reg.probe_input(&mut c, Some("vtt")).unwrap();
    assert_eq!(name, "webvtt");
}

#[test]
fn srt_probes_srt() {
    let reg = make_registry();
    let mut c = Cursor::new(b"1\n00:00:01,000 --> 00:00:02,000\nhi\n\n".to_vec());
    let name = reg.probe_input(&mut c, Some("srt")).unwrap();
    assert_eq!(name, "srt");
}

// ASS probe test moved to the `oxideav-ass` crate.

#[test]
fn demuxer_yields_one_packet_per_cue() {
    let reg = make_registry();
    let src =
        b"1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n2\n00:00:03,000 --> 00:00:04,500\nsecond\n"
            .to_vec();
    let mut c = Cursor::new(src);
    let name = reg.probe_input(&mut c, Some("srt")).unwrap();
    let mut dmx = reg.open_demuxer(&name, Box::new(c)).unwrap();
    let p1 = dmx.next_packet().unwrap();
    assert_eq!(p1.pts, Some(1_000_000));
    let p2 = dmx.next_packet().unwrap();
    assert_eq!(p2.pts, Some(3_000_000));
    assert!(dmx.next_packet().is_err());
}

#[test]
fn mux_srt_reemits_cues() {
    use oxideav_codec::CodecRegistry;
    use oxideav_core::{CodecId, CodecParameters, Frame, MediaType};

    let mut codecs = CodecRegistry::new();
    oxideav_subtitle::register_codecs(&mut codecs);

    let reg = make_registry();
    // Demux a source first.
    let src =
        b"1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n2\n00:00:03,000 --> 00:00:04,500\nsecond\n"
            .to_vec();
    let c = Cursor::new(src);
    let mut dmx = reg.open_demuxer("srt", Box::new(c)).unwrap();
    let stream = dmx.streams()[0].clone();
    let mut packets = Vec::new();
    while let Ok(p) = dmx.next_packet() {
        packets.push(p);
    }
    assert_eq!(packets.len(), 2);

    // Re-mux.
    let buf = Cursor::new(Vec::<u8>::new());
    let mut mux = reg
        .open_muxer("srt", Box::new(buf), std::slice::from_ref(&stream))
        .unwrap();
    mux.write_header().unwrap();
    for p in &packets {
        mux.write_packet(p).unwrap();
    }
    mux.write_trailer().unwrap();
    drop(mux);

    // Decode packets through the codec for good measure.
    let dec_params = CodecParameters {
        codec_id: CodecId::new("subrip"),
        media_type: MediaType::Subtitle,
        sample_rate: None,
        channels: None,
        sample_format: None,
        width: None,
        height: None,
        pixel_format: None,
        frame_rate: None,
        extradata: Vec::new(),
        bit_rate: None,
    };
    let mut dec = codecs.make_decoder(&dec_params).unwrap();
    dec.send_packet(&packets[0]).unwrap();
    let f = dec.receive_frame().unwrap();
    match f {
        Frame::Subtitle(cue) => {
            assert_eq!(cue.start_us, 1_000_000);
        }
        _ => panic!("expected subtitle frame"),
    }
}
