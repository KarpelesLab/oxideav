//! Vorbis roundtrip comparison tests against ffmpeg.
//!
//! Vorbis is lossy, so we use relaxed thresholds.

use oxideav_container::{ReadSeek, WriteSeek};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat, StreamInfo, TimeBase,
};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;

/// Encode PCM with our Vorbis encoder and wrap in an Ogg container.
fn encode_with_ours(pcm: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = oxideav_vorbis::encoder::make_encoder(&params).expect("make vorbis encoder");

    let block_size = 2048usize;
    let stride = channels as usize;

    for chunk in pcm.chunks(block_size * stride) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame {
            format: SampleFormat::S16,
            sample_rate,
            channels,
            samples: (chunk.len() / stride) as u32,
            data: vec![bytes],
            pts: None,
            time_base: TimeBase::new(1, sample_rate as i64),
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send");
    }
    enc.flush().expect("flush");

    let out_params = enc.output_params().clone();
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(Error::NeedMore | Error::Eof) => break,
            Err(e) => panic!("encode error: {e:?}"),
        }
    }

    // Write via Ogg muxer to a temp file
    let mux_path = tmp("oxideav-vorbis-mux-tmp.ogg");
    {
        let reg = oxideav::Registries::with_all_features();
        let stream = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, sample_rate as i64),
            duration: None,
            start_time: Some(0),
            params: out_params,
        };
        let f = std::fs::File::create(&mux_path).expect("create mux file");
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = reg
            .containers
            .open_muxer("ogg", ws, &[stream])
            .expect("open ogg muxer");
        mux.write_header().expect("write header");
        for pkt in &packets {
            mux.write_packet(pkt).expect("write packet");
        }
        mux.write_trailer().expect("write trailer");
    }
    std::fs::read(&mux_path).expect("read muxed ogg")
}

/// Decode an Ogg/Vorbis file with our decoder via demuxer.
fn decode_with_ours(ogg_data: &[u8]) -> Vec<i16> {
    let reg = oxideav::Registries::with_all_features();
    let mut file: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(ogg_data.to_vec()));
    let format = reg
        .containers
        .probe_input(&mut *file, Some("ogg"))
        .expect("probe ogg");
    let mut dmx = reg
        .containers
        .open_demuxer(&format, file)
        .expect("open ogg demuxer");
    let params = dmx.streams()[0].params.clone();
    let mut dec = reg
        .codecs
        .make_decoder(&params)
        .expect("make vorbis decoder");
    let mut out = Vec::new();
    loop {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        };
        dec.send_packet(&pkt).expect("send");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    let bytes = &a.data[0];
                    for chunk in bytes.chunks_exact(2) {
                        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore | Error::Eof) => break,
                Err(e) => panic!("decode error: {e:?}"),
            }
        }
    }
    out
}

/// Encoder test: our encoder vs ffmpeg encoder, both decoded by ffmpeg.
#[test]
fn encoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-vorbis-enc-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with our lib
    let our_ogg = encode_with_ours(&pcm, SAMPLE_RATE, CHANNELS);
    let our_ogg_path = tmp("oxideav-vorbis-enc-ours.ogg");
    std::fs::write(&our_ogg_path, &our_ogg).expect("write our ogg");

    // Encode with ffmpeg
    let ffmpeg_ogg_path = tmp("oxideav-vorbis-enc-ffmpeg.ogg");
    assert!(
        ffmpeg(&[
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            "-i",
            raw_path.to_str().unwrap(),
            "-c:a",
            "libvorbis",
            "-b:a",
            "128k",
            ffmpeg_ogg_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode both with ffmpeg
    let our_decoded_path = tmp("oxideav-vorbis-enc-ours-decoded.raw");
    let ffmpeg_decoded_path = tmp("oxideav-vorbis-enc-ffmpeg-decoded.raw");
    assert!(
        ffmpeg(&[
            "-i",
            our_ogg_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            our_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of our ogg failed"
    );
    assert!(
        ffmpeg(&[
            "-i",
            ffmpeg_ogg_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            ffmpeg_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of ffmpeg ogg failed"
    );

    let our_decoded = read_pcm_s16le(&our_decoded_path);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);
    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== Vorbis encoder comparison ===");
    report(
        "encoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "Vorbis encoder RMS {rms:.6} too large (> 1.0)");
}

/// Decoder test: ffmpeg-encoded Vorbis, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-vorbis-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg (libvorbis)
    let ogg_path = tmp("oxideav-vorbis-dec-test.ogg");
    assert!(
        ffmpeg(&[
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            "-i",
            raw_path.to_str().unwrap(),
            "-c:a",
            "libvorbis",
            "-b:a",
            "128k",
            ogg_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-vorbis-dec-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-i",
            ogg_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            ffmpeg_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode failed"
    );

    // Decode with our decoder
    let ogg_data = std::fs::read(&ogg_path).expect("read ogg");
    let our_decoded = decode_with_ours(&ogg_data);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== Vorbis decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "Vorbis decoder RMS {rms:.6} too large (> 1.0)");
}
