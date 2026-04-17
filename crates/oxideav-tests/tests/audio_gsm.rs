//! GSM Full Rate (06.10) roundtrip comparison tests against ffmpeg.
//!
//! GSM is 8 kHz mono only. Our encoder produces raw 33-byte ETSI frames.
//! ffmpeg can read/write raw .gsm files (concatenated 33-byte frames).

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 8000;
const CHANNELS: u16 = 1;
const DURATION: f32 = 2.0;

/// Encode PCM with our GSM encoder, return concatenated raw GSM frames.
fn encode_with_ours(pcm: &[i16]) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new("gsm"));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(CHANNELS);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = oxideav_gsm::encoder::make_encoder(&params).expect("make gsm encoder");

    let tb = TimeBase::new(1, SAMPLE_RATE as i64);
    // Feed 160-sample chunks (one GSM frame = 160 samples)
    let chunk_samples = 160;
    let mut pts: i64 = 0;
    for chunk in pcm.chunks(chunk_samples) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame {
            format: SampleFormat::S16,
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            samples: chunk.len() as u32,
            pts: Some(pts),
            time_base: tb,
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send");
        pts += chunk.len() as i64;
    }
    enc.flush().expect("flush");

    let mut out = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => out.extend_from_slice(&pkt.data),
            Err(Error::NeedMore | Error::Eof) => break,
            Err(e) => panic!("encode error: {e:?}"),
        }
    }
    out
}

/// Decode raw GSM frames with our decoder.
fn decode_with_ours(gsm_data: &[u8]) -> Vec<i16> {
    let mut params = CodecParameters::audio(CodecId::new("gsm"));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(CHANNELS);
    let mut dec = oxideav_gsm::decoder::make_decoder(&params).expect("make gsm decoder");

    let tb = TimeBase::new(1, SAMPLE_RATE as i64);
    let frame_size = 33; // standard GSM frame is 33 bytes
    let mut out = Vec::new();

    for frame_bytes in gsm_data.chunks(frame_size) {
        if frame_bytes.len() < frame_size {
            break;
        }
        let pkt = Packet::new(0, tb, frame_bytes.to_vec());
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
    let raw_path = tmp("oxideav-gsm-enc-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with our lib
    let our_gsm = encode_with_ours(&pcm);
    let our_gsm_path = tmp("oxideav-gsm-enc-ours.gsm");
    std::fs::write(&our_gsm_path, &our_gsm).expect("write our gsm");

    // Encode with ffmpeg (libgsm)
    let ffmpeg_gsm_path = tmp("oxideav-gsm-enc-ffmpeg.gsm");
    if !ffmpeg(&[
        "-f",
        "s16le",
        "-ar",
        &SAMPLE_RATE.to_string(),
        "-ac",
        &CHANNELS.to_string(),
        "-i",
        raw_path.to_str().unwrap(),
        "-c:a",
        "libgsm",
        "-f",
        "gsm",
        ffmpeg_gsm_path.to_str().unwrap(),
    ]) {
        eprintln!("skip: ffmpeg libgsm encode failed (libgsm may not be available)");
        return;
    }

    // Decode both with ffmpeg
    let our_decoded_path = tmp("oxideav-gsm-enc-ours-decoded.raw");
    let ffmpeg_decoded_path = tmp("oxideav-gsm-enc-ffmpeg-decoded.raw");
    assert!(
        ffmpeg(&[
            "-f",
            "gsm",
            "-i",
            our_gsm_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            our_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of our gsm failed"
    );
    assert!(
        ffmpeg(&[
            "-f",
            "gsm",
            "-i",
            ffmpeg_gsm_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            ffmpeg_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of ffmpeg gsm failed"
    );

    let our_decoded = read_pcm_s16le(&our_decoded_path);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);
    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== GSM encoder comparison ===");
    report(
        "encoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "GSM encoder RMS {rms:.6} too large (> 1.0)");
}

/// Decoder test: ffmpeg-encoded GSM, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-gsm-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg (libgsm)
    let gsm_path = tmp("oxideav-gsm-dec-test.gsm");
    if !ffmpeg(&[
        "-f",
        "s16le",
        "-ar",
        &SAMPLE_RATE.to_string(),
        "-ac",
        &CHANNELS.to_string(),
        "-i",
        raw_path.to_str().unwrap(),
        "-c:a",
        "libgsm",
        "-f",
        "gsm",
        gsm_path.to_str().unwrap(),
    ]) {
        eprintln!("skip: ffmpeg libgsm encode failed (libgsm may not be available)");
        return;
    }

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-gsm-dec-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-f",
            "gsm",
            "-i",
            gsm_path.to_str().unwrap(),
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
    let gsm_data = std::fs::read(&gsm_path).expect("read gsm");
    let our_decoded = decode_with_ours(&gsm_data);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== GSM decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "GSM decoder RMS {rms:.6} too large (> 1.0)");
}
