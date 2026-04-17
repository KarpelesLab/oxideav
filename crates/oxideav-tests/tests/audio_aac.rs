//! AAC-LC roundtrip comparison tests against ffmpeg.
//!
//! AAC is lossy. Our encoder produces raw ADTS frames that ffmpeg can
//! decode directly. For the decoder test, ffmpeg encodes to ADTS (.aac)
//! and we decode frame-by-frame using the ADTS header parser.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;
const BITRATE: u64 = 128_000;

/// Encode PCM with our AAC encoder, return raw ADTS bytes.
fn encode_with_ours(pcm: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new("aac"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.bit_rate = Some(BITRATE);
    let mut enc = oxideav_aac::encoder::make_encoder(&params).expect("make aac encoder");

    let stride = channels as usize;
    let total_samples = pcm.len() / stride;
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let frame = Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate,
        samples: total_samples as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, sample_rate as i64),
        data: vec![bytes],
    });
    enc.send_frame(&frame).expect("send_frame");
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

/// Decode raw ADTS AAC frames with our decoder.
/// Scans for ADTS sync words and feeds each frame as a packet.
fn decode_adts_with_ours(aac_data: &[u8], sample_rate: u32, channels: u16) -> Vec<i16> {
    // Parse ADTS frames
    let frames = iter_adts_frames(aac_data);
    if frames.is_empty() {
        return Vec::new();
    }

    let mut params = CodecParameters::audio(CodecId::new("aac"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    let mut dec = oxideav_aac::decoder::make_decoder(&params).expect("make aac decoder");
    let tb = TimeBase::new(1, sample_rate as i64);
    let mut out = Vec::new();

    for (i, &(off, len)) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, aac_data[off..off + len].to_vec()).with_pts(i as i64 * 1024);
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

/// Find ADTS frame boundaries in a raw AAC bitstream.
fn iter_adts_frames(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 7 < bytes.len() {
        if bytes[i] != 0xFF || (bytes[i + 1] & 0xF0) != 0xF0 {
            i += 1;
            continue;
        }
        match oxideav_aac::adts::parse_adts_header(&bytes[i..]) {
            Ok(h) => {
                if h.frame_length == 0 || i + h.frame_length > bytes.len() {
                    break;
                }
                out.push((i, h.frame_length));
                i += h.frame_length;
            }
            Err(_) => i += 1,
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
    let raw_path = tmp("oxideav-aac-enc-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with our lib (raw ADTS)
    let our_aac = encode_with_ours(&pcm, SAMPLE_RATE, CHANNELS);
    let our_aac_path = tmp("oxideav-aac-enc-ours.aac");
    std::fs::write(&our_aac_path, &our_aac).expect("write our aac");

    // Encode with ffmpeg (native aac encoder to ADTS)
    let ffmpeg_aac_path = tmp("oxideav-aac-enc-ffmpeg.aac");
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
            "aac",
            "-b:a",
            "128k",
            ffmpeg_aac_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode both with ffmpeg
    let our_decoded_path = tmp("oxideav-aac-enc-ours-decoded.raw");
    let ffmpeg_decoded_path = tmp("oxideav-aac-enc-ffmpeg-decoded.raw");
    assert!(
        ffmpeg(&[
            "-i",
            our_aac_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            our_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of our aac failed"
    );
    assert!(
        ffmpeg(&[
            "-i",
            ffmpeg_aac_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            ffmpeg_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of ffmpeg aac failed"
    );

    let our_decoded = read_pcm_s16le(&our_decoded_path);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);
    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== AAC encoder comparison ===");
    report(
        "encoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "AAC encoder RMS {rms:.6} too large (> 1.0)");
}

/// Decoder test: ffmpeg-encoded AAC (ADTS), our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-aac-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg to ADTS (raw .aac)
    let aac_path = tmp("oxideav-aac-dec-test.aac");
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
            "aac",
            "-b:a",
            "128k",
            aac_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-aac-dec-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-i",
            aac_path.to_str().unwrap(),
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

    // Decode with our decoder (ADTS path)
    let aac_data = std::fs::read(&aac_path).expect("read aac");
    let our_decoded = decode_adts_with_ours(&aac_data, SAMPLE_RATE, CHANNELS);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== AAC decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "AAC decoder RMS {rms:.6} too large (> 1.0)");
}
