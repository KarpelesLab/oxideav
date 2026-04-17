//! MP2 (MPEG-1 Audio Layer II) decode-only comparison tests against ffmpeg.
//!
//! Our crate has no MP2 encoder. ffmpeg's mp2 encoder is widely available.
//! The MP3 container demuxer rejects Layer II frames, so we manually split
//! frames using the MP2 header parser and feed packets directly.

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;

/// Split an elementary MP2 bitstream into individual frames using our
/// header parser.
fn split_mp2_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        if data[i] != 0xFF || (data[i + 1] & 0xF0) != 0xF0 {
            i += 1;
            continue;
        }
        let Ok(h) = oxideav_mp2::header::parse_header(&data[i..]) else {
            i += 1;
            continue;
        };
        let len = h.frame_length();
        if len == 0 || i + len > data.len() {
            break;
        }
        frames.push(&data[i..i + len]);
        i += len;
    }
    frames
}

/// Decode MP2 frames with our decoder by feeding packets directly.
fn decode_with_ours(mp2_data: &[u8], sample_rate: u32, channels: u16) -> Vec<i16> {
    let mut params = CodecParameters::audio(CodecId::new("mp2"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = oxideav_mp2::decoder::make_decoder(&params).expect("make mp2 decoder");

    let tb = TimeBase::new(1, sample_rate as i64);
    let frames = split_mp2_frames(mp2_data);
    let mut out = Vec::new();

    for frame_data in &frames {
        let pkt = Packet::new(0, tb, frame_data.to_vec());
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

/// Decoder test: ffmpeg-encoded MP2, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-mp2-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg
    let mp2_path = tmp("oxideav-mp2-dec-test.mp2");
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
        "mp2",
        "-b:a",
        "192k",
        "-f",
        "mp2",
        mp2_path.to_str().unwrap(),
    ]) {
        eprintln!("skip: ffmpeg mp2 encode failed");
        return;
    }

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-mp2-dec-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-i",
            mp2_path.to_str().unwrap(),
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
    let mp2_data = std::fs::read(&mp2_path).expect("read mp2");
    let our_decoded = decode_with_ours(&mp2_data, SAMPLE_RATE, CHANNELS);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== MP2 decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "MP2 decoder RMS {rms:.6} too large (> 1.0)");
}
