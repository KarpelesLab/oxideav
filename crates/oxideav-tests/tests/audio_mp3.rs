//! MP3 roundtrip comparison tests against ffmpeg.
//!
//! MP3 is lossy. This mirrors the per-crate pipeline_comparison.rs
//! but uses the shared test helpers for consistency with the other
//! codec tests.

use oxideav_container::ReadSeek;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat, TimeBase};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;
const BITRATE_KBPS: u32 = 192;

/// Encode PCM with our MP3 encoder, return raw MP3 bytes.
fn encode_with_ours(pcm: &[i16], sample_rate: u32, channels: u16, bitrate: u32) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.bit_rate = Some(bitrate as u64 * 1000);
    let mut enc = oxideav_mp3::encoder::make_encoder(&params).expect("make mp3 encoder");

    let samples_per_frame = 1152;
    let stride = channels as usize;
    let mut mp3_bytes = Vec::new();

    for chunk in pcm.chunks(samples_per_frame * stride) {
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
        loop {
            match enc.receive_packet() {
                Ok(pkt) => mp3_bytes.extend_from_slice(&pkt.data),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("encode error: {e:?}"),
            }
        }
    }
    enc.flush().expect("flush");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => mp3_bytes.extend_from_slice(&pkt.data),
            Err(Error::NeedMore | Error::Eof) => break,
            Err(e) => panic!("flush error: {e:?}"),
        }
    }
    mp3_bytes
}

/// Decode MP3 bytes with our decoder via the MP3 container demuxer.
fn decode_with_ours(mp3: &[u8]) -> Vec<i16> {
    let reg = oxideav::Registries::with_all_features();
    let mut file: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(mp3.to_vec()));
    let format = reg
        .containers
        .probe_input(&mut *file, Some("mp3"))
        .expect("probe mp3");
    let mut dmx = reg
        .containers
        .open_demuxer(&format, file)
        .expect("open mp3 demuxer");
    let params = dmx.streams()[0].params.clone();
    let mut dec = reg.codecs.make_decoder(&params).expect("make mp3 decoder");
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
    let raw_path = tmp("oxideav-mp3-enc2-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with our lib
    let our_mp3 = encode_with_ours(&pcm, SAMPLE_RATE, CHANNELS, BITRATE_KBPS);
    let our_mp3_path = tmp("oxideav-mp3-enc2-ours.mp3");
    std::fs::write(&our_mp3_path, &our_mp3).expect("write our mp3");

    // Encode with ffmpeg
    let ffmpeg_mp3_path = tmp("oxideav-mp3-enc2-ffmpeg.mp3");
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
            "libmp3lame",
            "-b:a",
            &format!("{BITRATE_KBPS}k"),
            ffmpeg_mp3_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode both with ffmpeg
    let our_decoded_path = tmp("oxideav-mp3-enc2-ours-decoded.raw");
    let ffmpeg_decoded_path = tmp("oxideav-mp3-enc2-ffmpeg-decoded.raw");
    assert!(
        ffmpeg(&[
            "-i",
            our_mp3_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            our_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of our mp3 failed"
    );
    assert!(
        ffmpeg(&[
            "-i",
            ffmpeg_mp3_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            &CHANNELS.to_string(),
            ffmpeg_decoded_path.to_str().unwrap(),
        ]),
        "ffmpeg decode of ffmpeg mp3 failed"
    );

    let our_decoded = read_pcm_s16le(&our_decoded_path);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);
    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== MP3 encoder comparison ===");
    report(
        "encoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    // MP3 encoder is crude (no psychoacoustic model, long-blocks-only),
    // so the threshold is relaxed. Matches pipeline_comparison.rs (2.0).
    assert!(rms < 2.0, "MP3 encoder RMS {rms:.6} too large (> 2.0)");
}

/// Decoder test: ffmpeg-encoded MP3, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-mp3-dec2-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg
    let mp3_path = tmp("oxideav-mp3-dec2-test.mp3");
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
            "libmp3lame",
            "-b:a",
            &format!("{BITRATE_KBPS}k"),
            mp3_path.to_str().unwrap(),
        ]),
        "ffmpeg encode failed"
    );

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-mp3-dec2-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-i",
            mp3_path.to_str().unwrap(),
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
    let mp3_data = std::fs::read(&mp3_path).expect("read mp3");
    let our_decoded = decode_with_ours(&mp3_data);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== MP3 decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "MP3 decoder RMS {rms:.6} too large (> 1.0)");
}
