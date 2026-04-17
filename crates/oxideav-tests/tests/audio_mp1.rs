//! MP1 (MPEG-1 Audio Layer I) decode-only comparison tests against ffmpeg.
//!
//! Our crate has no MP1 encoder. ffmpeg's built-in MP1 encoder (mp1 /
//! mp1fixed) may or may not be available. If it is, we encode with ffmpeg,
//! then compare our decode vs ffmpeg's decode.

use oxideav_core::{Error, Frame};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;

/// Decode raw MPEG audio frames with our MP1 decoder.
/// MP1 frames are self-delimiting via their headers, so we use the
/// MP3 container demuxer (which handles all MPEG audio layers).
fn decode_with_ours(mp1_data: &[u8]) -> Vec<i16> {
    let reg = oxideav::Registries::with_all_features();
    let mut file: Box<dyn oxideav::container::ReadSeek> =
        Box::new(std::io::Cursor::new(mp1_data.to_vec()));
    let format = reg
        .containers
        .probe_input(&mut *file, Some("mp1"))
        .expect("probe mp1");
    let mut dmx = reg
        .containers
        .open_demuxer(&format, file)
        .expect("open mp1 demuxer");
    let params = dmx.streams()[0].params.clone();
    let mut dec = reg.codecs.make_decoder(&params).expect("make mp1 decoder");
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

/// Decoder test: ffmpeg-encoded MP1, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-mp1-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Try encoding with ffmpeg's MP1 encoder.
    // ffmpeg may not have mp1 encoder built in (it's deprecated), so we
    // try mp1 first, then mp1fixed, and skip if neither works.
    let mp1_path = tmp("oxideav-mp1-dec-test.mp1");
    let encoded = ffmpeg(&[
        "-f",
        "s16le",
        "-ar",
        &SAMPLE_RATE.to_string(),
        "-ac",
        &CHANNELS.to_string(),
        "-i",
        raw_path.to_str().unwrap(),
        "-c:a",
        "mp1",
        "-b:a",
        "192k",
        "-f",
        "mp3",
        mp1_path.to_str().unwrap(),
    ]) || ffmpeg(&[
        "-f",
        "s16le",
        "-ar",
        &SAMPLE_RATE.to_string(),
        "-ac",
        &CHANNELS.to_string(),
        "-i",
        raw_path.to_str().unwrap(),
        "-c:a",
        "mp1fixed",
        "-b:a",
        "192k",
        "-f",
        "mp3",
        mp1_path.to_str().unwrap(),
    ]);

    if !encoded {
        eprintln!("skip: ffmpeg lacks MP1 encoder (mp1/mp1fixed)");
        return;
    }

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-mp1-dec-ffmpeg.raw");
    assert!(
        ffmpeg(&[
            "-i",
            mp1_path.to_str().unwrap(),
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
    let mp1_data = std::fs::read(&mp1_path).expect("read mp1");
    let our_decoded = decode_with_ours(&mp1_data);
    let ffmpeg_decoded = read_pcm_s16le(&ffmpeg_decoded_path);

    let rms = audio_rms_diff(&our_decoded, &ffmpeg_decoded);
    let psnr = audio_psnr(&our_decoded, &ffmpeg_decoded);

    eprintln!("=== MP1 decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "MP1 decoder RMS {rms:.6} too large (> 1.0)");
}
