//! Speex decode-only comparison tests against ffmpeg.
//!
//! Our crate has no Speex encoder. ffmpeg's libspeex encoder produces
//! Ogg/Speex files, which our Ogg demuxer + Speex decoder handles.
//!
//! Speex supports narrowband (8 kHz) and wideband (16 kHz). We test
//! narrowband since it's the most commonly used.

use oxideav_core::{Error, Frame};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 8000;
const CHANNELS: u16 = 1;
const DURATION: f32 = 2.0;

/// Decode an Ogg/Speex file with our decoder via demuxer.
fn decode_with_ours(ogg_data: &[u8]) -> Vec<i16> {
    let reg = oxideav::Registries::with_all_features();
    let mut file: Box<dyn oxideav::container::ReadSeek> =
        Box::new(std::io::Cursor::new(ogg_data.to_vec()));
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
        .expect("make speex decoder");
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

/// Decoder test: ffmpeg-encoded Speex, our decode vs ffmpeg decode.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-speex-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg (libspeex)
    let ogg_path = tmp("oxideav-speex-dec-test.ogg");
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
        "libspeex",
        "-b:a",
        "15k",
        ogg_path.to_str().unwrap(),
    ]) {
        eprintln!("skip: ffmpeg libspeex encode failed (libspeex may not be available)");
        return;
    }

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-speex-dec-ffmpeg.raw");
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

    eprintln!("=== Speex decoder comparison ===");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    assert!(rms < 1.0, "Speex decoder RMS {rms:.6} too large (> 1.0)");
}
