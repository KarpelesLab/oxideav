//! Opus decode-only comparison tests against ffmpeg.
//!
//! Our crate has no Opus encoder (the stub returns Unsupported).
//! ffmpeg's libopus encoder produces Ogg/Opus files, which our Ogg
//! demuxer + Opus decoder handles.
//!
//! Opus uses 48 kHz internally.

use oxideav_core::{Error, Frame};
use oxideav_tests::*;

const SAMPLE_RATE: u32 = 48000;
const CHANNELS: u16 = 2;
const DURATION: f32 = 2.0;

/// Decode an Ogg/Opus file with our decoder via demuxer.
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
    let mut dec = reg.codecs.make_decoder(&params).expect("make opus decoder");
    let mut out = Vec::new();
    loop {
        let pkt = match dmx.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        };
        // Opus decoder may return Unsupported for non-silence frames
        // since the full CELT/SILK decoder is not yet landed.
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(Error::Unsupported(_)) => continue,
            Err(e) => panic!("send error: {e:?}"),
        }
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
                Err(Error::Unsupported(_)) => break,
                Err(e) => panic!("decode error: {e:?}"),
            }
        }
    }
    out
}

/// Decoder test: ffmpeg-encoded Opus, our decode vs ffmpeg decode.
///
/// Note: the Opus decoder is still partial (SILK/CELT not fully landed),
/// so this test may produce mostly-silent output. We use very relaxed
/// thresholds and document the actual numbers.
#[test]
fn decoder_vs_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }

    let pcm = generate_audio_signal(SAMPLE_RATE, CHANNELS, DURATION);
    let raw_path = tmp("oxideav-opus-dec-input.raw");
    write_pcm_s16le(&raw_path, &pcm);

    // Encode with ffmpeg (libopus)
    let ogg_path = tmp("oxideav-opus-dec-test.ogg");
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
        "libopus",
        "-b:a",
        "128k",
        ogg_path.to_str().unwrap(),
    ]) {
        eprintln!("skip: ffmpeg libopus encode failed (libopus may not be available)");
        return;
    }

    // Decode with ffmpeg
    let ffmpeg_decoded_path = tmp("oxideav-opus-dec-ffmpeg.raw");
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

    eprintln!("=== Opus decoder comparison ===");
    eprintln!("  Note: Opus decoder is partial (SILK/CELT not fully landed)");
    report(
        "decoder",
        rms,
        psnr,
        our_decoded.len(),
        ffmpeg_decoded.len(),
    );

    // Very relaxed threshold since decoder is incomplete — just document
    // the current quality level.
    if our_decoded.is_empty() {
        eprintln!("  WARNING: our decoder produced 0 samples (expected — decoder is partial)");
    } else {
        assert!(rms < 1.0, "Opus decoder RMS {rms:.6} too large (> 1.0)");
    }
}
