//! End-to-end encode → decode roundtrip tests for the pure-Rust MP1
//! encoder.
//!
//! Generates a tone, encodes with our MP1 encoder, decodes with our
//! MP1 decoder, and asserts PSNR is above a reasonable floor. Layer I
//! with a greedy allocator won't compete with a perceptual encoder but
//! a settled sine should reconstruct well above noise floor.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, MediaType, Packet, SampleFormat, TimeBase,
};
use oxideav_mp1::decoder::make_decoder;
use oxideav_mp1::encoder::make_encoder;
use oxideav_mp1::header::FrameHeader;
use oxideav_mp1::CODEC_ID_STR;

/// Build an interleaved PCM s16le tone.
fn make_tone(duration_s: f32, sample_rate: u32, channels: u16, freq: f32) -> Vec<u8> {
    let n = (duration_s * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(n * 2 * channels as usize);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (0.5 * (2.0 * std::f32::consts::PI * freq * t).sin() * 32767.0) as i16;
        for _ch in 0..channels {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

fn encode_all(pcm: &[u8], sample_rate: u32, channels: u16, bitrate_kbps: u32) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = Some((bitrate_kbps as u64) * 1000);
    let mut enc = make_encoder(&params).expect("build encoder");

    let total_samples = (pcm.len() / (2 * channels as usize)) as u32;
    let frame = AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate,
        samples: total_samples,
        pts: Some(0),
        time_base: TimeBase::new(1, sample_rate as i64),
        data: vec![pcm.to_vec()],
    };
    enc.send_frame(&Frame::Audio(frame)).expect("send_frame");
    let mut bytes = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        bytes.extend_from_slice(&p.data);
    }
    enc.flush().expect("flush");
    while let Ok(p) = enc.receive_packet() {
        bytes.extend_from_slice(&p.data);
    }
    bytes
}

fn split_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        if data[i] != 0xFF || (data[i + 1] & 0xF0) != 0xF0 {
            i += 1;
            continue;
        }
        let Ok(h) = FrameHeader::parse(&data[i..]) else {
            i += 1;
            continue;
        };
        let len = h.frame_size();
        if i + len > data.len() {
            break;
        }
        frames.push(&data[i..i + len]);
        i += len;
    }
    frames
}

fn decode_all(data: &[u8]) -> (Vec<i16>, u32, u16) {
    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("build decoder");
    let mut samples: Vec<i16> = Vec::new();
    let mut sr = 0u32;
    let mut ch = 0u16;
    for fr in split_frames(data) {
        let pkt = Packet {
            stream_index: 0,
            time_base: TimeBase::new(1, 48_000),
            pts: None,
            dts: None,
            duration: None,
            flags: Default::default(),
            data: fr.to_vec(),
        };
        dec.send_packet(&pkt).expect("send_packet");
        if let Ok(Frame::Audio(a)) = dec.receive_frame() {
            sr = a.sample_rate;
            ch = a.channels;
            let bytes = &a.data[0];
            for chunk in bytes.chunks_exact(2) {
                samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    (samples, sr, ch)
}

/// Compute PSNR in dB vs a reference signal. The decoded signal is
/// delayed by the polyphase group delay (~481 samples for Layer I); we
/// align by skipping the first `skip_recon` samples of `recon`.
fn psnr_db(ref_signal: &[i16], recon: &[i16], skip_recon: usize) -> f64 {
    if skip_recon >= recon.len() {
        return 0.0;
    }
    let r = recon;
    let n = ref_signal.len().min(r.len() - skip_recon);
    if n == 0 {
        return 0.0;
    }
    let mut sq_err = 0.0f64;
    for i in 0..n {
        let d = ref_signal[i] as f64 - r[i + skip_recon] as f64;
        sq_err += d * d;
    }
    let mse = sq_err / n as f64;
    if mse < 1.0 {
        return 120.0;
    }
    let peak = 32767.0f64;
    10.0 * ((peak * peak) / mse).log10()
}

#[test]
fn roundtrip_mono_44k_1khz_tone() {
    let sr = 44_100u32;
    let freq = 1000.0f32;
    // 100 ms of tone per the task prompt; extend a touch so we clear the
    // analysis/synthesis warm-up before measuring.
    let pcm = make_tone(0.5, sr, 1, freq);
    let encoded = encode_all(&pcm, sr, 1, 192);
    assert!(!encoded.is_empty(), "encoder produced no data");
    let (decoded, dec_sr, dec_ch) = decode_all(&encoded);
    assert_eq!(dec_sr, sr);
    assert_eq!(dec_ch, 1);
    assert!(
        decoded.len() > 10_000,
        "decoded too few samples: {}",
        decoded.len()
    );

    let ref_samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    // Polyphase analysis + synthesis has a ~481-sample group delay on
    // top of a Layer I frame size of 384 samples; scan an offset window
    // and keep the best PSNR.
    let mut best = -1000.0f64;
    for offset in 0..1500 {
        let p = psnr_db(&ref_samples, &decoded, offset);
        if p > best {
            best = p;
        }
    }
    println!("roundtrip PSNR (mono, 44.1k, 192kbps, 1kHz): {best:.2} dB");
    assert!(best >= 28.0, "PSNR too low: {best:.2} dB");
}

#[test]
fn roundtrip_mono_44k_silence() {
    let sr = 44_100u32;
    // 100 ms of silence.
    let n = (sr as f32 * 0.1) as usize;
    let pcm = vec![0u8; n * 2];
    let encoded = encode_all(&pcm, sr, 1, 128);
    assert!(!encoded.is_empty(), "encoder produced no data");
    let (decoded, dec_sr, dec_ch) = decode_all(&encoded);
    assert_eq!(dec_sr, sr);
    assert_eq!(dec_ch, 1);

    // Pure-tone encoder filtering silence yields silence to within
    // float roundoff. Check the post-warmup tail settles near zero.
    let warmup = 1000.min(decoded.len() / 2);
    let tail = &decoded[warmup..];
    let max_abs = tail.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
    println!("silence max |s| post-warmup: {max_abs}");
    assert!(max_abs < 32, "silence should stay near zero, got {max_abs}");
}
