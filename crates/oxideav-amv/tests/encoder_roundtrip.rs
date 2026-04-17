//! AMV IMA-ADPCM encoder roundtrip tests.
//!
//! Generates a known PCM S16 mono signal (sine wave), feeds it through the
//! `AmvAudioEncoder`, then re-decodes each emitted packet with the existing
//! `AmvAudioDecoder`. Because IMA-ADPCM is lossy the roundtrip is not
//! bit-exact; we assert on PSNR instead, which for well-behaved sine input
//! sits comfortably in the 30–45 dB range characteristic of 4-bit ADPCM.

use oxideav_codec::CodecRegistry;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

const SAMPLE_RATE: u32 = 22_050;

fn make_sine(num_samples: usize, freq_hz: f32, amp: f32) -> Vec<i16> {
    let mut out = Vec::with_capacity(num_samples);
    let two_pi_f_over_fs = 2.0 * std::f32::consts::PI * freq_hz / SAMPLE_RATE as f32;
    for n in 0..num_samples {
        let s = (amp * (two_pi_f_over_fs * n as f32).sin()) * i16::MAX as f32;
        out.push(s.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

fn pcm_to_frame(samples: &[i16]) -> AudioFrame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        samples: samples.len() as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        data: vec![bytes],
    }
}

fn psnr(reference: &[i16], decoded: &[i16]) -> f64 {
    assert_eq!(reference.len(), decoded.len(), "length mismatch");
    let n = reference.len() as f64;
    let mut sse: f64 = 0.0;
    for (a, b) in reference.iter().zip(decoded.iter()) {
        let e = (*a as f64) - (*b as f64);
        sse += e * e;
    }
    let mse = sse / n;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    let peak = i16::MAX as f64;
    10.0 * (peak * peak / mse).log10()
}

fn register_and_run_roundtrip(samples_per_chunk: usize, total_samples: usize) -> f64 {
    let mut reg = CodecRegistry::new();
    oxideav_amv::register_codecs(&mut reg);

    let id = CodecId::new(oxideav_amv::AUDIO_CODEC_ID_STR);
    assert!(reg.has_decoder(&id), "decoder should be registered");
    assert!(reg.has_encoder(&id), "encoder should be registered");

    // A 1 kHz sine at ~90% full scale is a textbook IMA-ADPCM workload.
    let input = make_sine(total_samples, 1000.0, 0.9);

    let mut enc_params = CodecParameters::audio(id.clone());
    enc_params.sample_rate = Some(SAMPLE_RATE);
    enc_params.channels = Some(1);
    enc_params.sample_format = Some(SampleFormat::S16);
    let mut encoder = reg.make_encoder(&enc_params).expect("make encoder");

    let mut dec_params = CodecParameters::audio(id);
    dec_params.sample_rate = Some(SAMPLE_RATE);
    dec_params.channels = Some(1);
    let mut decoder = reg.make_decoder(&dec_params).expect("make decoder");

    // Feed the encoder one chunk at a time.
    let mut collected: Vec<i16> = Vec::with_capacity(total_samples);
    let mut packets: Vec<Packet> = Vec::new();
    for chunk in input.chunks(samples_per_chunk) {
        let frame = Frame::Audio(pcm_to_frame(chunk));
        encoder.send_frame(&frame).expect("encoder accepts frame");
        loop {
            match encoder.receive_packet() {
                Ok(p) => packets.push(p),
                Err(oxideav_core::Error::NeedMore) => break,
                Err(e) => panic!("unexpected encoder error: {:?}", e),
            }
        }
    }
    encoder.flush().expect("encoder flush");
    while let Ok(p) = encoder.receive_packet() {
        packets.push(p);
    }

    // Sanity: at least one packet emitted, each with header + nibble bytes.
    assert!(!packets.is_empty(), "encoder emitted no packets");
    for p in &packets {
        assert!(p.data.len() >= 8, "packet missing 8-byte header");
    }

    // Decode each packet individually — each chunk reseeds the decoder via
    // its own 8-byte header, which is exactly the AMV container semantic.
    for p in packets {
        decoder.send_packet(&p).expect("send packet to decoder");
        let frame = decoder.receive_frame().expect("receive frame");
        let Frame::Audio(af) = frame else {
            panic!("expected audio frame");
        };
        let bytes = &af.data[0];
        for pair in bytes.chunks_exact(2) {
            collected.push(i16::from_le_bytes([pair[0], pair[1]]));
        }
    }

    // Encoder emits two samples per byte, rounding odd lengths up with a
    // zero-nibble tail sample. Trim to the input length before scoring.
    assert!(
        collected.len() >= input.len(),
        "decoded fewer samples ({}) than input ({})",
        collected.len(),
        input.len()
    );
    collected.truncate(input.len());

    psnr(&input, &collected)
}

#[test]
fn sine_roundtrip_single_chunk() {
    // 1 second of 1 kHz sine at 22.05 kHz, as a single big chunk.
    let psnr_db = register_and_run_roundtrip(SAMPLE_RATE as usize, SAMPLE_RATE as usize);
    eprintln!("AMV IMA-ADPCM sine PSNR (single chunk) = {:.2} dB", psnr_db);
    assert!(
        psnr_db >= 30.0,
        "PSNR {:.2} dB below expected 30 dB floor",
        psnr_db
    );
}

#[test]
fn sine_roundtrip_chunked() {
    // Same signal, but emitted via many small (≈100 ms) chunks. Each chunk
    // re-headers the decoder to the encoder's running state, so the PSNR
    // should be essentially the same as the single-chunk case.
    let chunk = (SAMPLE_RATE / 10) as usize; // ~2205 samples
    let psnr_db = register_and_run_roundtrip(chunk, SAMPLE_RATE as usize);
    eprintln!("AMV IMA-ADPCM sine PSNR (chunked)      = {:.2} dB", psnr_db);
    assert!(
        psnr_db >= 30.0,
        "PSNR {:.2} dB below expected 30 dB floor",
        psnr_db
    );
}

#[test]
fn encode_silent_input_is_near_zero() {
    // Silence in → packets whose decoded output is also silence. With an
    // all-zero start state, feeding zero samples yields zero nibbles,
    // which expand back to zero predictor — exact roundtrip.
    let mut reg = CodecRegistry::new();
    oxideav_amv::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_amv::AUDIO_CODEC_ID_STR);

    let mut params = CodecParameters::audio(id.clone());
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut enc = reg.make_encoder(&params).unwrap();
    let mut dec = reg.make_decoder(&params).unwrap();

    let input = vec![0i16; 512];
    enc.send_frame(&Frame::Audio(pcm_to_frame(&input))).unwrap();
    let pkt = enc.receive_packet().unwrap();
    assert_eq!(pkt.data.len(), 8 + input.len() / 2);
    dec.send_packet(&pkt).unwrap();
    let Frame::Audio(af) = dec.receive_frame().unwrap() else {
        panic!("expected audio");
    };
    let bytes = &af.data[0];
    for pair in bytes.chunks_exact(2) {
        assert_eq!(i16::from_le_bytes([pair[0], pair[1]]), 0);
    }
}
