//! Encode → decode round-trip for a pure tone.
//!
//! G.722 at 64 kbit/s is lossy but preserves enough of a single-frequency
//! sinusoid that we can assert a non-trivial signal-to-noise ratio. 20 dB
//! PSNR is the floor for a well-behaved quantiser + predictor on a tone —
//! typical numbers land 5–15 dB higher, so this is a generous margin.

// These traits must be in scope so we can call `send_frame` /
// `send_packet` etc. on the `Box<dyn _>` returned by the factories.
#[allow(unused_imports)]
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase};
use oxideav_g722::{decoder, encoder, CODEC_ID_STR};

fn params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(16_000);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn sine(len_samples: usize, freq: f32) -> Vec<i16> {
    let two_pi = 2.0f32 * std::f32::consts::PI;
    (0..len_samples)
        .map(|n| {
            let t = n as f32 / 16_000.0;
            ((two_pi * freq * t).sin() * 10_000.0) as i16
        })
        .collect()
}

fn audio_frame(samples: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: 16_000,
        samples: samples.len() as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, 16_000),
        data: vec![bytes],
    })
}

fn psnr(original: &[i16], decoded: &[i16]) -> f64 {
    // Best-integer-delay alignment — the QMF introduces ~23 samples of
    // group delay that varies slightly with the ADPCM predictor.
    let mut best = f64::NEG_INFINITY;
    let n_total = original.len().min(decoded.len());
    let skip_head = 96.min(n_total / 4); // settle transient
    for delay in 0..64 {
        let mut err = 0.0f64;
        let mut sig = 0.0f64;
        if skip_head + delay >= n_total {
            continue;
        }
        let n = n_total - skip_head - delay;
        for i in 0..n {
            let x = original[skip_head + i] as f64;
            let y = decoded[skip_head + delay + i] as f64;
            let e = x - y;
            err += e * e;
            sig += x * x;
        }
        let psnr_v = if err == 0.0 {
            200.0
        } else if sig == 0.0 {
            0.0
        } else {
            10.0 * (sig / err).log10()
        };
        if psnr_v > best {
            best = psnr_v;
        }
    }
    best
}

#[test]
fn roundtrip_1khz_sine_psnr_above_20db() {
    // 200 ms at 16 kHz = 3200 samples.
    let input = sine(3200, 1000.0);

    let mut enc = encoder::make_encoder(&params()).expect("encoder");
    enc.send_frame(&audio_frame(&input)).expect("send_frame");
    enc.flush().expect("flush");

    let mut dec = decoder::make_decoder(&params()).expect("decoder");
    let mut decoded: Vec<i16> = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for chunk in af.data[0].chunks_exact(2) {
                        decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(_) => break,
                Err(oxideav_core::Error::NeedMore) => break,
                Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("decode error: {e}"),
            }
        }
    }
    dec.flush().ok();

    assert!(
        !decoded.is_empty(),
        "decoder produced no samples for a 200 ms input"
    );
    assert!(
        decoded.len() >= input.len() / 2,
        "decoded length too short: {} vs input {}",
        decoded.len(),
        input.len()
    );

    let snr = psnr(&input, &decoded);
    eprintln!("G.722 1 kHz sine PSNR = {snr:.2} dB");
    assert!(
        snr > 20.0,
        "PSNR {snr:.2} dB below the 20 dB floor for a 1 kHz sine"
    );
}
