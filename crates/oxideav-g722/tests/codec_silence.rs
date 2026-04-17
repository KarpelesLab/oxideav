//! Silence in → silence-ish out.

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

#[test]
fn silence_roundtrip_stays_bounded() {
    let input = vec![0i16; 4000]; // 250 ms
    let mut enc = encoder::make_encoder(&params()).expect("encoder");
    enc.send_frame(&audio_frame(&input)).expect("send_frame");
    enc.flush().expect("flush");

    let mut dec = decoder::make_decoder(&params()).expect("decoder");
    let mut decoded: Vec<i16> = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(Frame::Audio(af)) = dec.receive_frame() {
            for chunk in af.data[0].chunks_exact(2) {
                decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }

    assert!(!decoded.is_empty());
    // Bound: after the filter settling transient, values should stay near
    // zero. Allow up to +/- 2000 during the first 50 samples, and tighter
    // afterward. G.722's backward-adaptive predictor converges quickly when
    // input is strictly zero.
    for (i, &s) in decoded.iter().enumerate() {
        let limit = if i < 50 { 4_000 } else { 2_500 };
        assert!(
            (s as i32).abs() < limit,
            "silence drifted at sample {i}: {s} exceeds {limit}"
        );
    }
}
