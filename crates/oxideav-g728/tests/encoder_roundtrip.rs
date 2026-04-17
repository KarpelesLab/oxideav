//! G.728 encoder → decoder roundtrip.
//!
//! The encoder shares its shape / gain codebooks and its backward-adaptive
//! LPC + log-gain predictors with the decoder. The shape codebook in
//! `crate::tables::SHAPE_CB` is a deterministic unit-RMS placeholder — it
//! is *not* the ITU Annex A `CODEBK` table — so we cannot reasonably hope
//! to approach reference-grade SNR numbers. These tests assert pipeline
//! properties rather than exact reconstructions:
//!
//!   1. Sinewave input produces finite output whose energy is at least
//!      5 % of the input energy (i.e. the encoder emits non-trivial
//!      excitation and the decoder reconstructs audible content).
//!   2. Silence in → near-silence out after the initial transient, with
//!      bounded peak amplitude.
//!   3. PTS values on the produced packets rise monotonically.
//!   4. `register` exposes both decode and encode factories.

use oxideav_codec::CodecRegistry;
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_g728::encoder::{PACKET_BYTES, PACKET_SAMPLES};
use oxideav_g728::{CODEC_ID_STR, SAMPLE_RATE};

// 400 ms of audio at 8 kHz → 3200 samples → 160 G.728 packets.
const PACKETS: usize = 160;
const TOTAL_SAMPLES: usize = PACKETS * PACKET_SAMPLES;

fn make_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn make_encoder() -> Box<dyn Encoder> {
    let params = make_params();
    oxideav_g728::encoder::make_encoder(&params).expect("encoder ctor")
}

fn make_decoder() -> Box<dyn Decoder> {
    let params = make_params();
    oxideav_g728::decoder::make_decoder(&params).expect("decoder ctor")
}

/// Build a 400 Hz sine wave at 8 kHz. Amplitude chosen to sit comfortably
/// within the placeholder-codebook scale — the decoder's gain predictor
/// operates in a loose log-domain envelope that maps our ±1000 LSB sine
/// to an output of broadly similar amplitude.
fn build_sine(n: usize) -> Vec<i16> {
    let sr = SAMPLE_RATE as f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let s = 1000.0 * (2.0 * core::f32::consts::PI * 400.0 * t).sin();
        out.push(s.round().clamp(-32768.0, 32767.0) as i16);
    }
    out
}

fn pack_audio_frame(samples: &[i16]) -> AudioFrame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        samples: samples.len() as u32,
        pts: None,
        time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        data: vec![bytes],
    }
}

fn encode_all(enc: &mut Box<dyn Encoder>, input: &[i16]) -> Vec<Packet> {
    let chunk = PACKET_SAMPLES; // feed one packet's worth at a time
    let mut packets = Vec::new();
    for block in input.chunks(chunk) {
        let af = pack_audio_frame(block);
        enc.send_frame(&Frame::Audio(af)).expect("send_frame");
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        }
    }
    enc.flush().expect("encoder flush");
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_packet post-flush: {e}"),
        }
    }
    packets
}

fn decode_all(dec: &mut Box<dyn Decoder>, packets: &[Packet]) -> Vec<i16> {
    let mut pcm = Vec::new();
    for p in packets {
        dec.send_packet(p).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for chunk in af.data[0].chunks_exact(2) {
                        pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    let _ = dec.flush();
    pcm
}

fn energy(x: &[i16]) -> f64 {
    let mut e = 0.0f64;
    for &v in x {
        let v = v as f64;
        e += v * v;
    }
    e
}

#[test]
fn sine_roundtrip_has_nontrivial_energy() {
    let input = build_sine(TOTAL_SAMPLES);

    let mut enc = make_encoder();
    let packets = encode_all(&mut enc, &input);
    assert_eq!(
        packets.len(),
        PACKETS,
        "expected {PACKETS} packets, got {}",
        packets.len()
    );
    for p in &packets {
        assert_eq!(
            p.data.len(),
            PACKET_BYTES,
            "G.728 packets must be 5 bytes (40 bits = 4 × 10-bit indices)"
        );
    }

    let mut dec = make_decoder();
    let out = decode_all(&mut dec, &packets);
    assert!(!out.is_empty(), "decoder produced no audio");

    // Make sure the decoded output isn't constant (catches silent-decoder bugs).
    let distinct = {
        let mut set = std::collections::HashSet::new();
        for &s in out.iter() {
            set.insert(s);
        }
        set.len()
    };
    assert!(
        distinct > 4,
        "decoded output is near-constant ({distinct} distinct values)"
    );

    // Energy check — placeholder codebook caps the SNR, but the
    // encoder↔decoder pair must still route at least 5 % of the input
    // energy through to the output. Much lower would mean the
    // analysis-by-synthesis search is not actually tracking the target.
    let e_in = energy(&input);
    let e_out = energy(&out);
    assert!(e_in > 0.0, "input energy is zero — test setup bug");
    let ratio = e_out / e_in;
    assert!(
        ratio >= 0.05,
        "decoded/input energy ratio {ratio:.3} is below the 5 % floor \
         (e_in={e_in:.0}, e_out={e_out:.0})"
    );
    assert!(
        ratio < 1_000.0,
        "decoded energy {e_out:.0} is absurdly larger than input {e_in:.0}"
    );
}

#[test]
fn silence_roundtrip_stays_bounded() {
    let input = vec![0i16; TOTAL_SAMPLES];

    let mut enc = make_encoder();
    let packets = encode_all(&mut enc, &input);
    assert_eq!(packets.len(), PACKETS);
    for p in &packets {
        assert_eq!(p.data.len(), PACKET_BYTES);
    }

    let mut dec = make_decoder();
    let out = decode_all(&mut dec, &packets);

    // The encoder has to pick *some* codeword for every vector — the
    // gain codebook doesn't include an exact zero — so the decoder will
    // emit a bit of hiss on pure silence input. After the first few
    // adaptation cycles the log-gain predictor ratchets the excitation
    // scale downward and the output should stay well below ±2500. The
    // leading transient (before the predictor catches up) is allowed
    // to peak higher, so we test the tail of the stream only.
    let tail_start = out.len() / 4; // skip the first 100 ms of transient
    let tail = &out[tail_start..];
    let peak = tail.iter().map(|s| s.abs() as i32).max().unwrap_or(0);
    assert!(
        peak < 2500,
        "silence decoded to |{peak}| after transient — expected near-zero"
    );
}

#[test]
fn packet_pts_rises_across_vectors() {
    let input = build_sine(TOTAL_SAMPLES);
    let mut enc = make_encoder();
    let packets = encode_all(&mut enc, &input);
    assert!(packets.len() >= 2);
    let mut prev: Option<i64> = None;
    for p in &packets {
        let pts = p.pts.expect("encoder must stamp pts on every packet");
        if let Some(prev) = prev {
            assert!(pts > prev, "pts went backwards: {prev} then {pts}");
            assert_eq!(
                pts - prev,
                PACKET_SAMPLES as i64,
                "pts increments must be one packet ({PACKET_SAMPLES} samples)"
            );
        }
        prev = Some(pts);
    }
}

#[test]
fn register_exposes_both_directions() {
    let mut reg = CodecRegistry::new();
    oxideav_g728::register(&mut reg);
    let id = CodecId::new(CODEC_ID_STR);
    assert!(reg.has_decoder(&id), "decoder factory must be registered");
    assert!(reg.has_encoder(&id), "encoder factory must be registered");
}
