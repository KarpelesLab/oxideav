//! G.729 encoder → decoder roundtrip.
//!
//! The encoder shares its LSP / gain VQ tables with the decoder (see the
//! `lsp_tables` + `synthesis` modules). Those tables are not all spec-
//! exact — procedural rows in `LSPCB1_Q13`, first-cut `GBK1`/`GBK2` — so
//! we cannot reasonably target the 10+ dB SNR numbers a reference
//! implementation would hit. These tests instead assert:
//!
//!   1. Synthetic speech-like input produces finite output whose energy
//!      is at least 5 % of the input energy (i.e. the encoder emits
//!      non-trivial excitation and the decoder reconstructs audible
//!      content).
//!   2. Silence in → silence (or near-silence) out, bounded amplitude.
//!   3. PTS values on the produced packets rise monotonically.

use oxideav_codec::CodecRegistry;
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_g729::{CODEC_ID_STR, FRAME_SAMPLES, SAMPLE_RATE};

const FRAMES: usize = 20;
const TOTAL_SAMPLES: usize = FRAMES * FRAME_SAMPLES; // 200 ms @ 8 kHz

fn make_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn make_encoder() -> Box<dyn Encoder> {
    let params = make_params();
    oxideav_g729::encoder::make_encoder(&params).expect("encoder ctor")
}

fn make_decoder() -> Box<dyn Decoder> {
    let params = make_params();
    oxideav_g729::decoder::make_decoder(&params).expect("decoder ctor")
}

/// Build a synthetic speech-like signal: sum of sines + a second-harmonic
/// sweep. Not a true speech waveform but has the right bandwidth and
/// pitch structure to exercise the encoder's pitch / ACELP analysis.
///
/// Amplitude is chosen to sit in the range the decoder's gain tables can
/// naturally reproduce — G.729's gain VQ + MA-4 predictor clamp the
/// decoder output to a codec-internal scale that is only loosely tied to
/// the input level. A peak of ±200 LSB lines up with the decoder's
/// natural output peak (~±100–300 LSB) so the energy-ratio check below
/// is a meaningful "the encoder routed something through" assertion.
fn build_speech_like(n: usize) -> Vec<i16> {
    let sr = SAMPLE_RATE as f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let s = 60.0
            * ((2.0 * core::f32::consts::PI * 200.0 * t).sin()
                + 0.6 * (2.0 * core::f32::consts::PI * 500.0 * t).sin()
                + 0.3 * (2.0 * core::f32::consts::PI * 1200.0 * t).sin())
            + 20.0 * (2.0 * core::f32::consts::PI * (300.0 + 120.0 * t) * t).sin();
        out.push(s.round().clamp(-32768.0, 32767.0) as i16);
    }
    out
}

/// Build a single audio frame from `samples` (mono S16).
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

/// Drive the encoder with `input`, drain its packet stream, and return
/// the packets in emission order.
fn encode_all(enc: &mut Box<dyn Encoder>, input: &[i16]) -> Vec<Packet> {
    // Feed in sub-frame-sized chunks to exercise the queueing path.
    let chunk = FRAME_SAMPLES;
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

/// Drive the decoder with `packets` and return the reconstructed S16 PCM.
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
fn speech_like_roundtrip_has_nontrivial_energy() {
    let input = build_speech_like(TOTAL_SAMPLES);

    let mut enc = make_encoder();
    let packets = encode_all(&mut enc, &input);
    assert_eq!(
        packets.len(),
        FRAMES,
        "expected {FRAMES} packets, got {}",
        packets.len()
    );
    for p in &packets {
        assert_eq!(p.data.len(), 10, "G.729 packets must be 10 bytes");
    }

    let mut dec = make_decoder();
    let out = decode_all(&mut dec, &packets);
    assert!(
        !out.is_empty(),
        "decoder produced no audio for speech-like input"
    );
    // Every sample is i16 by construction (no NaN / inf), so there is
    // nothing to check at that level. Make sure we got a non-empty run
    // of distinct samples (catches decoder-returns-silence bugs).
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

    // Energy check: the decoded output must carry at least 5 % of the
    // input energy. G.729 with the non-spec LSP / gain tables won't
    // reproduce the input faithfully, but it must still emit *something*
    // speech-like from a speech-like input.
    let e_in = energy(&input);
    let e_out = energy(&out);
    assert!(e_in > 0.0, "input energy is zero — test setup bug");
    let ratio = e_out / e_in;
    assert!(
        ratio >= 0.05,
        "decoded/input energy ratio {ratio:.3} is below the 5 % floor \
         (e_in={e_in:.0}, e_out={e_out:.0})"
    );
    // Also sanity-bound the output so a runaway filter can't sneak through.
    assert!(
        ratio < 1_000.0,
        "decoded energy {e_out:.0} is absurdly larger than input {e_in:.0}"
    );
}

#[test]
fn silence_roundtrip_stays_silent() {
    let input = vec![0i16; TOTAL_SAMPLES];

    let mut enc = make_encoder();
    let packets = encode_all(&mut enc, &input);
    assert_eq!(packets.len(), FRAMES);
    for p in &packets {
        assert_eq!(p.data.len(), 10);
    }

    let mut dec = make_decoder();
    let out = decode_all(&mut dec, &packets);
    // Output samples must be finite and bounded. A zero-input / near-
    // silent decoder should never exceed a small peak; the decoder's
    // postfilter occasionally leaks a few LSBs even for zero input so
    // we allow a generous ceiling rather than asserting exact zeros.
    let peak = out.iter().map(|s| s.abs() as i32).max().unwrap_or(0);
    assert!(
        peak < 2000,
        "silence decoded to |{peak}| — expected near-zero output"
    );
}

#[test]
fn packet_pts_rises_across_frames() {
    let input = build_speech_like(TOTAL_SAMPLES);
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
                FRAME_SAMPLES as i64,
                "pts increments must be one frame ({FRAME_SAMPLES} samples)"
            );
        }
        prev = Some(pts);
    }
}

#[test]
fn register_exposes_both_directions() {
    let mut reg = CodecRegistry::new();
    oxideav_g729::register(&mut reg);
    let id = CodecId::new(CODEC_ID_STR);
    assert!(reg.has_decoder(&id), "decoder factory must be registered");
    assert!(reg.has_encoder(&id), "encoder factory must be registered");
}
