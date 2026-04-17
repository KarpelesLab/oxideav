//! End-to-end smoke tests for the G.728 decoder.
//!
//! These cover the public `Decoder` trait surface:
//! - packet-in / audio-frame-out with arbitrary but valid indices,
//! - bit-stream sizing edge cases (partial-byte tails are ignored),
//! - synthesis-filter stability on zero excitation over many vectors.
//!
//! The numeric output is not bit-exact to ITU-T G.728 because the
//! codebook tables + autocorrelation windowing in this crate are
//! documented approximations. These tests assert properties (non-zero
//! energy, finite samples, bounded amplitude) rather than exact values.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_g728::{CODEC_ID_STR, INDEX_BITS, SAMPLE_RATE, VECTOR_SIZE};

/// Build a configured decoder for the standard G.728 setup.
fn make_dec() -> Box<dyn Decoder> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    oxideav_g728::decoder::make_decoder(&params).expect("make_decoder should succeed")
}

/// Pack a slice of 10-bit indices into an MSB-first byte stream.
fn pack_indices(indices: &[u16]) -> Vec<u8> {
    let total_bits = indices.len() * INDEX_BITS as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; total_bytes];
    let mut bit_pos: usize = 0;
    for &idx in indices {
        let v = idx & ((1 << INDEX_BITS) - 1);
        for b in (0..INDEX_BITS).rev() {
            let bit = ((v >> b) & 1) as u8;
            let byte_idx = bit_pos / 8;
            let shift = 7 - (bit_pos % 8);
            out[byte_idx] |= bit << shift;
            bit_pos += 1;
        }
    }
    out
}

#[test]
fn codec_emits_nonsilent_output() {
    // Build a 10-vector packet (50 samples, 6.25 ms) with varied
    // indices so the excitation has non-trivial content.
    let indices: Vec<u16> = (0..10u16)
        .map(|i| {
            let shape: u16 = (i * 13 + 3) & 0x7F;
            let sign: u16 = i & 1;
            let mag: u16 = (i >> 1) & 3;
            (shape << 3) | (sign << 2) | mag
        })
        .collect();
    let bytes = pack_indices(&indices);

    let mut dec = make_dec();
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected audio frame");
    };

    assert_eq!(a.sample_rate, SAMPLE_RATE);
    assert_eq!(a.channels, 1);
    assert_eq!(a.samples as usize, indices.len() * VECTOR_SIZE);
    assert_eq!(a.data.len(), 1);
    assert_eq!(a.data[0].len(), a.samples as usize * 2);

    // Energy + finiteness checks.
    let mut max_abs: i32 = 0;
    let mut sum_sq: u64 = 0;
    for chunk in a.data[0].chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        let v = s.unsigned_abs() as i32;
        if v > max_abs {
            max_abs = v;
        }
        sum_sq += (v as u64) * (v as u64);
    }
    assert!(sum_sq > 0, "decoder produced all-silent output");
    assert!(max_abs > 0, "decoder produced all-zero PCM");
    // Must not be saturated on every sample.
    assert!(max_abs < 32767 || sum_sq < (a.samples as u64) * 32767 * 32767);
}

#[test]
fn bitstream_length_check() {
    // The decoder should:
    //  - accept exact multiples of 10 bits (packed into ceil(bits/8) bytes),
    //  - accept packets whose tail has fewer than 10 bits (and ignore them),
    //  - reject packets that can't hold even one 10-bit index.
    let mut dec = make_dec();

    // One 10-bit index = 2 bytes (8 usable in first byte + 2 spill). Accept.
    let bytes = pack_indices(&[0x1AAu16]);
    assert_eq!(bytes.len(), 2);
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes);
    dec.send_packet(&pkt).unwrap();
    let f = dec.receive_frame().unwrap();
    if let Frame::Audio(a) = f {
        assert_eq!(a.samples as usize, VECTOR_SIZE);
    } else {
        panic!("expected audio");
    }

    // Eight 10-bit indices = 80 bits = 10 bytes exactly. Accept.
    let bytes = pack_indices(&[0x2A5u16; 8]);
    assert_eq!(bytes.len(), 10);
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes);
    dec.send_packet(&pkt).unwrap();
    let f = dec.receive_frame().unwrap();
    if let Frame::Audio(a) = f {
        assert_eq!(a.samples as usize, 8 * VECTOR_SIZE);
    } else {
        panic!("expected audio");
    }

    // One 10-bit index padded into 3 bytes (24 bits = 2 indices + 4 leftover).
    // Should decode 2 vectors and discard the 4-bit tail.
    let three = pack_indices(&[0x155u16, 0x2AAu16]);
    // Pack yields 20 bits = 3 bytes. Verify we decode 2 vectors.
    assert_eq!(three.len(), 3);
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), three);
    dec.send_packet(&pkt).unwrap();
    let f = dec.receive_frame().unwrap();
    if let Frame::Audio(a) = f {
        assert_eq!(a.samples as usize, 2 * VECTOR_SIZE);
    } else {
        panic!("expected audio");
    }

    // Tail of 9 bits (one byte + 1 bit) can't form a 10-bit index.
    // We pad the single byte with a trailing zero bit by giving 1 byte:
    // 8 bits < 10, so the decoder must reject (too short).
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), vec![0xFFu8]);
    dec.send_packet(&pkt).unwrap();
    assert!(dec.receive_frame().is_err(), "one-byte packet must error");
}

#[test]
fn synthesis_filter_stable_on_zero_excitation() {
    // Feed 200 vectors of all-zero indices (shape=0, sign=0, mag=0).
    // The shape[0] row is not actually zero (placeholder random entries)
    // but the excitation magnitude is tiny. The backward-adaptive LPC
    // must not blow up over many vectors.
    let mut dec = make_dec();
    let bytes = pack_indices(&vec![0u16; 200]);
    let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes);
    dec.send_packet(&pkt).unwrap();
    let Frame::Audio(a) = dec.receive_frame().unwrap() else {
        panic!("expected audio");
    };
    let mut max_abs: i32 = 0;
    for chunk in a.data[0].chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        // i16 is trivially in-range; the real assertion is that the
        // envelope doesn't saturate at i16::MIN for extended runs.
        assert!(s != i16::MIN, "synthesis clipped to i16::MIN: {}", s);
        if s.unsigned_abs() as i32 > max_abs {
            max_abs = s.unsigned_abs() as i32;
        }
    }
    // Must stay well below full-scale on a near-silent input.
    assert!(
        max_abs < 10_000,
        "synthesis filter amplified zero-input to {max_abs}",
    );
}

#[test]
fn decoder_handles_multiple_packets_in_sequence() {
    let mut dec = make_dec();
    // Drive 5 packets of 32 vectors each (= 160 samples = 20 ms each).
    let indices: Vec<u16> = (0..32u16)
        .map(|i| ((i * 7) & 0x7F) << 3 | ((i & 4) >> 2) << 2 | (i & 3))
        .collect();
    let bytes = pack_indices(&indices);
    let mut saw_nonzero = false;
    for i in 0..5 {
        let pkt =
            Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bytes.clone()).with_pts(i * 160);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio");
        };
        assert_eq!(a.samples as usize, 32 * VECTOR_SIZE);
        for chunk in a.data[0].chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            // Saturation everywhere would indicate a filter blowup.
            assert!(s != i16::MIN, "decoder saturated to i16::MIN");
            if s != 0 {
                saw_nonzero = true;
            }
        }
    }
    assert!(
        saw_nonzero,
        "multi-packet decode produced no non-zero samples"
    );
}
