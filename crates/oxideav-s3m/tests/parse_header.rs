//! Synthetic S3M fixture: exercises the full header → pattern →
//! samples → decoder pipeline without relying on a real .s3m file.
//!
//! The fixture lays out a tiny 4-channel module with one 16-sample
//! square-wave PCM instrument and one pattern that triggers that
//! instrument at a few pitches.

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase};
use oxideav_s3m::container::OUTPUT_SAMPLE_RATE;
use oxideav_s3m::decoder;
use oxideav_s3m::header::{parse_header, S3M_SIGNATURE};
use oxideav_s3m::pattern::unpack_all;
use oxideav_s3m::samples::extract_samples;

/// Build a minimal valid S3M byte sequence.
///
/// Layout (offsets fed in sequence — we track them manually):
/// ```text
///   0x00..0x60   Top-level header (96 bytes)
///   0x60..0x62   Order table (2 entries: pattern 0, 0xFF end)
///   0x62..0x64   Instrument parapointer table (1 entry)
///   0x64..0x66   Pattern parapointer table (1 entry)
///   align to 16  Instrument header (80 bytes)
///   align to 16  Pattern body (2-byte length + packed records)
///   align to 16  Sample body (16 bytes of 8-bit unsigned)
/// ```
fn build_synth_s3m() -> (Vec<u8>, SynthLayout) {
    let mut out = vec![0u8; 0x60];

    // 0x00: song name (up to 27 chars + 0x00).
    let name = b"SYNTH-TEST";
    out[..name.len()].copy_from_slice(name);
    out[0x1C] = 0x1A; // EOT marker
    out[0x1D] = 0x10; // type = S3M

    // Counts: 2 orders, 1 inst, 1 pattern.
    out[0x20..0x22].copy_from_slice(&2u16.to_le_bytes());
    out[0x22..0x24].copy_from_slice(&1u16.to_le_bytes());
    out[0x24..0x26].copy_from_slice(&1u16.to_le_bytes());
    // flags
    out[0x26..0x28].copy_from_slice(&0u16.to_le_bytes());
    // tracker version
    out[0x28..0x2A].copy_from_slice(&0x1320u16.to_le_bytes());
    // FFI: 2 = unsigned samples (the common ST3 case)
    out[0x2A..0x2C].copy_from_slice(&2u16.to_le_bytes());
    // Signature.
    out[0x2C..0x30].copy_from_slice(S3M_SIGNATURE);
    // Global volume = 64.
    out[0x30] = 64;
    // Initial speed = 6.
    out[0x31] = 6;
    // Initial tempo = 125.
    out[0x32] = 125;
    // Master volume = 0x30 | 0x80 (stereo).
    out[0x33] = 0x30 | 0x80;
    // Default-pan flag: not 0xFC (use channel-derived pans).
    out[0x35] = 0x00;
    // Channel settings: 4 active channels (slots 0..=3), rest 0xFF.
    for (i, c) in out[0x40..0x40 + 32].iter_mut().enumerate() {
        *c = if i < 4 { i as u8 } else { 0xFF };
    }

    // Order table (2 bytes): pattern 0, 0xFF end.
    out.extend_from_slice(&[0, 0xFF]);
    // Instrument parapointer table (2 bytes): placeholder, patched below.
    let ins_pp_off = out.len();
    out.extend_from_slice(&[0, 0]);
    // Pattern parapointer table (2 bytes): placeholder, patched below.
    let pat_pp_off = out.len();
    out.extend_from_slice(&[0, 0]);

    // Pad to next 16-byte boundary for the instrument header.
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let inst_off = out.len();
    assert_eq!(inst_off % 16, 0);
    let inst_parapointer = (inst_off >> 4) as u16;

    // 80-byte instrument header.
    let mut inst = vec![0u8; 80];
    inst[0] = 1; // PCM
                 // DOS filename "square.wav"
    let dn = b"square.wav";
    inst[1..1 + dn.len()].copy_from_slice(dn);
    // MemSeg (parapointer to sample body) — patched once sample offset is known.
    // length = 16 samples (LE u32 at 0x10).
    inst[0x10..0x14].copy_from_slice(&16u32.to_le_bytes());
    // loop_start = 0, loop_end = 16, flag bit0 (loop).
    inst[0x14..0x18].copy_from_slice(&0u32.to_le_bytes());
    inst[0x18..0x1C].copy_from_slice(&16u32.to_le_bytes());
    inst[0x1C] = 64; // volume
    inst[0x1F] = 1; // flags: loop
                    // C5 speed = 8363 Hz.
    inst[0x20..0x24].copy_from_slice(&8363u32.to_le_bytes());
    // Sample name.
    let nm = b"square wave";
    inst[0x30..0x30 + nm.len()].copy_from_slice(nm);
    inst[0x4C..0x50].copy_from_slice(b"SCRS");
    out.extend_from_slice(&inst);

    // Pattern body, 16-byte aligned.
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let pat_off = out.len();
    let pat_parapointer = (pat_off >> 4) as u16;
    // Pattern: 4 rows with note-triggers on channel 0, then empty rows.
    // Record layout: flags | note | inst | [vol] | [cmd info]
    // We'll use flags = 0x20 (note+inst) so channel=0, no volume/cmd.
    let mut pat_body = Vec::new();
    let notes: [u8; 4] = [0x50, 0x52, 0x54, 0x55]; // C5, D5, E5, F5 (approx)
    for &n in &notes {
        pat_body.push(0x20); // channel 0, note+inst flag
        pat_body.push(n);
        pat_body.push(1); // instrument 1
        pat_body.push(0); // row terminator
    }
    // Fill remaining 60 rows with just terminators.
    pat_body.resize(pat_body.len() + 60, 0);
    // 2-byte length header (includes itself).
    let total_len = (2 + pat_body.len()) as u16;
    out.extend_from_slice(&total_len.to_le_bytes());
    out.extend_from_slice(&pat_body);

    // Sample body, 16-byte aligned.
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let sample_off = out.len();
    let sample_parapointer = (sample_off >> 4) as u32;
    // 16-sample unsigned 8-bit square wave: 8 high (0xE0), 8 low (0x20).
    for i in 0..16 {
        out.push(if i < 8 { 0xE0 } else { 0x20 });
    }

    // Patch instrument memseg with sample parapointer.
    // memseg: hi at 0x0D, lo at 0x0E..0x10 (LE).
    let mem_hi = (sample_parapointer >> 16) as u8;
    let mem_lo = (sample_parapointer & 0xFFFF) as u16;
    out[inst_off + 0x0D] = mem_hi;
    out[inst_off + 0x0E..inst_off + 0x10].copy_from_slice(&mem_lo.to_le_bytes());

    // Patch parapointer tables.
    out[ins_pp_off..ins_pp_off + 2].copy_from_slice(&inst_parapointer.to_le_bytes());
    out[pat_pp_off..pat_pp_off + 2].copy_from_slice(&pat_parapointer.to_le_bytes());

    let layout = SynthLayout {
        inst_off,
        pat_off,
        sample_off,
    };
    (out, layout)
}

#[allow(dead_code)]
struct SynthLayout {
    inst_off: usize,
    pat_off: usize,
    sample_off: usize,
}

#[test]
fn header_parses() {
    let (bytes, _) = build_synth_s3m();
    let h = parse_header(&bytes).unwrap();
    assert_eq!(h.song_name, "SYNTH-TEST");
    assert_eq!(h.ord_num, 2);
    assert_eq!(h.ins_num, 1);
    assert_eq!(h.pat_num, 1);
    assert_eq!(h.initial_speed, 6);
    assert_eq!(h.initial_tempo, 125);
    assert_eq!(h.global_volume, 64);
    assert!(h.stereo);
    assert_eq!(h.enabled_channels, 4);
    assert_eq!(h.instruments.len(), 1);
    assert_eq!(h.instruments[0].kind, 1);
    assert_eq!(h.instruments[0].length, 16);
    assert_eq!(h.instruments[0].c5_speed, 8363);
    assert_eq!(h.instruments[0].name, "square wave");
    assert_eq!(h.instruments[0].tag, *b"SCRS");
}

#[test]
fn samples_and_patterns_parse() {
    let (bytes, _) = build_synth_s3m();
    let h = parse_header(&bytes).unwrap();
    let samples = extract_samples(&h, &bytes);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].pcm.len(), 16);
    // First half should be strongly positive (0xE0 = 224, - 128 = 96, *256 ≈ 24576).
    assert!(samples[0].pcm[0] > 10_000);
    // Second half should be strongly negative.
    assert!(samples[0].pcm[8] < -10_000);

    let patterns = unpack_all(&h, &bytes);
    assert_eq!(patterns.len(), 1);
    assert_eq!(patterns[0].rows.len(), 64);
    // First four rows trigger a note on channel 0.
    for r in 0..4 {
        assert_ne!(patterns[0].rows[r][0].note, 0xFF);
        assert_eq!(patterns[0].rows[r][0].instrument, 1);
    }
}

#[test]
fn container_roundtrip_and_metadata() {
    use std::io::Cursor;

    let (bytes, _) = build_synth_s3m();
    let mut registries_containers = ContainerRegistry::new();
    oxideav_s3m::register_containers(&mut registries_containers);

    // Open via the registry by format name.
    let cur = Box::new(Cursor::new(bytes.clone()));
    let mut demux = registries_containers
        .open_demuxer("s3m", cur)
        .expect("demuxer opens synthetic s3m");

    let md = demux.metadata();
    assert!(md.iter().any(|(k, _)| k == "title"));
    assert!(md.iter().any(|(k, _)| k == "sample"));
    assert!(md.iter().any(|(k, _)| k == "extra_info"));
    assert!(demux.duration_micros().unwrap_or(0) > 0);

    let pkt = demux.next_packet().unwrap();
    assert!(pkt.data.len() > 0x60);
    // Second next_packet must be Eof.
    matches!(demux.next_packet(), Err(Error::Eof));
}

#[test]
fn decoder_emits_nonsilent_pcm() {
    let (bytes, _) = build_synth_s3m();

    let mut codec_reg = CodecRegistry::new();
    decoder::register(&mut codec_reg);

    let params = CodecParameters::audio(CodecId::new("s3m"));
    let mut dec = codec_reg.make_decoder(&params).expect("build s3m decoder");

    let tb = TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64);
    let pkt = Packet::new(0, tb, bytes);
    dec.send_packet(&pkt).unwrap();

    let mut total_samples = 0u64;
    let mut total_nonzero = 0u64;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                assert_eq!(a.channels, 2);
                assert_eq!(a.sample_rate, OUTPUT_SAMPLE_RATE);
                assert_eq!(a.format, SampleFormat::S16);
                total_samples += a.samples as u64;
                for chunk in a.data[0].chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    if s != 0 {
                        total_nonzero += 1;
                    }
                }
            }
            Ok(_) => unreachable!("S3M emits audio only"),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected decode error: {e:?}"),
        }
    }
    assert!(
        total_samples > 1000,
        "expected substantial sample output, got {total_samples}"
    );
    assert!(
        total_nonzero > 100,
        "expected non-silent PCM, got {total_nonzero} non-zero samples"
    );

    // Expected duration: 64 rows × 6 ticks/row × (44100 * 2.5 / 125) =
    // 64 × 6 × 882 = 338688 frames.
    // Tolerate a small ± drift due to integer rounding in samples_per_tick.
    assert!(
        (300_000..400_000).contains(&total_samples),
        "total frames {} out of expected window",
        total_samples
    );
}

/// Smoke-test `Axx` (set speed): speed=12 should roughly double the
/// pattern's wall time compared to default speed=6. We build a variant
/// where the pattern's first row carries command A with info=12.
#[test]
fn effect_axx_doubles_runtime() {
    let bytes_fast = build_synth_s3m().0; // default speed=6
    let bytes_slow = build_synth_s3m_with_axx(12);

    let fast = total_frames_for(&bytes_fast);
    let slow = total_frames_for(&bytes_slow);
    // speed=12 means twice as many ticks per row → ~2x sample count.
    // Allow +/- 15% tolerance.
    let ratio = slow as f32 / fast as f32;
    assert!(
        (1.7..2.3).contains(&ratio),
        "expected ~2x runtime for double-speed (got ratio {:.2}; fast={fast}, slow={slow})",
        ratio
    );
}

fn total_frames_for(bytes: &[u8]) -> u64 {
    let mut codec_reg = CodecRegistry::new();
    decoder::register(&mut codec_reg);
    let params = CodecParameters::audio(CodecId::new("s3m"));
    let mut dec = codec_reg.make_decoder(&params).unwrap();
    let tb = TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64);
    let pkt = Packet::new(0, tb, bytes.to_vec());
    dec.send_packet(&pkt).unwrap();

    let mut total = 0u64;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => total += a.samples as u64,
            Ok(_) => unreachable!(),
            Err(Error::Eof) => break,
            Err(e) => panic!("decode error: {e:?}"),
        }
    }
    total
}

/// Variant of `build_synth_s3m` that adds an Axx (set speed) effect on
/// row 0 channel 0 — the rest of the song plays at the new speed.
fn build_synth_s3m_with_axx(new_speed: u8) -> Vec<u8> {
    // Rebuild the synthetic module from scratch but swap the pattern body
    // to include an Axx on the very first row.
    let mut out = vec![0u8; 0x60];
    let name = b"SYNTH-AXX";
    out[..name.len()].copy_from_slice(name);
    out[0x1C] = 0x1A;
    out[0x1D] = 0x10;
    out[0x20..0x22].copy_from_slice(&2u16.to_le_bytes());
    out[0x22..0x24].copy_from_slice(&1u16.to_le_bytes());
    out[0x24..0x26].copy_from_slice(&1u16.to_le_bytes());
    out[0x28..0x2A].copy_from_slice(&0x1320u16.to_le_bytes());
    out[0x2A..0x2C].copy_from_slice(&2u16.to_le_bytes());
    out[0x2C..0x30].copy_from_slice(S3M_SIGNATURE);
    out[0x30] = 64;
    out[0x31] = 6;
    out[0x32] = 125;
    out[0x33] = 0x30 | 0x80;
    for (i, c) in out[0x40..0x40 + 32].iter_mut().enumerate() {
        *c = if i < 4 { i as u8 } else { 0xFF };
    }
    out.extend_from_slice(&[0, 0xFF]);
    let ins_pp_off = out.len();
    out.extend_from_slice(&[0, 0]);
    let pat_pp_off = out.len();
    out.extend_from_slice(&[0, 0]);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let inst_off = out.len();
    let inst_parapointer = (inst_off >> 4) as u16;
    let mut inst = vec![0u8; 80];
    inst[0] = 1;
    inst[0x10..0x14].copy_from_slice(&16u32.to_le_bytes());
    inst[0x18..0x1C].copy_from_slice(&16u32.to_le_bytes());
    inst[0x1C] = 64;
    inst[0x1F] = 1;
    inst[0x20..0x24].copy_from_slice(&8363u32.to_le_bytes());
    inst[0x4C..0x50].copy_from_slice(b"SCRS");
    out.extend_from_slice(&inst);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let pat_off = out.len();
    let pat_parapointer = (pat_off >> 4) as u16;
    // Row 0: channel 0 with note+inst AND command Axx.
    // flags = 0x20 (note+inst) | 0x80 (cmd+info) = 0xA0
    let mut pat_body: Vec<u8> = vec![
        0xA0, 0x50, // C-5
        1, 1, // command A (speed)
        new_speed, 0, // end row 0
    ];
    // Rows 1..=3: just triggers.
    for n in [0x52u8, 0x54, 0x55] {
        pat_body.extend_from_slice(&[0x20, n, 1, 0]);
    }
    pat_body.resize(pat_body.len() + 60, 0);
    let total_len = (2 + pat_body.len()) as u16;
    out.extend_from_slice(&total_len.to_le_bytes());
    out.extend_from_slice(&pat_body);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    let sample_off = out.len();
    let sample_parapointer = (sample_off >> 4) as u32;
    for i in 0..16 {
        out.push(if i < 8 { 0xE0 } else { 0x20 });
    }
    let mem_hi = (sample_parapointer >> 16) as u8;
    let mem_lo = (sample_parapointer & 0xFFFF) as u16;
    out[inst_off + 0x0D] = mem_hi;
    out[inst_off + 0x0E..inst_off + 0x10].copy_from_slice(&mem_lo.to_le_bytes());
    out[ins_pp_off..ins_pp_off + 2].copy_from_slice(&inst_parapointer.to_le_bytes());
    out[pat_pp_off..pat_pp_off + 2].copy_from_slice(&pat_parapointer.to_le_bytes());
    out
}
