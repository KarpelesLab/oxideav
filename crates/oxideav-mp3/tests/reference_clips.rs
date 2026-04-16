//! Integration tests against /tmp/ref-mp3-*.mp3 files.
//!
//! These tests are guarded by the presence of the files — if they are
//! missing (e.g., cold CI environment) the tests silently pass as
//! skipped. Regenerate with:
//!
//!     ffmpeg -f lavfi -i "sine=f=440:d=1:sample_rate=48000" \
//!            -ac 1 -b:a 128k /tmp/ref-mp3-mono-cbr.mp3
//!     ffmpeg -f lavfi -i "sine=f=1000:d=1:sample_rate=48000" \
//!            -ac 2 -b:a 192k /tmp/ref-mp3-stereo-cbr.mp3

use std::fs;
use std::path::Path;

use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::frame::{parse_frame_header, FrameHeader};
use oxideav_mp3::sideinfo::SideInfo;
use oxideav_mp3::CODEC_ID_STR;

fn skip_id3v2(data: &[u8]) -> usize {
    if data.len() >= 10 && &data[0..3] == b"ID3" {
        let size = ((data[6] as u32 & 0x7F) << 21)
            | ((data[7] as u32 & 0x7F) << 14)
            | ((data[8] as u32 & 0x7F) << 7)
            | (data[9] as u32 & 0x7F);
        10 + size as usize
    } else {
        0
    }
}

fn find_first_sync(data: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[test]
fn parse_mono_first_frame_header_and_sideinfo() {
    let p = Path::new("/tmp/ref-mp3-mono-cbr.mp3");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let offset = find_first_sync(&data, start).expect("no sync word");
    let hdr: FrameHeader = parse_frame_header(&data[offset..]).expect("header parse");
    assert_eq!(hdr.sample_rate, 48_000);
    assert_eq!(hdr.channels(), 1);
    // Skip header (4) + CRC (0 when no_crc) + side info (17 for mono).
    let side_start = offset + 4;
    let si = SideInfo::parse_mpeg1(&hdr, &data[side_start..]).expect("side info");
    assert_eq!(si.channels, 1);
}

#[test]
fn parse_stereo_first_frame_header_and_sideinfo() {
    let p = Path::new("/tmp/ref-mp3-stereo-cbr.mp3");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let offset = find_first_sync(&data, start).expect("no sync word");
    let hdr: FrameHeader = parse_frame_header(&data[offset..]).expect("header parse");
    assert_eq!(hdr.sample_rate, 48_000);
    assert_eq!(hdr.channels(), 2);
    let si = SideInfo::parse_mpeg1(&hdr, &data[offset + 4..]).expect("side info");
    assert_eq!(si.channels, 2);
}

#[test]
fn count_mono_frames() {
    let p = Path::new("/tmp/ref-mp3-mono-cbr.mp3");
    if !p.exists() {
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let mut pos = find_first_sync(&data, start).expect("no sync");
    let mut frames = 0usize;
    while pos + 4 <= data.len() {
        let Ok(hdr) = parse_frame_header(&data[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else {
            break;
        };
        frames += 1;
        pos += flen as usize;
    }
    // Should be ~41.7 frames for 1s @ 48kHz MPEG-1 L3 (1152/48000 s each).
    // Depending on encoder padding (Info tag frame at start) it's 38-42.
    assert!((30..=60).contains(&frames), "frames={}", frames);
}

/// Decode the first frame of the mono clip end-to-end. Now that
/// Huffman tables 7-13, 15, 16, 24 are populated this should produce a
/// real audio frame (1152 samples, mono).
#[test]
fn decode_mono_first_frame() {
    let p = Path::new("/tmp/ref-mp3-mono-cbr.mp3");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let offset = find_first_sync(&data, start).expect("no sync");
    let hdr = parse_frame_header(&data[offset..]).unwrap();
    let flen = hdr.frame_bytes().unwrap() as usize;

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    let pkt = Packet::new(
        0,
        TimeBase::new(1, 48_000),
        data[offset..offset + flen].to_vec(),
    );
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    if let oxideav_core::Frame::Audio(a) = frame {
        assert_eq!(a.samples, 1152);
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, hdr.sample_rate);
    } else {
        panic!("not an audio frame");
    }
}

/// Decode the first frame of the stereo clip end-to-end.
#[test]
fn decode_stereo_first_frame() {
    let p = Path::new("/tmp/ref-mp3-stereo-cbr.mp3");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let offset = find_first_sync(&data, start).expect("no sync");
    let hdr = parse_frame_header(&data[offset..]).unwrap();
    let flen = hdr.frame_bytes().unwrap() as usize;

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    let pkt = Packet::new(
        0,
        TimeBase::new(1, hdr.sample_rate as i64),
        data[offset..offset + flen].to_vec(),
    );
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    if let oxideav_core::Frame::Audio(a) = frame {
        assert_eq!(a.samples, 1152);
        assert_eq!(a.channels, 2);
        assert_eq!(a.sample_rate, hdr.sample_rate);
        // Interleaved S16: 1152 samples * 2 channels * 2 bytes.
        assert_eq!(a.data.len(), 1);
        assert_eq!(a.data[0].len(), 1152 * 2 * 2);
    } else {
        panic!("not an audio frame");
    }
}

/// Decode several frames of the mono 440 Hz tone and verify via Goertzel
/// that the dominant frequency component is near 440 Hz with comparable
/// magnitude to the input. The decoder warms up over the first couple of
/// frames (bit reservoir + IMDCT overlap), so we skip the very first
/// frame and analyse the next few.
#[test]
fn decode_mono_dominant_frequency_is_440hz() {
    let p = Path::new("/tmp/ref-mp3-mono-cbr.mp3");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let mut pos = find_first_sync(&data, start).expect("no sync");

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    // Decode ~20 frames and concatenate samples.
    let mut all_pcm: Vec<f32> = Vec::with_capacity(20 * 1152);
    let mut sample_rate = 0u32;
    for _ in 0..20 {
        let Ok(hdr) = parse_frame_header(&data[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > data.len() {
            break;
        }
        let pkt = Packet::new(
            0,
            TimeBase::new(1, hdr.sample_rate as i64),
            data[pos..pos + flen].to_vec(),
        );
        if dec.send_packet(&pkt).is_err() {
            break;
        }
        let Ok(frame) = dec.receive_frame() else {
            break;
        };
        if let oxideav_core::Frame::Audio(a) = frame {
            sample_rate = a.sample_rate;
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                all_pcm.push(s);
            }
        }
        pos += flen;
    }
    assert!(
        all_pcm.len() >= 4 * 1152,
        "need enough frames; got {}",
        all_pcm.len()
    );

    // Goertzel at 440 Hz over samples after the first frame's warm-up.
    let warmup = 1152usize;
    let pcm = &all_pcm[warmup..];
    let n = pcm.len();
    let target = 440.0f32;
    let k = (n as f32 * target / sample_rate as f32).round();
    let omega = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    for &x in pcm {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    let power_440 = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;

    // Compute total signal energy for ratio.
    let total_energy: f32 = pcm.iter().map(|x| x * x).sum();
    let ratio = power_440 / (total_energy + 1e-12);

    eprintln!(
        "mono Goertzel @ 440Hz: power={:.4e} total_energy={:.4e} ratio={:.4}",
        power_440, total_energy, ratio
    );
    // For a clean 440 Hz tone we expect the 440 Hz bin to dominate (the
    // Goertzel power is ~ N/2 * |X(k)|^2 normalised). A real MP3-encoded
    // tone leaks a little; require >= 25% of total energy at 440 Hz.
    // Using a generous threshold so encoder/decoder noise doesn't false-fail.
    assert!(
        ratio > 0.25,
        "440 Hz energy ratio too low: {:.4} (decoder may be wrong)",
        ratio
    );
}

/// Dump which Huffman tables the first few frames select — diagnostic
/// for populating missing tables in a follow-up session.
#[test]
fn huffman_tables_used_by_mono_clip() {
    let p = Path::new("/tmp/ref-mp3-mono-cbr.mp3");
    if !p.exists() {
        return;
    }
    let data = fs::read(p).unwrap();
    let start = skip_id3v2(&data);
    let mut pos = find_first_sync(&data, start).expect("no sync");
    let mut used = std::collections::BTreeSet::new();
    let mut count1_used = std::collections::BTreeSet::new();
    let mut window_switch = 0usize;
    let mut long_block = 0usize;
    let mut block_type_hist = [0usize; 4];
    let mut count = 0usize;
    while pos + 4 <= data.len() && count < 10 {
        let Ok(hdr) = parse_frame_header(&data[pos..]) else {
            break;
        };
        let si_bytes = hdr.side_info_bytes();
        if pos + 4 + si_bytes > data.len() {
            break;
        }
        let si = SideInfo::parse_mpeg1(&hdr, &data[pos + 4..]).expect("si");
        for gr in 0..2 {
            for ch in 0..(si.channels as usize) {
                let gc = si.granules[gr][ch];
                for t in gc.table_select.iter() {
                    used.insert(*t);
                }
                count1_used.insert(gc.count1table_select as u8);
                if gc.window_switching_flag {
                    window_switch += 1;
                    block_type_hist[gc.block_type as usize] += 1;
                } else {
                    long_block += 1;
                }
            }
        }
        let flen = hdr.frame_bytes().unwrap() as usize;
        pos += flen;
        count += 1;
    }
    eprintln!("MP3 mono first {count} frames:");
    eprintln!("  Huffman big-value tables used: {:?}", used);
    eprintln!("  count1 tables used: {:?}", count1_used);
    eprintln!(
        "  window-switching granules: {}, long: {}",
        window_switch, long_block
    );
    eprintln!("  block_type histogram: {:?}", block_type_hist);
}

/// Decode a synthesized MP3 with table-0-only encoding (silence) to
/// verify the synthesis pipeline doesn't blow up. This exercises the
/// reservoir, side-info parse, and the long-block long-block IMDCT +
/// synthesis path with all-zero coefficients. PCM should be silence.
#[test]
fn decode_silence_path_via_reservoir() {
    // Synthesise an MPEG-1 L3 mono frame with all-zero main data.
    // Header: FF FB 94 C0 (MPEG-1 L3 noCRC 128k 48kHz noPad mono)
    // Side info: 17 bytes of zeros  (main_data_begin=0, all zero
    // granules → big_values=0 ⇒ skip Huffman, scalefac_compress=0
    // ⇒ skip scalefactors, part2_3_length=0 ⇒ no bits to read).
    let header = [0xFF, 0xFB, 0x94, 0xC0];
    let side_info = [0u8; 17];
    let frame_bytes = 144 * 128_000 / 48_000; // 384
    let mut frame = Vec::with_capacity(frame_bytes);
    frame.extend_from_slice(&header);
    frame.extend_from_slice(&side_info);
    frame.resize(frame_bytes, 0);

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();
    let pkt = Packet::new(0, TimeBase::new(1, 48_000), frame);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    if let oxideav_core::Frame::Audio(a) = frame {
        assert_eq!(a.samples, 1152);
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, 48_000);
        assert_eq!(a.data.len(), 1);
        // Silence: every sample exactly zero.
        for chunk in a.data[0].chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(s, 0, "expected silence, got {}", s);
        }
    } else {
        panic!("not an audio frame");
    }
}
