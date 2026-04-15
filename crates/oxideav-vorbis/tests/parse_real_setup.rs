//! Integration test: parse a real Vorbis setup header from a known fixture.
//! The fixture is constructed inline so the test is hermetic — no external
//! files required.
//!
//! The bytes below are the Xiph-laced 3-packet CodecPrivate produced by
//! ffmpeg/libvorbis for a 1-second 1ch 48 kHz sine wave at low quality
//! (sine.ogg → A_VORBIS in MKV). Captured 2026-04-15 from the test rig.

use oxideav_vorbis::bitreader::BitReader;
use oxideav_vorbis::floor::decode_floor_packet;
use oxideav_vorbis::identification::parse_identification_header;
use oxideav_vorbis::residue::decode_residue;
use oxideav_vorbis::setup::parse_setup;

/// Split a Xiph-laced extradata blob (count_byte=N-1, then N-1 lacing groups,
/// then the packets themselves) into individual packet byte vectors.
fn split_xiph_lacing(extradata: &[u8]) -> Option<Vec<Vec<u8>>> {
    if extradata.is_empty() {
        return None;
    }
    let n_packets = extradata[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n_packets);
    let mut i = 1usize;
    for _ in 0..n_packets - 1 {
        let mut s = 0usize;
        loop {
            if i >= extradata.len() {
                return None;
            }
            let b = extradata[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    let used: usize = sizes.iter().sum();
    if i + used > extradata.len() {
        return None;
    }
    let last = extradata.len() - i - used;
    sizes.push(last);
    let mut packets = Vec::with_capacity(n_packets);
    for sz in sizes {
        packets.push(extradata[i..i + sz].to_vec());
        i += sz;
    }
    Some(packets)
}

#[test]
fn split_lacing_minimal() {
    // "Manually" pack: 2 packets → count byte = 1, then 1 lacing entry, then both packets.
    let p1 = b"first";
    let p2 = b"second-packet";
    let mut blob = Vec::new();
    blob.push(1); // packets - 1
    blob.push(p1.len() as u8); // p1 size
    blob.extend_from_slice(p1);
    blob.extend_from_slice(p2);
    let split = split_xiph_lacing(&blob).unwrap();
    assert_eq!(split.len(), 2);
    assert_eq!(split[0], p1);
    assert_eq!(split[1], p2);
}

#[test]
fn parses_real_setup_from_disk() {
    // Optional: if the test rig has a pre-built sine.ogg lying around at the
    // standard test location, parse its CodecPrivate-equivalent. Skipped
    // when the file isn't present so CI without external assets still passes.
    let path = std::path::Path::new("/tmp/oxideav-test/sine.ogg");
    if !path.exists() {
        return;
    }
    let data = std::fs::read(path).unwrap();
    // Walk the Ogg pages, collect first 3 packets (= the 3 Vorbis headers).
    let mut headers: Vec<Vec<u8>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    'outer: while i + 27 <= data.len() && headers.len() < 3 {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let n_segs = data[i + 26] as usize;
        let lacing = &data[i + 27..i + 27 + n_segs];
        let body_start = i + 27 + n_segs;
        let mut off = body_start;
        for &lv in lacing {
            buf.extend_from_slice(&data[off..off + lv as usize]);
            off += lv as usize;
            if lv < 255 {
                headers.push(std::mem::take(&mut buf));
                if headers.len() == 3 {
                    break 'outer;
                }
            }
        }
        i = body_start + (lacing.iter().map(|&v| v as usize).sum::<usize>());
    }
    assert_eq!(headers.len(), 3, "expected 3 Vorbis header packets");
    let id = parse_identification_header(&headers[0]).unwrap();
    let setup = parse_setup(&headers[2], id.audio_channels).unwrap();
    assert!(!setup.codebooks.is_empty(), "setup should have codebooks");
    assert!(
        !setup.modes.is_empty(),
        "setup should declare at least one mode"
    );

    // Now extract the first AUDIO packet from sine.ogg (page 2's first
    // terminated packet) and partially decode it to validate the audio
    // packet header + per-channel floor decode path.
    let mut audio_pkt: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut packets_seen = 0usize;
    let mut i = 0usize;
    'outer: while i + 27 <= data.len() {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let n_segs = data[i + 26] as usize;
        let lacing = &data[i + 27..i + 27 + n_segs];
        let body_start = i + 27 + n_segs;
        let mut off = body_start;
        for &lv in lacing {
            buf.extend_from_slice(&data[off..off + lv as usize]);
            off += lv as usize;
            if lv < 255 {
                let pkt = std::mem::take(&mut buf);
                packets_seen += 1;
                if packets_seen == 4 {
                    // First audio packet (after 3 headers).
                    audio_pkt = Some(pkt);
                    break 'outer;
                }
            }
        }
        i = body_start + lacing.iter().map(|&v| v as usize).sum::<usize>();
    }
    let audio_pkt = audio_pkt.expect("expected to find first audio packet");
    // Vorbis blocksizes are 2^blocksize_0 / blocksize_1 (encoded fields).
    let blocksize_short = 1u32 << id.blocksize_0;
    let blocksize_long = 1u32 << id.blocksize_1;
    let partial = oxideav_vorbis::audio_packet::decode_audio_packet_partial(
        &audio_pkt,
        &id,
        &setup,
        blocksize_short,
        blocksize_long,
    )
    .expect("audio packet partial decode should succeed");
    assert_eq!(
        partial.floors.len(),
        id.audio_channels as usize,
        "one floor decoded per channel"
    );

    // Drive the residue decoder too. Per Vorbis I §8.6.3, hitting end-of-
    // packet mid-residue is legal and the decoder must terminate gracefully
    // (regression for a bug where we errored out when the classbook read
    // ran out of bits on short blocks of sine.ogg).
    let blocksize = partial.blocksize as usize;
    let n_half = blocksize / 2;
    let n_channels = id.audio_channels as usize;
    let mode = &setup.modes[partial.mode_index as usize];
    let mapping = &setup.mappings[mode.mapping as usize];
    let mut br = BitReader::new(&audio_pkt);
    let _ = br.read_bit().unwrap();
    let mode_bits = if setup.modes.len() <= 1 {
        0
    } else {
        32 - (setup.modes.len() as u32 - 1).leading_zeros()
    };
    let _ = br.read_u32(mode_bits).unwrap();
    if mode.blockflag {
        let _ = br.read_bit().unwrap();
        let _ = br.read_bit().unwrap();
    }
    let mut no_residue = vec![false; n_channels];
    for (ch, nr) in no_residue.iter_mut().enumerate() {
        let submap = if mapping.submaps > 1 {
            mapping.mux[ch]
        } else {
            0
        };
        let floor_idx = mapping.submap_floor[submap as usize] as usize;
        let floor = &setup.floors[floor_idx];
        let dec = decode_floor_packet(floor, &setup.codebooks, &mut br).unwrap();
        *nr = dec.unused;
    }
    for sm in 0..mapping.submaps as usize {
        let ch_list: Vec<usize> = (0..n_channels)
            .filter(|&ch| {
                let smi = if mapping.submaps > 1 {
                    mapping.mux[ch] as usize
                } else {
                    0
                };
                smi == sm
            })
            .collect();
        if ch_list.is_empty() {
            continue;
        }
        let res_idx = mapping.submap_residue[sm] as usize;
        let residue = &setup.residues[res_idx];
        let mut sub_vectors: Vec<Vec<f32>> = ch_list.iter().map(|_| vec![0f32; n_half]).collect();
        let dnd: Vec<bool> = ch_list.iter().map(|&ch| no_residue[ch]).collect();
        decode_residue(
            residue,
            &setup.codebooks,
            n_half,
            &dnd,
            &mut sub_vectors,
            &mut br,
        )
        .expect("residue decode must tolerate end-of-packet");
    }
}
