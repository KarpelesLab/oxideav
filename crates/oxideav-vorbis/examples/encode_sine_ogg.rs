//! End-to-end encode: synthesize a 1 kHz sine, run it through our Vorbis
//! encoder, then wrap the output packets in an Ogg container so ffmpeg can
//! probe and decode the result. Writes /tmp/oxideav-test/ours-vorbis.ogg.
//!
//! This is the practical interop test referenced in the encoder's quality
//! work: ffmpeg's libvorbis must accept what we produce.

#![allow(clippy::needless_range_loop)]

use std::fs;
use std::path::PathBuf;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, MediaType, SampleFormat, TimeBase,
};
use oxideav_vorbis::encoder::make_encoder;

const SR: u32 = 48_000;
const N_SAMPLES: usize = 48_000; // 1 second
const FREQ_HZ: f64 = 1000.0;
const AMP: f64 = 0.5;

fn write_ogg_pages(
    out: &mut Vec<u8>,
    serial: u32,
    mut seq_no: u32,
    packets: Vec<(Vec<u8>, i64, bool, bool)>,
) {
    // Each entry: (packet_bytes, granule_pos, is_first, is_last)
    for (data, granule, bos, eos) in packets {
        let mut pos = 0usize;
        let mut continued = false;
        while pos < data.len() || (pos == 0 && data.is_empty()) {
            let max_segments = 255usize;
            let max_bytes = max_segments * 255;
            let bytes_this_page = (data.len() - pos).min(max_bytes);
            let n_full = bytes_this_page / 255;
            let last = bytes_this_page % 255;
            // If the packet exactly fills 255-byte segments, we still need
            // a 0-length terminator segment to indicate end-of-packet.
            let mut lacing: Vec<u8> = vec![255u8; n_full];
            let last_packet_segment = if bytes_this_page == data.len() - pos {
                lacing.push(last as u8);
                true
            } else {
                false
            };
            let n_segs = lacing.len();
            // Header byte 5 flags:
            let mut hdr_flags: u8 = 0;
            if continued {
                hdr_flags |= 0x01;
            }
            if bos && pos == 0 {
                hdr_flags |= 0x02;
            }
            // EOS only on the last page of the last packet of the stream.
            let granule_for_page: i64 = if last_packet_segment { granule } else { -1 };
            if eos && last_packet_segment {
                hdr_flags |= 0x04;
            }

            let mut page_hdr: Vec<u8> = Vec::with_capacity(27 + n_segs);
            page_hdr.extend_from_slice(b"OggS");
            page_hdr.push(0);
            page_hdr.push(hdr_flags);
            page_hdr.extend_from_slice(&granule_for_page.to_le_bytes());
            page_hdr.extend_from_slice(&serial.to_le_bytes());
            page_hdr.extend_from_slice(&seq_no.to_le_bytes());
            page_hdr.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
            page_hdr.push(n_segs as u8);
            page_hdr.extend_from_slice(&lacing);

            let mut page = page_hdr.clone();
            page.extend_from_slice(&data[pos..pos + bytes_this_page]);

            // Compute CRC and patch.
            let crc = crc32_ogg(&page);
            page[22..26].copy_from_slice(&crc.to_le_bytes());

            out.extend_from_slice(&page);
            pos += bytes_this_page;
            continued = true;
            seq_no += 1;
            if data.is_empty() {
                break;
            }
        }
        let _ = continued;
    }
}

fn crc32_ogg(data: &[u8]) -> u32 {
    // Ogg uses CRC-32 with polynomial 0x04C11DB7, no inversion.
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for i in 0..256u32 {
            let mut r = i << 24;
            for _ in 0..8 {
                r = if r & 0x8000_0000 != 0 {
                    (r << 1) ^ 0x04C1_1DB7
                } else {
                    r << 1
                };
            }
            t[i as usize] = r;
        }
        t
    });
    let mut crc: u32 = 0;
    for &b in data {
        crc = (crc << 8) ^ table[((crc >> 24) as u8 ^ b) as usize];
    }
    crc
}

fn split_xiph_lacing(blob: &[u8]) -> Vec<Vec<u8>> {
    let n = blob[0] as usize + 1;
    let mut sizes = Vec::with_capacity(n);
    let mut i = 1usize;
    for _ in 0..n - 1 {
        let mut s = 0usize;
        loop {
            let b = blob[i];
            i += 1;
            s += b as usize;
            if b < 255 {
                break;
            }
        }
        sizes.push(s);
    }
    sizes.push(blob.len() - i - sizes.iter().sum::<usize>());
    let mut out = Vec::with_capacity(n);
    for sz in sizes {
        out.push(blob[i..i + sz].to_vec());
        i += sz;
    }
    out
}

fn make_encoder_for(channels: u16) -> Box<dyn Encoder> {
    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.media_type = MediaType::Audio;
    params.channels = Some(channels);
    params.sample_rate = Some(SR);
    make_encoder(&params).expect("make_encoder")
}

fn encode(channels: u16, samples: &[i16]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut enc = make_encoder_for(channels);
    let extradata = enc.output_params().extradata.clone();
    let n_per_ch = samples.len() / channels as usize;
    let mut data = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    let frame = Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels,
        sample_rate: SR,
        samples: n_per_ch as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, SR as i64),
        data: vec![data],
    });
    enc.send_frame(&frame).expect("send_frame");
    enc.flush().expect("flush");
    let mut packets = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        packets.push(p.data);
    }
    (extradata, packets)
}

fn write_ogg(channels: u16, audio_packets: &[Vec<u8>], extradata: &[u8], out_path: &PathBuf) {
    // Split extradata into 3 header packets.
    let headers = split_xiph_lacing(extradata);
    assert_eq!(headers.len(), 3);

    let mut bytes = Vec::new();
    let serial = 0xCAFEu32;

    // Packet 1: identification — granule 0, BOS.
    write_ogg_pages(
        &mut bytes,
        serial,
        0,
        vec![(headers[0].clone(), 0, true, false)],
    );
    // Packet 2 + 3: comment + setup, on a single page (granule 0).
    let mut combined = headers[1].clone();
    combined.extend_from_slice(&headers[2]);
    // Unfortunately, two distinct packets need to be on separate lacing entries.
    // Easier: write a synthetic page by hand.
    {
        // Build a single page with two complete packets via lacing.
        let p2 = &headers[1];
        let p3 = &headers[2];
        let mut lacing: Vec<u8> = Vec::new();
        let mut sz = p2.len();
        while sz >= 255 {
            lacing.push(255);
            sz -= 255;
        }
        lacing.push(sz as u8);
        let mut sz = p3.len();
        while sz >= 255 {
            lacing.push(255);
            sz -= 255;
        }
        lacing.push(sz as u8);
        let n_segs = lacing.len();
        let mut page = Vec::with_capacity(27 + n_segs + p2.len() + p3.len());
        page.extend_from_slice(b"OggS");
        page.push(0);
        page.push(0); // flags
        page.extend_from_slice(&0i64.to_le_bytes()); // granule = 0
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&1u32.to_le_bytes()); // seq_no = 1
        page.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
        page.push(n_segs as u8);
        page.extend_from_slice(&lacing);
        page.extend_from_slice(p2);
        page.extend_from_slice(p3);
        let crc = crc32_ogg(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&page);
    }

    // Audio packets: pack with running granule positions.
    let blocksize_long = 2048u64;
    let mut granule: i64 = 0;
    let mut seq_no: u32 = 2;
    for (i, pkt) in audio_packets.iter().enumerate() {
        let is_last = i + 1 == audio_packets.len();
        // Vorbis granule position is the last sample number in the page,
        // increments by blocksize_long/2 for each long block (post-OLA).
        granule += (blocksize_long / 2) as i64;
        let g = granule;
        let mut to_write = vec![(pkt.clone(), g, false, is_last)];
        if i == 0 {
            // First audio page — granule may be 0 since first block emits no
            // OLA samples. But ffmpeg accepts non-zero too.
        }
        // Use simple per-packet pages.
        write_ogg_pages(&mut bytes, serial, seq_no, std::mem::take(&mut to_write));
        seq_no += 1;
    }
    let _ = combined;
    let _ = channels;

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(out_path, &bytes).expect("write ogg");
    println!("wrote {} ({} bytes)", out_path.display(), bytes.len());
}

fn main() {
    // Mono 1 kHz sine.
    let mut mono = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        let t = i as f64 / SR as f64;
        let s = (2.0 * std::f64::consts::PI * FREQ_HZ * t).sin() * AMP;
        mono.push((s * 32768.0) as i16);
    }
    let (extra, packets) = encode(1, &mono);
    let total: usize = packets.iter().map(|p| p.len()).sum();
    println!(
        "mono: {} packets, {} bytes total ({} bps for 1s)",
        packets.len(),
        total,
        total * 8
    );
    write_ogg(
        1,
        &packets,
        &extra,
        &PathBuf::from("/tmp/oxideav-test/ours-vorbis-mono.ogg"),
    );

    // Stereo 1 kHz sine (same on both).
    let mut stereo: Vec<i16> = Vec::with_capacity(N_SAMPLES * 2);
    for i in 0..N_SAMPLES {
        let t = i as f64 / SR as f64;
        let s = (2.0 * std::f64::consts::PI * FREQ_HZ * t).sin() * AMP;
        let q = (s * 32768.0) as i16;
        stereo.push(q);
        stereo.push(q);
    }
    let (extra_s, packets_s) = encode(2, &stereo);
    let total_s: usize = packets_s.iter().map(|p| p.len()).sum();
    println!(
        "stereo: {} packets, {} bytes total ({} bps for 1s)",
        packets_s.len(),
        total_s,
        total_s * 8
    );
    write_ogg(
        2,
        &packets_s,
        &extra_s,
        &PathBuf::from("/tmp/oxideav-test/ours-vorbis-stereo.ogg"),
    );
}
