//! Read a 16-bit PCM WAV file (mono or stereo, ≤2 channels) and encode it
//! with our Vorbis encoder, then wrap the output as Ogg. Used to test
//! ffmpeg interop against arbitrary content (sine, noise, mixed).
//!
//! Usage: `cargo run --example encode_wav -- <input.wav> <output.ogg>`

#![allow(clippy::needless_range_loop)]

use std::env;
use std::fs;
use std::path::PathBuf;

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, MediaType, SampleFormat, TimeBase,
};
use oxideav_vorbis::encoder::make_encoder;

fn write_ogg_pages(
    out: &mut Vec<u8>,
    serial: u32,
    mut seq_no: u32,
    packets: Vec<(Vec<u8>, i64, bool, bool)>,
) -> u32 {
    for (data, granule, bos, eos) in packets {
        let mut pos = 0usize;
        let mut continued = false;
        loop {
            let max_segments = 255usize;
            let max_bytes = max_segments * 255;
            let bytes_this_page = (data.len() - pos).min(max_bytes);
            let n_full = bytes_this_page / 255;
            let last = bytes_this_page % 255;
            let mut lacing: Vec<u8> = vec![255u8; n_full];
            let last_packet_segment = if bytes_this_page == data.len() - pos {
                lacing.push(last as u8);
                true
            } else {
                false
            };
            let n_segs = lacing.len();
            let mut hdr_flags: u8 = 0;
            if continued {
                hdr_flags |= 0x01;
            }
            if bos && pos == 0 {
                hdr_flags |= 0x02;
            }
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
            page_hdr.extend_from_slice(&0u32.to_le_bytes());
            page_hdr.push(n_segs as u8);
            page_hdr.extend_from_slice(&lacing);

            let mut page = page_hdr;
            page.extend_from_slice(&data[pos..pos + bytes_this_page]);
            let crc = crc32_ogg(&page);
            page[22..26].copy_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&page);

            pos += bytes_this_page;
            continued = true;
            seq_no += 1;
            if pos >= data.len() {
                break;
            }
        }
    }
    seq_no
}

fn crc32_ogg(data: &[u8]) -> u32 {
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

fn read_wav(path: &PathBuf) -> (u32, u16, Vec<i16>) {
    let bytes = fs::read(path).expect("read");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut i = 12usize;
    let mut sr: u32 = 0;
    let mut nch: u16 = 0;
    let mut data_off = 0usize;
    let mut data_len = 0usize;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            nch = u16::from_le_bytes(bytes[i + 10..i + 12].try_into().unwrap());
            sr = u32::from_le_bytes(bytes[i + 12..i + 16].try_into().unwrap());
        } else if id == b"data" {
            data_off = i + 8;
            data_len = sz;
            break;
        }
        i += 8 + sz;
    }
    let pcm: Vec<i16> = bytes[data_off..data_off + data_len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    (sr, nch, pcm)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_path = args.get(1).expect("usage: encode_wav <in.wav> <out.ogg>");
    let out_path = args.get(2).expect("usage: encode_wav <in.wav> <out.ogg>");
    let (sr, nch, pcm) = read_wav(&PathBuf::from(in_path));
    println!(
        "input: {} Hz, {} ch, {} samples/ch",
        sr,
        nch,
        pcm.len() / nch as usize
    );

    let mut params = CodecParameters::audio(CodecId::new("vorbis"));
    params.media_type = MediaType::Audio;
    params.channels = Some(nch);
    params.sample_rate = Some(sr);
    let mut enc = make_encoder(&params).expect("make_encoder");
    let extradata = enc.output_params().extradata.clone();

    let mut data = Vec::with_capacity(pcm.len() * 2);
    for s in &pcm {
        data.extend_from_slice(&s.to_le_bytes());
    }
    let frame = Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: nch,
        sample_rate: sr,
        samples: (pcm.len() / nch as usize) as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, sr as i64),
        data: vec![data],
    });
    enc.send_frame(&frame).expect("send_frame");
    enc.flush().expect("flush");
    let mut packets = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        packets.push(p.data);
    }
    let total: usize = packets.iter().map(|p| p.len()).sum();
    println!(
        "encoded: {} packets, {} bytes ({} bps)",
        packets.len(),
        total,
        total * 8 * sr as usize / (pcm.len() / nch as usize)
    );

    // Wrap as Ogg.
    let headers = split_xiph_lacing(&extradata);
    assert_eq!(headers.len(), 3);
    let serial = 0xCAFEu32;
    let mut bytes = Vec::new();

    let mut seq_no = write_ogg_pages(
        &mut bytes,
        serial,
        0,
        vec![(headers[0].clone(), 0, true, false)],
    );

    // Comment + setup as separate packets on the same page.
    {
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
        page.push(0);
        page.extend_from_slice(&0i64.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&seq_no.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes());
        page.push(n_segs as u8);
        page.extend_from_slice(&lacing);
        page.extend_from_slice(p2);
        page.extend_from_slice(p3);
        let crc = crc32_ogg(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&page);
        seq_no += 1;
    }

    let blocksize_long: i64 = 2048;
    let mut granule: i64 = 0;
    for (i, pkt) in packets.iter().enumerate() {
        let is_last = i + 1 == packets.len();
        granule += blocksize_long / 2;
        seq_no = write_ogg_pages(
            &mut bytes,
            serial,
            seq_no,
            vec![(pkt.clone(), granule, false, is_last)],
        );
    }

    fs::write(out_path, &bytes).expect("write");
    println!("wrote {} ({} bytes)", out_path, bytes.len());
}
