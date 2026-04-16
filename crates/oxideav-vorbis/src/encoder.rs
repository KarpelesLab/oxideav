//! Vorbis encoder.
//!
//! Two-tier output: blocks below the silence-energy threshold are emitted as
//! a 1-byte "floor unused" packet (legally decodes to zero PCM); blocks with
//! signal go through the full pipeline (sin window → forward MDCT → floor1
//! Y-quantisation → residue VQ search) and produce a real spectrum-bearing
//! audio packet.
//!
//! The setup header is intentionally minimal — three codebooks, one
//! floor1, one residue, one mode — enough to encode recognisable audio
//! through our own decoder. It is **not** a high-quality encoder: the
//! floor analysis is per-bin nearest-dB quantisation and the residue VQ
//! is dim-2 ternary {-1, 0, +1}. The goal is round-trippable audio, not
//! perceptually-optimal compression.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::bitwriter::BitWriter;
use crate::codebook::{parse_codebook, Codebook};
use crate::dbtable::FLOOR1_INVERSE_DB;
use crate::floor::synth_floor1;
use crate::imdct::{forward_mdct_naive, sin_window_sample};
use crate::setup::Floor1;

/// Default short blocksize (power-of-two exponent). 256 samples matches
/// libvorbis's standard low-bitrate configuration.
pub const DEFAULT_BLOCKSIZE_SHORT_LOG2: u8 = 8; // 1 << 8 = 256
/// Default long blocksize. 2048 samples matches libvorbis for music content.
pub const DEFAULT_BLOCKSIZE_LONG_LOG2: u8 = 11; // 1 << 11 = 2048

/// Assemble the Vorbis Identification header (§4.2.2).
pub fn build_identification_header(
    channels: u8,
    sample_rate: u32,
    bitrate_nominal: i32,
    blocksize_0_log2: u8,
    blocksize_1_log2: u8,
) -> Vec<u8> {
    assert!(channels >= 1, "Vorbis requires at least one channel");
    assert!(sample_rate > 0, "Vorbis requires a non-zero sample rate");
    assert!(
        (6..=13).contains(&blocksize_0_log2)
            && (6..=13).contains(&blocksize_1_log2)
            && blocksize_0_log2 <= blocksize_1_log2,
        "Vorbis blocksize exponents must be in 6..=13 and short <= long"
    );

    let mut out = Vec::with_capacity(30);
    out.push(0x01);
    out.extend_from_slice(b"vorbis");
    out.extend_from_slice(&0u32.to_le_bytes()); // vorbis_version
    out.push(channels);
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // bitrate_maximum (0 = unset)
    out.extend_from_slice(&bitrate_nominal.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // bitrate_minimum
                                                // blocksize byte: low nibble = blocksize_0, high nibble = blocksize_1.
    out.push((blocksize_1_log2 << 4) | (blocksize_0_log2 & 0x0F));
    out.push(0x01); // framing bit (per Vorbis I §4.2.2)
    out
}

/// Assemble the Vorbis Comment header (§5). Uses a fixed vendor string
/// identifying this encoder; `comments` is an optional list of
/// `KEY=VALUE` strings.
pub fn build_comment_header(comments: &[String]) -> Vec<u8> {
    let vendor = concat!("oxideav-vorbis ", env!("CARGO_PKG_VERSION")).as_bytes();
    let mut out = Vec::with_capacity(32 + vendor.len());
    out.push(0x03);
    out.extend_from_slice(b"vorbis");
    out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor);
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in comments {
        let bytes = c.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out.push(0x01); // framing bit
    out
}

/// Assemble the Vorbis Setup header with a **minimal** configuration:
/// one channel (passed through `channels` for the mapping mux), one floor1
/// with a single partition class and two posts (X=0 and X=blocksize/2),
/// one residue type-2, and two modes (short / long).
///
/// The returned setup is a placeholder: decoders accept it but no real
/// content is encoded yet. Used to unblock muxer roundtrips so the
/// audio-packet encoder can be written against a known good setup shape.
pub fn build_placeholder_setup_header(channels: u8) -> Vec<u8> {
    let _ = channels;
    let mut w = BitWriter::with_capacity(64);
    // Setup packet header.
    for &b in &[0x05u32, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73] {
        w.write_u32(b, 8);
    }
    // codebook_count = 1 (minus 1 encoded).
    w.write_u32(0, 8);
    // One codebook: 1 dim, 2 entries, both length 1 (identity-ish tree).
    // Sync: 0x564342 (24 bits, LSB-first in bytes gives 'B' 'C' 'V').
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16); // dimensions = 1
    w.write_u32(2, 24); // entries = 2
    w.write_bit(false); // ordered flag
    w.write_bit(false); // sparse flag
                        // Per-entry length-1 (stored as length-1 = 0 → write 0).
    for _ in 0..2 {
        w.write_u32(0, 5); // codeword_length - 1 = 0 → length 1
    }
    w.write_u32(0, 4); // lookup_type = 0 (no VQ)

    // time_count = 0 (minus 1), placeholder value = 0 (6 bits).
    w.write_u32(0, 6);
    w.write_u32(0, 16);

    // floor_count = 0 (minus 1).
    w.write_u32(0, 6);
    // Floor0_type = 1 (floor1).
    w.write_u32(1, 16);
    // floor1 body: partitions=1 (5 bits), classes=[0] (4 bits).
    w.write_u32(1, 5);
    w.write_u32(0, 4);
    // class_dimensions[0] = 1 (stored as 1 minus one = 0).
    w.write_u32(0, 3);
    // class_subclasses[0] = 0.
    w.write_u32(0, 2);
    // No master book since subclasses=0.
    // subbooks for 1 << subclasses = 1 slot: value 0 → book_index = -1
    // (spec treats "0" as "no book", actual book = value-1).
    w.write_u32(0, 8);
    // multiplier (2 bits): 2 (stored minus-one = 1 → value 1 == mult=2).
    w.write_u32(1, 2);
    // rangebits (4 bits): ilog(n/2). For blocksize 256, n/2=128, ilog=7.
    // Use 7 so the xlist holds 7-bit X values (0..=127). This is
    // sufficient for the short block; long-block floor setup would need
    // a separate floor.
    w.write_u32(7, 4);
    // No per-partition X values because class_dimensions[0]=1 and
    // partitions=1 → dim=1 extra X after the 2 implicit (0 and 128).
    // Wait — the partition class list above points at class 0 which has
    // cdim=1, so we DO read 1 X value here (not zero). Write X=64 as
    // something in the middle.
    w.write_u32(64, 7);

    // residue_count = 0 (minus 1).
    w.write_u32(0, 6);
    // Residue0_type = 2 (residue2).
    w.write_u32(2, 16);
    w.write_u32(0, 24); // begin
    w.write_u32(0, 24); // end  (spec: values past blocksize/2 are skipped)
    w.write_u32(0, 24); // partition_size = 0+1 = 1
    w.write_u32(0, 6); // classifications-1 = 0 → 1 class
    w.write_u32(0, 8); // classbook = 0
                       // Cascade per class: 3 low bits + maybe 5 high bits.
    w.write_u32(0, 3); // low bits
    w.write_bit(false); // bitflag
                        // No cascade bits set, so no books.

    // mapping_count = 0 (minus 1).
    w.write_u32(0, 6);
    // mapping_type = 0.
    w.write_u32(0, 16);
    // submaps flag (bit): 0 (use 1 submap).
    w.write_bit(false);
    // coupling flag: 0.
    w.write_bit(false);
    // reserved 2 bits.
    w.write_u32(0, 2);
    // No mux since submaps == 1. Submap 0:
    w.write_u32(0, 8); // time index
    w.write_u32(0, 8); // floor index
    w.write_u32(0, 8); // residue index

    // mode_count = 0 (minus 1) — 1 mode.
    w.write_u32(0, 6);
    // Mode 0: blockflag=0 (short), windowtype=0, transformtype=0, mapping=0.
    w.write_bit(false);
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);

    // Framing bit.
    w.write_bit(true);
    w.finish()
}

// =================== Simple "real audio" setup header ====================

/// X positions for the floor1 X-list (excluding the 2 implicit values
/// `0` and `2^rangebits`). For blocksize_short = 256, n_half = 128, so the
/// implicit values are 0 and 128. We add 6 evenly-spaced posts in between.
const FLOOR1_EXTRA_X: [u32; 6] = [16, 32, 48, 64, 80, 96];

/// Total floor1 post count: 2 implicit + 6 extra = 8 posts.
pub const FLOOR1_N_POSTS: usize = 2 + FLOOR1_EXTRA_X.len();

/// Floor1 multiplier setting (range = 128, amp_bits = 7).
const FLOOR1_MULTIPLIER: u8 = 2;

/// Number of frequency bins per partition for the residue.
const RESIDUE_PARTITION_SIZE: u32 = 2;

/// Number of usable codebook-2 (residue VQ) entries: 3^2 = 9.
const VQ_BOOK_ENTRIES: u32 = 9;

/// Reverse the low `bits` bits of `v` (MSB↔LSB swap). Used when emitting
/// a Huffman codeword whose decoder reads bits MSB-first into `code` while
/// our `BitWriter` writes LSB-first.
fn bit_reverse(v: u32, bits: u8) -> u32 {
    let mut r = 0u32;
    for i in 0..bits {
        if (v >> i) & 1 != 0 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

/// Emit Huffman codeword for `entry` of `cb` to `w`. Bit-reverses the
/// codebook's marker code so the LSB-first stream matches the decoder's
/// MSB-first accumulation in `decode_scalar`.
fn write_huffman(w: &mut BitWriter, cb: &Codebook, entry: u32) {
    let len = cb.codeword_lengths[entry as usize];
    if len == 0 {
        return;
    }
    let code = cb.codewords[entry as usize];
    let rev = bit_reverse(code, len);
    w.write_u32(rev, len as u32);
}

/// Write codebook 0: dim=1, entries=128, all length 7, lookup_type=0. The
/// codewords end up as the natural enumeration 0..127 (every length-7 slot
/// is filled), which is what we want — entry k has marker code k.
fn write_setup_codebook_0(w: &mut BitWriter) {
    w.write_u32(0x564342, 24); // sync
    w.write_u32(1, 16); // dimensions
    w.write_u32(128, 24); // entries
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    for _ in 0..128 {
        w.write_u32(6, 5); // length - 1 = 6 → length 7
    }
    w.write_u32(0, 4); // lookup_type = 0
}

/// Write codebook 1: residue classification book. dim=1, entries=2, both
/// length 1. Codewords are 0 and 1 — perfect prefix code. We always emit
/// entry 0 (= class 0).
fn write_setup_codebook_1(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(1, 16); // dim
    w.write_u32(2, 24); // entries
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    for _ in 0..2 {
        w.write_u32(0, 5); // length-1 = 0 → length 1
    }
    w.write_u32(0, 4); // lookup_type = 0
}

/// Write codebook 2: residue VQ book. dim=2, entries=9, all length 4
/// (uniform 16-leaf tree, of which we use 9). Lookup type 1 with min=-1,
/// delta=1, value_bits=2, sequence_p=false, multiplicands=[0, 1, 2].
/// Per-dim VQ value = multiplicand - 1, so each component is in {-1, 0, +1}.
fn write_setup_codebook_2(w: &mut BitWriter) {
    w.write_u32(0x564342, 24);
    w.write_u32(2, 16); // dim
    w.write_u32(VQ_BOOK_ENTRIES, 24); // entries = 9
    w.write_bit(false); // ordered
    w.write_bit(false); // sparse
    for _ in 0..VQ_BOOK_ENTRIES {
        w.write_u32(3, 5); // length-1 = 3 → length 4
    }
    w.write_u32(1, 4); // lookup_type = 1
    write_vorbis_float(w, -1.0); // min
    write_vorbis_float(w, 1.0); // delta
    w.write_u32(1, 4); // value_bits-1 = 1 → 2 bits
    w.write_bit(false); // sequence_p
    for &m in &[0u32, 1, 2] {
        w.write_u32(m, 2);
    }
}

/// Encode `value` as a 32-bit "Vorbis float" (Vorbis I §9.2.2 reverse of
/// `read_vorbis_float`). Used for codebook lookup parameters in the setup
/// header. The Vorbis representation is (sign | biased_exponent | mantissa)
/// where `value = sign * mantissa * 2^(exponent - 788)`.
fn write_vorbis_float(w: &mut BitWriter, value: f32) {
    if value == 0.0 {
        w.write_u32(0, 32);
        return;
    }
    let abs = value.abs() as f64;
    let mut mantissa = abs;
    let mut exp: i32 = 0;
    // Normalise so mantissa fits in 21 bits and uses bit 20 as MSB.
    while mantissa < (1u64 << 20) as f64 {
        mantissa *= 2.0;
        exp -= 1;
    }
    while mantissa >= (1u64 << 21) as f64 {
        mantissa /= 2.0;
        exp += 1;
    }
    let m = mantissa as u32 & 0x001F_FFFF;
    let biased = (exp + 788) as u32;
    debug_assert!(biased < 1024, "Vorbis float exponent out of range");
    let sign_bit = if value < 0.0 { 0x8000_0000u32 } else { 0 };
    let raw = sign_bit | ((biased & 0x3FF) << 21) | m;
    w.write_u32(raw, 32);
}

/// Build a setup header capable of carrying real audio: 3 codebooks
/// (Y-encoder, residue classbook, residue VQ), one floor1, one residue
/// type 1, one mapping, one mode.
///
/// Designed for blocksize_short = 256 (n_half = 128).
pub fn build_simple_setup_header(channels: u8) -> Vec<u8> {
    let _ = channels;
    let mut w = BitWriter::with_capacity(256);
    // Setup packet header.
    for &b in &[0x05u32, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73] {
        w.write_u32(b, 8);
    }
    // codebook_count - 1 = 2.
    w.write_u32(2, 8);
    write_setup_codebook_0(&mut w);
    write_setup_codebook_1(&mut w);
    write_setup_codebook_2(&mut w);

    // time_count - 1 = 0, placeholder = 0 (16 bits).
    w.write_u32(0, 6);
    w.write_u32(0, 16);

    // floor_count - 1 = 0 → 1 floor.
    w.write_u32(0, 6);
    // floor type = 1.
    w.write_u32(1, 16);
    // floor1 body: partitions = 1, partition_class_list = [0].
    w.write_u32(1, 5);
    w.write_u32(0, 4);
    // Class 0: dim = FLOOR1_EXTRA_X.len(), subclasses = 0.
    w.write_u32((FLOOR1_EXTRA_X.len() as u32) - 1, 3);
    w.write_u32(0, 2);
    // No master book (subclasses=0).
    // Subbook list (1 << subclasses = 1 entry): book index 0 stored as raw=1
    // (decoder: raw_value - 1 = book index).
    w.write_u32(1, 8);
    // multiplier (stored - 1) = 1 → multiplier 2.
    w.write_u32(1, 2);
    // rangebits = 7 (so implicit second X = 128 = blocksize_short / 2).
    w.write_u32(7, 4);
    // Per-partition X values (cdim=6 values).
    for &x in &FLOOR1_EXTRA_X {
        w.write_u32(x, 7);
    }

    // residue_count - 1 = 0 → 1 residue.
    w.write_u32(0, 6);
    // residue_type = 1.
    w.write_u32(1, 16);
    w.write_u32(0, 24); // begin
    w.write_u32(128, 24); // end = blocksize_short / 2
    w.write_u32(RESIDUE_PARTITION_SIZE - 1, 24); // partition_size - 1 = 1
    w.write_u32(0, 6); // classifications - 1 = 0 → 1 class
    w.write_u32(1, 8); // classbook = 1
                       // Cascade for class 0: low-3-bits = 0b001 (pass 0 has a book), bitflag = 0.
    w.write_u32(0b001, 3);
    w.write_bit(false);
    // books for class 0, pass 0 (cascade bit 0 set).
    w.write_u32(2, 8); // book index 2 (the VQ book)

    // mapping_count - 1 = 0 → 1 mapping.
    w.write_u32(0, 6);
    w.write_u32(0, 16); // mapping type = 0
    w.write_bit(false); // submaps flag = 0 → 1 submap
    w.write_bit(false); // coupling flag = 0
    w.write_u32(0, 2); // reserved
                       // No mux since submaps == 1. Submap 0:
    w.write_u32(0, 8); // time index (discarded)
    w.write_u32(0, 8); // floor index
    w.write_u32(0, 8); // residue index

    // mode_count - 1 = 0 → 1 mode.
    w.write_u32(0, 6);
    w.write_bit(false); // blockflag = 0 (short)
    w.write_u32(0, 16);
    w.write_u32(0, 16);
    w.write_u32(0, 8);

    // Framing bit.
    w.write_bit(true);
    w.finish()
}

/// Decode the codebooks back from our own setup header so the encoder
/// can use them for Huffman emission. Returns the three codebooks.
fn extract_codebooks(setup: &[u8]) -> Result<Vec<Codebook>> {
    use crate::bitreader::BitReader;
    if setup.len() < 7 || setup[0] != 0x05 || &setup[1..7] != b"vorbis" {
        return Err(Error::invalid("Vorbis encoder setup magic"));
    }
    let mut br = BitReader::new(&setup[7..]);
    let count = (br.read_u32(8)? + 1) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(parse_codebook(&mut br)?);
    }
    Ok(out)
}

/// Build extradata: 3 Xiph-laced headers.
pub fn build_extradata(id: &[u8], comment: &[u8], setup: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + id.len() + comment.len() + setup.len() + 8);
    out.push(2); // packet count - 1
                 // Lacing for id and comment (setup length inferred from trailing bytes).
    for sz in [id.len(), comment.len()] {
        let mut rem = sz;
        while rem >= 255 {
            out.push(255);
            rem -= 255;
        }
        out.push(rem as u8);
    }
    out.extend_from_slice(id);
    out.extend_from_slice(comment);
    out.extend_from_slice(setup);
    out
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("Vorbis encoder: channels required"))?;
    if !(1..=2).contains(&channels) {
        return Err(Error::unsupported(format!(
            "Vorbis encoder: {channels}-channel encode not supported yet (mono + stereo only)"
        )));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("Vorbis encoder: sample_rate required"))?;

    let id_hdr = build_identification_header(
        channels as u8,
        sample_rate,
        0,
        DEFAULT_BLOCKSIZE_SHORT_LOG2,
        DEFAULT_BLOCKSIZE_LONG_LOG2,
    );
    let comment_hdr = build_comment_header(&[]);
    let setup_hdr = build_simple_setup_header(channels as u8);
    let extradata = build_extradata(&id_hdr, &comment_hdr, &setup_hdr);
    let codebooks = extract_codebooks(&setup_hdr)?;

    let mut out_params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    out_params.media_type = MediaType::Audio;
    out_params.channels = Some(channels);
    out_params.sample_rate = Some(sample_rate);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.extradata = extradata;

    let blocksize = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
    // Build a Floor1 struct mirroring the one in our setup, used by the
    // encoder to drive `synth_floor1` for the residue normalisation step.
    let floor1 = Floor1 {
        partition_class_list: vec![0],
        class_dimensions: vec![FLOOR1_EXTRA_X.len() as u8],
        class_subclasses: vec![0],
        class_masterbook: vec![0],
        class_subbook: vec![vec![0i16]], // book index 0
        multiplier: FLOOR1_MULTIPLIER,
        rangebits: 7,
        xlist: {
            let mut v = vec![0u32, 1u32 << 7];
            v.extend_from_slice(&FLOOR1_EXTRA_X);
            v
        },
    };

    Ok(Box::new(VorbisEncoder {
        codec_id: CodecId::new(crate::CODEC_ID_STR),
        out_params,
        time_base: TimeBase::new(1, sample_rate as i64),
        channels,
        blocksize,
        input_buf: vec![Vec::with_capacity(blocksize * 4); channels as usize],
        output_queue: VecDeque::new(),
        pts: 0,
        flushed: false,
        codebooks,
        floor1,
    }))
}

struct VorbisEncoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    time_base: TimeBase,
    channels: u16,
    blocksize: usize,
    /// Per-channel pending input samples (planar f32 in [-1, 1]).
    input_buf: Vec<Vec<f32>>,
    /// Encoded packets waiting to be drained by `receive_packet`.
    output_queue: VecDeque<Packet>,
    /// PTS counter — incremented by output samples per emitted packet.
    pts: i64,
    flushed: bool,
    /// Decoded codebooks (extracted from our own setup header) — used to
    /// look up Huffman codewords during audio packet emission.
    codebooks: Vec<Codebook>,
    /// Floor1 description used for floor curve reconstruction during residue
    /// normalisation. Mirrors the floor definition in the setup header.
    floor1: Floor1,
}

impl VorbisEncoder {
    /// Append samples from an [`AudioFrame`] into the per-channel input
    /// buffers, converting from whatever sample format the caller used.
    fn push_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.channels != self.channels {
            return Err(Error::invalid(format!(
                "Vorbis encoder: expected {} channels, got {}",
                self.channels, frame.channels
            )));
        }
        let n = frame.samples as usize;
        if n == 0 {
            return Ok(());
        }
        // Decode planar/interleaved into per-channel f32. Limited
        // sample-format support for now — extend as needed.
        match frame.format {
            SampleFormat::S16 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("S16 frame missing data plane"))?;
                let stride = self.channels as usize * 2;
                if plane.len() < n * stride {
                    return Err(Error::invalid("S16 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 2;
                        let sample = i16::from_le_bytes([plane[off], plane[off + 1]]);
                        self.input_buf[ch].push(sample as f32 / 32768.0);
                    }
                }
            }
            SampleFormat::F32 => {
                let plane = frame
                    .data
                    .first()
                    .ok_or_else(|| Error::invalid("F32 frame missing data plane"))?;
                let stride = self.channels as usize * 4;
                if plane.len() < n * stride {
                    return Err(Error::invalid("F32 frame: data plane too short"));
                }
                for i in 0..n {
                    for ch in 0..self.channels as usize {
                        let off = i * stride + ch * 4;
                        let v = f32::from_le_bytes([
                            plane[off],
                            plane[off + 1],
                            plane[off + 2],
                            plane[off + 3],
                        ]);
                        self.input_buf[ch].push(v);
                    }
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "Vorbis encoder: input sample format {other:?} not supported yet"
                )));
            }
        }
        Ok(())
    }

    /// Drain full blocksize-worth of input by emitting one audio packet
    /// per block. Below the silence threshold we emit the 1-byte "floor
    /// unused" packet that decodes to PCM zeros. Otherwise we run the
    /// full real-audio pipeline.
    fn drain_blocks(&mut self, force_short: bool) {
        let _ = force_short;
        let blocksize = self.blocksize;
        while self.input_buf[0].len() >= blocksize {
            // Snapshot per-channel samples for this block.
            let mut block: Vec<Vec<f32>> = Vec::with_capacity(self.channels as usize);
            for ch in 0..self.channels as usize {
                let mut v = Vec::with_capacity(blocksize);
                v.extend_from_slice(&self.input_buf[ch][..blocksize]);
                block.push(v);
            }
            // Consume blocksize samples from each channel.
            for ch in 0..self.channels as usize {
                self.input_buf[ch].drain(..blocksize);
            }
            let pkt = self.emit_block_packet(&block);
            self.output_queue.push_back(pkt);
        }
    }

    /// Decide whether the block has enough energy to warrant a real audio
    /// packet, then emit either a real packet or the silence fallback.
    fn emit_block_packet(&mut self, block: &[Vec<f32>]) -> Packet {
        let mut max_abs = 0f32;
        for ch in block {
            for &s in ch {
                let a = s.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
        }
        // Threshold chosen so explicit-zero S16 input still hits the silence
        // path (max_abs == 0). Anything with measurable energy goes through
        // the real encoder.
        if max_abs < 1.0e-6 {
            return self.emit_silent_packet();
        }
        match self.encode_block(block) {
            Some(data) => {
                let pts = self.pts;
                self.pts += self.blocksize as i64;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = Some(pts);
                pkt.dts = Some(pts);
                pkt.duration = Some(self.blocksize as i64);
                pkt.flags.keyframe = true;
                pkt
            }
            None => self.emit_silent_packet(),
        }
    }

    fn emit_silent_packet(&mut self) -> Packet {
        let mut w = BitWriter::with_capacity(2);
        // Audio packet header bit: 0 (audio).
        w.write_bit(false);
        // mode_bits = ilog(modes_count - 1) = ilog(0) = 0 with our
        // single-mode setup → no mode bits.
        // Per-channel floor: read nonzero bit = 0 → unused, no residue
        // values follow.
        for _ in 0..self.channels as usize {
            w.write_bit(false);
        }
        let data = w.finish();
        let pts = self.pts;
        self.pts += self.blocksize as i64;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(self.blocksize as i64);
        pkt.flags.keyframe = true;
        pkt
    }

    /// Run the full encode pipeline on `block` (per-channel samples already
    /// drained from the input buffer). Returns the packed packet bytes, or
    /// `None` if the encoder can't produce a valid packet (caller falls back
    /// to silence).
    fn encode_block(&self, block: &[Vec<f32>]) -> Option<Vec<u8>> {
        let n = self.blocksize;
        let n_half = n / 2;
        let n_channels = self.channels as usize;

        // 1. Window + forward MDCT for each channel.
        let mut floor_y: Vec<Vec<i32>> = Vec::with_capacity(n_channels);
        let mut residues_q: Vec<Vec<u32>> = Vec::with_capacity(n_channels);

        for ch in 0..n_channels {
            let mut windowed = vec![0f32; n];
            for i in 0..n {
                windowed[i] = block[ch][i] * sin_window_sample(i, n);
            }
            let mut spec = vec![0f32; n_half];
            forward_mdct_naive(&windowed, &mut spec);

            // 2. Floor1 analysis: per-post Y values.
            let y = self.analyse_floor(&spec, n_half);

            // 3. Reconstruct floor curve: feed unit input through synth_floor1.
            let mut curve = vec![1f32; n_half];
            let decoded = crate::floor::Floor1Decoded {
                unused: false,
                y: y.clone(),
            };
            if synth_floor1(&self.floor1, &decoded, n_half, &mut curve).is_err() {
                return None;
            }

            // 4. Residue = spectrum / floor_curve, then quantise per dim-2 pair.
            let mut quant: Vec<u32> = Vec::with_capacity(n_half / RESIDUE_PARTITION_SIZE as usize);
            let mut p = 0;
            while p + 1 < n_half {
                let r0 = if curve[p].abs() > 1e-30 {
                    spec[p] / curve[p]
                } else {
                    0.0
                };
                let r1 = if curve[p + 1].abs() > 1e-30 {
                    spec[p + 1] / curve[p + 1]
                } else {
                    0.0
                };
                // Find the best ternary {-1, 0, +1}^2 codebook entry.
                quant.push(quantise_pair(r0, r1));
                p += 2;
            }

            floor_y.push(y);
            residues_q.push(quant);
        }

        // 5. Bit-pack the audio packet.
        let mut w = BitWriter::with_capacity(64);
        w.write_bit(false); // header bit = audio
                            // mode_bits = ilog(0) = 0 (single mode).

        // Per-channel floor1 packet.
        let book0 = &self.codebooks[0];
        for ch in 0..n_channels {
            let y = &floor_y[ch];
            // nonzero = 1.
            w.write_bit(true);
            // Y[0], Y[1] as 7-bit raw.
            w.write_u32(y[0].clamp(0, 127) as u32, 7);
            w.write_u32(y[1].clamp(0, 127) as u32, 7);
            // Class 0 has subclasses=0 → no master codebook to write.
            // Then cdim values via book 0.
            for k in 2..y.len() {
                let val = y[k].clamp(0, 127) as u32;
                write_huffman(&mut w, book0, val);
            }
        }

        // Per-channel residue (type 1 layout — per-channel concatenated).
        let book1 = &self.codebooks[1]; // classbook
        let book2 = &self.codebooks[2]; // VQ book
        let n_partitions = n_half / RESIDUE_PARTITION_SIZE as usize;
        // Cascade pass 0:
        // For each partition group of `classwords_per_codeword` partitions
        // (=1 for our classbook with dim=1), per channel emit class code,
        // then per partition per channel emit VQ entries.
        //
        // Implementation matches `decode_partitioned`'s outer loop.
        for partition_idx in 0..n_partitions {
            // pass 0: emit class codeword for each channel.
            for _ch in 0..n_channels {
                // Always class 0 → entry 0 of book 1 → length-1 code "0".
                write_huffman(&mut w, book1, 0);
            }
            // Decode/encode partition values for this group of 1 partition.
            for ch in 0..n_channels {
                let entry = residues_q[ch][partition_idx];
                write_huffman(&mut w, book2, entry);
            }
        }

        Some(w.finish())
    }

    /// Per-post Y quantisation: for each X position in the floor's xlist, look
    /// at the magnitude of the spectrum near that bin and choose the Y in
    /// 0..127 whose `FLOOR1_INVERSE_DB[Y * multiplier]` is closest in log
    /// space to that magnitude. Posts beyond the spectrum length use the
    /// last valid bin.
    fn analyse_floor(&self, spec: &[f32], n_half: usize) -> Vec<i32> {
        let xlist = &self.floor1.xlist;
        let mut y = Vec::with_capacity(xlist.len());
        let mult = self.floor1.multiplier as usize;
        for &x in xlist {
            let bin = (x as usize).min(n_half.saturating_sub(1));
            // Compute peak magnitude across a small window around `bin` so
            // narrowband sinusoids don't fall through the cracks of our
            // sparse post grid.
            let lo = bin.saturating_sub(4);
            let hi = (bin + 4).min(n_half);
            let mut mag = 0f32;
            for v in &spec[lo..hi] {
                let a = v.abs();
                if a > mag {
                    mag = a;
                }
            }
            // Find the Y minimising |log(table[Y*mult]) - log(mag)|.
            let target = mag.max(1e-30).ln();
            let mut best_y = 0i32;
            let mut best_diff = f32::MAX;
            for cand in 0..128 {
                let idx = (cand * mult).min(255);
                let table_v = FLOOR1_INVERSE_DB[idx];
                let diff = (table_v.ln() - target).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_y = cand as i32;
                }
            }
            y.push(best_y);
        }
        y
    }
}

/// Find the entry in codebook 2 (3x3 ternary VQ on values {-1, 0, +1}) that
/// best matches the residue pair `(r0, r1)` in L2 distance.
fn quantise_pair(r0: f32, r1: f32) -> u32 {
    // Saturating-quantise each component to {-1, 0, +1}. Bin boundaries at
    // ±0.5 give a uniform Voronoi diagram on the 3x3 grid.
    let q = |v: f32| -> u32 {
        if v < -0.5 {
            0
        } else if v < 0.5 {
            1
        } else {
            2
        }
    };
    let m0 = q(r0);
    let m1 = q(r1);
    // Codebook entry e is constructed as: d=0 uses (e % 3), d=1 uses (e/3 % 3).
    // Solve: m0 = e % 3, m1 = (e / 3) % 3 → e = m0 + 3 * m1.
    m0 + 3 * m1
}

impl Encoder for VorbisEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::other("encoder already flushed"));
        }
        match frame {
            Frame::Audio(a) => {
                self.push_audio_frame(a)?;
                self.drain_blocks(false);
                Ok(())
            }
            Frame::Video(_) => Err(Error::invalid("Vorbis encoder received a video frame")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output_queue.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // Pad final partial block with zeros and emit one last packet
        // so total emitted samples cover the input.
        let pending = self.input_buf[0].len();
        if pending > 0 {
            for ch in 0..self.channels as usize {
                self.input_buf[ch].resize(self.blocksize, 0.0);
            }
            self.drain_blocks(true);
        }
        self.flushed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identification::parse_identification_header;
    use crate::setup::parse_setup;

    #[test]
    fn identification_header_roundtrip() {
        let bytes = build_identification_header(2, 48_000, 128_000, 8, 11);
        let id = parse_identification_header(&bytes).expect("parse");
        assert_eq!(id.audio_channels, 2);
        assert_eq!(id.audio_sample_rate, 48_000);
        assert_eq!(id.bitrate_nominal, 128_000);
        assert_eq!(id.blocksize_0, 8);
        assert_eq!(id.blocksize_1, 11);
    }

    #[test]
    fn comment_header_signature() {
        let bytes = build_comment_header(&["TITLE=Test".to_string()]);
        assert_eq!(bytes[0], 0x03);
        assert_eq!(&bytes[1..7], b"vorbis");
        // Last byte is framing bit.
        assert_eq!(*bytes.last().unwrap() & 0x01, 0x01);
    }

    #[test]
    fn placeholder_setup_parses() {
        let bytes = build_placeholder_setup_header(1);
        // Feed through our own parser to verify it's syntactically valid.
        let setup = parse_setup(&bytes, 1).expect("our placeholder setup must parse");
        assert_eq!(setup.codebooks.len(), 1);
        assert_eq!(setup.floors.len(), 1);
        assert_eq!(setup.residues.len(), 1);
        assert_eq!(setup.mappings.len(), 1);
        assert_eq!(setup.modes.len(), 1);
    }

    #[test]
    fn simple_setup_parses() {
        let bytes = build_simple_setup_header(1);
        let setup = parse_setup(&bytes, 1).expect("simple setup must parse");
        assert_eq!(setup.codebooks.len(), 3);
        assert_eq!(setup.floors.len(), 1);
        assert_eq!(setup.residues.len(), 1);
        assert_eq!(setup.modes.len(), 1);
        // Spot-check codebook 0: 128 entries, dim 1, lookup type 0.
        assert_eq!(setup.codebooks[0].entries, 128);
        assert_eq!(setup.codebooks[0].dimensions, 1);
        assert!(setup.codebooks[0].vq.is_none());
        // Codebook 2: dim 2, 9 entries, VQ type 1.
        assert_eq!(setup.codebooks[2].entries, 9);
        assert_eq!(setup.codebooks[2].dimensions, 2);
        let vq = setup.codebooks[2].vq.as_ref().unwrap();
        assert_eq!(vq.lookup_type, 1);
        assert!((vq.min - -1.0).abs() < 1e-5);
        assert!((vq.delta - 1.0).abs() < 1e-5);
    }

    #[test]
    fn vorbis_float_roundtrip() {
        use crate::bitreader::BitReader;
        for &v in &[1.0f32, -1.0, 0.5, -0.25, 16.0, 1e-5] {
            let mut w = BitWriter::new();
            write_vorbis_float(&mut w, v);
            let bytes = w.finish();
            let mut br = BitReader::new(&bytes);
            let decoded = br.read_vorbis_float().unwrap();
            assert!(
                (decoded - v).abs() / v.abs() < 1e-4,
                "roundtrip {v} → {decoded}"
            );
        }
    }

    #[test]
    fn bit_reverse_basic() {
        assert_eq!(bit_reverse(0b1011, 4), 0b1101);
        assert_eq!(bit_reverse(0b1, 1), 0b1);
        assert_eq!(bit_reverse(0b10, 2), 0b01);
        assert_eq!(bit_reverse(0b110, 3), 0b011);
    }

    #[test]
    fn extradata_lacing_splits_back() {
        let id = build_identification_header(1, 48_000, 0, 8, 11);
        let comm = build_comment_header(&[]);
        let setup = build_placeholder_setup_header(1);
        let blob = build_extradata(&id, &comm, &setup);
        assert_eq!(blob[0], 2); // packet count - 1

        // Decode via the same Xiph lacing the decoder uses.
        let n_packets = blob[0] as usize + 1;
        let mut sizes = Vec::new();
        let mut i = 1usize;
        for _ in 0..n_packets - 1 {
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
        assert_eq!(sizes[0], id.len());
        assert_eq!(sizes[1], comm.len());
        assert_eq!(sizes[2], setup.len());
    }

    #[test]
    fn make_encoder_emits_headers() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let enc = make_encoder(&params).expect("make_encoder");
        assert!(!enc.output_params().extradata.is_empty());
    }

    #[test]
    fn send_frame_emits_silent_packet_per_block() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        // Send exactly one block of S16 silence.
        let block = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: block as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; block * 2]],
        });
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc.receive_packet().expect("packet");
        assert_eq!(pkt.pts, Some(0));
        assert_eq!(pkt.duration, Some(block as i64));
        // Packet body: 1 header bit + 1 floor-nonzero bit = 2 bits → 1 byte.
        assert_eq!(pkt.data.len(), 1);
        // Both bits zero.
        assert_eq!(pkt.data[0], 0);
        // No more packets pending until next send_frame.
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }

    #[test]
    fn flush_emits_final_padded_packet() {
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(2);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();
        // Send less than one block — encoder buffers but emits nothing.
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 2,
            sample_rate: 48_000,
            samples: 64,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; 64 * 4]],
        });
        enc.send_frame(&frame).unwrap();
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
        enc.flush().unwrap();
        // After flush, a final padded packet appears.
        let pkt = enc.receive_packet().expect("flush emits packet");
        assert_eq!(pkt.pts, Some(0));
        // Then EOF.
        assert!(matches!(enc.receive_packet(), Err(Error::Eof)));
    }

    #[test]
    fn roundtrip_sine_via_our_decoder() {
        // Encode a 1 kHz sine at 48 kHz mono and check the decoded PCM is
        // non-trivial (RMS > 100, peak < 32768).
        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();

        let block = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
        let n_blocks = 8;
        let total = block * n_blocks;
        // Build a 1 kHz sine wave at amplitude 0.5.
        let mut samples = Vec::with_capacity(total);
        for i in 0..total {
            let t = i as f64 / 48_000.0;
            let s = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            let q = (s * 32768.0) as i16;
            samples.extend_from_slice(&q.to_le_bytes());
        }
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: total as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![samples],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();

        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        assert_eq!(packets.len(), n_blocks);
        // Sanity: at least one packet should be larger than the silence packet
        // (1 byte) — that proves the encoder went through the real path.
        let max_size = packets.iter().map(|p| p.data.len()).max().unwrap_or(0);
        assert!(
            max_size > 1,
            "expected at least one real audio packet, all are silence"
        );

        // Decode through our own decoder.
        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");

        let mut decoded_pcm: Vec<i16> = Vec::new();
        for pkt in &packets {
            dec.send_packet(pkt).expect("send_packet");
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    decoded_pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
        assert!(
            !decoded_pcm.is_empty(),
            "expected decoded samples, got nothing"
        );
        let mut sum_sq: f64 = 0.0;
        let mut peak: i32 = 0;
        for &s in &decoded_pcm {
            sum_sq += (s as f64) * (s as f64);
            let a = (s as i32).abs();
            if a > peak {
                peak = a;
            }
        }
        let rms = (sum_sq / decoded_pcm.len() as f64).sqrt();

        // Goertzel-style energy check at the input frequency vs random
        // off-frequencies. This proves the encoded signal isn't just noise.
        fn goertzel_mag(samples: &[i16], freq: f64, sr: f64) -> f64 {
            let omega = 2.0 * std::f64::consts::PI * freq / sr;
            let coeff = 2.0 * omega.cos();
            let mut s_prev = 0f64;
            let mut s_prev2 = 0f64;
            for &s in samples {
                let s_now = s as f64 + coeff * s_prev - s_prev2;
                s_prev2 = s_prev;
                s_prev = s_now;
            }
            (s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2).sqrt()
        }
        let target_mag = goertzel_mag(&decoded_pcm, 1000.0, 48_000.0);
        let off_freq_mag = goertzel_mag(&decoded_pcm, 7000.0, 48_000.0);
        eprintln!(
            "sine roundtrip: rms={rms} peak={peak} samples={} target_mag={target_mag} off_mag={off_freq_mag}",
            decoded_pcm.len()
        );
        assert!(rms > 100.0, "RMS too low: {rms} (expected > 100)");
        assert!(peak < 32768, "peak hit ceiling: {peak}");
        // The 1 kHz input should produce more energy at 1 kHz than at 7 kHz.
        assert!(
            target_mag > off_freq_mag,
            "target freq energy ({target_mag}) should exceed off-freq energy ({off_freq_mag})"
        );
    }

    #[test]
    fn roundtrip_silence_via_our_decoder() {
        // Encode N blocks of silence, then decode the resulting packets
        // through our own VorbisDecoder. Output should be silent PCM
        // matching the input sample count (modulo Vorbis's first-packet
        // discard).
        use crate::decoder::make_decoder as make_dec;
        let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        params.channels = Some(1);
        params.sample_rate = Some(48_000);
        let mut enc = make_encoder(&params).unwrap();

        let block = 1usize << DEFAULT_BLOCKSIZE_SHORT_LOG2;
        let n_blocks = 4;
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
            samples: (block * n_blocks) as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, 48_000),
            data: vec![vec![0u8; block * n_blocks * 2]],
        });
        enc.send_frame(&frame).unwrap();
        enc.flush().unwrap();

        let mut packets = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            packets.push(p);
        }
        assert_eq!(packets.len(), n_blocks);

        // Round-trip via our decoder using the encoder's emitted extradata.
        let mut dec_params = enc.output_params().clone();
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = make_dec(&dec_params).expect("decoder accepts our extradata");

        let mut emitted = 0usize;
        for pkt in &packets {
            dec.send_packet(pkt).expect("send_packet");
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                emitted += a.samples as usize;
                // All bytes should be zero (silence).
                for plane in &a.data {
                    assert!(plane.iter().all(|&b| b == 0), "expected silence");
                }
            }
        }
        // First packet emits 0 samples (warm-up); remaining n-1 emit
        // n_half samples each.
        assert!(emitted > 0, "expected some samples decoded");
    }
}
