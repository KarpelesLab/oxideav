//! Build MP4 `stsd` sample-entry payloads for specific codecs.
//!
//! This is the *only* place in the muxer where codec knowledge is encoded.
//! All other muxer code is codec-agnostic — it just appends opaque packet bytes.
//!
//! Each `sample_entry_for` returns:
//! - `fourcc`: the 4-byte sample entry type (e.g. `b"sowt"`, `b"mp4a"`, `b"fLaC"`,
//!   `b"avc1"`).
//! - `body`: the contents of the sample entry box (i.e. everything after the
//!   8-byte box header). For audio entries this begins with the 28-byte
//!   `AudioSampleEntryV0` preamble; for video entries it begins with the
//!   78-byte `VisualSampleEntry` preamble. Codec-specific subboxes
//!   (`dfLa`, `esds`, `avcC`, …) follow.
//!
//! References: ISO/IEC 14496-12 §8.5, ISO/IEC 14496-14, ISO/IEC 23003-5
//! (FLAC-in-ISOBMFF).

use oxideav_core::{CodecParameters, Error, MediaType, Result};

/// A complete sample-entry description.
pub(crate) struct SampleEntry {
    /// Sample-entry FourCC (the box type that goes inside `stsd`).
    pub fourcc: [u8; 4],
    /// Payload of the sample-entry box (everything after the 8-byte box header).
    pub body: Vec<u8>,
}

/// Build the sample entry for a stream. Errors with `Unsupported` if the codec
/// has no MP4 packaging in our table.
pub(crate) fn sample_entry_for(params: &CodecParameters) -> Result<SampleEntry> {
    match params.codec_id.as_str() {
        "pcm_s16le" => pcm_sowt(params),
        "flac" => flac_entry(params),
        "aac" => aac_entry(params),
        "h264" => h264_entry(params),
        "mjpeg" => mjpeg_entry(params),
        other => Err(Error::unsupported(format!(
            "mp4 muxer: no sample entry for codec {other}"
        ))),
    }
}

/// Motion JPEG sample entry. Modern ISOBMFF uses the `jpeg` FourCC with a
/// plain VisualSampleEntry; each sample is a self-contained JPEG byte
/// stream. (The legacy QuickTime `mjpa`/`mjpb` forms have extra quirks
/// we don't emit today.)
fn mjpeg_entry(params: &CodecParameters) -> Result<SampleEntry> {
    if params.media_type != MediaType::Video {
        return Err(Error::invalid("mp4 muxer: mjpeg must be video"));
    }
    let width = params
        .width
        .ok_or_else(|| Error::invalid("mp4 muxer: mjpeg requires width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("mp4 muxer: mjpeg requires height"))?;
    let body = visual_preamble(width, height).to_vec();
    Ok(SampleEntry {
        fourcc: *b"jpeg",
        body,
    })
}

/// 28-byte AudioSampleEntryV0 preamble.
fn audio_preamble(channels: u16, sample_size: u16, sample_rate: u32) -> [u8; 28] {
    let mut out = [0u8; 28];
    // 6 bytes reserved
    // data_reference_index = 1
    out[6] = 0;
    out[7] = 1;
    // 8 bytes reserved (version/revision/vendor in QT-style, all zero in ISO)
    // channel_count at offset 16
    out[16..18].copy_from_slice(&channels.to_be_bytes());
    // sample_size at offset 18
    out[18..20].copy_from_slice(&sample_size.to_be_bytes());
    // 2 bytes pre_defined + 2 bytes reserved
    // sample_rate as 16.16 fixed-point at offset 24
    let sr_fixed = sample_rate << 16;
    out[24..28].copy_from_slice(&sr_fixed.to_be_bytes());
    out
}

/// 78-byte VisualSampleEntry preamble.
fn visual_preamble(width: u32, height: u32) -> [u8; 78] {
    let mut out = [0u8; 78];
    // 6 bytes reserved
    // data_reference_index = 1
    out[6] = 0;
    out[7] = 1;
    // 16 bytes pre_defined/reserved (offsets 8..24)
    // width at offset 24 (u16)
    let w = width as u16;
    let h = height as u16;
    out[24..26].copy_from_slice(&w.to_be_bytes());
    out[26..28].copy_from_slice(&h.to_be_bytes());
    // horizresolution 72 dpi as 16.16 at offset 28
    let dpi = 72u32 << 16;
    out[28..32].copy_from_slice(&dpi.to_be_bytes());
    // vertresolution 72 dpi as 16.16 at offset 32
    out[32..36].copy_from_slice(&dpi.to_be_bytes());
    // reserved u32 at offset 36
    // frame_count u16 = 1 at offset 40
    out[40..42].copy_from_slice(&1u16.to_be_bytes());
    // 32 bytes compressorname (length-prefixed Pascal string) at offset 42
    // depth u16 = 0x0018 at offset 74
    out[74..76].copy_from_slice(&0x0018u16.to_be_bytes());
    // pre_defined i16 = -1 at offset 76
    out[76..78].copy_from_slice(&(-1i16).to_be_bytes());
    out
}

fn pcm_sowt(params: &CodecParameters) -> Result<SampleEntry> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("mp4 muxer: PCM requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("mp4 muxer: PCM requires sample_rate"))?;
    // sowt is 16-bit signed little-endian PCM; hard-coded 16 bps.
    let body = audio_preamble(channels, 16, sample_rate).to_vec();
    Ok(SampleEntry {
        fourcc: *b"sowt",
        body,
    })
}

fn flac_entry(params: &CodecParameters) -> Result<SampleEntry> {
    if params.media_type != MediaType::Audio {
        return Err(Error::invalid("mp4 muxer: flac must be audio"));
    }
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("mp4 muxer: flac requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("mp4 muxer: flac requires sample_rate"))?;
    // Bits per sample: pick from sample_format; default to 16.
    let bps = params
        .sample_format
        .map(|f| (f.bytes_per_sample() * 8) as u16)
        .unwrap_or(16);
    let mut body = audio_preamble(channels, bps, sample_rate).to_vec();

    // dfLa subbox: FullBox (version 0 + 3 bytes flags) followed by the
    // FLAC metadata blocks. oxideav-flac extradata is already the concatenated
    // metadata blocks (each with 4-byte header + payload).
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "mp4 muxer: flac stream missing extradata (STREAMINFO)",
        ));
    }
    let mut dfla_body = Vec::with_capacity(4 + params.extradata.len());
    dfla_body.extend_from_slice(&[0, 0, 0, 0]); // version 0 + 3 bytes flags
    dfla_body.extend_from_slice(&params.extradata);
    body.extend_from_slice(&write_simple_box(b"dfLa", &dfla_body));

    Ok(SampleEntry {
        fourcc: *b"fLaC",
        body,
    })
}

fn aac_entry(params: &CodecParameters) -> Result<SampleEntry> {
    if params.media_type != MediaType::Audio {
        return Err(Error::invalid("mp4 muxer: aac must be audio"));
    }
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("mp4 muxer: aac requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("mp4 muxer: aac requires sample_rate"))?;
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "mp4 muxer: aac stream missing extradata (AudioSpecificConfig)",
        ));
    }
    let mut body = audio_preamble(channels, 16, sample_rate).to_vec();

    // esds box (full box): ES_Descriptor wrapping DecoderConfigDescriptor wrapping
    // DecoderSpecificInfo (the AudioSpecificConfig). ObjectTypeIndication = 0x40
    // (AAC). StreamType = 0x05 (audio). See ISO/IEC 14496-1 §7.2.6.
    let asc = &params.extradata;
    // DecoderSpecificInfo (tag 0x05): length = asc.len()
    let mut dsi = Vec::new();
    dsi.push(0x05);
    append_ber_length(&mut dsi, asc.len() as u32);
    dsi.extend_from_slice(asc);

    // DecoderConfigDescriptor (tag 0x04): 13 bytes header + DSI
    let mut dcd = Vec::new();
    dcd.push(0x04);
    let dcd_payload_len = 13 + dsi.len() as u32;
    append_ber_length(&mut dcd, dcd_payload_len);
    dcd.push(0x40); // object type: AAC
    dcd.push((0x05 << 2) | 0x01); // stream type (audio=5) | upstream=0 | reserved=1
                                  // buffer_size_db (24-bit) = 0
    dcd.extend_from_slice(&[0, 0, 0]);
    // max_bitrate (32-bit) = 0
    dcd.extend_from_slice(&[0, 0, 0, 0]);
    // avg_bitrate (32-bit) = 0
    dcd.extend_from_slice(&[0, 0, 0, 0]);
    dcd.extend_from_slice(&dsi);

    // SLConfigDescriptor (tag 0x06): 1 byte predefined=2
    let mut slc = Vec::new();
    slc.push(0x06);
    append_ber_length(&mut slc, 1);
    slc.push(0x02);

    // ES_Descriptor (tag 0x03): 3-byte header + DCD + SLC
    let mut esd = Vec::new();
    esd.push(0x03);
    let esd_payload_len = 3 + dcd.len() as u32 + slc.len() as u32;
    append_ber_length(&mut esd, esd_payload_len);
    // ES_ID = 0, flags = 0
    esd.extend_from_slice(&[0, 0, 0]);
    esd.extend_from_slice(&dcd);
    esd.extend_from_slice(&slc);

    // esds FullBox: 4 bytes version/flags + ES_Descriptor
    let mut esds_body = Vec::with_capacity(4 + esd.len());
    esds_body.extend_from_slice(&[0, 0, 0, 0]);
    esds_body.extend_from_slice(&esd);
    body.extend_from_slice(&write_simple_box(b"esds", &esds_body));

    Ok(SampleEntry {
        fourcc: *b"mp4a",
        body,
    })
}

fn h264_entry(params: &CodecParameters) -> Result<SampleEntry> {
    if params.media_type != MediaType::Video {
        return Err(Error::invalid("mp4 muxer: h264 must be video"));
    }
    let width = params
        .width
        .ok_or_else(|| Error::invalid("mp4 muxer: h264 requires width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("mp4 muxer: h264 requires height"))?;
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "mp4 muxer: h264 stream missing extradata (AVCC configuration)",
        ));
    }
    let mut body = visual_preamble(width, height).to_vec();
    // avcC box: extradata assumed to already be AVCConfigurationRecord bytes.
    body.extend_from_slice(&write_simple_box(b"avcC", &params.extradata));
    Ok(SampleEntry {
        fourcc: *b"avc1",
        body,
    })
}

/// Write a simple (non-FullBox) box: 4-byte size + 4-byte fourcc + body.
fn write_simple_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let total = 8 + body.len() as u32;
    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// Append a BER-style variable-length encoding (as used in MPEG-4 descriptors).
fn append_ber_length(out: &mut Vec<u8>, mut value: u32) {
    // Emit 4 bytes: high-7-bits first, continuation flag = 0x80. We always emit
    // 4 bytes so the length is stable and easy to parse.
    let mut bytes = [0u8; 4];
    for i in (0..4).rev() {
        bytes[i] = (value & 0x7F) as u8;
        value >>= 7;
    }
    for b in &mut bytes[..3] {
        *b |= 0x80;
    }
    out.extend_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, SampleFormat};

    #[test]
    fn pcm_sowt_shape() {
        let mut p = CodecParameters::audio(CodecId::new("pcm_s16le"));
        p.channels = Some(2);
        p.sample_rate = Some(48_000);
        p.sample_format = Some(SampleFormat::S16);
        let e = sample_entry_for(&p).unwrap();
        assert_eq!(&e.fourcc, b"sowt");
        assert_eq!(e.body.len(), 28);
        // channels big-endian at offset 16
        assert_eq!(u16::from_be_bytes([e.body[16], e.body[17]]), 2);
        // sample size at offset 18
        assert_eq!(u16::from_be_bytes([e.body[18], e.body[19]]), 16);
    }

    #[test]
    fn flac_entry_has_dfla() {
        let mut p = CodecParameters::audio(CodecId::new("flac"));
        p.channels = Some(2);
        p.sample_rate = Some(48_000);
        p.sample_format = Some(SampleFormat::S16);
        // Minimal extradata: one STREAMINFO metadata block header+payload.
        let mut extradata = Vec::new();
        extradata.extend_from_slice(&[0x80, 0, 0, 34]); // last block, type=STREAMINFO, length=34
        extradata.extend_from_slice(&[0u8; 34]);
        p.extradata = extradata;
        let e = sample_entry_for(&p).unwrap();
        assert_eq!(&e.fourcc, b"fLaC");
        // Body: 28 byte audio preamble + dfLa box (8 header + 4 version/flags + 38 metadata)
        assert_eq!(e.body.len(), 28 + 8 + 4 + 38);
        // Check the dfLa box is present at offset 28.
        assert_eq!(&e.body[32..36], b"dfLa");
    }

    #[test]
    fn unsupported_codec_errors() {
        let p = CodecParameters::audio(CodecId::new("vorbis"));
        assert!(sample_entry_for(&p).is_err());
    }
}
