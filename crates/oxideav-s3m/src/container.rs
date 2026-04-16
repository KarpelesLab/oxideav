//! S3M container shim.
//!
//! Like MOD, S3M files are self-contained — there's no packet-level
//! framing. The container reads the whole file, parses the top-level
//! header so it can expose `CodecParameters` and metadata, and delivers
//! the entire blob to the decoder as a single packet.

use std::io::Read;

use oxideav_container::{ContainerRegistry, Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::header::{parse_header, S3mHeader};

/// Output sample rate used by the decoder. 44.1 kHz matches what the
/// MOD crate emits, keeping the pipeline uniform across tracker formats.
pub const OUTPUT_SAMPLE_RATE: u32 = 44_100;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("s3m", open);
    reg.register_extension("s3m", "s3m");
}

fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut blob = Vec::new();
    input.read_to_end(&mut blob)?;
    if blob.len() < 0x60 {
        return Err(Error::invalid("S3M: file shorter than minimum header"));
    }
    let header = parse_header(&blob)?;

    let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(2);
    params.sample_rate = Some(OUTPUT_SAMPLE_RATE);
    params.sample_format = Some(SampleFormat::S16);
    params.extradata = blob.clone();

    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64),
        duration: None,
        start_time: Some(0),
        params,
    };

    let metadata = build_metadata(&header);
    // Approximate upper-bound duration. We use the initial speed/tempo
    // and the count of non-marker orders; effects may retime the song
    // so this is only a ceiling.
    let order_count = header
        .order
        .iter()
        .take_while(|&&v| v != 0xFF)
        .filter(|&&v| v != 0xFE)
        .count() as i64;
    let speed = header.initial_speed.max(1) as i64;
    let bpm = header.initial_tempo.max(1) as i64;
    // ticks_per_sec = 2 * bpm / 5 (the Amiga formula, ST3 uses the same).
    let ticks_per_sec = (2 * bpm).max(1) / 5;
    let total_ticks = order_count * 64 * speed;
    let duration_micros = if ticks_per_sec > 0 && total_ticks > 0 {
        total_ticks.saturating_mul(1_000_000) / ticks_per_sec.max(1)
    } else {
        0
    };

    Ok(Box::new(S3mDemuxer {
        streams: vec![stream],
        blob,
        consumed: false,
        metadata,
        duration_micros,
        _header: header,
    }))
}

fn build_metadata(h: &S3mHeader) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    if !h.song_name.is_empty() {
        out.push(("title".into(), h.song_name.clone()));
    }
    for inst in h.instruments.iter() {
        let nm = if !inst.name.is_empty() {
            inst.name.clone()
        } else if !inst.dos_name.is_empty() {
            inst.dos_name.clone()
        } else {
            continue;
        };
        out.push(("sample".into(), nm));
    }
    let n_nonempty = h
        .instruments
        .iter()
        .filter(|i| i.is_pcm() && i.length > 0)
        .count();
    out.push((
        "extra_info".into(),
        format!(
            "{} patterns, {} channels, {}/{} samples",
            h.pat_num, h.enabled_channels, n_nonempty, h.ins_num
        ),
    ));
    out
}

struct S3mDemuxer {
    streams: Vec<StreamInfo>,
    blob: Vec<u8>,
    consumed: bool,
    metadata: Vec<(String, String)>,
    duration_micros: i64,
    _header: S3mHeader,
}

impl Demuxer for S3mDemuxer {
    fn format_name(&self) -> &str {
        "s3m"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.consumed {
            return Err(Error::Eof);
        }
        self.consumed = true;
        let data = std::mem::take(&mut self.blob);
        let stream = &self.streams[0];
        let mut pkt = Packet::new(0, stream.time_base, data);
        pkt.pts = Some(0);
        pkt.dts = Some(0);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        if self.duration_micros > 0 {
            Some(self.duration_micros)
        } else {
            None
        }
    }
}
