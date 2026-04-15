//! MOD as a container format.
//!
//! MOD files are self-contained and don't have a natural packetisation,
//! so the container here is a thin shim: it reads the whole file into
//! memory, parses the header to populate the stream's `CodecParameters`
//! (channel count, sample rate, sample format), then delivers the entire
//! file as a single packet to the codec.

use std::io::Read;

use oxideav_container::{ContainerRegistry, Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::header::parse_header;

/// Output sample rate used by the decoder. 44.1 kHz is a common choice
/// that matches most "modern" MOD players; the Amiga Paula chip ran at
/// 7093789.2 Hz / divider so there's no "native" rate.
pub const OUTPUT_SAMPLE_RATE: u32 = 44_100;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("mod", open);
    reg.register_extension("mod", "mod");
}

fn open(mut input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let mut blob = Vec::new();
    input.read_to_end(&mut blob)?;
    if blob.len() < crate::header::HEADER_FIXED_SIZE {
        return Err(Error::invalid("MOD: file shorter than 1084-byte header"));
    }
    let header = parse_header(&blob)?;

    let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(2); // mixed stereo output
    params.sample_rate = Some(OUTPUT_SAMPLE_RATE);
    params.sample_format = Some(SampleFormat::S16);
    params.extradata = blob.clone();

    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64),
        duration: None, // computed lazily by the decoder
        start_time: Some(0),
        params,
    };

    Ok(Box::new(ModDemuxer {
        streams: vec![stream],
        blob,
        consumed: false,
        _header: header,
    }))
}

struct ModDemuxer {
    streams: Vec<StreamInfo>,
    blob: Vec<u8>,
    consumed: bool,
    _header: crate::header::ModHeader,
}

impl Demuxer for ModDemuxer {
    fn format_name(&self) -> &str {
        "mod"
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
}
