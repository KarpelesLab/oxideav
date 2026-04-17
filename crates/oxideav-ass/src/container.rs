//! Container for the standalone ASS/SSA subtitle format.
//!
//! The demuxer reads the file into memory, parses it, and queues one
//! [`Packet`] per cue. The muxer accumulates incoming packets and emits
//! the full file in `write_trailer`.
//!
//! The on-wire payload per packet is a `Dialogue:` line. The codec
//! parameters' `extradata` carries the `[Script Info]` + `[V4+ Styles]` +
//! `[Events]` lead-in.

use std::collections::VecDeque;
use std::io::{Read, SeekFrom, Write};

use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, StreamInfo, TimeBase,
};

use oxideav_subtitle::ir::SubtitleTrack;

pub use crate::codec::ASS_CODEC_ID;

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("ass", open_ass);
    reg.register_muxer("ass", mux_ass);
    reg.register_extension("ass", "ass");
    reg.register_extension("ssa", "ass");
    reg.register_probe("ass", probe_ass);
}

fn probe_ass(p: &ProbeData) -> u8 {
    if super::looks_like_ass(p.buf) {
        100
    } else {
        0
    }
}

fn read_all(mut input: Box<dyn ReadSeek>) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    input.seek(SeekFrom::Start(0))?;
    input.read_to_end(&mut buf)?;
    drop(input);
    Ok(buf)
}

fn open_ass(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let buf = read_all(input)?;
    let track = super::parse(&buf)?;
    Ok(Box::new(AssDemuxer::new(track)))
}

fn mux_ass(out: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(AssMuxer::new(out, streams)?))
}

// ---------------------------------------------------------------------------
// Demuxer — one packet per cue.

struct AssDemuxer {
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl AssDemuxer {
    fn new(track: SubtitleTrack) -> Self {
        let time_base = TimeBase::new(1, 1_000_000); // microseconds
        let mut params = CodecParameters::audio(CodecId::new(ASS_CODEC_ID)); // fields reset below
        params.media_type = MediaType::Subtitle;
        params.sample_rate = None;
        params.channels = None;
        params.sample_format = None;
        params.extradata = track.extradata.clone();

        let total_us: i64 = track.cues.last().map(|c| c.end_us).unwrap_or(0);

        let mut packets: VecDeque<Packet> = VecDeque::with_capacity(track.cues.len());
        for cue in &track.cues {
            let payload = super::cue_to_bytes(cue);
            let mut pkt = Packet::new(0, time_base, payload);
            pkt.pts = Some(cue.start_us);
            pkt.dts = Some(cue.start_us);
            pkt.duration = Some((cue.end_us - cue.start_us).max(0));
            pkt.flags.keyframe = true;
            packets.push_back(pkt);
        }

        let stream = StreamInfo {
            index: 0,
            time_base,
            duration: Some(total_us),
            start_time: Some(0),
            params,
        };

        Self {
            streams: [stream],
            packets,
        }
    }
}

impl Demuxer for AssDemuxer {
    fn format_name(&self) -> &str {
        "ass"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }

    fn duration_micros(&self) -> Option<i64> {
        self.streams[0].duration
    }
}

// ---------------------------------------------------------------------------
// Muxer — buffer cues, reassemble file in write_trailer.

struct AssMuxer {
    out: Box<dyn WriteSeek>,
    extradata: Vec<u8>,
    buffered: Vec<Packet>,
    header_written: bool,
}

impl AssMuxer {
    fn new(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Self> {
        if streams.len() != 1 {
            return Err(Error::invalid(
                "ass muxer: exactly one subtitle stream required",
            ));
        }
        let s = &streams[0];
        let id = s.params.codec_id.as_str();
        if id != ASS_CODEC_ID {
            return Err(Error::invalid(format!(
                "ass muxer: expected codec `{}`, got `{}`",
                ASS_CODEC_ID, id
            )));
        }
        Ok(Self {
            out: output,
            extradata: s.params.extradata.clone(),
            buffered: Vec::new(),
            header_written: false,
        })
    }
}

impl Muxer for AssMuxer {
    fn format_name(&self) -> &str {
        "ass"
    }

    fn write_header(&mut self) -> Result<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid("ass muxer: write_header not called"));
        }
        self.buffered.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let track = rebuild_track(&self.extradata, &self.buffered)?;
        let bytes = super::write(&track);
        self.out.write_all(&bytes)?;
        Ok(())
    }
}

fn rebuild_track(extradata: &[u8], packets: &[Packet]) -> Result<SubtitleTrack> {
    let mut track = SubtitleTrack {
        extradata: extradata.to_vec(),
        ..SubtitleTrack::default()
    };
    // Seed styles / script-info from the extradata so the writer can
    // re-emit them. We do this by parsing the extradata as a truncated
    // file of this format — works because extradata carries the exact
    // header the demuxer saw.
    if !extradata.is_empty() {
        if let Ok(p) = super::parse(extradata) {
            track.styles = p.styles;
            track.metadata = p.metadata;
        }
    }
    for pkt in packets {
        let cue = super::bytes_to_cue(&pkt.data)?;
        track.cues.push(cue);
    }
    Ok(track)
}
