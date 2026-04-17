//! Containers for the standalone text subtitle formats.
//!
//! Each demuxer reads the file into memory, parses it via the relevant
//! module (`crate::<fmt>::parse`), and queues one [`Packet`] per cue.
//! Each muxer accumulates incoming packets and emits the full file in
//! `write_trailer` via `crate::<fmt>::write`.
//!
//! The on-wire payload per packet is the cue's natural textual form in
//! that format. The codec parameters' `extradata` carries the
//! file-level header (empty for SRT/MPL2/etc., the `WEBVTT ...`/STYLE
//! blocks for WebVTT, the GSI+styles block for EBU STL, …).
//!
//! Shared-extension disambiguation: MicroDVD / MPsub / SubViewer 1 /
//! SubViewer 2 all claim `.sub`; MicroDVD + VPlayer both claim `.txt`.
//! Each format ships a content-based probe; the container registry
//! dispatches to the highest-scoring one. SubViewer 2's `[INFORMATION]`
//! header scores 95; MicroDVD scores up to 80 on its `{n}{m}` pattern;
//! MPsub scores 85 on `FORMAT=TIME`; SubViewer 1 scores 80 on its
//! `**START SCRIPT**` marker; VPlayer scores 70 on the generic
//! `HH:MM:SS:` line shape.

use std::collections::VecDeque;
use std::io::{Read, SeekFrom, Write};

use oxideav_container::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, MediaType, Packet, Result, StreamInfo, TimeBase,
};

use crate::ir::SubtitleTrack;
use crate::{
    aqtitle, ebu_stl, jacosub, microdvd, mpl2, mpsub, pjs, realtext, sami, srt, subviewer1,
    subviewer2, ttml, vplayer, webvtt,
};

/// Codec ids emitted by this crate's containers (kept for backward
/// compat with callers that expected them here rather than on the
/// individual modules).
pub const SRT_CODEC_ID: &str = "subrip";
pub const WEBVTT_CODEC_ID: &str = "webvtt";

pub fn register(reg: &mut ContainerRegistry) {
    // ---- SRT + WebVTT (kept separate for historical clarity) ----
    reg.register_demuxer("srt", open_srt);
    reg.register_muxer("srt", mux_srt);
    reg.register_extension("srt", "srt");
    reg.register_probe("srt", probe_srt);

    reg.register_demuxer("webvtt", open_webvtt);
    reg.register_muxer("webvtt", mux_webvtt);
    reg.register_extension("vtt", "webvtt");
    reg.register_probe("webvtt", probe_webvtt);

    // ---- Everything else uses the generic per-format dispatch ----
    register_fmt::<MicroDvd>(reg, "microdvd", &["sub", "txt"]);
    register_fmt::<Mpl2>(reg, "mpl2", &["mpl"]);
    register_fmt::<MpSub>(reg, "mpsub", &["sub"]);
    register_fmt::<VPlayer>(reg, "vplayer", &["txt", "vpl"]);
    register_fmt::<Pjs>(reg, "pjs", &["pjs"]);
    register_fmt::<AqTitle>(reg, "aqtitle", &["aqt"]);
    register_fmt::<JacoSub>(reg, "jacosub", &["jss", "js"]);
    register_fmt::<RealText>(reg, "realtext", &["rt"]);
    register_fmt::<SubViewer1>(reg, "subviewer1", &["sub"]);
    register_fmt::<SubViewer2>(reg, "subviewer2", &["sub"]);
    register_fmt::<Ttml>(reg, "ttml", &["ttml", "dfxp", "xml"]);
    register_fmt::<Sami>(reg, "sami", &["smi", "sami"]);
    register_fmt::<EbuStl>(reg, "ebu_stl", &["stl"]);
}

fn register_fmt<F: FormatOps>(reg: &mut ContainerRegistry, name: &'static str, exts: &[&str]) {
    reg.register_demuxer(name, open_fmt::<F>);
    reg.register_muxer(name, mux_fmt::<F>);
    for e in exts {
        reg.register_extension(e, name);
    }
    reg.register_probe(name, probe_fmt::<F>);
}

fn probe_srt(p: &ProbeData) -> u8 {
    if srt::looks_like_srt(p.buf) {
        75
    } else {
        0
    }
}

fn probe_webvtt(p: &ProbeData) -> u8 {
    if webvtt::looks_like_webvtt(p.buf) {
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

fn open_srt(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let buf = read_all(input)?;
    let track = srt::parse(&buf)?;
    Ok(Box::new(TextSubtitleDemuxer::new(
        "srt",
        SRT_CODEC_ID,
        track,
        &|c| srt::cue_to_bytes(c),
    )))
}

fn open_webvtt(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let buf = read_all(input)?;
    let track = webvtt::parse(&buf)?;
    Ok(Box::new(TextSubtitleDemuxer::new(
        "webvtt",
        WEBVTT_CODEC_ID,
        track,
        &|c| webvtt::cue_to_bytes(c),
    )))
}

fn mux_srt(out: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(LegacyTextSubtitleMuxer::new(out, streams, "srt")?))
}

fn mux_webvtt(out: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(LegacyTextSubtitleMuxer::new(
        out, streams, "webvtt",
    )?))
}

// ---------------------------------------------------------------------------
// Generic per-format ops + registration helpers
//
// Every new format provides the same primitive set (parse, write, probe,
// cue_to_bytes, bytes_to_cue, CODEC_ID). `FormatOps` collects them into
// a trait-table we can parameterise register/open/mux/probe with.

trait FormatOps: Send + Sync + 'static {
    const FORMAT_NAME: &'static str;
    const CODEC_ID: &'static str;
    fn parse(buf: &[u8]) -> Result<SubtitleTrack>;
    fn write(track: &SubtitleTrack) -> Result<Vec<u8>>;
    fn probe(buf: &[u8]) -> u8;
    fn cue_to_bytes(cue: &oxideav_core::SubtitleCue) -> Vec<u8>;
    fn bytes_to_cue(bytes: &[u8]) -> Result<oxideav_core::SubtitleCue>;
}

fn open_fmt<F: FormatOps>(input: Box<dyn ReadSeek>) -> Result<Box<dyn Demuxer>> {
    let buf = read_all(input)?;
    let track = F::parse(&buf)?;
    Ok(Box::new(TextSubtitleDemuxer::new(
        F::FORMAT_NAME,
        F::CODEC_ID,
        track,
        &|c| F::cue_to_bytes(c),
    )))
}

fn mux_fmt<F: FormatOps>(
    out: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(GenericTextSubtitleMuxer::<F>::new(out, streams)?))
}

fn probe_fmt<F: FormatOps>(p: &ProbeData) -> u8 {
    F::probe(p.buf)
}

/// Uniform-signature adapter. Each module has `parse` + `probe` with
/// the same shape, but `write` / `cue_to_bytes` / `bytes_to_cue`
/// shapes vary (some return `Vec<u8>` vs `Result<Vec<u8>>`, microdvd
/// takes an extra fps argument). Hand-written impls normalise all of
/// them to the trait shape below.
macro_rules! fmt_ops_simple {
    ($ty:ident, $name:literal, $mod:ident) => {
        struct $ty;
        impl FormatOps for $ty {
            const FORMAT_NAME: &'static str = $name;
            const CODEC_ID: &'static str = $mod::CODEC_ID;
            fn parse(buf: &[u8]) -> Result<SubtitleTrack> {
                $mod::parse(buf)
            }
            fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
                $mod::write(track)
            }
            fn probe(buf: &[u8]) -> u8 {
                $mod::probe(buf)
            }
            fn cue_to_bytes(cue: &oxideav_core::SubtitleCue) -> Vec<u8> {
                $mod::cue_to_bytes(cue)
            }
            fn bytes_to_cue(bytes: &[u8]) -> Result<oxideav_core::SubtitleCue> {
                $mod::bytes_to_cue(bytes)
            }
        }
    };
}

macro_rules! fmt_ops_infallible_write {
    ($ty:ident, $name:literal, $mod:ident) => {
        struct $ty;
        impl FormatOps for $ty {
            const FORMAT_NAME: &'static str = $name;
            const CODEC_ID: &'static str = $mod::CODEC_ID;
            fn parse(buf: &[u8]) -> Result<SubtitleTrack> {
                $mod::parse(buf)
            }
            fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
                Ok($mod::write(track))
            }
            fn probe(buf: &[u8]) -> u8 {
                $mod::probe(buf)
            }
            fn cue_to_bytes(cue: &oxideav_core::SubtitleCue) -> Vec<u8> {
                $mod::cue_to_bytes(cue)
            }
            fn bytes_to_cue(bytes: &[u8]) -> Result<oxideav_core::SubtitleCue> {
                $mod::bytes_to_cue(bytes)
            }
        }
    };
}

// MicroDVD needs an fps argument at the packet boundary; use the 25 fps
// default (matches the MicroDVD legacy convention + what the encoder
// falls back to when `{1}{1}<fps>` isn't present).
struct MicroDvd;
impl FormatOps for MicroDvd {
    const FORMAT_NAME: &'static str = "microdvd";
    const CODEC_ID: &'static str = microdvd::CODEC_ID;
    fn parse(buf: &[u8]) -> Result<SubtitleTrack> {
        microdvd::parse(buf)
    }
    fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
        microdvd::write(track)
    }
    fn probe(buf: &[u8]) -> u8 {
        microdvd::probe(buf)
    }
    fn cue_to_bytes(cue: &oxideav_core::SubtitleCue) -> Vec<u8> {
        microdvd::cue_to_bytes(cue, 25.0)
    }
    fn bytes_to_cue(bytes: &[u8]) -> Result<oxideav_core::SubtitleCue> {
        microdvd::bytes_to_cue(bytes, 25.0)
    }
}

fmt_ops_simple!(Mpl2, "mpl2", mpl2);
fmt_ops_simple!(MpSub, "mpsub", mpsub);
fmt_ops_simple!(VPlayer, "vplayer", vplayer);
fmt_ops_simple!(Pjs, "pjs", pjs);
fmt_ops_simple!(AqTitle, "aqtitle", aqtitle);
fmt_ops_simple!(JacoSub, "jacosub", jacosub);
fmt_ops_simple!(RealText, "realtext", realtext);
fmt_ops_simple!(SubViewer1, "subviewer1", subviewer1);
fmt_ops_simple!(SubViewer2, "subviewer2", subviewer2);
fmt_ops_simple!(EbuStl, "ebu_stl", ebu_stl);
fmt_ops_infallible_write!(Ttml, "ttml", ttml);
fmt_ops_infallible_write!(Sami, "sami", sami);

// ---------------------------------------------------------------------------
// Shared text subtitle demuxer — one packet per cue, closure-based cue
// serialisation so the demuxer works for every format without a dispatch
// branch per format.

struct TextSubtitleDemuxer {
    format_name: &'static str,
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl TextSubtitleDemuxer {
    fn new(
        format_name: &'static str,
        codec_id: &'static str,
        track: SubtitleTrack,
        to_bytes: &dyn Fn(&oxideav_core::SubtitleCue) -> Vec<u8>,
    ) -> Self {
        let time_base = TimeBase::new(1, 1_000_000); // microseconds
        let mut params = CodecParameters::audio(CodecId::new(codec_id));
        params.media_type = MediaType::Subtitle;
        params.sample_rate = None;
        params.channels = None;
        params.sample_format = None;
        params.extradata = track.extradata.clone();

        let total_us: i64 = track.cues.last().map(|c| c.end_us).unwrap_or(0);

        let mut packets: VecDeque<Packet> = VecDeque::with_capacity(track.cues.len());
        for cue in &track.cues {
            let payload = to_bytes(cue);
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
            format_name,
            streams: [stream],
            packets,
        }
    }
}

impl Demuxer for TextSubtitleDemuxer {
    fn format_name(&self) -> &str {
        self.format_name
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
// Generic muxer — collects packets, re-parses each into a cue via
// FormatOps::bytes_to_cue, and writes the whole track via FormatOps::write
// on write_trailer.

struct GenericTextSubtitleMuxer<F: FormatOps> {
    out: Box<dyn WriteSeek>,
    extradata: Vec<u8>,
    buffered: Vec<Packet>,
    header_written: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: FormatOps> GenericTextSubtitleMuxer<F> {
    fn new(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Self> {
        if streams.len() != 1 {
            return Err(Error::invalid(format!(
                "{} muxer: exactly one subtitle stream required",
                F::FORMAT_NAME
            )));
        }
        let s = &streams[0];
        let id = s.params.codec_id.as_str();
        if id != F::CODEC_ID {
            return Err(Error::invalid(format!(
                "{} muxer: expected codec `{}`, got `{}`",
                F::FORMAT_NAME,
                F::CODEC_ID,
                id
            )));
        }
        Ok(Self {
            out: output,
            extradata: s.params.extradata.clone(),
            buffered: Vec::new(),
            header_written: false,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: FormatOps> Muxer for GenericTextSubtitleMuxer<F> {
    fn format_name(&self) -> &str {
        F::FORMAT_NAME
    }

    fn write_header(&mut self) -> Result<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid("subtitle muxer: write_header not called"));
        }
        self.buffered.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let mut track = SubtitleTrack {
            extradata: self.extradata.clone(),
            ..SubtitleTrack::default()
        };
        // Seed track header state from extradata where meaningful.
        // Formats that need this (WebVTT styles, STL GSI, TTML styles)
        // re-parse the extradata as a truncated file. Ignore parse
        // errors — an empty seed is preferable to refusing to mux.
        if !self.extradata.is_empty() {
            if let Ok(parsed) = F::parse(&self.extradata) {
                track.styles = parsed.styles;
                track.metadata = parsed.metadata;
            }
        }
        for pkt in &self.buffered {
            let cue = F::bytes_to_cue(&pkt.data)?;
            track.cues.push(cue);
        }
        let bytes = F::write(&track)?;
        self.out.write_all(&bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Legacy muxer wrapping srt / webvtt (kept for the existing public path
// and docstrings that reference it).

struct LegacyTextSubtitleMuxer {
    out: Box<dyn WriteSeek>,
    format: &'static str,
    extradata: Vec<u8>,
    buffered: Vec<Packet>,
    header_written: bool,
}

impl LegacyTextSubtitleMuxer {
    fn new(
        output: Box<dyn WriteSeek>,
        streams: &[StreamInfo],
        format: &'static str,
    ) -> Result<Self> {
        if streams.len() != 1 {
            return Err(Error::invalid(format!(
                "{format} muxer: exactly one subtitle stream required"
            )));
        }
        let s = &streams[0];
        let id = s.params.codec_id.as_str();
        let expected = match format {
            "srt" => SRT_CODEC_ID,
            "webvtt" => WEBVTT_CODEC_ID,
            _ => "",
        };
        if id != expected {
            return Err(Error::invalid(format!(
                "{format} muxer: expected codec `{expected}`, got `{id}`"
            )));
        }
        Ok(Self {
            out: output,
            format,
            extradata: s.params.extradata.clone(),
            buffered: Vec::new(),
            header_written: false,
        })
    }
}

impl Muxer for LegacyTextSubtitleMuxer {
    fn format_name(&self) -> &str {
        self.format
    }

    fn write_header(&mut self) -> Result<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid("subtitle muxer: write_header not called"));
        }
        self.buffered.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let mut track = SubtitleTrack {
            extradata: self.extradata.clone(),
            ..SubtitleTrack::default()
        };
        if !self.extradata.is_empty() {
            if self.format == "webvtt" {
                if let Ok(parsed) = webvtt::parse(&self.extradata) {
                    track.styles = parsed.styles;
                    track.metadata = parsed.metadata;
                }
            }
        }
        for pkt in &self.buffered {
            let cue = match self.format {
                "srt" => srt::bytes_to_cue(&pkt.data)?,
                "webvtt" => webvtt::bytes_to_cue(&pkt.data)?,
                other => return Err(Error::invalid(format!("unknown subtitle format {other}"))),
            };
            track.cues.push(cue);
        }
        let bytes = match self.format {
            "srt" => srt::write(&track),
            "webvtt" => webvtt::write(&track),
            other => return Err(Error::invalid(format!("unknown subtitle format {other}"))),
        };
        self.out.write_all(&bytes)?;
        Ok(())
    }
}
