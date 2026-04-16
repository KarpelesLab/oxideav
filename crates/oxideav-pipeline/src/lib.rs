//! Pipeline composition.
//!
//! The main abstraction is [`Pipeline`] which routes source streams to
//! per-stream [`Sink`]s, deciding per-route whether to copy packets
//! verbatim or transcode (decode + optional re-encode). Streams with
//! no bound sink stay inactive — the demuxer is told to skip them.
//!
//! Legacy helpers [`remux`] and [`transcode_simple`] are preserved for
//! backward compatibility.

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_container::{Demuxer, Muxer};
use oxideav_core::{
    CodecParameters, Error, Frame, MediaType, Packet, Result, StreamInfo, TimeBase,
};

// ───────────────────────── Sink trait ─────────────────────────

/// What a [`Sink`] wants for a given source stream.
#[derive(Clone, Debug)]
pub enum SinkAcceptance {
    /// Take raw packets as-is — stream copy, no decoding.
    Copy,
    /// Want decoded frames that match `target` parameters. If the
    /// source already matches the target's core params, the pipeline
    /// degenerates this to a Copy route automatically.
    Transcode { target: CodecParameters },
    /// Not interested in this stream.
    Reject,
}

/// A per-stream output destination.
///
/// Sinks declare what they accept via [`accepts`](Sink::accepts), then
/// receive either raw packets (copy path) or decoded frames (transcode
/// path) depending on how the pipeline resolves the route.
pub trait Sink: Send {
    /// Inspect the source stream's parameters and declare interest.
    fn accepts(&self, src_params: &CodecParameters) -> SinkAcceptance;

    /// Copy path: receive a raw encoded packet.
    fn write_packet(&mut self, pkt: &Packet) -> Result<()>;

    /// Decoded/transcoded path: receive a decoded frame.
    fn write_frame(&mut self, frame: &Frame) -> Result<()>;

    /// Called once after all data has been sent through this sink.
    fn flush(&mut self) -> Result<()>;
}

// ───────────────────────── Pipeline ──────────────────────────

/// Processing mode resolved for one source→sink binding.
enum RouteMode {
    Copy,
    Decode {
        decoder: Box<dyn Decoder>,
    },
    Transcode {
        decoder: Box<dyn Decoder>,
        encoder: Box<dyn Encoder>,
    },
}

struct Route {
    source_stream: u32,
    sink: Box<dyn Sink>,
    mode: RouteMode,
}

/// Multi-stream pipeline: source demuxer → per-stream sinks.
pub struct Pipeline {
    source: Box<dyn Demuxer>,
    routes: Vec<Route>,
    codecs: CodecRegistry,
    prepared: bool,
}

/// Stats returned by [`Pipeline::run`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineStats {
    pub packets_read: u64,
    pub packets_copied: u64,
    pub frames_decoded: u64,
    pub packets_encoded: u64,
    pub packets_dropped: u64,
}

impl Pipeline {
    /// Create a pipeline from a demuxer and codec registry.
    pub fn new(source: Box<dyn Demuxer>, codecs: CodecRegistry) -> Self {
        Self {
            source,
            routes: Vec::new(),
            codecs,
            prepared: false,
        }
    }

    /// Inspect the source container's streams before binding sinks.
    pub fn streams(&self) -> &[StreamInfo] {
        self.source.streams()
    }

    /// Bind a sink to a specific source stream by index.
    pub fn bind(&mut self, stream_index: u32, sink: Box<dyn Sink>) -> Result<()> {
        if self.prepared {
            return Err(Error::other("pipeline already prepared"));
        }
        let stream = self
            .source
            .streams()
            .iter()
            .find(|s| s.index == stream_index)
            .ok_or_else(|| Error::invalid(format!("no stream with index {stream_index}")))?;
        let acceptance = sink.accepts(&stream.params);
        let mode = self.resolve_mode(&stream.params, &acceptance)?;
        self.routes.push(Route {
            source_stream: stream_index,
            sink,
            mode,
        });
        Ok(())
    }

    /// Bind a sink to the first source stream matching a media type.
    pub fn bind_first(&mut self, media_type: MediaType, sink: Box<dyn Sink>) -> Result<()> {
        let idx = self
            .source
            .streams()
            .iter()
            .find(|s| s.params.media_type == media_type)
            .map(|s| s.index)
            .ok_or_else(|| Error::invalid(format!("no {media_type:?} stream in source")))?;
        self.bind(idx, sink)
    }

    /// Finalize routing: tell the demuxer which streams are active.
    pub fn prepare(&mut self) -> Result<()> {
        if self.prepared {
            return Ok(());
        }
        let active: Vec<u32> = self.routes.iter().map(|r| r.source_stream).collect();
        self.source.set_active_streams(&active);
        self.prepared = true;
        Ok(())
    }

    /// Run the pipeline to completion.
    pub fn run(&mut self) -> Result<PipelineStats> {
        if !self.prepared {
            self.prepare()?;
        }
        let mut stats = PipelineStats::default();
        loop {
            let pkt = match self.source.next_packet() {
                Ok(p) => p,
                Err(Error::Eof) => break,
                Err(e) => return Err(e),
            };
            stats.packets_read += 1;
            let route = match self
                .routes
                .iter_mut()
                .find(|r| r.source_stream == pkt.stream_index)
            {
                Some(r) => r,
                None => {
                    stats.packets_dropped += 1;
                    continue;
                }
            };
            match &mut route.mode {
                RouteMode::Copy => {
                    route.sink.write_packet(&pkt)?;
                    stats.packets_copied += 1;
                }
                RouteMode::Decode { decoder } => {
                    decoder.send_packet(&pkt)?;
                    drain_decoder_to_sink(&mut **decoder, &mut *route.sink, &mut stats)?;
                }
                RouteMode::Transcode { decoder, encoder } => {
                    decoder.send_packet(&pkt)?;
                    drain_transcode_to_sink(
                        &mut **decoder,
                        &mut **encoder,
                        &mut *route.sink,
                        &mut stats,
                    )?;
                }
            }
        }
        // Flush all routes.
        for route in &mut self.routes {
            match &mut route.mode {
                RouteMode::Copy => {}
                RouteMode::Decode { decoder } => {
                    decoder.flush()?;
                    drain_decoder_to_sink(&mut **decoder, &mut *route.sink, &mut stats)?;
                }
                RouteMode::Transcode { decoder, encoder } => {
                    decoder.flush()?;
                    drain_transcode_to_sink(
                        &mut **decoder,
                        &mut **encoder,
                        &mut *route.sink,
                        &mut stats,
                    )?;
                    encoder.flush()?;
                    drain_encoder_to_sink(&mut **encoder, &mut *route.sink, &mut stats)?;
                }
            }
            route.sink.flush()?;
        }
        Ok(stats)
    }

    fn resolve_mode(
        &self,
        src_params: &CodecParameters,
        acceptance: &SinkAcceptance,
    ) -> Result<RouteMode> {
        match acceptance {
            SinkAcceptance::Reject => Err(Error::other("sink rejected stream")),
            SinkAcceptance::Copy => Ok(RouteMode::Copy),
            SinkAcceptance::Transcode { target } => {
                if src_params.matches_core(target) {
                    Ok(RouteMode::Copy)
                } else if target.codec_id == src_params.codec_id
                    && target.media_type == src_params.media_type
                {
                    // Same codec but different params — decode only, let sink
                    // handle the format difference (e.g. resample).
                    let decoder = self.codecs.make_decoder(src_params)?;
                    Ok(RouteMode::Decode { decoder })
                } else {
                    let decoder = self.codecs.make_decoder(src_params)?;
                    let encoder = self.codecs.make_encoder(target)?;
                    Ok(RouteMode::Transcode { decoder, encoder })
                }
            }
        }
    }
}

fn drain_decoder_to_sink(
    decoder: &mut dyn Decoder,
    sink: &mut dyn Sink,
    stats: &mut PipelineStats,
) -> Result<()> {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                stats.frames_decoded += 1;
                sink.write_frame(&frame)?;
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

fn drain_transcode_to_sink(
    decoder: &mut dyn Decoder,
    encoder: &mut dyn Encoder,
    sink: &mut dyn Sink,
    stats: &mut PipelineStats,
) -> Result<()> {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                stats.frames_decoded += 1;
                encoder.send_frame(&frame)?;
                drain_encoder_to_sink(encoder, sink, stats)?;
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

fn drain_encoder_to_sink(
    encoder: &mut dyn Encoder,
    sink: &mut dyn Sink,
    stats: &mut PipelineStats,
) -> Result<()> {
    loop {
        match encoder.receive_packet() {
            Ok(pkt) => {
                stats.packets_encoded += 1;
                sink.write_packet(&pkt)?;
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

// ─────────────────── Legacy helpers (preserved) ──────────────

/// Copy all packets from `demuxer` into `muxer` without re-encoding.
pub fn remux(demuxer: &mut dyn Demuxer, muxer: &mut dyn Muxer) -> Result<u64> {
    muxer.write_header()?;
    let mut packets = 0u64;
    loop {
        match demuxer.next_packet() {
            Ok(pkt) => {
                muxer.write_packet(&pkt)?;
                packets += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }
    muxer.write_trailer()?;
    Ok(packets)
}

/// Plan describing how to derive the output stream from the input stream.
#[derive(Clone, Debug)]
pub enum StreamPlan {
    /// Stream-copy (codec passthrough, no decode).
    Copy,
    /// Decode then re-encode using the named codec. The encoder is constructed
    /// with parameters carried over from the input (sample rate, channels…)
    /// and the chosen codec id.
    Reencode { output_codec: String },
}

/// Single-input single-output transcode (or copy).
///
/// `make_output_streams` lets the caller customise the output StreamInfo
/// (e.g. choose container-specific time bases) before the muxer is opened.
/// For the common case where the output stream layout matches the encoder's
/// declared parameters, just clone & adapt.
pub fn transcode_simple(
    demuxer: &mut dyn Demuxer,
    muxer_open: impl FnOnce(&[StreamInfo]) -> Result<Box<dyn Muxer>>,
    codecs: &CodecRegistry,
    plan: &StreamPlan,
) -> Result<TranscodeStats> {
    let in_streams = demuxer.streams().to_vec();
    if in_streams.len() != 1 {
        return Err(Error::unsupported(
            "transcode_simple only handles single-stream inputs today",
        ));
    }
    let in_stream = &in_streams[0];

    match plan {
        StreamPlan::Copy => {
            let mut muxer = muxer_open(&in_streams)?;
            let n = remux(demuxer, &mut *muxer)?;
            Ok(TranscodeStats {
                packets_in: n,
                packets_out: n,
                frames_decoded: 0,
            })
        }
        StreamPlan::Reencode { output_codec } => {
            // Build a decoder from input parameters.
            let mut decoder = codecs.make_decoder(&in_stream.params)?;

            // Build encoder parameters from the input stream's audio
            // properties + the requested codec id.
            let mut enc_params = CodecParameters::audio(output_codec.as_str().into());
            enc_params.media_type = MediaType::Audio;
            enc_params.sample_rate = in_stream.params.sample_rate;
            enc_params.channels = in_stream.params.channels;
            enc_params.sample_format = in_stream.params.sample_format;
            let mut encoder = codecs.make_encoder(&enc_params)?;
            let out_params = encoder.output_params().clone();

            let out_time_base = match out_params.sample_rate {
                Some(sr) if sr > 0 => TimeBase::new(1, sr as i64),
                _ => in_stream.time_base,
            };
            let out_stream = StreamInfo {
                index: 0,
                time_base: out_time_base,
                duration: in_stream.duration,
                start_time: Some(0),
                params: out_params,
            };
            let mut muxer = muxer_open(std::slice::from_ref(&out_stream))?;
            muxer.write_header()?;

            let mut stats = TranscodeStats::default();

            // Drive the decode→encode loop.
            'outer: loop {
                match demuxer.next_packet() {
                    Ok(pkt) => {
                        stats.packets_in += 1;
                        decoder.send_packet(&pkt)?;
                        loop {
                            match decoder.receive_frame() {
                                Ok(frame) => {
                                    stats.frames_decoded += 1;
                                    let frame = adapt_frame_for_encoder(frame, &out_stream)?;
                                    encoder.send_frame(&frame)?;
                                    drain_encoder(&mut *encoder, &mut *muxer, &mut stats)?;
                                }
                                Err(Error::NeedMore) => break,
                                Err(Error::Eof) => break 'outer,
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    Err(Error::Eof) => {
                        decoder.flush()?;
                        // Drain remaining frames.
                        loop {
                            match decoder.receive_frame() {
                                Ok(frame) => {
                                    stats.frames_decoded += 1;
                                    let frame = adapt_frame_for_encoder(frame, &out_stream)?;
                                    encoder.send_frame(&frame)?;
                                    drain_encoder(&mut *encoder, &mut *muxer, &mut stats)?;
                                }
                                Err(Error::NeedMore) | Err(Error::Eof) => break,
                                Err(e) => return Err(e),
                            }
                        }
                        encoder.flush()?;
                        drain_encoder(&mut *encoder, &mut *muxer, &mut stats)?;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }

            muxer.write_trailer()?;
            Ok(stats)
        }
    }
}

fn drain_encoder(
    encoder: &mut dyn oxideav_codec::Encoder,
    muxer: &mut dyn Muxer,
    stats: &mut TranscodeStats,
) -> Result<()> {
    loop {
        match encoder.receive_packet() {
            Ok(pkt) => {
                muxer.write_packet(&pkt)?;
                stats.packets_out += 1;
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

/// If the decoder's output frame doesn't quite match the encoder's input
/// expectations (e.g. time_base differs by container choice), adapt minimal
/// fields here. Sample-format conversion is **not** performed yet — the
/// caller must wire compatible decoder and encoder formats.
fn adapt_frame_for_encoder(frame: Frame, out_stream: &StreamInfo) -> Result<Frame> {
    let time_base = out_stream.time_base;
    Ok(match frame {
        Frame::Audio(mut a) => {
            a.time_base = time_base;
            Frame::Audio(a)
        }
        Frame::Video(mut v) => {
            v.time_base = time_base;
            Frame::Video(v)
        }
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TranscodeStats {
    pub packets_in: u64,
    pub packets_out: u64,
    pub frames_decoded: u64,
}

// Compatibility re-export: keep the old `transcode` symbol for now (returns Unsupported).
#[deprecated(note = "use transcode_simple instead")]
pub fn transcode(
    _demuxer: &mut dyn Demuxer,
    _muxer: &mut dyn Muxer,
    _codecs: &CodecRegistry,
    _plans: &[StreamPlan],
) -> Result<()> {
    Err(Error::unsupported("use transcode_simple"))
}
