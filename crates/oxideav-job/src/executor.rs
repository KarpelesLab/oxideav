//! DAG executor. Two modes share the same DAG-resolution + track-runtime
//! building code:
//!
//! - **Serial mode** (`threads == 1`): one packet pulled, one packet routed,
//!   one frame at a time. The original path — minimal, predictable, easy
//!   to reason about.
//! - **Pipelined mode** (`threads ≥ 2`, the default when
//!   `available_parallelism()` reports ≥ 2 cores): one thread per stage
//!   per track, bounded mpsc channels in between, mux on the caller
//!   thread so sinks don't need to be `Send`. See [`crate::pipeline`].
//!
//! Outputs within one job still run sequentially — multi-output
//! parallelism is a deliberate follow-up.

use std::collections::HashMap;
use std::path::PathBuf;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_container::{ContainerRegistry, Demuxer, ReadSeek};
use oxideav_core::{
    CodecId, CodecParameters, Error, ExecutionContext, Frame, MediaType, Packet, PixelFormat,
    Result, StreamInfo, TimeBase,
};
use oxideav_pixfmt::{convert as pixfmt_convert, ConvertOptions};
use oxideav_source::SourceRegistry;

use crate::dag::{codec_accepted_pixel_formats, Dag, DagNode, MuxTrack, ResolvedSelector};
use crate::pipeline;
use crate::schema::{is_reserved_sink, Job};
use crate::sinks::{open_file_write, FileSink, NullSink};

/// A user-installable output sink. Implementations receive either raw
/// packets (copy path) or decoded frames (transcode path without an
/// encoder node, e.g. live-play).
///
/// The sink is not required to be `Send` — the executor is single-threaded
/// today. Any future parallel scheduler will have to revisit this.
pub trait JobSink {
    /// Called once after all encoders are constructed and the output
    /// stream layout is known. Muxer-style sinks usually write the
    /// container header here.
    fn start(&mut self, streams: &[StreamInfo]) -> Result<()>;
    fn write_packet(&mut self, kind: MediaType, pkt: &Packet) -> Result<()>;
    fn write_frame(&mut self, kind: MediaType, frm: &Frame) -> Result<()>;
    /// Drain any remaining internal state and finalise the output.
    fn finish(&mut self) -> Result<()>;
}

/// Job runner. Constructed with a validated `Job` + registries; dispatches
/// to serial or pipelined execution based on the effective thread budget.
pub struct Executor<'a> {
    job: &'a Job,
    codecs: &'a CodecRegistry,
    containers: &'a ContainerRegistry,
    sources: &'a SourceRegistry,
    sink_overrides: HashMap<String, Box<dyn JobSink>>,
    /// Explicit thread budget from the caller. `None` = resolve from
    /// `job.threads` or autodetect. `Some(0)` is treated as auto as well.
    explicit_threads: Option<usize>,
}

impl<'a> Executor<'a> {
    pub fn new(
        job: &'a Job,
        codecs: &'a CodecRegistry,
        containers: &'a ContainerRegistry,
        sources: &'a SourceRegistry,
    ) -> Self {
        Self {
            job,
            codecs,
            containers,
            sources,
            sink_overrides: HashMap::new(),
            explicit_threads: None,
        }
    }

    /// Replace the sink for a named output. Typically used to bind a
    /// live-playback sink to `@display`/`@out`.
    pub fn with_sink_override(mut self, name: &str, sink: Box<dyn JobSink>) -> Self {
        self.sink_overrides.insert(name.to_string(), sink);
        self
    }

    /// Override the thread budget. `0` means "auto" (use the value from the
    /// job's `threads` key, or fall back to `available_parallelism()`).
    /// `1` forces strictly serial execution; `≥ 2` requests pipelined.
    pub fn with_threads(mut self, n: usize) -> Self {
        self.explicit_threads = Some(n);
        self
    }

    /// Validate, resolve, and run the job. Processes outputs in their
    /// document order.
    pub fn run(mut self) -> Result<ExecutorStats> {
        self.job.validate()?;
        let dag = self.job.to_dag()?;
        let threads = self.resolve_threads();
        let mut stats = ExecutorStats::default();
        let names: Vec<String> = dag.roots.keys().cloned().collect();
        for name in names {
            let out_stats = if threads >= 2 {
                self.run_output_pipelined(&dag, &name, threads)?
            } else {
                self.run_output(&dag, &name)?
            };
            stats.merge(&out_stats);
        }
        Ok(stats)
    }

    /// Resolve the effective thread budget. Priority: explicit
    /// `with_threads(n > 0)` > `job.threads` > `available_parallelism()`
    /// > `1`.
    fn resolve_threads(&self) -> usize {
        if let Some(n) = self.explicit_threads {
            if n > 0 {
                return n;
            }
        }
        if let Some(n) = self.job.threads {
            if n > 0 {
                return n;
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    fn run_output(&mut self, dag: &Dag, name: &str) -> Result<ExecutorStats> {
        let root_id = dag.roots[name];
        let tracks: Vec<MuxTrack> = match dag.node(root_id) {
            DagNode::Mux { tracks, .. } => tracks.clone(),
            other => {
                return Err(Error::invalid(format!(
                    "job: output {name}: expected Mux root, got {other:?}"
                )));
            }
        };

        // Walk each track's upstream chain to find the leaf Demuxer + the
        // stack of stages (select/decode/filter/encode) to apply.
        let mut pipelines: Vec<TrackRuntime> = Vec::new();
        for t in &tracks {
            pipelines.push(self.build_track_runtime(dag, t)?);
        }

        // Open every unique demuxer source exactly once.
        let mut dmx_by_uri: HashMap<String, Box<dyn Demuxer>> = HashMap::new();
        for pl in &pipelines {
            if !dmx_by_uri.contains_key(&pl.source_uri) {
                let dmx = self.open_demuxer(&pl.source_uri)?;
                dmx_by_uri.insert(pl.source_uri.clone(), dmx);
            }
        }

        // Resolve each pipeline's stream index now that we can inspect
        // the demuxer's stream list.
        for pl in &mut pipelines {
            let dmx = dmx_by_uri.get(&pl.source_uri).unwrap();
            pl.source_stream = select_stream(dmx.streams(), &pl.selector)?;
            // The pipeline's input params come from the actual demuxer stream.
            let info = dmx
                .streams()
                .iter()
                .find(|s| s.index == pl.source_stream)
                .ok_or_else(|| Error::invalid("selected stream not in demuxer"))?;
            pl.input_params = info.params.clone();
            pl.input_time_base = info.time_base;
        }

        // Auto-insert pixel-format conversion stages now that we know
        // the source stream's pixel format. Runs after the demuxer is
        // open and before codec instantiation.
        for pl in &mut pipelines {
            pl.apply_pixel_format_auto_insert(self.codecs);
        }

        // Instantiate decoders / filters / encoders for each track. The
        // serial path tells codecs to stay single-threaded; the pipelined
        // path passes its own thread budget below.
        let ctx = ExecutionContext::serial();
        for pl in &mut pipelines {
            pl.instantiate(self.codecs, &ctx)?;
        }

        // Build the per-track output stream infos + open (or replace) the sink.
        let out_streams: Vec<StreamInfo> = pipelines
            .iter()
            .enumerate()
            .map(|(i, pl)| StreamInfo {
                index: i as u32,
                time_base: pl.output_time_base(),
                duration: None,
                start_time: Some(0),
                params: pl.output_params().clone(),
            })
            .collect();

        let mut sink = self.open_sink(name, &out_streams)?;
        sink.start(&out_streams)?;

        // Main pump. Read packets from every demuxer in round-robin until
        // all are EOF. Route each packet to every pipeline that consumes it.
        let mut stats = ExecutorStats::default();
        let mut eof: HashMap<String, bool> =
            dmx_by_uri.keys().map(|k| (k.clone(), false)).collect();
        let uris: Vec<String> = dmx_by_uri.keys().cloned().collect();
        while eof.values().any(|e| !e) {
            for uri in &uris {
                if eof[uri] {
                    continue;
                }
                let dmx = dmx_by_uri.get_mut(uri).unwrap();
                let pkt = match dmx.next_packet() {
                    Ok(p) => p,
                    Err(Error::Eof) => {
                        eof.insert(uri.clone(), true);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                stats.packets_read += 1;
                for (track_idx, pl) in pipelines.iter_mut().enumerate() {
                    if pl.source_uri != *uri {
                        continue;
                    }
                    if pkt.stream_index != pl.source_stream {
                        continue;
                    }
                    pl.feed_packet(&pkt, track_idx as u32, sink.as_mut(), &mut stats)?;
                }
            }
        }
        // EOF — drain each pipeline.
        for (track_idx, pl) in pipelines.iter_mut().enumerate() {
            pl.drain(track_idx as u32, sink.as_mut(), &mut stats)?;
        }
        sink.finish()?;
        Ok(stats)
    }

    pub(crate) fn build_track_runtime(&self, dag: &Dag, track: &MuxTrack) -> Result<TrackRuntime> {
        // Walk upstream chain, accumulating stages in reverse (top-down).
        // The chain ends at a Demuxer (leaf). The pixel-format
        // auto-insert pass runs later, once the demuxer is open and
        // the source's `CodecParameters.pixel_format` is known —
        // see `TrackRuntime::apply_pixel_format_auto_insert`.
        let mut stages: Vec<StageSpec> = Vec::new();
        let mut cur = track.upstream;
        let (source_uri, selector) = loop {
            match dag.node(cur) {
                DagNode::Demuxer { source } => {
                    break (source.clone(), ResolvedSelector::any());
                }
                DagNode::Select { upstream, selector } => match dag.node(*upstream) {
                    DagNode::Demuxer { source } => {
                        break (source.clone(), selector.clone());
                    }
                    _ => {
                        return Err(Error::other(
                            "job: nested Select above non-Demuxer is not yet supported",
                        ));
                    }
                },
                DagNode::Decode { upstream } => {
                    stages.push(StageSpec::Decode);
                    cur = *upstream;
                }
                DagNode::Filter {
                    upstream,
                    kind,
                    name,
                    params,
                } => {
                    stages.push(StageSpec::Filter {
                        kind: kind.clone(),
                        name: name.clone(),
                        params: params.clone(),
                    });
                    cur = *upstream;
                }
                DagNode::PixConvert { upstream, target } => {
                    stages.push(StageSpec::Convert { target: *target });
                    cur = *upstream;
                }
                DagNode::Encode {
                    upstream,
                    codec,
                    params,
                } => {
                    stages.push(StageSpec::Encode {
                        codec: codec.clone(),
                        params: params.clone(),
                    });
                    cur = *upstream;
                }
                DagNode::Mux { .. } => {
                    return Err(Error::other(
                        "job: walked into a Mux node while building track runtime",
                    ));
                }
            }
        };
        stages.reverse();
        Ok(TrackRuntime::new(
            source_uri, selector, track.kind, track.copy, stages,
        ))
    }

    pub(crate) fn open_demuxer(&self, uri: &str) -> Result<Box<dyn Demuxer>> {
        let file = self.sources.open(uri)?;
        let mut file: Box<dyn ReadSeek> = file;
        let ext = ext_from_uri(uri);
        let format = self.containers.probe_input(&mut *file, ext.as_deref())?;
        self.containers.open_demuxer(&format, file)
    }

    /// Pipelined counterpart to [`Self::run_output`]. Builds the same
    /// per-track runtimes + demuxers + sink, then hands them off to
    /// [`crate::pipeline::run_pipelined`] which spawns a stage-per-thread
    /// worker graph.
    fn run_output_pipelined(
        &mut self,
        dag: &Dag,
        name: &str,
        threads: usize,
    ) -> Result<ExecutorStats> {
        let root_id = dag.roots[name];
        let tracks: Vec<MuxTrack> = match dag.node(root_id) {
            DagNode::Mux { tracks, .. } => tracks.clone(),
            other => {
                return Err(Error::invalid(format!(
                    "job: output {name}: expected Mux root, got {other:?}"
                )));
            }
        };

        let mut pipelines: Vec<TrackRuntime> = Vec::new();
        for t in &tracks {
            pipelines.push(self.build_track_runtime(dag, t)?);
        }
        let mut dmx_by_uri: HashMap<String, Box<dyn Demuxer>> = HashMap::new();
        for pl in &pipelines {
            if !dmx_by_uri.contains_key(&pl.source_uri) {
                let dmx = self.open_demuxer(&pl.source_uri)?;
                dmx_by_uri.insert(pl.source_uri.clone(), dmx);
            }
        }
        for pl in &mut pipelines {
            let dmx = dmx_by_uri.get(&pl.source_uri).unwrap();
            pl.source_stream = select_stream(dmx.streams(), &pl.selector)?;
            let info = dmx
                .streams()
                .iter()
                .find(|s| s.index == pl.source_stream)
                .ok_or_else(|| Error::invalid("selected stream not in demuxer"))?;
            pl.input_params = info.params.clone();
            pl.input_time_base = info.time_base;
        }
        // Auto-insert pixel-format conversion stages now that we know
        // the source stream's pixel format.
        for pl in &mut pipelines {
            pl.apply_pixel_format_auto_insert(self.codecs);
        }
        let ctx = ExecutionContext::with_threads(threads);
        for pl in &mut pipelines {
            pl.instantiate(self.codecs, &ctx)?;
        }
        let out_streams: Vec<StreamInfo> = pipelines
            .iter()
            .enumerate()
            .map(|(i, pl)| StreamInfo {
                index: i as u32,
                time_base: pl.output_time_base(),
                duration: None,
                start_time: Some(0),
                params: pl.output_params().clone(),
            })
            .collect();
        let sink = self.open_sink(name, &out_streams)?;
        pipeline::run_pipelined(pipelines, dmx_by_uri, sink, out_streams)
    }

    pub(crate) fn open_sink(
        &mut self,
        name: &str,
        out_streams: &[StreamInfo],
    ) -> Result<Box<dyn JobSink>> {
        if let Some(s) = self.sink_overrides.remove(name) {
            return Ok(s);
        }
        if name == "@null" {
            return Ok(Box::new(NullSink::new()));
        }
        if name.starts_with('@') {
            if is_reserved_sink(name) {
                return Err(Error::unsupported(format!(
                    "job: no handler registered for reserved sink {name} (use with_sink_override)"
                )));
            }
            return Err(Error::invalid(format!(
                "job: alias {name} cannot be used as an output sink"
            )));
        }
        // File sink — open a muxer matching the path extension.
        let path = PathBuf::from(name);
        let fout = open_file_write(&path)?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| Error::invalid(format!("job: output {name}: no extension")))?;
        let format = self
            .containers
            .container_for_extension(ext)
            .ok_or_else(|| {
                Error::FormatNotFound(format!("no muxer registered for extension .{ext}"))
            })?
            .to_owned();
        let muxer = self.containers.open_muxer(&format, fout, out_streams)?;
        Ok(Box::new(FileSink::new(path, muxer)))
    }
}

// ───────────────────────── per-track runtime ─────────────────────────

#[derive(Clone, Debug)]
pub(crate) enum StageSpec {
    Decode,
    Filter {
        kind: crate::dag::FilterKind,
        name: String,
        params: serde_json::Value,
    },
    /// Pixel-format conversion stage (video only). Inserted either from
    /// an explicit `DagNode::PixConvert` or by the auto-insert pass in
    /// [`Executor::build_track_runtime`] when a codec declares
    /// `accepted_pixel_formats`.
    Convert {
        target: PixelFormat,
    },
    Encode {
        codec: String,
        params: serde_json::Value,
    },
}

/// One track's execution state: decoder + per-frame stage chain +
/// encoder, plus the resolved source URI + selected stream index. Used
/// by both the serial executor and the pipelined runner in
/// [`crate::pipeline`].
pub(crate) struct TrackRuntime {
    pub(crate) source_uri: String,
    pub(crate) selector: ResolvedSelector,
    pub(crate) source_stream: u32,
    pub(crate) kind: MediaType,
    pub(crate) copy: bool,
    pub(crate) stages: Vec<StageSpec>,
    pub(crate) input_params: CodecParameters,
    pub(crate) input_time_base: TimeBase,
    pub(crate) decoder: Option<Box<dyn Decoder>>,
    /// Per-frame stages in order (filters + pixel-format conversions)
    /// between the decoder and the encoder. The pipelined runner
    /// drains this vector to spawn one worker per stage.
    pub(crate) frame_stages: Vec<FrameStage>,
    pub(crate) encoder: Option<Box<dyn Encoder>>,
    pub(crate) encoder_time_base: Option<TimeBase>,
}

pub(crate) enum RuntimeFilter {
    #[cfg(feature = "audio_filter")]
    Audio(Box<dyn oxideav_audio_filter::AudioFilter>),
    #[allow(dead_code)] // only constructed when audio_filter feature is disabled
    Unsupported(String),
}

/// A per-frame stage in the decoded-frame pipeline. Decoder output
/// flows through these in order to reach the encoder.
pub(crate) enum FrameStage {
    Filter(RuntimeFilter),
    /// Pixel-format conversion — calls
    /// [`oxideav_pixfmt::convert`] on each video frame. Non-video
    /// frames pass through unchanged.
    PixConvert(PixelFormat),
}

impl TrackRuntime {
    fn new(
        source_uri: String,
        selector: ResolvedSelector,
        kind: MediaType,
        copy: bool,
        stages: Vec<StageSpec>,
    ) -> Self {
        Self {
            source_uri,
            selector,
            source_stream: 0,
            kind,
            copy,
            stages,
            input_params: CodecParameters::audio(CodecId::new("")),
            input_time_base: TimeBase::new(1, 1),
            decoder: None,
            frame_stages: Vec::new(),
            encoder: None,
            encoder_time_base: None,
        }
    }

    pub(crate) fn instantiate(
        &mut self,
        codecs: &CodecRegistry,
        ctx: &ExecutionContext,
    ) -> Result<()> {
        // Track running frame format through the stage stack so the encoder
        // can be constructed with a realistic parameter set.
        let mut running = self.input_params.clone();
        for stage in &self.stages {
            match stage {
                StageSpec::Decode => {
                    if self.decoder.is_none() {
                        let mut d = codecs.make_decoder(&self.input_params)?;
                        d.set_execution_context(ctx);
                        self.decoder = Some(d);
                    }
                }
                StageSpec::Filter { kind, name, params } => {
                    let src_rate = running.sample_rate.unwrap_or(48_000);
                    let f = build_filter(kind.clone(), name, params, src_rate)?;
                    self.frame_stages.push(FrameStage::Filter(f));
                    // Filter output params are assumed to match input for now;
                    // resample filters override via explicit rate params we
                    // surface before handing to the encoder below.
                    if let Some(new_rate) = params.get("rate").and_then(|r| r.as_u64()) {
                        running.sample_rate = Some(new_rate as u32);
                    }
                }
                StageSpec::Convert { target } => {
                    self.frame_stages.push(FrameStage::PixConvert(*target));
                    // Propagate the target format into the running
                    // CodecParameters so a downstream encoder sees the
                    // correct `pixel_format`.
                    running.pixel_format = Some(*target);
                }
                StageSpec::Encode { codec, params } => {
                    let mut enc_params = running.clone();
                    enc_params.codec_id = CodecId::new(codec.as_str());
                    // Map a handful of common params directly onto
                    // CodecParameters. Everything else is ignored by the
                    // generic path — codec-specific crates can read extras
                    // via their own param-parsing when we add that layer.
                    if let Some(br) = params.get("bitrate").and_then(|b| b.as_u64()) {
                        enc_params.bit_rate = Some(br);
                    }
                    if let Some(sr) = params.get("sample_rate").and_then(|b| b.as_u64()) {
                        enc_params.sample_rate = Some(sr as u32);
                    }
                    if let Some(ch) = params.get("channels").and_then(|b| b.as_u64()) {
                        enc_params.channels = Some(ch as u16);
                    }
                    if let Some(w) = params.get("width").and_then(|b| b.as_u64()) {
                        enc_params.width = Some(w as u32);
                    }
                    if let Some(h) = params.get("height").and_then(|b| b.as_u64()) {
                        enc_params.height = Some(h as u32);
                    }
                    let mut encoder = codecs.make_encoder(&enc_params)?;
                    encoder.set_execution_context(ctx);
                    let out_params = encoder.output_params().clone();
                    running = out_params.clone();
                    self.encoder_time_base = Some(match out_params.sample_rate {
                        Some(sr) if sr > 0 => TimeBase::new(1, sr as i64),
                        _ => self.input_time_base,
                    });
                    self.encoder = Some(encoder);
                }
            }
        }
        Ok(())
    }

    /// Rewrite `self.stages` to insert `StageSpec::Convert` in front of
    /// any `Encode` whose codec declares a non-empty
    /// `accepted_pixel_formats` set that does not include the currently
    /// running pixel format.
    ///
    /// Must be called after `input_params` has been populated from the
    /// demuxer and before `instantiate`. Audio tracks (where
    /// `input_params.pixel_format` is `None`) are a no-op — the pass
    /// only applies when a concrete source pixel format is available.
    ///
    /// **Caveat:** we assume filters preserve the pixel format
    /// (they do today — only audio filters exist — but this assumption
    /// will have to be revisited when video filters land and can
    /// change the pixel format).
    pub(crate) fn apply_pixel_format_auto_insert(&mut self, codecs: &CodecRegistry) {
        let src_fmt = match self.input_params.pixel_format {
            Some(f) => f,
            None => return,
        };
        let mut rewritten: Vec<StageSpec> = Vec::with_capacity(self.stages.len() + 2);
        let mut running: Option<PixelFormat> = None;
        for stage in std::mem::take(&mut self.stages).into_iter() {
            match &stage {
                StageSpec::Decode => {
                    running = Some(src_fmt);
                    rewritten.push(stage);
                }
                StageSpec::Filter { .. } => {
                    rewritten.push(stage);
                }
                StageSpec::Convert { target } => {
                    running = Some(*target);
                    rewritten.push(stage);
                }
                StageSpec::Encode { codec, .. } => {
                    if let Some(cur_fmt) = running {
                        if let Some(accepted) = codec_accepted_pixel_formats(codecs, codec) {
                            if !accepted.contains(&cur_fmt) {
                                let target = accepted[0];
                                rewritten.push(StageSpec::Convert { target });
                                running = Some(target);
                            }
                        }
                    }
                    rewritten.push(stage);
                }
            }
        }
        self.stages = rewritten;
    }

    pub(crate) fn output_params(&self) -> &CodecParameters {
        if let Some(enc) = &self.encoder {
            enc.output_params()
        } else {
            &self.input_params
        }
    }

    pub(crate) fn output_time_base(&self) -> TimeBase {
        self.encoder_time_base.unwrap_or(self.input_time_base)
    }

    fn feed_packet(
        &mut self,
        pkt: &Packet,
        track_index: u32,
        sink: &mut dyn JobSink,
        stats: &mut ExecutorStats,
    ) -> Result<()> {
        if self.copy {
            // Pure copy: retag stream index + forward.
            let mut out = pkt.clone();
            out.stream_index = track_index;
            sink.write_packet(self.kind, &out)?;
            stats.packets_copied += 1;
            return Ok(());
        }
        let frames = if let Some(dec) = &mut self.decoder {
            dec.send_packet(pkt)?;
            drain_decoder(dec.as_mut(), stats)?
        } else {
            Vec::new()
        };
        for frame in frames {
            self.pump_frame(frame, track_index, sink, stats)?;
        }
        Ok(())
    }

    fn pump_frame(
        &mut self,
        frame: Frame,
        track_index: u32,
        sink: &mut dyn JobSink,
        stats: &mut ExecutorStats,
    ) -> Result<()> {
        let mut frames: Vec<Frame> = vec![frame];
        for stage in &mut self.frame_stages {
            let mut next = Vec::new();
            for f in frames {
                let produced = run_frame_stage(stage, f)?;
                next.extend(produced);
            }
            frames = next;
        }

        if let Some(enc) = &mut self.encoder {
            for frame in frames {
                enc.send_frame(&frame)?;
                loop {
                    match enc.receive_packet() {
                        Ok(mut p) => {
                            p.stream_index = track_index;
                            sink.write_packet(self.kind, &p)?;
                            stats.packets_encoded += 1;
                        }
                        Err(Error::NeedMore) | Err(Error::Eof) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
        } else {
            // Raw frame to sink (player sink consumes this).
            for f in frames {
                sink.write_frame(self.kind, &f)?;
                stats.frames_written += 1;
            }
        }
        Ok(())
    }

    fn drain(
        &mut self,
        track_index: u32,
        sink: &mut dyn JobSink,
        stats: &mut ExecutorStats,
    ) -> Result<()> {
        if self.copy {
            return Ok(());
        }
        let tail_from_decoder = if let Some(dec) = &mut self.decoder {
            dec.flush()?;
            drain_decoder(dec.as_mut(), stats)?
        } else {
            Vec::new()
        };
        for frame in tail_from_decoder {
            self.pump_frame(frame, track_index, sink, stats)?;
        }
        // Flush frame stages. Filters may hold internal buffers
        // (resampler tail, noise-gate decay); pixel-format converts
        // are stateless so they drain trivially. After each stage
        // flushes, push its residual frames through the remaining
        // stages downstream.
        let mut tail: Vec<Frame> = Vec::new();
        for i in 0..self.frame_stages.len() {
            let mut flushed = flush_frame_stage(&mut self.frame_stages[i])?;
            // Push the flushed frames through the remaining downstream
            // stages so a later pixel-format convert still sees them.
            for j in (i + 1)..self.frame_stages.len() {
                let mut next: Vec<Frame> = Vec::new();
                for f in flushed.drain(..) {
                    let produced = run_frame_stage(&mut self.frame_stages[j], f)?;
                    next.extend(produced);
                }
                flushed = next;
            }
            tail.extend(flushed);
        }
        if let Some(enc) = &mut self.encoder {
            for frame in tail {
                enc.send_frame(&frame)?;
                loop {
                    match enc.receive_packet() {
                        Ok(mut p) => {
                            p.stream_index = track_index;
                            sink.write_packet(self.kind, &p)?;
                            stats.packets_encoded += 1;
                        }
                        Err(Error::NeedMore) | Err(Error::Eof) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
            enc.flush()?;
            loop {
                match enc.receive_packet() {
                    Ok(mut p) => {
                        p.stream_index = track_index;
                        sink.write_packet(self.kind, &p)?;
                        stats.packets_encoded += 1;
                    }
                    Err(Error::NeedMore) | Err(Error::Eof) => break,
                    Err(e) => return Err(e),
                }
            }
        } else {
            for f in tail {
                sink.write_frame(self.kind, &f)?;
                stats.frames_written += 1;
            }
        }
        Ok(())
    }
}

pub(crate) fn drain_decoder(
    dec: &mut dyn Decoder,
    stats: &mut ExecutorStats,
) -> Result<Vec<Frame>> {
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(frame) => {
                stats.frames_decoded += 1;
                out.push(frame);
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(out),
            Err(e) => return Err(e),
        }
    }
}

/// Drive one frame through a [`FrameStage`]. Pixel-format converts on
/// a non-video frame pass through unchanged so tracks that mix audio
/// and video (rare today, but cheap to support) don't bounce off the
/// convert stage.
pub(crate) fn run_frame_stage(stage: &mut FrameStage, frame: Frame) -> Result<Vec<Frame>> {
    match stage {
        FrameStage::Filter(f) => run_filter(f, frame),
        FrameStage::PixConvert(target) => match frame {
            Frame::Video(v) => {
                let out = pixfmt_convert(&v, *target, &ConvertOptions::default())?;
                Ok(vec![Frame::Video(out)])
            }
            other => Ok(vec![other]),
        },
    }
}

/// Flush any residual frames held inside a [`FrameStage`]. Audio
/// filters may have tail samples; pixel-format converts are stateless
/// and return nothing.
pub(crate) fn flush_frame_stage(stage: &mut FrameStage) -> Result<Vec<Frame>> {
    match stage {
        FrameStage::Filter(f) => flush_filter(f),
        FrameStage::PixConvert(_) => Ok(Vec::new()),
    }
}

pub(crate) fn run_filter(filter: &mut RuntimeFilter, frame: Frame) -> Result<Vec<Frame>> {
    match filter {
        #[cfg(feature = "audio_filter")]
        RuntimeFilter::Audio(f) => match frame {
            Frame::Audio(a) => {
                let outs = f.process(&a)?;
                Ok(outs.into_iter().map(Frame::Audio).collect())
            }
            _ => Err(Error::invalid(
                "job: audio filter received a non-audio frame",
            )),
        },
        RuntimeFilter::Unsupported(name) => Err(Error::unsupported(format!(
            "job: filter {name} is not supported at execution time"
        ))),
    }
}

pub(crate) fn flush_filter(filter: &mut RuntimeFilter) -> Result<Vec<Frame>> {
    match filter {
        #[cfg(feature = "audio_filter")]
        RuntimeFilter::Audio(f) => {
            let outs = f.flush()?;
            Ok(outs.into_iter().map(Frame::Audio).collect())
        }
        RuntimeFilter::Unsupported(_) => Ok(Vec::new()),
    }
}

fn build_filter(
    kind: crate::dag::FilterKind,
    name: &str,
    params: &serde_json::Value,
    src_rate: u32,
) -> Result<RuntimeFilter> {
    use crate::dag::FilterKind;
    match kind {
        FilterKind::Video => Err(Error::unsupported(format!(
            "job: video filter '{name}' — no video filters are wired in yet"
        ))),
        FilterKind::Audio => {
            #[cfg(feature = "audio_filter")]
            {
                let f = build_audio_filter(name, params, src_rate)?;
                Ok(RuntimeFilter::Audio(f))
            }
            #[cfg(not(feature = "audio_filter"))]
            {
                let _ = (name, params, src_rate);
                Ok(RuntimeFilter::Unsupported(
                    "audio filters disabled at compile time (enable the `audio_filter` feature)"
                        .into(),
                ))
            }
        }
    }
}

#[cfg(feature = "audio_filter")]
fn build_audio_filter(
    name: &str,
    params: &serde_json::Value,
    src_rate: u32,
) -> Result<Box<dyn oxideav_audio_filter::AudioFilter>> {
    use oxideav_audio_filter::{NoiseGate, Resample, Volume};
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    match name {
        "volume" => {
            // Accept either `gain` (linear) or `gain_db`; convert the latter to
            // the linear form the constructor expects.
            if let Some(db) = get_f64("gain_db") {
                let linear = 10f32.powf((db as f32) / 20.0);
                Ok(Box::new(Volume::new(linear)))
            } else if let Some(g) = get_f64("gain") {
                Ok(Box::new(Volume::new(g as f32)))
            } else {
                Err(Error::invalid(
                    "job: filter 'volume' needs `gain` or `gain_db`",
                ))
            }
        }
        "noise_gate" => {
            let threshold_db = get_f64("threshold_db").unwrap_or(-40.0) as f32;
            let attack_ms = get_f64("attack_ms").unwrap_or(10.0) as f32;
            let release_ms = get_f64("release_ms").unwrap_or(100.0) as f32;
            let hold_ms = get_f64("hold_ms").unwrap_or(50.0) as f32;
            Ok(Box::new(NoiseGate::new(
                threshold_db,
                attack_ms,
                release_ms,
                hold_ms,
            )))
        }
        "resample" => {
            let dst_rate = get_u64("rate").ok_or_else(|| {
                Error::invalid("job: filter 'resample' needs `rate` (output sample rate)")
            })?;
            Ok(Box::new(Resample::new(src_rate, dst_rate as u32)?))
        }
        other => Err(Error::unsupported(format!(
            "job: unknown audio filter '{other}'"
        ))),
    }
}

pub(crate) fn select_stream(streams: &[StreamInfo], sel: &ResolvedSelector) -> Result<u32> {
    let filtered: Vec<&StreamInfo> = streams
        .iter()
        .filter(|s| match sel.kind {
            Some(k) => s.params.media_type == k,
            None => true,
        })
        .collect();
    if filtered.is_empty() {
        return Err(Error::invalid(format!(
            "job: no streams match selector {sel:?}"
        )));
    }
    let idx = sel.index.unwrap_or(0) as usize;
    let picked = filtered
        .get(idx)
        .ok_or_else(|| Error::invalid(format!("job: selector index {idx} out of range")))?;
    Ok(picked.index)
}

pub(crate) fn ext_from_uri(uri: &str) -> Option<String> {
    let last = uri.rsplit('/').next().unwrap_or(uri);
    let last = last.split('?').next().unwrap_or(last);
    let dot = last.rfind('.')?;
    Some(last[dot + 1..].to_ascii_lowercase())
}

// ───────────────────────── stats ─────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutorStats {
    pub packets_read: u64,
    pub packets_copied: u64,
    pub packets_encoded: u64,
    pub frames_decoded: u64,
    pub frames_written: u64,
}

impl ExecutorStats {
    pub(crate) fn merge(&mut self, other: &Self) {
        self.packets_read += other.packets_read;
        self.packets_copied += other.packets_copied;
        self.packets_encoded += other.packets_encoded;
        self.frames_decoded += other.frames_decoded;
        self.frames_written += other.frames_written;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_from_uri_basic() {
        assert_eq!(ext_from_uri("foo.mp3").as_deref(), Some("mp3"));
        assert_eq!(
            ext_from_uri("https://x/y.mkv?token=1").as_deref(),
            Some("mkv")
        );
        assert_eq!(ext_from_uri("/no/ext"), None);
    }
}
