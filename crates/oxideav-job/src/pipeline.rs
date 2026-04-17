//! Pipelined (stage-per-thread) executor.
//!
//! Called by [`Executor::run`] when the thread budget is `≥ 2`. Spawns
//! one worker thread per pipeline stage per track, connected by bounded
//! `mpsc::sync_channel`s, and drives the mux/sink loop on the caller's
//! thread. Sinks therefore don't need to be `Send`.
//!
//! Data flow per output:
//!
//! ```text
//!   [one dmx thread per URI] ──► per-track packet channel ─┐
//!                                                           ├─► decode ─► filter… ─► encode ─► output channel
//!                                                           ┴─► (copy mode: output channel directly)
//!
//!   main thread (mux loop): recv across all output channels → sink.write_packet
//! ```
//!
//! End-of-stream is signalled with [`Msg::Eof`] rather than by dropping
//! the sender, so downstream stages can reliably flush their internal
//! buffers before exiting. Errors in any stage are funnelled through
//! [`AbortState`]; the first error wins, other stages bail cleanly.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use oxideav_codec::{Decoder, Encoder};
use oxideav_container::Demuxer;
use oxideav_core::{Error, Frame, MediaType, Packet, Result, StreamInfo};

use crate::executor::{
    drain_decoder, flush_frame_stage, run_frame_stage, ExecutorStats, FrameStage, JobSink,
    TrackRuntime,
};

/// Packet-channel depth. Small enough that a stalled consumer back-pressures
/// the demuxer before memory blows up; large enough to amortise the mutex
/// cost on each send.
const PACKET_CAP: usize = 16;

/// Frame-channel depth. Smaller than `PACKET_CAP` because decoded frames
/// are much larger than compressed packets.
const FRAME_CAP: usize = 8;

/// Messages across channels. `Eof` is an in-band signal so workers can
/// flush their state before exiting.
enum Msg<T> {
    Data(T),
    Eof,
}

/// Shared counters. Each worker increments its relevant field; the mux
/// thread reads them out at the end into [`ExecutorStats`].
#[derive(Default)]
struct PipelineCounters {
    packets_read: AtomicU64,
    packets_copied: AtomicU64,
    packets_encoded: AtomicU64,
    frames_decoded: AtomicU64,
    frames_written: AtomicU64,
}

impl PipelineCounters {
    fn snapshot(&self) -> ExecutorStats {
        ExecutorStats {
            packets_read: self.packets_read.load(Ordering::SeqCst),
            packets_copied: self.packets_copied.load(Ordering::SeqCst),
            packets_encoded: self.packets_encoded.load(Ordering::SeqCst),
            frames_decoded: self.frames_decoded.load(Ordering::SeqCst),
            frames_written: self.frames_written.load(Ordering::SeqCst),
        }
    }
}

/// Shared state used to coordinate clean shutdown across all worker
/// threads in one output's pipeline.
struct AbortState {
    /// Set by any worker that errors out (or by the mux thread at EOF).
    /// Workers poll it between iterations and bail cleanly.
    abort: AtomicBool,
    /// First `Err(_)` seen. Later errors are dropped so the caller
    /// gets the root cause rather than a cascading symptom.
    first_err: Mutex<Option<Error>>,
}

impl AbortState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            abort: AtomicBool::new(false),
            first_err: Mutex::new(None),
        })
    }

    fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    fn record_error(&self, e: Error) {
        let mut slot = self.first_err.lock().unwrap();
        if slot.is_none() {
            *slot = Some(e);
        }
        self.abort.store(true, Ordering::SeqCst);
    }

    fn take_error(&self) -> Option<Error> {
        self.first_err.lock().unwrap().take()
    }
}

/// One per-track output channel item — retains the track index so the
/// mux thread can tag packets with the right stream index.
struct OutputItem {
    track_index: u32,
    kind: MediaType,
    payload: OutputPayload,
}

enum OutputPayload {
    Packet(Packet),
    Frame(Frame),
}

/// Run one output's pipeline. The caller has already instantiated all
/// decoders/filters/encoders via `TrackRuntime::instantiate`, opened the
/// demuxers, and prepared the sink (but not called `start` on it).
pub(crate) fn run_pipelined(
    mut pipelines: Vec<TrackRuntime>,
    dmx_by_uri: HashMap<String, Box<dyn Demuxer>>,
    mut sink: Box<dyn JobSink>,
    out_streams: Vec<StreamInfo>,
) -> Result<ExecutorStats> {
    sink.start(&out_streams)?;

    let abort = AbortState::new();
    let counters = Arc::new(PipelineCounters::default());
    let mut handles: Vec<JoinHandle<()>> = Vec::new();

    // Per-track output channel: stage workers send processed packets /
    // frames on tx; the mux loop on the caller thread reads rx.
    let mut track_output_rx: Vec<Receiver<Msg<OutputItem>>> = Vec::new();
    let mut track_output_tx: Vec<SyncSender<Msg<OutputItem>>> = Vec::new();
    for _ in 0..pipelines.len() {
        let (tx, rx) = mpsc::sync_channel::<Msg<OutputItem>>(PACKET_CAP);
        track_output_tx.push(tx);
        track_output_rx.push(rx);
    }

    // Route table: per source URI, the list of (source_stream, packet_tx)
    // pairs the demuxer thread fans packets out to.
    let mut routes_by_uri: HashMap<String, Vec<(u32, SyncSender<Msg<Packet>>)>> = HashMap::new();

    // Build + spawn each track's stage chain. We consume the Vec so the
    // decoder/encoder/filters can be moved into worker threads.
    for (track_idx, mut pl) in pipelines.drain(..).enumerate() {
        let out_tx = track_output_tx[track_idx].clone();
        let kind = pl.kind;
        let source_uri = pl.source_uri.clone();
        let source_stream = pl.source_stream;

        // Every track has a packet-input channel from the demuxer
        // regardless of copy / transcode — the demuxer thread doesn't
        // need to know which mode each consumer uses.
        let (pkt_tx, pkt_rx) = mpsc::sync_channel::<Msg<Packet>>(PACKET_CAP);
        routes_by_uri
            .entry(source_uri)
            .or_default()
            .push((source_stream, pkt_tx));

        if pl.copy {
            let abort_c = abort.clone();
            let counters_c = counters.clone();
            let name = format!("copy-{track_idx}");
            handles.push(spawn_stage(abort_c, name, move |abort| {
                run_copy_stage(pkt_rx, out_tx, track_idx as u32, kind, abort, counters_c)
            }));
            continue;
        }

        // Transcode: decoder → frame stages → encoder-or-fanout.
        // Each FrameStage runs on its own worker thread so audio
        // filters, pixel-format converts, and future video filters
        // can overlap the encoder's back-pressure.
        let decoder = pl.decoder.take().ok_or_else(|| {
            Error::other("pipeline: non-copy track without a decoder is not supported")
        })?;
        let frame_stages = std::mem::take(&mut pl.frame_stages);
        let encoder = pl.encoder.take();

        let (frame0_tx, frame0_rx) = mpsc::sync_channel::<Msg<Frame>>(FRAME_CAP);
        {
            let abort_d = abort.clone();
            let counters_d = counters.clone();
            let name = format!("decode-{track_idx}");
            handles.push(spawn_stage(abort_d, name, move |abort| {
                run_decode_stage(decoder, pkt_rx, frame0_tx, abort, counters_d)
            }));
        }

        let mut upstream: Receiver<Msg<Frame>> = frame0_rx;
        for (fidx, stage) in frame_stages.into_iter().enumerate() {
            let (ftx, frx) = mpsc::sync_channel::<Msg<Frame>>(FRAME_CAP);
            let label = match &stage {
                FrameStage::Filter(_) => "filter",
                FrameStage::PixConvert(_) => "convert",
            };
            let name = format!("{label}-{track_idx}-{fidx}");
            let abort_f = abort.clone();
            handles.push(spawn_stage(abort_f, name, move |abort| {
                run_frame_stage_worker(stage, upstream, ftx, abort)
            }));
            upstream = frx;
        }

        if let Some(enc) = encoder {
            let abort_e = abort.clone();
            let counters_e = counters.clone();
            let out_tx = out_tx.clone();
            let name = format!("encode-{track_idx}");
            handles.push(spawn_stage(abort_e, name, move |abort| {
                run_encode_stage(
                    enc,
                    upstream,
                    out_tx,
                    track_idx as u32,
                    kind,
                    abort,
                    counters_e,
                )
            }));
        } else {
            // No encoder — raw frames flow into the mux (player scenario).
            let abort_r = abort.clone();
            let out_tx = out_tx.clone();
            let name = format!("frame-fanout-{track_idx}");
            handles.push(spawn_stage(abort_r, name, move |abort| {
                run_frame_fanout(upstream, out_tx, track_idx as u32, kind, abort)
            }));
        }
    }

    // Drop the master copies of the output channels; only workers hold
    // senders now so `recv_timeout` sees RecvTimeoutError::Disconnected
    // when every stage has finished.
    drop(track_output_tx);

    // Spawn one demuxer thread per URI.
    for (uri, dmx) in dmx_by_uri {
        let routes = routes_by_uri.remove(&uri).unwrap_or_default();
        if routes.is_empty() {
            continue;
        }
        let abort_d = abort.clone();
        let counters_d = counters.clone();
        let name = format!("demux-{uri}");
        handles.push(spawn_stage(abort_d, name, move |abort| {
            run_demuxer_stage(dmx, routes, abort, counters_d)
        }));
    }

    // Mux loop on the caller thread — recv across every track output
    // channel until all are EOF or abort is set.
    let mut eof_count = 0usize;
    let total = track_output_rx.len();
    let mut i = 0usize;
    while eof_count < total {
        if abort.is_aborted() {
            break;
        }
        let rx = &track_output_rx[i];
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Msg::Data(item)) => match item.payload {
                OutputPayload::Packet(mut p) => {
                    p.stream_index = item.track_index;
                    if let Err(e) = sink.write_packet(item.kind, &p) {
                        abort.record_error(e);
                        break;
                    }
                }
                OutputPayload::Frame(f) => {
                    if let Err(e) = sink.write_frame(item.kind, &f) {
                        abort.record_error(e);
                        break;
                    }
                    counters.frames_written.fetch_add(1, Ordering::SeqCst);
                }
            },
            Ok(Msg::Eof) => {
                eof_count += 1;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Producer panicked or exited without sending Eof —
                // count as EOF to avoid hanging. Any error was already
                // recorded on the abort state.
                eof_count += 1;
            }
        }
        i = (i + 1) % total;
    }

    // Drain abort flag + wait for workers regardless of exit path.
    abort.abort.store(true, Ordering::SeqCst);
    for h in handles {
        let _ = h.join();
    }
    if let Some(err) = abort.take_error() {
        return Err(err);
    }
    sink.finish()?;
    Ok(counters.snapshot())
}

/// Spawn a worker thread that runs `work` under `abort`. If `work`
/// returns `Err`, record it on `abort` (first-wins) and flip the abort
/// flag so peers can bail.
fn spawn_stage<F>(abort: Arc<AbortState>, name: String, work: F) -> JoinHandle<()>
where
    F: FnOnce(Arc<AbortState>) -> Result<()> + Send + 'static,
{
    thread::Builder::new()
        .name(format!("oxideav-job:{name}"))
        .spawn(move || {
            if let Err(e) = work(abort.clone()) {
                abort.record_error(e);
            }
        })
        .expect("pipeline: thread spawn")
}

// ───────────────────────── stage workers ─────────────────────────

/// Demuxer thread: read packets until EOF, fan out to each route whose
/// source_stream matches. Broadcasts `Msg::Eof` to every route on EOF.
fn run_demuxer_stage(
    mut dmx: Box<dyn Demuxer>,
    routes: Vec<(u32, SyncSender<Msg<Packet>>)>,
    abort: Arc<AbortState>,
    counters: Arc<PipelineCounters>,
) -> Result<()> {
    loop {
        if abort.is_aborted() {
            break;
        }
        match dmx.next_packet() {
            Ok(pkt) => {
                counters.packets_read.fetch_add(1, Ordering::SeqCst);
                for (stream_idx, tx) in &routes {
                    if *stream_idx == pkt.stream_index {
                        if tx.send(Msg::Data(pkt.clone())).is_err() {
                            // Consumer gone; likely aborted.
                            abort.abort.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }
    for (_, tx) in routes {
        let _ = tx.send(Msg::Eof);
    }
    Ok(())
}

/// Copy track: packets straight to the output channel.
fn run_copy_stage(
    rx: Receiver<Msg<Packet>>,
    out_tx: SyncSender<Msg<OutputItem>>,
    track_index: u32,
    kind: MediaType,
    abort: Arc<AbortState>,
    counters: Arc<PipelineCounters>,
) -> Result<()> {
    loop {
        if abort.is_aborted() {
            break;
        }
        match rx.recv() {
            Ok(Msg::Data(pkt)) => {
                if out_tx
                    .send(Msg::Data(OutputItem {
                        track_index,
                        kind,
                        payload: OutputPayload::Packet(pkt),
                    }))
                    .is_err()
                {
                    break;
                }
                counters.packets_copied.fetch_add(1, Ordering::SeqCst);
            }
            Ok(Msg::Eof) | Err(_) => break,
        }
    }
    let _ = out_tx.send(Msg::Eof);
    Ok(())
}

/// Decoder stage: packets -> frames.
fn run_decode_stage(
    mut decoder: Box<dyn Decoder>,
    rx: Receiver<Msg<Packet>>,
    tx: SyncSender<Msg<Frame>>,
    abort: Arc<AbortState>,
    counters: Arc<PipelineCounters>,
) -> Result<()> {
    let mut scratch = ExecutorStats::default();
    loop {
        if abort.is_aborted() {
            break;
        }
        match rx.recv() {
            Ok(Msg::Data(pkt)) => {
                decoder.send_packet(&pkt)?;
                let frames = drain_decoder(decoder.as_mut(), &mut scratch)?;
                counters
                    .frames_decoded
                    .fetch_add(frames.len() as u64, Ordering::SeqCst);
                for f in frames {
                    if tx.send(Msg::Data(f)).is_err() {
                        abort.abort.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
            Ok(Msg::Eof) => {
                decoder.flush()?;
                let frames = drain_decoder(decoder.as_mut(), &mut scratch)?;
                counters
                    .frames_decoded
                    .fetch_add(frames.len() as u64, Ordering::SeqCst);
                for f in frames {
                    let _ = tx.send(Msg::Data(f));
                }
                break;
            }
            Err(_) => break,
        }
    }
    let _ = tx.send(Msg::Eof);
    Ok(())
}

/// Frame-stage worker: consumes frames, runs them through an audio
/// filter or pixel-format conversion, and forwards to the next stage.
/// Used for both `FrameStage::Filter` and `FrameStage::PixConvert`.
fn run_frame_stage_worker(
    mut stage: FrameStage,
    rx: Receiver<Msg<Frame>>,
    tx: SyncSender<Msg<Frame>>,
    abort: Arc<AbortState>,
) -> Result<()> {
    loop {
        if abort.is_aborted() {
            break;
        }
        match rx.recv() {
            Ok(Msg::Data(frame)) => {
                let outs = run_frame_stage(&mut stage, frame)?;
                for o in outs {
                    if tx.send(Msg::Data(o)).is_err() {
                        abort.abort.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
            Ok(Msg::Eof) => {
                let outs = flush_frame_stage(&mut stage)?;
                for o in outs {
                    let _ = tx.send(Msg::Data(o));
                }
                break;
            }
            Err(_) => break,
        }
    }
    let _ = tx.send(Msg::Eof);
    Ok(())
}

/// Encoder stage: frames -> packets -> OutputItem.
fn run_encode_stage(
    mut encoder: Box<dyn Encoder>,
    rx: Receiver<Msg<Frame>>,
    out_tx: SyncSender<Msg<OutputItem>>,
    track_index: u32,
    kind: MediaType,
    abort: Arc<AbortState>,
    counters: Arc<PipelineCounters>,
) -> Result<()> {
    loop {
        if abort.is_aborted() {
            break;
        }
        match rx.recv() {
            Ok(Msg::Data(frame)) => {
                encoder.send_frame(&frame)?;
                drain_and_send(encoder.as_mut(), &out_tx, track_index, kind, &counters)?;
            }
            Ok(Msg::Eof) => {
                encoder.flush()?;
                drain_and_send(encoder.as_mut(), &out_tx, track_index, kind, &counters)?;
                break;
            }
            Err(_) => break,
        }
    }
    let _ = out_tx.send(Msg::Eof);
    Ok(())
}

/// Frame fan-out (no encoder): just forwards raw frames to the mux /
/// sink. Used when the output sink is something like the SDL2 player.
fn run_frame_fanout(
    rx: Receiver<Msg<Frame>>,
    out_tx: SyncSender<Msg<OutputItem>>,
    track_index: u32,
    kind: MediaType,
    abort: Arc<AbortState>,
) -> Result<()> {
    loop {
        if abort.is_aborted() {
            break;
        }
        match rx.recv() {
            Ok(Msg::Data(f)) => {
                if out_tx
                    .send(Msg::Data(OutputItem {
                        track_index,
                        kind,
                        payload: OutputPayload::Frame(f),
                    }))
                    .is_err()
                {
                    break;
                }
            }
            Ok(Msg::Eof) | Err(_) => break,
        }
    }
    let _ = out_tx.send(Msg::Eof);
    Ok(())
}

fn drain_and_send(
    encoder: &mut dyn Encoder,
    out_tx: &SyncSender<Msg<OutputItem>>,
    track_index: u32,
    kind: MediaType,
    counters: &PipelineCounters,
) -> Result<()> {
    loop {
        match encoder.receive_packet() {
            Ok(p) => {
                if out_tx
                    .send(Msg::Data(OutputItem {
                        track_index,
                        kind,
                        payload: OutputPayload::Packet(p),
                    }))
                    .is_err()
                {
                    return Ok(()); // consumer gone; caller will see abort
                }
                counters.packets_encoded.fetch_add(1, Ordering::SeqCst);
            }
            Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}
