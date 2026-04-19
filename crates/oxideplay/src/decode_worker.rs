//! Background decode worker.
//!
//! Owns the demuxer + decoders and runs them on its own thread. Produces
//! a stream of [`DecodedUnit`] values into a bounded `sync_channel` that
//! the main thread drains on every tick.
//!
//! The worker does NOT handle timing: audio/video frames carry their
//! source-stream PTS and the main thread decides when to present them
//! (audio goes into the driver's SDL-backed queue, video into a small
//! VecDeque that the render loop walks in wallclock order).
//!
//! Shutdown happens via (a) an [`AtomicBool`] flag the worker checks at
//! the top of each iteration, and (b) a [`DecodeCmd::Shutdown`] command
//! so a blocked `send()` still returns. `DecodeWorker::drop` sets the
//! flag, sends Shutdown, and joins.
//!
//! Seeks flow through the command channel: main sends
//! [`DecodeCmd::Seek`], worker performs it, resets decoders, emits
//! [`DecodedUnit::Seeked`]. Main is responsible for draining stale
//! `Audio/Video` units queued BEFORE the `Seeked` marker — the worker
//! can't do that because its output is already in-flight.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use oxideav_codec::Decoder;
use oxideav_container::Demuxer;
use oxideav_core::{AudioFrame, Error, Frame, Packet, VideoFrame};

/// Commands sent from the main thread to the worker.
pub enum DecodeCmd {
    /// Seek the demuxer on `stream_idx` to `pts`. Worker resets its
    /// decoders and emits [`DecodedUnit::Seeked`] once done.
    Seek { stream_idx: u32, pts: i64 },
    /// Stop decoding and exit.
    Shutdown,
}

/// One item the worker produces. Main thread consumes these off the
/// output channel on every tick.
pub enum DecodedUnit {
    Audio(AudioFrame),
    Video(VideoFrame),
    /// Worker completed a seek — carries the landed pts in the seeked
    /// stream's time base, same value `Demuxer::seek_to` returned.
    Seeked(i64),
    /// Demuxer hit EOF, decoders flushed. Main can wait for audio to
    /// drain then exit.
    Eof,
    /// Unrecoverable error. Worker is exiting.
    Err(String),
}

/// Bounded output-channel capacity. Small enough to keep decoded-frame
/// memory bounded (one 4K YUV420 frame is ~12 MiB; we don't expect 4K
/// here, but 48 × 640×480 is ~22 MiB worst-case).
const OUT_CAP: usize = 48;

/// Handle to the background decode thread. Drops cleanly — the
/// destructor signals shutdown and joins the thread.
pub struct DecodeWorker {
    handle: Option<JoinHandle<()>>,
    cmd_tx: mpsc::Sender<DecodeCmd>,
    out_rx: Receiver<DecodedUnit>,
    shutdown: Arc<AtomicBool>,
}

impl DecodeWorker {
    /// Spawn a new worker. Returns immediately; decoding happens on the
    /// spawned thread.
    pub fn spawn(
        demuxer: Box<dyn Demuxer>,
        audio_decoder: Option<Box<dyn Decoder>>,
        video_decoder: Option<Box<dyn Decoder>>,
        audio_idx: Option<u32>,
        video_idx: Option<u32>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<DecodeCmd>();
        let (out_tx, out_rx) = mpsc::sync_channel::<DecodedUnit>(OUT_CAP);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_worker = shutdown.clone();
        let handle = thread::Builder::new()
            .name("oxideplay-decode".into())
            .spawn(move || {
                let ctx = WorkerCtx {
                    demuxer,
                    audio_decoder,
                    video_decoder,
                    audio_idx,
                    video_idx,
                    cmd_rx,
                    out_tx,
                    shutdown: shutdown_worker,
                };
                ctx.run();
            })
            .expect("spawn decode thread");
        Self {
            handle: Some(handle),
            cmd_tx,
            out_rx,
            shutdown,
        }
    }

    /// Try to receive one decoded unit without blocking.
    pub fn try_recv(&self) -> Option<DecodedUnit> {
        self.out_rx.try_recv().ok()
    }

    /// Send a seek command. Returns `false` if the worker has exited.
    pub fn seek(&self, stream_idx: u32, pts: i64) -> bool {
        self.cmd_tx
            .send(DecodeCmd::Seek { stream_idx, pts })
            .is_ok()
    }
}

impl Drop for DecodeWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.cmd_tx.send(DecodeCmd::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ─────────────────────────── worker internals ────────────────────────

struct WorkerCtx {
    demuxer: Box<dyn Demuxer>,
    audio_decoder: Option<Box<dyn Decoder>>,
    video_decoder: Option<Box<dyn Decoder>>,
    audio_idx: Option<u32>,
    video_idx: Option<u32>,
    cmd_rx: Receiver<DecodeCmd>,
    out_tx: SyncSender<DecodedUnit>,
    shutdown: Arc<AtomicBool>,
}

impl WorkerCtx {
    fn run(mut self) {
        let mut eof = false;
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return;
            }
            if !self.poll_commands(&mut eof) {
                return;
            }
            if eof {
                // Demuxer is done and we've flushed decoders. Keep the
                // thread alive so it can still service a late seek
                // command; sleep a bit to avoid spinning.
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            match self.read_routed_packet() {
                ReadResult::Packet(pkt) => {
                    let ok = if Some(pkt.stream_index) == self.audio_idx {
                        self.decode_audio(&pkt)
                    } else if Some(pkt.stream_index) == self.video_idx {
                        self.decode_video(&pkt)
                    } else {
                        true
                    };
                    if !ok {
                        return;
                    }
                }
                ReadResult::Eof => {
                    if !self.drain_on_eof() {
                        return;
                    }
                    eof = true;
                }
                ReadResult::Err(e) => {
                    let _ = self.out_tx.send(DecodedUnit::Err(e));
                    return;
                }
                ReadResult::Shutdown => return,
            }
        }
    }

    /// Drain any pending commands. Returns `false` if the worker should
    /// exit (Shutdown received or command channel disconnected).
    fn poll_commands(&mut self, eof: &mut bool) -> bool {
        loop {
            match self.cmd_rx.try_recv() {
                Ok(DecodeCmd::Seek { stream_idx, pts }) => {
                    match self.demuxer.seek_to(stream_idx, pts) {
                        Ok(landed) => {
                            if let Some(d) = self.audio_decoder.as_mut() {
                                let _ = d.reset();
                            }
                            if let Some(d) = self.video_decoder.as_mut() {
                                let _ = d.reset();
                            }
                            *eof = false;
                            if self.out_tx.send(DecodedUnit::Seeked(landed)).is_err() {
                                return false;
                            }
                        }
                        Err(e) => {
                            if self
                                .out_tx
                                .send(DecodedUnit::Err(format!("seek: {e}")))
                                .is_err()
                            {
                                return false;
                            }
                        }
                    }
                }
                Ok(DecodeCmd::Shutdown) => return false,
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    /// Pull the next packet belonging to a routed stream (audio or
    /// video), discarding uninteresting (subtitle / data) packets inline.
    fn read_routed_packet(&mut self) -> ReadResult {
        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return ReadResult::Shutdown;
            }
            match self.demuxer.next_packet() {
                Ok(p) => {
                    let idx = Some(p.stream_index);
                    if idx == self.audio_idx || idx == self.video_idx {
                        return ReadResult::Packet(p);
                    }
                    // Unrouted — skip.
                }
                Err(Error::Eof) => return ReadResult::Eof,
                Err(e) => return ReadResult::Err(e.to_string()),
            }
        }
    }

    fn decode_audio(&mut self, pkt: &Packet) -> bool {
        let Some(dec) = self.audio_decoder.as_mut() else {
            return true;
        };
        if let Err(e) = dec.send_packet(pkt) {
            if !matches!(e, Error::NeedMore) {
                let _ = self
                    .out_tx
                    .send(DecodedUnit::Err(format!("audio decode: {e}")));
            }
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    if self.out_tx.send(DecodedUnit::Audio(af)).is_err() {
                        return false;
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => return true,
                Err(e) => {
                    let _ = self
                        .out_tx
                        .send(DecodedUnit::Err(format!("audio recv: {e}")));
                    return true;
                }
            }
        }
    }

    fn decode_video(&mut self, pkt: &Packet) -> bool {
        let Some(dec) = self.video_decoder.as_mut() else {
            return true;
        };
        if let Err(e) = dec.send_packet(pkt) {
            if !matches!(e, Error::NeedMore) {
                let _ = self
                    .out_tx
                    .send(DecodedUnit::Err(format!("video decode: {e}")));
            }
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    if self.out_tx.send(DecodedUnit::Video(vf)).is_err() {
                        return false;
                    }
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => return true,
                Err(e) => {
                    let _ = self
                        .out_tx
                        .send(DecodedUnit::Err(format!("video recv: {e}")));
                    return true;
                }
            }
        }
    }

    /// Handle demuxer EOF: flush decoders, emit all buffered frames,
    /// then send [`DecodedUnit::Eof`]. Returns `false` if the channel
    /// is gone.
    fn drain_on_eof(&mut self) -> bool {
        if let Some(d) = self.audio_decoder.as_mut() {
            let _ = d.flush();
            while let Ok(Frame::Audio(af)) = d.receive_frame() {
                if self.out_tx.send(DecodedUnit::Audio(af)).is_err() {
                    return false;
                }
            }
        }
        if let Some(d) = self.video_decoder.as_mut() {
            let _ = d.flush();
            while let Ok(Frame::Video(vf)) = d.receive_frame() {
                if self.out_tx.send(DecodedUnit::Video(vf)).is_err() {
                    return false;
                }
            }
        }
        self.out_tx.send(DecodedUnit::Eof).is_ok()
    }
}

enum ReadResult {
    Packet(Packet),
    Eof,
    Err(String),
    Shutdown,
}
