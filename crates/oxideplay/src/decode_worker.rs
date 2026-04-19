//! Threaded demux + decode pipeline.
//!
//! Three worker threads behind a single [`DecodeWorker`] handle:
//!
//! 1. **Demux thread** — owns the demuxer. Reads packets, drops
//!    subtitle / data streams inline, and routes audio and video
//!    packets onto per-stream bounded channels.
//! 2. **Audio decode thread** — pulls packets off the audio channel,
//!    decodes, and sends `DecodedUnit::Audio` into the shared output
//!    channel to the main thread.
//! 3. **Video decode thread** — symmetric; sends `DecodedUnit::Video`.
//!
//! The split is what keeps audio smooth: a slow video decode (28 ms
//! per frame in debug builds on 640×480) would otherwise serialise
//! with audio on a single worker and underrun SDL's audio device.
//! With this split, audio decode runs freely on its own core regardless
//! of how long video decode takes.
//!
//! Seek is handled by the demux thread via a command channel. On
//! seek it drains both packet channels, instructs the decoders to
//! reset, and resumes producing. The `Seeked` marker is injected into
//! the output stream so the main thread can discard pre-seek
//! Audio/Video units still in flight.
//!
//! Shutdown: `DecodeWorker::Drop` sets an `AtomicBool` and sends
//! `Shutdown` on the command channel; all three threads observe it
//! at their next loop iteration and exit. Channel senders are
//! dropped which unblocks any pending receives.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use oxideav_codec::Decoder;
use oxideav_container::Demuxer;
use oxideav_core::{AudioFrame, Error, Frame, Packet, VideoFrame};

/// Commands from the main thread to the demux thread.
pub enum DecodeCmd {
    Seek { stream_idx: u32, pts: i64 },
    Shutdown,
}

/// Events sent on the control channel. Seek / EOF / error markers
/// don't belong on the per-stream decoded channels because those might
/// be full of audio frames when main needs to see a Seeked ack.
pub enum DecodedCtl {
    Seeked(i64),
    Eof,
    Err(String),
}

/// One item produced by the decode pipeline. Main thread consumes these
/// from three separate channels:
///
/// * Audio frames — lots of them, small each, never blocked by video.
/// * Video frames — fewer, large each, can back up without starving
///   audio.
/// * Control markers — Seeked / Eof / Err.
pub enum DecodedUnit {
    Audio(AudioFrame),
    Video(VideoFrame),
    Ctl(DecodedCtl),
}

/// Per-stream decoded-channel depths. Audio is sized generously (up to
/// a few seconds of buffer) so a main-thread stall can't starve SDL.
/// Video is smaller because each frame is large and main caps at one
/// present per tick anyway.
const AUDIO_OUT_CAP: usize = 128;
const VIDEO_OUT_CAP: usize = 8;
const CTL_CAP: usize = 16;

/// Packet-channel depth per decode stage.
const AUDIO_PKT_CAP: usize = 32;
const VIDEO_PKT_CAP: usize = 16;

/// Handle to the pipeline. Drops cleanly.
pub struct DecodeWorker {
    demux_handle: Option<JoinHandle<()>>,
    audio_handle: Option<JoinHandle<()>>,
    video_handle: Option<JoinHandle<()>>,
    cmd_tx: mpsc::Sender<DecodeCmd>,
    audio_rx: Receiver<AudioFrame>,
    video_rx: Receiver<VideoFrame>,
    ctl_rx: Receiver<DecodedCtl>,
    shutdown: Arc<AtomicBool>,
}

impl DecodeWorker {
    pub fn spawn(
        demuxer: Box<dyn Demuxer>,
        audio_decoder: Option<Box<dyn Decoder>>,
        video_decoder: Option<Box<dyn Decoder>>,
        audio_idx: Option<u32>,
        video_idx: Option<u32>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<DecodeCmd>();
        let (audio_tx, audio_rx) = mpsc::sync_channel::<AudioFrame>(AUDIO_OUT_CAP);
        let (video_tx, video_rx) = mpsc::sync_channel::<VideoFrame>(VIDEO_OUT_CAP);
        let (ctl_tx, ctl_rx) = mpsc::sync_channel::<DecodedCtl>(CTL_CAP);
        let shutdown = Arc::new(AtomicBool::new(false));

        // Per-stream packet channels from demuxer → decoders.
        let (audio_pkt_tx, audio_pkt_rx) = mpsc::sync_channel::<PktMsg>(AUDIO_PKT_CAP);
        let (video_pkt_tx, video_pkt_rx) = mpsc::sync_channel::<PktMsg>(VIDEO_PKT_CAP);

        let audio_handle = audio_decoder.map(|dec| {
            let ctl_tx = ctl_tx.clone();
            let shutdown = shutdown.clone();
            thread::Builder::new()
                .name("oxideplay-audio".into())
                .spawn(move || {
                    decode_loop(
                        dec,
                        audio_pkt_rx,
                        audio_tx,
                        ctl_tx,
                        shutdown,
                        |f| {
                            if let Frame::Audio(af) = f {
                                Some(af)
                            } else {
                                None
                            }
                        },
                        "audio",
                    );
                })
                .expect("spawn audio decode thread")
        });

        let video_handle = video_decoder.map(|dec| {
            let ctl_tx = ctl_tx.clone();
            let shutdown = shutdown.clone();
            thread::Builder::new()
                .name("oxideplay-video".into())
                .spawn(move || {
                    decode_loop(
                        dec,
                        video_pkt_rx,
                        video_tx,
                        ctl_tx,
                        shutdown,
                        |f| {
                            if let Frame::Video(vf) = f {
                                Some(vf)
                            } else {
                                None
                            }
                        },
                        "video",
                    );
                })
                .expect("spawn video decode thread")
        });

        let shutdown_demux = shutdown.clone();
        let demux_handle = thread::Builder::new()
            .name("oxideplay-demux".into())
            .spawn(move || {
                let ctx = DemuxCtx {
                    demuxer,
                    audio_idx,
                    video_idx,
                    audio_pkt_tx,
                    video_pkt_tx,
                    cmd_rx,
                    ctl_tx,
                    shutdown: shutdown_demux,
                };
                ctx.run();
            })
            .expect("spawn demux thread");

        Self {
            demux_handle: Some(demux_handle),
            audio_handle,
            video_handle,
            cmd_tx,
            audio_rx,
            video_rx,
            ctl_rx,
            shutdown,
        }
    }

    /// Try to pull one decoded unit. Audio is checked first so the main
    /// thread always services audio before video — audio is the master
    /// clock and starving SDL is catastrophic; a dropped video frame
    /// is a non-event.
    pub fn try_recv(&self) -> Option<DecodedUnit> {
        if let Ok(af) = self.audio_rx.try_recv() {
            return Some(DecodedUnit::Audio(af));
        }
        if let Ok(ctl) = self.ctl_rx.try_recv() {
            return Some(DecodedUnit::Ctl(ctl));
        }
        if let Ok(vf) = self.video_rx.try_recv() {
            return Some(DecodedUnit::Video(vf));
        }
        None
    }

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
        // The demux thread drops its senders on exit, which hangs up
        // the audio/video packet channels and unblocks the decoders.
        if let Some(h) = self.demux_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.audio_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.video_handle.take() {
            let _ = h.join();
        }
    }
}

// ─────────────────────── per-decoder packet stream ───────────────────

/// In-band message on a packet channel. `Reset` tells the decoder to
/// wipe its internal state (post-seek) and is sent by the demux thread
/// after seeking.
enum PktMsg {
    Pkt(Packet),
    Reset,
}

/// Runs inside a decoder thread. Receives packets, decodes, forwards
/// frames via the per-stream `frame_tx` (`SyncSender<AudioFrame>` or
/// `SyncSender<VideoFrame>`). Errors go out on the shared `ctl_tx`.
fn decode_loop<T, G>(
    mut dec: Box<dyn Decoder>,
    pkt_rx: Receiver<PktMsg>,
    frame_tx: SyncSender<T>,
    ctl_tx: SyncSender<DecodedCtl>,
    shutdown: Arc<AtomicBool>,
    mut filter: G,
    label: &'static str,
) where
    G: FnMut(Frame) -> Option<T>,
{
    while !shutdown.load(Ordering::SeqCst) {
        let msg = match pkt_rx.recv() {
            Ok(m) => m,
            Err(_) => return, // demux side closed
        };
        match msg {
            PktMsg::Reset => {
                let _ = dec.reset();
            }
            PktMsg::Pkt(pkt) => {
                if let Err(e) = dec.send_packet(&pkt) {
                    if !matches!(e, Error::NeedMore) {
                        let _ = ctl_tx.send(DecodedCtl::Err(format!("{label} decode: {e}")));
                    }
                }
                loop {
                    match dec.receive_frame() {
                        Ok(frame) => {
                            if let Some(f) = filter(frame) {
                                if frame_tx.send(f).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(Error::NeedMore) | Err(Error::Eof) => break,
                        Err(e) => {
                            let _ = ctl_tx.send(DecodedCtl::Err(format!("{label} recv: {e}")));
                            break;
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────── demux thread ────────────────────────────

struct DemuxCtx {
    demuxer: Box<dyn Demuxer>,
    audio_idx: Option<u32>,
    video_idx: Option<u32>,
    audio_pkt_tx: SyncSender<PktMsg>,
    video_pkt_tx: SyncSender<PktMsg>,
    cmd_rx: Receiver<DecodeCmd>,
    ctl_tx: SyncSender<DecodedCtl>,
    shutdown: Arc<AtomicBool>,
}

impl DemuxCtx {
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
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            match self.demuxer.next_packet() {
                Ok(p) => {
                    let idx = Some(p.stream_index);
                    if idx == self.audio_idx {
                        // Audio is the critical path — block until the
                        // audio decoder's queue has room. If we dropped
                        // audio here SDL would underrun.
                        if self.audio_pkt_tx.send(PktMsg::Pkt(p)).is_err() {
                            return;
                        }
                    } else if idx == self.video_idx {
                        // Video is discardable — if the video decoder
                        // can't keep up, DROP the packet rather than
                        // block the demux thread. Blocking here would
                        // stop us from ever reading the next audio
                        // packet either (head-of-line blocking), which
                        // is far worse than a dropped video frame.
                        match self.video_pkt_tx.try_send(PktMsg::Pkt(p)) {
                            Ok(()) => {}
                            Err(TrySendError::Full(_)) => {
                                // drop silently; video decoder catches
                                // up on the next I-VOP.
                            }
                            Err(TrySendError::Disconnected(_)) => return,
                        }
                    }
                    // else: subtitle / data / unknown — discard.
                }
                Err(Error::Eof) => {
                    // Signal decoders to flush by closing their input
                    // channels — we simply drop `self.audio_pkt_tx` and
                    // `self.video_pkt_tx` at thread exit. But we need
                    // one more message to tell main that EOF was seen.
                    let _ = self.ctl_tx.send(DecodedCtl::Eof);
                    eof = true;
                }
                Err(e) => {
                    let _ = self.ctl_tx.send(DecodedCtl::Err(format!("demux: {e}")));
                    return;
                }
            }
        }
    }

    /// Poll commands from main. Returns `false` to exit.
    fn poll_commands(&mut self, eof: &mut bool) -> bool {
        loop {
            match self.cmd_rx.try_recv() {
                Ok(DecodeCmd::Seek { stream_idx, pts }) => {
                    match self.demuxer.seek_to(stream_idx, pts) {
                        Ok(landed) => {
                            // Tell the decoders to drop any buffered
                            // state before they see post-seek packets.
                            let _ = self.audio_pkt_tx.send(PktMsg::Reset);
                            let _ = self.video_pkt_tx.send(PktMsg::Reset);
                            *eof = false;
                            if self.ctl_tx.send(DecodedCtl::Seeked(landed)).is_err() {
                                return false;
                            }
                        }
                        Err(e) => {
                            if self
                                .ctl_tx
                                .send(DecodedCtl::Err(format!("seek: {e}")))
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
}
