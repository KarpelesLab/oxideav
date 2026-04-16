//! Prefetch-ring-buffer wrapper around any `ReadSeek`.
//!
//! A worker thread owns the inner source and continuously fills a ring
//! buffer ahead of the read cursor. Reads serve from the ring; seeks
//! either move the cursor inside the ring (no IO) or restart the worker
//! at the new offset.
//!
//! Designed for streaming playback over a slow source (HTTP).

use std::collections::VecDeque;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use oxideav_container::ReadSeek;

/// Worker reads at most this many bytes per `inner.read` call.
const BLOCK: usize = 256 * 1024;

/// Shared state between reader and worker.
struct RingState {
    /// Bytes prefetched, oldest first. `buf[0]` corresponds to `ring_start`.
    buf: VecDeque<u8>,
    /// Absolute offset of `buf[0]` in the inner source.
    ring_start: u64,
    /// Maximum number of bytes the ring may hold.
    capacity: usize,
    /// Total length of inner source, if known.
    total_len: Option<u64>,
    /// Worker has reached EOF at the current ring tail.
    eof: bool,
    /// Sticky error from the worker; surfaced on the next reader call.
    err: Option<io::Error>,
    /// Reader has set this to ask the worker to discard the ring and
    /// reposition the inner source. The worker clears it when it has acted.
    target_pos: Option<u64>,
    /// Reader is gone; worker should exit promptly.
    stop: bool,
}

struct Shared {
    state: Mutex<RingState>,
    not_full: Condvar,
    not_empty: Condvar,
}

/// Buffered, prefetching wrapper around any `ReadSeek`.
pub struct BufferedSource {
    shared: Arc<Shared>,
    /// Reader's logical position in the inner source.
    pos: u64,
    /// Worker handle. `None` only between drop signal and join.
    worker: Option<JoinHandle<()>>,
}

impl BufferedSource {
    /// Wrap `inner`, allocating up to `capacity` bytes for the prefetch
    /// ring. Spawns one worker thread that takes ownership of `inner`.
    /// `capacity` is rounded up to at least 4 × `BLOCK` so the worker
    /// always has room to make forward progress.
    pub fn new(mut inner: Box<dyn ReadSeek>, capacity: usize) -> io::Result<Self> {
        let capacity = capacity.max(4 * BLOCK);

        // Determine total length up front (cheap for File / HttpSource).
        let pos = inner.stream_position()?;
        let end = inner.seek(SeekFrom::End(0))?;
        let total_len = Some(end);
        // Restore position.
        inner.seek(SeekFrom::Start(pos))?;

        let state = RingState {
            buf: VecDeque::with_capacity(capacity),
            ring_start: pos,
            capacity,
            total_len,
            eof: total_len == Some(pos),
            err: None,
            target_pos: None,
            stop: false,
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        });

        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || worker_loop(worker_shared, inner));

        Ok(Self {
            shared,
            pos,
            worker: Some(worker),
        })
    }

    /// Total length of the inner source, if known.
    pub fn len(&self) -> Option<u64> {
        self.shared.state.lock().unwrap().total_len
    }

    /// Whether the inner source is known to be empty. Returns `false` if
    /// the length couldn't be determined (treat as non-empty).
    pub fn is_empty(&self) -> bool {
        matches!(self.len(), Some(0))
    }
}

fn worker_loop(shared: Arc<Shared>, mut inner: Box<dyn ReadSeek>) {
    let mut scratch = vec![0u8; BLOCK];
    loop {
        // Phase 1: handle stop / seek requests, wait if ring is full.
        let to_read: usize;
        {
            let mut st = shared.state.lock().unwrap();
            loop {
                if st.stop {
                    return;
                }
                if let Some(target) = st.target_pos.take() {
                    st.buf.clear();
                    st.ring_start = target;
                    st.eof = matches!(st.total_len, Some(end) if target >= end);
                    st.err = None;
                    // Reader may already be sleeping on not_empty waiting
                    // for data at the new position. Wake it so it sees the
                    // updated ring_start / eof state.
                    shared.not_empty.notify_all();
                    drop(st);
                    if let Err(e) = inner.seek(SeekFrom::Start(target)) {
                        let mut st = shared.state.lock().unwrap();
                        st.err = Some(e);
                        shared.not_empty.notify_all();
                        return;
                    }
                    st = shared.state.lock().unwrap();
                    continue;
                }
                if st.eof {
                    // No more data to fetch; sleep until reader seeks or drops.
                    st = shared.not_full.wait(st).unwrap();
                    continue;
                }
                let free = st.capacity - st.buf.len();
                if free == 0 {
                    // Wait for reader to drain.
                    st = shared.not_full.wait(st).unwrap();
                    continue;
                }
                to_read = free.min(BLOCK);
                break;
            }
        }

        // Phase 2: read into scratch outside the lock.
        let read_result = inner.read(&mut scratch[..to_read]);

        // Phase 3: deposit in ring or surface error / EOF.
        let mut st = shared.state.lock().unwrap();
        // Reader may have requested a seek while we were reading; if so,
        // discard what we just read and let phase 1 handle it next loop.
        if st.target_pos.is_some() || st.stop {
            continue;
        }
        match read_result {
            Ok(0) => {
                st.eof = true;
                shared.not_empty.notify_all();
            }
            Ok(n) => {
                st.buf.extend(scratch[..n].iter().copied());
                shared.not_empty.notify_all();
            }
            Err(e) => {
                st.err = Some(e);
                shared.not_empty.notify_all();
                return;
            }
        }
    }
}

impl Read for BufferedSource {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let mut st = self.shared.state.lock().unwrap();
        loop {
            if let Some(e) = st.err.take() {
                return Err(e);
            }
            // Position relative to ring_start.
            let rel = self.pos.saturating_sub(st.ring_start) as usize;
            // If reader is somehow before ring_start (shouldn't happen — Seek
            // bumps target_pos), surface as InvalidInput.
            if self.pos < st.ring_start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "BufferedSource: reader behind ring start",
                ));
            }
            if rel < st.buf.len() {
                // Hit. Copy out.
                let avail = st.buf.len() - rel;
                let n = avail.min(out.len());
                // VecDeque slice view via into-iterator — copy element-wise.
                for (i, byte) in st.buf.iter().skip(rel).take(n).enumerate() {
                    out[i] = *byte;
                }
                self.pos += n as u64;
                // If we've consumed past the front of the ring, drop those
                // bytes so the worker can refill.
                let drop_n = rel + n;
                // But keep some slack so backward seeks within recent past
                // still hit. Use 1/8 of capacity as the "rear" the reader
                // can lookback into without re-fetching.
                let rear = st.capacity / 8;
                if drop_n > rear {
                    let to_drop = drop_n - rear;
                    st.buf.drain(..to_drop);
                    st.ring_start += to_drop as u64;
                    self.shared.not_full.notify_one();
                }
                return Ok(n);
            }
            // Miss: at or past the end of the ring.
            if st.eof {
                return Ok(0);
            }
            // Wait for worker to push more bytes — bounded so a stuck
            // worker becomes visible rather than deadlocking forever.
            let (new_st, wait_result) = self
                .shared
                .not_empty
                .wait_timeout(st, Duration::from_secs(30))
                .unwrap();
            st = new_st;
            if wait_result.timed_out() && st.err.is_none() && !st.eof {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "BufferedSource: prefetch timeout (30s)",
                ));
            }
        }
    }
}

impl Seek for BufferedSource {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let mut st = self.shared.state.lock().unwrap();
        let total = st.total_len;
        let new_pos: u64 = match from {
            SeekFrom::Start(n) => n,
            SeekFrom::Current(d) => add_signed(self.pos, d)?,
            SeekFrom::End(d) => {
                let end = total.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::Unsupported, "stream length unknown")
                })?;
                add_signed(end, d)?
            }
        };
        // If the new position is inside the current ring window, just
        // update the cursor — no IO needed.
        let ring_end = st.ring_start + st.buf.len() as u64;
        if new_pos >= st.ring_start && new_pos <= ring_end {
            self.pos = new_pos;
            return Ok(new_pos);
        }
        // Otherwise tell the worker to reposition the inner source and
        // restart prefetch from `new_pos`.
        st.target_pos = Some(new_pos);
        st.eof = false;
        st.err = None;
        self.pos = new_pos;
        self.shared.not_full.notify_all();
        self.shared.not_empty.notify_all();
        Ok(new_pos)
    }
}

fn add_signed(base: u64, delta: i64) -> io::Result<u64> {
    if delta >= 0 {
        base.checked_add(delta as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek overflow"))
    } else {
        let mag = delta.unsigned_abs();
        base.checked_sub(mag)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek before start"))
    }
}

impl Drop for BufferedSource {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.stop = true;
        }
        self.shared.not_full.notify_all();
        self.shared.not_empty.notify_all();
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}
