//! `JobSink` implementation that pushes decoded frames into the SDL2
//! driver.
//!
//! Scope (matches the oxideav-job "first cut" plan): fire-and-forget
//! playback — there is no pause / seek / volume-keyboard loop yet. The
//! user quits with Ctrl-C. A follow-up can thread the event loop in
//! parallel with the executor.

use std::time::Duration;

use oxideav::job::JobSink;
use oxideav_core::{Error, Frame, MediaType, Packet, Result, StreamInfo};

use crate::driver::OutputDriver;
use crate::drivers::sdl2_driver::Sdl2Driver;

pub struct PlayerSink {
    driver: Option<Sdl2Driver>,
    mute: bool,
    want_video: bool,
    /// Sample rate of the first audio stream (for back-pressure).
    audio_rate: u32,
    /// Cap how far the audio queue may run ahead of the speakers. If the
    /// driver buffer goes beyond this we sleep briefly in write_frame to
    /// throttle decoding.
    max_buffered: Duration,
}

impl PlayerSink {
    pub fn new(mute: bool, want_video: bool) -> Self {
        Self {
            driver: None,
            mute,
            want_video,
            audio_rate: 48_000,
            max_buffered: Duration::from_secs(2),
        }
    }

    fn driver_mut(&mut self) -> Result<&mut Sdl2Driver> {
        self.driver
            .as_mut()
            .ok_or_else(|| Error::other("PlayerSink used before start()"))
    }
}

impl JobSink for PlayerSink {
    fn start(&mut self, streams: &[StreamInfo]) -> Result<()> {
        // Pick output parameters from the first audio + first video stream.
        let mut sr = 48_000u32;
        let mut ch = 2u16;
        let mut video_dims: Option<(u32, u32)> = None;
        for s in streams {
            match s.params.media_type {
                MediaType::Audio => {
                    sr = s.params.sample_rate.unwrap_or(48_000);
                    ch = s.params.channels.unwrap_or(2);
                }
                MediaType::Video => {
                    if let (Some(w), Some(h)) = (s.params.width, s.params.height) {
                        video_dims = Some((w, h));
                    }
                }
                _ => {}
            }
        }
        if !self.want_video {
            video_dims = None;
        }
        self.audio_rate = sr.max(1);
        let mut d = Sdl2Driver::new(sr, ch, video_dims)?;
        if self.mute {
            d.set_volume(0.0);
        }
        self.driver = Some(d);
        Ok(())
    }

    fn write_packet(&mut self, _kind: MediaType, _pkt: &Packet) -> Result<()> {
        // The job executor inserts a Decode stage before this sink when
        // the track has no `codec`; raw packet arrivals mean the track
        // was set to stream-copy to @display, which doesn't make sense.
        Err(Error::unsupported(
            "oxideplay: @display sink needs decoded frames; remove `codec` or set it to the source codec with a decoder",
        ))
    }

    fn write_frame(&mut self, _kind: MediaType, frame: &Frame) -> Result<()> {
        let max_samples = (self.audio_rate as u64 * self.max_buffered.as_millis() as u64) / 1000;
        // Back-pressure: if the audio queue is already full, pause the
        // decode loop briefly so we don't build an unbounded memory
        // buffer while SDL drains the device.
        while self.driver_mut()?.audio_queue_len_samples() > max_samples {
            std::thread::sleep(Duration::from_millis(5));
        }
        let d = self.driver_mut()?;
        match frame {
            Frame::Audio(a) => d.queue_audio(a),
            Frame::Video(v) => d.present_video(v),
            // oxideplay doesn't render subtitles yet — silently drop them.
            _ => Ok(()),
        }
    }

    fn finish(&mut self) -> Result<()> {
        // Wait for the audio queue to drain so playback isn't cut off.
        if let Some(d) = self.driver.as_mut() {
            while d.audio_queue_len_samples() > 0 {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        Ok(())
    }
}
