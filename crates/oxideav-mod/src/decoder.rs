//! MOD codec decoder — initial scaffold.
//!
//! This decoder parses the MOD header, precomputes song duration, and
//! produces silent PCM frames of the right shape. Paula-channel emulation,
//! effect processing, and sample mixing follow in a dedicated session;
//! the decoder is wired in now so the pipeline (probe / demux / mux)
//! works end-to-end today and later work just fills in the mixer.

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, Error, Frame, Packet, Result,
    SampleFormat, TimeBase,
};

use crate::container::OUTPUT_SAMPLE_RATE;
use crate::header::{parse_header, ModHeader};

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("mod_sw")
        .with_lossy(false)
        .with_lossless(true)
        .with_intra_only(false)
        .with_max_channels(32)
        .with_max_sample_rate(OUTPUT_SAMPLE_RATE);
    reg.register_decoder_impl(CodecId::new(crate::CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(ModDecoder {
        codec_id: CodecId::new(crate::CODEC_ID_STR),
        state: DecoderState::AwaitingPacket,
    }))
}

struct ModDecoder {
    codec_id: CodecId,
    state: DecoderState,
}

enum DecoderState {
    /// Haven't seen the file yet.
    AwaitingPacket,
    /// File parsed; emitting `remaining_frames` of silent output in chunks.
    /// `_header` will be load-bearing once the Paula mixer lands.
    Playing {
        _header: ModHeader,
        remaining_frames: u64,
        emit_pts: i64,
    },
    /// All samples produced.
    Done,
}

const CHUNK_FRAMES: u32 = 1024;

impl Decoder for ModDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // The MOD "container" delivers the whole file in one packet.
        if !matches!(self.state, DecoderState::AwaitingPacket) {
            return Err(Error::other(
                "MOD decoder received a second packet; only one is expected per song",
            ));
        }
        let header = parse_header(&packet.data)?;
        let total_frames = estimate_total_frames(&header);
        self.state = DecoderState::Playing {
            _header: header,
            remaining_frames: total_frames,
            emit_pts: 0,
        };
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match &mut self.state {
            DecoderState::AwaitingPacket => Err(Error::NeedMore),
            DecoderState::Done => Err(Error::Eof),
            DecoderState::Playing {
                remaining_frames,
                emit_pts,
                ..
            } => {
                if *remaining_frames == 0 {
                    self.state = DecoderState::Done;
                    return Err(Error::Eof);
                }
                let frames = (*remaining_frames).min(CHUNK_FRAMES as u64) as u32;
                let channels = 2u16;
                let bytes = (frames as usize) * (channels as usize) * 2; // S16 stereo
                let data = vec![0u8; bytes]; // silent PCM until the mixer lands
                let pts = *emit_pts;
                *emit_pts += frames as i64;
                *remaining_frames -= frames as u64;
                Ok(Frame::Audio(AudioFrame {
                    format: SampleFormat::S16,
                    channels,
                    sample_rate: OUTPUT_SAMPLE_RATE,
                    samples: frames,
                    pts: Some(pts),
                    time_base: TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64),
                    data: vec![data],
                }))
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        if let DecoderState::Playing { .. } = self.state {
            // Draining is implicit — `receive_frame` will return Eof once
            // `remaining_frames` hits zero.
        }
        Ok(())
    }
}

/// Rough duration estimate: song-length patterns × 64 rows × 6 ticks × ~882
/// samples (= 44100 / 50 Hz VBlank). Real MODs can change tempo via effect
/// Fxx — this is an upper-bound placeholder until the mixer lands.
fn estimate_total_frames(header: &ModHeader) -> u64 {
    const ROWS_PER_PATTERN: u64 = 64;
    const TICKS_PER_ROW: u64 = 6;
    const SAMPLES_PER_TICK: u64 = OUTPUT_SAMPLE_RATE as u64 / 50; // 882 @ 44.1 kHz
    let patterns = header.song_length as u64;
    patterns * ROWS_PER_PATTERN * TICKS_PER_ROW * SAMPLES_PER_TICK
}
