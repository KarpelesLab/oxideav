//! S3M codec decoder — drives a `PlayerState` and emits stereo S16 PCM.

use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecParameters, Error, Frame, Packet, Result,
    SampleFormat, TimeBase,
};

use crate::container::OUTPUT_SAMPLE_RATE;
use crate::header::parse_header;
use crate::pattern::unpack_all;
use crate::player::PlayerState;
use crate::samples::extract_samples;

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("s3m_sw")
        .with_lossy(false)
        .with_lossless(true)
        .with_intra_only(false)
        .with_max_channels(32)
        .with_max_sample_rate(OUTPUT_SAMPLE_RATE);
    reg.register_decoder_impl(CodecId::new(crate::CODEC_ID_STR), caps, make_decoder);
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(S3mDecoder {
        codec_id: CodecId::new(crate::CODEC_ID_STR),
        state: DecoderState::AwaitingPacket,
    }))
}

struct S3mDecoder {
    codec_id: CodecId,
    state: DecoderState,
}

enum DecoderState {
    AwaitingPacket,
    Playing {
        player: Box<PlayerState>,
        emit_pts: i64,
    },
    Done,
}

const CHUNK_FRAMES: u32 = 1024;

impl Decoder for S3mDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if !matches!(self.state, DecoderState::AwaitingPacket) {
            return Err(Error::other(
                "S3M decoder received a second packet; only one is expected per song",
            ));
        }
        let header = parse_header(&packet.data)?;
        let samples = extract_samples(&header, &packet.data);
        let patterns = unpack_all(&header, &packet.data);
        let player = PlayerState::new(&header, samples, patterns, OUTPUT_SAMPLE_RATE);
        self.state = DecoderState::Playing {
            player: Box::new(player),
            emit_pts: 0,
        };
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match &mut self.state {
            DecoderState::AwaitingPacket => Err(Error::NeedMore),
            DecoderState::Done => Err(Error::Eof),
            DecoderState::Playing { player, emit_pts } => {
                let mut pcm = vec![0i16; CHUNK_FRAMES as usize * 2];
                let produced = player.render(&mut pcm);
                if produced == 0 {
                    self.state = DecoderState::Done;
                    return Err(Error::Eof);
                }
                pcm.truncate(produced * 2);

                let mut bytes = Vec::with_capacity(pcm.len() * 2);
                for s in &pcm {
                    bytes.extend_from_slice(&s.to_le_bytes());
                }

                let pts = *emit_pts;
                *emit_pts += produced as i64;
                Ok(Frame::Audio(AudioFrame {
                    format: SampleFormat::S16,
                    channels: 2,
                    sample_rate: OUTPUT_SAMPLE_RATE,
                    samples: produced as u32,
                    pts: Some(pts),
                    time_base: TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64),
                    data: vec![bytes],
                }))
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
