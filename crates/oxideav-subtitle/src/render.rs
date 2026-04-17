//! Wrapper decoder that turns `Frame::Subtitle` into `Frame::Video` RGBA
//! by running each cue through a [`Compositor`].
//!
//! The wrapper is transparent: it forwards `send_packet` / `flush` to the
//! inner decoder and calls `receive_frame` on the inner to pull cues. On
//! each new distinct cue, it rasterises into a fresh RGBA video frame
//! and returns `Frame::Video`.
//!
//! "Distinct" is measured with a stable 64-bit hash of the cue's visible
//! content (segments + style ref + positioning). Re-emitting the exact
//! same cue — whatever the reason (decoder idempotency, container-level
//! re-send, …) — yields `Error::NeedMore` so downstream pipelines don't
//! do redundant work or ship duplicate frames.
//!
//! On cue *exit* (the inner decoder returning `Error::NeedMore` with no
//! new cue available), this wrapper also returns `Error::NeedMore`. There
//! is no automatic transparent frame emitted at cue end — downstream
//! video compositor code should treat "no new subtitle frame" as
//! "continue showing the last one until the next change", which matches
//! the typical overlay compositor behaviour.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CuePosition, Error, Frame, Packet, PixelFormat, Result, Segment, SubtitleCue,
    TextAlign, TimeBase, VideoFrame, VideoPlane,
};

use crate::compositor::Compositor;

/// Decoder that wraps another subtitle decoder and emits rasterised RGBA
/// video frames instead of raw subtitle cues.
pub struct RenderedSubtitleDecoder {
    inner: Box<dyn Decoder>,
    compositor: Compositor,
    last_rendered_hash: Option<u64>,
    codec_id: CodecId,
}

impl RenderedSubtitleDecoder {
    pub fn new(inner: Box<dyn Decoder>, width: u32, height: u32) -> Self {
        let codec_id = inner.codec_id().clone();
        Self {
            inner,
            compositor: Compositor::new(width, height),
            last_rendered_hash: None,
            codec_id,
        }
    }

    /// Mutable access to the underlying compositor. Use this to tune
    /// colours, margin, or outline width before the first frame.
    pub fn compositor_mut(&mut self) -> &mut Compositor {
        &mut self.compositor
    }
}

impl Decoder for RenderedSubtitleDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.inner.send_packet(packet)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        loop {
            let frame = self.inner.receive_frame()?;
            let cue = match frame {
                Frame::Subtitle(c) => c,
                // Pass through anything non-subtitle unchanged.
                other => return Ok(other),
            };
            let hash = hash_cue(&cue);
            if self.last_rendered_hash == Some(hash) {
                // Identical content — don't re-emit.
                return Err(Error::NeedMore);
            }
            self.last_rendered_hash = Some(hash);
            let vf = render_cue_to_video_frame(&cue, &self.compositor);
            return Ok(Frame::Video(vf));
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

/// Factory: wrap an existing subtitle decoder in a RenderedSubtitleDecoder
/// at the given output resolution.
pub fn make_rendered_decoder(inner: Box<dyn Decoder>, width: u32, height: u32) -> Box<dyn Decoder> {
    Box::new(RenderedSubtitleDecoder::new(inner, width, height))
}

fn render_cue_to_video_frame(cue: &SubtitleCue, comp: &Compositor) -> VideoFrame {
    let rgba = comp.render(cue);
    let stride = comp.width as usize * 4;
    let duration = (cue.end_us - cue.start_us).max(0);
    let _ = duration; // duration is currently carried on packets, not frames.
    VideoFrame {
        format: PixelFormat::Rgba,
        width: comp.width,
        height: comp.height,
        pts: Some(cue.start_us),
        time_base: TimeBase::new(1, 1_000_000),
        planes: vec![VideoPlane { stride, data: rgba }],
    }
}

// ---------------------------------------------------------------------------
// Cue hashing
// ---------------------------------------------------------------------------

fn hash_cue(cue: &SubtitleCue) -> u64 {
    let mut h = DefaultHasher::new();
    cue.start_us.hash(&mut h);
    cue.end_us.hash(&mut h);
    cue.style_ref.hash(&mut h);
    hash_position(&cue.positioning, &mut h);
    hash_segments(&cue.segments, &mut h);
    h.finish()
}

fn hash_position(p: &Option<CuePosition>, h: &mut DefaultHasher) {
    match p {
        Some(pos) => {
            b"pos".hash(h);
            pos.x.map(|v| v.to_bits()).hash(h);
            pos.y.map(|v| v.to_bits()).hash(h);
            match pos.align {
                TextAlign::Start => 0u8,
                TextAlign::Center => 1u8,
                TextAlign::End => 2u8,
                TextAlign::Left => 3u8,
                TextAlign::Right => 4u8,
            }
            .hash(h);
            pos.size.map(|v| v.to_bits()).hash(h);
        }
        None => {
            b"nopos".hash(h);
        }
    }
}

fn hash_segments(segs: &[Segment], h: &mut DefaultHasher) {
    for s in segs {
        hash_segment(s, h);
    }
    b";".hash(h);
}

fn hash_segment(seg: &Segment, h: &mut DefaultHasher) {
    match seg {
        Segment::Text(s) => {
            b"T".hash(h);
            s.hash(h);
        }
        Segment::LineBreak => {
            b"N".hash(h);
        }
        Segment::Bold(c) => {
            b"B".hash(h);
            hash_segments(c, h);
        }
        Segment::Italic(c) => {
            b"I".hash(h);
            hash_segments(c, h);
        }
        Segment::Underline(c) => {
            b"U".hash(h);
            hash_segments(c, h);
        }
        Segment::Strike(c) => {
            b"S".hash(h);
            hash_segments(c, h);
        }
        Segment::Color { rgb, children } => {
            b"C".hash(h);
            rgb.hash(h);
            hash_segments(children, h);
        }
        Segment::Font {
            family,
            size,
            children,
        } => {
            b"F".hash(h);
            family.hash(h);
            size.map(|v| v.to_bits()).hash(h);
            hash_segments(children, h);
        }
        Segment::Voice { name, children } => {
            b"V".hash(h);
            name.hash(h);
            hash_segments(children, h);
        }
        Segment::Class { name, children } => {
            b"L".hash(h);
            name.hash(h);
            hash_segments(children, h);
        }
        Segment::Karaoke { cs, children } => {
            b"K".hash(h);
            cs.hash(h);
            hash_segments(children, h);
        }
        Segment::Timestamp { offset_us } => {
            b"M".hash(h);
            offset_us.hash(h);
        }
        Segment::Raw(s) => {
            b"R".hash(h);
            s.hash(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mkcue(text: &str) -> SubtitleCue {
        SubtitleCue {
            start_us: 1_000_000,
            end_us: 2_000_000,
            style_ref: None,
            positioning: None,
            segments: vec![Segment::Text(text.to_string())],
        }
    }

    #[test]
    fn identical_cues_share_hash() {
        let a = mkcue("Hello");
        let b = mkcue("Hello");
        assert_eq!(hash_cue(&a), hash_cue(&b));
    }

    #[test]
    fn different_text_different_hash() {
        let a = mkcue("Hello");
        let b = mkcue("World");
        assert_ne!(hash_cue(&a), hash_cue(&b));
    }
}
