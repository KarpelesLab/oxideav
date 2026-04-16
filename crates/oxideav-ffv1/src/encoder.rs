//! FFV1 frame encoder.
//!
//! Emits a single-slice FFV1 v3 packet per input video frame. The output
//! stream's `extradata` (available via `output_params().extradata`) contains
//! the configuration record; muxers (e.g. Matroska) should read it from
//! there.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::config::ConfigRecord;
use crate::slice::{encode_single_slice_frame, PlaneGeom, SlicePlanes};

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("FFV1 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("FFV1 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    let yuv444 = match pix {
        PixelFormat::Yuv420P => false,
        PixelFormat::Yuv444P => true,
        _ => {
            return Err(Error::unsupported(format!(
                "FFV1 encoder: pixel format {:?}",
                pix
            )))
        }
    };

    let config = ConfigRecord::new_simple(yuv444);
    let extradata = config.encode();

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);
    output_params.extradata = extradata;

    Ok(Box::new(Ffv1Encoder {
        output_params,
        width,
        height,
        pix,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Ffv1Encoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Ffv1Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid("FFV1 encoder: video frames only"));
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "FFV1 encoder: frame dimensions do not match encoder config",
            ));
        }
        if v.format != self.pix {
            return Err(Error::invalid(format!(
                "FFV1 encoder: frame format {:?} vs encoder {:?}",
                v.format, self.pix
            )));
        }
        let data = encode_frame(v)?;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn encode_frame(v: &VideoFrame) -> Result<Vec<u8>> {
    let width = v.width;
    let height = v.height;
    if v.planes.len() != 3 {
        return Err(Error::invalid("FFV1 encoder: expected 3 planes"));
    }
    let (cw, ch) = match v.format {
        PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
        PixelFormat::Yuv444P => (width, height),
        _ => {
            return Err(Error::unsupported(format!(
                "FFV1 encoder: format {:?}",
                v.format
            )))
        }
    };

    // Flatten Y / U / V planes into contiguous w*h buffers (removing any
    // stride padding).
    let y_flat = flatten_plane(&v.planes[0].data, v.planes[0].stride, width, height);
    let u_flat = flatten_plane(&v.planes[1].data, v.planes[1].stride, cw, ch);
    let v_flat = flatten_plane(&v.planes[2].data, v.planes[2].stride, cw, ch);

    let planes = SlicePlanes {
        y: &y_flat,
        u: Some(&u_flat),
        v: Some(&v_flat),
        y_geom: PlaneGeom { width, height },
        c_geom: PlaneGeom {
            width: cw,
            height: ch,
        },
    };

    // RFC 9043 single-slice v3 packet: keyframe bit, slice header, planes,
    // 3-byte slice_size footer. We keep `ec = false` in the config record,
    // which means no CRC/error_status byte is appended per slice — FFmpeg
    // handles either form depending on the ec flag it reads from extradata.
    Ok(encode_single_slice_frame(&planes, false))
}

fn flatten_plane(data: &[u8], stride: usize, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &data[y * stride..y * stride + w];
        out.extend_from_slice(row);
    }
    out
}
