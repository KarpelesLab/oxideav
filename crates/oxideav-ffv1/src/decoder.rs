//! FFV1 packet decoder.
//!
//! Each compressed packet holds one frame: a leading range-coded keyframe
//! bit embedded in the first slice's range coder, followed by one or more
//! slices. Each slice ends with a trailer (`slice_size` + optional
//! `error_status` + `slice_crc_parity`) per RFC 9043 §4.5; the trailer
//! format is selected by the `ec` flag in the configuration record.
//!
//! Our decoder supports arbitrary `num_h_slices × num_v_slices` grids as
//! long as each slice lands on a single cell; it still restricts the rest
//! of the feature space to 8-bit YCbCr 4:2:0 / 4:4:4 with `coder_type = 1`.

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::config::ConfigRecord;
use crate::slice::decode_frame;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let config = if params.extradata.is_empty() {
        // Allow extradata-less streams at our own default.
        let yuv444 = matches!(params.pixel_format, Some(PixelFormat::Yuv444P));
        ConfigRecord::new_simple(yuv444)
    } else {
        ConfigRecord::parse(&params.extradata)?
    };
    let width = params
        .width
        .ok_or_else(|| Error::invalid("FFV1 decoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("FFV1 decoder: missing height"))?;
    Ok(Box::new(Ffv1Decoder {
        codec_id: params.codec_id.clone(),
        config,
        width,
        height,
        pending: None,
        eof: false,
    }))
}

struct Ffv1Decoder {
    codec_id: CodecId,
    config: ConfigRecord,
    width: u32,
    height: u32,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for Ffv1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "FFV1 decoder: receive_frame must be called before another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_packet(&self.config, &pkt, self.width, self.height)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn decode_packet(
    config: &ConfigRecord,
    pkt: &Packet,
    width: u32,
    height: u32,
) -> Result<VideoFrame> {
    let data = &pkt.data;
    if data.is_empty() {
        return Err(Error::invalid("FFV1 decode: empty packet"));
    }

    // Map the pixel format from the config record.
    let pix_fmt = if config.is_yuv420() {
        PixelFormat::Yuv420P
    } else if config.is_yuv444() {
        PixelFormat::Yuv444P
    } else {
        return Err(Error::unsupported("FFV1: unsupported chroma subsampling"));
    };

    let has_chroma = config.chroma_planes;
    let log2_h_sub = config.log2_h_chroma_subsample;
    let log2_v_sub = config.log2_v_chroma_subsample;
    let ec = config.ec != 0;

    let decoded = decode_frame(
        data,
        width,
        height,
        config.num_h_slices,
        config.num_v_slices,
        has_chroma,
        log2_h_sub,
        log2_v_sub,
        ec,
    )?;

    let y_plane = VideoPlane {
        stride: width as usize,
        data: decoded.y,
    };
    let cw = decoded.c_geom.width as usize;
    let u_plane = VideoPlane {
        stride: cw,
        data: decoded.u.unwrap_or_default(),
    };
    let v_plane = VideoPlane {
        stride: cw,
        data: decoded.v.unwrap_or_default(),
    };

    Ok(VideoFrame {
        format: pix_fmt,
        width,
        height,
        pts: pkt.pts,
        time_base: pkt.time_base,
        planes: vec![y_plane, u_plane, v_plane],
    })
}
