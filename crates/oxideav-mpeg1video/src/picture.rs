//! Picture-level assembly buffers (Y / Cb / Cr) + display-order reorder.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{PixelFormat, TimeBase, VideoFrame};

use crate::headers::PictureType;

/// Allocate per-picture YUV buffers sized to the macroblock-aligned image.
pub struct PictureBuffer {
    pub width: usize,
    pub height: usize,
    pub mb_width: usize,
    pub mb_height: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub y_stride: usize,
    pub c_stride: usize,
    pub picture_type: PictureType,
    pub temporal_reference: u16,
}

impl PictureBuffer {
    pub fn new(width: usize, height: usize, picture_type: PictureType, tr: u16) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let y_stride = mb_w * 16;
        let c_stride = mb_w * 8;
        let y_h = mb_h * 16;
        let c_h = mb_h * 8;
        Self {
            width,
            height,
            mb_width: mb_w,
            mb_height: mb_h,
            y: vec![0u8; y_stride * y_h],
            cb: vec![0u8; c_stride * c_h],
            cr: vec![0u8; c_stride * c_h],
            y_stride,
            c_stride,
            picture_type,
            temporal_reference: tr,
        }
    }

    /// Copy the MB-aligned luma / chroma buffers into a tight `VideoFrame`
    /// with no padding.
    pub fn to_video_frame(&self, pts: Option<i64>, time_base: TimeBase) -> VideoFrame {
        let w = self.width;
        let h = self.height;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            y[row * w..row * w + w]
                .copy_from_slice(&self.y[row * self.y_stride..row * self.y_stride + w]);
        }
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for row in 0..ch {
            cb[row * cw..row * cw + cw]
                .copy_from_slice(&self.cb[row * self.c_stride..row * self.c_stride + cw]);
            cr[row * cw..row * cw + cw]
                .copy_from_slice(&self.cr[row * self.c_stride..row * self.c_stride + cw]);
        }
        VideoFrame {
            format: PixelFormat::Yuv420P,
            width: w as u32,
            height: h as u32,
            pts,
            time_base,
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: cb,
                },
                VideoPlane {
                    stride: cw,
                    data: cr,
                },
            ],
        }
    }
}
