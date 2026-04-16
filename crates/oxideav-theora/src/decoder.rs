//! Theora video decoder.
//!
//! Status: header parsing (identification + comment + setup) from extradata,
//! codec parameter population, frame-header classification, full **intra
//! frame** decode, and **inter (P-frame)** decode with motion compensation
//! against the previous reference frame and the golden frame.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat as CorePixelFormat, Rational,
    Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::bitreader::BitReader;
use crate::block::IntraFrameDecoder;
use crate::coded_order::FrameLayout;
use crate::headers::{parse_headers_from_extradata, Headers, PixelFormat};
use crate::inter::Mode;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let headers = if params.extradata.is_empty() {
        None
    } else {
        Some(parse_headers_from_extradata(&params.extradata)?)
    };
    Ok(Box::new(TheoraDecoder {
        codec_id: params.codec_id.clone(),
        headers,
        ready_frames: VecDeque::new(),
        pending_pts: None,
        pending_tb: TimeBase::new(1, 90_000),
        eof: false,
        prev_ref: None,
        golden_ref: None,
    }))
}

pub struct TheoraDecoder {
    codec_id: CodecId,
    pub(crate) headers: Option<Headers>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
    /// Previous reconstructed frame (post-loop-filter) used as motion-comp
    /// reference for inter frames (RFI=1).
    prev_ref: Option<[Vec<u8>; 3]>,
    /// Golden reference frame; updated only on keyframes.
    golden_ref: Option<[Vec<u8>; 3]>,
}

/// Theora "frame type": false = intra, true = inter (spec §7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    Intra,
    Inter,
}

/// Classification of a packet: the three headers (bit 0 = 1) or a data frame
/// (bit 0 = 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketKind {
    Identification,
    Comment,
    Setup,
    Frame(FrameType),
}

/// Inspect the leading bits of a Theora packet to identify its kind.
pub fn classify_packet(packet: &[u8]) -> Result<PacketKind> {
    if packet.is_empty() {
        return Err(Error::invalid("empty Theora packet"));
    }
    let b0 = packet[0];
    // Header packet: MSB (bit 0 in Theora numbering) = 1. The next 7 bits are
    // the header type code: 0x80 = ID, 0x81 = comment, 0x82 = setup.
    if b0 & 0x80 != 0 {
        return match b0 {
            0x80 => Ok(PacketKind::Identification),
            0x81 => Ok(PacketKind::Comment),
            0x82 => Ok(PacketKind::Setup),
            other => Err(Error::invalid(format!(
                "Theora: unknown header packet type 0x{other:02X}"
            ))),
        };
    }
    // Frame packet: bit 0 is 0. Bit 1 is FTYPE: 0 = intra, 1 = inter.
    let ftype_bit = (b0 & 0x40) != 0;
    Ok(PacketKind::Frame(if ftype_bit {
        FrameType::Inter
    } else {
        FrameType::Intra
    }))
}

impl TheoraDecoder {
    fn ensure_headers_from_packet(&mut self, packet: &[u8]) -> Result<bool> {
        match classify_packet(packet)? {
            PacketKind::Identification | PacketKind::Comment | PacketKind::Setup => Ok(true),
            PacketKind::Frame(_) => Ok(false),
        }
    }

    fn decode_frame(&mut self, packet: &[u8]) -> Result<()> {
        let headers_ptr: *const Headers = match self.headers.as_ref() {
            Some(h) => h as *const _,
            None => {
                return Err(Error::invalid(
                    "Theora frame packet before headers were parsed",
                ))
            }
        };
        // SAFETY: `headers` outlives this function call; we only need a shared
        // reference for the duration of decode_frame.
        let headers: &Headers = unsafe { &*headers_ptr };

        let kind = classify_packet(packet)?;
        let frame_type = match kind {
            PacketKind::Frame(ft) => ft,
            _ => return Err(Error::invalid("Theora: expected frame packet")),
        };

        let mut br = BitReader::new(packet);
        // §7.1 frame header prelude.
        let marker = br.read_bit()?;
        if marker {
            return Err(Error::invalid("Theora: frame marker non-zero"));
        }
        let _ftype = br.read_bit()?;

        let mut frame = IntraFrameDecoder::new(headers);
        match frame_type {
            FrameType::Intra => {
                frame.is_inter = false;
                frame.read_intra_frame_header(&mut br)?;
                frame.fill_bcoded_intra();
                // §7.4 macro-block modes: intra frame — all INTRA (implicit).
                for m in frame.modes.iter_mut() {
                    *m = Mode::Intra;
                }
                frame.read_qiis(&mut br)?;
                frame.decode_coefficients(&mut br)?;
                frame.undo_dc_prediction();
                let mut planes_buf = frame.reconstruct();
                frame.loop_filter(&mut planes_buf);
                self.emit_frame(headers, &planes_buf)?;
                // Keyframe updates BOTH golden and prev refs.
                self.golden_ref = Some(planes_buf.clone());
                self.prev_ref = Some(planes_buf);
            }
            FrameType::Inter => {
                if self.prev_ref.is_none() {
                    return Err(Error::invalid("Theora: inter frame without prior keyframe"));
                }
                frame.is_inter = true;
                frame.read_inter_frame_header(&mut br)?;
                // §7.3.2/§7.3.3 BCODED + SBPMs.
                crate::inter::read_inter_bcoded(&mut br, &frame.layout, &mut frame.bcoded)?;
                // §7.3.4 macro-block modes.
                read_inter_modes(&mut br, &mut frame)?;
                // §7.3.5 motion vectors.
                read_inter_mvs(&mut br, &mut frame)?;
                // §7.6 qii.
                frame.read_qiis(&mut br)?;
                // §7.7.3 DCT coefficients.
                frame.decode_coefficients(&mut br)?;
                // §7.8.2 undo DC prediction.
                frame.undo_dc_prediction();
                // §7.9.4 reconstruction with motion compensation.
                let prev = self.prev_ref.as_ref().unwrap();
                let golden = self.golden_ref.as_ref();
                let mut planes_buf = frame.reconstruct_with_refs(Some(prev), golden);
                // §7.10 loop filter.
                frame.loop_filter(&mut planes_buf);
                self.emit_frame(headers, &planes_buf)?;
                self.prev_ref = Some(planes_buf);
            }
        }
        Ok(())
    }

    fn emit_frame(&mut self, headers: &Headers, planes_buf: &[Vec<u8>; 3]) -> Result<()> {
        let id = &headers.identification;
        let (cropped_planes, out_w, out_h) = crop_to_picture(id, planes_buf);
        let core_fmt = pixel_format_to_core(id.pf);
        let planes = build_video_planes(&cropped_planes, out_w, out_h, id.pf);
        let video = VideoFrame {
            format: core_fmt,
            width: out_w,
            height: out_h,
            pts: self.pending_pts,
            time_base: self.pending_tb,
            planes,
        };
        self.ready_frames.push_back(video);
        Ok(())
    }
}

/// Macroblock-Hilbert order within one super-block (spec Figure 2.6).
/// Order: bottom-left, top-left, top-right, bottom-right.
/// In Figure 2.6 the diagram "1 2 / 0 3" labels the position with its order:
///   (0, 0) lower-left = order 0
///   (0, 1) upper-left = order 1
///   (1, 1) upper-right = order 2
///   (1, 0) lower-right = order 3
const MB_HILBERT: [(u32, u32); 4] = [(0, 0), (0, 1), (1, 1), (1, 0)];

/// Iterate macroblocks in Theora coded order. For each MB yields its
/// (mbx, mby) in luma-MB coordinates (i.e. half of luma-block coords).
fn for_each_mb<F: FnMut(u32, u32) -> Result<()>>(layout: &FrameLayout, mut f: F) -> Result<()> {
    let plane = &layout.planes[0];
    let nbw = plane.nbw;
    let nbh = plane.nbh;
    // Super-blocks in luma plane.
    let sbw = nbw.div_ceil(4);
    let sbh = nbh.div_ceil(4);
    let mbw = nbw / 2;
    let mbh = nbh / 2;
    for sby in 0..sbh {
        for sbx in 0..sbw {
            // 4 MBs per SB (2x2). MB coords inside SB: from MB_HILBERT.
            for &(dx, dy) in &MB_HILBERT {
                let mbx = sbx * 2 + dx;
                let mby = sby * 2 + dy;
                if mbx < mbw && mby < mbh {
                    f(mbx, mby)?;
                }
            }
        }
    }
    Ok(())
}

/// Decode per-MB modes and broadcast to the 4 luma blocks + 1 chroma block per
/// chroma plane (4:2:0). Block-level modes follow the macroblock's mode (Table
/// 7.18 row notes).
fn read_inter_modes(br: &mut BitReader<'_>, frame: &mut IntraFrameDecoder<'_>) -> Result<()> {
    use crate::inter::{decode_mode_rank, read_mode_alphabet, read_raw_mode, ModeScheme};
    let scheme = read_mode_alphabet(br)?;
    let (alphabet, raw_mode) = match scheme {
        ModeScheme::Alphabet(a) => (a, false),
        ModeScheme::Raw => ([Mode::InterNoMv; 8], true),
    };

    let layout = frame.layout.clone();
    let pf = frame.headers.identification.pf;
    // Walk MBs, collect modes to apply.
    let mut decisions: Vec<(Mode, [usize; 4], Vec<usize>)> = Vec::new();
    for_each_mb(&layout, |mbx, mby| {
        let bxs = [mbx * 2, mbx * 2 + 1, mbx * 2, mbx * 2 + 1];
        let bys = [mby * 2, mby * 2, mby * 2 + 1, mby * 2 + 1];
        let bis: [usize; 4] = [
            layout.global_coded(0, bxs[0], bys[0]) as usize,
            layout.global_coded(0, bxs[1], bys[1]) as usize,
            layout.global_coded(0, bxs[2], bys[2]) as usize,
            layout.global_coded(0, bxs[3], bys[3]) as usize,
        ];
        let any_coded = bis.iter().any(|&bi| frame.bcoded[bi]);
        let mb_mode = if !any_coded {
            Mode::InterNoMv
        } else if raw_mode {
            read_raw_mode(br)?
        } else {
            let r = decode_mode_rank(br)? as usize;
            alphabet[r]
        };
        // Collect chroma block indices.
        let mut chroma = Vec::new();
        for pli in 1..3 {
            let cplane = &layout.planes[pli];
            let cnbw = cplane.nbw;
            let cnbh = cplane.nbh;
            let (cbx, cby, dxr, dyr) = match pf {
                PixelFormat::Yuv420 => (mbx, mby, 1u32, 1u32),
                PixelFormat::Yuv422 => (mbx, mby * 2, 1u32, 2u32),
                PixelFormat::Yuv444 | PixelFormat::Reserved => (mbx * 2, mby * 2, 2u32, 2u32),
            };
            for dy in 0..dyr {
                for dx in 0..dxr {
                    let x = cbx + dx;
                    let y = cby + dy;
                    if x < cnbw && y < cnbh {
                        chroma.push(layout.global_coded(pli, x, y) as usize);
                    }
                }
            }
        }
        decisions.push((mb_mode, bis, chroma));
        Ok(())
    })?;
    for (mode, bis, chroma) in decisions {
        for bi in bis {
            frame.modes[bi] = mode;
        }
        for bj in chroma {
            frame.modes[bj] = mode;
        }
    }
    Ok(())
}

/// Decode motion vectors per spec §7.3.5.
fn read_inter_mvs(br: &mut BitReader<'_>, frame: &mut IntraFrameDecoder<'_>) -> Result<()> {
    use crate::inter::{decode_mv_component_raw, decode_mv_component_vlc};
    // 1 bit: MVMODE (0 = VLC, 1 = raw 6-bit per component).
    let mvmode_raw = br.read_bit()?;
    let read_mv = |br: &mut BitReader<'_>| -> Result<(i32, i32)> {
        if mvmode_raw {
            let x = decode_mv_component_raw(br)?;
            let y = decode_mv_component_raw(br)?;
            Ok((x, y))
        } else {
            let x = decode_mv_component_vlc(br)?;
            let y = decode_mv_component_vlc(br)?;
            Ok((x, y))
        }
    };
    let mut last_mv = (0i32, 0i32);
    let mut last_mv2 = (0i32, 0i32);

    let layout = frame.layout.clone();
    let pf = frame.headers.identification.pf;

    // Build per-MB plan first.
    struct Plan {
        bis: [usize; 4],
        chroma: Vec<usize>,
        mode: Mode,
    }
    let mut plans: Vec<Plan> = Vec::new();
    for_each_mb(&layout, |mbx, mby| {
        let bxs = [mbx * 2, mbx * 2 + 1, mbx * 2, mbx * 2 + 1];
        let bys = [mby * 2, mby * 2, mby * 2 + 1, mby * 2 + 1];
        let bis: [usize; 4] = [
            layout.global_coded(0, bxs[0], bys[0]) as usize,
            layout.global_coded(0, bxs[1], bys[1]) as usize,
            layout.global_coded(0, bxs[2], bys[2]) as usize,
            layout.global_coded(0, bxs[3], bys[3]) as usize,
        ];
        let mode = bis
            .iter()
            .find(|&&bi| frame.bcoded[bi])
            .map(|&bi| frame.modes[bi])
            .unwrap_or(Mode::InterNoMv);
        let mut chroma = Vec::new();
        for pli in 1..3 {
            let cplane = &layout.planes[pli];
            let cnbw = cplane.nbw;
            let cnbh = cplane.nbh;
            let (cbx, cby, dxr, dyr) = match pf {
                PixelFormat::Yuv420 => (mbx, mby, 1u32, 1u32),
                PixelFormat::Yuv422 => (mbx, mby * 2, 1u32, 2u32),
                PixelFormat::Yuv444 | PixelFormat::Reserved => (mbx * 2, mby * 2, 2u32, 2u32),
            };
            for dy in 0..dyr {
                for dx in 0..dxr {
                    let x = cbx + dx;
                    let y = cby + dy;
                    if x < cnbw && y < cnbh {
                        chroma.push(layout.global_coded(pli, x, y) as usize);
                    }
                }
            }
        }
        plans.push(Plan { bis, chroma, mode });
        Ok(())
    })?;
    for plan in plans {
        let mb_mv: (i32, i32) = match plan.mode {
            Mode::InterMv | Mode::InterGoldenMv => {
                let mv = read_mv(br)?;
                if plan.mode == Mode::InterMv {
                    last_mv2 = last_mv;
                    last_mv = mv;
                }
                mv
            }
            Mode::InterMvLast => last_mv,
            Mode::InterMvLast2 => {
                let m = last_mv2;
                std::mem::swap(&mut last_mv, &mut last_mv2);
                m
            }
            Mode::InterMvFour => {
                let mut mvs4 = [(0i32, 0i32); 4];
                for i in 0..4 {
                    if frame.bcoded[plan.bis[i]] {
                        mvs4[i] = read_mv(br)?;
                    }
                }
                for i in 0..4 {
                    frame.mvs[plan.bis[i]] = mvs4[i];
                }
                last_mv2 = last_mv;
                last_mv = mvs4[3];
                let avg_x = mvs4.iter().map(|m| m.0).sum::<i32>();
                let avg_y = mvs4.iter().map(|m| m.1).sum::<i32>();
                (avg_x.div_euclid(4), avg_y.div_euclid(4))
            }
            _ => (0, 0),
        };
        if plan.mode != Mode::InterMvFour {
            for &bi in &plan.bis {
                frame.mvs[bi] = mb_mv;
            }
        }
        for bj in plan.chroma {
            frame.mvs[bj] = mb_mv;
        }
    }
    Ok(())
}

impl Decoder for TheoraDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        if packet.data.is_empty() {
            return Ok(());
        }
        // In-band header parsing: if the decoder was constructed without
        // extradata, fall through the 3 header packets before accepting data.
        if self.headers.is_none() {
            if self.ensure_headers_from_packet(&packet.data)? {
                // We cannot build `Headers` until we have all three; users
                // passing headers in-band are expected to use extradata
                // instead. Report the current limitation.
                return Err(Error::unsupported(
                    "Theora: in-band header parsing not implemented; pass Xiph-laced extradata",
                ));
            }
            return Err(Error::invalid(
                "Theora: frame packet received before headers",
            ));
        }
        self.decode_frame(&packet.data)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Crop three full-frame planes (top-down) to the picture region.
/// Returns `(planes, picture_width, picture_height)`.
fn crop_to_picture(
    id: &crate::headers::Identification,
    full: &[Vec<u8>; 3],
) -> (Vec<Vec<u8>>, u32, u32) {
    let pw = id.picw;
    let ph = id.pich;
    let mut out = Vec::with_capacity(3);
    for pli in 0..3 {
        let (frame_w, frame_h) =
            crate::coded_order::plane_pixel_dims(id.fmbw as u32, id.fmbh as u32, id.pf, pli);
        // Chroma offsets & dimensions scale with subsampling.
        let sx = if pli == 0 { 0 } else { id.pf.chroma_shift_x() };
        let sy = if pli == 0 { 0 } else { id.pf.chroma_shift_y() };
        let picx = id.picx >> sx;
        let picy = id.picy >> sy;
        let out_w = pw >> sx;
        let out_h = ph >> sy;
        let mut plane = Vec::with_capacity((out_w * out_h) as usize);
        // picy is from the BOTTOM of the frame, but our buffer is top-down.
        // The picture's bottom row lives at (frame_h - 1 - picy), its top row
        // at (frame_h - 1 - (picy + out_h - 1)) = (frame_h - picy - out_h).
        let top_row_in_full = frame_h.saturating_sub(picy + out_h);
        for row in 0..out_h {
            let src_row = (top_row_in_full + row) as usize;
            let src_off = src_row * frame_w as usize + picx as usize;
            plane.extend_from_slice(&full[pli][src_off..src_off + out_w as usize]);
        }
        out.push(plane);
    }
    (out, pw, ph)
}

fn build_video_planes(cropped: &[Vec<u8>], pw: u32, ph: u32, pf: PixelFormat) -> Vec<VideoPlane> {
    let sx = pf.chroma_shift_x();
    let sy = pf.chroma_shift_y();
    let y_stride = pw as usize;
    let c_w = pw >> sx;
    let _c_h = ph >> sy;
    vec![
        VideoPlane {
            stride: y_stride,
            data: cropped[0].clone(),
        },
        VideoPlane {
            stride: c_w as usize,
            data: cropped[1].clone(),
        },
        VideoPlane {
            stride: c_w as usize,
            data: cropped[2].clone(),
        },
    ]
}

fn pixel_format_to_core(pf: PixelFormat) -> CorePixelFormat {
    match pf {
        PixelFormat::Yuv420 | PixelFormat::Reserved => CorePixelFormat::Yuv420P,
        PixelFormat::Yuv422 => CorePixelFormat::Yuv422P,
        PixelFormat::Yuv444 => CorePixelFormat::Yuv444P,
    }
}

/// Build a `CodecParameters` for a Theora stream given the identification
/// header.
pub fn codec_parameters_from_identification(
    id: &crate::headers::Identification,
) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(id.picw);
    params.height = Some(id.pich);
    params.pixel_format = Some(id.pf.to_core());
    if id.frd > 0 {
        params.frame_rate = Some(Rational::new(id.frn as i64, id.frd as i64));
    }
    if id.nombr > 0 {
        params.bit_rate = Some(id.nombr as u64);
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_headers() {
        assert_eq!(
            classify_packet(&[0x80]).unwrap(),
            PacketKind::Identification
        );
        assert_eq!(classify_packet(&[0x81]).unwrap(), PacketKind::Comment);
        assert_eq!(classify_packet(&[0x82]).unwrap(), PacketKind::Setup);
    }

    #[test]
    fn classify_frames() {
        // bit 0 clear → frame; bit 1 clear → intra.
        assert_eq!(
            classify_packet(&[0b0000_0000]).unwrap(),
            PacketKind::Frame(FrameType::Intra)
        );
        // bit 0 clear, bit 1 set → inter.
        assert_eq!(
            classify_packet(&[0b0100_0000]).unwrap(),
            PacketKind::Frame(FrameType::Inter)
        );
    }

    #[test]
    fn rejects_empty() {
        assert!(classify_packet(&[]).is_err());
    }
}
