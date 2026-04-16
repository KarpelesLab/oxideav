//! Theora encoder (intra + P-frame).
//!
//! Strategy:
//!
//! * Identification + comment headers are constructed from the
//!   [`CodecParameters`] passed at build time.
//! * The setup header (loop filter limits, AC/DC scale tables, quant base
//!   matrices, quant ranges, 80 Huffman trees) is **embedded verbatim** from
//!   a known-good libtheora reference.
//! * Per intra frame we run forward DCT (`fdct8x8`) → integer dequant rounding
//!   → forward DC prediction → token RLE → Huffman encoding using the embedded
//!   setup's trees.
//! * Per inter (P) frame we run an SAD-based mode decision per macro-block
//!   with full-pel motion search inside `ME_RANGE`, considering modes
//!   {INTER_NOMV, INTRA, INTER_MV, INTER_MV_LAST, INTER_MV_LAST2,
//!   INTER_GOLDEN_NOMV, INTER_GOLDEN_MV}. The encoder emits SBPM/BCODED,
//!   the mode alphabet (MSCHEME=0 with the natural alphabet), the MV stream
//!   (Table 7.23 VLC), then DCT the residual against the chosen predictor and
//!   Huffman-codes it with the inter-frame table group.
//!
//! After encoding we always reconstruct the frame (apply the same
//! dequant/IDCT pipeline as the decoder) to update `prev_ref` (and
//! `golden_ref` on keyframes), keeping our reference store bit-exact for the
//! decoder to follow.
//!
//! GOP structure: a keyframe at frame 0 and every `keyint` frames thereafter
//! (default [`DEFAULT_KEYINT`]). Callers can override via
//! [`EncoderOptions::keyint`] + [`make_encoder_with_options`].
//!
//! Current limitations (future work):
//!   * Integer-pel motion estimation only (no sub-pel refinement).
//!   * No rate control: QI is fixed at [`DEFAULT_QI`].
//!   * 4-MV (`INTER_MV_FOUR`) mode is not produced.
//!   * Motion search is a brute-force full-pel scan within ±`ME_RANGE`.
//!
//! The encoder always picks HTI 0 for the Huffman group selectors and always
//! emits `MSCHEME=0` so the alphabet is the natural 0..7 ordering shipped with
//! the decoder.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat as CorePixelFormat,
    Result, TimeBase, VideoFrame,
};

use crate::coded_order::FrameLayout;
use crate::dct::{idct2d, INV_ZIGZAG};
use crate::encoder_huffman::{build_all, HuffCode, HuffTable};
use crate::fdct::fdct8x8;
use crate::headers::{parse_setup_header, PixelFormat as TheoraPixelFormat, Setup};
use crate::inter::{Mode, MODE_ALPHABETS};
use crate::quant::build_qmat;

/// Standard libtheora setup header.
const STANDARD_SETUP: &[u8] = include_bytes!("encoder_data/standard_setup.bin");

/// Quantisation index used for every block. 32 is mid-quality.
pub const DEFAULT_QI: u8 = 32;

/// Default keyframe interval (one I-frame every N frames).
pub const DEFAULT_KEYINT: u32 = 30;

/// Default maximum motion-vector search radius in luma pixels. Task spec
/// asks for ±16; we use 15 to stay within the MV-VLC ±31 half-pel range
/// (since our MV magnitudes encode as full_pel × 2).
pub const DEFAULT_ME_RANGE: i32 = 15;

/// SAD threshold (sum of absolute diffs over a 16x16 luma MB) below which an
/// MB is encoded as INTER_NOMV with all blocks marked uncoded (skip).
const SKIP_SAD_THRESHOLD: i32 = 384;

/// SAD improvement (over zero-MV) required for the encoder to invest the cost
/// of an INTER_MV codeword over INTER_NOMV.
const MV_GAIN_THRESHOLD: i32 = 256;

/// SAD-bias credit given to INTER_MV_LAST / INTER_MV_LAST2 matches over a
/// newly-coded INTER_MV. These modes cost no MV bits, so if the predictor is
/// nearly as good we save a whole Table 7.23 codeword.
const LAST_MV_BONUS: i32 = 128;

/// Extra cost charged against switching to the GOLDEN reference (each switch
/// introduces a RFI flip during DC prediction + breaks the last_mv chain).
const GOLDEN_PENALTY: i32 = 128;

/// Encoder tunable options exposed by [`make_encoder_with_options`].
#[derive(Clone, Debug)]
pub struct EncoderOptions {
    /// Fixed quantisation index (0..63). Higher = lower quality, smaller packets.
    pub qi: u8,
    /// Keyframe interval: emit an I-frame every `keyint` frames (≥ 1).
    pub keyint: u32,
    /// Full-pel motion-estimation search radius, in luma pixels.
    /// Clamped to `1..=15` internally (MV VLC codec range).
    pub me_range: i32,
    /// If true, consider INTER_GOLDEN_NOMV / INTER_GOLDEN_MV against the
    /// golden (last-keyframe) reference during mode decision.
    pub use_golden: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            qi: DEFAULT_QI,
            keyint: DEFAULT_KEYINT,
            me_range: DEFAULT_ME_RANGE,
            use_golden: true,
        }
    }
}

/// Build a Theora encoder for the given [`CodecParameters`], using
/// [`EncoderOptions::default`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_options(params, EncoderOptions::default())
}

/// Build a Theora encoder with the given tunable options.
pub fn make_encoder_with_options(
    params: &CodecParameters,
    opts: EncoderOptions,
) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("Theora encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("Theora encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(CorePixelFormat::Yuv420P);
    if pix != CorePixelFormat::Yuv420P {
        return Err(Error::unsupported(format!(
            "Theora encoder: only Yuv420P supported (got {pix:?})"
        )));
    }
    if width == 0 || height == 0 {
        return Err(Error::invalid("Theora encoder: zero frame dimensions"));
    }
    if opts.keyint == 0 {
        return Err(Error::invalid("Theora encoder: keyint must be >= 1"));
    }
    if opts.qi > 63 {
        return Err(Error::invalid("Theora encoder: qi must be <= 63"));
    }
    let me_range = opts.me_range.clamp(1, 15);

    let theora_pf = TheoraPixelFormat::Yuv420;
    let fmbw = width.div_ceil(16) as u16;
    let fmbh = height.div_ceil(16) as u16;
    let frame_w = fmbw as u32 * 16;
    let frame_h = fmbh as u32 * 16;

    let setup = parse_setup_header(STANDARD_SETUP)?;
    let huff_tables = build_all(&setup.huffs);

    let layout = FrameLayout::new(fmbw as u32, fmbh as u32, theora_pf);

    let frn = params.frame_rate.map(|r| r.num as u32).unwrap_or(24);
    let frd = params.frame_rate.map(|r| r.den as u32).unwrap_or(1);

    let id_packet = build_identification_packet(BuildIdParams {
        fmbw,
        fmbh,
        picw: width,
        pich: height,
        picx: 0,
        picy: 0,
        frn,
        frd,
        pf: theora_pf,
    });
    let comment_packet = build_comment_packet("oxideav-theora");
    let setup_packet = STANDARD_SETUP.to_vec();

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(crate::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(CorePixelFormat::Yuv420P);
    output_params.extradata = xiph_lace(&[&id_packet, &comment_packet, &setup_packet]);

    let time_base = params
        .frame_rate
        .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));

    Ok(Box::new(TheoraEncoder {
        output_params,
        width,
        height,
        frame_w,
        frame_h,
        layout,
        setup,
        huff_tables,
        time_base,
        pending: VecDeque::new(),
        eof: false,
        emitted_headers: false,
        id_packet,
        comment_packet,
        setup_packet,
        qi: opts.qi,
        keyint: opts.keyint,
        me_range,
        use_golden: opts.use_golden,
        frame_index: 0,
        prev_ref: None,
        golden_ref: None,
    }))
}

struct TheoraEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    frame_w: u32,
    frame_h: u32,
    layout: FrameLayout,
    setup: Setup,
    huff_tables: Vec<HuffTable>,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
    emitted_headers: bool,
    id_packet: Vec<u8>,
    comment_packet: Vec<u8>,
    setup_packet: Vec<u8>,
    qi: u8,
    keyint: u32,
    me_range: i32,
    use_golden: bool,
    frame_index: u32,
    /// Previous reconstructed frame (post-loop-filter pixel buffers).
    /// Each plane is stored TOP-DOWN at the coded frame size.
    prev_ref: Option<[Vec<u8>; 3]>,
    /// Golden reference; updated on every keyframe.
    golden_ref: Option<[Vec<u8>; 3]>,
}

impl Encoder for TheoraEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("Theora encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "Theora encoder: frame dimensions do not match encoder config",
            ));
        }
        if v.format != CorePixelFormat::Yuv420P {
            return Err(Error::invalid(format!(
                "Theora encoder: expected Yuv420P, got {:?}",
                v.format
            )));
        }
        if !self.emitted_headers {
            self.emit_header_packet(self.id_packet.clone(), v.pts);
            self.emit_header_packet(self.comment_packet.clone(), v.pts);
            self.emit_header_packet(self.setup_packet.clone(), v.pts);
            self.emitted_headers = true;
        }
        // Decide frame type.
        let is_keyframe = self.prev_ref.is_none() || self.frame_index % self.keyint == 0;
        let (data, recon) = if is_keyframe {
            self.encode_intra_frame(v)?
        } else {
            self.encode_inter_frame(v)?
        };
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = is_keyframe;
        self.pending.push_back(pkt);
        if is_keyframe {
            self.golden_ref = Some(recon.clone());
        }
        self.prev_ref = Some(recon);
        self.frame_index += 1;
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

impl TheoraEncoder {
    fn emit_header_packet(&mut self, data: Vec<u8>, pts: Option<i64>) {
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.flags.header = true;
        self.pending.push_back(pkt);
    }

    /// Encode an intra (key) frame and return the bitstream + reconstructed
    /// planes for use as the future reference.
    fn encode_intra_frame(&self, frame: &VideoFrame) -> Result<(Vec<u8>, [Vec<u8>; 3])> {
        let nbs = self.layout.nbs as usize;
        let mut coeffs = vec![0i32; nbs * 64];

        let qmats: [[i32; 64]; 3] = [
            build_qmat(&self.setup, 0, 0, self.qi),
            build_qmat(&self.setup, 0, 1, self.qi),
            build_qmat(&self.setup, 0, 2, self.qi),
        ];

        // Quantised coefficient table (zig-zag layout).
        for bi in 0..nbs {
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            let block = self.fetch_block_pixels(frame, pli, bx, by);
            let mut f = block;
            for v in f.iter_mut() {
                *v -= 128.0;
            }
            fdct8x8(&mut f);
            let mut quant_zz = [0i32; 64];
            for ci in 0..64 {
                let q = qmats[pli][ci].max(1);
                let raw = f[ci] * 4.0;
                let qc = (raw / q as f32).round() as i32;
                let qc = qc.clamp(-32768, 32767);
                let zzi = INV_ZIGZAG[ci];
                quant_zz[zzi] = qc;
            }
            for ti in 0..64 {
                coeffs[bi * 64 + ti] = quant_zz[ti];
            }
        }

        // Apply forward DC prediction (intra-only path: every block
        // contributes; rfi=0 throughout).
        let modes = vec![Mode::Intra; nbs];
        let bcoded = vec![true; nbs];
        let raw_dc = self.snapshot_dc(&coeffs);
        self.apply_forward_dc_prediction(&mut coeffs, &modes, &bcoded);

        // Bit-pack the frame.
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1); // data packet
        bw.write_bits(0, 1); // intra
        bw.write_bits(self.qi as u32, 6);
        bw.write_bits(0, 1); // NQIS extension flag
        bw.write_bits(0, 3); // reserved (intra only)

        self.encode_coefficients(&mut bw, &coeffs, &bcoded)?;

        let bitstream = bw.finish();

        // Reconstruct.
        let recon = self.reconstruct_frame(
            &raw_dc,
            &coeffs,
            &bcoded,
            &modes,
            &vec![(0i32, 0i32); nbs],
            None,
            None,
        );
        Ok((bitstream, recon))
    }

    /// Encode a P (inter) frame. Returns bitstream + reconstructed planes.
    fn encode_inter_frame(&self, frame: &VideoFrame) -> Result<(Vec<u8>, [Vec<u8>; 3])> {
        let prev = self
            .prev_ref
            .as_ref()
            .ok_or_else(|| Error::invalid("Theora encoder: P-frame without prior reference"))?;
        let nbs = self.layout.nbs as usize;

        // Per-MB decisions.
        let mb_decisions = self.decide_mb_modes(frame, prev);

        // Build per-block bcoded, modes, and MVs from MB decisions.
        let mut bcoded = vec![false; nbs];
        let mut modes = vec![Mode::InterNoMv; nbs];
        let mut mvs = vec![(0i32, 0i32); nbs];

        let pf = TheoraPixelFormat::Yuv420;
        for d in &mb_decisions {
            // Luma blocks.
            for &bi in &d.luma_bis {
                modes[bi] = d.mode;
                mvs[bi] = d.mv;
                bcoded[bi] = d.bcoded;
            }
            // Chroma blocks: derive from MB.
            for &bj in &d.chroma_bis {
                modes[bj] = d.mode;
                // Chroma MV: spec uses /2 round-towards-zero for 4:2:0.
                let (cmx, _) = crate::inter::chroma_mv_split(d.mv.0, pf, false);
                let (cmy, _) = crate::inter::chroma_mv_split(d.mv.1, pf, true);
                mvs[bj] = (cmx, cmy);
                bcoded[bj] = d.bcoded;
            }
        }

        // Build quantised coefficients for coded blocks. Inter qti = 1.
        let qmats_intra: [[i32; 64]; 3] = [
            build_qmat(&self.setup, 0, 0, self.qi),
            build_qmat(&self.setup, 0, 1, self.qi),
            build_qmat(&self.setup, 0, 2, self.qi),
        ];
        let qmats_inter: [[i32; 64]; 3] = [
            build_qmat(&self.setup, 1, 0, self.qi),
            build_qmat(&self.setup, 1, 1, self.qi),
            build_qmat(&self.setup, 1, 2, self.qi),
        ];

        let mut coeffs = vec![0i32; nbs * 64];
        for bi in 0..nbs {
            if !bcoded[bi] {
                continue;
            }
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            let block = self.fetch_block_pixels(frame, pli, bx, by);
            let mode = modes[bi];
            let qti = if mode == Mode::Intra { 0 } else { 1 };
            let qmat = if qti == 0 {
                &qmats_intra[pli]
            } else {
                &qmats_inter[pli]
            };

            let mut residual = [0.0f32; 64];
            if mode == Mode::Intra {
                for k in 0..64 {
                    residual[k] = block[k] - 128.0;
                }
            } else {
                // Build motion-compensated predictor in BOTTOM-UP row order to
                // match `fetch_block_pixels` orientation. Pick reference
                // per-mode (rfi=1 prev, rfi=2 golden).
                let ref_buf = match mode.rfi() {
                    2 => self.golden_ref.as_ref().unwrap_or(prev),
                    _ => prev,
                };
                let pred = self.predict_inter_block_bottom_up(ref_buf, pli, bx, by, mvs[bi]);
                for k in 0..64 {
                    residual[k] = block[k] - pred[k] as f32;
                }
            }

            fdct8x8(&mut residual);
            let mut quant_zz = [0i32; 64];
            for ci in 0..64 {
                let q = qmat[ci].max(1);
                let raw = residual[ci] * 4.0;
                let qc = (raw / q as f32).round() as i32;
                let qc = qc.clamp(-32768, 32767);
                let zzi = INV_ZIGZAG[ci];
                quant_zz[zzi] = qc;
            }
            for ti in 0..64 {
                coeffs[bi * 64 + ti] = quant_zz[ti];
            }
        }

        let raw_dc = self.snapshot_dc(&coeffs);
        self.apply_forward_dc_prediction(&mut coeffs, &modes, &bcoded);

        // ---- Bit-pack ----
        let mut bw = BitWriter::new();
        bw.write_bits(0, 1); // data packet
        bw.write_bits(1, 1); // inter
        bw.write_bits(self.qi as u32, 6);
        bw.write_bits(0, 1); // NQIS extension flag

        // SBPMs + per-block BCODED.
        write_inter_bcoded(&mut bw, &self.layout, &bcoded);

        // Mode header: pick MSCHEME=0, write 8-mode alphabet table mapping
        // mode_index -> rank. We choose alphabet[i]=Mode::from_index(i), so
        // every mode's rank in the alphabet equals its mode index.
        bw.write_bits(0, 3); // MSCHEME = 0
        for mode in 0..8u8 {
            // For mscheme=0, decoder reads 8 × 3-bit values mi; sets
            // alphabet[mi] = mode_for(loop_var). We want alphabet[i] = mode i,
            // so mi = i.
            bw.write_bits(mode as u32, 3);
        }
        // Per-MB mode codeword: only for MBs that have any coded block.
        write_inter_modes(&mut bw, &mb_decisions);

        // MV stream.
        bw.write_bits(0, 1); // MVMODE = 0 (VLC)
        write_inter_mvs(&mut bw, &mb_decisions);

        // QIIs: NQIS = 1 → no QII bits.

        // Coefficient encoding (only for coded blocks).
        self.encode_coefficients(&mut bw, &coeffs, &bcoded)?;

        let bitstream = bw.finish();

        let recon = self.reconstruct_frame(
            &raw_dc,
            &coeffs,
            &bcoded,
            &modes,
            &mvs,
            Some(prev),
            self.golden_ref.as_ref(),
        );
        Ok((bitstream, recon))
    }

    fn snapshot_dc(&self, coeffs: &[i32]) -> Vec<i32> {
        let nbs = self.layout.nbs as usize;
        let mut out = vec![0i32; nbs];
        for bi in 0..nbs {
            out[bi] = coeffs[bi * 64];
        }
        out
    }

    /// Read 8x8 block of luma/chroma samples into a buffer indexed in
    /// **bottom-up** row order: `out[ri * 8 + ci]` is row `ri` of the block
    /// counted from the BOTTOM in coded-frame coordinates.
    fn fetch_block_pixels(&self, frame: &VideoFrame, pli: usize, bx: u32, by: u32) -> [f32; 64] {
        let plane = &frame.planes[pli];
        let stride = plane.stride;
        let (sw, sh) = match pli {
            0 => (frame.width as i32, frame.height as i32),
            _ => ((frame.width / 2) as i32, (frame.height / 2) as i32),
        };
        let (_cw, ch) = match pli {
            0 => (self.frame_w as i32, self.frame_h as i32),
            _ => ((self.frame_w / 2) as i32, (self.frame_h / 2) as i32),
        };
        let bx_px = (bx * 8) as i32;
        let by_px_bottom = (by * 8) as i32;
        let top_row_coded = ch - 1 - (by_px_bottom + 7);
        let pic_top_offset = ch - sh;

        let mut out = [0.0f32; 64];
        for ri in 0..8i32 {
            let coded_y_top = top_row_coded + (7 - ri);
            let src_y = (coded_y_top - pic_top_offset).clamp(0, sh - 1);
            for ci in 0..8i32 {
                let src_x = (bx_px + ci).clamp(0, sw - 1);
                let idx = src_y as usize * stride + src_x as usize;
                let p = if idx < plane.data.len() {
                    plane.data[idx] as f32
                } else {
                    128.0
                };
                out[ri as usize * 8 + ci as usize] = p;
            }
        }
        out
    }

    /// Build a motion-compensated predictor for an inter block, output in
    /// BOTTOM-UP row order (matches `fetch_block_pixels`). The reference
    /// buffer is in TOP-DOWN row order at the coded plane size.
    fn predict_inter_block_bottom_up(
        &self,
        refs: &[Vec<u8>; 3],
        pli: usize,
        bx: u32,
        by: u32,
        mv: (i32, i32),
    ) -> [i32; 64] {
        let plane = &self.layout.planes[pli];
        let pw = (plane.nbw * 8) as i32;
        let ph = (plane.nbh * 8) as i32;
        let bx_px = (bx * 8) as i32;
        let by_px_bottom = (by * 8) as i32;
        // top-left in top-down coords
        let bx_top = bx_px;
        let by_top = ph - 1 - (by_px_bottom + 7);
        let pf = TheoraPixelFormat::Yuv420;
        let (sub_x, sub_y) = if pli == 0 {
            (false, false)
        } else {
            (pf.chroma_shift_x() == 1, pf.chroma_shift_y() == 1)
        };
        let mut pred_top_down = [0i32; 64];
        crate::inter::motion_compensate(
            &refs[pli],
            pw,
            ph,
            bx_top,
            by_top,
            mv.0,
            mv.1,
            sub_x,
            sub_y,
            &mut pred_top_down,
        );
        // Flip vertically into bottom-up order.
        let mut out = [0i32; 64];
        for ry in 0..8 {
            for rx in 0..8 {
                out[(7 - ry) * 8 + rx] = pred_top_down[ry * 8 + rx];
            }
        }
        out
    }

    /// Sum of absolute differences for a single 8x8 luma block residual at MV
    /// `mv` (luma half-pel units; here we always pass full-pel × 2).
    fn block_sad(
        &self,
        frame: &VideoFrame,
        prev: &[Vec<u8>; 3],
        pli: usize,
        bx: u32,
        by: u32,
        mv: (i32, i32),
    ) -> i32 {
        let src = self.fetch_block_pixels(frame, pli, bx, by);
        let pred = self.predict_inter_block_bottom_up(prev, pli, bx, by, mv);
        let mut s = 0i32;
        for k in 0..64 {
            s += (src[k] as i32 - pred[k]).abs();
        }
        s
    }

    /// Sum of absolute differences for the four luma blocks of an MB at the
    /// given MV.
    fn mb_sad(
        &self,
        frame: &VideoFrame,
        prev: &[Vec<u8>; 3],
        bxs: [u32; 4],
        bys: [u32; 4],
        mv: (i32, i32),
    ) -> i32 {
        let mut s = 0i32;
        for i in 0..4 {
            s += self.block_sad(frame, prev, 0, bxs[i], bys[i], mv);
        }
        s
    }

    /// Sum of absolute deviations of an MB's source from its mean (proxy for
    /// intra coding cost; high value → block has lots of detail).
    fn mb_sad_intra(&self, frame: &VideoFrame, bxs: [u32; 4], bys: [u32; 4]) -> i32 {
        let mut s = 0i32;
        for i in 0..4 {
            let src = self.fetch_block_pixels(frame, 0, bxs[i], bys[i]);
            // Mean.
            let mut m = 0.0f32;
            for k in 0..64 {
                m += src[k];
            }
            m /= 64.0;
            for k in 0..64 {
                s += (src[k] - m).abs() as i32;
            }
        }
        s
    }

    fn decide_mb_modes(&self, frame: &VideoFrame, prev: &[Vec<u8>; 3]) -> Vec<MbDecision> {
        let layout = &self.layout;
        let plane0 = &layout.planes[0];
        let mbw = plane0.nbw / 2;
        let mbh = plane0.nbh / 2;
        let sbw = plane0.nbw.div_ceil(4);
        let sbh = plane0.nbh.div_ceil(4);

        // LAST-MV tracking must walk MBs in *the same* coded order the
        // decoder uses (spec §7.3.5): only MBs that actually transmit a new
        // MV (InterMv / InterGoldenMv / InterMvFour) update last_mv; the LAST
        // / LAST2 modes reuse previously emitted MVs. Mirror the decoder here.
        let mut last_mv = (0i32, 0i32);
        let mut last_mv2 = (0i32, 0i32);

        let mut out = Vec::with_capacity((mbw * mbh) as usize);
        for sby in 0..sbh {
            for sbx in 0..sbw {
                for &(dx, dy) in &MB_HILBERT {
                    let mbx = sbx * 2 + dx;
                    let mby = sby * 2 + dy;
                    if mbx >= mbw || mby >= mbh {
                        continue;
                    }
                    let bxs = [mbx * 2, mbx * 2 + 1, mbx * 2, mbx * 2 + 1];
                    let bys = [mby * 2, mby * 2, mby * 2 + 1, mby * 2 + 1];
                    let bis: [usize; 4] = [
                        layout.global_coded(0, bxs[0], bys[0]) as usize,
                        layout.global_coded(0, bxs[1], bys[1]) as usize,
                        layout.global_coded(0, bxs[2], bys[2]) as usize,
                        layout.global_coded(0, bxs[3], bys[3]) as usize,
                    ];
                    // Chroma blocks (4:2:0: one 8x8 per chroma plane per MB).
                    let mut chroma_bis = Vec::with_capacity(2);
                    for pli in 1..3 {
                        let cplane = &layout.planes[pli];
                        if mbx < cplane.nbw && mby < cplane.nbh {
                            chroma_bis.push(layout.global_coded(pli, mbx, mby) as usize);
                        }
                    }

                    // Zero-MV reference SAD.
                    let sad_zero = self.mb_sad(frame, prev, bxs, bys, (0, 0));

                    // Skip (uncoded) heuristic: very small residual against
                    // LAST at (0,0) — the frame barely changed here.
                    if sad_zero <= SKIP_SAD_THRESHOLD {
                        out.push(MbDecision {
                            mbx,
                            mby,
                            mode: Mode::InterNoMv,
                            mv: (0, 0),
                            bcoded: false,
                            luma_bis: bis,
                            chroma_bis,
                        });
                        continue;
                    }

                    // INTRA cost (mean-absolute-deviation proxy).
                    let intra_cost = self.mb_sad_intra(frame, bxs, bys);

                    // Full-pel motion search against LAST within ±me_range.
                    // Encoded MV is in half-pel units: full_pel × 2.
                    let mut best_mv = (0i32, 0i32);
                    let mut best_sad = sad_zero;
                    let r = self.me_range;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let mv = (dx * 2, dy * 2);
                            if mv == (0, 0) {
                                continue;
                            }
                            let sad = self.mb_sad(frame, prev, bxs, bys, mv);
                            if sad < best_sad {
                                best_sad = sad;
                                best_mv = mv;
                            }
                        }
                    }

                    // LAST / LAST2 candidates: evaluate SAD at the two prior
                    // MVs. If either beats INTER_NOMV and is close to the
                    // best-found MV, prefer them (no MV bits).
                    let sad_last = if last_mv != (0, 0) {
                        self.mb_sad(frame, prev, bxs, bys, last_mv)
                    } else {
                        i32::MAX
                    };
                    let sad_last2 = if last_mv2 != (0, 0) && last_mv2 != last_mv {
                        self.mb_sad(frame, prev, bxs, bys, last_mv2)
                    } else {
                        i32::MAX
                    };

                    // GOLDEN candidates.
                    let golden_ref = if self.use_golden {
                        self.golden_ref.as_ref()
                    } else {
                        None
                    };
                    let (best_gmv, best_gsad) = if let Some(golden) = golden_ref {
                        let sad_gzero = self.mb_sad(frame, golden, bxs, bys, (0, 0));
                        let mut g_best_mv = (0i32, 0i32);
                        let mut g_best_sad = sad_gzero;
                        // Coarser search grid for golden (step 2) — it matters
                        // less for compression.
                        let step = 2i32;
                        let mut dy = -r;
                        while dy <= r {
                            let mut dx = -r;
                            while dx <= r {
                                let mv = (dx * 2, dy * 2);
                                if mv != (0, 0) {
                                    let sad = self.mb_sad(frame, golden, bxs, bys, mv);
                                    if sad < g_best_sad {
                                        g_best_sad = sad;
                                        g_best_mv = mv;
                                    }
                                }
                                dx += step;
                            }
                            dy += step;
                        }
                        (g_best_mv, g_best_sad)
                    } else {
                        ((0, 0), i32::MAX)
                    };

                    // Compose candidate scoreboard. Lower is better. Each
                    // candidate is (score, mode, mv, updates_last).
                    //
                    //   INTER_NOMV     : sad_zero                                   (0 MV bits)
                    //   INTER_MV       : best_sad + MV_GAIN_THRESHOLD                (two MV VLCs)
                    //   INTER_MV_LAST  : sad_last - LAST_MV_BONUS                    (rank bits only)
                    //   INTER_MV_LAST2 : sad_last2 - LAST_MV_BONUS/2                 (rank bits only, rarer)
                    //   INTER_GOLDEN_NOMV : sad_g0 + GOLDEN_PENALTY                 (rank bits only)
                    //   INTER_GOLDEN_MV : best_gsad + GOLDEN_PENALTY + MV_GAIN_THRESHOLD
                    //   INTRA          : intra_cost                                  (plain DCT bits)
                    let no_mv_score = sad_zero;
                    let mv_score = best_sad + MV_GAIN_THRESHOLD;
                    let last_score = sad_last.saturating_sub(LAST_MV_BONUS);
                    let last2_score = sad_last2.saturating_sub(LAST_MV_BONUS / 2);
                    let intra_score = intra_cost;
                    let gnm_score = best_gsad.saturating_add(GOLDEN_PENALTY).saturating_add(
                        if best_gmv == (0, 0) {
                            0
                        } else {
                            MV_GAIN_THRESHOLD
                        },
                    );

                    #[derive(Clone, Copy)]
                    struct Cand {
                        score: i32,
                        mode: Mode,
                        mv: (i32, i32),
                    }
                    let cands = [
                        Cand {
                            score: no_mv_score,
                            mode: Mode::InterNoMv,
                            mv: (0, 0),
                        },
                        Cand {
                            score: mv_score,
                            mode: Mode::InterMv,
                            mv: best_mv,
                        },
                        Cand {
                            score: last_score,
                            mode: Mode::InterMvLast,
                            mv: last_mv,
                        },
                        Cand {
                            score: last2_score,
                            mode: Mode::InterMvLast2,
                            mv: last_mv2,
                        },
                        Cand {
                            score: intra_score,
                            mode: Mode::Intra,
                            mv: (0, 0),
                        },
                        Cand {
                            score: gnm_score,
                            mode: if best_gmv == (0, 0) {
                                Mode::InterGoldenNoMv
                            } else {
                                Mode::InterGoldenMv
                            },
                            mv: best_gmv,
                        },
                    ];
                    let mut best_cand = cands[0];
                    for c in &cands[1..] {
                        if c.score < best_cand.score {
                            best_cand = *c;
                        }
                    }

                    // Update LAST / LAST2 only on modes that transmit a new MV
                    // (spec §7.3.5.3 Table 7.33).
                    match best_cand.mode {
                        Mode::InterMv | Mode::InterGoldenMv => {
                            last_mv2 = last_mv;
                            last_mv = best_cand.mv;
                        }
                        Mode::InterMvLast2 => {
                            std::mem::swap(&mut last_mv, &mut last_mv2);
                        }
                        _ => {}
                    }

                    out.push(MbDecision {
                        mbx,
                        mby,
                        mode: best_cand.mode,
                        mv: best_cand.mv,
                        bcoded: true,
                        luma_bis: bis,
                        chroma_bis,
                    });
                }
            }
        }
        out
    }

    /// Apply forward DC prediction. After this, `coeffs[bi*64]` holds
    /// `q_dc - predicted_dc`. Mirrors `block::undo_dc_prediction`'s structure.
    fn apply_forward_dc_prediction(&self, coeffs: &mut [i32], modes: &[Mode], bcoded: &[bool]) {
        for pli in 0..3 {
            let mut last_dc = [0i32; 3];
            let plane = &self.layout.planes[pli];
            let nbw = plane.nbw;
            let nbh = plane.nbh;
            let mut q_dc = vec![0i32; (nbw * nbh) as usize];
            for by in 0..nbh {
                for bx in 0..nbw {
                    let bi = self.layout.global_coded(pli, bx, by) as usize;
                    q_dc[(by * nbw + bx) as usize] = coeffs[bi * 64];
                }
            }
            let q_at = |bx: u32, by: u32| -> i32 { q_dc[(by * nbw + bx) as usize] };
            for by in 0..nbh {
                for bx in 0..nbw {
                    let bi = self.layout.global_coded(pli, bx, by) as usize;
                    if !bcoded[bi] {
                        continue;
                    }
                    let rfi = modes[bi].rfi() as usize;
                    let neighbour_ok = |x: u32, y: u32| -> bool {
                        let j = self.layout.global_coded(pli, x, y) as usize;
                        bcoded[j] && modes[j].rfi() as usize == rfi
                    };
                    let mut p = [false; 4];
                    let mut pv = [0i32; 4];
                    if bx > 0 && neighbour_ok(bx - 1, by) {
                        p[0] = true;
                        pv[0] = q_at(bx - 1, by);
                    }
                    if bx > 0 && by > 0 && neighbour_ok(bx - 1, by - 1) {
                        p[1] = true;
                        pv[1] = q_at(bx - 1, by - 1);
                    }
                    if by > 0 && neighbour_ok(bx, by - 1) {
                        p[2] = true;
                        pv[2] = q_at(bx, by - 1);
                    }
                    if bx + 1 < nbw && by > 0 && neighbour_ok(bx + 1, by - 1) {
                        p[3] = true;
                        pv[3] = q_at(bx + 1, by - 1);
                    }
                    let dcpred = if !p.iter().any(|&b| b) {
                        last_dc[rfi]
                    } else {
                        let (weights, pdiv) = weights_for(p);
                        let mut sum: i32 = 0;
                        for i in 0..4 {
                            if p[i] {
                                sum += weights[i] * pv[i];
                            }
                        }
                        let mut pred = if sum < 0 {
                            -((-sum) / pdiv)
                        } else {
                            sum / pdiv
                        };
                        if p[0] && p[1] && p[2] {
                            if (pred - pv[2]).abs() > 128 {
                                pred = pv[2];
                            } else if (pred - pv[0]).abs() > 128 {
                                pred = pv[0];
                            } else if (pred - pv[1]).abs() > 128 {
                                pred = pv[1];
                            }
                        }
                        pred
                    };
                    let q = q_at(bx, by);
                    let residual = ((q - dcpred) as i16) as i32;
                    coeffs[bi * 64] = residual;
                    last_dc[rfi] = q;
                }
            }
        }
    }

    /// Reconstruct a frame from quantised coefficients (raw DC, BEFORE
    /// forward DC prediction was applied), using the same dequant + IDCT +
    /// loop-filter pipeline as the decoder.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_frame(
        &self,
        raw_dc: &[i32],
        coeffs_after_pred: &[i32],
        bcoded: &[bool],
        modes: &[Mode],
        mvs: &[(i32, i32)],
        prev_ref: Option<&[Vec<u8>; 3]>,
        golden_ref: Option<&[Vec<u8>; 3]>,
    ) -> [Vec<u8>; 3] {
        let id = &self.layout;
        let pf = TheoraPixelFormat::Yuv420;
        let nbs = self.layout.nbs as usize;

        // Restore the original quantised coefficients (un-predict DC) into
        // a working copy, then dequant + IDCT.
        let mut coeffs = coeffs_after_pred.to_vec();
        for bi in 0..nbs {
            coeffs[bi * 64] = raw_dc[bi];
        }

        let setup = &self.setup;
        let mut dc_qmats: [[[i32; 64]; 3]; 2] = [[[0i32; 64]; 3]; 2];
        let mut ac_qmats: [[[i32; 64]; 3]; 2] = [[[0i32; 64]; 3]; 2];
        for qti in 0..2 {
            for pli in 0..3 {
                dc_qmats[qti][pli] = build_qmat(setup, qti, pli, self.qi);
                ac_qmats[qti][pli] = build_qmat(setup, qti, pli, self.qi);
            }
        }

        let mut planes_out: [Vec<u8>; 3] = Default::default();
        for pli in 0..3 {
            let plane = &id.planes[pli];
            let pw = (plane.nbw * 8) as usize;
            let ph = (plane.nbh * 8) as usize;
            planes_out[pli] = vec![0u8; pw * ph];
        }

        // For inter frames pre-fill from prev_ref to handle uncoded blocks.
        if let Some(prev) = prev_ref {
            for pli in 0..3 {
                planes_out[pli].copy_from_slice(&prev[pli]);
            }
        }

        for bi in 0..nbs {
            if !bcoded[bi] {
                continue;
            }
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            let plane = &self.layout.planes[pli];
            let pw = (plane.nbw * 8) as i32;
            let ph = (plane.nbh * 8) as i32;
            let mode = modes[bi];
            let qti = if mode == Mode::Intra { 0 } else { 1 };

            // Dequant.
            let mut dqc = [0i32; 64];
            dqc[0] = ((coeffs[bi * 64] * dc_qmats[qti][pli][0]) as i16) as i32;
            for ci in 1..64 {
                let zzi = INV_ZIGZAG[ci];
                let c = coeffs[bi * 64 + zzi] * ac_qmats[qti][pli][ci];
                dqc[ci] = (c as i16) as i32;
            }
            let res = idct2d(&dqc);

            // Predictor.
            let mut pred_top_down = [128i32; 64];
            if mode != Mode::Intra {
                let rfi = mode.rfi();
                let refbuf = match rfi {
                    1 => prev_ref,
                    2 => golden_ref,
                    _ => None,
                };
                if let Some(rb) = refbuf {
                    let bx_px = (bx * 8) as i32;
                    let by_px_bottom = (by * 8) as i32;
                    let bx_top = bx_px;
                    let by_top = ph - 1 - (by_px_bottom + 7);
                    let (sub_x, sub_y) = if pli == 0 {
                        (false, false)
                    } else {
                        (pf.chroma_shift_x() == 1, pf.chroma_shift_y() == 1)
                    };
                    crate::inter::motion_compensate(
                        &rb[pli],
                        pw,
                        ph,
                        bx_top,
                        by_top,
                        mvs[bi].0,
                        mvs[bi].1,
                        sub_x,
                        sub_y,
                        &mut pred_top_down,
                    );
                }
            }

            // Write block back to the top-down output buffer.
            let bx_px = bx * 8;
            let by_px_bottom = by * 8;
            let pw_u = pw as u32;
            let ph_u = ph as u32;
            let out = &mut planes_out[pli];
            for ry_top in 0..8u32 {
                let py = ph_u - 1 - (by_px_bottom + 7 - ry_top);
                let ry_bot = 7 - ry_top;
                for rx in 0..8u32 {
                    let px = bx_px + rx;
                    let predv = if mode == Mode::Intra {
                        pred_top_down[(ry_bot as usize) * 8 + rx as usize]
                    } else {
                        pred_top_down[(ry_top as usize) * 8 + rx as usize]
                    };
                    let resv = res[(ry_bot as usize) * 8 + rx as usize];
                    let p = resv + predv;
                    let clamped = p.clamp(0, 255) as u8;
                    out[(py * pw_u + px) as usize] = clamped;
                }
            }
        }

        // Loop filter (mirroring decoder).
        let l = self.setup.lflims[self.qi as usize] as i32;
        if l != 0 {
            apply_loop_filter(&self.layout, &mut planes_out, bcoded, l);
        }

        planes_out
    }

    /// Encode all DCT coefficients per spec §7.7.3.
    fn encode_coefficients(
        &self,
        bw: &mut BitWriter,
        coeffs: &[i32],
        bcoded: &[bool],
    ) -> Result<()> {
        let nbs = self.layout.nbs as usize;
        let nlbs = (self.layout.planes[0].nbw * self.layout.planes[0].nbh) as usize;

        let mut programs: Vec<Vec<TokenAtTi>> = Vec::with_capacity(nbs);
        for bi in 0..nbs {
            if bcoded[bi] {
                programs.push(build_block_token_program(&coeffs[bi * 64..bi * 64 + 64]));
            } else {
                programs.push(Vec::new());
            }
        }

        let hti_l: u8 = 0;
        let hti_c: u8 = 0;

        let mut prog_idx = vec![0usize; nbs];
        for ti in 0u8..64 {
            if ti == 0 || ti == 1 {
                bw.write_bits(hti_l as u32, 4);
                bw.write_bits(hti_c as u32, 4);
            }
            let hg = huffman_group_for_ti(ti) as usize;
            let table_idx_l = 16 * hg + hti_l as usize;
            let table_idx_c = 16 * hg + hti_c as usize;
            for bi in 0..nbs {
                if !bcoded[bi] {
                    continue;
                }
                if prog_idx[bi] >= programs[bi].len() {
                    continue;
                }
                let entry = &programs[bi][prog_idx[bi]];
                if entry.ti_pos != ti {
                    continue;
                }
                let tbl = if bi < nlbs {
                    &self.huff_tables[table_idx_l]
                } else {
                    &self.huff_tables[table_idx_c]
                };
                emit_token(bw, tbl, entry)?;
                prog_idx[bi] += 1;
            }
        }
        Ok(())
    }
}

/// Per-MB encoding decision.
#[derive(Clone, Debug)]
struct MbDecision {
    #[allow(dead_code)]
    mbx: u32,
    #[allow(dead_code)]
    mby: u32,
    mode: Mode,
    /// Luma motion vector in half-pel units (stored as encoded).
    mv: (i32, i32),
    /// True if the MB's blocks are coded (i.e. residuals get DCT-encoded).
    bcoded: bool,
    luma_bis: [usize; 4],
    chroma_bis: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct TokenAtTi {
    token: u8,
    extra_bits: u32,
    extra_len: u8,
    ti_pos: u8,
}

fn build_block_token_program(zz_coeffs: &[i32]) -> Vec<TokenAtTi> {
    debug_assert_eq!(zz_coeffs.len(), 64);
    let mut last_nz = -1i32;
    for ti in 0..64usize {
        if zz_coeffs[ti] != 0 {
            last_nz = ti as i32;
        }
    }
    let mut out: Vec<TokenAtTi> = Vec::new();
    if last_nz < 0 {
        out.push(TokenAtTi {
            token: 0,
            extra_bits: 0,
            extra_len: 0,
            ti_pos: 0,
        });
        return out;
    }
    let mut ti = 0usize;
    while ti as i32 <= last_nz {
        let v = zz_coeffs[ti];
        if v == 0 {
            let mut run = 1usize;
            while ti + run <= last_nz as usize && zz_coeffs[ti + run] == 0 {
                run += 1;
            }
            if run <= 8 {
                out.push(TokenAtTi {
                    token: 7,
                    extra_bits: (run - 1) as u32,
                    extra_len: 3,
                    ti_pos: ti as u8,
                });
            } else {
                out.push(TokenAtTi {
                    token: 8,
                    extra_bits: (run - 1) as u32,
                    extra_len: 6,
                    ti_pos: ti as u8,
                });
            }
            ti += run;
        } else {
            let entry = encode_value_token(v, ti as u8);
            out.push(entry);
            ti += 1;
        }
    }
    if (last_nz as usize) + 1 < 64 {
        out.push(TokenAtTi {
            token: 0,
            extra_bits: 0,
            extra_len: 0,
            ti_pos: (last_nz + 1) as u8,
        });
    }
    out
}

fn encode_value_token(v: i32, ti: u8) -> TokenAtTi {
    let abs = v.unsigned_abs() as i32;
    let neg = v < 0;
    if abs == 1 {
        return TokenAtTi {
            token: if neg { 10 } else { 9 },
            extra_bits: 0,
            extra_len: 0,
            ti_pos: ti,
        };
    }
    if abs == 2 {
        return TokenAtTi {
            token: if neg { 12 } else { 11 },
            extra_bits: 0,
            extra_len: 0,
            ti_pos: ti,
        };
    }
    if (3..=6).contains(&abs) {
        let token = (abs + 10) as u8;
        return TokenAtTi {
            token,
            extra_bits: u32::from(neg),
            extra_len: 1,
            ti_pos: ti,
        };
    }
    if (7..=8).contains(&abs) {
        let mag_off = abs - 7;
        return TokenAtTi {
            token: 17,
            extra_bits: (u32::from(neg) << 1) | mag_off as u32,
            extra_len: 2,
            ti_pos: ti,
        };
    }
    if (9..=12).contains(&abs) {
        let mag_off = abs - 9;
        return TokenAtTi {
            token: 18,
            extra_bits: (u32::from(neg) << 2) | mag_off as u32,
            extra_len: 3,
            ti_pos: ti,
        };
    }
    if (13..=20).contains(&abs) {
        let mag_off = abs - 13;
        return TokenAtTi {
            token: 19,
            extra_bits: (u32::from(neg) << 3) | mag_off as u32,
            extra_len: 4,
            ti_pos: ti,
        };
    }
    if (21..=36).contains(&abs) {
        let mag_off = abs - 21;
        return TokenAtTi {
            token: 20,
            extra_bits: (u32::from(neg) << 4) | mag_off as u32,
            extra_len: 5,
            ti_pos: ti,
        };
    }
    if (37..=68).contains(&abs) {
        let mag_off = abs - 37;
        return TokenAtTi {
            token: 21,
            extra_bits: (u32::from(neg) << 5) | mag_off as u32,
            extra_len: 6,
            ti_pos: ti,
        };
    }
    let abs = abs.min(580);
    let mag_off = abs - 69;
    TokenAtTi {
        token: 22,
        extra_bits: (u32::from(neg) << 9) | mag_off as u32,
        extra_len: 10,
        ti_pos: ti,
    }
}

fn emit_token(bw: &mut BitWriter, tbl: &HuffTable, entry: &TokenAtTi) -> Result<()> {
    let hc: HuffCode = tbl[entry.token as usize];
    if hc.len == 0 {
        return Err(Error::invalid(format!(
            "Theora encoder: token {} not in current Huffman table",
            entry.token
        )));
    }
    bw.write_bits(hc.code, hc.len as u32);
    if entry.extra_len > 0 {
        bw.write_bits(entry.extra_bits, entry.extra_len as u32);
    }
    Ok(())
}

fn huffman_group_for_ti(ti: u8) -> u8 {
    match ti {
        0 => 0,
        1..=5 => 1,
        6..=14 => 2,
        15..=27 => 3,
        _ => 4,
    }
}

/// Same DC-prediction weights as the decoder (mirrors `block::weights_for`).
fn weights_for(p: [bool; 4]) -> ([i32; 4], i32) {
    let key = (p[0] as u8) | ((p[1] as u8) << 1) | ((p[2] as u8) << 2) | ((p[3] as u8) << 3);
    match key {
        0b0000 => ([0, 0, 0, 0], 1),
        0b0001 => ([1, 0, 0, 0], 1),
        0b0010 => ([0, 1, 0, 0], 1),
        0b0011 => ([1, 0, 0, 0], 1),
        0b0100 => ([0, 0, 1, 0], 1),
        0b0101 => ([1, 0, 1, 0], 2),
        0b0110 => ([0, 0, 1, 0], 1),
        0b0111 => ([29, -26, 29, 0], 32),
        0b1000 => ([0, 0, 0, 1], 1),
        0b1001 => ([75, 0, 0, 53], 128),
        0b1010 => ([0, 1, 0, 1], 2),
        0b1011 => ([75, 0, 0, 53], 128),
        0b1100 => ([0, 0, 1, 0], 1),
        0b1101 => ([75, 0, 0, 53], 128),
        0b1110 => ([0, 3, 10, 3], 16),
        0b1111 => ([29, -26, 29, 0], 32),
        _ => ([0, 0, 0, 0], 1),
    }
}

// ---- inter-frame structural encoding (SBPM/BCODED, modes, MVs) ----------

const MB_HILBERT: [(u32, u32); 4] = [(0, 0), (0, 1), (1, 1), (1, 0)];

/// Iterate super-blocks across the three planes in coded order, yielding for
/// each the in-coded-order list of global block indices. Mirrors the
/// decoder's `SbCodedIter`.
fn for_each_sb<F: FnMut(usize, &[usize])>(layout: &FrameLayout, mut f: F) {
    let mut sb_idx = 0usize;
    for pli in 0..3 {
        let plane = &layout.planes[pli];
        let sbw = plane.nbw.div_ceil(4);
        let sbh = plane.nbh.div_ceil(4);
        for sby in 0..sbh {
            for sbx in 0..sbw {
                let mut blocks = Vec::with_capacity(16);
                for &(dx, dy) in &crate::coded_order::HILBERT_XY {
                    let bx = sbx * 4 + dx as u32;
                    let by = sby * 4 + dy as u32;
                    if bx < plane.nbw && by < plane.nbh {
                        blocks.push(layout.global_coded(pli, bx, by) as usize);
                    }
                }
                f(sb_idx, &blocks);
                sb_idx += 1;
            }
        }
    }
}

fn write_inter_bcoded(bw: &mut BitWriter, layout: &FrameLayout, bcoded: &[bool]) {
    // Determine SB_PARTIAL[]: 1 = mixed, 0 = uniform.
    let mut sb_partial: Vec<bool> = Vec::new();
    let mut sb_full: Vec<bool> = Vec::new();
    let mut partial_blocks: Vec<bool> = Vec::new();
    for_each_sb(layout, |_idx, blocks| {
        let v0 = bcoded[blocks[0]];
        let uniform = blocks.iter().all(|&bi| bcoded[bi] == v0);
        if uniform {
            sb_partial.push(false);
            sb_full.push(v0);
        } else {
            sb_partial.push(true);
            for &bi in blocks {
                partial_blocks.push(bcoded[bi]);
            }
        }
    });
    write_long_run_bitstring(bw, &sb_partial);
    if !sb_full.is_empty() {
        write_long_run_bitstring(bw, &sb_full);
    }
    if !partial_blocks.is_empty() {
        write_short_run_bitstring(bw, &partial_blocks);
    }
}

/// Write per-MB modes in coded order. Only MBs with at least one coded block
/// emit a codeword. We use the natural alphabet (alphabet[i] = mode i), so
/// the rank of each mode equals its mode index.
fn write_inter_modes(bw: &mut BitWriter, decisions: &[MbDecision]) {
    for d in decisions {
        if !d.bcoded {
            continue; // implicit INTER_NOMV
        }
        let rank = d.mode as u8;
        // Codeword: r leading 1-bits then a 0 (for ranks 0..6); rank 7 = seven 1s.
        if rank == 7 {
            for _ in 0..7 {
                bw.write_bits(1, 1);
            }
        } else {
            for _ in 0..rank {
                bw.write_bits(1, 1);
            }
            bw.write_bits(0, 1);
        }
    }
}

/// Write the MV stream (MVMODE=0, VLC). Mirrors `decode_mv_component_vlc` /
/// `read_inter_mvs` in the decoder. For each MB:
///   - InterMv / InterGoldenMv: emit one (x,y).
///   - InterMvLast / Last2: nothing.
///   - InterMvFour: emit one MV per coded luma block.
///   - InterNoMv / Intra / GoldenNoMv: nothing.
fn write_inter_mvs(bw: &mut BitWriter, decisions: &[MbDecision]) {
    for d in decisions {
        if !d.bcoded {
            continue;
        }
        match d.mode {
            Mode::InterMv | Mode::InterGoldenMv => {
                write_mv_component_vlc(bw, d.mv.0);
                write_mv_component_vlc(bw, d.mv.1);
            }
            Mode::InterMvFour => {
                // Not produced by the v1 mode chooser yet; skip safely.
                for _ in 0..4 {
                    write_mv_component_vlc(bw, d.mv.0);
                    write_mv_component_vlc(bw, d.mv.1);
                }
            }
            _ => {}
        }
    }
}

/// Encode one MV component using the spec §7.3.5 / Table 7.23 VLC. The
/// encoder writes the same prefix tree the decoder reads: see
/// `decode_mv_component_vlc`.
fn write_mv_component_vlc(bw: &mut BitWriter, mv: i32) {
    // Clamp to the encodable range (±31).
    let v = mv.clamp(-31, 31);
    let abs = v.unsigned_abs() as i32;
    let neg = v < 0;
    match abs {
        0 => {
            bw.write_bits(0b000, 3);
        }
        1 => {
            if neg {
                bw.write_bits(0b010, 3);
            } else {
                bw.write_bits(0b001, 3);
            }
        }
        2 => {
            // 011 + sign bit
            bw.write_bits(0b011, 3);
            bw.write_bits(u32::from(neg), 1);
        }
        3 => {
            bw.write_bits(0b100, 3);
            bw.write_bits(u32::from(neg), 1);
        }
        4..=7 => {
            bw.write_bits(0b101, 3);
            // 3 extra bits: (mag-4)<<1 | sign
            let mag_off = (abs - 4) as u32;
            bw.write_bits((mag_off << 1) | u32::from(neg), 3);
        }
        8..=15 => {
            bw.write_bits(0b110, 3);
            let mag_off = (abs - 8) as u32;
            bw.write_bits((mag_off << 1) | u32::from(neg), 4);
        }
        16..=23 => {
            bw.write_bits(0b1110, 4);
            let mag_off = (abs - 16) as u32;
            bw.write_bits((mag_off << 1) | u32::from(neg), 4);
        }
        _ => {
            // 24..=31
            bw.write_bits(0b1111, 4);
            let mag_off = (abs - 24) as u32;
            bw.write_bits((mag_off << 1) | u32::from(neg), 4);
        }
    }
}

// ---- long/short-run bit string writers (mirror decoder readers) ---------

/// Long-run encoder: emit `bits` in run-length-encoded form per spec §7.2.1.
/// Strategy: greedy — collapse maximal runs of equal value into a single
/// codeword, choosing the smallest run-prefix that fits.
fn write_long_run_bitstring(bw: &mut BitWriter, bits: &[bool]) {
    if bits.is_empty() {
        return;
    }
    // First bit value.
    bw.write_bits(u32::from(bits[0]), 1);
    let mut i = 0usize;
    let mut current = bits[0];
    while i < bits.len() {
        // Count run length of `current`.
        let mut run = 0usize;
        while i + run < bits.len() && bits[i + run] == current {
            run += 1;
        }
        // Encode run (clamped to 4129 max in spec).
        while run > 0 {
            let take = run.min(4129);
            emit_long_run(bw, take as u32);
            run -= take;
            if run > 0 {
                // Same value continues; the spec marks max-length runs with
                // an explicit re-read of the bit value (see decoder), so we
                // emit the same bit again to keep current.
                bw.write_bits(u32::from(current), 1);
            }
        }
        i += {
            // Total run we just consumed (at this point `run` is 0).
            // Find how many bits were of `current`.
            let mut k = 0usize;
            while i + k < bits.len() && bits[i + k] == current {
                k += 1;
            }
            k
        };
        current = !current;
    }
}

/// Encode a single long-run length using the (rstart, rbits) buckets from
/// Table 7.7. Picks the smallest bucket that contains `rlen`.
fn emit_long_run(bw: &mut BitWriter, rlen: u32) {
    // (prefix, prefix_len, rstart, rbits)
    const TABLE: &[(u32, u8, u32, u32)] = &[
        (0b0, 1, 1, 0),
        (0b10, 2, 2, 1),
        (0b110, 3, 4, 1),
        (0b1110, 4, 6, 2),
        (0b11110, 5, 10, 3),
        (0b111110, 6, 18, 4),
        (0b111111, 6, 34, 12),
    ];
    for &(prefix, plen, rstart, rbits) in TABLE {
        let max_offset = if rbits == 0 { 0 } else { (1u32 << rbits) - 1 };
        if rlen >= rstart && rlen <= rstart + max_offset {
            bw.write_bits(prefix, plen as u32);
            if rbits > 0 {
                bw.write_bits(rlen - rstart, rbits);
            }
            return;
        }
    }
    // Should never happen for rlen <= 4129. Fall back to longest bucket.
    let (prefix, plen, rstart, rbits) = (0b111111u32, 6u8, 34u32, 12u32);
    let off = (rlen.saturating_sub(rstart)).min((1 << rbits) - 1);
    bw.write_bits(prefix, plen as u32);
    bw.write_bits(off, rbits);
}

/// Short-run encoder per spec §7.2.2.
fn write_short_run_bitstring(bw: &mut BitWriter, bits: &[bool]) {
    if bits.is_empty() {
        return;
    }
    bw.write_bits(u32::from(bits[0]), 1);
    let mut i = 0usize;
    let mut current = bits[0];
    while i < bits.len() {
        let mut run = 0usize;
        while i + run < bits.len() && bits[i + run] == current {
            run += 1;
        }
        while run > 0 {
            let take = run.min(30);
            emit_short_run(bw, take as u32);
            run -= take;
        }
        // Advance past consumed bits.
        let mut k = 0usize;
        while i + k < bits.len() && bits[i + k] == current {
            k += 1;
        }
        i += k;
        current = !current;
    }
}

fn emit_short_run(bw: &mut BitWriter, rlen: u32) {
    // Table 7.11 buckets.
    const TABLE: &[(u32, u8, u32, u32)] = &[
        (0b0, 1, 1, 1),
        (0b10, 2, 3, 1),
        (0b110, 3, 5, 1),
        (0b1110, 4, 7, 2),
        (0b11110, 5, 11, 2),
        (0b11111, 5, 15, 4),
    ];
    for &(prefix, plen, rstart, rbits) in TABLE {
        let max_offset = if rbits == 0 { 0 } else { (1u32 << rbits) - 1 };
        if rlen >= rstart && rlen <= rstart + max_offset {
            bw.write_bits(prefix, plen as u32);
            if rbits > 0 {
                bw.write_bits(rlen - rstart, rbits);
            }
            return;
        }
    }
    // Fall back to longest bucket.
    let (prefix, plen, rstart, rbits) = (0b11111u32, 5u8, 15u32, 4u32);
    let off = (rlen.saturating_sub(rstart)).min((1 << rbits) - 1);
    bw.write_bits(prefix, plen as u32);
    bw.write_bits(off, rbits);
}

// ---- loop filter (encoder uses the same logic the decoder applies) ------

fn apply_loop_filter(layout: &FrameLayout, planes: &mut [Vec<u8>; 3], bcoded: &[bool], l: i32) {
    for bi in 0..(layout.nbs as usize) {
        if !bcoded[bi] {
            continue;
        }
        let (pli, bx, by) = layout.global_xy(bi as u32);
        let plane = &layout.planes[pli];
        let pw = (plane.nbw * 8) as i32;
        let ph = (plane.nbh * 8) as i32;
        let bx_px = (bx * 8) as i32;
        let by_px_bottom = (by * 8) as i32;
        let buf = &mut planes[pli];
        if bx_px > 0 {
            horizontal_filter(buf, pw, ph, bx_px - 2, by_px_bottom, l);
        }
        if by_px_bottom > 0 {
            vertical_filter(buf, pw, ph, bx_px, by_px_bottom - 2, l);
        }
        if bx + 1 < plane.nbw {
            let bj = layout.global_coded(pli, bx + 1, by) as usize;
            if !bcoded[bj] {
                horizontal_filter(buf, pw, ph, bx_px + 6, by_px_bottom, l);
            }
        }
        if by + 1 < plane.nbh {
            let bj = layout.global_coded(pli, bx, by + 1) as usize;
            if !bcoded[bj] {
                vertical_filter(buf, pw, ph, bx_px, by_px_bottom + 6, l);
            }
        }
    }
}

fn horizontal_filter(buf: &mut [u8], pw: i32, ph: i32, fx: i32, fy: i32, l: i32) {
    for by in 0..8 {
        let row_bottom = fy + by;
        if row_bottom < 0 || row_bottom >= ph {
            continue;
        }
        if fx < 0 || fx + 3 >= pw {
            continue;
        }
        let top_y = ph - 1 - row_bottom;
        let row = (top_y * pw) as usize;
        let a = buf[row + fx as usize] as i32;
        let b = buf[row + (fx + 1) as usize] as i32;
        let c = buf[row + (fx + 2) as usize] as i32;
        let d = buf[row + (fx + 3) as usize] as i32;
        let r = (a - 3 * b + 3 * c - d + 4) >> 3;
        let r_lim = lflim(r, l);
        let nb = (b + r_lim).clamp(0, 255) as u8;
        let nc = (c - r_lim).clamp(0, 255) as u8;
        buf[row + (fx + 1) as usize] = nb;
        buf[row + (fx + 2) as usize] = nc;
    }
}

fn vertical_filter(buf: &mut [u8], pw: i32, ph: i32, fx: i32, fy: i32, l: i32) {
    for bx in 0..8 {
        let col = fx + bx;
        if col < 0 || col >= pw {
            continue;
        }
        if fy < 0 || fy + 3 >= ph {
            continue;
        }
        let rowy = |yb: i32| -> usize { ((ph - 1 - yb) * pw + col) as usize };
        let a = buf[rowy(fy)] as i32;
        let b = buf[rowy(fy + 1)] as i32;
        let c = buf[rowy(fy + 2)] as i32;
        let d = buf[rowy(fy + 3)] as i32;
        let r = (a - 3 * b + 3 * c - d + 4) >> 3;
        let r_lim = lflim(r, l);
        let nb = (b + r_lim).clamp(0, 255) as u8;
        let nc = (c - r_lim).clamp(0, 255) as u8;
        buf[rowy(fy + 1)] = nb;
        buf[rowy(fy + 2)] = nc;
    }
}

fn lflim(r: i32, l: i32) -> i32 {
    if r <= -2 * l {
        0
    } else if r <= -l {
        -r - 2 * l
    } else if r < l {
        r
    } else if r < 2 * l {
        -r + 2 * l
    } else {
        0
    }
}

// --- header construction --------------------------------------------------

struct BuildIdParams {
    fmbw: u16,
    fmbh: u16,
    picw: u32,
    pich: u32,
    picx: u32,
    picy: u32,
    frn: u32,
    frd: u32,
    pf: TheoraPixelFormat,
}

fn build_identification_packet(p: BuildIdParams) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_bytes(&[0x80, b't', b'h', b'e', b'o', b'r', b'a']);
    bw.write_bits(3, 8);
    bw.write_bits(2, 8);
    bw.write_bits(1, 8);
    bw.write_bits(p.fmbw as u32, 16);
    bw.write_bits(p.fmbh as u32, 16);
    bw.write_bits(p.picw, 24);
    bw.write_bits(p.pich, 24);
    bw.write_bits(p.picx & 0xFF, 8);
    bw.write_bits(p.picy & 0xFF, 8);
    bw.write_bits(p.frn, 32);
    bw.write_bits(p.frd, 32);
    bw.write_bits(1, 24);
    bw.write_bits(1, 24);
    bw.write_bits(0, 8);
    bw.write_bits(0, 24);
    bw.write_bits(32, 6);
    bw.write_bits(6, 5);
    let pf_code = match p.pf {
        TheoraPixelFormat::Yuv420 => 0u32,
        TheoraPixelFormat::Yuv422 => 2,
        TheoraPixelFormat::Yuv444 => 3,
        TheoraPixelFormat::Reserved => 0,
    };
    bw.write_bits(pf_code, 2);
    bw.write_bits(0, 3);
    bw.finish()
}

fn build_comment_packet(vendor: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x81);
    out.extend_from_slice(b"theora");
    let v = vendor.as_bytes();
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v);
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn xiph_lace(packets: &[&[u8]]) -> Vec<u8> {
    let n = packets.len();
    let mut out = Vec::new();
    out.push((n - 1) as u8);
    for p in &packets[..n - 1] {
        let mut sz = p.len();
        while sz >= 255 {
            out.push(255);
            sz -= 255;
        }
        out.push(sz as u8);
    }
    for p in packets {
        out.extend_from_slice(p);
    }
    out
}

// --- bit writer (MSB-first, no byte stuffing) -----------------------------

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    buf: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self::default()
    }

    fn write_bits(&mut self, value: u32, len: u32) {
        if len == 0 {
            return;
        }
        debug_assert!(len <= 32);
        let mask = if len == 32 {
            u32::MAX
        } else {
            (1u32 << len) - 1
        };
        let v = (value & mask) as u64;
        let acc = ((self.buf as u64) << len) | v;
        let mut nbits = self.nbits + len;
        while nbits >= 8 {
            nbits -= 8;
            let b = ((acc >> nbits) & 0xFF) as u8;
            self.out.push(b);
        }
        let keep_mask = if nbits == 0 {
            0u64
        } else {
            (1u64 << nbits) - 1
        };
        self.buf = (acc & keep_mask) as u32;
        self.nbits = nbits;
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        if self.nbits == 0 {
            self.out.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.write_bits(b as u32, 8);
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
        self.out
    }
}

// Suppress unused-variable warning for `MODE_ALPHABETS`: imported to make the
// intent clear that we know about the alphabets even though MSCHEME=0
// supplies its own.
#[allow(dead_code)]
const _: [[Mode; 8]; 7] = MODE_ALPHABETS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_packet_round_trips_through_parser() {
        let p = build_identification_packet(BuildIdParams {
            fmbw: 4,
            fmbh: 4,
            picw: 64,
            pich: 64,
            picx: 0,
            picy: 0,
            frn: 24,
            frd: 1,
            pf: TheoraPixelFormat::Yuv420,
        });
        let id = crate::headers::parse_identification_header(&p).expect("parse id");
        assert_eq!(id.fmbw, 4);
        assert_eq!(id.fmbh, 4);
        assert_eq!(id.picw, 64);
        assert_eq!(id.pich, 64);
        assert_eq!(id.frn, 24);
        assert_eq!(id.frd, 1);
        assert_eq!(id.pf, TheoraPixelFormat::Yuv420);
    }

    #[test]
    fn comment_packet_round_trips() {
        let p = build_comment_packet("oxideav-theora");
        let c = crate::headers::parse_comment_header(&p).expect("parse comment");
        assert_eq!(c.vendor, "oxideav-theora");
        assert!(c.comments.is_empty());
    }

    #[test]
    fn standard_setup_parses() {
        let s = parse_setup_header(STANDARD_SETUP).expect("parse standard setup");
        assert_eq!(s.huffs.len(), 80);
    }

    fn make_test_encoder() -> Box<dyn Encoder> {
        let mut params = CodecParameters::video(CodecId::new("theora"));
        params.media_type = MediaType::Video;
        params.width = Some(64);
        params.height = Some(64);
        params.pixel_format = Some(CorePixelFormat::Yuv420P);
        params.frame_rate = Some(oxideav_core::Rational::new(24, 1));
        make_encoder(&params).unwrap()
    }

    fn round_trip_one(frame: VideoFrame) -> VideoFrame {
        use crate::decoder::make_decoder;
        let mut enc = make_test_encoder();
        enc.send_frame(&Frame::Video(frame)).unwrap();
        let mut pkts = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            pkts.push(p);
        }
        let extradata = enc.output_params().extradata.clone();
        let mut p2 = CodecParameters::video(CodecId::new("theora"));
        p2.extradata = extradata;
        let mut dec = make_decoder(&p2).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 24), pkts[3].data.clone());
        dec.send_packet(&pkt).unwrap();
        match dec.receive_frame().unwrap() {
            Frame::Video(v) => v,
            _ => panic!(),
        }
    }

    fn make_yuv420_frame(y: Vec<u8>, u: Vec<u8>, v: Vec<u8>) -> VideoFrame {
        VideoFrame {
            format: CorePixelFormat::Yuv420P,
            width: 64,
            height: 64,
            pts: Some(0),
            time_base: TimeBase::new(1, 24),
            planes: vec![
                oxideav_core::VideoPlane {
                    stride: 64,
                    data: y,
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: u,
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: v,
                },
            ],
        }
    }

    #[test]
    fn debug_round_trip_constant_gray() {
        use crate::decoder::make_decoder;
        let mut params = CodecParameters::video(CodecId::new("theora"));
        params.media_type = MediaType::Video;
        params.width = Some(64);
        params.height = Some(64);
        params.pixel_format = Some(CorePixelFormat::Yuv420P);
        params.frame_rate = Some(oxideav_core::Rational::new(24, 1));
        let mut enc = make_encoder(&params).unwrap();

        let f = VideoFrame {
            format: CorePixelFormat::Yuv420P,
            width: 64,
            height: 64,
            pts: Some(0),
            time_base: TimeBase::new(1, 24),
            planes: vec![
                oxideav_core::VideoPlane {
                    stride: 64,
                    data: vec![128; 64 * 64],
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: vec![128; 32 * 32],
                },
                oxideav_core::VideoPlane {
                    stride: 32,
                    data: vec![128; 32 * 32],
                },
            ],
        };
        enc.send_frame(&Frame::Video(f)).unwrap();
        let mut pkts = Vec::new();
        while let Ok(p) = enc.receive_packet() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 4);

        let extradata = enc.output_params().extradata.clone();
        let mut p2 = CodecParameters::video(CodecId::new("theora"));
        p2.extradata = extradata;
        let mut dec = make_decoder(&p2).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 24), pkts[3].data.clone());
        dec.send_packet(&pkt).unwrap();
        let out = match dec.receive_frame().unwrap() {
            Frame::Video(v) => v,
            _ => panic!(),
        };
        let y = &out.planes[0].data;
        let mn = *y.iter().min().unwrap() as i32;
        let mx = *y.iter().max().unwrap() as i32;
        let mean: i32 = y.iter().map(|&v| v as i32).sum::<i32>() / (y.len() as i32);
        eprintln!("constant gray: Y min={} max={} mean={}", mn, mx, mean);
        assert!((mean - 128).abs() <= 3, "mean Y = {}, want ~128", mean);
    }

    #[test]
    fn round_trip_horizontal_gradient() {
        let mut y = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                y[j * 64 + i] = (i * 4) as u8;
            }
        }
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];
        let out = round_trip_one(make_yuv420_frame(y.clone(), u, v));
        let dec_y = &out.planes[0].data;
        let mut max_diff: u32 = 0;
        for k in 0..(64 * 64) {
            let d = (dec_y[k] as i32 - y[k] as i32).unsigned_abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff <= 6, "max_diff = {max_diff}");
    }

    #[test]
    fn bit_writer_round_trip_u32() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b1, 1);
        bw.write_bits(0b01, 2);
        bw.write_bits(0xABCD, 16);
        bw.write_bits(0b101, 3);
        let buf = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&buf);
        assert_eq!(br.read_u32(1).unwrap(), 0b1);
        assert_eq!(br.read_u32(2).unwrap(), 0b01);
        assert_eq!(br.read_u32(16).unwrap(), 0xABCD);
        assert_eq!(br.read_u32(3).unwrap(), 0b101);
    }

    #[test]
    fn long_run_round_trip_basic() {
        let bits = vec![true; 10];
        let mut bw = BitWriter::new();
        write_long_run_bitstring(&mut bw, &bits);
        let buf = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&buf);
        let out = crate::block::read_long_run_bitstring(&mut br, bits.len()).unwrap();
        assert_eq!(out, bits);
    }

    #[test]
    fn long_run_round_trip_alternating() {
        let bits: Vec<bool> = (0..30).map(|i| i % 2 == 0).collect();
        let mut bw = BitWriter::new();
        write_long_run_bitstring(&mut bw, &bits);
        let buf = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&buf);
        let out = crate::block::read_long_run_bitstring(&mut br, bits.len()).unwrap();
        assert_eq!(out, bits);
    }

    #[test]
    fn short_run_round_trip_mixed() {
        let bits = vec![
            true, true, false, false, false, true, false, true, true, true, true, false,
        ];
        let mut bw = BitWriter::new();
        write_short_run_bitstring(&mut bw, &bits);
        let buf = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&buf);
        let out = crate::block::read_short_run_bitstring(&mut br, bits.len()).unwrap();
        assert_eq!(out, bits);
    }

    #[test]
    fn mv_vlc_round_trip() {
        for v in -31..=31i32 {
            let mut bw = BitWriter::new();
            write_mv_component_vlc(&mut bw, v);
            let buf = bw.finish();
            let mut br = crate::bitreader::BitReader::new(&buf);
            let got = crate::inter::decode_mv_component_vlc(&mut br).unwrap();
            assert_eq!(got, v, "mismatch at v={v}");
        }
    }
}
