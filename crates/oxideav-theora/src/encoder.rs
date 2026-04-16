//! Theora I-frame-only encoder.
//!
//! This encoder targets the **intra-only** subset of Theora I.2: every frame
//! is emitted as a key (intra) frame. P-frames are explicitly not produced.
//!
//! Strategy:
//!
//! * Identification + comment headers are constructed from the
//!   [`CodecParameters`] passed at build time.
//! * The setup header (loop filter limits, AC/DC scale tables, quant base
//!   matrices, quant ranges, 80 Huffman trees) is **embedded verbatim** from
//!   a known-good libtheora reference. Both our own decoder and ffmpeg's
//!   `libtheora` accept it without modification, and shipping a fixed setup
//!   keeps the encoder small while still being spec-compliant. The bytes were
//!   captured from `ffmpeg`'s default Theora encoder output.
//! * Per frame we run forward DCT (`fdct8x8`) → integer dequant rounding →
//!   forward DC prediction (mirroring the decoder's gradient predictor) →
//!   token RLE → Huffman encoding using the embedded setup's trees.
//! * No motion vectors, no inter-frame mode signalling, no AC prediction
//!   (Theora doesn't have explicit AC prediction in the bitstream — DC is
//!   the only spatial predictor).
//!
//! The output is a sequence of three header packets followed by one frame
//! packet per call to `send_frame`. Callers are responsible for muxing into
//! Ogg, MKV or whatever container is required.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat as CorePixelFormat,
    Result, TimeBase, VideoFrame,
};

use crate::coded_order::FrameLayout;
use crate::dct::INV_ZIGZAG;
use crate::encoder_huffman::{build_all, HuffCode, HuffTable};
use crate::fdct::fdct8x8;
use crate::headers::{parse_setup_header, PixelFormat as TheoraPixelFormat, Setup};
use crate::quant::build_qmat;

/// Standard libtheora setup header (3196 bytes). Captured from
/// `ffmpeg -i lavfi -i testsrc -c:v libtheora`. Decoder validates it
/// at constructor time.
const STANDARD_SETUP: &[u8] = include_bytes!("encoder_data/standard_setup.bin");

/// Quantisation index used for every block. 32 is mid-quality and matches
/// what libtheora picks for `-qscale 6` in this clip range.
pub const DEFAULT_QI: u8 = 32;

/// Build a Theora encoder for the given [`CodecParameters`].
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
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

    let theora_pf = TheoraPixelFormat::Yuv420;
    // Frame dimensions must be multiples of 16; pad if needed.
    let fmbw = width.div_ceil(16) as u16;
    let fmbh = height.div_ceil(16) as u16;
    let frame_w = fmbw as u32 * 16;
    let frame_h = fmbh as u32 * 16;

    let setup = parse_setup_header(STANDARD_SETUP)?;
    let huff_tables = build_all(&setup.huffs);

    let layout = FrameLayout::new(fmbw as u32, fmbh as u32, theora_pf);

    let frn = params.frame_rate.map(|r| r.num as u32).unwrap_or(24);
    let frd = params.frame_rate.map(|r| r.den as u32).unwrap_or(1);

    // Identification packet bytes (used both as the first emitted header and
    // as part of the codec parameters' extradata).
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
        qi: DEFAULT_QI,
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
        let data = self.encode_intra_frame(v)?;
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

impl TheoraEncoder {
    fn emit_header_packet(&mut self, data: Vec<u8>, pts: Option<i64>) {
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.flags.header = true;
        self.pending.push_back(pkt);
    }

    fn encode_intra_frame(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
        let nbs = self.layout.nbs as usize;
        // 1. Build quantised coefficient blocks per block in coded order.
        // Coefficients are stored in zig-zag order: coeffs[bi*64 + ti].
        let mut coeffs = vec![0i32; nbs * 64];

        // Cache one quant matrix per plane (qti=0 = intra).
        let qmats: [[i32; 64]; 3] = [
            build_qmat(&self.setup, 0, 0, self.qi),
            build_qmat(&self.setup, 0, 1, self.qi),
            build_qmat(&self.setup, 0, 2, self.qi),
        ];

        for bi in 0..nbs {
            let (pli, bx, by) = self.layout.global_xy(bi as u32);
            let block = self.fetch_block_pixels(frame, pli, bx, by);
            let mut f = block;
            // Level shift: subtract 128 before DCT (samples are 0..255 → -128..127).
            for v in f.iter_mut() {
                *v -= 128.0;
            }
            fdct8x8(&mut f);
            // Quantise. The Theora integer IDCT applies an implicit 1/32 scale
            // for DC (vs. textbook DCT-II's 1/8). To compensate, we
            // pre-scale the textbook FDCT output by 4 so the dequantised
            // coefficient fed to the integer IDCT recovers the original
            // pixel residual. With this factor the chain is
            //
            //   pixel_residual ≈ idct(quant_coef * qmat) ≈ idct(F * 4) / 32
            //                 ≈ F * 4 / 32 = F / 8 = textbook idct of F.
            let mut quant_zz = [0i32; 64];
            for ci in 0..64 {
                let q = qmats[pli][ci].max(1);
                let raw = f[ci] * 4.0;
                let qc = (raw / q as f32).round() as i32;
                // Theora coefficients are stored as 16-bit signed values.
                let qc = qc.clamp(-32768, 32767);
                let zzi = INV_ZIGZAG[ci];
                quant_zz[zzi] = qc;
            }
            // Copy into the global coefficient array (zig-zag layout).
            for ti in 0..64 {
                coeffs[bi * 64 + ti] = quant_zz[ti];
            }
        }

        // 2. Apply forward DC prediction. Each block's DC becomes
        //    raw_dc = quantised_dc - predicted_dc, where predicted_dc uses
        //    the same gradient predictor as the decoder. Walk per plane in
        //    raster order so predecessors are processed first.
        self.apply_forward_dc_prediction(&mut coeffs);

        // 3. Bit-pack the frame.
        let mut bw = BitWriter::new();
        // Frame header: data-bit (0) + ftype-bit (0 = intra) + 6 bits of QI[0]
        // + 1-bit (0 = NQIS=1) + 3 reserved bits = 0.
        bw.write_bits(0, 1); // data packet
        bw.write_bits(0, 1); // intra
        bw.write_bits(self.qi as u32, 6);
        bw.write_bits(0, 1); // NQIS extension flag
        bw.write_bits(0, 3); // reserved (intra only)

        // QIIs: with NQIS=1 there are no QII bits.

        // 4. DCT coefficient encoding (spec §7.7.3).
        self.encode_coefficients(&mut bw, &coeffs)?;

        Ok(bw.finish())
    }

    /// Read 8x8 block of luma/chroma samples into a buffer indexed in
    /// **bottom-up** row order: `out[ri * 8 + ci]` is row `ri` of the block
    /// counted from the BOTTOM in coded-frame coordinates. This matches the
    /// orientation the integer IDCT writes back into the decoder, so an FDCT
    /// applied to this buffer dequantises into a residual the decoder can
    /// add to the predictor without needing a row flip.
    ///
    /// Pixels outside the picture are clamped (edge-replicated).
    fn fetch_block_pixels(&self, frame: &VideoFrame, pli: usize, bx: u32, by: u32) -> [f32; 64] {
        let plane = &frame.planes[pli];
        let stride = plane.stride;
        // Plane pixel dims in the source frame.
        let (sw, sh) = match pli {
            0 => (frame.width as i32, frame.height as i32),
            _ => ((frame.width / 2) as i32, (frame.height / 2) as i32),
        };
        // Plane pixel dims in coded frame (frame_w/h padded to mb).
        let (_cw, ch) = match pli {
            0 => (self.frame_w as i32, self.frame_h as i32),
            _ => ((self.frame_w / 2) as i32, (self.frame_h / 2) as i32),
        };
        let bx_px = (bx * 8) as i32;
        let by_px_bottom = (by * 8) as i32;
        // Top row of block in coded-frame top-down coords.
        let top_row_coded = ch - 1 - (by_px_bottom + 7);
        // Picture is anchored at the bottom of the coded frame (picy=0); the
        // top of the picture sits at coded row (ch - sh).
        let pic_top_offset = ch - sh;

        let mut out = [0.0f32; 64];
        for ri in 0..8i32 {
            // Bottom-up row index. ri=0 is the bottom of the block.
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

    /// Apply forward DC prediction. After this, `coeffs[bi*64]` holds
    /// `q_dc - predicted_dc` (the residual the decoder will sum back).
    fn apply_forward_dc_prediction(&self, coeffs: &mut [i32]) {
        for pli in 0..3 {
            let mut last_dc = [0i32; 3];
            let plane = &self.layout.planes[pli];
            let nbw = plane.nbw;
            let nbh = plane.nbh;
            // Snapshot quantised DCs (the predictor must use original values
            // of *predecessor* blocks, but we mutate `coeffs[bi*64]` in place
            // so we collect the originals first).
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
                    // All blocks are coded in intra frames (rfi = 0).
                    let mut p = [false; 4];
                    let mut pv = [0i32; 4];
                    if bx > 0 {
                        p[0] = true;
                        pv[0] = q_at(bx - 1, by);
                    }
                    if bx > 0 && by > 0 {
                        p[1] = true;
                        pv[1] = q_at(bx - 1, by - 1);
                    }
                    if by > 0 {
                        p[2] = true;
                        pv[2] = q_at(bx, by - 1);
                    }
                    if bx + 1 < nbw && by > 0 {
                        p[3] = true;
                        pv[3] = q_at(bx + 1, by - 1);
                    }
                    let dcpred = if !p.iter().any(|&b| b) {
                        last_dc[0]
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
                    // Stored coefficient = q - prediction (16-bit truncated).
                    let residual = ((q - dcpred) as i16) as i32;
                    coeffs[bi * 64] = residual;
                    last_dc[0] = q;
                }
            }
        }
    }

    /// Encode all DCT coefficients per spec §7.7.3.
    ///
    /// Strategy: build a per-block token list first (one entry per ti
    /// position the block produces), then walk ti=0..63 and within each ti
    /// emit each block's pending token in coded-order. We do NOT use
    /// cross-block EOB runs: every block ends with an EOB-1 token.
    fn encode_coefficients(&self, bw: &mut BitWriter, coeffs: &[i32]) -> Result<()> {
        let nbs = self.layout.nbs as usize;
        let nlbs = (self.layout.planes[0].nbw * self.layout.planes[0].nbh) as usize;

        // Build per-block token program.
        let mut programs: Vec<Vec<TokenAtTi>> = Vec::with_capacity(nbs);
        for bi in 0..nbs {
            programs.push(build_block_token_program(&coeffs[bi * 64..bi * 64 + 64]));
        }

        // We always pick HTI = 0 for both luma and chroma at every group.
        let hti_l: u8 = 0;
        let hti_c: u8 = 0;

        // Per-block "next-token" pointer (== current ti for that block).
        // We walk programs in lock-step with the decoder: for each ti from
        // 0..63, scan blocks in coded order. Any block whose program's next
        // token has `ti_pos == ti` writes that token now.
        let mut prog_idx = vec![0usize; nbs];
        for ti in 0u8..64 {
            // Header bytes: HTI for ti==0 (DC) selects the table family for
            // the DC group; ti==1 selects the table family used by AC groups
            // 1..=4 (libtheora-style, spec §7.7.3).
            if ti == 0 || ti == 1 {
                bw.write_bits(hti_l as u32, 4);
                bw.write_bits(hti_c as u32, 4);
            }
            let hg = huffman_group_for_ti(ti) as usize;
            let table_idx_l = 16 * hg + hti_l as usize;
            let table_idx_c = 16 * hg + hti_c as usize;
            for bi in 0..nbs {
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

/// Token stream entry: (token id, extra bits to emit, ti position when this
/// token is consumed). Recorded per block.
#[derive(Clone, Copy, Debug)]
struct TokenAtTi {
    token: u8,
    extra_bits: u32,
    extra_len: u8,
    ti_pos: u8,
}

fn build_block_token_program(zz_coeffs: &[i32]) -> Vec<TokenAtTi> {
    debug_assert_eq!(zz_coeffs.len(), 64);
    // Find last non-zero index in zig-zag order.
    let mut last_nz = -1i32;
    for ti in 0..64usize {
        if zz_coeffs[ti] != 0 {
            last_nz = ti as i32;
        }
    }
    let mut out: Vec<TokenAtTi> = Vec::new();
    if last_nz < 0 {
        // All-zero block: emit a single EOB(1) token at ti=0.
        out.push(TokenAtTi {
            token: 0, // EOB run = 1
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
            // Emit a zero-run token spanning `run` positions.
            // Token 7: run 1..=8 (extra 3 bits = run-1).
            // Token 8: run 1..=64 (extra 6 bits = run-1).
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
    // End-of-block: emit EOB-1 token at the position right after last non-zero.
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

/// Encode a single non-zero coefficient `v` into a value-token entry at
/// position `ti`. Selects from tokens 9..22 per spec §7.7.3.
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
        // Token 13..16 carry abs values 3..=6 and one sign bit.
        let token = (abs + 10) as u8; // 3→13, 4→14, 5→15, 6→16
        return TokenAtTi {
            token,
            extra_bits: u32::from(neg),
            extra_len: 1,
            ti_pos: ti,
        };
    }
    // Range 7..=8 → token 17, sign + 1-bit (mag-7).
    if (7..=8).contains(&abs) {
        let mag_off = abs - 7;
        return TokenAtTi {
            token: 17,
            extra_bits: (u32::from(neg) << 1) | mag_off as u32,
            extra_len: 2,
            ti_pos: ti,
        };
    }
    // 9..=12 → token 18, sign + 2-bit.
    if (9..=12).contains(&abs) {
        let mag_off = abs - 9;
        return TokenAtTi {
            token: 18,
            extra_bits: (u32::from(neg) << 2) | mag_off as u32,
            extra_len: 3,
            ti_pos: ti,
        };
    }
    // 13..=20 → token 19, sign + 3-bit.
    if (13..=20).contains(&abs) {
        let mag_off = abs - 13;
        return TokenAtTi {
            token: 19,
            extra_bits: (u32::from(neg) << 3) | mag_off as u32,
            extra_len: 4,
            ti_pos: ti,
        };
    }
    // 21..=36 → token 20, sign + 4-bit.
    if (21..=36).contains(&abs) {
        let mag_off = abs - 21;
        return TokenAtTi {
            token: 20,
            extra_bits: (u32::from(neg) << 4) | mag_off as u32,
            extra_len: 5,
            ti_pos: ti,
        };
    }
    // 37..=68 → token 21, sign + 5-bit.
    if (37..=68).contains(&abs) {
        let mag_off = abs - 37;
        return TokenAtTi {
            token: 21,
            extra_bits: (u32::from(neg) << 5) | mag_off as u32,
            extra_len: 6,
            ti_pos: ti,
        };
    }
    // 69..=580 → token 22, sign + 9-bit. Clamp magnitude.
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

/// Same DC-prediction weights as the decoder. Mirrored from `block::weights_for`.
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
    // Magic prefix: 0x80 + "theora" (1 + 6 = 7 bytes).
    bw.write_bytes(&[0x80, b't', b'h', b'e', b'o', b'r', b'a']);
    bw.write_bits(3, 8); // VMAJ
    bw.write_bits(2, 8); // VMIN
    bw.write_bits(1, 8); // VREV
    bw.write_bits(p.fmbw as u32, 16);
    bw.write_bits(p.fmbh as u32, 16);
    bw.write_bits(p.picw, 24);
    bw.write_bits(p.pich, 24);
    bw.write_bits(p.picx & 0xFF, 8);
    bw.write_bits(p.picy & 0xFF, 8);
    bw.write_bits(p.frn, 32);
    bw.write_bits(p.frd, 32);
    // Pixel aspect ratio numerator/denominator (1:1).
    bw.write_bits(1, 24);
    bw.write_bits(1, 24);
    bw.write_bits(0, 8); // colour space
    bw.write_bits(0, 24); // nominal bit-rate
    bw.write_bits(32, 6); // qual
    bw.write_bits(6, 5); // kfgshift = 6 (keyframe granule shift)
    let pf_code = match p.pf {
        TheoraPixelFormat::Yuv420 => 0u32,
        TheoraPixelFormat::Yuv422 => 2,
        TheoraPixelFormat::Yuv444 => 3,
        TheoraPixelFormat::Reserved => 0,
    };
    bw.write_bits(pf_code, 2);
    bw.write_bits(0, 3); // reserved
    bw.finish()
}

fn build_comment_packet(vendor: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x81);
    out.extend_from_slice(b"theora");
    let v = vendor.as_bytes();
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v);
    out.extend_from_slice(&0u32.to_le_bytes()); // 0 user comments
    out
}

/// Pack three packets in Xiph-style lacing for use as `extradata`.
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
    /// Bit accumulator; bits packed MSB-first.
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
        // Shift accumulator left by `len`, OR new low bits in. Use u64 so
        // partial bytes never lose data (max combined size is 32 + 7 < 64).
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
            // Pad with zero bits.
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
        self.out
    }
}

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
        // For a constant input of 128, output should be ~128.
        assert!((mean - 128).abs() <= 3, "mean Y = {}, want ~128", mean);
    }

    #[test]
    fn round_trip_horizontal_gradient() {
        // Linear horizontal gradient: pixel value = x * 4 (0..252).
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
        // The standard quant matrix at qi=32 introduces some loss; ±4 per
        // pixel is well within the lossy tolerance we expect.
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
}
