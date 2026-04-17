//! EBU STL (EBU Tech 3264 / ISO 18041) subtitle parser + writer.
//!
//! Binary format:
//!
//! * **GSI block** — 1024 bytes of header metadata (code page, disk
//!   format code, frame rate, totals, max row/col).
//! * **TTI blocks** — 128 bytes each, one per subtitle line (extension
//!   blocks via `EBN != 0xFF` are merged into a single cue).
//!
//! GSI fields handled:
//! * `CPN` (code page 3 bytes — interpreted as ASCII/Latin-1 fallback)
//! * `DFC` (8 bytes, disk format code, e.g. `STL25.01` → 25 fps)
//! * `TNB` (total TTI blocks, 5-digit ASCII)
//! * `TNS` (total subtitles, 5-digit ASCII)
//! * `MNC` (max chars per row, 2-digit ASCII)
//! * `MNR` (max rows, 2-digit ASCII)
//!
//! Each TTI:
//! * SGN (1), SN (2 LE), EBN (1), CS (1)
//! * TCI (4: HH MM SS FF)
//! * TCO (4: HH MM SS FF)
//! * VP (1), JC (1), CF (1)
//! * TF (112) — text field with control codes
//!
//! Text-field control codes (§ Tech 3264):
//! * `0x00..=0x07` — italic/underline/boxing on/off (0x80/0x81 italic
//!   on/off; 0x82/0x83 underline on/off; 0x84/0x85 boxing on/off —
//!   mapping per EBU ARIB rework here uses 0x80..0x87).
//! * `0x8A` — CRLF (line break)
//! * `0x8F` — pad (trailing, ignored)
//!
//! Colour codes are emitted as [`Segment::Raw`] for lossless round-trip
//! in this first cut.
//!
//! Text mode only — bitmap subtitles (DFC `STL*.22`) are not decoded.

use oxideav_core::{Error, Result, Segment, SubtitleCue};

use crate::ir::{SourceFormat, SubtitleTrack};

pub const GSI_SIZE: usize = 1024;
pub const TTI_SIZE: usize = 128;
/// Codec id string.
pub const CODEC_ID: &str = "ebu_stl";

/// Parse an EBU STL file into a track.
pub fn parse(bytes: &[u8]) -> Result<SubtitleTrack> {
    if bytes.len() < GSI_SIZE {
        return Err(Error::invalid("EBU STL: truncated GSI header"));
    }
    let gsi = &bytes[..GSI_SIZE];

    let dfc = std::str::from_utf8(&gsi[3..11])
        .unwrap_or("STL25.01")
        .trim();
    let fps = fps_from_dfc(dfc);

    let cpn = std::str::from_utf8(&gsi[0..3]).unwrap_or("850").trim();
    let mnc = ascii_u32(&gsi[248..250]).unwrap_or(40);
    let mnr = ascii_u32(&gsi[250..252]).unwrap_or(23);
    let tnb = ascii_u32(&gsi[238..243]).unwrap_or(0);
    let tns = ascii_u32(&gsi[243..248]).unwrap_or(0);

    let mut track = SubtitleTrack {
        source: Some(SourceFormat::Srt),
        ..SubtitleTrack::default()
    };
    track
        .metadata
        .push(("source_format".into(), "ebu_stl".into()));
    track.metadata.push(("dfc".into(), dfc.to_string()));
    track.metadata.push(("cpn".into(), cpn.to_string()));
    track.metadata.push(("mnc".into(), mnc.to_string()));
    track.metadata.push(("mnr".into(), mnr.to_string()));
    track.metadata.push(("tnb".into(), tnb.to_string()));
    track.metadata.push(("tns".into(), tns.to_string()));

    // Walk TTI records.
    let tail = &bytes[GSI_SIZE..];
    if tail.len() % TTI_SIZE != 0 && tail.len() / TTI_SIZE == 0 {
        return Err(Error::invalid("EBU STL: TTI section empty / not aligned"));
    }

    let mut i = 0;
    while i + TTI_SIZE <= tail.len() {
        let rec = &tail[i..i + TTI_SIZE];
        i += TTI_SIZE;

        let _sgn = rec[0];
        let _sn = u16::from_le_bytes([rec[1], rec[2]]);
        let ebn = rec[3];
        let _cs = rec[4];
        let tci = (rec[5], rec[6], rec[7], rec[8]);
        let tco = (rec[9], rec[10], rec[11], rec[12]);
        let _vp = rec[13];
        let _jc = rec[14];
        let cf = rec[15];
        let tf = &rec[16..128];

        if cf != 0 {
            // Comment flag set — skip.
            continue;
        }

        let start_us = tc_to_us(tci, fps);
        let end_us = tc_to_us(tco, fps);

        // Extension blocks carry continuation text of the prior cue
        // (EBN values 0x00..0xFE = extension index; 0xFF = last / only).
        if ebn != 0xFF && ebn != 0x00 {
            // Continuation — merge into previous cue.
            if let Some(last) = track.cues.last_mut() {
                let addl = decode_text_field(tf, cpn);
                extend_segments(&mut last.segments, addl);
                continue;
            }
        }

        let segments = decode_text_field(tf, cpn);
        track.cues.push(SubtitleCue {
            start_us,
            end_us,
            style_ref: None,
            positioning: None,
            segments,
        });
    }

    Ok(track)
}

/// Write a track as EBU STL bytes.
pub fn write(track: &SubtitleTrack) -> Result<Vec<u8>> {
    let dfc = meta_or(&track.metadata, "dfc", "STL25.01");
    let cpn = meta_or(&track.metadata, "cpn", "850");
    let mnc: u32 = meta_or(&track.metadata, "mnc", "40").parse().unwrap_or(40);
    let mnr: u32 = meta_or(&track.metadata, "mnr", "23").parse().unwrap_or(23);
    let fps = fps_from_dfc(&dfc);

    let mut gsi = [0x20u8; GSI_SIZE];
    write_ascii_fixed(&mut gsi[0..3], &cpn, 3);
    write_ascii_fixed(&mut gsi[3..11], &dfc, 8);
    // DSC — "1" (teletext latin), filled by default spaces if unchanged.
    gsi[11] = b'1';
    // CCT — "00" default.
    write_ascii_fixed(&mut gsi[12..14], "00", 2);
    // LC (language code) — 2 bytes, default "00".
    write_ascii_fixed(&mut gsi[14..16], "00", 2);

    // Count TTI rows we'll actually write.
    let tti_rows = track.cues.len() as u32;
    write_ascii_fixed(&mut gsi[238..243], &format!("{:05}", tti_rows), 5);
    write_ascii_fixed(&mut gsi[243..248], &format!("{:05}", track.cues.len()), 5);
    write_ascii_fixed(&mut gsi[248..250], &format!("{:02}", mnc), 2);
    write_ascii_fixed(&mut gsi[250..252], &format!("{:02}", mnr), 2);
    // TCS — "1" (TCS format = hh:mm:ss:ff).
    gsi[252] = b'1';

    let mut out = Vec::with_capacity(GSI_SIZE + track.cues.len() * TTI_SIZE);
    out.extend_from_slice(&gsi);

    for (idx, cue) in track.cues.iter().enumerate() {
        let mut tti = [0u8; TTI_SIZE];
        tti[0] = 0; // SGN
        let sn = (idx as u16).to_le_bytes();
        tti[1] = sn[0];
        tti[2] = sn[1];
        tti[3] = 0xFF; // EBN = last (we don't split extensions here)
        tti[4] = 0; // CS
        let tci = us_to_tc(cue.start_us, fps);
        tti[5] = tci.0;
        tti[6] = tci.1;
        tti[7] = tci.2;
        tti[8] = tci.3;
        let tco = us_to_tc(cue.end_us, fps);
        tti[9] = tco.0;
        tti[10] = tco.1;
        tti[11] = tco.2;
        tti[12] = tco.3;
        tti[13] = (mnr as u8).saturating_sub(1); // VP — bottom-ish
        tti[14] = 0x02; // JC — centered
        tti[15] = 0; // CF
                     // Fill text field.
        let encoded = encode_text_field(&cue.segments, &cpn);
        let copy_len = encoded.len().min(112);
        tti[16..16 + copy_len].copy_from_slice(&encoded[..copy_len]);
        // Pad with 0x8F.
        for b in &mut tti[16 + copy_len..128] {
            *b = 0x8F;
        }
        out.extend_from_slice(&tti);
    }

    Ok(out)
}

/// Probe score — needs the full GSI header to be confident.
pub fn probe(buf: &[u8]) -> u8 {
    looks_like_ebu_stl(buf)
}

pub fn looks_like_ebu_stl(buf: &[u8]) -> u8 {
    if buf.len() < 16 {
        return 0;
    }
    // CPN: 3 ASCII digits (e.g. "850", "437").
    let cpn_ok = buf[..3].iter().all(|&b| b.is_ascii_digit());
    // DFC begins with "STL".
    let dfc_ok = &buf[3..6] == b"STL";
    let mut score = 0u8;
    if cpn_ok {
        score += 25;
    }
    if dfc_ok {
        score += 70;
    }
    // DSC is an ASCII char.
    if buf.len() >= 12 && buf[11].is_ascii() && !buf[11].is_ascii_control() {
        score = score.saturating_add(5);
    }
    score.min(100)
}

pub fn make_decoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Decoder>> {
    crate::codec::make_decoder(params)
}

pub fn make_encoder(
    params: &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_codec::Encoder>> {
    crate::codec::make_encoder(params)
}

// ---------------------------------------------------------------------------
// Cue <-> bytes helpers (used by the codec wiring).

pub(crate) fn cue_to_bytes(cue: &SubtitleCue) -> Vec<u8> {
    // For the per-packet transport, emit a single TTI row (128 bytes).
    let fps = 25.0; // default if not known
    let mut tti = [0u8; TTI_SIZE];
    tti[3] = 0xFF; // EBN = last
    let tci = us_to_tc(cue.start_us, fps);
    tti[5] = tci.0;
    tti[6] = tci.1;
    tti[7] = tci.2;
    tti[8] = tci.3;
    let tco = us_to_tc(cue.end_us, fps);
    tti[9] = tco.0;
    tti[10] = tco.1;
    tti[11] = tco.2;
    tti[12] = tco.3;
    tti[14] = 0x02; // centered
    let encoded = encode_text_field(&cue.segments, "850");
    let copy_len = encoded.len().min(112);
    tti[16..16 + copy_len].copy_from_slice(&encoded[..copy_len]);
    for b in &mut tti[16 + copy_len..128] {
        *b = 0x8F;
    }
    tti.to_vec()
}

pub(crate) fn bytes_to_cue(bytes: &[u8]) -> Result<SubtitleCue> {
    if bytes.len() < TTI_SIZE {
        return Err(Error::invalid("EBU STL TTI: short"));
    }
    let rec = &bytes[..TTI_SIZE];
    let tci = (rec[5], rec[6], rec[7], rec[8]);
    let tco = (rec[9], rec[10], rec[11], rec[12]);
    let tf = &rec[16..128];
    let fps = 25.0;
    Ok(SubtitleCue {
        start_us: tc_to_us(tci, fps),
        end_us: tc_to_us(tco, fps),
        style_ref: None,
        positioning: None,
        segments: decode_text_field(tf, "850"),
    })
}

// ---------------------------------------------------------------------------
// Text-field codec.

fn decode_text_field(tf: &[u8], _cpn: &str) -> Vec<Segment> {
    // Flat run-based decoder. Each time a style flips we close the
    // current run (text) and push it wrapped in the appropriate
    // Segment::Italic / Underline nests. Colour / unknown control bytes
    // are emitted as Raw so a round-trip can replay them.
    let mut segs: Vec<Segment> = Vec::new();
    let mut run = String::new();
    let mut stack_italic = false;
    let mut stack_underline = false;

    fn flush(buf: &mut String, it: bool, un: bool, out: &mut Vec<Segment>) {
        if buf.is_empty() {
            return;
        }
        let text = std::mem::take(buf);
        let mut node: Vec<Segment> = vec![Segment::Text(text)];
        if un {
            node = vec![Segment::Underline(node)];
        }
        if it {
            node = vec![Segment::Italic(node)];
        }
        out.extend(node);
    }

    for &b in tf {
        if b == 0x8F {
            // Trailing pad — stop.
            break;
        }
        if b == 0x8A {
            // Line break.
            flush(&mut run, stack_italic, stack_underline, &mut segs);
            segs.push(Segment::LineBreak);
            continue;
        }
        // Style toggles (one common mapping — 0x80..0x87 are attribute
        // start/end in Tech 3264; we map italic + underline + boxing).
        match b {
            0x80 => {
                // Italic on.
                flush(&mut run, stack_italic, stack_underline, &mut segs);
                stack_italic = true;
                continue;
            }
            0x81 => {
                // Italic off.
                flush(&mut run, stack_italic, stack_underline, &mut segs);
                stack_italic = false;
                continue;
            }
            0x82 => {
                // Underline on.
                flush(&mut run, stack_italic, stack_underline, &mut segs);
                stack_underline = true;
                continue;
            }
            0x83 => {
                // Underline off.
                flush(&mut run, stack_italic, stack_underline, &mut segs);
                stack_underline = false;
                continue;
            }
            _ => {}
        }
        // Colour codes 0x00..0x07 (teletext) — preserve raw for round-trip.
        if b <= 0x07 {
            flush(&mut run, stack_italic, stack_underline, &mut segs);
            segs.push(Segment::Raw(format!("\\x{:02X}", b)));
            continue;
        }
        // Other control codes 0x08..0x1F, 0x84..0x8F (excluding the ones
        // handled above).
        if b < 0x20 || (0x80..=0x9F).contains(&b) {
            flush(&mut run, stack_italic, stack_underline, &mut segs);
            segs.push(Segment::Raw(format!("\\x{:02X}", b)));
            continue;
        }
        // Printable — Latin-1 style (our simplified CCIR-1 interpretation
        // for this first cut).
        run.push(b as char);
    }
    flush(&mut run, stack_italic, stack_underline, &mut segs);
    segs
}

fn encode_text_field(segs: &[Segment], _cpn: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut italic = false;
    let mut underline = false;
    walk_encode(segs, &mut out, &mut italic, &mut underline);
    // If styles are still open, close them.
    if italic {
        out.push(0x81);
    }
    if underline {
        out.push(0x83);
    }
    out
}

fn walk_encode(segs: &[Segment], out: &mut Vec<u8>, italic: &mut bool, underline: &mut bool) {
    for s in segs {
        match s {
            Segment::Text(t) => {
                for c in t.chars() {
                    if (c as u32) <= 0xFF {
                        let b = c as u8;
                        // Avoid emitting a control byte as text.
                        if b < 0x20 || (0x80..=0x9F).contains(&b) {
                            // Replace with '?'.
                            out.push(b'?');
                        } else {
                            out.push(b);
                        }
                    } else {
                        out.push(b'?');
                    }
                }
            }
            Segment::LineBreak => out.push(0x8A),
            Segment::Italic(c) => {
                if !*italic {
                    out.push(0x80);
                    *italic = true;
                }
                walk_encode(c, out, italic, underline);
                out.push(0x81);
                *italic = false;
            }
            Segment::Bold(c) => walk_encode(c, out, italic, underline),
            Segment::Underline(c) => {
                if !*underline {
                    out.push(0x82);
                    *underline = true;
                }
                walk_encode(c, out, italic, underline);
                out.push(0x83);
                *underline = false;
            }
            Segment::Strike(c)
            | Segment::Color { children: c, .. }
            | Segment::Font { children: c, .. }
            | Segment::Voice { children: c, .. }
            | Segment::Class { children: c, .. }
            | Segment::Karaoke { children: c, .. } => {
                walk_encode(c, out, italic, underline);
            }
            Segment::Timestamp { .. } => {}
            Segment::Raw(r) => {
                // Support the `\xNN` placeholder we emit on decode.
                if let Some(rest) = r.strip_prefix("\\x") {
                    if rest.len() == 2 {
                        if let Ok(v) = u8::from_str_radix(rest, 16) {
                            out.push(v);
                            continue;
                        }
                    }
                }
                // Otherwise, write printable chars from the string.
                for c in r.chars() {
                    if (c as u32) <= 0xFF && !((c as u8) < 0x20) {
                        out.push(c as u8);
                    }
                }
            }
        }
    }
}

/// Merge a new segment list onto the end of an existing one, inserting a
/// line break between them (used for extension blocks).
fn extend_segments(dst: &mut Vec<Segment>, mut addl: Vec<Segment>) {
    if !dst.is_empty() {
        dst.push(Segment::LineBreak);
    }
    dst.append(&mut addl);
}

// ---------------------------------------------------------------------------
// Timecode helpers.

fn fps_from_dfc(dfc: &str) -> f32 {
    // DFC examples: "STL25.01", "STL30.01", "STL24.01".
    let trimmed = dfc.trim();
    if trimmed.starts_with("STL") && trimmed.len() >= 5 {
        let body = &trimmed[3..];
        let num: String = body.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(v) = num.parse::<u32>() {
            return v as f32;
        }
    }
    25.0
}

fn tc_to_us(tc: (u8, u8, u8, u8), fps: f32) -> i64 {
    let (h, m, s, f) = tc;
    let fps = fps.max(1.0);
    let frame_us = 1_000_000.0 / fps;
    (h as i64) * 3_600_000_000
        + (m as i64) * 60_000_000
        + (s as i64) * 1_000_000
        + (f as i64) * (frame_us as i64)
}

fn us_to_tc(us: i64, fps: f32) -> (u8, u8, u8, u8) {
    let us = us.max(0);
    let fps = fps.max(1.0);
    let total_s = us / 1_000_000;
    let remain_us = us - total_s * 1_000_000;
    let frames = ((remain_us as f64) * (fps as f64) / 1_000_000.0).floor() as i64;
    let h = (total_s / 3_600) as u8;
    let m = ((total_s / 60) % 60) as u8;
    let s = (total_s % 60) as u8;
    let f = (frames as u8).min((fps as u8).saturating_sub(1).max(24));
    (h, m, s, f)
}

// ---------------------------------------------------------------------------
// GSI helpers.

fn ascii_u32(bytes: &[u8]) -> Option<u32> {
    let s = std::str::from_utf8(bytes).ok()?.trim();
    s.parse::<u32>().ok()
}

fn write_ascii_fixed(dst: &mut [u8], s: &str, len: usize) {
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(len);
    for (i, b) in dst.iter_mut().enumerate().take(len) {
        *b = if i < copy_len { bytes[i] } else { b' ' };
    }
}

fn meta_or(meta: &[(String, String)], key: &str, fallback: &str) -> String {
    meta.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| fallback.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_synthetic() -> Vec<u8> {
        let mut out = vec![0x20u8; GSI_SIZE];
        out[0..3].copy_from_slice(b"850");
        out[3..11].copy_from_slice(b"STL25.01");
        out[11] = b'1';
        out[12..14].copy_from_slice(b"00");
        out[14..16].copy_from_slice(b"00");
        // TNB = 2, TNS = 2, MNC = 40, MNR = 23.
        out[238..243].copy_from_slice(b"00002");
        out[243..248].copy_from_slice(b"00002");
        out[248..250].copy_from_slice(b"40");
        out[250..252].copy_from_slice(b"23");
        out[252] = b'1';

        // Two TTI blocks.
        for (idx, (tci, tco, text)) in [
            ((0u8, 0u8, 1u8, 0u8), (0u8, 0u8, 3u8, 0u8), "Hello world"),
            ((0u8, 0u8, 5u8, 0u8), (0u8, 0u8, 7u8, 0u8), "Second line"),
        ]
        .iter()
        .enumerate()
        {
            let mut tti = [0u8; TTI_SIZE];
            tti[0] = 0;
            let sn = (idx as u16).to_le_bytes();
            tti[1] = sn[0];
            tti[2] = sn[1];
            tti[3] = 0xFF;
            tti[4] = 0;
            tti[5] = tci.0;
            tti[6] = tci.1;
            tti[7] = tci.2;
            tti[8] = tci.3;
            tti[9] = tco.0;
            tti[10] = tco.1;
            tti[11] = tco.2;
            tti[12] = tco.3;
            tti[13] = 22;
            tti[14] = 0x02;
            tti[15] = 0;
            let text_bytes = text.as_bytes();
            tti[16..16 + text_bytes.len()].copy_from_slice(text_bytes);
            for b in &mut tti[16 + text_bytes.len()..128] {
                *b = 0x8F;
            }
            out.extend_from_slice(&tti);
        }
        out
    }

    #[test]
    fn parse_synthetic() {
        let buf = build_synthetic();
        let t = parse(&buf).unwrap();
        assert_eq!(t.cues.len(), 2);
        assert_eq!(t.cues[0].start_us, 1_000_000);
        assert_eq!(t.cues[0].end_us, 3_000_000);
        match &t.cues[0].segments[0] {
            Segment::Text(s) => assert_eq!(s, "Hello world"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_synthetic() {
        let buf = build_synthetic();
        let t = parse(&buf).unwrap();
        let out = write(&t).unwrap();
        assert_eq!(out.len(), GSI_SIZE + 2 * TTI_SIZE);
        // Parse again and check timings preserved.
        let t2 = parse(&out).unwrap();
        assert_eq!(t2.cues.len(), 2);
        assert_eq!(t2.cues[0].start_us, 1_000_000);
        assert_eq!(t2.cues[1].start_us, 5_000_000);
    }

    #[test]
    fn probe_detects() {
        let buf = build_synthetic();
        assert!(probe(&buf) > 60);
        assert_eq!(probe(b"1\n00:00:01,000"), 0);
    }

    #[test]
    fn italic_encoded_as_control_bytes() {
        let cue = SubtitleCue {
            start_us: 0,
            end_us: 1_000_000,
            style_ref: None,
            positioning: None,
            segments: vec![Segment::Italic(vec![Segment::Text("hi".into())])],
        };
        let track = SubtitleTrack {
            cues: vec![cue],
            ..SubtitleTrack::default()
        };
        let out = write(&track).unwrap();
        let tti = &out[GSI_SIZE..GSI_SIZE + TTI_SIZE];
        let tf = &tti[16..];
        assert_eq!(tf[0], 0x80, "expected italic-on at start of TF");
    }
}
