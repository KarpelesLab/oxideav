//! Pure-Rust Matroska (MKV/WebM) container.
//!
//! Implements the EBML primitives plus enough of the Matroska schema to
//! demux the audio codecs oxideav already understands (FLAC, Opus, Vorbis,
//! PCM). The muxer can write back any codec we can carry — there are no
//! codec-specific assumptions in the container layer.
//!
//! WebM is exposed as a first-class peer of Matroska: the demuxer is
//! shared (the two formats are byte-identical except for the DocType
//! string and a restricted codec set), but the registry holds separate
//! `"matroska"` and `"webm"` entries with their own probes and muxer
//! factories. The WebM muxer enforces the [WebM container
//! guidelines](https://www.webmproject.org/docs/container/): only VP8,
//! VP9, AV1 for video and Vorbis, Opus for audio.

pub mod codec_id;
pub mod demux;
pub mod ebml;
pub mod ids;
pub mod mux;

use oxideav_container::ContainerRegistry;

/// Register both the `"matroska"` and `"webm"` containers.
///
/// They share a demuxer factory but each gets its own probe and muxer
/// factory, so callers asking for `"webm"` explicitly get the WebM muxer
/// (codec whitelist enforced, `DocType="webm"` in the EBML header) and
/// callers asking for `"matroska"` get the general muxer.
pub fn register(reg: &mut ContainerRegistry) {
    // Matroska entry.
    reg.register_demuxer("matroska", demux::open);
    reg.register_muxer("matroska", mux::open);
    reg.register_probe("matroska", probe_matroska);

    // WebM entry — same demuxer, dedicated muxer and probe.
    reg.register_demuxer("webm", demux::open);
    reg.register_muxer("webm", mux::open_webm);
    reg.register_probe("webm", probe_webm);

    // Extensions.
    reg.register_extension("mkv", "matroska");
    reg.register_extension("mka", "matroska");
    reg.register_extension("mks", "matroska");
    reg.register_extension("webm", "webm");
}

/// EBML signature at offset 0 — common to both Matroska and WebM.
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

/// Probe score returned when the on-disk DocType matches exactly. The
/// registry picks the highest scorer across all registered probes, so
/// this needs to beat the "signature-only" fallback score.
const SCORE_DOCTYPE_MATCH: u8 = 100;

/// Probe score returned when we recognise the EBML signature but the
/// DocType points at the other flavour (still a valid fallback — either
/// demuxer can read either flavour).
const SCORE_SIGNATURE_ONLY: u8 = 60;

/// Matroska probe: high score if DocType reads "matroska", moderate
/// score on any EBML file (accepts WebM as a fallback).
fn probe_matroska(p: &oxideav_container::ProbeData) -> u8 {
    match probe_doctype(p.buf) {
        DocTypeProbe::Matroska => SCORE_DOCTYPE_MATCH,
        DocTypeProbe::Webm => SCORE_SIGNATURE_ONLY,
        DocTypeProbe::EbmlOnly => SCORE_SIGNATURE_ONLY,
        DocTypeProbe::NotEbml => 0,
    }
}

/// WebM probe: high score if DocType reads "webm", zero otherwise so a
/// plain .mkv does not get reported as webm. (A truly ambiguous case —
/// EBML magic but no readable DocType — falls through to the matroska
/// entry, which is what ffmpeg/Chromium do.)
fn probe_webm(p: &oxideav_container::ProbeData) -> u8 {
    match probe_doctype(p.buf) {
        DocTypeProbe::Webm => SCORE_DOCTYPE_MATCH,
        DocTypeProbe::Matroska => 0,
        DocTypeProbe::EbmlOnly => 0,
        DocTypeProbe::NotEbml => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DocTypeProbe {
    /// EBML signature seen AND DocType reads exactly "matroska".
    Matroska,
    /// EBML signature seen AND DocType reads exactly "webm".
    Webm,
    /// EBML signature seen, DocType either missing from the buffer or
    /// set to something else.
    EbmlOnly,
    /// No EBML signature at the head of the buffer.
    NotEbml,
}

/// Scan `buf` for the EBML magic and, if found, the `DocType` (0x4282)
/// child element inside the EBML header. Returns a classification the
/// two probe functions can act on. Does not consume I/O; parses the
/// buffer in-place.
fn probe_doctype(buf: &[u8]) -> DocTypeProbe {
    if buf.len() < 4 || buf[0..4] != EBML_MAGIC {
        return DocTypeProbe::NotEbml;
    }
    // Parse the EBML header size (VINT after the 4-byte ID) and walk its
    // children looking for DocType (0x4282). If we can't parse it, return
    // EbmlOnly — neither probe should fail catastrophically on a torn
    // read.
    let mut cur = std::io::Cursor::new(&buf[4..]);
    let (hdr_size, _) = match ebml::read_vint(&mut cur, false) {
        Ok(v) => v,
        Err(_) => return DocTypeProbe::EbmlOnly,
    };
    let hdr_start = 4 + cur.position() as usize;
    let hdr_end = hdr_start.saturating_add(hdr_size as usize);
    let scan_end = hdr_end.min(buf.len());
    let mut pos = hdr_start;
    while pos < scan_end {
        let mut sub = std::io::Cursor::new(&buf[pos..scan_end]);
        let (id, id_len) = match ebml::read_vint(&mut sub, true) {
            Ok(v) => v,
            Err(_) => return DocTypeProbe::EbmlOnly,
        };
        let (size, size_len) = match ebml::read_vint(&mut sub, false) {
            Ok(v) => v,
            Err(_) => return DocTypeProbe::EbmlOnly,
        };
        let data_start = pos + id_len + size_len;
        let data_end = data_start.saturating_add(size as usize);
        if data_end > scan_end {
            return DocTypeProbe::EbmlOnly;
        }
        if id == ids::EBML_DOC_TYPE as u64 {
            let slice = &buf[data_start..data_end];
            // Strip trailing NULs, the common Matroska padding.
            let trimmed = slice.split(|&b| b == 0).next().unwrap_or(&[]);
            return match trimmed {
                b"matroska" => DocTypeProbe::Matroska,
                b"webm" => DocTypeProbe::Webm,
                _ => DocTypeProbe::EbmlOnly,
            };
        }
        pos = data_end;
    }
    DocTypeProbe::EbmlOnly
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny EBML header with a given DocType plus enough
    /// surrounding bytes to look like a real file head.
    fn synth_ebml_head(doc_type: &str) -> Vec<u8> {
        use crate::ebml::{write_element_id, write_vint};
        // Body: EBMLVersion(1), DocType(...), DocTypeVersion(2),
        // DocTypeReadVersion(2).
        let mut body = Vec::new();
        // EBMLVersion (0x4286) = 1
        body.extend_from_slice(&write_element_id(ids::EBML_VERSION));
        body.extend_from_slice(&write_vint(1, 0));
        body.push(0x01);
        // DocType
        body.extend_from_slice(&write_element_id(ids::EBML_DOC_TYPE));
        body.extend_from_slice(&write_vint(doc_type.len() as u64, 0));
        body.extend_from_slice(doc_type.as_bytes());
        // DocTypeVersion
        body.extend_from_slice(&write_element_id(ids::EBML_DOC_TYPE_VERSION));
        body.extend_from_slice(&write_vint(1, 0));
        body.push(0x02);
        // DocTypeReadVersion
        body.extend_from_slice(&write_element_id(ids::EBML_DOC_TYPE_READ_VERSION));
        body.extend_from_slice(&write_vint(1, 0));
        body.push(0x02);
        // Wrap in EBML header element.
        let mut out = Vec::new();
        out.extend_from_slice(&write_element_id(ids::EBML_HEADER));
        out.extend_from_slice(&write_vint(body.len() as u64, 0));
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn probe_doctype_webm() {
        let head = synth_ebml_head("webm");
        assert_eq!(probe_doctype(&head), DocTypeProbe::Webm);
    }

    #[test]
    fn probe_doctype_matroska() {
        let head = synth_ebml_head("matroska");
        assert_eq!(probe_doctype(&head), DocTypeProbe::Matroska);
    }

    #[test]
    fn probe_doctype_non_ebml() {
        let buf = b"RIFF....WAVEfmt ";
        assert_eq!(probe_doctype(buf), DocTypeProbe::NotEbml);
    }

    #[test]
    fn webm_probe_prefers_webm() {
        let head = synth_ebml_head("webm");
        let p = oxideav_container::ProbeData {
            buf: &head,
            ext: None,
        };
        let webm_score = probe_webm(&p);
        let mkv_score = probe_matroska(&p);
        assert!(
            webm_score > mkv_score,
            "webm probe ({}) should outrank matroska probe ({}) on DocType=webm",
            webm_score,
            mkv_score
        );
        assert_eq!(webm_score, SCORE_DOCTYPE_MATCH);
    }

    #[test]
    fn matroska_probe_prefers_matroska() {
        let head = synth_ebml_head("matroska");
        let p = oxideav_container::ProbeData {
            buf: &head,
            ext: None,
        };
        let webm_score = probe_webm(&p);
        let mkv_score = probe_matroska(&p);
        assert!(
            mkv_score > webm_score,
            "matroska probe ({}) should outrank webm probe ({}) on DocType=matroska",
            mkv_score,
            webm_score
        );
        assert_eq!(mkv_score, SCORE_DOCTYPE_MATCH);
        assert_eq!(webm_score, 0);
    }

    #[test]
    fn registry_extension_mapping() {
        let mut reg = ContainerRegistry::new();
        register(&mut reg);
        assert_eq!(reg.container_for_extension("webm"), Some("webm"));
        assert_eq!(reg.container_for_extension("mkv"), Some("matroska"));
        assert_eq!(reg.container_for_extension("mka"), Some("matroska"));
        assert_eq!(reg.container_for_extension("mks"), Some("matroska"));
    }
}
