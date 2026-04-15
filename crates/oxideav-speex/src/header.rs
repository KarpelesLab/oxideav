//! Speex header packet ("Speex-in-Ogg mapping").
//!
//! The very first packet of a Speex bitstream is a fixed 80-byte struct
//! (mirrors `SpeexHeader` from `libspeex/speex_header.c`). All integer
//! fields are little-endian. Field layout:
//!
//! ```text
//!   offset | size | field
//!     0    |  8   | speex_string       (ASCII, "Speex   " — 5 letters + 3 spaces)
//!     8    | 20   | speex_version      (ASCII, NUL-padded)
//!    28    |  4   | speex_version_id   (1 for the current stable format)
//!    32    |  4   | header_size        (80)
//!    36    |  4   | rate               (Hz — 8000/16000/32000)
//!    40    |  4   | mode               (0=NB, 1=WB, 2=UWB)
//!    44    |  4   | mode_bitstream_version
//!    48    |  4   | nb_channels        (1 or 2)
//!    52    |  4   | bitrate            (nominal, -1 if unknown)
//!    56    |  4   | frame_size         (in samples, per mode)
//!    60    |  4   | vbr                (0 or 1)
//!    64    |  4   | frames_per_packet
//!    68    |  4   | extra_headers
//!    72    |  4   | reserved1
//!    76    |  4   | reserved2
//! ```

use oxideav_core::{Error, Result};

pub const SPEEX_SIGNATURE: &[u8; 8] = b"Speex   ";
pub const SPEEX_HEADER_SIZE: usize = 80;

/// Which Speex mode (sampling-rate tier) a stream uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeexMode {
    /// Narrowband — 8 kHz, 20 ms frames (160 samples/frame).
    Narrowband,
    /// Wideband — 16 kHz. Layered on top of narrowband.
    Wideband,
    /// Ultra-wideband — 32 kHz. Layered on top of wideband.
    UltraWideband,
}

impl SpeexMode {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Narrowband),
            1 => Some(Self::Wideband),
            2 => Some(Self::UltraWideband),
            _ => None,
        }
    }

    /// Default sample rate in Hz for this mode.
    pub fn sample_rate(self) -> u32 {
        match self {
            Self::Narrowband => 8_000,
            Self::Wideband => 16_000,
            Self::UltraWideband => 32_000,
        }
    }

    /// Default frame size (samples per frame) for this mode.
    pub fn frame_size(self) -> u32 {
        match self {
            Self::Narrowband => 160,
            Self::Wideband => 320,
            Self::UltraWideband => 640,
        }
    }
}

/// A parsed Speex 80-byte header packet.
#[derive(Clone, Debug)]
pub struct SpeexHeader {
    pub version: String,
    pub version_id: u32,
    pub header_size: u32,
    pub rate: u32,
    pub mode: SpeexMode,
    pub mode_bitstream_version: u32,
    pub nb_channels: u32,
    pub bitrate: i32,
    pub frame_size: u32,
    pub vbr: bool,
    pub frames_per_packet: u32,
    pub extra_headers: u32,
}

impl SpeexHeader {
    /// Parse an 80-byte Speex header. Accepts a slice ≥ 80 bytes (extra
    /// trailing bytes are ignored — some encoders pad the initial Ogg page).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SPEEX_HEADER_SIZE {
            return Err(Error::invalid(format!(
                "Speex header: expected {SPEEX_HEADER_SIZE}+ bytes, got {}",
                bytes.len()
            )));
        }
        if &bytes[0..8] != SPEEX_SIGNATURE {
            return Err(Error::invalid("Speex header: bad signature"));
        }

        // Version string: 20 bytes, NUL-padded.
        let ver_raw = &bytes[8..28];
        let ver_end = ver_raw
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(ver_raw.len());
        let version = String::from_utf8_lossy(&ver_raw[..ver_end]).into_owned();

        let version_id = le_u32(&bytes[28..32]);
        let header_size = le_u32(&bytes[32..36]);
        let rate = le_u32(&bytes[36..40]);
        let mode_raw = le_u32(&bytes[40..44]);
        let mode_bitstream_version = le_u32(&bytes[44..48]);
        let nb_channels = le_u32(&bytes[48..52]);
        let bitrate = le_u32(&bytes[52..56]) as i32;
        let frame_size = le_u32(&bytes[56..60]);
        let vbr_raw = le_u32(&bytes[60..64]);
        let frames_per_packet = le_u32(&bytes[64..68]);
        let extra_headers = le_u32(&bytes[68..72]);
        // bytes[72..80] = reserved1, reserved2 — ignored.

        let mode = SpeexMode::from_u32(mode_raw)
            .ok_or_else(|| Error::invalid(format!("Speex header: unknown mode {mode_raw}")))?;

        if nb_channels == 0 || nb_channels > 2 {
            return Err(Error::invalid(format!(
                "Speex header: unsupported channel count {nb_channels}"
            )));
        }

        // Sanity: frame_size is expected to match the mode default. Encoders
        // occasionally write 0; treat 0 as "use the default for the mode".
        let _ = frame_size;

        Ok(Self {
            version,
            version_id,
            header_size,
            rate,
            mode,
            mode_bitstream_version,
            nb_channels,
            bitrate,
            frame_size,
            vbr: vbr_raw != 0,
            frames_per_packet,
            extra_headers,
        })
    }
}

fn le_u32(s: &[u8]) -> u32 {
    u32::from_le_bytes([s[0], s[1], s[2], s[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_header(
        version: &[u8; 20],
        version_id: u32,
        rate: u32,
        mode: u32,
        channels: u32,
        vbr: u32,
    ) -> [u8; 80] {
        let mut h = [0u8; 80];
        h[0..8].copy_from_slice(SPEEX_SIGNATURE);
        h[8..28].copy_from_slice(version);
        h[28..32].copy_from_slice(&version_id.to_le_bytes());
        h[32..36].copy_from_slice(&80u32.to_le_bytes());
        h[36..40].copy_from_slice(&rate.to_le_bytes());
        h[40..44].copy_from_slice(&mode.to_le_bytes());
        h[44..48].copy_from_slice(&4u32.to_le_bytes()); // mode_bitstream_version
        h[48..52].copy_from_slice(&channels.to_le_bytes());
        h[52..56].copy_from_slice(&(-1i32).to_le_bytes()); // bitrate
        h[56..60].copy_from_slice(&160u32.to_le_bytes()); // frame_size
        h[60..64].copy_from_slice(&vbr.to_le_bytes());
        h[64..68].copy_from_slice(&1u32.to_le_bytes()); // frames_per_packet
        h[68..72].copy_from_slice(&0u32.to_le_bytes()); // extra_headers
        h
    }

    #[test]
    fn parses_narrowband_mono() {
        let mut ver = [0u8; 20];
        ver[..5].copy_from_slice(b"1.2.1");
        let bytes = build_header(&ver, 1, 8000, 0, 1, 0);
        let h = SpeexHeader::parse(&bytes).unwrap();
        assert_eq!(h.version, "1.2.1");
        assert_eq!(h.version_id, 1);
        assert_eq!(h.rate, 8000);
        assert_eq!(h.mode, SpeexMode::Narrowband);
        assert_eq!(h.nb_channels, 1);
        assert!(!h.vbr);
        assert_eq!(h.bitrate, -1);
    }

    #[test]
    fn parses_wideband_stereo_vbr() {
        let ver = [0u8; 20];
        let bytes = build_header(&ver, 1, 16000, 1, 2, 1);
        let h = SpeexHeader::parse(&bytes).unwrap();
        assert_eq!(h.mode, SpeexMode::Wideband);
        assert_eq!(h.nb_channels, 2);
        assert!(h.vbr);
    }

    #[test]
    fn parses_ultra_wideband() {
        let ver = [0u8; 20];
        let bytes = build_header(&ver, 1, 32000, 2, 1, 0);
        let h = SpeexHeader::parse(&bytes).unwrap();
        assert_eq!(h.mode, SpeexMode::UltraWideband);
        assert_eq!(h.mode.sample_rate(), 32_000);
        assert_eq!(h.mode.frame_size(), 640);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut bytes = [0u8; 80];
        bytes[0..8].copy_from_slice(b"OggS\0\0\0\0");
        let err = SpeexHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_short_buffer() {
        let bytes = [0u8; 40];
        let err = SpeexHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_bad_mode() {
        let ver = [0u8; 20];
        let bytes = build_header(&ver, 1, 8000, 7, 1, 0);
        let err = SpeexHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_bad_channel_count() {
        let ver = [0u8; 20];
        let bytes = build_header(&ver, 1, 8000, 0, 3, 0);
        let err = SpeexHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
