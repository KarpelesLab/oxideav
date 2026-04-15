//! ProTracker / SoundTracker MOD header parser.
//!
//! Layout (little-endian is not used — all multi-byte fields are
//! big-endian):
//!
//! ```text
//! Offset 0        20 bytes      Song title (null-padded ASCII)
//! Offset 20      31 * 30 bytes  Sample definitions:
//!                                 22 bytes name
//!                                  2 bytes length (in 16-bit words, BE)
//!                                  1 byte  finetune  (low 4 bits, signed)
//!                                  1 byte  volume    (0..64)
//!                                  2 bytes repeat-start (words, BE)
//!                                  2 bytes repeat-length (words, BE)
//! Offset 950      1 byte        Song length (1..128)
//! Offset 951      1 byte        Restart byte (0x7F typical)
//! Offset 952    128 bytes       Pattern-order table
//! Offset 1080     4 bytes       Signature: "M.K.", "M!K!", "4CHN",
//!                                "6CHN", "8CHN", "xxCH" (xx 10..32)
//! Offset 1084      …            Pattern data: 64 rows × channels × 4 bytes
//! After patterns               Raw sample bodies (signed 8-bit)
//! ```

use oxideav_core::{Error, Result};

pub const HEADER_FIXED_SIZE: usize = 1084;
pub const PATTERN_ROWS: usize = 64;
pub const SAMPLE_COUNT: usize = 31;
pub const ORDER_TABLE_SIZE: usize = 128;

#[derive(Clone, Debug)]
pub struct Sample {
    pub name: String,
    /// Sample length in *samples* (spec stores words — we've doubled).
    pub length: u32,
    /// Finetune value, signed 4-bit (-8..=7).
    pub finetune: i8,
    /// Volume 0..=64.
    pub volume: u8,
    /// Loop start in samples.
    pub repeat_start: u32,
    /// Loop length in samples (0 or 2 = no loop).
    pub repeat_length: u32,
}

#[derive(Clone, Debug)]
pub struct ModHeader {
    pub title: String,
    pub samples: Vec<Sample>,
    pub song_length: u8,
    pub restart: u8,
    pub order: Vec<u8>,
    pub signature: [u8; 4],
    pub channels: u8,
    /// Number of distinct patterns referenced by the order table.
    pub n_patterns: u8,
}

impl ModHeader {
    /// Total size of the header block preceding sample data (in bytes).
    pub fn pattern_data_offset(&self) -> usize {
        HEADER_FIXED_SIZE
    }

    /// Size of the pattern data region in bytes.
    pub fn pattern_data_size(&self) -> usize {
        self.n_patterns as usize * PATTERN_ROWS * self.channels as usize * 4
    }

    /// Absolute offset where sample bodies begin.
    pub fn sample_data_offset(&self) -> usize {
        HEADER_FIXED_SIZE + self.pattern_data_size()
    }
}

pub fn parse_header(bytes: &[u8]) -> Result<ModHeader> {
    if bytes.len() < HEADER_FIXED_SIZE {
        return Err(Error::NeedMore);
    }
    let title = read_padded_ascii(&bytes[0..20]);

    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for i in 0..SAMPLE_COUNT {
        let off = 20 + i * 30;
        let name = read_padded_ascii(&bytes[off..off + 22]);
        let len_words = u16::from_be_bytes([bytes[off + 22], bytes[off + 23]]) as u32;
        let finetune_raw = bytes[off + 24] & 0x0F;
        let finetune = if finetune_raw & 0x08 != 0 {
            (finetune_raw as i8) - 16
        } else {
            finetune_raw as i8
        };
        let volume = bytes[off + 25].min(64);
        let repeat_start_words = u16::from_be_bytes([bytes[off + 26], bytes[off + 27]]) as u32;
        let repeat_length_words = u16::from_be_bytes([bytes[off + 28], bytes[off + 29]]) as u32;
        samples.push(Sample {
            name,
            length: len_words.saturating_mul(2),
            finetune,
            volume,
            repeat_start: repeat_start_words.saturating_mul(2),
            repeat_length: repeat_length_words.saturating_mul(2),
        });
    }

    let song_length = bytes[950];
    let restart = bytes[951];
    let order: Vec<u8> = bytes[952..952 + ORDER_TABLE_SIZE].to_vec();

    let mut signature = [0u8; 4];
    signature.copy_from_slice(&bytes[1080..1084]);
    let channels = channels_from_signature(&signature)?;

    let n_patterns = 1 + *order.iter().take(song_length as usize).max().unwrap_or(&0);

    Ok(ModHeader {
        title,
        samples,
        song_length,
        restart,
        order,
        signature,
        channels,
        n_patterns,
    })
}

fn channels_from_signature(sig: &[u8; 4]) -> Result<u8> {
    match sig {
        b"M.K." | b"M!K!" | b"FLT4" | b"4CHN" => Ok(4),
        b"6CHN" => Ok(6),
        b"8CHN" | b"OCTA" | b"CD81" | b"FLT8" => Ok(8),
        // "xxCH" with xx in 10..=32 (Fast Tracker / TakeTracker).
        other if other[2] == b'C' && other[3] == b'H' => {
            let tens = (other[0] as char).to_digit(10);
            let ones = (other[1] as char).to_digit(10);
            match (tens, ones) {
                (Some(t), Some(o)) => {
                    let n = (t * 10 + o) as u8;
                    if (10..=32).contains(&n) {
                        Ok(n)
                    } else {
                        Err(Error::unsupported(format!(
                            "MOD: unsupported channel count {n}"
                        )))
                    }
                }
                _ => Err(Error::invalid(format!(
                    "MOD: unknown signature {:?}",
                    std::str::from_utf8(other).unwrap_or("????")
                ))),
            }
        }
        _ => Err(Error::invalid(format!(
            "MOD: unknown signature {:?}",
            std::str::from_utf8(sig).unwrap_or("????")
        ))),
    }
}

fn read_padded_ascii(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end])
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fake_mod(channels: &[u8; 4], song_length: u8) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_FIXED_SIZE];
        out[0..8].copy_from_slice(b"test\0\0\0\0");
        // sample 0: empty
        // song length + order table
        out[950] = song_length;
        out[951] = 0x7F;
        for i in 0..song_length as usize {
            out[952 + i] = 0;
        }
        out[1080..1084].copy_from_slice(channels);
        out
    }

    #[test]
    fn signature_mk() {
        let h = parse_header(&make_fake_mod(b"M.K.", 1)).unwrap();
        assert_eq!(h.channels, 4);
        assert_eq!(h.signature, *b"M.K.");
        assert_eq!(h.song_length, 1);
        assert_eq!(h.samples.len(), 31);
    }

    #[test]
    fn signature_6chn() {
        let h = parse_header(&make_fake_mod(b"6CHN", 2)).unwrap();
        assert_eq!(h.channels, 6);
    }

    #[test]
    fn signature_14ch() {
        let h = parse_header(&make_fake_mod(b"14CH", 1)).unwrap();
        assert_eq!(h.channels, 14);
    }

    #[test]
    fn rejects_unknown_signature() {
        let bytes = make_fake_mod(b"XXXX", 1);
        assert!(parse_header(&bytes).is_err());
    }
}
