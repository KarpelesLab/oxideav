//! Attached picture metadata (cover art, artist photos, etc.).
//!
//! Used by containers that can carry embedded image data — notably MP3
//! (via `APIC` / `PIC` frames in an ID3v2 tag), FLAC (via
//! `METADATA_BLOCK_PICTURE`), MP4 (`covr` atoms), and Ogg/Vorbis (base64-
//! encoded `METADATA_BLOCK_PICTURE` inside a Vorbis comment).
//!
//! The picture type values follow the ID3v2 `APIC` spec (which FLAC and
//! Ogg reuse verbatim), so a `PictureType` round-trips cleanly between
//! all four containers.

/// A single attached picture as it appears in a file's metadata.
///
/// `data` is the raw encoded image bytes exactly as stored in the file —
/// the container does not decode the image. Callers that want pixels
/// should feed `data` through the appropriate image decoder (JPEG, PNG,
/// ...) using `mime_type` as a routing hint.
#[derive(Clone, Debug)]
pub struct AttachedPicture {
    /// MIME type of the image payload (`"image/jpeg"`, `"image/png"`,
    /// ...). The special value `"-->"` means `data` is a URL string
    /// pointing to an external image rather than inline bytes (ID3v2).
    pub mime_type: String,
    /// Semantic role of the picture (cover art, artist photo, ...).
    pub picture_type: PictureType,
    /// Human-readable description supplied by the tagger. Often empty.
    pub description: String,
    /// Raw image bytes (or URL bytes when `mime_type == "-->"`).
    pub data: Vec<u8>,
}

/// ID3v2 `APIC` picture-type taxonomy (also reused by FLAC and Vorbis).
///
/// The numeric values match the ID3v2 specification and are stable:
/// callers are free to cast a `PictureType` to `u8`. New values added to
/// future revisions of the spec will land as `Unknown` rather than
/// breaking existing code — the enum is `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum PictureType {
    Other = 0x00,
    FileIcon32x32 = 0x01,
    FileIcon = 0x02,
    FrontCover = 0x03,
    BackCover = 0x04,
    LeafletPage = 0x05,
    Media = 0x06,
    LeadArtist = 0x07,
    Artist = 0x08,
    Conductor = 0x09,
    BandOrchestra = 0x0A,
    Composer = 0x0B,
    Lyricist = 0x0C,
    RecordingLocation = 0x0D,
    DuringRecording = 0x0E,
    DuringPerformance = 0x0F,
    MovieScreenCapture = 0x10,
    ABrightColouredFish = 0x11,
    Illustration = 0x12,
    BandLogo = 0x13,
    PublisherLogo = 0x14,
    /// Catch-all for unrecognised or out-of-range codes.
    Unknown = 0xFF,
}

impl PictureType {
    /// Convert a raw ID3v2/FLAC picture-type byte into a `PictureType`.
    /// Unknown or reserved codes (> 0x14) collapse to `Unknown`.
    pub fn from_u8(b: u8) -> Self {
        match b {
            0x00 => Self::Other,
            0x01 => Self::FileIcon32x32,
            0x02 => Self::FileIcon,
            0x03 => Self::FrontCover,
            0x04 => Self::BackCover,
            0x05 => Self::LeafletPage,
            0x06 => Self::Media,
            0x07 => Self::LeadArtist,
            0x08 => Self::Artist,
            0x09 => Self::Conductor,
            0x0A => Self::BandOrchestra,
            0x0B => Self::Composer,
            0x0C => Self::Lyricist,
            0x0D => Self::RecordingLocation,
            0x0E => Self::DuringRecording,
            0x0F => Self::DuringPerformance,
            0x10 => Self::MovieScreenCapture,
            0x11 => Self::ABrightColouredFish,
            0x12 => Self::Illustration,
            0x13 => Self::BandLogo,
            0x14 => Self::PublisherLogo,
            _ => Self::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picture_type_known_values() {
        assert_eq!(PictureType::from_u8(0x03), PictureType::FrontCover);
        assert_eq!(PictureType::from_u8(0x07), PictureType::LeadArtist);
        assert_eq!(PictureType::from_u8(0x14), PictureType::PublisherLogo);
    }

    #[test]
    fn picture_type_unknown_collapses() {
        assert_eq!(PictureType::from_u8(0x15), PictureType::Unknown);
        assert_eq!(PictureType::from_u8(0xAB), PictureType::Unknown);
    }
}
