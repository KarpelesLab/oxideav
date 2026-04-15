//! Media-type and sample/pixel format enumerations.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MediaType {
    Audio,
    Video,
    Subtitle,
    Data,
    Unknown,
}

/// Audio sample format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit, interleaved.
    U8,
    /// Signed 8-bit, interleaved. Native format of Amiga 8SVX and MOD samples.
    S8,
    /// Signed 16-bit little-endian, interleaved.
    S16,
    /// Signed 24-bit packed (3 bytes/sample) little-endian, interleaved.
    S24,
    /// Signed 32-bit little-endian, interleaved.
    S32,
    /// 32-bit IEEE float, interleaved.
    F32,
    /// 64-bit IEEE float, interleaved.
    F64,
    /// Planar variants — one plane per channel.
    U8P,
    S16P,
    S32P,
    F32P,
    F64P,
}

impl SampleFormat {
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::U8P | Self::S16P | Self::S32P | Self::F32P | Self::F64P
        )
    }

    /// Bytes per sample *per channel*.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 | Self::U8P | Self::S8 => 1,
            Self::S16 | Self::S16P => 2,
            Self::S24 => 3,
            Self::S32 | Self::S32P | Self::F32 | Self::F32P => 4,
            Self::F64 | Self::F64P => 8,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F32P | Self::F64P)
    }
}

/// Video pixel format. Only a handful are declared up front; more can be added
/// as codec crates land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit YUV 4:2:0, planar (Y, U, V).
    Yuv420P,
    /// 8-bit YUV 4:2:2, planar.
    Yuv422P,
    /// 8-bit YUV 4:4:4, planar.
    Yuv444P,
    /// Packed 8-bit RGB, 3 bytes/pixel.
    Rgb24,
    /// Packed 8-bit RGBA, 4 bytes/pixel.
    Rgba,
    /// Packed 8-bit grayscale.
    Gray8,
}
