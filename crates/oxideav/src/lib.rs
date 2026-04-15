//! Aggregator crate for oxideav.
//!
//! Depend on this crate to pull in codecs and containers from the wider
//! oxideav ecosystem, gated by Cargo features. Each format crate maps to
//! exactly one feature here:
//!
//! ```toml
//! [dependencies]
//! oxideav = { version = "*", features = ["basic", "ogg", "vorbis", "flac"] }
//! ```

pub use oxideav_codec as codec;
pub use oxideav_container as container;
pub use oxideav_core as core;
pub use oxideav_pipeline as pipeline;

#[cfg(feature = "basic")]
pub use oxideav_basic as basic;
#[cfg(feature = "flac")]
pub use oxideav_flac as flac;
#[cfg(feature = "mkv")]
pub use oxideav_mkv as mkv;
#[cfg(feature = "mp4")]
pub use oxideav_mp4 as mp4;
#[cfg(feature = "ogg")]
pub use oxideav_ogg as ogg;
#[cfg(feature = "opus")]
pub use oxideav_opus as opus;
#[cfg(feature = "vorbis")]
pub use oxideav_vorbis as vorbis;

/// A pair of registries populated with every format enabled at build time.
pub struct Registries {
    pub codecs: oxideav_codec::CodecRegistry,
    pub containers: oxideav_container::ContainerRegistry,
}

impl Registries {
    /// Build a fresh set of registries containing all compiled-in formats.
    pub fn with_all_features() -> Self {
        #[allow(unused_mut)]
        let mut codecs = oxideav_codec::CodecRegistry::new();
        #[allow(unused_mut)]
        let mut containers = oxideav_container::ContainerRegistry::new();

        #[cfg(feature = "basic")]
        {
            oxideav_basic::register_codecs(&mut codecs);
            oxideav_basic::register_containers(&mut containers);
        }
        #[cfg(feature = "ogg")]
        {
            oxideav_ogg::register(&mut containers);
        }
        #[cfg(feature = "vorbis")]
        {
            oxideav_vorbis::register(&mut codecs);
        }
        #[cfg(feature = "opus")]
        {
            oxideav_opus::register(&mut codecs);
        }
        #[cfg(feature = "flac")]
        {
            oxideav_flac::register_codecs(&mut codecs);
            oxideav_flac::register_containers(&mut containers);
        }
        #[cfg(feature = "mkv")]
        {
            oxideav_mkv::register(&mut containers);
        }
        #[cfg(feature = "mp4")]
        {
            oxideav_mp4::register(&mut containers);
        }

        Self { codecs, containers }
    }
}

impl Default for Registries {
    fn default() -> Self {
        Self::with_all_features()
    }
}
