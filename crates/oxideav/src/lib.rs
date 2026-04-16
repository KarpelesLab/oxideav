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
pub use oxideav_source as source;

#[cfg(feature = "http")]
pub use oxideav_http as http;

#[cfg(feature = "aac")]
pub use oxideav_aac as aac;
#[cfg(feature = "amv")]
pub use oxideav_amv as amv;
#[cfg(feature = "audio_filter")]
pub use oxideav_audio_filter as audio_filter;
#[cfg(feature = "av1")]
pub use oxideav_av1 as av1;
#[cfg(feature = "avi")]
pub use oxideav_avi as avi;
#[cfg(feature = "basic")]
pub use oxideav_basic as basic;
#[cfg(feature = "celt")]
pub use oxideav_celt as celt;
#[cfg(feature = "flac")]
pub use oxideav_flac as flac;
#[cfg(feature = "g7231")]
pub use oxideav_g7231 as g7231;
#[cfg(feature = "g728")]
pub use oxideav_g728 as g728;
#[cfg(feature = "g729")]
pub use oxideav_g729 as g729;
#[cfg(feature = "gsm")]
pub use oxideav_gsm as gsm;
#[cfg(feature = "h263")]
pub use oxideav_h263 as h263;
#[cfg(feature = "h264")]
pub use oxideav_h264 as h264;
#[cfg(feature = "h265")]
pub use oxideav_h265 as h265;
#[cfg(feature = "iff")]
pub use oxideav_iff as iff;
#[cfg(feature = "mjpeg")]
pub use oxideav_mjpeg as mjpeg;
#[cfg(feature = "mkv")]
pub use oxideav_mkv as mkv;
#[cfg(feature = "amiga_mod")]
pub use oxideav_mod as amiga_mod;
#[cfg(feature = "mp1")]
pub use oxideav_mp1 as mp1;
#[cfg(feature = "mp2")]
pub use oxideav_mp2 as mp2;
#[cfg(feature = "mp3")]
pub use oxideav_mp3 as mp3;
#[cfg(feature = "mp4")]
pub use oxideav_mp4 as mp4;
#[cfg(feature = "mpeg1video")]
pub use oxideav_mpeg1video as mpeg1video;
#[cfg(feature = "mpeg4video")]
pub use oxideav_mpeg4video as mpeg4video;
#[cfg(feature = "ogg")]
pub use oxideav_ogg as ogg;
#[cfg(feature = "opus")]
pub use oxideav_opus as opus;
#[cfg(feature = "s3m")]
pub use oxideav_s3m as s3m;
#[cfg(feature = "speex")]
pub use oxideav_speex as speex;
#[cfg(feature = "theora")]
pub use oxideav_theora as theora;
#[cfg(feature = "vorbis")]
pub use oxideav_vorbis as vorbis;
#[cfg(feature = "vp8")]
pub use oxideav_vp8 as vp8;
#[cfg(feature = "vp9")]
pub use oxideav_vp9 as vp9;
#[cfg(feature = "webp")]
pub use oxideav_webp as webp;

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
        #[cfg(feature = "avi")]
        {
            oxideav_avi::register(&mut containers);
        }
        #[cfg(feature = "iff")]
        {
            oxideav_iff::register(&mut containers);
        }
        #[cfg(feature = "amiga_mod")]
        {
            oxideav_mod::register_codecs(&mut codecs);
            oxideav_mod::register_containers(&mut containers);
        }
        #[cfg(feature = "s3m")]
        {
            oxideav_s3m::register_codecs(&mut codecs);
            oxideav_s3m::register_containers(&mut containers);
        }
        #[cfg(feature = "mp1")]
        {
            oxideav_mp1::register(&mut codecs);
        }
        #[cfg(feature = "mp2")]
        {
            oxideav_mp2::register(&mut codecs);
        }
        #[cfg(feature = "mp3")]
        {
            oxideav_mp3::register_codecs(&mut codecs);
            oxideav_mp3::register_containers(&mut containers);
        }
        #[cfg(feature = "mjpeg")]
        {
            oxideav_mjpeg::register(&mut codecs);
        }
        #[cfg(feature = "mpeg1video")]
        {
            oxideav_mpeg1video::register(&mut codecs);
        }
        #[cfg(feature = "aac")]
        {
            oxideav_aac::register(&mut codecs);
        }
        #[cfg(feature = "celt")]
        {
            oxideav_celt::register(&mut codecs);
        }
        #[cfg(feature = "g7231")]
        {
            oxideav_g7231::register(&mut codecs);
        }
        #[cfg(feature = "g728")]
        {
            oxideav_g728::register(&mut codecs);
        }
        #[cfg(feature = "g729")]
        {
            oxideav_g729::register(&mut codecs);
        }
        #[cfg(feature = "gsm")]
        {
            oxideav_gsm::register(&mut codecs);
        }
        #[cfg(feature = "speex")]
        {
            oxideav_speex::register(&mut codecs);
        }
        #[cfg(feature = "mpeg4video")]
        {
            oxideav_mpeg4video::register(&mut codecs);
        }
        #[cfg(feature = "theora")]
        {
            oxideav_theora::register(&mut codecs);
        }
        #[cfg(feature = "vp9")]
        {
            oxideav_vp9::register(&mut codecs);
        }
        #[cfg(feature = "h265")]
        {
            oxideav_h265::register(&mut codecs);
        }
        #[cfg(feature = "h264")]
        {
            oxideav_h264::register(&mut codecs);
        }
        #[cfg(feature = "h263")]
        {
            oxideav_h263::register(&mut codecs);
        }
        #[cfg(feature = "vp8")]
        {
            oxideav_vp8::register(&mut codecs, &mut containers);
        }
        #[cfg(feature = "webp")]
        {
            oxideav_webp::register(&mut codecs, &mut containers);
        }
        #[cfg(feature = "amv")]
        {
            oxideav_amv::register(&mut codecs, &mut containers);
        }
        #[cfg(feature = "av1")]
        {
            oxideav_av1::register(&mut codecs);
        }

        Self { codecs, containers }
    }
}

impl Default for Registries {
    fn default() -> Self {
        Self::with_all_features()
    }
}
