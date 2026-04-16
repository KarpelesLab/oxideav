//! Core types for the oxideav framework.
//!
//! This crate intentionally depends on nothing but `thiserror`. All codecs,
//! containers, filters, and frontends build on top of the primitives defined
//! here.

pub mod capabilities;
pub mod error;
pub mod format;
pub mod frame;
pub mod packet;
pub mod rational;
pub mod stream;
pub mod time;

pub use capabilities::{CodecCapabilities, CodecPreferences, DEFAULT_PRIORITY};
pub use error::{Error, Result};
pub use format::{MediaType, PixelFormat, SampleFormat};
pub use frame::{AudioFrame, Frame, VideoFrame, VideoPlane};
pub use packet::Packet;
pub use rational::Rational;
pub use stream::{CodecId, CodecParameters, StreamInfo};
pub use time::{TimeBase, Timestamp};
