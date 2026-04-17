//! CABAC (Context-based Adaptive Binary Arithmetic Coding) for H.264.
//!
//! See ITU-T H.264 (07/2019) §9.3.

pub mod binarize;
pub mod context;
pub mod engine;
pub mod tables;
