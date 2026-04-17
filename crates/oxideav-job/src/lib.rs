//! JSON-based transcode job graph for oxideav.
//!
//! A *job* is a JSON object whose keys are either output-file paths or named
//! aliases (keys that start with `@`). Each value describes a set of tracks —
//! `audio`, `video`, `subtitle`, or `all` — and each track declares how its
//! samples/frames flow from an input: a source file, an alias reference, or
//! a chain of filters that themselves have inputs.
//!
//! Example:
//!
//! ```json
//! {
//!   "@input": { "all": [{ "from": "movie.mp4" }] },
//!   "out.mkv": {
//!     "audio": [{ "from": "@input", "stream_selector": {"type": "audio", "index": 0} }],
//!     "video": [{ "from": "@input", "codec": "h264", "params": {"crf": 23} }]
//!   }
//! }
//! ```
//!
//! Reserved sink names:
//! - `@null`   — discard (useful for dry-runs + tests)
//! - `@display` / `@out` — live-playback sink (bound by `oxideplay`)

pub mod dag;
pub mod executor;
pub mod pipeline;
pub mod schema;
pub mod sinks;
pub mod validate;

pub use dag::{Dag, DagNode, NodeId};
pub use executor::{Executor, JobSink};
pub use schema::{FilterNode, Job, OutputSpec, SourceRef, StreamSelector, TrackInput, TrackSpec};
pub use sinks::{FileSink, NullSink};
