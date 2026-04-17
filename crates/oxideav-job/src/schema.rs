//! Schema for the JSON job graph + serde (de)serialisation.
//!
//! The top-level document is a JSON object. Keys that start with `@` define
//! named aliases consumable by other entries; all other keys are treated as
//! output sinks (file paths or reserved sink names like `@null`).
//!
//! `TrackInput` is the recursive node type — each filter takes exactly one
//! upstream input today (multi-input fan-in is a future extension).

use indexmap::IndexMap;
use oxideav_core::{Error, MediaType, Result};
use serde::{Deserialize, Serialize};

/// Top-level job: a set of named outputs + aliases.
#[derive(Clone, Debug, Default)]
pub struct Job {
    /// Output targets keyed by filename or reserved sink name (`@null`,
    /// `@display`, `@out`).
    pub outputs: IndexMap<String, OutputSpec>,
    /// Named intermediate aliases (keys starting with `@` that are not
    /// reserved sink names).
    pub aliases: IndexMap<String, OutputSpec>,
    /// Advisory thread budget for the executor. `None` = auto-detect
    /// (use the number of logical CPUs). `Some(1)` forces the serial
    /// executor; `Some(n)` with n ≥ 2 requests pipelined execution.
    /// Explicit CLI overrides (`Executor::with_threads`) take precedence.
    pub threads: Option<usize>,
}

/// Per-file/per-alias spec: track lists grouped by media type.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<TrackSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub video: Vec<TrackSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subtitle: Vec<TrackSpec>,
    /// Tracks that should be pulled across media types. Resolved to
    /// kind-specific lists at DAG-build time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub all: Vec<TrackSpec>,
}

impl OutputSpec {
    /// True when no tracks at all are declared — an error at validation time.
    pub fn is_empty(&self) -> bool {
        self.audio.is_empty()
            && self.video.is_empty()
            && self.subtitle.is_empty()
            && self.all.is_empty()
    }
}

/// A single track: an input chain plus optional encoder settings.
///
/// We do not use `deny_unknown_fields` here because `#[serde(flatten)]` on
/// `input` lifts either `SourceRef` or `FilterNode` fields up to the track
/// level — strict rejection wouldn't distinguish them from truly unknown
/// keys. The builder still catches empty / inconsistent specs in the DAG
/// resolve step.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrackSpec {
    /// Recursive input (source or filter). We flatten so callers can write
    /// either `{"from": ...}` or `{"filter": ..., "input": ...}` directly
    /// on the track.
    #[serde(flatten)]
    pub input: TrackInput,
    /// Output codec id (e.g. `"h264"`, `"flac"`). If omitted the track is
    /// stream-copied — only valid when the upstream directly resolves to a
    /// demuxer packet of a codec the target muxer accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    /// Codec-specific tuning (e.g. `{"crf": 23}`). Opaque to the schema —
    /// codec crates interpret their own keys. Named `codec_params` rather
    /// than `params` so it cannot collide with a flattened filter's
    /// `params` when the track itself is a filter node.
    #[serde(default, rename = "codec_params", skip_serializing_if = "is_null_or_empty")]
    pub params: serde_json::Value,
    /// Optional stream filter applied after the upstream source/filter
    /// emits N streams.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_selector: Option<StreamSelector>,
}

/// Recursive input node.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TrackInput {
    /// `{"from": "path-or-@alias"}`.
    Source(SourceRef),
    /// `{"filter": "name", "params": {...}, "input": <TrackInput>}`.
    Filter(FilterNode),
}

/// Leaf input: either a file path or an alias reference.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceRef {
    /// Filename opened via the source registry, or `@alias` referencing
    /// another top-level entry.
    pub from: String,
}

/// Filter node — single-input for now.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilterNode {
    /// Filter id. Unknown filters error at DAG-build, not parse — so the
    /// caller can still report a precise location.
    pub filter: String,
    /// Filter-specific parameters (opaque JSON).
    #[serde(default, skip_serializing_if = "is_null_or_empty")]
    pub params: serde_json::Value,
    /// Upstream node.
    pub input: Box<TrackInput>,
}

/// Selector for multi-stream inputs. When `kind` is omitted we default to
/// the context kind (e.g. a selector inside `"audio": [...]` only pulls
/// audio streams even if the upstream produces more).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSelector {
    /// `"audio"` / `"video"` / `"subtitle"`. Case-insensitive on the wire.
    #[serde(
        default,
        rename = "type",
        alias = "kind",
        skip_serializing_if = "Option::is_none",
        deserialize_with = "de_media_type_opt",
        serialize_with = "ser_media_type_opt"
    )]
    pub kind: Option<MediaType>,
    /// 0-based index within the filtered pool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

fn is_null_or_empty(v: &serde_json::Value) -> bool {
    v.is_null() || v.as_object().map(|m| m.is_empty()).unwrap_or(false)
}

fn de_media_type_opt<'de, D>(d: D) -> std::result::Result<Option<MediaType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(d)?;
    Ok(match s.as_deref().map(|s| s.trim().to_ascii_lowercase()) {
        Some(ref s) if s == "audio" => Some(MediaType::Audio),
        Some(ref s) if s == "video" => Some(MediaType::Video),
        Some(ref s) if s == "subtitle" || s == "subtitles" => Some(MediaType::Subtitle),
        Some(ref s) if s == "data" => Some(MediaType::Data),
        None => None,
        Some(other) => {
            return Err(serde::de::Error::custom(format!(
                "unknown stream type {other:?} (expected audio|video|subtitle|data)"
            )));
        }
    })
}

fn ser_media_type_opt<S>(v: &Option<MediaType>, s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match v {
        None => s.serialize_none(),
        Some(MediaType::Audio) => s.serialize_str("audio"),
        Some(MediaType::Video) => s.serialize_str("video"),
        Some(MediaType::Subtitle) => s.serialize_str("subtitle"),
        Some(MediaType::Data) => s.serialize_str("data"),
        Some(MediaType::Unknown) => s.serialize_str("unknown"),
    }
}

/// Reserved sink names (all start with `@`). These are **not** aliases —
/// they bind to built-in or caller-supplied sinks at execution time.
pub const RESERVED_SINKS: &[&str] = &["@null", "@display", "@out", "@stdout"];

impl Job {
    /// Parse a `Job` from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| Error::invalid(format!("job: JSON parse error: {e}")))?;
        Self::from_value(v)
    }

    /// Parse a `Job` from an already-decoded `serde_json::Value`.
    pub fn from_value(v: serde_json::Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| Error::invalid("job: top level must be an object"))?;
        let mut job = Job::default();
        // Reserved meta keys — they describe the job itself rather than
        // a named alias or output. Pulled off before the walker runs so
        // parse errors on them give precise messages.
        if let Some(t) = obj.get("threads") {
            let n = t
                .as_u64()
                .ok_or_else(|| Error::invalid("job: `threads` must be a non-negative integer"))?;
            if n == 0 {
                return Err(Error::invalid(
                    "job: `threads` must be ≥ 1 (use CLI `--threads 0` for auto)",
                ));
            }
            job.threads = Some(n as usize);
        }
        for (key, val) in obj {
            if is_meta_key(key) {
                continue;
            }
            let spec: OutputSpec = serde_json::from_value(val.clone())
                .map_err(|e| Error::invalid(format!("job: {key}: {e}")))?;
            if key.is_empty() {
                return Err(Error::invalid("job: empty top-level key"));
            }
            if key.starts_with('@') && !RESERVED_SINKS.contains(&key.as_str()) {
                job.aliases.insert(key.clone(), spec);
            } else {
                job.outputs.insert(key.clone(), spec);
            }
        }
        Ok(job)
    }

    /// Serialise back to pretty-printed JSON (useful for `dry-run` dumps).
    pub fn to_json_pretty(&self) -> String {
        let mut merged: IndexMap<&String, &OutputSpec> = IndexMap::new();
        for (k, v) in &self.aliases {
            merged.insert(k, v);
        }
        for (k, v) in &self.outputs {
            merged.insert(k, v);
        }
        serde_json::to_string_pretty(&merged).unwrap_or_default()
    }
}

/// True when the given top-level key is a reserved sink name.
pub fn is_reserved_sink(name: &str) -> bool {
    RESERVED_SINKS.contains(&name)
}

/// Keys reserved for job metadata (not outputs or aliases). Pulled off
/// the top-level object before the output/alias walk.
const META_KEYS: &[&str] = &["threads"];

fn is_meta_key(name: &str) -> bool {
    META_KEYS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_alias_and_output() {
        let job = Job::from_json(
            r#"{
                "@input": {"all": [{"from": "a.mp4"}]},
                "out.mkv": {
                    "audio": [{"from": "@input"}],
                    "video": [{"from": "@input"}]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(job.aliases.len(), 1);
        assert_eq!(job.outputs.len(), 1);
        assert!(job.aliases.contains_key("@input"));
        assert!(job.outputs.contains_key("out.mkv"));
        let out = &job.outputs["out.mkv"];
        assert_eq!(out.audio.len(), 1);
        assert_eq!(out.video.len(), 1);
    }

    #[test]
    fn parses_filter_chain() {
        let job = Job::from_json(
            r#"{
                "out.flac": {
                    "audio": [{
                        "filter": "volume",
                        "params": {"gain_db": -3},
                        "input": {
                            "filter": "resample",
                            "params": {"rate": 48000},
                            "input": {"from": "in.wav"}
                        }
                    }]
                }
            }"#,
        )
        .unwrap();
        let track = &job.outputs["out.flac"].audio[0];
        match &track.input {
            TrackInput::Filter(f) => {
                assert_eq!(f.filter, "volume");
                match f.input.as_ref() {
                    TrackInput::Filter(inner) => assert_eq!(inner.filter, "resample"),
                    _ => panic!("expected inner filter"),
                }
            }
            _ => panic!("expected outer filter"),
        }
    }

    #[test]
    fn stream_selector_accepts_type_and_kind() {
        let j = Job::from_json(
            r#"{"o.wav": {"audio": [{"from": "x", "stream_selector": {"type": "audio", "index": 1}}]}}"#,
        ).unwrap();
        let sel = j.outputs["o.wav"].audio[0].stream_selector.as_ref().unwrap();
        assert_eq!(sel.kind, Some(MediaType::Audio));
        assert_eq!(sel.index, Some(1));

        let j = Job::from_json(
            r#"{"o.wav": {"audio": [{"from": "x", "stream_selector": {"kind": "subtitles"}}]}}"#,
        )
        .unwrap();
        let sel = j.outputs["o.wav"].audio[0].stream_selector.as_ref().unwrap();
        assert_eq!(sel.kind, Some(MediaType::Subtitle));
    }

    #[test]
    fn parses_threads_meta_key() {
        let j = Job::from_json(
            r#"{"threads": 4, "out.wav": {"audio": [{"from": "in.wav"}]}}"#,
        )
        .unwrap();
        assert_eq!(j.threads, Some(4));
        assert_eq!(j.outputs.len(), 1);
        assert!(j.aliases.is_empty());
    }

    #[test]
    fn rejects_zero_threads() {
        let e = Job::from_json(r#"{"threads": 0, "out.wav": {"audio": []}}"#).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("≥ 1") || msg.contains(">= 1"), "got: {msg}");
    }

    #[test]
    fn reserved_sink_is_not_alias() {
        let j = Job::from_json(r#"{"@display": {"video": [{"from": "x"}]}}"#).unwrap();
        assert!(j.outputs.contains_key("@display"));
        assert!(j.aliases.is_empty());
    }

    #[test]
    fn rejects_non_object_top_level() {
        assert!(Job::from_json("42").is_err());
        assert!(Job::from_json("[]").is_err());
    }

    #[test]
    fn parses_codec_params_field() {
        // Track-level encoder tuning lives under `codec_params` so it can't
        // collide with a flattened filter's own `params`.
        let j = Job::from_json(
            r#"{"o.mkv": {"video": [{"from": "x", "codec": "h264", "codec_params": {"crf": 23}}]}}"#,
        )
        .unwrap();
        let t = &j.outputs["o.mkv"].video[0];
        assert_eq!(t.codec.as_deref(), Some("h264"));
        assert_eq!(t.params, serde_json::json!({"crf": 23}));
    }
}
