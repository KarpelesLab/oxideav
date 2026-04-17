//! Resolved intermediate representation: a DAG per output.
//!
//! The resolver walks the parsed [`Job`](crate::Job), inlines alias
//! references, and emits a `Dag` with one `Mux` node per output. Execution
//! works bottom-up from the source demuxer(s) to the mux; copy, decode,
//! filter, and encode nodes sit in between.
//!
//! Aliases are inlined by reference — if the same alias is used by two
//! outputs they each get their own subgraph so the executor can share a
//! single demuxer with per-consumer replay buffering.

use indexmap::IndexMap;
use oxideav_core::{Error, MediaType, PixelFormat, Result};

use crate::schema::{
    parse_pixel_format, Job, OutputSpec, SourceRef, StreamSelector, TrackInput, TrackSpec,
};

/// Opaque index into `Dag::nodes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A single node in the resolved DAG.
#[derive(Clone, Debug)]
pub enum DagNode {
    /// Open a demuxer from a source URI. Emits packets stream-by-stream;
    /// downstream `Select` nodes filter.
    Demuxer { source: String },
    /// Constrain upstream streams to those matching `selector`.
    Select {
        upstream: NodeId,
        selector: ResolvedSelector,
    },
    /// Decode packets to frames.
    Decode { upstream: NodeId },
    /// Apply a filter to frames.
    Filter {
        upstream: NodeId,
        kind: FilterKind,
        name: String,
        params: serde_json::Value,
    },
    /// Convert video frames to a target pixel format. Emitted either
    /// explicitly from a `{"convert": "..."}` schema node, or
    /// auto-inserted by the resolver when an encoder has a non-empty
    /// `accepted_pixel_formats` that excludes the upstream format.
    PixConvert {
        upstream: NodeId,
        target: PixelFormat,
    },
    /// Encode frames to packets using the named codec.
    Encode {
        upstream: NodeId,
        codec: String,
        params: serde_json::Value,
    },
    /// Terminal mux node — collects tracks by kind and writes to `target`.
    Mux {
        target: String,
        tracks: Vec<MuxTrack>,
    },
}

#[derive(Clone, Debug)]
pub enum FilterKind {
    Audio,
    Video,
}

#[derive(Clone, Debug)]
pub struct ResolvedSelector {
    pub kind: Option<MediaType>,
    pub index: Option<u32>,
}

impl ResolvedSelector {
    pub fn any() -> Self {
        Self {
            kind: None,
            index: None,
        }
    }
    pub fn kind(k: MediaType) -> Self {
        Self {
            kind: Some(k),
            index: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MuxTrack {
    /// Intended media type for this output track.
    pub kind: MediaType,
    /// Upstream producing either packets (copy path) or frames (transcode
    /// path). The executor distinguishes based on whether the node is an
    /// `Encode` (packets) or a `Select`/`Demuxer` in copy mode.
    pub upstream: NodeId,
    /// True when the upstream is a raw-packet stream that should be
    /// packet-copied into the muxer. False when the upstream is an
    /// `Encode` node (already packets, but produced by us) — same I/O,
    /// different provenance for stats.
    pub copy: bool,
}

/// Resolved DAG: nodes addressed by NodeId, with one `roots` entry per output.
#[derive(Clone, Debug, Default)]
pub struct Dag {
    nodes: Vec<DagNode>,
    /// Output name → Mux node id.
    pub roots: IndexMap<String, NodeId>,
}

impl Dag {
    pub fn node(&self, id: NodeId) -> &DagNode {
        &self.nodes[id.0]
    }
    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    fn push(&mut self, node: DagNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }
}

impl Job {
    /// Resolve this job into a [`Dag`]. Call [`Job::validate`] first to get
    /// readable errors on malformed input — `to_dag` also validates defensively
    /// but reports terser messages.
    ///
    /// The resolver honours explicit `{"convert": ...}` schema nodes and
    /// emits [`DagNode::PixConvert`] nodes for them. Auto-insertion of
    /// `PixConvert` based on an encoder's `accepted_pixel_formats` is
    /// carried out by the executor at track-runtime build time (that's
    /// when the upstream stream's actual pixel format becomes known
    /// from the demuxer's [`oxideav_core::CodecParameters`]).
    pub fn to_dag(&self) -> Result<Dag> {
        let mut dag = Dag::default();
        for (name, spec) in &self.outputs {
            let node = self.build_mux(&mut dag, name, spec)?;
            dag.roots.insert(name.clone(), node);
        }
        Ok(dag)
    }

    fn build_mux(&self, dag: &mut Dag, name: &str, spec: &OutputSpec) -> Result<NodeId> {
        let mut tracks: Vec<MuxTrack> = Vec::new();

        // Explicit kind-specific tracks.
        for t in &spec.audio {
            tracks.push(self.build_track(dag, name, MediaType::Audio, t)?);
        }
        for t in &spec.video {
            tracks.push(self.build_track(dag, name, MediaType::Video, t)?);
        }
        for t in &spec.subtitle {
            tracks.push(self.build_track(dag, name, MediaType::Subtitle, t)?);
        }
        // `all` entries fan out into one MuxTrack per media type — since we
        // don't know the upstream's kind statically, the executor treats the
        // selector as "any" and routes by the packet's actual media type.
        for t in &spec.all {
            // A single track spec becomes one MuxTrack with kind=Unknown so
            // the executor preserves whatever the upstream produces.
            tracks.push(self.build_track(dag, name, MediaType::Unknown, t)?);
        }

        if tracks.is_empty() {
            return Err(Error::invalid(format!("job: {name}: no tracks")));
        }

        let target = name.to_string();
        Ok(dag.push(DagNode::Mux { target, tracks }))
    }

    fn build_track(
        &self,
        dag: &mut Dag,
        ctx: &str,
        ctx_kind: MediaType,
        track: &TrackSpec,
    ) -> Result<MuxTrack> {
        let selector = match (&track.stream_selector, ctx_kind) {
            (Some(sel), _) => ResolvedSelector {
                kind: sel.kind.or(match ctx_kind {
                    MediaType::Unknown => None,
                    other => Some(other),
                }),
                index: sel.index,
            },
            (None, MediaType::Unknown) => ResolvedSelector::any(),
            (None, other) => ResolvedSelector::kind(other),
        };

        let upstream = self.build_input(dag, ctx, &track.input, &selector)?;

        // If a codec is named, we build Decode → (Filter / PixConvert
        // chain already in upstream if any) → Encode. If no codec is
        // named, the upstream is packet-producing: either a `Select`
        // straight from the demuxer (copy path) or a previous `Encode`
        // node.
        let (top, copy) = match (&track.codec, self.is_packet_producing(dag, upstream)?) {
            (None, true) => (upstream, true),
            (None, false) => {
                return Err(Error::invalid(format!(
                    "job: {ctx}: track ends with frames but has no `codec` \
                     (add a codec or remove the terminating filter)"
                )));
            }
            (Some(c), true) => {
                // Need to decode first, then encode.
                let dec = dag.push(DagNode::Decode { upstream });
                let enc = dag.push(DagNode::Encode {
                    upstream: dec,
                    codec: c.clone(),
                    params: track.params.clone(),
                });
                (enc, false)
            }
            (Some(c), false) => {
                let enc = dag.push(DagNode::Encode {
                    upstream,
                    codec: c.clone(),
                    params: track.params.clone(),
                });
                (enc, false)
            }
        };

        Ok(MuxTrack {
            kind: ctx_kind,
            upstream: top,
            copy,
        })
    }

    fn build_input(
        &self,
        dag: &mut Dag,
        ctx: &str,
        input: &TrackInput,
        selector: &ResolvedSelector,
    ) -> Result<NodeId> {
        match input {
            TrackInput::Source(src) => self.build_source(dag, ctx, src, selector),
            TrackInput::Filter(f) => {
                let upstream = self.build_input(dag, ctx, f.input.as_ref(), selector)?;
                let kind = guess_filter_kind(&f.filter);
                // Filter consumes frames; insert a Decode if the upstream is
                // still packet-producing.
                let frame_upstream = if self.is_packet_producing(dag, upstream)? {
                    dag.push(DagNode::Decode { upstream })
                } else {
                    upstream
                };
                Ok(dag.push(DagNode::Filter {
                    upstream: frame_upstream,
                    kind,
                    name: f.filter.clone(),
                    params: f.params.clone(),
                }))
            }
            TrackInput::Convert(c) => {
                let upstream = self.build_input(dag, ctx, c.input.as_ref(), selector)?;
                let target = parse_pixel_format(&c.convert)
                    .map_err(|e| Error::invalid(format!("job: {ctx}: convert: {e}")))?;
                // Convert consumes frames; insert a Decode if the upstream
                // is still packet-producing.
                let frame_upstream = if self.is_packet_producing(dag, upstream)? {
                    dag.push(DagNode::Decode { upstream })
                } else {
                    upstream
                };
                Ok(dag.push(DagNode::PixConvert {
                    upstream: frame_upstream,
                    target,
                }))
            }
        }
    }

    fn build_source(
        &self,
        dag: &mut Dag,
        ctx: &str,
        src: &SourceRef,
        selector: &ResolvedSelector,
    ) -> Result<NodeId> {
        if let Some(stripped) = src.from.strip_prefix('@') {
            if stripped.is_empty() {
                return Err(Error::invalid(format!("job: {ctx}: empty alias name")));
            }
            let alias_name = src.from.clone();
            let alias = self.aliases.get(&alias_name).ok_or_else(|| {
                Error::invalid(format!("job: {ctx}: undefined alias {alias_name}"))
            })?;
            // Inline the alias: pick its matching track(s). If the alias has
            // a single-track body in the matching kind, use that; otherwise
            // fall through to `all` which applies our selector.
            let track = pick_alias_track(alias, selector).ok_or_else(|| {
                Error::invalid(format!(
                    "job: {ctx}: alias {alias_name} does not provide a matching track"
                ))
            })?;
            self.build_input(dag, ctx, &track.input, selector)
        } else {
            let demux = dag.push(DagNode::Demuxer {
                source: src.from.clone(),
            });
            Ok(dag.push(DagNode::Select {
                upstream: demux,
                selector: selector.clone(),
            }))
        }
    }

    /// Does the node currently emit raw packets? (Demuxer/Select yes, Decode
    /// no, Filter/PixConvert no, Encode yes.)
    fn is_packet_producing(&self, dag: &Dag, id: NodeId) -> Result<bool> {
        Ok(matches!(
            dag.node(id),
            DagNode::Demuxer { .. } | DagNode::Select { .. } | DagNode::Encode { .. }
        ))
    }
}

/// Look up the first encoder implementation registered for `codec` that
/// declares a non-empty `accepted_pixel_formats`, returning that list.
/// Used by the executor's auto-insert pass.
///
/// Returns `None` when either:
/// - no encoder is registered for this codec id (resolution will fail
///   later with a clearer error), or
/// - every registered encoder declares an empty accepted set
///   (treat as "any format OK").
pub(crate) fn codec_accepted_pixel_formats(
    codecs: &oxideav_codec::CodecRegistry,
    codec: &str,
) -> Option<Vec<PixelFormat>> {
    let id = oxideav_core::CodecId::new(codec);
    for imp in codecs.implementations(&id) {
        if imp.make_encoder.is_none() {
            continue;
        }
        if !imp.caps.accepted_pixel_formats.is_empty() {
            return Some(imp.caps.accepted_pixel_formats.clone());
        }
    }
    None
}

/// Pick the single best track from an alias that matches the caller's
/// selector. Prefers kind-specific buckets; falls back to `all`.
fn pick_alias_track<'a>(
    alias: &'a OutputSpec,
    selector: &ResolvedSelector,
) -> Option<&'a TrackSpec> {
    let bucket = match selector.kind {
        Some(MediaType::Audio) if !alias.audio.is_empty() => Some(&alias.audio),
        Some(MediaType::Video) if !alias.video.is_empty() => Some(&alias.video),
        Some(MediaType::Subtitle) if !alias.subtitle.is_empty() => Some(&alias.subtitle),
        _ => None,
    };
    let src = bucket
        .or({
            if !alias.all.is_empty() {
                Some(&alias.all)
            } else {
                None
            }
        })
        .or({
            if !alias.audio.is_empty() {
                Some(&alias.audio)
            } else {
                None
            }
        })
        .or({
            if !alias.video.is_empty() {
                Some(&alias.video)
            } else {
                None
            }
        })
        .or({
            if !alias.subtitle.is_empty() {
                Some(&alias.subtitle)
            } else {
                None
            }
        })?;

    // Respect selector.index when provided; otherwise take the first entry.
    match selector.index {
        Some(i) => src.get(i as usize),
        None => src.first(),
    }
}

fn guess_filter_kind(name: &str) -> FilterKind {
    // Today we only have audio filters. Treat any name prefixed with
    // `video.` as a video filter (reserved; executor returns Unsupported).
    if name.starts_with("video.") || name.starts_with("v:") {
        FilterKind::Video
    } else {
        FilterKind::Audio
    }
}

impl StreamSelector {
    pub fn resolve(&self, default_kind: MediaType) -> ResolvedSelector {
        ResolvedSelector {
            kind: self.kind.or(Some(default_kind)),
            index: self.index,
        }
    }
}

impl Dag {
    /// Pretty-print the DAG in a stable form for `dry-run` output.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (name, root) in &self.roots {
            s.push_str(&format!("── output: {name}\n"));
            self.describe_node(*root, 1, &mut s);
        }
        s
    }

    fn describe_node(&self, id: NodeId, indent: usize, out: &mut String) {
        let pad = "  ".repeat(indent);
        match &self.nodes[id.0] {
            DagNode::Demuxer { source } => {
                out.push_str(&format!("{pad}demuxer({source})\n"));
            }
            DagNode::Select { upstream, selector } => {
                out.push_str(&format!(
                    "{pad}select(kind={:?}, index={:?})\n",
                    selector.kind, selector.index
                ));
                self.describe_node(*upstream, indent + 1, out);
            }
            DagNode::Decode { upstream } => {
                out.push_str(&format!("{pad}decode\n"));
                self.describe_node(*upstream, indent + 1, out);
            }
            DagNode::Filter {
                upstream,
                kind,
                name,
                params,
            } => {
                out.push_str(&format!(
                    "{pad}filter({kind:?}, {name}, {params})\n",
                    kind = kind,
                    name = name,
                    params = params
                ));
                self.describe_node(*upstream, indent + 1, out);
            }
            DagNode::PixConvert { upstream, target } => {
                out.push_str(&format!("{pad}convert({target:?})\n"));
                self.describe_node(*upstream, indent + 1, out);
            }
            DagNode::Encode {
                upstream,
                codec,
                params,
            } => {
                out.push_str(&format!("{pad}encode({codec}, {params})\n"));
                self.describe_node(*upstream, indent + 1, out);
            }
            DagNode::Mux { target, tracks } => {
                out.push_str(&format!("{pad}mux({target})\n"));
                for t in tracks {
                    let label = if t.copy { "copy" } else { "xcode" };
                    out.push_str(&format!(
                        "{pad}  track[{kind:?},{label}]\n",
                        kind = t.kind,
                        label = label
                    ));
                    self.describe_node(t.upstream, indent + 2, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_remux_builds_copy_path() {
        let j = Job::from_json(
            r#"{
                "out.mkv": {
                    "audio": [{"from": "in.mp3"}],
                    "video": [{"from": "in.mp3"}]
                }
            }"#,
        )
        .unwrap();
        let dag = j.to_dag().unwrap();
        let root = dag.roots["out.mkv"];
        match dag.node(root) {
            DagNode::Mux { tracks, .. } => {
                assert_eq!(tracks.len(), 2);
                for t in tracks {
                    assert!(t.copy);
                }
            }
            _ => panic!("expected mux root"),
        }
    }

    #[test]
    fn codec_triggers_decode_encode() {
        let j = Job::from_json(
            r#"{
                "out.flac": {
                    "audio": [{"from": "in.mp3", "codec": "flac"}]
                }
            }"#,
        )
        .unwrap();
        let dag = j.to_dag().unwrap();
        let root = dag.roots["out.flac"];
        let mux = match dag.node(root) {
            DagNode::Mux { tracks, .. } => tracks,
            _ => panic!(),
        };
        match dag.node(mux[0].upstream) {
            DagNode::Encode { codec, .. } => assert_eq!(codec, "flac"),
            n => panic!("expected Encode, got {n:?}"),
        }
    }

    #[test]
    fn filter_chain_inserts_decode_then_filter() {
        let j = Job::from_json(
            r#"{
                "out.wav": {
                    "audio": [{
                        "filter": "volume",
                        "params": {"gain_db": 3},
                        "input": {"from": "in.wav"},
                        "codec": "pcm_s16le"
                    }]
                }
            }"#,
        )
        .unwrap();
        let dag = j.to_dag().unwrap();
        let root = dag.roots["out.wav"];
        let mux = match dag.node(root) {
            DagNode::Mux { tracks, .. } => tracks,
            _ => panic!(),
        };
        let enc = match dag.node(mux[0].upstream) {
            DagNode::Encode { upstream, .. } => *upstream,
            _ => panic!(),
        };
        match dag.node(enc) {
            DagNode::Filter { name, .. } => assert_eq!(name, "volume"),
            n => panic!("expected Filter under Encode, got {n:?}"),
        }
    }

    #[test]
    fn alias_is_inlined() {
        let j = Job::from_json(
            r#"{
                "@in": {"all": [{"from": "a.mkv"}]},
                "out.mkv": {"audio": [{"from": "@in"}]}
            }"#,
        )
        .unwrap();
        let dag = j.to_dag().unwrap();
        let root = dag.roots["out.mkv"];
        let mux = match dag.node(root) {
            DagNode::Mux { tracks, .. } => tracks,
            _ => panic!(),
        };
        match dag.node(mux[0].upstream) {
            DagNode::Select { upstream, .. } => match dag.node(*upstream) {
                DagNode::Demuxer { source } => assert_eq!(source, "a.mkv"),
                _ => panic!(),
            },
            n => panic!("unexpected top node {n:?}"),
        }
    }

    #[test]
    fn explicit_convert_builds_pix_convert_node() {
        // An explicit `{"convert": "..."}` schema node must produce a
        // PixConvert DAG node with a Decode underneath (since the source
        // is still packet-producing).
        let j = Job::from_json(
            r#"{
                "out.mkv": {
                    "video": [{
                        "convert": "yuv420p",
                        "input": {"from": "in.mp4"},
                        "codec": "h264"
                    }]
                }
            }"#,
        )
        .unwrap();
        let dag = j.to_dag().unwrap();
        let root = dag.roots["out.mkv"];
        let mux = match dag.node(root) {
            DagNode::Mux { tracks, .. } => tracks,
            _ => panic!(),
        };
        let enc_up = match dag.node(mux[0].upstream) {
            DagNode::Encode { upstream, .. } => *upstream,
            n => panic!("expected Encode, got {n:?}"),
        };
        let pc_up = match dag.node(enc_up) {
            DagNode::PixConvert { upstream, target } => {
                assert_eq!(*target, PixelFormat::Yuv420P);
                *upstream
            }
            n => panic!("expected PixConvert, got {n:?}"),
        };
        match dag.node(pc_up) {
            DagNode::Decode { .. } => {}
            n => panic!("expected Decode under PixConvert, got {n:?}"),
        }
    }
}
