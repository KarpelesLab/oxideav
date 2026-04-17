//! Validate a parsed `Job`: reference integrity + alias-cycle detection.

use oxideav_core::{Error, Result};

use crate::schema::{is_reserved_sink, parse_pixel_format, Job, OutputSpec, TrackInput};

impl Job {
    /// Walk every track input to confirm:
    ///
    /// 1. Every output/alias has at least one track (no empty specs).
    /// 2. Every `@alias` reference inside a `from` field resolves — either to
    ///    a defined alias in this document, or to a reserved sink name.
    /// 3. Alias references do not form a cycle.
    ///
    /// Returns `Err(InvalidData)` with a pointer at the offending key on
    /// first failure.
    pub fn validate(&self) -> Result<()> {
        for (name, spec) in self.outputs.iter().chain(self.aliases.iter()) {
            if spec.is_empty() {
                return Err(Error::invalid(format!(
                    "job: {name}: no tracks (need at least one of audio/video/subtitle/all)"
                )));
            }
            self.check_refs_in_spec(name, spec)?;
        }
        for alias in self.aliases.keys() {
            self.check_no_cycle(alias)?;
        }
        Ok(())
    }

    fn check_refs_in_spec(&self, ctx_name: &str, spec: &OutputSpec) -> Result<()> {
        let all_tracks = spec
            .audio
            .iter()
            .chain(&spec.video)
            .chain(&spec.subtitle)
            .chain(&spec.all);
        for track in all_tracks {
            self.check_refs_in_input(ctx_name, &track.input)?;
        }
        Ok(())
    }

    fn check_refs_in_input(&self, ctx: &str, input: &TrackInput) -> Result<()> {
        match input {
            TrackInput::Source(src) => {
                if src.from.starts_with('@') {
                    if is_reserved_sink(&src.from) {
                        return Err(Error::invalid(format!(
                            "job: {ctx}: cannot use reserved sink {src} as a source",
                            src = src.from
                        )));
                    }
                    if !self.aliases.contains_key(&src.from) {
                        return Err(Error::invalid(format!(
                            "job: {ctx}: unresolved alias reference {src}",
                            src = src.from
                        )));
                    }
                } else if src.from.is_empty() {
                    return Err(Error::invalid(format!("job: {ctx}: empty `from`")));
                }
                Ok(())
            }
            TrackInput::Filter(f) => {
                if f.filter.trim().is_empty() {
                    return Err(Error::invalid(format!(
                        "job: {ctx}: filter node has empty `filter` name"
                    )));
                }
                self.check_refs_in_input(ctx, f.input.as_ref())
            }
            TrackInput::Convert(c) => {
                // Reject unknown pixel format names here so errors point at
                // the track context rather than the opaque DAG builder.
                parse_pixel_format(&c.convert)
                    .map_err(|e| Error::invalid(format!("job: {ctx}: convert: {e}")))?;
                self.check_refs_in_input(ctx, c.input.as_ref())
            }
        }
    }

    /// Depth-first search from `start` over the alias graph. Reports a cycle
    /// with the offending path if found.
    fn check_no_cycle(&self, start: &str) -> Result<()> {
        let mut stack: Vec<String> = vec![start.to_string()];
        let mut path: Vec<String> = vec![start.to_string()];
        self.visit_cycle(start, &mut stack, &mut path)
    }

    fn visit_cycle(
        &self,
        current: &str,
        stack: &mut Vec<String>,
        path: &mut Vec<String>,
    ) -> Result<()> {
        let spec = match self.aliases.get(current) {
            Some(s) => s,
            None => return Ok(()),
        };
        for refd in collect_alias_refs(spec) {
            if stack.iter().any(|s| s == &refd) {
                path.push(refd.clone());
                return Err(Error::invalid(format!(
                    "job: alias cycle detected: {}",
                    path.join(" -> ")
                )));
            }
            stack.push(refd.clone());
            path.push(refd.clone());
            self.visit_cycle(&refd, stack, path)?;
            stack.pop();
            path.pop();
        }
        Ok(())
    }
}

fn collect_alias_refs(spec: &OutputSpec) -> Vec<String> {
    let mut out = Vec::new();
    for t in spec
        .audio
        .iter()
        .chain(&spec.video)
        .chain(&spec.subtitle)
        .chain(&spec.all)
    {
        walk_input(&t.input, &mut out);
    }
    out
}

fn walk_input(input: &TrackInput, out: &mut Vec<String>) {
    match input {
        TrackInput::Source(s) => {
            if s.from.starts_with('@') {
                out.push(s.from.clone());
            }
        }
        TrackInput::Filter(f) => walk_input(f.input.as_ref(), out),
        TrackInput::Convert(c) => walk_input(c.input.as_ref(), out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_output() {
        let j = Job::from_json(r#"{"out.mkv": {}}"#).unwrap();
        let e = j.validate().unwrap_err();
        assert!(matches!(e, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_dangling_alias() {
        let j = Job::from_json(r#"{"out.mkv": {"audio": [{"from": "@missing"}]}}"#).unwrap();
        let e = j.validate().unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("unresolved alias"), "got: {msg}");
    }

    #[test]
    fn detects_direct_cycle() {
        // @a references @b and @b references @a.
        let j = Job::from_json(
            r#"{
                "@a": {"all": [{"from": "@b"}]},
                "@b": {"all": [{"from": "@a"}]},
                "out.mkv": {"audio": [{"from": "@a"}]}
            }"#,
        )
        .unwrap();
        let e = j.validate().unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("cycle"), "got: {msg}");
    }

    #[test]
    fn detects_self_cycle() {
        let j = Job::from_json(
            r#"{
                "@a": {"all": [{"from": "@a"}]},
                "out.mkv": {"audio": [{"from": "@a"}]}
            }"#,
        )
        .unwrap();
        assert!(j.validate().is_err());
    }

    #[test]
    fn accepts_legal_alias_chain() {
        let j = Job::from_json(
            r#"{
                "@in": {"all": [{"from": "a.mp4"}]},
                "@loud": {"audio": [{"filter": "volume", "params": {"gain_db": 3}, "input": {"from": "@in"}}]},
                "out.mkv": {"audio": [{"from": "@loud"}], "video": [{"from": "@in"}]}
            }"#,
        )
        .unwrap();
        j.validate().unwrap();
    }

    #[test]
    fn rejects_reserved_sink_as_source() {
        let j = Job::from_json(r#"{"out.mkv": {"audio": [{"from": "@display"}]}}"#).unwrap();
        assert!(j.validate().is_err());
    }
}
