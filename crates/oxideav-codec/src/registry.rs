//! In-process codec registry — supports multiple implementations per codec
//! id, ranked by capability + priority + user preferences with init-time
//! fallback.

use std::collections::HashMap;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecParameters, CodecPreferences, Error, Result,
};

use crate::{Decoder, DecoderFactory, Encoder, EncoderFactory};

/// One registered implementation: capability description + factories.
/// Either / both factories may be present depending on whether the impl
/// can decode, encode, or both.
#[derive(Clone)]
pub struct CodecImplementation {
    pub caps: CodecCapabilities,
    pub make_decoder: Option<DecoderFactory>,
    pub make_encoder: Option<EncoderFactory>,
}

#[derive(Default)]
pub struct CodecRegistry {
    impls: HashMap<CodecId, Vec<CodecImplementation>>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a codec implementation. The same codec id may be registered
    /// multiple times — for example a software FLAC decoder and (later) a
    /// hardware one would both register under id `"flac"`.
    pub fn register(&mut self, id: CodecId, implementation: CodecImplementation) {
        self.impls.entry(id).or_default().push(implementation);
    }

    /// Convenience: register a decoder-only implementation built from a
    /// caps + factory pair.
    pub fn register_decoder_impl(
        &mut self,
        id: CodecId,
        caps: CodecCapabilities,
        factory: DecoderFactory,
    ) {
        self.register(
            id,
            CodecImplementation {
                caps: caps.with_decode(),
                make_decoder: Some(factory),
                make_encoder: None,
            },
        );
    }

    /// Convenience: register an encoder-only implementation.
    pub fn register_encoder_impl(
        &mut self,
        id: CodecId,
        caps: CodecCapabilities,
        factory: EncoderFactory,
    ) {
        self.register(
            id,
            CodecImplementation {
                caps: caps.with_encode(),
                make_decoder: None,
                make_encoder: Some(factory),
            },
        );
    }

    /// Convenience: register a single implementation that handles both
    /// decode and encode for a codec id.
    pub fn register_both(
        &mut self,
        id: CodecId,
        caps: CodecCapabilities,
        decode: DecoderFactory,
        encode: EncoderFactory,
    ) {
        self.register(
            id,
            CodecImplementation {
                caps: caps.with_decode().with_encode(),
                make_decoder: Some(decode),
                make_encoder: Some(encode),
            },
        );
    }

    /// Backwards-compat shim: register a decoder-only impl with default
    /// software capabilities. Prefer `register_decoder_impl` for new code.
    pub fn register_decoder(&mut self, id: CodecId, factory: DecoderFactory) {
        let caps = CodecCapabilities::audio(id.as_str()).with_decode();
        self.register_decoder_impl(id, caps, factory);
    }

    /// Backwards-compat shim: register an encoder-only impl with default
    /// software capabilities.
    pub fn register_encoder(&mut self, id: CodecId, factory: EncoderFactory) {
        let caps = CodecCapabilities::audio(id.as_str()).with_encode();
        self.register_encoder_impl(id, caps, factory);
    }

    pub fn has_decoder(&self, id: &CodecId) -> bool {
        self.impls
            .get(id)
            .map(|v| v.iter().any(|i| i.make_decoder.is_some()))
            .unwrap_or(false)
    }

    pub fn has_encoder(&self, id: &CodecId) -> bool {
        self.impls
            .get(id)
            .map(|v| v.iter().any(|i| i.make_encoder.is_some()))
            .unwrap_or(false)
    }

    /// Build a decoder for `params`. Walks all implementations matching the
    /// codec id in increasing priority order, skipping any excluded by the
    /// caller's preferences. Init-time fallback: if a higher-priority impl's
    /// constructor returns an error, the next candidate is tried.
    pub fn make_decoder_with(
        &self,
        params: &CodecParameters,
        prefs: &CodecPreferences,
    ) -> Result<Box<dyn Decoder>> {
        let candidates = self
            .impls
            .get(&params.codec_id)
            .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
        let mut ranked: Vec<&CodecImplementation> = candidates
            .iter()
            .filter(|i| i.make_decoder.is_some() && !prefs.excludes(&i.caps))
            .filter(|i| caps_fit_params(&i.caps, params, false))
            .collect();
        ranked.sort_by_key(|i| prefs.effective_priority(&i.caps));
        let mut last_err: Option<Error> = None;
        for imp in ranked {
            match (imp.make_decoder.unwrap())(params) {
                Ok(d) => return Ok(d),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            Error::CodecNotFound(format!(
                "no decoder for {} accepts the requested parameters",
                params.codec_id
            ))
        }))
    }

    /// Build an encoder, with the same priority + fallback semantics.
    pub fn make_encoder_with(
        &self,
        params: &CodecParameters,
        prefs: &CodecPreferences,
    ) -> Result<Box<dyn Encoder>> {
        let candidates = self
            .impls
            .get(&params.codec_id)
            .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
        let mut ranked: Vec<&CodecImplementation> = candidates
            .iter()
            .filter(|i| i.make_encoder.is_some() && !prefs.excludes(&i.caps))
            .filter(|i| caps_fit_params(&i.caps, params, true))
            .collect();
        ranked.sort_by_key(|i| prefs.effective_priority(&i.caps));
        let mut last_err: Option<Error> = None;
        for imp in ranked {
            match (imp.make_encoder.unwrap())(params) {
                Ok(e) => return Ok(e),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            Error::CodecNotFound(format!(
                "no encoder for {} accepts the requested parameters",
                params.codec_id
            ))
        }))
    }

    /// Default-preference shorthand for `make_decoder_with`.
    pub fn make_decoder(&self, params: &CodecParameters) -> Result<Box<dyn Decoder>> {
        self.make_decoder_with(params, &CodecPreferences::default())
    }

    /// Default-preference shorthand for `make_encoder_with`.
    pub fn make_encoder(&self, params: &CodecParameters) -> Result<Box<dyn Encoder>> {
        self.make_encoder_with(params, &CodecPreferences::default())
    }

    /// Iterate codec ids that have at least one decoder implementation.
    pub fn decoder_ids(&self) -> impl Iterator<Item = &CodecId> {
        self.impls
            .iter()
            .filter(|(_, v)| v.iter().any(|i| i.make_decoder.is_some()))
            .map(|(id, _)| id)
    }

    pub fn encoder_ids(&self) -> impl Iterator<Item = &CodecId> {
        self.impls
            .iter()
            .filter(|(_, v)| v.iter().any(|i| i.make_encoder.is_some()))
            .map(|(id, _)| id)
    }

    /// All registered implementations of a given codec id.
    pub fn implementations(&self, id: &CodecId) -> &[CodecImplementation] {
        self.impls.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Iterator over every (codec_id, impl) pair — useful for `oxideav list`
    /// to show capability flags per implementation.
    pub fn all_implementations(&self) -> impl Iterator<Item = (&CodecId, &CodecImplementation)> {
        self.impls
            .iter()
            .flat_map(|(id, v)| v.iter().map(move |i| (id, i)))
    }
}

/// Check whether an implementation's restrictions are compatible with the
/// requested codec parameters. `for_encode` swaps the rare cases where a
/// restriction only applies one way.
fn caps_fit_params(caps: &CodecCapabilities, p: &CodecParameters, for_encode: bool) -> bool {
    let _ = for_encode; // reserved for future use (e.g. encode-only bitrate caps)
    if let (Some(max), Some(w)) = (caps.max_width, p.width) {
        if w > max {
            return false;
        }
    }
    if let (Some(max), Some(h)) = (caps.max_height, p.height) {
        if h > max {
            return false;
        }
    }
    if let (Some(max), Some(br)) = (caps.max_bitrate, p.bit_rate) {
        if br > max {
            return false;
        }
    }
    if let (Some(max), Some(sr)) = (caps.max_sample_rate, p.sample_rate) {
        if sr > max {
            return false;
        }
    }
    if let (Some(max), Some(ch)) = (caps.max_channels, p.channels) {
        if ch > max {
            return false;
        }
    }
    true
}
