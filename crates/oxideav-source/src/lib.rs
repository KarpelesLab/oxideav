//! Generic source registry for oxideav.
//!
//! Containers in oxideav take a `Box<dyn ReadSeek>`; this crate is what
//! turns a URI into one. The built-in `file` driver handles bare paths
//! and `file://` URIs; external drivers (e.g. `oxideav-http`) register
//! themselves into a [`SourceRegistry`] for additional schemes.
//!
//! ```no_run
//! use oxideav_source::SourceRegistry;
//!
//! let reg = SourceRegistry::with_defaults();
//! let _input = reg.open("/tmp/video.mp4").unwrap();
//! ```

use std::collections::HashMap;

pub use oxideav_container::ReadSeek;
use oxideav_core::{Error, Result};

mod buffered;
mod file;
mod uri;

pub use buffered::BufferedSource;
pub use file::open_file;

/// Function signature for a source driver. Receives the full URI string
/// and returns an opened reader.
pub type OpenSourceFn = fn(uri: &str) -> Result<Box<dyn ReadSeek>>;

/// Registry mapping URI schemes to opener functions.
pub struct SourceRegistry {
    schemes: HashMap<String, OpenSourceFn>,
}

impl SourceRegistry {
    /// Empty registry. Callers must register at least the `file` driver
    /// before calling [`open`](Self::open).
    pub fn new() -> Self {
        Self {
            schemes: HashMap::new(),
        }
    }

    /// Registry pre-populated with the built-in `file` driver. Bare paths
    /// (without a scheme) also dispatch to it.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register("file", open_file);
        r
    }

    /// Register an opener for a scheme. Schemes are normalised to ASCII
    /// lowercase. Replaces any prior registration.
    pub fn register(&mut self, scheme: &str, opener: OpenSourceFn) {
        self.schemes.insert(scheme.to_ascii_lowercase(), opener);
    }

    /// Open a URI. The URI's scheme determines which opener runs; bare
    /// paths (no scheme) and unrecognised schemes both fall back to the
    /// `file` driver if it is registered.
    pub fn open(&self, uri_str: &str) -> Result<Box<dyn ReadSeek>> {
        let (scheme, _) = uri::split(uri_str);
        let scheme = scheme.to_ascii_lowercase();
        if let Some(opener) = self.schemes.get(&scheme) {
            return opener(uri_str);
        }
        // Fall back to file driver for unknown schemes — useful when a
        // caller hands us "/path" or a Windows drive letter that uri::split
        // already mapped to "file" anyway.
        if let Some(opener) = self.schemes.get("file") {
            return opener(uri_str);
        }
        Err(Error::Unsupported(format!(
            "no source driver for scheme '{scheme}' (URI: {uri_str})"
        )))
    }

    /// Iterate the registered schemes (for diagnostics).
    pub fn schemes(&self) -> impl Iterator<Item = &str> {
        self.schemes.keys().map(|s| s.as_str())
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}
