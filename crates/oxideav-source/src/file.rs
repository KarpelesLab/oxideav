//! Built-in `file://` driver and bare-path fallback.

use std::fs::File;

use oxideav_container::ReadSeek;
use oxideav_core::{Error, Result};

use crate::uri;

/// Open a local file as a `Box<dyn ReadSeek>`. Accepts:
/// - bare paths: `/abs/path`, `rel/path`, `Cargo.toml`
/// - `file:///abs/path`
/// - `file:relative`
pub fn open_file(uri_str: &str) -> Result<Box<dyn ReadSeek>> {
    let (scheme, rest) = uri::split(uri_str);
    if scheme != "file" {
        return Err(Error::invalid(format!(
            "file driver invoked on non-file URI: {uri_str}"
        )));
    }
    let f = File::open(rest)?;
    Ok(Box::new(f))
}
