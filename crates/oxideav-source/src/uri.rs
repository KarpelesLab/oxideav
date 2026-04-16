//! Minimal URI scheme parsing.
//!
//! We only need to split off the leading `scheme://` (or `scheme:`) so the
//! registry can dispatch. Anything fancier (auth, query, fragment) is the
//! driver's problem.

/// Split a URI into `(scheme, rest)`. Bare paths (no scheme) report scheme
/// `"file"` and `rest = uri`. Path-like inputs that happen to start with
/// `c:` on Windows are treated as bare paths (the second char is `:`, but
/// no `//` follows and the part before `:` is a single ASCII letter).
pub fn split(uri: &str) -> (&str, &str) {
    if let Some(idx) = uri.find(':') {
        let (scheme, rest) = uri.split_at(idx);
        let rest = &rest[1..]; // skip ':'

        // Reject single-letter scheme that looks like a Windows drive letter.
        if scheme.len() == 1 && scheme.chars().next().unwrap().is_ascii_alphabetic() {
            return ("file", uri);
        }

        // Scheme must be ASCII alphanumeric / `+` / `-` / `.`, starting with a letter.
        let valid = !scheme.is_empty()
            && scheme.chars().next().unwrap().is_ascii_alphabetic()
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));

        if !valid {
            return ("file", uri);
        }

        // Strip leading `//` from rest if present.
        let rest = rest.strip_prefix("//").unwrap_or(rest);
        return (scheme, rest);
    }
    ("file", uri)
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn bare_path() {
        assert_eq!(split("/tmp/x.mp4"), ("file", "/tmp/x.mp4"));
        assert_eq!(split("relative/x.mp4"), ("file", "relative/x.mp4"));
        assert_eq!(split("Cargo.toml"), ("file", "Cargo.toml"));
    }

    #[test]
    fn file_scheme() {
        assert_eq!(split("file:///tmp/x.mp4"), ("file", "/tmp/x.mp4"));
        assert_eq!(split("file:relative"), ("file", "relative"));
    }

    #[test]
    fn http_scheme() {
        assert_eq!(
            split("https://example.com/a.mp4"),
            ("https", "example.com/a.mp4")
        );
        assert_eq!(
            split("http://example.com:8080/a"),
            ("http", "example.com:8080/a")
        );
    }

    #[test]
    fn windows_drive_letter_is_bare_path() {
        assert_eq!(
            split("C:\\Users\\file.mp4"),
            ("file", "C:\\Users\\file.mp4")
        );
    }
}
