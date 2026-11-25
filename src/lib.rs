//! Rust port of `fpgen` 1.3.0 (`scrapfly/fingerprint-generator` / `JWriter20/fpgen-js`).

/// Crate version aligned with upstream `fpgen` 1.3.0.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_fpgen_upstream() {
        assert_eq!(VERSION, "1.3.0");
    }
}
