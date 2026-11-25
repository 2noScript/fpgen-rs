//! Rust port of `fpgen` 1.3.0 (`scrapfly/fingerprint-generator` / `JWriter20/fpgen-js`).
//!
//! Generates coherent browser fingerprints from `fpgen`'s pre-trained Bayesian network model.

pub mod bayesian_network;
pub mod exceptions;
pub mod generator;
pub mod model;
pub mod pin;
pub mod pyjson;
pub mod trace;
pub mod utils;
pub mod zip;

/// Crate version aligned with upstream `fpgen` 1.3.0.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use exceptions::{FpgenError, Result};
pub use generator::{generate, generate_target, GenerateOptions, Generator, GeneratorOptions};
pub use model::{ensure_model, is_model_installed, model_dir, set_model_dir, EnsureModelOptions, FpgenModel};
pub use pin::{ModelPin, MODEL_PIN};
pub use trace::{trace, trace_with_evidence, TraceEntry, TraceResult};
pub use utils::{lookup_possibilities, query};

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_fpgen_upstream() {
        assert_eq!(VERSION, "1.3.0");
    }
}
