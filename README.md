# fpgen-rs

Rust port of `fpgen` `1.3.0` (`scrapfly/fingerprint-generator` / `JWriter20/fpgen-js`).

Generates coherent browser fingerprints using `fpgen`'s Bayesian network model with pinned SHA-256 verification and zstd-compressed value lookup.

## Module Layout

The crate keeps file names and API symbols aligned with upstream `fpgen` / `fpgen-js`:

- `exceptions` (`FpgenError`, `Result`)
- `pin` (`ModelPin`, `MODEL_PIN`)
- `zip` (`read_zip`)
- `pyjson` (`parse_ordered`, `parse_py_typed`, `casefold`, `py_equals`, `py_dumps`, `base85_to_int`)
- `bayesian_network` (`BayesianNetwork`, `BayesianNode`, `AllowedValues`, `CaseInsensitiveMap`)
- `model` (`FpgenModel`, `ensure_model`, `is_model_installed`, `model_dir`, `set_model_dir`, `verify_archive`, `install_archive`)
- `utils` (`query`, `lookup_possibilities`, `build_evidence`, `assert_conditions`, `flatten_dict`, `unflatten`)
- `trace` (`trace`, `trace_with_evidence`, `TraceEntry`, `TraceResult`)
- `generator` (`Generator`, `GeneratorOptions`, `GenerateOptions`, `generate`, `generate_target`)

## Usage

```rust
use serde_json::json;
use fpgen_rs::{generate, generate_target, query, trace, GenerateOptions, Generator, GeneratorOptions};

fn main() -> fpgen_rs::Result<()> {
    // One-shot fingerprint constrained to macOS:
    let fp = generate(Some(GenerateOptions {
        conditions: Some(json!({ "navigator.platform": "MacIntel" })),
        ..Default::default()
    }))?;
    assert_eq!(fp["navigator"]["platform"], "MacIntel");

    // Reusable generator instance:
    let gen = Generator::new(GeneratorOptions {
        conditions: Some(json!({ "os": "MacOS" })),
        strict: true,
        ..Default::default()
    })?;
    let sample = gen.generate(None)?;
    assert_eq!(sample["os"], "MacOS");

    // Partial target generation, querying, and marginal probability tracing:
    let partial = generate_target(&["navigator.platform", "screen.colorDepth"], None)?;
    let _platforms = query("navigator.platform", None)?;
    let _dist = trace(&["navigator.platform"], None, false, true, None)?;
    let _ = partial;
    Ok(())
}
```

## Testing

```bash
# Offline unit + synthetic network & archive verification tests:
cargo test

# Parity tests against the real pinned model (~212 MB, downloaded on demand if not cached):
cargo test -- --ignored

# Lint:
cargo clippy --all-targets -- -D warnings
```
