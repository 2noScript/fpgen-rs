use std::path::PathBuf;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use crate::bayesian_network::AllowedValues;
use crate::exceptions::{FpgenError, Result};
use crate::model::{get_model, FpgenModel};
use crate::trace::{trace_with_evidence, TraceResult};
use crate::utils::{
    assert_conditions, build_evidence, find_roots, lookup_possibilities, make_output_dict,
    reassemble_targets,
};

/// Options configuring a reusable `Generator` instance.
#[derive(Debug, Clone)]
pub struct GeneratorOptions {
    pub conditions: Option<Value>,
    pub strict: bool,
    pub unflatten: bool,
    pub model_dir: Option<PathBuf>,
}

impl Default for GeneratorOptions {
    fn default() -> Self {
        Self {
            conditions: None,
            strict: false,
            unflatten: true,
            model_dir: None,
        }
    }
}

/// Per-call options for `generate` and `generate_target`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    pub conditions: Option<Value>,
    pub strict: Option<bool>,
    pub unflatten: Option<bool>,
    pub model_dir: Option<PathBuf>,
}

/// Fingerprint generator backed by the pinned `fpgen` Bayesian network.
pub struct Generator {
    model: Arc<FpgenModel>,
    base_conditions: Option<Value>,
    base_evidence: IndexMap<String, AllowedValues>,
    strict: bool,
    unflatten: bool,
}

impl Generator {
    /// Creates a new `Generator`, loading the model and validating any base `conditions`.
    pub fn new(options: GeneratorOptions) -> Result<Self> {
        let model = get_model(options.model_dir.as_deref())?;
        let base_evidence =
            build_evidence(&model, options.conditions.as_ref(), options.strict)?;
        Ok(Self {
            model,
            base_conditions: options.conditions,
            base_evidence,
            strict: options.strict,
            unflatten: options.unflatten,
        })
    }

    pub fn model(&self) -> &FpgenModel {
        &self.model
    }

    fn resolve_evidence(
        &self,
        override_conditions: Option<&Value>,
        strict: bool,
    ) -> Result<IndexMap<String, AllowedValues>> {
        match override_conditions {
            None => Ok(self.base_evidence.clone()),
            Some(v) if v.is_null() => Ok(self.base_evidence.clone()),
            Some(overrides) => {
                assert_conditions(&self.model, self.base_conditions.as_ref(), Some(overrides), strict)
            }
        }
    }

    /// Generates a full coherent browser fingerprint JSON object.
    pub fn generate(&self, options: Option<&GenerateOptions>) -> Result<Value> {
        let strict = options.and_then(|o| o.strict).unwrap_or(self.strict);
        let unflatten = options.and_then(|o| o.unflatten).unwrap_or(self.unflatten);
        let evidence = self.resolve_evidence(
            options.and_then(|o| o.conditions.as_ref()),
            strict,
        )?;
        let sample = self.model.network.generate_consistent_sample(&evidence)?;
        make_output_dict(&sample, &self.model, unflatten)
    }

    /// Generates only the requested target nodes/paths coherently with any conditions.
    pub fn generate_target(
        &self,
        targets: &[&str],
        options: Option<&GenerateOptions>,
    ) -> Result<Value> {
        if targets.is_empty() {
            return Err(FpgenError::ValueError(
                "generate_target requires at least one target".into(),
            ));
        }
        let strict = options.and_then(|o| o.strict).unwrap_or(self.strict);
        let unflatten = options.and_then(|o| o.unflatten).unwrap_or(self.unflatten);
        let evidence = self.resolve_evidence(
            options.and_then(|o| o.conditions.as_ref()),
            strict,
        )?;

        let mut root_nodes: Vec<String> = Vec::new();
        for &target in targets {
            for root in find_roots(&self.model.network, target)? {
                if !root_nodes.contains(&root) {
                    root_nodes.push(root);
                }
            }
        }

        let sample = self
            .model
            .network
            .generate_certain_nodes(&evidence, Some(&root_nodes))?;
        let owned_targets: Vec<String> = targets.iter().map(|s| (*s).to_owned()).collect();
        reassemble_targets(&sample, &self.model, &owned_targets, unflatten, strict)
    }

    /// Queries all possible values for `target`.
    pub fn query(&self, target: &str) -> Result<Value> {
        lookup_possibilities(&self.model, target)
    }

    /// Traces the marginal probability distribution for `targets` under conditions.
    pub fn trace(
        &self,
        targets: &[&str],
        conditions: Option<&Value>,
        strict: Option<bool>,
        unflatten: Option<bool>,
    ) -> Result<TraceResult> {
        let effective_strict = strict.unwrap_or(self.strict);
        let effective_unflatten = unflatten.unwrap_or(self.unflatten);
        let evidence = self.resolve_evidence(conditions, effective_strict)?;
        trace_with_evidence(&self.model, targets, &evidence, effective_unflatten)
    }
}

/// One-shot helper generating a complete fingerprint with default or custom options.
pub fn generate(options: Option<GenerateOptions>) -> Result<Value> {
    let opts = options.unwrap_or_default();
    let gen = Generator::new(GeneratorOptions {
        conditions: opts.conditions.clone(),
        strict: opts.strict.unwrap_or(false),
        unflatten: opts.unflatten.unwrap_or(true),
        model_dir: opts.model_dir.clone(),
    })?;
    gen.generate(Some(&GenerateOptions {
        conditions: None,
        strict: opts.strict,
        unflatten: Some(opts.unflatten.unwrap_or(true)),
        model_dir: None,
    }))
}

/// One-shot helper generating specific fingerprint targets with optional constraints.
pub fn generate_target(targets: &[&str], options: Option<GenerateOptions>) -> Result<Value> {
    let opts = options.unwrap_or_default();
    let gen = Generator::new(GeneratorOptions {
        conditions: opts.conditions.clone(),
        strict: opts.strict.unwrap_or(false),
        unflatten: opts.unflatten.unwrap_or(true),
        model_dir: opts.model_dir.clone(),
    })?;
    gen.generate_target(
        targets,
        Some(&GenerateOptions {
            conditions: None,
            strict: opts.strict,
            unflatten: Some(opts.unflatten.unwrap_or(true)),
            model_dir: None,
        }),
    )
}
