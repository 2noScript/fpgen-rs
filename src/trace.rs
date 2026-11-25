use indexmap::IndexMap;
use serde_json::Value;

use crate::bayesian_network::AllowedValues;
use crate::exceptions::{FpgenError, Result};
use crate::model::{get_model, FpgenModel};
use crate::pyjson::{parse_py_typed, py_equals};
use crate::utils::{build_evidence, find_roots};

/// A single entry in a traced marginal probability distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEntry {
    pub value: Value,
    pub probability: f64,
}

/// Output of `trace` / `trace_with_evidence`: either a single distribution or nested target distributions.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceResult {
    Distribution(Vec<TraceEntry>),
    Multi(IndexMap<String, TraceResult>),
}

impl TraceResult {
    pub fn as_distribution(&self) -> Option<&[TraceEntry]> {
        match self {
            Self::Distribution(entries) => Some(entries),
            Self::Multi(_) => None,
        }
    }

    pub fn as_multi(&self) -> Option<&IndexMap<String, TraceResult>> {
        match self {
            Self::Multi(map) => Some(map),
            Self::Distribution(_) => None,
        }
    }
}

fn decode_distribution(
    model: &FpgenModel,
    raw_dist: &IndexMap<String, f64>,
) -> Result<Vec<TraceEntry>> {
    if raw_dist.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<String> = raw_dist.keys().cloned().collect();
    let probs: Vec<f64> = raw_dist.values().copied().collect();
    let raw_json_list = model.lookup_value_list(&ids)?;

    let mut aggregated: Vec<TraceEntry> = Vec::with_capacity(ids.len());
    for (raw_json, prob) in raw_json_list.into_iter().zip(probs) {
        let parsed = parse_py_typed(&raw_json)?;
        if let Some(existing) = aggregated.iter_mut().find(|e| py_equals(&e.value, &parsed)) {
            existing.probability += prob;
        } else {
            aggregated.push(TraceEntry {
                value: parsed,
                probability: prob,
            });
        }
    }

    let total: f64 = aggregated.iter().map(|e| e.probability).sum();
    if total > 0.0 {
        for entry in &mut aggregated {
            entry.probability /= total;
        }
    }
    Ok(aggregated)
}

fn trace_single_target(
    model: &FpgenModel,
    target: &str,
    evidence: &IndexMap<String, AllowedValues>,
    unflatten: bool,
) -> Result<TraceResult> {
    let roots = find_roots(&model.network, target)?;
    if roots.len() == 1 {
        let root_name = &roots[0];
        let raw_dist = model.network.trace(root_name, evidence)?;
        let dist = decode_distribution(model, &raw_dist)?;
        return Ok(TraceResult::Distribution(dist));
    }

    let mut map = IndexMap::new();
    for root_name in roots {
        let raw_dist = model.network.trace(&root_name, evidence)?;
        let dist = decode_distribution(model, &raw_dist)?;
        let key = if unflatten {
            root_name.rsplit('.').next().unwrap_or(&root_name).to_owned()
        } else {
            root_name
        };
        map.insert(key, TraceResult::Distribution(dist));
    }
    Ok(TraceResult::Multi(map))
}

fn insert_trace_nested(
    target: &mut IndexMap<String, TraceResult>,
    parts: &[&str],
    dist: TraceResult,
) {
    if parts.is_empty() {
        return;
    }
    if parts.len() == 1 {
        target.insert(parts[0].to_owned(), dist);
        return;
    }
    let head = parts[0].to_owned();
    let entry = target
        .entry(head)
        .or_insert_with(|| TraceResult::Multi(IndexMap::new()));
    if let TraceResult::Multi(child) = entry {
        insert_trace_nested(child, &parts[1..], dist);
    }
}

/// Computes the probability distribution for one or more targets given prebuilt network `evidence`.
pub fn trace_with_evidence(
    model: &FpgenModel,
    targets: &[&str],
    evidence: &IndexMap<String, AllowedValues>,
    unflatten: bool,
) -> Result<TraceResult> {
    if targets.is_empty() {
        return Err(FpgenError::ValueError(
            "trace requires at least one target".into(),
        ));
    }
    if targets.len() == 1 {
        return trace_single_target(model, targets[0], evidence, unflatten);
    }

    let mut root_map: IndexMap<String, TraceResult> = IndexMap::new();
    for &target in targets {
        let single = trace_single_target(model, target, evidence, unflatten)?;
        if unflatten {
            let parts: Vec<&str> = target.split('.').collect();
            insert_trace_nested(&mut root_map, &parts, single);
        } else {
            root_map.insert(target.to_owned(), single);
        }
    }
    Ok(TraceResult::Multi(root_map))
}

/// Traces target probability distributions under optional JSON `conditions`.
pub fn trace(
    targets: &[&str],
    conditions: Option<&Value>,
    strict: bool,
    unflatten: bool,
    model: Option<&FpgenModel>,
) -> Result<TraceResult> {
    match model {
        Some(m) => {
            let evidence = build_evidence(m, conditions, strict)?;
            trace_with_evidence(m, targets, &evidence, unflatten)
        }
        None => {
            let default_model = get_model(None)?;
            let evidence = build_evidence(&default_model, conditions, strict)?;
            trace_with_evidence(&default_model, targets, &evidence, unflatten)
        }
    }
}
