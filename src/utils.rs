use indexmap::{IndexMap, IndexSet};
use serde_json::{Map, Value};

use crate::bayesian_network::{is_plain_object, AllowedValues, BayesianNetwork, BayesianNode};
use crate::exceptions::{FpgenError, Result};
use crate::model::FpgenModel;
use crate::pyjson::{casefold, parse_py_typed, py_equals};

/// Deep-merges `incoming` into `base`, recursively merging nested JSON objects.
pub fn merge_dicts(base: &mut Map<String, Value>, incoming: &Map<String, Value>) {
    for (k, v) in incoming {
        match (base.get_mut(k), v) {
            (Some(Value::Object(existing)), Value::Object(inc_map)) => {
                merge_dicts(existing, inc_map);
            }
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Unflattens dotted keys (`"navigator.userAgent"`) into nested JSON objects, deep-merging collisions.
pub fn unflatten(data: &IndexMap<String, Value>) -> Value {
    let mut root = Map::new();
    for (key, value) in data {
        let parts: Vec<&str> = key.split('.').collect();
        insert_nested(&mut root, &parts, value.clone());
    }
    Value::Object(root)
}

fn insert_nested(target: &mut Map<String, Value>, parts: &[&str], value: Value) {
    if parts.is_empty() {
        return;
    }
    if parts.len() == 1 {
        let leaf = parts[0].to_owned();
        match (target.get_mut(&leaf), value) {
            (Some(Value::Object(existing)), Value::Object(inc)) => {
                merge_dicts(existing, &inc);
            }
            (_, v) => {
                target.insert(leaf, v);
            }
        }
        return;
    }
    let head = parts[0];
    if !matches!(target.get(head), Some(Value::Object(_))) {
        target.insert(head.to_owned(), Value::Object(Map::new()));
    }
    if let Some(Value::Object(child)) = target.get_mut(head) {
        insert_nested(child, &parts[1..], value);
    }
}

/// Flattens a nested JSON object into dotted keys (`sep`, typically `"."`).
/// When `fold` is true, each leaf's full dotted key is passed through `casefold`.
pub fn flatten_dict(
    map: &Map<String, Value>,
    sep: &str,
    fold: bool,
) -> IndexMap<String, Value> {
    let mut out = IndexMap::new();
    flatten_into(map, "", sep, fold, &mut out);
    out
}

fn flatten_into(
    map: &Map<String, Value>,
    prefix: &str,
    sep: &str,
    fold: bool,
    out: &mut IndexMap<String, Value>,
) {
    for (k, v) in map {
        let new_key = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}{sep}{k}")
        };
        if let Value::Object(sub) = v {
            if !sub.is_empty() {
                flatten_into(sub, &new_key, sep, fold, out);
                continue;
            }
        }
        let stored_key = if fold { casefold(&new_key) } else { new_key };
        out.insert(stored_key, v.clone());
    }
}

/// Flattens JSON objects into dotted keys; returns other values unchanged.
pub fn maybe_flatten(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let flat = flatten_dict(map, ".", false);
            let mut out = Map::with_capacity(flat.len());
            for (k, v) in flat {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Flattens a condition map into casefolded dotted keys without mutating intermediate keys prematurely.
pub fn flatten_conditions(map: &Map<String, Value>) -> IndexMap<String, Value> {
    flatten_dict(map, ".", true)
}

/// Deduplicates values using Python equality semantics while preserving first-seen order.
pub fn dedupe(values: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(values.len());
    for v in values {
        if !out.iter().any(|seen| py_equals(seen, &v)) {
            out.push(v);
        }
    }
    out
}

/// Traverses `data` along `path` using case-insensitive key lookup.
pub fn at_path(data: &Value, path: &[&str]) -> Result<Value> {
    let mut current = data;
    for &segment in path {
        let obj = current.as_object().ok_or_else(|| {
            FpgenError::NodePathError(format!(
                "Cannot traverse segment '{segment}' on non-object value"
            ))
        })?;
        if let Some(exact) = obj.get(segment) {
            current = exact;
            continue;
        }
        let target_fold = casefold(segment);
        let found = obj.iter().find(|(k, _)| casefold(k) == target_fold);
        match found {
            Some((_, v)) => current = v,
            None => {
                return Err(FpgenError::NodePathError(format!(
                    "Path segment '{segment}' not found"
                )))
            }
        }
    }
    Ok(current.clone())
}

fn find_at_path_opt(data: &Value, path: &[&str]) -> Option<Value> {
    at_path(data, path).ok()
}

/// Finds all Bayesian network root node names matching or containing `target`.
pub fn find_roots(net: &BayesianNetwork, target: &str) -> Result<Vec<String>> {
    let target_fold = casefold(target);
    let prefix_dot = format!("{target_fold}.");

    let mut children: Vec<String> = net
        .node_names
        .iter()
        .filter(|name| casefold(name).starts_with(&prefix_dot))
        .cloned()
        .collect();

    if let Ok(node) = net.node(target) {
        if children.is_empty() {
            return Ok(vec![node.name.clone()]);
        }
        let mut all = vec![node.name.clone()];
        all.append(&mut children);
        return Ok(all);
    }

    if !children.is_empty() {
        return Ok(children);
    }

    let parts: Vec<&str> = target.split('.').collect();
    for i in (1..parts.len()).rev() {
        let prefix = parts[..i].join(".");
        if let Ok(node) = net.node(&prefix) {
            return Ok(vec![node.name.clone()]);
        }
    }

    Err(FpgenError::InvalidNode(format!(
        "{target} is not a valid node"
    )))
}

/// Resolves and parses every possible value for a single root Bayesian node.
pub fn lookup_root_possibilities(node: &BayesianNode, model: &FpgenModel) -> Result<Vec<Value>> {
    let raw = model.lookup_value_list(&node.possible_values)?;
    raw.into_iter().map(|s| parse_py_typed(&s)).collect()
}

/// Extracts values along `sub_path` across all parsed root possibilities, deduplicated via `py_equals`.
pub fn search_downward(
    node: &BayesianNode,
    sub_path: &[&str],
    model: &FpgenModel,
) -> Result<Vec<Value>> {
    let roots = lookup_root_possibilities(node, model)?;
    if sub_path.is_empty() {
        return Ok(dedupe(roots));
    }
    let mut matches = Vec::new();
    for root in &roots {
        if let Some(val) = find_at_path_opt(root, sub_path) {
            matches.push(val);
        }
    }
    if matches.is_empty() {
        return Err(FpgenError::NodePathError(format!(
            "Sub-path '{}' not found inside node '{}'",
            sub_path.join("."),
            node.name
        )));
    }
    Ok(dedupe(matches))
}

/// Computes the relative path segments inside `root_name` for `target`.
pub(crate) fn sub_path_for<'a>(target: &'a str, root_name: &str) -> Vec<&'a str> {
    let target_fold = casefold(target);
    let root_fold = casefold(root_name);
    if target_fold == root_fold {
        return Vec::new();
    }
    let root_parts_len = root_name.split('.').count();
    let target_parts: Vec<&str> = target.split('.').collect();
    if target_parts.len() > root_parts_len {
        target_parts[root_parts_len..].to_vec()
    } else {
        Vec::new()
    }
}

/// Lists all possible values for `target` (JSON array for single node/path, JSON object for multi-node namespaces).
pub fn lookup_possibilities(model: &FpgenModel, target: &str) -> Result<Value> {
    let roots = find_roots(&model.network, target)?;
    if roots.len() == 1 {
        let root_name = &roots[0];
        let node = model.network.node(root_name)?;
        if casefold(root_name) == casefold(target) {
            return Ok(Value::Array(lookup_root_possibilities(node, model)?));
        }
        let target_fold = casefold(target);
        let root_fold = casefold(root_name);
        if root_fold.starts_with(&format!("{target_fold}.")) {
            let vals = lookup_root_possibilities(node, model)?;
            let prefix_len = target.split('.').count();
            let suffix = root_name.split('.').collect::<Vec<_>>()[prefix_len..].join(".");
            let mut m = Map::new();
            m.insert(suffix, Value::Array(vals));
            return Ok(Value::Object(m));
        }
        let sub = sub_path_for(target, root_name);
        let out = search_downward(node, &sub, model)?;
        return Ok(Value::Array(out));
    }

    let prefix_len = target.split('.').count();
    let mut map = IndexMap::new();
    for root_name in roots {
        if casefold(&root_name) == casefold(target) {
            continue;
        }
        let node = model.network.node(&root_name)?;
        let vals = lookup_root_possibilities(node, model)?;
        let parts: Vec<&str> = root_name.split('.').collect();
        let suffix = parts[prefix_len..].join(".");
        map.insert(suffix, Value::Array(vals));
    }
    Ok(unflatten(&map))
}

/// Queries all possible values for `target` using the given or default `FpgenModel`.
pub fn query(target: &str, model: Option<&FpgenModel>) -> Result<Value> {
    match model {
        Some(m) => lookup_possibilities(m, target),
        None => {
            let default_model = crate::model::get_model(None)?;
            lookup_possibilities(&default_model, target)
        }
    }
}

fn value_matches_condition(candidate: &Value, expected: &Value) -> bool {
    if py_equals(candidate, expected) {
        return true;
    }
    if let Value::Array(options) = expected {
        return options.iter().any(|opt| py_equals(candidate, opt));
    }
    false
}

/// Converts user condition constraints into Bayesian network node evidence.
pub fn build_evidence(
    model: &FpgenModel,
    conditions: Option<&Value>,
    strict: bool,
) -> Result<IndexMap<String, AllowedValues>> {
    let mut evidence: IndexMap<String, AllowedValues> = IndexMap::new();
    let Some(cond_val) = conditions else {
        return Ok(evidence);
    };
    if cond_val.is_null() {
        return Ok(evidence);
    }
    let cond_obj = cond_val.as_object().ok_or_else(|| {
        FpgenError::InvalidConstraints("Conditions must be a JSON object".into())
    })?;
    if cond_obj.is_empty() {
        return Ok(evidence);
    }

    let flat = flatten_conditions(cond_obj);
    for (cond_key, expected) in flat {
        if let Value::Array(arr) = &expected {
            if arr.is_empty() {
                if strict {
                    return Err(FpgenError::InvalidConstraints(format!(
                        "Empty constraint array for '{cond_key}'"
                    )));
                }
                continue;
            }
        }

        let roots = find_roots(&model.network, &cond_key)?;

        for root_name in roots {
            let node = model.network.node(&root_name)?;
            let root_fold = casefold(&root_name);
            let sub_parts: Vec<&str> = if root_fold == cond_key {
                Vec::new()
            } else if cond_key.starts_with(&format!("{root_fold}.")) {
                cond_key[root_fold.len() + 1..].split('.').collect()
            } else {
                continue;
            };

            let raw_values = model.lookup_value_list(&node.possible_values)?;
            let mut matched_ids: IndexSet<String> = IndexSet::new();
            let mut saw_path = sub_parts.is_empty();

            for (id, raw_json) in node.possible_values.iter().zip(raw_values.iter()) {
                let parsed = parse_py_typed(raw_json)?;
                if sub_parts.is_empty() {
                    if value_matches_condition(&parsed, &expected) {
                        matched_ids.insert(id.clone());
                    }
                } else if let Some(found) = find_at_path_opt(&parsed, &sub_parts) {
                    saw_path = true;
                    if value_matches_condition(&found, &expected) {
                        matched_ids.insert(id.clone());
                    }
                }
            }

            if !saw_path {
                if strict {
                    return Err(FpgenError::InvalidConstraints(format!(
                        "Constraint path '{cond_key}' not found in node '{root_name}'"
                    )));
                }
                continue;
            }

            if matched_ids.is_empty() {
                return Err(FpgenError::RestrictiveConstraints(format!(
                    "No possible values match constraint {cond_key}={expected}"
                )));
            }

            if let Some(AllowedValues::Set(existing)) = evidence.get_mut(&root_name) {
                existing.retain(|id| matched_ids.contains(id));
                if existing.is_empty() {
                    return Err(FpgenError::RestrictiveConstraints(format!(
                        "Conflicting constraints on node '{root_name}' at '{cond_key}'"
                    )));
                }
            } else {
                evidence.insert(root_name, AllowedValues::Set(matched_ids));
            }
        }
    }

    if !evidence.is_empty() {
        model.network.validate_evidence(&evidence)?;
    }

    Ok(evidence)
}

/// Validates and merges base + override condition dicts, returning built evidence.
pub fn assert_conditions(
    model: &FpgenModel,
    base: Option<&Value>,
    overrides: Option<&Value>,
    strict: bool,
) -> Result<IndexMap<String, AllowedValues>> {
    let merged = match (base, overrides) {
        (None, None) => return Ok(IndexMap::new()),
        (Some(b), None) => b.clone(),
        (None, Some(o)) => o.clone(),
        (Some(b), Some(o)) => {
            if b.is_null() {
                o.clone()
            } else if o.is_null() {
                b.clone()
            } else {
                let mut m = b
                    .as_object()
                    .ok_or_else(|| {
                        FpgenError::InvalidConstraints("Base conditions must be an object".into())
                    })?
                    .clone();
                let o_map = o.as_object().ok_or_else(|| {
                    FpgenError::InvalidConstraints("Override conditions must be an object".into())
                })?;
                merge_dicts(&mut m, o_map);
                Value::Object(m)
            }
        }
    };
    build_evidence(model, Some(&merged), strict)
}

/// Decodes a raw `{node_name: value_id}` sample into parsed JSON values, optionally unflattened.
pub fn make_output_dict(
    sample: &IndexMap<String, String>,
    model: &FpgenModel,
    unflatten_output: bool,
) -> Result<Value> {
    if sample.is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    let keys: Vec<&String> = sample.keys().collect();
    let ids: Vec<String> = sample.values().cloned().collect();
    let raw_values = model.lookup_value_list(&ids)?;

    let mut flat: IndexMap<String, Value> = IndexMap::with_capacity(keys.len());
    for (k, raw) in keys.into_iter().zip(raw_values) {
        let parsed = parse_py_typed(&raw)?;
        flat.insert(k.clone(), parsed);
    }

    if unflatten_output {
        Ok(unflatten(&flat))
    } else {
        let mut obj = Map::with_capacity(flat.len());
        for (k, v) in flat {
            obj.insert(k, v);
        }
        Ok(Value::Object(obj))
    }
}

/// Extracts and reassembles requested `targets` from a flat `{node_name: value_id}` sample.
pub fn reassemble_targets(
    sample: &IndexMap<String, String>,
    model: &FpgenModel,
    targets: &[String],
    unflatten_output: bool,
    strict: bool,
) -> Result<Value> {
    let full = make_output_dict(sample, model, true)?;
    let mut selected: IndexMap<String, Value> = IndexMap::new();

    for target in targets {
        let parts: Vec<&str> = target.split('.').collect();
        match at_path(&full, &parts) {
            Ok(v) => {
                if unflatten_output {
                    selected.insert(target.clone(), v);
                } else if is_plain_object(&v) {
                    if let Some(obj) = v.as_object() {
                        let flat = flatten_dict(obj, ".", false);
                        for (fk, fv) in flat {
                            selected.insert(format!("{target}.{fk}"), fv);
                        }
                    }
                } else {
                    selected.insert(target.clone(), v);
                }
            }
            Err(e) => {
                if strict {
                    return Err(e);
                }
            }
        }
    }

    if unflatten_output {
        Ok(unflatten(&selected))
    } else {
        let mut obj = Map::with_capacity(selected.len());
        for (k, v) in selected {
            obj.insert(k, v);
        }
        Ok(Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flatten_and_unflatten_roundtrip() {
        let input = json!({
            "navigator": {
                "userAgent": "Mozilla/5.0",
                "platform": "MacIntel"
            },
            "screen": {
                "width": 1920
            }
        });
        let flat = flatten_dict(input.as_object().unwrap(), ".", false);
        assert_eq!(flat.get("navigator.userAgent").unwrap(), "Mozilla/5.0");
        assert_eq!(flat.get("screen.width").unwrap(), 1920);
        let restored = unflatten(&flat);
        assert_eq!(restored, input);
    }

    #[test]
    fn dedupe_uses_py_equals() {
        let items = vec![json!(1), json!(1.0), json!(true), json!("1"), json!(2)];
        let out = dedupe(items);
        assert_eq!(out, vec![json!(1), json!("1"), json!(2)]);
    }
}
