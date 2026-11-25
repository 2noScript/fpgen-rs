use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use serde::Deserialize;
use serde_json::Value;

use crate::exceptions::{FpgenError, Result};
use crate::pyjson::casefold;

const BEAM_WIDTH: usize = 1000;

#[derive(Debug, Clone)]
pub enum AllowedValues {
    Set(IndexSet<String>),
    Substring(String),
}

impl AllowedValues {
    pub fn contains(&self, value: &str) -> bool {
        match self {
            Self::Set(set) => set.contains(value),
            Self::Substring(text) => text.contains(value),
        }
    }

    pub fn from_single(value: String) -> Self {
        let mut set = IndexSet::new();
        set.insert(value);
        Self::Set(set)
    }

    pub fn as_set(&self) -> Option<&IndexSet<String>> {
        match self {
            Self::Set(set) => Some(set),
            Self::Substring(_) => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ProbNode {
    Leaf(IndexMap<String, f64>),
    Branch(IndexMap<String, ProbNode>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeDefinition {
    pub name: String,
    pub parent_names: Vec<String>,
    pub possible_values: Vec<String>,
    pub conditional_probabilities: ProbNode,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkDefinition {
    pub nodes: Vec<NodeDefinition>,
}

pub trait ValueLookup: Send + Sync {
    fn lookup_value_list(&self, ids: &[String]) -> Result<Vec<String>>;
}

#[derive(Debug, Clone)]
pub struct BayesianNode {
    pub index: usize,
    pub name: String,
    pub parent_names: Vec<String>,
    pub possible_values: Vec<String>,
    pub probabilities: ProbNode,
}

impl BayesianNode {
    pub fn new(def: NodeDefinition, index: usize) -> Self {
        Self {
            index,
            name: def.name,
            parent_names: def.parent_names,
            possible_values: def.possible_values,
            probabilities: def.conditional_probabilities,
        }
    }

    pub fn get_probabilities_given_known_values(
        &self,
        parent_values: &[&str],
    ) -> IndexMap<String, f64> {
        let mut current = &self.probabilities;
        for parent_val in parent_values {
            match current {
                ProbNode::Branch(map) => match map.get(*parent_val) {
                    Some(next) => current = next,
                    None => return IndexMap::new(),
                },
                ProbNode::Leaf(_) => return IndexMap::new(),
            }
        }
        match current {
            ProbNode::Leaf(leaf) => leaf.clone(),
            ProbNode::Branch(_) => IndexMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CaseInsensitiveMap<V> {
    inner: IndexMap<String, V>,
}

impl<V> Default for CaseInsensitiveMap<V> {
    fn default() -> Self {
        Self {
            inner: IndexMap::new(),
        }
    }
}

impl<V> CaseInsensitiveMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.inner.get(&casefold(key))
    }

    pub fn insert(&mut self, key: &str, value: V) -> Option<V> {
        self.inner.insert(casefold(key), value)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(&casefold(key))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }
}

pub struct BayesianNetwork {
    values: Arc<dyn ValueLookup>,
    pub nodes_in_sampling_order: Vec<BayesianNode>,
    pub nodes_by_name: CaseInsensitiveMap<BayesianNode>,
    pub node_names: Vec<String>,
    ancestors_by_name: HashMap<String, HashSet<String>>,
}

impl BayesianNetwork {
    pub fn new(def: NetworkDefinition, values: Arc<dyn ValueLookup>) -> Result<Self> {
        let nodes_in_sampling_order: Vec<BayesianNode> = def
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, d)| BayesianNode::new(d, i))
            .collect();

        let mut by_name: IndexMap<String, BayesianNode> = IndexMap::new();
        for node in &nodes_in_sampling_order {
            by_name.insert(node.name.clone(), node.clone());
        }

        let mut nodes_by_name = CaseInsensitiveMap::new();
        let mut node_names = Vec::with_capacity(by_name.len());
        for (name, node) in by_name {
            node_names.push(name.clone());
            nodes_by_name.insert(&name, node);
        }

        let mut net = Self {
            values,
            nodes_in_sampling_order,
            nodes_by_name,
            node_names,
            ancestors_by_name: HashMap::new(),
        };

        let names: Vec<String> = net
            .nodes_in_sampling_order
            .iter()
            .map(|n| n.name.clone())
            .collect();
        for name in names {
            net.compute_ancestors(&name)?;
        }
        Ok(net)
    }

    pub fn node(&self, name: &str) -> Result<&BayesianNode> {
        self.nodes_by_name
            .get(name)
            .ok_or_else(|| FpgenError::InvalidNode(format!("KeyError: '{}'", casefold(name))))
    }

    fn compute_ancestors(&mut self, node_name: &str) -> Result<HashSet<String>> {
        if let Some(cached) = self.ancestors_by_name.get(node_name) {
            return Ok(cached.clone());
        }
        let parents = self.node(node_name)?.parent_names.clone();
        let mut ancestors = HashSet::new();
        for parent in parents {
            ancestors.insert(parent.clone());
            for a in self.compute_ancestors(&parent)? {
                ancestors.insert(a);
            }
        }
        self.ancestors_by_name
            .insert(node_name.to_owned(), ancestors.clone());
        Ok(ancestors)
    }

    pub fn get_all_ancestors(&self, node_name: &str) -> Result<&HashSet<String>> {
        self.ancestors_by_name
            .get(node_name)
            .ok_or_else(|| FpgenError::InvalidNode(format!("KeyError: '{}'", casefold(node_name))))
    }

    pub fn generate_consistent_sample(
        &self,
        evidence: &IndexMap<String, AllowedValues>,
    ) -> Result<IndexMap<String, String>> {
        let mut result = IndexMap::new();
        let mut current_evidence = evidence.clone();

        for node in &self.nodes_in_sampling_order {
            let node_name = &node.name;
            let sampled_value = if let Some(allowed_values) = current_evidence.get(node_name) {
                let mut search_evidence = current_evidence.clone();
                search_evidence.shift_remove(node_name);
                let distribution = self.trace(node_name, &search_evidence)?;

                let mut filtered = IndexMap::new();
                for (k, v) in distribution {
                    if allowed_values.contains(&k) {
                        filtered.insert(k, v);
                    }
                }
                let total: f64 = filtered.values().sum();
                if filtered.is_empty() || total <= 0.0 {
                    let allowed_set = allowed_values.as_set().ok_or_else(|| {
                        FpgenError::RestrictiveConstraints(format!(
                            "Cannot sample {node_name} from empty filtered distribution"
                        ))
                    })?;
                    let uniform = 1.0 / (allowed_set.len() as f64);
                    filtered = allowed_set.iter().map(|v| (v.clone(), uniform)).collect();
                } else {
                    for p in filtered.values_mut() {
                        *p /= total;
                    }
                }
                self.sample_value_from_distribution(&filtered)
            } else {
                let distribution = self.trace(node_name, &current_evidence)?;
                self.sample_value_from_distribution(&distribution)
            };

            result.insert(node_name.clone(), sampled_value.clone());
            current_evidence.insert(node_name.clone(), AllowedValues::from_single(sampled_value));
        }

        Ok(result)
    }

    pub fn generate_certain_nodes(
        &self,
        evidence: &IndexMap<String, AllowedValues>,
        targets: Option<&[String]>,
    ) -> Result<IndexMap<String, String>> {
        let Some(targets) = targets else {
            return self.generate_consistent_sample(evidence);
        };

        let mut result = IndexMap::new();
        for target_node in targets {
            let mut distribution = self.trace(target_node, evidence)?;
            if let Some(allowed_values) = evidence.get(target_node) {
                let mut filtered = IndexMap::new();
                for (k, v) in distribution {
                    if allowed_values.contains(&k) {
                        filtered.insert(k, v);
                    }
                }
                let total: f64 = filtered.values().sum();
                if filtered.is_empty() || total <= 0.0 {
                    return Err(FpgenError::RestrictiveConstraints(format!(
                        "Cannot generate fingerprint: No valid values for {target_node} with current conditions."
                    )));
                }
                for p in filtered.values_mut() {
                    *p /= total;
                }
                distribution = filtered;
            }
            if distribution.is_empty() {
                return Err(FpgenError::RestrictiveConstraints(format!(
                    "Cannot generate fingerprint: Empty distribution for {target_node}."
                )));
            }
            let sampled = self.sample_value_from_distribution(&distribution);
            result.insert(target_node.clone(), sampled);
        }
        Ok(result)
    }

    pub fn validate_evidence(&self, evidence: &IndexMap<String, AllowedValues>) -> Result<()> {
        if evidence.len() <= 1 {
            return Ok(());
        }

        for (node_name, allowed_values) in evidence {
            let mut fixed: IndexMap<String, AllowedValues> = IndexMap::new();
            let mut fixed_ids: Vec<String> = Vec::new();
            let mut fixed_keys: Vec<&str> = Vec::new();
            for (k, v) in evidence {
                if k != node_name {
                    if let Some(set) = v.as_set() {
                        if set.len() == 1 {
                            let val = set[0].clone();
                            fixed.insert(k.clone(), AllowedValues::Substring(val.clone()));
                            fixed_ids.push(val);
                            fixed_keys.push(k.as_str());
                        }
                    }
                }
            }
            if fixed.is_empty() {
                continue;
            }

            let dist = self.trace(node_name, &fixed)?;
            let allowed_set = match allowed_values.as_set() {
                Some(s) => s,
                None => continue,
            };
            if !dist.is_empty()
                && allowed_set
                    .iter()
                    .all(|val| dist.get(val).copied().unwrap_or(0.0) <= 0.0)
            {
                let preview: Vec<String> = allowed_set.iter().take(5).cloned().collect();
                let mut values_str = self.values.lookup_value_list(&preview)?.join(", ");
                if allowed_set.len() > 5 {
                    values_str.push_str(", ...");
                }
                let constraint_values = self.values.lookup_value_list(&fixed_ids)?;
                let constraints_str = fixed_keys
                    .iter()
                    .enumerate()
                    .map(|(i, k)| format!("{k}={}", constraint_values[i]))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(FpgenError::RestrictiveConstraints(format!(
                    "Cannot generate fingerprint: {node_name}=({values_str}) is impossible with constraint: {constraints_str}"
                )));
            }
        }
        Ok(())
    }

    pub fn trace(
        &self,
        target: &str,
        evidence: &IndexMap<String, AllowedValues>,
    ) -> Result<IndexMap<String, f64>> {
        let target_name = self.node(target)?.name.clone();
        let mut relevant: HashSet<String> = self.get_all_ancestors(&target_name)?.clone();
        relevant.insert(target_name.clone());

        for ev_node in evidence.keys() {
            if self.nodes_by_name.contains_key(ev_node) {
                relevant.insert(ev_node.clone());
                for a in self.get_all_ancestors(ev_node)? {
                    relevant.insert(a.clone());
                }
            }
        }

        let ordered: Vec<&BayesianNode> = self
            .nodes_in_sampling_order
            .iter()
            .filter(|n| relevant.contains(&n.name))
            .collect();

        let mut slot: HashMap<&str, usize> = HashMap::with_capacity(ordered.len());
        for (i, n) in ordered.iter().enumerate() {
            slot.insert(n.name.as_str(), i);
        }

        #[derive(Clone)]
        struct BeamItem {
            values: Vec<String>,
            prob: f64,
        }

        let mut beam = vec![BeamItem {
            values: Vec::new(),
            prob: 1.0,
        }];

        for node in &ordered {
            let allowed_values = evidence.get(&node.name);
            let mut parent_slots = Vec::with_capacity(node.parent_names.len());
            for p in &node.parent_names {
                let s = slot
                    .get(p.as_str())
                    .copied()
                    .ok_or_else(|| FpgenError::InvalidNode(format!("KeyError: '{p}'")))?;
                parent_slots.push(s);
            }

            let mut new_beam: Vec<BeamItem> = Vec::new();
            let mut cpt_cache: HashMap<String, IndexMap<String, f64>> = HashMap::new();

            for item in &beam {
                let parent_values: Vec<&str> = parent_slots
                    .iter()
                    .map(|&s| item.values[s].as_str())
                    .collect();
                let cache_key = parent_values.join("\u{0000}");

                let cpt = if let Some(cached) = cpt_cache.get(&cache_key) {
                    cached
                } else {
                    let mut table = node.get_probabilities_given_known_values(&parent_values);
                    if table.is_empty() && !node.possible_values.is_empty() {
                        let uniform = 1.0 / (node.possible_values.len() as f64);
                        table = node
                            .possible_values
                            .iter()
                            .map(|v| (v.clone(), uniform))
                            .collect();
                    }
                    cpt_cache.entry(cache_key).or_insert(table)
                };

                for (value, &p) in cpt {
                    if p > 0.0 && allowed_values.is_none_or(|allowed| allowed.contains(value)) {
                        let mut next = item.values.clone();
                        next.push(value.clone());
                        new_beam.push(BeamItem {
                            values: next,
                            prob: item.prob * p,
                        });
                    }
                }
            }

            if new_beam.is_empty() {
                return Ok(IndexMap::new());
            }
            if new_beam.len() > BEAM_WIDTH {
                new_beam.sort_by(|a, b| b.prob.total_cmp(&a.prob));
                new_beam.truncate(BEAM_WIDTH);
            }
            beam = new_beam;
        }

        let target_slot = slot[target_name.as_str()];
        let mut target_dist: IndexMap<String, f64> = IndexMap::new();
        let mut total_prob = 0.0;
        for item in beam {
            let val = item.values[target_slot].clone();
            *target_dist.entry(val).or_insert(0.0) += item.prob;
            total_prob += item.prob;
        }

        if total_prob > 0.0 {
            for p in target_dist.values_mut() {
                *p /= total_prob;
            }
            Ok(target_dist)
        } else {
            Ok(IndexMap::new())
        }
    }

    pub fn sample_value_from_distribution(&self, distribution: &IndexMap<String, f64>) -> String {
        let mut rng = rand::rng();
        let anchor: f64 = rng.random();
        let mut cumulative = 0.0;
        for (value, &probability) in distribution {
            cumulative += probability;
            if anchor < cumulative {
                return value.clone();
            }
        }
        distribution.keys().next().cloned().unwrap_or_default()
    }
}

pub fn is_plain_object(value: &Value) -> bool {
    matches!(value, Value::Object(_))
}
