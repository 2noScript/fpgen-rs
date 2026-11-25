use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{json, Value};

use fpgen_rs::bayesian_network::{
    AllowedValues, BayesianNetwork, NetworkDefinition, NodeDefinition, ProbNode, ValueLookup,
};
use fpgen_rs::exceptions::FpgenError;
use fpgen_rs::model::verify_archive;
use fpgen_rs::pin::MODEL_PIN;
use fpgen_rs::zip::read_zip;

struct MemoryLookup {
    values: HashMap<String, String>,
}

impl ValueLookup for MemoryLookup {
    fn lookup_value_list(&self, ids: &[String]) -> fpgen_rs::Result<Vec<String>> {
        ids.iter()
            .map(|id| {
                self.values
                    .get(id)
                    .cloned()
                    .ok_or_else(|| FpgenError::ValueError(format!("missing value id {id}")))
            })
            .collect()
    }
}

fn sample_network() -> BayesianNetwork {
    let mut os_probs = IndexMap::new();
    os_probs.insert("00".into(), 0.7);
    os_probs.insert("0R".into(), 0.3);

    let mut mac_depth_probs = IndexMap::new();
    mac_depth_probs.insert("00".into(), 0.0);
    mac_depth_probs.insert("0R".into(), 1.0);

    let mut win_depth_probs = IndexMap::new();
    win_depth_probs.insert("00".into(), 0.8);
    win_depth_probs.insert("0R".into(), 0.2);

    let mut depth_branches = IndexMap::new();
    depth_branches.insert("00".into(), ProbNode::Leaf(win_depth_probs));
    depth_branches.insert("0R".into(), ProbNode::Leaf(mac_depth_probs));

    let def = NetworkDefinition {
        nodes: vec![
            NodeDefinition {
                name: "navigator.platform".into(),
                parent_names: vec![],
                possible_values: vec!["00".into(), "0R".into()],
                conditional_probabilities: ProbNode::Leaf(os_probs),
            },
            NodeDefinition {
                name: "screen.colorDepth".into(),
                parent_names: vec!["navigator.platform".into()],
                possible_values: vec!["00".into(), "0R".into()],
                conditional_probabilities: ProbNode::Branch(depth_branches),
            },
        ],
    };

    let mut values = HashMap::new();
    values.insert("00".into(), "\"Win32\"".into());
    values.insert("0R".into(), "\"MacIntel\"".into());

    BayesianNetwork::new(def, Arc::new(MemoryLookup { values })).unwrap()
}

#[test]
fn synthetic_network_trace_and_sampling_respect_constraints() {
    let net = sample_network();

    let mut evidence = IndexMap::new();
    evidence.insert(
        "navigator.platform".into(),
        AllowedValues::from_single("0R".into()),
    );

    let dist = net.trace("screen.colorDepth", &evidence).unwrap();
    assert!((dist.get("0R").copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
    assert!((dist.get("00").copied().unwrap_or(0.0) - 0.0).abs() < 1e-12);

    for _ in 0..10 {
        let sample = net.generate_consistent_sample(&evidence).unwrap();
        assert_eq!(sample.get("navigator.platform").unwrap(), "0R");
        assert_eq!(sample.get("screen.colorDepth").unwrap(), "0R");
    }
}

#[test]
fn impossible_evidence_raises_restrictive_constraints() {
    let net = sample_network();
    let mut evidence = IndexMap::new();
    evidence.insert(
        "navigator.platform".into(),
        AllowedValues::from_single("0R".into()),
    );
    evidence.insert(
        "screen.colorDepth".into(),
        AllowedValues::from_single("00".into()),
    );

    let err = net.validate_evidence(&evidence).unwrap_err();
    assert!(matches!(err, FpgenError::RestrictiveConstraints(_)));
}

#[test]
fn verify_archive_rejects_corrupted_zip() {
    let mut buf = Vec::new();
    {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let options: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default();
        writer.start_file("bogus.txt", options).unwrap();
        writer.write_all(b"invalid").unwrap();
        writer.finish().unwrap();
    }

    let entries = read_zip(&buf).unwrap();
    assert_eq!(entries.get("bogus.txt").unwrap(), b"invalid");

    let err = verify_archive(&buf, &MODEL_PIN).unwrap_err();
    assert!(matches!(err, FpgenError::ModelVerificationError(_)));
}

#[test]
fn json_value_unflatten_merge_is_deterministic() {
    let mut flat = IndexMap::new();
    flat.insert("navigator.platform".into(), json!("MacIntel"));
    flat.insert("navigator.hardwareConcurrency".into(), json!(8));
    flat.insert("screen".into(), json!({ "width": 1512, "colorDepth": 30 }));
    let nested: Value = fpgen_rs::utils::unflatten(&flat);
    assert_eq!(nested["navigator"]["platform"], "MacIntel");
    assert_eq!(nested["navigator"]["hardwareConcurrency"], 8);
    assert_eq!(nested["screen"]["colorDepth"], 30);
}
