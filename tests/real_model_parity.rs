use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::json;

use fpgen_rs::{
    ensure_model, generate, generate_target, is_model_installed, query, set_model_dir, trace,
    EnsureModelOptions, FpgenError, GenerateOptions, Generator, GeneratorOptions, MODEL_PIN,
};

fn setup_cached_or_downloaded_model() -> PathBuf {
    let mac_cache = dirs::home_dir()
        .unwrap()
        .join("Library/Caches/camoufox/fpgen");
    if is_model_installed(&mac_cache, &MODEL_PIN) {
        set_model_dir(Some(mac_cache.clone()));
        return mac_cache;
    }
    ensure_model(EnsureModelOptions::default())
        .expect("fpgen model must be installed or downloadable")
}

#[test]
#[ignore = "requires pre-installed or downloadable fpgen model (~212 MB)"]
fn query_and_trace_match_fpgen_js_golden() {
    let _dir = setup_cached_or_downloaded_model();

    let platforms = query("navigator.platform", None).unwrap();
    assert_eq!(
        platforms,
        json!(["Linux x86_64", "MacIntel", "Win32", "Linux armv81"])
    );

    let depths = query("screen.colorDepth", None).unwrap();
    assert_eq!(depths, json!([24, 30, 16]));

    let nav_tree = query("navigator", None).unwrap();
    let nav_obj = nav_tree.as_object().expect("navigator query returns object");
    assert!(nav_obj.contains_key("userAgent"));
    assert!(nav_obj.contains_key("platform"));
    assert!(nav_obj.contains_key("hardwareConcurrency"));

    let dist = trace(&["navigator.platform"], None, false, true, None).unwrap();
    let entries = dist.as_distribution().expect("single target distribution");

    let expected = [
        ("Win32", 0.5379113018597999_f64),
        ("MacIntel", 0.29852169766332853_f64),
        ("Linux x86_64", 0.16309012875536497_f64),
        ("Linux armv81", 0.00047687172150691495_f64),
    ];
    let mut sum = 0.0;
    for (plat, exp_prob) in expected {
        let found = entries
            .iter()
            .find(|e| e.value == json!(plat))
            .expect("platform missing in trace");
        assert!(
            (found.probability - exp_prob).abs() < 1e-12,
            "prob mismatch for {plat}: {} vs {exp_prob}",
            found.probability
        );
        sum += found.probability;
    }
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
#[ignore = "requires pre-installed or downloadable fpgen model (~212 MB)"]
fn generator_and_target_respect_conditions_and_keys() {
    let _dir = setup_cached_or_downloaded_model();

    let gen = Generator::new(GeneratorOptions {
        conditions: Some(json!({ "navigator": { "platform": "MacIntel" } })),
        strict: true,
        unflatten: true,
        model_dir: None,
    })
    .unwrap();

    let fp = gen.generate(None).unwrap();
    let obj = fp.as_object().unwrap();
    let keys: HashSet<&str> = obj.keys().map(String::as_str).collect();
    assert!(keys.contains("navigator"));
    assert!(keys.contains("screen"));
    assert!(keys.contains("webgl"));
    assert!(keys.contains("headers"));
    assert_eq!(fp["navigator"]["platform"], "MacIntel");

    let partial = generate_target(
        &["navigator.platform", "screen.colorDepth"],
        Some(GenerateOptions {
            conditions: Some(json!({ "navigator.platform": "MacIntel" })),
            strict: Some(true),
            unflatten: Some(false),
            model_dir: None,
        }),
    )
    .unwrap();
    assert_eq!(partial["navigator.platform"], "MacIntel");
    assert!(partial["screen.colorDepth"].is_i64());

    let bad = generate(Some(GenerateOptions {
        conditions: Some(json!({ "nonexistent.foo": 1 })),
        strict: Some(true),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(matches!(bad, FpgenError::InvalidNode(_)));

    let impossible = generate(Some(GenerateOptions {
        conditions: Some(json!({
            "navigator.platform": "MacIntel",
            "navigator.oscpu": "Windows NT 10.0; Win64; x64"
        })),
        strict: Some(true),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(matches!(impossible, FpgenError::RestrictiveConstraints(_)));

    let ci_fp = generate(Some(GenerateOptions {
        conditions: Some(json!({ "browser": "firefox", "os": "macOS" })),
        strict: Some(true),
        ..Default::default()
    }))
    .unwrap();
    assert_eq!(ci_fp["navigator"]["platform"], "MacIntel");
}
