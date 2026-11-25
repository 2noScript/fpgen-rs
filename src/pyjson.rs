use std::cmp::Ordering;

use serde_json::Value;

use crate::exceptions::{FpgenError, Result};

const B85_ALPHABET: &[u8; 85] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";

const fn build_b85_table() -> [i8; 256] {
    let mut table = [-1i8; 256];
    let mut i = 0;
    while i < 85 {
        table[B85_ALPHABET[i] as usize] = i as i8;
        i += 1;
    }
    table
}

const B85_DECODE: [i8; 256] = build_b85_table();

/// Parses JSON preserving object key document order (via `serde_json` `preserve_order`).
pub fn parse_ordered(text: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| FpgenError::ValueError(e.to_string()))
}

/// Parses a JSON token keeping Python's `int` vs `float` distinction (`1` vs `1.0`).
pub fn parse_py_typed(text: &str) -> Result<Value> {
    parse_ordered(text)
}

/// Python's `str.casefold()` over ASCII and Unicode special-casing characters.
pub fn casefold(s: &str) -> String {
    let lower = s.to_lowercase();
    if lower.is_ascii() {
        return lower;
    }
    lower
        .replace(['ß', 'ẞ'], "ss")
        .replace('ς', "σ")
        .replace('ſ', "s")
        .replace('ﬀ', "ff")
        .replace('ﬁ', "fi")
        .replace('ﬂ', "fl")
        .replace('ﬃ', "ffi")
        .replace('ﬄ', "ffl")
        .replace(['ﬅ', 'ﬆ'], "st")
}

fn py_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Python `==` over JSON values (`1 == 1.0 == True`, objects compared unordered).
pub fn py_equals(a: &Value, b: &Value) -> bool {
    if let (Some(na), Some(nb)) = (py_number(a), py_number(b)) {
        return na == nb;
    }
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Array(la), Value::Array(lb)) => {
            la.len() == lb.len() && la.iter().zip(lb).all(|(x, y)| py_equals(x, y))
        }
        (Value::Object(ma), Value::Object(mb)) => {
            ma.len() == mb.len()
                && ma
                    .iter()
                    .all(|(k, va)| mb.get(k).is_some_and(|vb| py_equals(va, vb)))
        }
        _ => false,
    }
}

/// Compact JSON serialization matching `orjson.dumps(value)` / `pyDumps`.
pub fn py_dumps(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(|e| FpgenError::ValueError(e.to_string()))
}

/// Decodes a RFC 1924 / Python `base64.b85decode` value ID to an integer offset index.
pub fn base85_to_int(s: &str) -> Result<usize> {
    let raw = s.as_bytes();
    let padding = (5 - (raw.len() % 5)) % 5;
    let mut padded = Vec::with_capacity(raw.len() + padding);
    padded.extend_from_slice(raw);
    padded.resize(raw.len() + padding, b'~');

    let mut bytes = Vec::with_capacity((padded.len() / 5) * 4);
    for (chunk_idx, chunk) in padded.chunks_exact(5).enumerate() {
        let mut acc: u64 = 0;
        for (offset, &byte) in chunk.iter().enumerate() {
            let d = B85_DECODE[byte as usize];
            if d < 0 {
                return Err(FpgenError::ValueError(format!(
                    "bad base85 character at position {}",
                    chunk_idx * 5 + offset
                )));
            }
            acc = acc * 85 + (d as u64);
        }
        if acc > 0xffff_ffff {
            return Err(FpgenError::ValueError(format!(
                "base85 overflow in hunk starting at byte {}",
                chunk_idx * 5
            )));
        }
        bytes.extend_from_slice(&(acc as u32).to_be_bytes());
    }
    if padding > 0 {
        bytes.truncate(bytes.len() - padding);
    }

    let mut n: u128 = 0;
    for b in bytes {
        n = n * 256 + u128::from(b);
    }
    usize::try_from(n).map_err(|_| FpgenError::ValueError(format!("base85 value too large: {s}")))
}

/// Python's default ordering for a homogeneous group (`str` by Unicode scalar value).
pub fn py_compare(a: &Value, b: &Value) -> Ordering {
    if let (Some(na), Some(nb)) = (py_number(a), py_number(b)) {
        return na.partial_cmp(&nb).unwrap_or(Ordering::Equal);
    }
    if let (Value::String(sa), Value::String(sb)) = (a, b) {
        return sa.cmp(sb);
    }
    Ordering::Equal
}

/// Python's `type(x).__name__` for a JSON value.
pub fn py_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn distinguishes_int_from_float() {
        let int_val: Value = parse_py_typed("1").unwrap();
        let float_val: Value = parse_py_typed("1.0").unwrap();
        assert_eq!(py_type_name(&int_val), "int");
        assert_eq!(py_type_name(&float_val), "float");
        assert!(py_equals(&int_val, &float_val));
        assert!(py_equals(&int_val, &json!(true)));
    }

    #[test]
    fn decodes_base85_indices() {
        assert_eq!(base85_to_int("00").unwrap(), 0);
        assert_eq!(base85_to_int("0R").unwrap(), 1);
        assert_eq!(base85_to_int("RR").unwrap(), 85);
        assert_eq!(base85_to_int("|Nj").unwrap(), 65535);
        assert_eq!(base85_to_int("00001").unwrap(), 1);
    }
}
