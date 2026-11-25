use std::collections::HashMap;
use std::io::{Cursor, Read};

use zip::ZipArchive;

use crate::exceptions::{FpgenError, Result};

fn malformed(why: impl std::fmt::Display) -> FpgenError {
    FpgenError::ModelVerificationError(format!("malformed zip archive: {why}"))
}

/// Reads all members from a zip archive buffer into an in-memory map, failing
/// on duplicate names or malformed entries (`fpgen-js/src/zip.ts`).
pub fn read_zip(buf: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    let mut archive = ZipArchive::new(Cursor::new(buf)).map_err(malformed)?;
    let mut members = HashMap::with_capacity(archive.len());
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(malformed)?;
        let name = entry.name().to_owned();
        if members.contains_key(&name) {
            return Err(malformed(format!("{name} appears twice")));
        }
        let mut data = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut data).map_err(malformed)?;
        members.insert(name, data);
    }
    Ok(members)
}
