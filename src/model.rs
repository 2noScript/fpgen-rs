use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use crate::bayesian_network::{BayesianNetwork, NetworkDefinition, ValueLookup};
use crate::exceptions::{FpgenError, Result};
use crate::pin::{ModelPin, MODEL_PIN};
use crate::pyjson::{base85_to_int, parse_ordered};
use crate::zip::read_zip;

const STAMP_FILE: &str = ".pinned-model";
const NETWORK_ZST: &str = "fingerprint-network.json.zst";
const VALUES_JSON_ZST: &str = "values.json.zst";
const VALUES_DAT_ZST: &str = "values.dat.zst";
const VALUES_DAT: &str = "values.dat";
pub const LOCK_DIR: &str = ".install.lock";
pub const STALE_LOCK_SECS: u64 = 10 * 60;

static CONFIGURED_DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static LOADED_MODELS: OnceLock<Mutex<HashMap<PathBuf, Arc<FpgenModel>>>> = OnceLock::new();

fn configured_dir_lock() -> &'static Mutex<Option<PathBuf>> {
    CONFIGURED_DIR.get_or_init(|| Mutex::new(None))
}

fn loaded_models_lock() -> &'static Mutex<HashMap<PathBuf, Arc<FpgenModel>>> {
    LOADED_MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn user_cache_dir(app_name: &str) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("AppData")
                    .join("Local")
            });
        return base.join(app_name).join(app_name).join("Cache");
    }
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Library")
            .join("Caches")
            .join(app_name)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let base = std::env::var("XDG_CACHE_HOME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".cache")
            });
        base.join(app_name)
    }
}

pub fn model_dir() -> PathBuf {
    if let Ok(guard) = configured_dir_lock().lock() {
        if let Some(dir) = guard.as_ref() {
            return dir.clone();
        }
    }
    if let Ok(env_dir) = std::env::var("FPGEN_DATA_DIR") {
        if !env_dir.trim().is_empty() {
            return PathBuf::from(env_dir);
        }
    }
    user_cache_dir("fpgen")
}

pub fn set_model_dir(dir: Option<PathBuf>) {
    if let Ok(mut guard) = configured_dir_lock().lock() {
        *guard = dir;
    }
}

pub fn is_model_installed(dir: &Path, pin: &ModelPin) -> bool {
    let Ok(stamp) = fs::read_to_string(dir.join(STAMP_FILE)) else {
        return false;
    };
    if stamp.trim() != pin.sha256 {
        return false;
    }
    pin.files.iter().all(|f| dir.join(f).exists())
}

#[derive(Debug, Clone, Default)]
pub struct EnsureModelOptions {
    pub dir: Option<PathBuf>,
    pub force: bool,
}

fn with_install_lock<T>(dir: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    fs::create_dir_all(dir)?;
    let lock = dir.join(LOCK_DIR);
    loop {
        match fs::create_dir(&lock) {
            Ok(()) => break,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if let Ok(meta) = fs::metadata(&lock) {
                    if let Ok(modified) = meta.modified() {
                        if SystemTime::now()
                            .duration_since(modified)
                            .unwrap_or_default()
                            > Duration::from_secs(STALE_LOCK_SECS)
                        {
                            let _ = fs::remove_dir_all(&lock);
                            continue;
                        }
                    }
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(err) => return Err(err.into()),
        }
    }
    let out = f();
    let _ = fs::remove_dir_all(&lock);
    out
}

pub fn ensure_model(options: EnsureModelOptions) -> Result<PathBuf> {
    let dir = options.dir.unwrap_or_else(model_dir);
    if !options.force && is_model_installed(&dir, &MODEL_PIN) && dat_is_ready(&dir) {
        return Ok(dir);
    }

    with_install_lock(&dir, || {
        if options.force || !is_model_installed(&dir, &MODEL_PIN) {
            let archive = download_archive(&MODEL_PIN)?;
            install_archive(&archive, &dir, &MODEL_PIN)?;
        }
        decompress_values_dat(&dir)?;
        Ok(())
    })?;

    Ok(dir)
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn download_archive(pin: &ModelPin) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("fpgen-rs")
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| FpgenError::ModelVerificationError(e.to_string()))?;

    let mut last_err = String::new();
    for _attempt in 0..3 {
        match client.get(pin.url).send() {
            Ok(resp) => {
                let status = resp.status();
                if !status.is_success() {
                    last_err = format!(
                        "fpgen model download failed: HTTP {} for {}",
                        status.as_u16(),
                        pin.url
                    );
                    if status.is_client_error() {
                        break;
                    }
                    continue;
                }
                let bytes = resp
                    .bytes()
                    .map_err(|e| FpgenError::ModelVerificationError(e.to_string()))?;
                verify_archive(&bytes, pin)?;
                return Ok(bytes.to_vec());
            }
            Err(err) => {
                last_err = err.to_string();
            }
        }
    }
    Err(FpgenError::ModelVerificationError(last_err))
}

pub fn verify_archive(buf: &[u8], pin: &ModelPin) -> Result<()> {
    if buf.len() != pin.size {
        return Err(FpgenError::ModelVerificationError(format!(
            "fpgen model size mismatch: got {}, expected {}",
            buf.len(),
            pin.size
        )));
    }
    let got = sha256_hex(buf);
    if got != pin.sha256 {
        return Err(FpgenError::ModelVerificationError(format!(
            "fpgen model sha256 mismatch:\n  got      {got}\n  expected {}",
            pin.sha256
        )));
    }
    Ok(())
}

pub fn install_archive(archive: &[u8], dir: &Path, pin: &ModelPin) -> Result<()> {
    verify_archive(archive, pin)?;
    let zip = read_zip(archive)?;
    let mut members = Vec::with_capacity(pin.files.len());
    for name in pin.files {
        let data = zip.get(name).ok_or_else(|| {
            FpgenError::ModelVerificationError(format!("fpgen model archive is missing {name}"))
        })?;
        let expected = pin.member_sha256(name).ok_or_else(|| {
            FpgenError::ModelVerificationError(format!("unexpected member in pin: {name}"))
        })?;
        if sha256_hex(data) != expected {
            return Err(FpgenError::ModelVerificationError(format!(
                "fpgen model member {name} does not match its pinned sha256"
            )));
        }
        members.push((name, data));
    }

    fs::create_dir_all(dir)?;
    let previous = fs::read_to_string(dir.join(STAMP_FILE))
        .map(|s| s.trim().to_owned())
        .unwrap_or_default();
    let _ = fs::remove_file(dir.join(STAMP_FILE));
    if previous != pin.sha256 {
        let _ = fs::remove_file(dir.join(VALUES_DAT));
    }
    for (name, data) in members {
        atomic_write(&dir.join(name), data)?;
    }
    fs::write(dir.join(STAMP_FILE), format!("{}\n", pin.sha256))?;
    Ok(())
}

fn atomic_write(target: &Path, data: &[u8]) -> Result<()> {
    let tmp = target.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&tmp, data)?;
    fs::rename(&tmp, target)?;
    Ok(())
}

fn expected_dat_size(pairs: &[(u64, usize)]) -> u64 {
    pairs
        .iter()
        .map(|&(offset, len)| offset + (len as u64))
        .max()
        .unwrap_or(0)
}

fn dat_is_complete(dat_path: &Path, pairs: &[(u64, usize)]) -> bool {
    fs::metadata(dat_path).is_ok_and(|meta| meta.len() == expected_dat_size(pairs))
}

fn dat_is_ready(dir: &Path) -> bool {
    read_value_pairs(dir)
        .map(|pairs| dat_is_complete(&dir.join(VALUES_DAT), &pairs))
        .unwrap_or(false)
}

fn decompress_values_dat(dir: &Path) -> Result<()> {
    let dat_path = dir.join(VALUES_DAT);
    let pairs = read_value_pairs(dir)?;
    if dat_is_complete(&dat_path, &pairs) {
        return Ok(());
    }
    let tmp = dir.join(format!("{VALUES_DAT}.{}.tmp", std::process::id()));
    {
        let src = BufReader::new(File::open(dir.join(VALUES_DAT_ZST))?);
        let mut decoder = zstd::stream::read::Decoder::new(src)?;
        let mut dst = BufWriter::new(File::create(&tmp)?);
        std::io::copy(&mut decoder, &mut dst)?;
        dst.flush()?;
    }
    fs::rename(&tmp, &dat_path)?;
    if !dat_is_complete(&dat_path, &pairs) {
        return Err(FpgenError::ModelVerificationError(format!(
            "decompressed {} does not match values.json",
            dat_path.display()
        )));
    }
    Ok(())
}

fn read_zst_text(file: &Path) -> Result<String> {
    let f = BufReader::new(File::open(file)?);
    let mut decoder = zstd::stream::read::Decoder::new(f)?;
    let mut text = String::new();
    decoder.read_to_string(&mut text)?;
    Ok(text)
}

fn read_value_pairs(dir: &Path) -> Result<Vec<(u64, usize)>> {
    let text = read_zst_text(&dir.join(VALUES_JSON_ZST))?;
    let json = parse_ordered(&text)?;
    let obj = json.as_object().ok_or_else(|| {
        FpgenError::ModelVerificationError("values.json is not an object".into())
    })?;
    let mut pairs = Vec::with_capacity(obj.len());
    for (hex_offset, len_val) in obj {
        let offset = u64::from_str_radix(hex_offset, 16)
            .map_err(|e| FpgenError::ModelVerificationError(e.to_string()))?;
        let length = len_val.as_u64().ok_or_else(|| {
            FpgenError::ModelVerificationError("values.json length is not an integer".into())
        })? as usize;
        pairs.push((offset, length));
    }
    Ok(pairs)
}

struct ValueStore {
    value_pairs: Vec<(u64, usize)>,
    dat_file: Mutex<File>,
}

impl ValueLookup for ValueStore {
    fn lookup_value_list(&self, ids: &[String]) -> Result<Vec<String>> {
        let mut sorted: Vec<(usize, usize)> = ids
            .iter()
            .enumerate()
            .map(|(pos, id)| Ok((base85_to_int(id)?, pos)))
            .collect::<Result<Vec<_>>>()?;
        sorted.sort_unstable_by_key(|&(idx, pos)| (idx, pos));

        let mut out = vec![String::new(); ids.len()];
        let mut file = self
            .dat_file
            .lock()
            .map_err(|_| FpgenError::ValueError("value store lock poisoned".into()))?;

        for (idx, pos) in sorted {
            let &(offset, length) = self.value_pairs.get(idx).ok_or_else(|| {
                FpgenError::ValueError(format!("list index out of range: value id {idx}"))
            })?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; length];
            file.read_exact(&mut buf)?;
            out[pos] = String::from_utf8(buf)
                .map_err(|e| FpgenError::ValueError(format!("invalid UTF-8 in values.dat: {e}")))?;
        }
        Ok(out)
    }
}

pub struct FpgenModel {
    pub dir: PathBuf,
    pub network: BayesianNetwork,
    store: Arc<ValueStore>,
}

impl FpgenModel {
    pub fn load(dir: &Path) -> Result<Self> {
        let value_pairs = read_value_pairs(dir)?;
        decompress_values_dat(dir)?;
        let dat_file = Mutex::new(File::open(dir.join(VALUES_DAT))?);
        let store = Arc::new(ValueStore {
            value_pairs,
            dat_file,
        });

        let network_text = read_zst_text(&dir.join(NETWORK_ZST))?;
        let def: NetworkDefinition = serde_json::from_str(&network_text)
            .map_err(|e| FpgenError::ModelVerificationError(e.to_string()))?;
        let network = BayesianNetwork::new(def, Arc::clone(&store) as Arc<dyn ValueLookup>)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            network,
            store,
        })
    }

    pub fn lookup_value(&self, id: &str) -> Result<String> {
        let mut list = self.lookup_value_list(&[id.to_owned()])?;
        Ok(list.pop().unwrap_or_default())
    }

    pub fn lookup_value_list(&self, ids: &[String]) -> Result<Vec<String>> {
        self.store.lookup_value_list(ids)
    }
}

pub fn get_model(dir: Option<&Path>) -> Result<Arc<FpgenModel>> {
    let target = dir.map(Path::to_path_buf).unwrap_or_else(model_dir);
    let key = fs::canonicalize(&target).unwrap_or(target);

    {
        let cache = loaded_models_lock()
            .lock()
            .map_err(|_| FpgenError::ValueError("model cache lock poisoned".into()))?;
        if let Some(model) = cache.get(&key) {
            return Ok(Arc::clone(model));
        }
    }

    if !is_model_installed(&key, &MODEL_PIN) {
        return Err(FpgenError::ModelNotInstalled(format!(
            "The fpgen model ({}) is not installed in {}. Call `ensure_model(...)` before generating fingerprints.",
            MODEL_PIN.tag,
            key.display()
        )));
    }

    let model = Arc::new(FpgenModel::load(&key)?);
    let mut cache = loaded_models_lock()
        .lock()
        .map_err(|_| FpgenError::ValueError("model cache lock poisoned".into()))?;
    Ok(Arc::clone(cache.entry(key).or_insert(model)))
}
