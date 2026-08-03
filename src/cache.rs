//! Inspection of the on-disk caches llama.cpp uses for `-hf` models.
//!
//! This is the only live view of a download tinyinference has: llama-server
//! writes its progress bar to a terminal, and tinyinference reads its output
//! through a pipe, so a download is otherwise completely silent.
//!
//! Two layouts are read. The current one is the Hugging Face hub cache, where
//! `models--owner--repo/blobs/<oid>` holds the payload and a file still being
//! fetched carries a `.downloadInProgress` suffix. The older one is a flat
//! `<cache>/owner_repo_file.gguf` per download, which has no object ids.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread,
};

use directories::BaseDirs;

use crate::config::ModelSource;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const PARTIAL_SUFFIXES: [&str; 2] = [".downloadInProgress", ".incomplete"];

/// A payload file in the hub cache, complete or not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedBlob {
    /// LFS object id, which the repository listing also reports.
    pub oid: String,
    pub bytes: u64,
    pub in_flight: bool,
}

/// What a repository currently occupies on this machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheScan {
    pub blobs: Vec<CachedBlob>,
    /// Bytes held in the legacy flat layout, which carries no object ids.
    pub flat_bytes: u64,
}

#[derive(Debug)]
pub struct PendingScan {
    receiver: Receiver<CacheScan>,
}

impl PendingScan {
    pub fn take(&self) -> Option<CacheScan> {
        self.receiver.try_recv().ok()
    }
}

impl CacheScan {
    pub fn total_bytes(&self) -> u64 {
        self.blobs
            .iter()
            .map(|blob| blob.bytes)
            .fold(self.flat_bytes, u64::saturating_add)
    }

    /// Bytes belonging to a known set of object ids, ignoring blobs from other
    /// quantizations of the same repository.
    pub fn bytes_of<S: AsRef<str>>(&self, oids: &[S]) -> u64 {
        self.blobs
            .iter()
            .filter(|blob| oids.iter().any(|oid| oid.as_ref() == blob.oid))
            .map(|blob| blob.bytes)
            .fold(0, u64::saturating_add)
    }

    pub fn in_flight(&self) -> impl Iterator<Item = &CachedBlob> {
        self.blobs.iter().filter(|blob| blob.in_flight)
    }
}

/// Look at every cache directory this machine might use for `repo`.
pub fn scan(repo: &str) -> CacheScan {
    let Some((owner, name)) = split_repo(repo) else {
        return CacheScan::default();
    };
    let hub_directory = format!("models--{owner}--{name}");
    let flat_prefix = format!("{owner}_{name}_");
    let mut scan = CacheScan::default();
    for root in hub_roots() {
        scan.blobs
            .extend(read_blobs(&root.join(&hub_directory).join("blobs")));
    }
    for root in flat_roots() {
        scan.flat_bytes = scan
            .flat_bytes
            .saturating_add(prefixed_file_bytes(&root, &flat_prefix));
    }
    scan
}

pub fn scan_async(repo: &str) -> PendingScan {
    let (sender, receiver) = mpsc::channel();
    let repo = repo.to_string();
    thread::spawn(move || {
        let _ = sender.send(scan(&repo));
    });
    PendingScan { receiver }
}

/// A GGUF model found in the local Hugging Face hub or llama.cpp cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub source: DiscoveredSource,
    /// Largest complete GGUF (or blob) seen for this entry, for sorting.
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveredSource {
    /// `owner/repo` from a `models--owner--repo` hub directory.
    HuggingFace(String),
    /// Absolute path to a `.gguf` in the flat llama.cpp cache.
    Local(PathBuf),
}

#[derive(Debug)]
pub struct PendingDiscover {
    receiver: Receiver<Vec<DiscoveredModel>>,
}

impl PendingDiscover {
    pub fn take(&self) -> Option<Vec<DiscoveredModel>> {
        self.receiver.try_recv().ok()
    }
}

/// List GGUF models already present in the on-disk caches llama-server uses.
pub fn discover_models() -> Vec<DiscoveredModel> {
    let mut hub: BTreeMap<String, u64> = BTreeMap::new();
    for root in hub_roots() {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some((owner, repo)) = parse_hub_dir(name) else {
                continue;
            };
            let Some(bytes) = hub_model_gguf_bytes(&path, &repo) else {
                continue;
            };
            let id = format!("{owner}/{repo}");
            hub.entry(id)
                .and_modify(|existing| *existing = (*existing).max(bytes))
                .or_insert(bytes);
        }
    }

    let mut models: Vec<DiscoveredModel> = hub
        .into_iter()
        .map(|(repo, bytes)| DiscoveredModel {
            source: DiscoveredSource::HuggingFace(repo),
            bytes,
        })
        .collect();

    for root in flat_roots() {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if PARTIAL_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                continue;
            }
            if !name.to_ascii_lowercase().ends_with(".gguf") {
                continue;
            }
            models.push(DiscoveredModel {
                source: DiscoveredSource::Local(path),
                bytes: metadata.len(),
            });
        }
    }

    models.sort_by(|left, right| {
        right
            .bytes
            .cmp(&left.bytes)
            .then_with(|| left.label().cmp(&right.label()))
    });
    models
}

pub fn discover_models_async() -> PendingDiscover {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(discover_models());
    });
    PendingDiscover { receiver }
}

impl DiscoveredModel {
    pub fn label(&self) -> String {
        match &self.source {
            DiscoveredSource::HuggingFace(repo) => repo.clone(),
            DiscoveredSource::Local(path) => path.display().to_string(),
        }
    }
}

/// What a local delete removed from disk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeleteReport {
    pub removed_paths: Vec<PathBuf>,
    pub freed_bytes: u64,
}

/// Whether this model still has files on disk that tinyinference can remove.
///
/// For Hugging Face sources this means a hub cache directory and/or flat
/// llama.cpp cache files. For local sources it means the GGUF path exists.
pub fn has_local_files(source: &ModelSource) -> bool {
    match source {
        ModelSource::HuggingFace(repo) => huggingface_cache_paths(repo)
            .into_iter()
            .any(|path| path.exists()),
        ModelSource::Local(path) => path.is_file(),
    }
}

/// Approximate bytes occupied locally by this model source.
pub fn local_bytes(source: &ModelSource) -> u64 {
    match source {
        ModelSource::HuggingFace(repo) => huggingface_cache_paths(repo)
            .into_iter()
            .map(|path| path_bytes(&path))
            .fold(0, u64::saturating_add),
        ModelSource::Local(path) => {
            let paths = crate::config::split_gguf_paths(path).unwrap_or_else(|| vec![path.clone()]);
            paths
                .iter()
                .map(|path| path_bytes(path))
                .fold(0, u64::saturating_add)
        }
    }
}

/// Remove a model's local cache files.
///
/// Hugging Face repos are deleted the same way as `hf cache rm model/<repo>`:
/// the whole `models--owner--repo` hub directory (and matching llama.cpp flat
/// cache files / lock dirs). Local GGUF paths delete the file itself.
pub fn delete_local_files(source: &ModelSource) -> Result<DeleteReport, String> {
    match source {
        ModelSource::HuggingFace(repo) => delete_huggingface_repo(repo),
        ModelSource::Local(path) => {
            let paths = crate::config::split_gguf_paths(path).unwrap_or_else(|| vec![path.clone()]);
            delete_files(&paths)
        }
    }
}

fn delete_huggingface_repo(repo: &str) -> Result<DeleteReport, String> {
    let paths = huggingface_cache_paths(repo);
    if paths.is_empty() {
        return Err("Not a valid Hugging Face repository id.".into());
    }
    let existing: Vec<PathBuf> = paths.into_iter().filter(|path| path.exists()).collect();
    if existing.is_empty() {
        return Err("No local cache found for this model.".into());
    }

    let mut report = DeleteReport::default();
    for path in existing {
        let bytes = path_bytes(&path);
        remove_path(&path)?;
        report.freed_bytes = report.freed_bytes.saturating_add(bytes);
        report.removed_paths.push(path);
    }
    Ok(report)
}

fn delete_files(paths: &[PathBuf]) -> Result<DeleteReport, String> {
    let mut targets = Vec::new();
    for path in paths {
        if path.is_file() {
            targets.push(path.clone());
        }
        for suffix in PARTIAL_SUFFIXES {
            let partial = PathBuf::from(format!("{}{suffix}", path.display()));
            if partial.is_file() {
                targets.push(partial);
            }
        }
    }
    targets.sort();
    targets.dedup();
    if targets.is_empty() {
        return Err("No local model file found to delete.".into());
    }

    let mut report = DeleteReport::default();
    for path in targets {
        let bytes = path_bytes(&path);
        remove_path(&path)?;
        report.freed_bytes = report.freed_bytes.saturating_add(bytes);
        report.removed_paths.push(path);
    }
    Ok(report)
}

fn huggingface_cache_paths(repo: &str) -> Vec<PathBuf> {
    huggingface_cache_paths_in(repo, &hub_roots(), &flat_roots())
}

fn huggingface_cache_paths_in(repo: &str, hubs: &[PathBuf], flats: &[PathBuf]) -> Vec<PathBuf> {
    let Some((owner, name)) = split_repo(repo) else {
        return Vec::new();
    };
    let hub_directory = format!("models--{owner}--{name}");
    let flat_prefix = format!("{owner}_{name}_");
    let mut paths = Vec::new();
    for root in hubs {
        let model_dir = root.join(&hub_directory);
        if model_dir.exists() {
            paths.push(model_dir);
        }
        let lock_dir = root.join(".locks").join(&hub_directory);
        if lock_dir.exists() {
            paths.push(lock_dir);
        }
    }
    for root in flats {
        let Ok(entries) = fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.starts_with(&flat_prefix) {
                paths.push(path);
            }
        }
    }
    paths
}

fn remove_path(path: &Path) -> Result<(), String> {
    let result = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    result.map_err(|error| format!("could not delete {}: {error}", path.display()))
}

fn path_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    }
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(child);
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

/// `models--owner--repo` → `(owner, repo)`.
fn parse_hub_dir(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix("models--")?;
    let (owner, repo) = rest.split_once("--")?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some((owner.to_string(), repo.to_string()))
}

/// Bytes of the largest complete GGUF for a hub model directory, if any.
fn hub_model_gguf_bytes(model_dir: &Path, repo_name: &str) -> Option<u64> {
    if let Some(bytes) = preferred_snapshot_gguf_bytes(model_dir) {
        return Some(bytes);
    }
    // Snapshots are missing while a fetch is still landing. For GGUF-named
    // repos, a finished blob is enough to treat the model as available.
    if !repo_name.to_ascii_lowercase().contains("gguf") {
        return None;
    }
    read_blobs(&model_dir.join("blobs"))
        .into_iter()
        .filter(|blob| !blob.in_flight)
        .map(|blob| blob.bytes)
        .max()
        .filter(|bytes| *bytes > 0)
}

fn preferred_snapshot_gguf_bytes(model_dir: &Path) -> Option<u64> {
    let snapshots = model_dir.join("snapshots");
    if let Ok(commit) = fs::read_to_string(model_dir.join("refs").join("main")) {
        let commit = commit.trim();
        if !commit.is_empty() {
            let bytes = largest_gguf_bytes(&snapshots.join(commit));
            if bytes > 0 {
                return Some(bytes);
            }
        }
    }
    let Ok(entries) = fs::read_dir(&snapshots) else {
        return None;
    };
    entries
        .flatten()
        .map(|entry| largest_gguf_bytes(&entry.path()))
        .filter(|bytes| *bytes > 0)
        .max()
}

fn largest_gguf_bytes(directory: &Path) -> u64 {
    let mut largest = 0u64;
    let mut stack = vec![directory.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            if !metadata.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.to_ascii_lowercase().ends_with(".gguf") {
                largest = largest.max(metadata.len());
            }
        }
    }
    largest
}

/// Whether a repository looks like it still has to be fetched.
///
/// This only decides whether the indicator is visible for the second or so
/// before the real size arrives from Hugging Face, so the comparison against
/// the configured estimate is deliberately loose: it separates "nothing yet"
/// from "already downloaded" rather than measuring anything.
pub fn looks_incomplete(repo: &str, estimated_gib: f64) -> bool {
    looks_incomplete_scan(&scan(repo), estimated_gib)
}

pub fn looks_incomplete_scan(scan: &CacheScan, estimated_gib: f64) -> bool {
    let bytes = scan.total_bytes();
    if bytes == 0 {
        return true;
    }
    if !estimated_gib.is_finite() || estimated_gib <= 0.0 {
        return false;
    }
    (bytes as f64 / GIB) < estimated_gib * 0.95
}

/// True when a llama-server log line shows the weights are being read.
///
/// Current llama.cpp is silent while downloading, so this marks the end of a
/// download; older builds that do log a download are covered by the same line.
pub fn is_model_load_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("llama_model_loader:")
        || lower.contains("load_tensors:")
        || lower.contains("load_model:")
        || lower.contains("loading model")
        || lower.contains("model loaded")
}

fn split_repo(repo: &str) -> Option<(String, String)> {
    // Repositories may carry a `:quant` tag, which is not part of the path.
    let without_tag = repo.trim().split(':').next()?.trim_end_matches('/');
    let (owner, name) = without_tag.split_once('/')?;
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    Some((owner.to_string(), name.to_string()))
}

fn read_blobs(directory: &Path) -> Vec<CachedBlob> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            if !metadata.is_file() {
                return None;
            }
            let name = entry.file_name().into_string().ok()?;
            let (oid, in_flight) = match PARTIAL_SUFFIXES
                .iter()
                .find_map(|suffix| name.strip_suffix(suffix))
            {
                Some(oid) => (oid.to_string(), true),
                None => (name, false),
            };
            Some(CachedBlob {
                oid,
                bytes: metadata.len(),
                in_flight,
            })
        })
        .collect()
}

/// Preferred Hugging Face hub cache root for new downloads.
pub fn default_hub_root() -> PathBuf {
    hub_roots().into_iter().next().unwrap_or_else(|| {
        BaseDirs::new()
            .map(|dirs| {
                dirs.home_dir()
                    .join(".cache")
                    .join("huggingface")
                    .join("hub")
            })
            .unwrap_or_else(|| PathBuf::from(".cache").join("huggingface").join("hub"))
    })
}

fn hub_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    push_root(&mut roots, env_path("HF_HUB_CACHE"));
    push_root(&mut roots, env_path("HUGGINGFACE_HUB_CACHE"));
    push_root(&mut roots, env_path("HF_HOME").map(|path| path.join("hub")));
    push_root(
        &mut roots,
        env_path("XDG_CACHE_HOME").map(|path| path.join("huggingface").join("hub")),
    );
    if let Some(dirs) = BaseDirs::new() {
        push_root(
            &mut roots,
            Some(
                dirs.home_dir()
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ),
        );
    }
    roots
}

fn flat_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    push_root(&mut roots, env_path("LLAMA_CACHE"));
    if let Some(dirs) = BaseDirs::new() {
        push_root(&mut roots, Some(dirs.cache_dir().join("llama.cpp")));
        push_root(&mut roots, Some(dirs.cache_dir().join("llama-cpp")));
    }
    roots
}

fn push_root(roots: &mut Vec<PathBuf>, candidate: Option<PathBuf>) {
    if let Some(path) = candidate
        && !path.as_os_str().is_empty()
        && !roots.contains(&path)
    {
        roots.push(path);
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn prefixed_file_bytes(root: &Path, prefix: &str) -> u64 {
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(prefix))
        })
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs_in(directory: &Path) -> Vec<CachedBlob> {
        let mut blobs = read_blobs(directory);
        blobs.sort_by(|left, right| left.oid.cmp(&right.oid));
        blobs
    }

    #[test]
    fn repository_tags_are_not_part_of_the_cache_path() {
        assert_eq!(
            split_repo("ggml-org/gpt-oss-120b-GGUF:Q4_K_M"),
            Some(("ggml-org".into(), "gpt-oss-120b-GGUF".into()))
        );
        assert_eq!(split_repo("no-slash"), None);
        assert_eq!(split_repo("owner/"), None);
    }

    #[test]
    fn a_file_being_fetched_is_reported_under_its_object_id() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("da2572.downloadInProgress"), vec![0; 512]).unwrap();
        fs::write(dir.path().join("beef01"), vec![0; 128]).unwrap();
        assert_eq!(
            blobs_in(dir.path()),
            [
                CachedBlob {
                    oid: "beef01".into(),
                    bytes: 128,
                    in_flight: false,
                },
                CachedBlob {
                    oid: "da2572".into(),
                    bytes: 512,
                    in_flight: true,
                },
            ]
        );
    }

    #[test]
    fn only_blobs_of_the_wanted_files_are_counted() {
        let scan = CacheScan {
            blobs: vec![
                CachedBlob {
                    oid: "wanted".into(),
                    bytes: 300,
                    in_flight: true,
                },
                CachedBlob {
                    oid: "other-quant".into(),
                    bytes: 900,
                    in_flight: false,
                },
            ],
            flat_bytes: 7,
        };
        assert_eq!(scan.bytes_of(&["wanted"]), 300);
        assert_eq!(scan.bytes_of(&["wanted", "other-quant"]), 1200);
        assert_eq!(scan.total_bytes(), 1207);
        assert_eq!(scan.in_flight().count(), 1);
    }

    #[test]
    fn flat_cache_files_are_matched_by_repository_prefix() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("owner_model_weights.gguf"), vec![0; 100]).unwrap();
        fs::write(
            dir.path()
                .join("owner_model_weights.gguf.downloadInProgress"),
            vec![0; 20],
        )
        .unwrap();
        fs::write(dir.path().join("other_model_weights.gguf"), vec![0; 999]).unwrap();
        assert_eq!(prefixed_file_bytes(dir.path(), "owner_model_"), 120);
    }

    #[test]
    fn missing_caches_report_nothing() {
        assert!(blobs_in(Path::new("/tinyinference/does/not/exist")).is_empty());
        assert_eq!(scan("not-a-repository"), CacheScan::default());
        assert!(looks_incomplete("tinyinference/never-downloaded", 59.1));
    }

    #[test]
    fn the_end_of_a_download_is_recognised_in_the_log() {
        assert!(is_model_load_line("llama_model_loader: loaded meta data"));
        assert!(is_model_load_line(
            "srv    load_model: initializing, n_slots = 4"
        ));
        assert!(is_model_load_line("main: llama_server: model loaded"));
        assert!(!is_model_load_line(
            "srv  llama_server: listening on http://x"
        ));
    }

    #[test]
    fn hub_directory_names_decode_to_owner_and_repo() {
        assert_eq!(
            parse_hub_dir("models--ggml-org--gpt-oss-120b-GGUF"),
            Some(("ggml-org".into(), "gpt-oss-120b-GGUF".into()))
        );
        assert_eq!(parse_hub_dir("models--only-owner"), None);
        assert_eq!(parse_hub_dir("datasets--owner--name"), None);
    }

    #[test]
    fn snapshot_gguf_files_mark_a_hub_model_as_present() {
        let root = tempfile::tempdir().unwrap();
        let model = root
            .path()
            .join("models--owner--tiny-GGUF");
        let snapshot = model.join("snapshots").join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(model.join("refs")).unwrap();
        fs::write(model.join("refs").join("main"), "abc123\n").unwrap();
        fs::write(snapshot.join("weights.gguf"), vec![0; 2048]).unwrap();
        fs::write(snapshot.join("readme.md"), b"nope").unwrap();
        assert_eq!(hub_model_gguf_bytes(&model, "tiny-GGUF"), Some(2048));
    }

    #[test]
    fn non_gguf_hub_repos_are_ignored_without_gguf_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let model = root
            .path()
            .join("models--owner--embeddings");
        let snapshot = model.join("snapshots").join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.safetensors"), vec![0; 512]).unwrap();
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::write(model.join("blobs").join("oid1"), vec![0; 512]).unwrap();
        assert_eq!(hub_model_gguf_bytes(&model, "embeddings"), None);
    }

    #[test]
    fn finished_blobs_count_for_gguf_named_repos_without_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("models--owner--weights-GGUF");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::write(model.join("blobs").join("oid1"), vec![0; 4096]).unwrap();
        fs::write(
            model.join("blobs").join("oid2.downloadInProgress"),
            vec![0; 100],
        )
        .unwrap();
        assert_eq!(hub_model_gguf_bytes(&model, "weights-GGUF"), Some(4096));
    }

    #[test]
    fn hub_cache_paths_include_model_and_lock_directories() {
        let hub = tempfile::tempdir().unwrap();
        let flat = tempfile::tempdir().unwrap();
        let model = hub.path().join("models--owner--tiny-GGUF");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::write(model.join("blobs").join("oid1"), vec![0; 128]).unwrap();
        let locks = hub.path().join(".locks").join("models--owner--tiny-GGUF");
        fs::create_dir_all(&locks).unwrap();
        let flat_file = flat.path().join("owner_tiny-GGUF_weights.gguf");
        fs::write(&flat_file, vec![0; 32]).unwrap();

        let paths = huggingface_cache_paths_in(
            "owner/tiny-GGUF",
            &[hub.path().to_path_buf()],
            &[flat.path().to_path_buf()],
        );
        assert!(paths.contains(&model));
        assert!(paths.contains(&locks));
        assert!(paths.contains(&flat_file));
    }

    #[test]
    fn deleting_hub_paths_removes_model_cache_directories() {
        let hub = tempfile::tempdir().unwrap();
        let model = hub.path().join("models--owner--tiny-GGUF");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::write(model.join("blobs").join("oid1"), vec![0; 128]).unwrap();
        let locks = hub.path().join(".locks").join("models--owner--tiny-GGUF");
        fs::create_dir_all(&locks).unwrap();

        let mut report = DeleteReport::default();
        for path in huggingface_cache_paths_in(
            "owner/tiny-GGUF",
            &[hub.path().to_path_buf()],
            &[],
        ) {
            let bytes = path_bytes(&path);
            remove_path(&path).unwrap();
            report.freed_bytes = report.freed_bytes.saturating_add(bytes);
            report.removed_paths.push(path);
        }
        assert!(!model.exists());
        assert!(!locks.exists());
        assert!(report.freed_bytes >= 128);
    }

    #[test]
    fn deleting_a_local_gguf_removes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.gguf");
        fs::write(&path, vec![0; 64]).unwrap();
        let source = ModelSource::Local(path.clone());
        assert!(has_local_files(&source));
        let report = delete_local_files(&source).unwrap();
        assert!(!path.exists());
        assert_eq!(report.freed_bytes, 64);
        assert!(!has_local_files(&source));
    }
}
