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
    env, fs,
    path::{Path, PathBuf},
};

use directories::BaseDirs;

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

/// Whether a repository looks like it still has to be fetched.
///
/// This only decides whether the indicator is visible for the second or so
/// before the real size arrives from Hugging Face, so the comparison against
/// the configured estimate is deliberately loose: it separates "nothing yet"
/// from "already downloaded" rather than measuring anything.
pub fn looks_incomplete(repo: &str, estimated_gib: f64) -> bool {
    let bytes = scan(repo).total_bytes();
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
}
