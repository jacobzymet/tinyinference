//! The Hugging Face file listing for a model repository.
//!
//! llama-server prints nothing while it downloads (its progress bar is written
//! only to a terminal, and tinyinference reads it through a pipe), so the real
//! size of a download has to come from the repository listing. Cached blobs are
//! named after their LFS object id, which is exactly the id this listing
//! carries, so the two can be matched without guessing which quantization
//! llama.cpp chose.

use std::{
    fmt::Write as _,
    sync::mpsc::{Receiver, TryRecvError},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

/// One file in a repository, as tinyinference needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub path: String,
    /// LFS object id, which is also the file name of the cached blob.
    pub oid: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    oid: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsEntry>,
}

#[derive(Debug, Deserialize)]
struct LfsEntry {
    #[serde(default)]
    oid: String,
    #[serde(default)]
    size: u64,
}

/// List the files of `repo`, ignoring any `:quant` tag on it.
pub fn list_files(repo: &str) -> Result<Vec<RemoteFile>> {
    let (owner, name) = repository_parts(repo)?;
    let path = format!("{owner}/{name}");
    let url = format!(
        "https://huggingface.co/api/models/{}/{}/tree/main?recursive=true",
        encode_segment(&owner),
        encode_segment(&name)
    );
    let body = get(&url).with_context(|| format!("could not list files for {path}"))?;
    let files = parse_tree(&body)?;
    if files.is_empty() {
        bail!("Hugging Face returned no files for {path}");
    }
    Ok(files)
}

/// A listing being fetched off the interface thread.
#[derive(Debug)]
pub struct PendingListing {
    receiver: Receiver<Result<Vec<RemoteFile>, String>>,
}

impl PendingListing {
    /// The result, once it arrives. `None` while the request is still running.
    pub fn take(&mut self) -> Option<Result<Vec<RemoteFile>, String>> {
        match self.receiver.try_recv() {
            Ok(files) => Some(files),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("listing worker stopped".into())),
        }
    }
}

/// Start listing `repo` in the background.
pub fn list_files_async(repo: &str) -> PendingListing {
    let (sender, receiver) = std::sync::mpsc::channel();
    let repo = repo.to_string();
    thread::spawn(move || {
        let result = list_files(&repo).map_err(|error| format!("{error:#}"));
        let _ = sender.send(result);
    });
    PendingListing { receiver }
}

fn get(url: &str) -> Result<String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut response = agent.get(url).call().context("request failed")?;
    if response.status() != 200 {
        bail!("Hugging Face responded with {}", response.status());
    }
    response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_string()
        .context("could not read the response")
}

fn parse_tree(body: &str) -> Result<Vec<RemoteFile>> {
    let files = serde_json::from_str::<Vec<TreeEntry>>(body)
        .context("Hugging Face returned invalid JSON")?
        .into_iter()
        .filter(|entry| entry.r#type == "file")
        .map(|entry| {
            // Large files are stored through LFS, where the blob id and the
            // real size live under `lfs`; small files carry them inline.
            let (oid, size) = match entry.lfs {
                Some(lfs) if !lfs.oid.is_empty() => (lfs.oid, lfs.size),
                _ => (entry.oid, entry.size),
            };
            RemoteFile {
                path: entry.path,
                oid,
                size,
            }
        })
        .filter(|file| !file.oid.is_empty())
        .collect();
    Ok(files)
}

fn repository_parts(repo: &str) -> Result<(String, String)> {
    let path = repo.trim().split(':').next().unwrap_or_default().trim();
    let Some((owner, name)) = path.split_once('/') else {
        bail!("{repo} is not an owner/model repository");
    };
    if owner.is_empty()
        || name.is_empty()
        || name.contains('/')
        || [owner, name].iter().any(|part| {
            part.chars().any(|character| {
                character.is_control()
                    || character.is_whitespace()
                    || matches!(character, '?' | '#' | '\\')
            })
        })
    {
        bail!("{repo} is not a valid owner/model repository");
    }
    Ok((owner.to_string(), name.to_string()))
}

fn encode_segment(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

/// Every file llama.cpp fetches when it is asked for `path`.
///
/// A split GGUF is only usable once all of its shards have arrived, so
/// `model-00001-of-00003.gguf` stands for all three.
pub fn family<'a>(files: &'a [RemoteFile], path: &str) -> Vec<&'a RemoteFile> {
    let family = shard_family(path);
    files
        .iter()
        .filter(|file| match &family {
            Some(family) => shard_family(&file.path).as_ref() == Some(family),
            None => file.path == path,
        })
        .collect()
}

/// Normalize pasted Hugging Face URLs / ids into `owner/model` or `owner/model:quant`.
pub fn normalize_repo_id(input: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("repository cannot be empty");
    }
    let without_scheme = trimmed
        .strip_prefix("hf://")
        .unwrap_or(trimmed)
        .trim()
        .trim_end_matches('/');
    let path = without_scheme
        .strip_prefix("https://huggingface.co/")
        .or_else(|| without_scheme.strip_prefix("http://huggingface.co/"))
        .unwrap_or(without_scheme);
    let path = path.split(['?', '#']).next().unwrap_or(path);
    let path = path.strip_prefix("models/").unwrap_or(path);
    let path = path.trim().trim_end_matches('/');
    let (owner, name) = repository_parts(path)?;
    let quant = path
        .split_once(':')
        .map(|(_, tag)| tag.trim())
        .filter(|tag| !tag.is_empty());
    Ok(match quant {
        Some(tag) => format!("{owner}/{name}:{tag}"),
        None => format!("{owner}/{name}"),
    })
}

/// Bytes of the GGUF artifact a repository is expected to map.
///
/// Split shards are counted together. When `repo` includes a `:quant` tag,
/// families whose path contains that tag are preferred; otherwise the largest
/// GGUF family in the listing is used (multi-quant repos ship several).
pub fn primary_gguf_bytes(files: &[RemoteFile], repo: &str) -> Option<u64> {
    let selected = primary_gguf_files(files, repo);
    (!selected.is_empty()).then(|| selected.iter().map(|file| file.size).sum())
}

/// The GGUF file(s) llama.cpp / tinyinference should download for `repo`.
pub fn primary_gguf_files<'a>(files: &'a [RemoteFile], repo: &str) -> Vec<&'a RemoteFile> {
    let quant = repo
        .trim()
        .split_once(':')
        .map(|(_, tag)| tag.trim())
        .filter(|tag| !tag.is_empty())
        .map(|tag| tag.to_ascii_lowercase());

    let mut families: Vec<(String, u64, bool, Vec<&'a RemoteFile>)> = Vec::new();
    for file in files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".gguf"))
    {
        let key = match shard_family(&file.path) {
            Some((prefix, total)) => format!("{prefix}-of-{total}"),
            None => file.path.clone(),
        };
        let matches_quant = quant
            .as_ref()
            .is_some_and(|tag| file.path.to_ascii_lowercase().contains(tag));
        if let Some(entry) = families.iter_mut().find(|(name, _, _, _)| name == &key) {
            entry.1 = entry.1.saturating_add(file.size);
            entry.2 |= matches_quant;
            entry.3.push(file);
        } else {
            families.push((key, file.size, matches_quant, vec![file]));
        }
    }

    if quant.is_some()
        && let Some((_, _, _, files)) = families
            .iter()
            .filter(|(_, _, matches, _)| *matches)
            .max_by_key(|(_, bytes, _, _)| *bytes)
    {
        return files.clone();
    }

    families
        .into_iter()
        .max_by_key(|(_, bytes, _, _)| *bytes)
        .map(|(_, _, _, files)| files)
        .unwrap_or_default()
}

/// The `(prefix, total)` shared by every shard of a split GGUF file.
fn shard_family(path: &str) -> Option<(String, usize)> {
    let stem = path.strip_suffix(".gguf")?;
    let (indexed, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = indexed.rsplit_once('-')?;
    let index = index.parse::<usize>().ok()?;
    let total = total.parse::<usize>().ok()?;
    (index >= 1 && index <= total && total > 1).then(|| (prefix.to_string(), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TREE: &str = r#"[
        {"type":"file","oid":"aaa","size":1803,"path":".gitattributes"},
        {"type":"file","oid":"ptr1","size":135,"lfs":{"oid":"da2572","size":428970080},"path":"Qwen3-0.6B-Q4_0.gguf"},
        {"type":"directory","oid":"ccc","size":0,"path":"nested"},
        {"type":"file","oid":"ptr2","size":135,"lfs":{"oid":"beef01","size":10},"path":"m-00001-of-00002.gguf"},
        {"type":"file","oid":"ptr3","size":135,"lfs":{"oid":"beef02","size":20},"path":"m-00002-of-00002.gguf"}
    ]"#;

    #[test]
    fn lfs_files_report_the_blob_id_and_real_size() {
        let files = parse_tree(TREE).unwrap();
        assert_eq!(files.len(), 4);
        let gguf = files
            .iter()
            .find(|file| file.path == "Qwen3-0.6B-Q4_0.gguf")
            .unwrap();
        assert_eq!(gguf.oid, "da2572");
        assert_eq!(gguf.size, 428_970_080);
        let plain = files
            .iter()
            .find(|file| file.path == ".gitattributes")
            .unwrap();
        assert_eq!(plain.oid, "aaa");
        assert_eq!(plain.size, 1803);
    }

    #[test]
    fn a_split_model_counts_every_shard() {
        let files = parse_tree(TREE).unwrap();
        let sizes = |path| family(&files, path).iter().map(|f| f.size).sum::<u64>();
        assert_eq!(sizes("m-00001-of-00002.gguf"), 30);
        assert_eq!(sizes("m-00002-of-00002.gguf"), 30);
        assert_eq!(sizes("Qwen3-0.6B-Q4_0.gguf"), 428_970_080);
        assert_eq!(sizes("absent.gguf"), 0);
    }

    #[test]
    fn primary_gguf_bytes_prefers_quant_tag_then_largest_family() {
        let files = parse_tree(TREE).unwrap();
        assert_eq!(primary_gguf_bytes(&files, "owner/model"), Some(428_970_080));
        assert_eq!(
            primary_gguf_bytes(&files, "owner/model:Q4_0"),
            Some(428_970_080)
        );
        assert_eq!(primary_gguf_bytes(&files, "owner/model:m"), Some(30));
        assert_eq!(primary_gguf_bytes(&[], "owner/model"), None);
        assert_eq!(
            primary_gguf_files(&files, "owner/model:Q4_0")
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec!["Qwen3-0.6B-Q4_0.gguf"]
        );
    }

    #[test]
    fn normalize_repo_id_accepts_urls() {
        assert_eq!(
            normalize_repo_id("https://huggingface.co/ggml-org/gpt-oss-120b-GGUF").unwrap(),
            "ggml-org/gpt-oss-120b-GGUF"
        );
        assert_eq!(
            normalize_repo_id("owner/model:Q4_K_M").unwrap(),
            "owner/model:Q4_K_M"
        );
    }

    #[test]
    fn malformed_listings_do_not_panic() {
        assert!(parse_tree("not json").is_err());
        assert!(parse_tree("[]").unwrap().is_empty());
        assert!(
            parse_tree(r#"[{"type":"file","path":"x"}]"#)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn repository_names_are_validated_before_a_request() {
        assert!(list_files("").is_err());
        assert!(list_files("no-slash").is_err());
        assert!(list_files("owner/model/extra").is_err());
    }
}
