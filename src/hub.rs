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
