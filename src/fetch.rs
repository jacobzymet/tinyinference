//! Download GGUF weights into the Hugging Face hub cache.
//!
//! Writes the same `models--owner--repo/blobs` layout llama-server and
//! `hf` use, so a model added here is immediately usable with `-hf`.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::{
    cache,
    hub::{self, RemoteFile},
};

const LIST_TIMEOUT: Duration = Duration::from_secs(20);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60 * 12);
const PROGRESS_EVERY_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone)]
pub enum FetchEvent {
    Started {
        repo: String,
        file: String,
        total: u64,
    },
    Progress {
        downloaded: u64,
        total: u64,
        file: String,
    },
    Finished {
        bytes: u64,
    },
    Error(String),
}

pub struct PendingFetch {
    receiver: Receiver<FetchEvent>,
    cancel: Arc<AtomicBool>,
}

impl PendingFetch {
    pub fn take(&self) -> Option<FetchEvent> {
        match self.receiver.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

/// Fetch the primary GGUF for `repo` into the local hub cache in the background.
pub fn fetch_primary_gguf_async(repo: &str) -> PendingFetch {
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_thread = Arc::clone(&cancel);
    let repo = repo.to_string();
    thread::spawn(move || {
        let result = fetch_primary_gguf(&repo, &cancel_thread, |event| {
            let _ = sender.send(event);
        });
        match result {
            Ok(bytes) => {
                let _ = sender.send(FetchEvent::Finished { bytes });
            }
            Err(error) => {
                let _ = sender.send(FetchEvent::Error(format!("{error:#}")));
            }
        }
    });
    PendingFetch { receiver, cancel }
}

fn fetch_primary_gguf(
    repo: &str,
    cancel: &AtomicBool,
    mut emit: impl FnMut(FetchEvent),
) -> Result<u64> {
    let repo = hub::normalize_repo_id(repo)?;
    let (owner, name) = hub_parts(&repo)?;
    let files = hub::list_files(&repo).context("could not list repository files")?;
    let selected = hub::primary_gguf_files(&files, &repo);
    if selected.is_empty() {
        bail!("no GGUF files found in {repo}");
    }
    let total: u64 = selected.iter().map(|file| file.size).sum();
    let label = selected
        .first()
        .map(|file| file.path.clone())
        .unwrap_or_else(|| repo.clone());
    emit(FetchEvent::Started {
        repo: repo.clone(),
        file: label.clone(),
        total,
    });

    let root = cache::default_hub_root();
    let model_dir = root.join(format!("models--{owner}--{name}"));
    let blobs_dir = model_dir.join("blobs");
    fs::create_dir_all(&blobs_dir)
        .with_context(|| format!("could not create {}", blobs_dir.display()))?;

    let revision = revision_sha(&owner, &name).unwrap_or_else(|| "main".into());
    let snapshot_dir = model_dir.join("snapshots").join(&revision);
    fs::create_dir_all(&snapshot_dir)
        .with_context(|| format!("could not create {}", snapshot_dir.display()))?;
    fs::create_dir_all(model_dir.join("refs"))
        .with_context(|| format!("could not create refs in {}", model_dir.display()))?;
    fs::write(model_dir.join("refs").join("main"), format!("{revision}\n"))
        .context("could not write refs/main")?;

    let mut downloaded = 0u64;
    for file in &selected {
        if cancel.load(Ordering::SeqCst) {
            bail!("download cancelled");
        }
        let already = blob_path(&blobs_dir, &file.oid);
        if already.is_file()
            && fs::metadata(&already)
                .map(|meta| meta.len() == file.size)
                .unwrap_or(false)
        {
            downloaded = downloaded.saturating_add(file.size);
            emit(FetchEvent::Progress {
                downloaded,
                total,
                file: file.path.clone(),
            });
        } else {
            downloaded = download_blob(
                (&owner, &name),
                file,
                &blobs_dir,
                downloaded,
                total,
                cancel,
                &mut emit,
            )?;
        }
        link_snapshot_file(&blobs_dir.join(&file.oid), &snapshot_dir.join(&file.path))?;
    }

    Ok(total)
}

fn download_blob(
    (owner, name): (&str, &str),
    file: &RemoteFile,
    blobs_dir: &Path,
    mut downloaded: u64,
    total: u64,
    cancel: &AtomicBool,
    emit: &mut impl FnMut(FetchEvent),
) -> Result<u64> {
    let url = format!(
        "https://huggingface.co/{}/{}/resolve/main/{}?download=true",
        encode(owner),
        encode(name),
        file.path
            .split('/')
            .map(encode)
            .collect::<Vec<_>>()
            .join("/")
    );
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(DOWNLOAD_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut response = agent.get(&url).call().context("download request failed")?;
    if response.status() != 200 {
        bail!(
            "Hugging Face responded with {} for {}",
            response.status(),
            file.path
        );
    }

    let partial = blobs_dir.join(format!("{}.incomplete", file.oid));
    let final_path = blobs_dir.join(&file.oid);
    if let Some(parent) = partial.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = File::create(&partial)
        .with_context(|| format!("could not create {}", partial.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0u8; 256 * 1024];
    let mut file_bytes = 0u64;
    let mut since_progress = 0u64;
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(output);
            let _ = fs::remove_file(&partial);
            bail!("download cancelled");
        }
        let read = reader.read(&mut buffer).context("download stream failed")?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .context("could not write download chunk")?;
        file_bytes = file_bytes.saturating_add(read as u64);
        downloaded = downloaded.saturating_add(read as u64);
        since_progress = since_progress.saturating_add(read as u64);
        if since_progress >= PROGRESS_EVERY_BYTES {
            since_progress = 0;
            emit(FetchEvent::Progress {
                downloaded,
                total,
                file: file.path.clone(),
            });
        }
    }
    output.flush().ok();
    drop(output);

    if file.size > 0 && file_bytes != file.size {
        let _ = fs::remove_file(&partial);
        bail!(
            "{} ended early ({file_bytes} of {} bytes)",
            file.path,
            file.size
        );
    }
    if final_path.exists() {
        fs::remove_file(&final_path).ok();
    }
    fs::rename(&partial, &final_path)
        .with_context(|| format!("could not finalize {}", final_path.display()))?;
    emit(FetchEvent::Progress {
        downloaded,
        total,
        file: file.path.clone(),
    });
    Ok(downloaded)
}

fn link_snapshot_file(blob: &Path, snapshot_file: &Path) -> Result<()> {
    if let Some(parent) = snapshot_file.parent() {
        fs::create_dir_all(parent)?;
    }
    if snapshot_file.exists() {
        fs::remove_file(snapshot_file).ok();
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        if let Some(relative) = relative_blob_link(blob, snapshot_file)
            && symlink(&relative, snapshot_file).is_ok()
        {
            return Ok(());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::symlink_file;
        if symlink_file(blob, snapshot_file).is_ok() {
            return Ok(());
        }
    }
    if fs::hard_link(blob, snapshot_file).is_ok() {
        return Ok(());
    }
    fs::copy(blob, snapshot_file)
        .map(|_| ())
        .with_context(|| format!("could not place {}", snapshot_file.display()))
}

#[cfg(unix)]
fn relative_blob_link(blob: &Path, snapshot_file: &Path) -> Option<PathBuf> {
    let model_dir = blob.parent()?.parent()?;
    let snapshot_parent = snapshot_file.parent()?;
    let rest = snapshot_parent.strip_prefix(model_dir).ok()?;
    let mut relative = PathBuf::new();
    for _ in rest.components() {
        relative.push("..");
    }
    relative.push("blobs");
    relative.push(blob.file_name()?);
    Some(relative)
}

fn blob_path(blobs_dir: &Path, oid: &str) -> PathBuf {
    blobs_dir.join(oid)
}

fn hub_parts(repo: &str) -> Result<(String, String)> {
    let path = repo.trim().split(':').next().unwrap_or_default();
    let Some((owner, name)) = path.split_once('/') else {
        bail!("{repo} is not an owner/model repository");
    };
    Ok((owner.to_string(), name.to_string()))
}

fn encode(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write as _;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    sha: String,
}

fn revision_sha(owner: &str, name: &str) -> Option<String> {
    let url = format!(
        "https://huggingface.co/api/models/{}/{}",
        encode(owner),
        encode(name)
    );
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(LIST_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut response = agent.get(&url).call().ok()?;
    if response.status() != 200 {
        return None;
    }
    let body = response.body_mut().read_to_string().ok()?;
    let info = serde_json::from_str::<ModelInfo>(&body).ok()?;
    let sha = info.sha.trim();
    (!sha.is_empty()).then(|| sha.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_accepts_urls_and_ids() {
        assert_eq!(
            hub::normalize_repo_id("https://huggingface.co/ggml-org/gpt-oss-120b-GGUF").unwrap(),
            "ggml-org/gpt-oss-120b-GGUF"
        );
        assert_eq!(
            hub::normalize_repo_id("hf://owner/model:Q4_K_M").unwrap(),
            "owner/model:Q4_K_M"
        );
    }
}
