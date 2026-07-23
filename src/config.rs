use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL: &str = "ggml-org/gpt-oss-120b-GGUF";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerConfig {
    pub executable: PathBuf,
    pub host: String,
    pub port: u16,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    pub source: ModelSource,
    pub estimated_size_gib: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum ModelSource {
    HuggingFace(String),
    Local(PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RuntimeConfig {
    pub context_size: u32,
    pub batch_size: u32,
    pub micro_batch_size: u32,
    pub parallel: u16,
    pub cpu_only: bool,
    pub mmap: bool,
    pub fit: bool,
    pub repack: bool,
    pub warmup: bool,
    pub cache_ram_mib: u32,
    pub context_checkpoints: u32,
    pub multimodal_projector: bool,
    pub jinja: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("llama-server"),
            host: "127.0.0.1".into(),
            port: 8080,
            extra_args: Vec::new(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::HuggingFace(DEFAULT_MODEL.into()),
            // The current official MXFP4 artifact is 63.4 decimal GB, or about
            // 59.1 binary GiB. This is editable because remote repositories change.
            estimated_size_gib: 59.1,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            context_size: 8192,
            batch_size: 8,
            micro_batch_size: 8,
            parallel: 1,
            cpu_only: true,
            mmap: true,
            fit: false,
            repack: false,
            warmup: false,
            cache_ram_mib: 0,
            context_checkpoints: 0,
            multimodal_projector: false,
            jinja: true,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("could not read {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("invalid config at {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("could not create {}", parent.display()))?;
        }
        let raw = toml::to_string_pretty(self).context("could not serialize configuration")?;
        fs::write(path, raw).with_context(|| format!("could not write {}", path.display()))
    }

    pub fn default_path() -> PathBuf {
        ProjectDirs::from("", "", "tinyinference")
            .map(|dirs| dirs.config_dir().join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("tinyinference.toml"))
    }

    pub fn model_label(&self) -> String {
        match &self.model.source {
            ModelSource::HuggingFace(id) => id.clone(),
            ModelSource::Local(path) => path.display().to_string(),
        }
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}:{}", self.server.host, self.server.port)
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.server.host.trim().is_empty() {
            errors.push("host cannot be empty".into());
        }
        if self.server.port == 0 {
            errors.push("port must be between 1 and 65535".into());
        }
        if self.runtime.context_size == 0 {
            errors.push("context size must be greater than zero".into());
        }
        if self.runtime.batch_size == 0 {
            errors.push("batch size must be greater than zero".into());
        }
        if self.runtime.micro_batch_size == 0 {
            errors.push("micro-batch size must be greater than zero".into());
        }
        if self.runtime.micro_batch_size > self.runtime.batch_size {
            errors.push("micro-batch size cannot exceed batch size".into());
        }
        if self.runtime.parallel == 0 {
            errors.push("parallel slots must be greater than zero".into());
        }
        match &self.model.source {
            ModelSource::HuggingFace(id) if id.trim().is_empty() => {
                errors.push("Hugging Face model ID cannot be empty".into())
            }
            ModelSource::Local(path) => {
                if !path.is_file() {
                    errors.push(format!("model file does not exist: {}", path.display()));
                } else if let Some(paths) = split_gguf_paths(path)
                    && let Some(missing) = paths.iter().find(|shard| !shard.is_file())
                {
                    errors.push(format!("model shard does not exist: {}", missing.display()));
                }
            }
            _ => {}
        }
        errors
    }

    pub fn local_model_size_gib(&self) -> Option<f64> {
        match &self.model.source {
            ModelSource::Local(path) => {
                let paths = split_gguf_paths(path).unwrap_or_else(|| vec![path.clone()]);
                let bytes = paths.iter().try_fold(0_u64, |total, shard| {
                    fs::metadata(shard)
                        .ok()
                        .and_then(|metadata| total.checked_add(metadata.len()))
                })?;
                Some(bytes as f64 / 1024_f64.powi(3))
            }
            ModelSource::HuggingFace(_) => None,
        }
    }
}

fn split_gguf_paths(path: &Path) -> Option<Vec<PathBuf>> {
    let file_name = path.file_name()?.to_str()?;
    let extension_start = file_name.len().checked_sub(5)?;
    let extension = file_name.get(extension_start..)?;
    if !extension.eq_ignore_ascii_case(".gguf") {
        return None;
    }
    let stem = file_name.get(..extension_start)?;
    let (indexed_name, total_text) = stem.rsplit_once("-of-")?;
    let (prefix, index_text) = indexed_name.rsplit_once('-')?;
    let index = index_text.parse::<usize>().ok()?;
    let total = total_text.parse::<usize>().ok()?;
    if index == 0 || index > total || total <= 1 || total > 9_999 {
        return None;
    }
    let width = index_text.len().max(total_text.len());
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    Some(
        (1..=total)
            .map(|part| {
                parent.join(format!(
                    "{prefix}-{part:0width$}-of-{total:0width$}{extension}"
                ))
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_low_memory_command() {
        let cfg = Config::default();
        assert!(cfg.runtime.cpu_only);
        assert!(cfg.runtime.mmap);
        assert!(!cfg.runtime.fit);
        assert!(!cfg.runtime.repack);
        assert!(!cfg.runtime.warmup);
        assert_eq!(cfg.runtime.context_size, 8192);
        assert_eq!(cfg.runtime.cache_ram_mib, 0);
        assert_eq!(cfg.runtime.context_checkpoints, 0);
    }

    #[test]
    fn config_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let expected = Config::default();
        expected.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), expected);
    }

    #[test]
    fn invalid_micro_batch_is_reported() {
        let mut cfg = Config::default();
        cfg.runtime.batch_size = 4;
        cfg.runtime.micro_batch_size = 8;
        assert!(cfg.validate().iter().any(|e| e.contains("micro-batch")));
    }

    #[test]
    fn example_config_matches_defaults() {
        let example: Config =
            toml::from_str(include_str!("../tinyinference.example.toml")).unwrap();
        assert_eq!(example, Config::default());
    }

    #[test]
    fn split_gguf_size_includes_every_shard() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("model-00001-of-00003.gguf");
        fs::write(&first, vec![0; 10]).unwrap();
        fs::write(dir.path().join("model-00002-of-00003.gguf"), vec![0; 20]).unwrap();
        fs::write(dir.path().join("model-00003-of-00003.gguf"), vec![0; 30]).unwrap();
        let mut config = Config::default();
        config.model.source = ModelSource::Local(first);
        let expected = 60.0 / 1024_f64.powi(3);
        assert!((config.local_model_size_gib().unwrap() - expected).abs() < f64::EPSILON);
        assert!(config.validate().is_empty());
    }

    #[test]
    fn missing_split_gguf_shard_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("model-00001-of-00002.gguf");
        fs::write(&first, vec![0; 10]).unwrap();
        let mut config = Config::default();
        config.model.source = ModelSource::Local(first);
        assert!(config.local_model_size_gib().is_none());
        assert!(
            config
                .validate()
                .iter()
                .any(|error| error.contains("00002-of-00002"))
        );
    }
}
