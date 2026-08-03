use std::{
    fs::{self, OpenOptions},
    io::Write,
    net::IpAddr,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
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
    pub recent_models: Vec<ModelSource>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
            // Placeholder until the Hugging Face listing resolves the real GGUF
            // size. tinyinference overwrites this from the repository tree.
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
            warmup: true,
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
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
        let raw = toml::to_string_pretty(self).context("could not serialize configuration")?;
        let file_name = path
            .file_name()
            .context("configuration path has no file name")?
            .to_string_lossy();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), stamp));

        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)
                .with_context(|| format!("could not create {}", temporary.display()))?;
            file.write_all(raw.as_bytes())
                .with_context(|| format!("could not write {}", temporary.display()))?;
            file.sync_all()
                .with_context(|| format!("could not flush {}", temporary.display()))?;
            drop(file);
            replace_file(&temporary, path)
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
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

    pub fn effective_host(&self) -> String {
        let host = extra_option_value(&self.server.extra_args, "--host")
            .unwrap_or_else(|| self.server.host.clone());
        strip_brackets(host.trim()).to_string()
    }

    pub fn effective_port(&self) -> u16 {
        extra_option_value(&self.server.extra_args, "--port")
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.server.port)
    }

    pub fn connect_host(&self) -> String {
        loopback_host(&self.effective_host())
    }

    pub fn endpoint(&self) -> String {
        format!(
            "http://{}",
            format_authority(&self.effective_host(), self.effective_port())
        )
    }

    pub fn api_endpoint(&self) -> String {
        format!(
            "http://{}/v1",
            format_authority(&self.connect_host(), self.effective_port())
        )
    }

    pub fn remember_model(&mut self, source: ModelSource) {
        self.recent_models.retain(|recent| recent != &source);
        self.recent_models.insert(0, source);
        self.recent_models.truncate(8);
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let host = self.effective_host();
        let normalized_host = strip_brackets(host.trim());
        if normalized_host.is_empty() {
            errors.push("host cannot be empty".into());
        } else if host.starts_with('[') != host.ends_with(']') {
            errors.push("IPv6 hosts must use matching brackets".into());
        } else if normalized_host.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(character, '/' | '?' | '#')
        }) {
            errors.push("host contains invalid characters".into());
        } else if normalized_host.contains(':') && normalized_host.parse::<IpAddr>().is_err() {
            errors.push("host must be a valid IPv6 address or hostname".into());
        }
        if self.effective_port() == 0 {
            errors.push("port must be between 1 and 65535".into());
        }
        if has_extra_option(&self.server.extra_args, "--host") {
            match extra_option_value(&self.server.extra_args, "--host") {
                None => errors.push("extra --host requires a value".into()),
                Some(value) if value.trim().is_empty() || value.starts_with("--") => {
                    errors.push("extra --host requires a valid value".into())
                }
                Some(_) => {}
            }
        }
        if has_extra_option(&self.server.extra_args, "--port")
            && extra_option_value(&self.server.extra_args, "--port")
                .and_then(|value| value.parse::<u16>().ok())
                .is_none_or(|port| port == 0)
        {
            errors.push("extra --port must be between 1 and 65535".into());
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
            ModelSource::HuggingFace(id) => {
                if !valid_repository_id(id) {
                    errors.push("Hugging Face model ID must be owner/model".into());
                }
                if !self.model.estimated_size_gib.is_finite()
                    || !(0.1..=100_000.0).contains(&self.model.estimated_size_gib)
                {
                    errors.push("estimated file size must be between 0.1 and 100000 GiB".into());
                }
            }
            ModelSource::Local(path) => {
                if !path.is_file() {
                    errors.push(format!("model file does not exist: {}", path.display()));
                } else if !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                {
                    errors.push(format!(
                        "model file must have a .gguf extension: {}",
                        path.display()
                    ));
                } else if let Some(paths) = split_gguf_paths(path)
                    && let Some(missing) = paths.iter().find(|shard| !shard.is_file())
                {
                    errors.push(format!("model shard does not exist: {}", missing.display()));
                }
            }
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

fn replace_file(temporary: &Path, destination: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        match fs::rename(temporary, destination) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                fs::remove_file(destination)
                    .with_context(|| format!("could not replace {}", destination.display()))?;
                fs::rename(temporary, destination)
                    .with_context(|| format!("could not replace {}", destination.display()))
            }
            Err(error) => {
                Err(error).with_context(|| format!("could not replace {}", destination.display()))
            }
        }
    }
    #[cfg(not(windows))]
    {
        fs::rename(temporary, destination)
            .with_context(|| format!("could not replace {}", destination.display()))
    }
}

fn strip_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
}

fn loopback_host(host: &str) -> String {
    match strip_brackets(host) {
        "0.0.0.0" => "127.0.0.1".into(),
        "::" => "::1".into(),
        host => host.to_string(),
    }
}

fn format_authority(host: &str, port: u16) -> String {
    let host = strip_brackets(host);
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn has_extra_option(args: &[String], option: &str) -> bool {
    let prefix = format!("{option}=");
    args.iter()
        .any(|argument| argument == option || argument.starts_with(&prefix))
}

fn extra_option_value(args: &[String], option: &str) -> Option<String> {
    let prefix = format!("{option}=");
    let mut value = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == option {
            value = args.get(index + 1).cloned();
            index += 2;
        } else if let Some(argument) = args[index].strip_prefix(&prefix) {
            value = Some(argument.to_string());
            index += 1;
        } else {
            index += 1;
        }
    }
    value
}

fn valid_repository_id(id: &str) -> bool {
    let path = id.trim().split(':').next().unwrap_or_default();
    let Some((owner, name)) = path.split_once('/') else {
        return false;
    };
    !owner.is_empty()
        && !name.is_empty()
        && !name.contains('/')
        && [owner, name].iter().all(|part| {
            !part
                .chars()
                .any(|character| character.is_control() || character.is_whitespace())
                && !part
                    .chars()
                    .any(|character| matches!(character, '?' | '#' | '\\'))
        })
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
        assert!(cfg.runtime.warmup);
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
        let mut changed = expected.clone();
        changed.server.port = 9090;
        changed.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap().server.port, 9090);
    }

    #[test]
    fn invalid_micro_batch_is_reported() {
        let mut cfg = Config::default();
        cfg.runtime.batch_size = 4;
        cfg.runtime.micro_batch_size = 8;
        assert!(cfg.validate().iter().any(|e| e.contains("micro-batch")));
    }

    #[test]
    fn recent_models_are_unique_bounded_and_most_recent_first() {
        let mut config = Config::default();
        for index in 0..10 {
            config.remember_model(ModelSource::HuggingFace(format!("owner/model-{index}")));
        }
        config.remember_model(ModelSource::HuggingFace("owner/model-5".into()));
        let repeated = ModelSource::HuggingFace("owner/model-5".into());
        assert_eq!(config.recent_models.len(), 8);
        assert_eq!(config.recent_models.first(), Some(&repeated));
        assert!(
            config
                .recent_models
                .iter()
                .skip(1)
                .all(|model| model != &repeated)
        );
    }

    #[test]
    fn copied_api_endpoint_is_connectable() {
        let mut config = Config::default();
        config.server.host = "0.0.0.0".into();
        assert_eq!(config.api_endpoint(), "http://127.0.0.1:8080/v1");
        config.server.host = "::".into();
        assert_eq!(config.api_endpoint(), "http://[::1]:8080/v1");
    }

    #[test]
    fn effective_endpoint_honors_advanced_host_and_port_overrides() {
        let mut config = Config::default();
        config.server.extra_args = vec!["--host".into(), "::1".into(), "--port=9090".into()];
        assert_eq!(config.effective_host(), "::1");
        assert_eq!(config.effective_port(), 9090);
        assert_eq!(config.endpoint(), "http://[::1]:9090");
        assert_eq!(config.api_endpoint(), "http://[::1]:9090/v1");
        assert!(config.validate().is_empty());
    }

    #[test]
    fn invalid_remote_settings_are_reported_before_launch() {
        let mut config = Config::default();
        config.model.source = ModelSource::HuggingFace("owner/model/extra".into());
        config.model.estimated_size_gib = f64::NAN;
        config.server.extra_args = vec!["--port".into(), "not-a-port".into()];
        let errors = config.validate();
        assert!(errors.iter().any(|error| error.contains("owner/model")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("estimated file size"))
        );
        assert!(errors.iter().any(|error| error.contains("extra --port")));
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
