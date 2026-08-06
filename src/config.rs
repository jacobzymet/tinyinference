use std::{
    fs::{self, OpenOptions},
    io::Write,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::network::NetworkConfig;

pub const DEFAULT_MODEL: &str = "ggml-org/gpt-oss-120b-GGUF";
pub const DEFAULT_UI_HOST: &str = "127.0.0.1";
pub const DEFAULT_UI_PORT: u16 = 3920;
pub const DEFAULT_CACHE_TYPE: &str = "q8_0";
pub const CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];
/// Short blurbs for Configure; keep order aligned with [`CACHE_TYPES`].
pub const CACHE_TYPE_DESCRIPTIONS: &[(&str, &str)] = &[
    ("f32", "full precision; highest RAM"),
    ("f16", "half precision; strong quality baseline"),
    ("bf16", "brain float16; similar to f16"),
    ("q8_0", "~½ of f16 RAM; little quality loss (default)"),
    ("q4_0", "~¼ of f16 RAM; may drift on long context"),
    (
        "q4_1",
        "4-bit + bias; similar savings, often slightly better",
    ),
    (
        "iq4_nl",
        "4-bit non-linear; usually better than q4 at similar size",
    ),
    ("q5_0", "5-bit; between q4 and q8 on RAM/quality"),
    (
        "q5_1",
        "5-bit + bias; similar to q5_0, often slightly better",
    ),
];

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub ui: UiConfig,
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
    pub network: NetworkConfig,
    pub recent_models: Vec<ModelSource>,
    /// Self-signed cert for shared llama-server (not persisted in TOML).
    #[serde(skip)]
    pub tls_cert_file: Option<PathBuf>,
    /// Matching private key for [`Self::tls_cert_file`].
    #[serde(skip)]
    pub tls_key_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UiConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerConfig {
    pub executable: PathBuf,
    pub host: String,
    pub port: u16,
    /// Passed to llama-server as `--api-key` when non-empty (OpenAI Bearer).
    pub api_key: String,
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
    pub flash_attn: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub cache_ram_mib: u32,
    pub context_checkpoints: u32,
    pub multimodal_projector: bool,
    pub jinja: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_UI_HOST.into(),
            port: DEFAULT_UI_PORT,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("llama-server"),
            host: "127.0.0.1".into(),
            port: 8080,
            api_key: String::new(),
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
            flash_attn: true,
            cache_type_k: DEFAULT_CACHE_TYPE.into(),
            cache_type_v: DEFAULT_CACHE_TYPE.into(),
            cache_ram_mib: 0,
            context_checkpoints: 0,
            multimodal_projector: false,
            jinja: true,
        }
    }
}

/// Named runtime profiles shown in Configure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePreset {
    /// Current tinyinference default: CPU + mmap, no GPU offload.
    LowRam,
    /// GPU offload with auto-fit; leftover layers stay on CPU with mmap.
    GpuFit,
}

impl RuntimePreset {
    pub const ALL: [Self; 2] = [Self::LowRam, Self::GpuFit];

    pub fn id(self) -> &'static str {
        match self {
            Self::LowRam => "low_ram",
            Self::GpuFit => "gpu_fit",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LowRam => "Max RAM efficiency (default)",
            Self::GpuFit => "GPU + CPU spill",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::LowRam => {
                "Max RAM efficiency for low-RAM machines or oversized models. CPU-only + mmap: with enough RAM the working set stays cached at normal speed; otherwise as much as fits stays in RAM and the rest is read from disk. No GPU offload."
            }
            Self::GpuFit => {
                "Discrete GPU only: auto-fit into VRAM; leftover layers stay on CPU + mmap."
            }
        }
    }

    pub fn warning(self) -> Option<&'static str> {
        match self {
            Self::LowRam => None,
            Self::GpuFit => Some(
                "Unsafe on machines with a unified memory architecture (Apple Silicon and similar). GPU and CPU share one RAM pool, and this preset can immediately exhaust it and crash the machine if there is too little unified memory for the LLM.",
            ),
        }
    }

    pub fn blocked_on_unified_memory(self) -> bool {
        matches!(self, Self::GpuFit)
    }

    pub fn runtime(self) -> RuntimeConfig {
        match self {
            Self::LowRam => RuntimeConfig::default(),
            Self::GpuFit => RuntimeConfig {
                cpu_only: false,
                fit: true,
                ..RuntimeConfig::default()
            },
        }
    }

    pub fn matching(runtime: &RuntimeConfig) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|preset| &preset.runtime() == runtime)
    }
}

pub fn normalize_cache_type(value: &str) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase();
    if CACHE_TYPES.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        Err(format!(
            "Cache type must be one of: {}.",
            CACHE_TYPES.join(", ")
        ))
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

    pub fn ui_addr(&self) -> Result<SocketAddr> {
        parse_ui_addr(&self.ui.host, self.ui.port)
    }

    /// Resolve the UI listen address before the web server starts.
    ///
    /// Priority: `--bind` CLI flag, then `TINYINFERENCE_BIND`, then `[ui]`.
    /// The control panel and chat stay local; Network sharing exposes llama-server only.
    pub fn resolve_ui_bind(cli_bind: Option<SocketAddr>, config: &Self) -> Result<SocketAddr> {
        Self::resolve_ui_bind_with_env(cli_bind, std::env::var("TINYINFERENCE_BIND").ok(), config)
    }

    fn resolve_ui_bind_with_env(
        cli_bind: Option<SocketAddr>,
        env_bind: Option<String>,
        config: &Self,
    ) -> Result<SocketAddr> {
        let addr = if let Some(addr) = cli_bind {
            addr
        } else if let Some(raw) = env_bind {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                SocketAddr::from_str(trimmed)
                    .with_context(|| format!("invalid TINYINFERENCE_BIND value: {raw}"))?
            } else {
                config.desired_ui_bind()?
            }
        } else {
            config.desired_ui_bind()?
        };
        Ok(force_loopback_socket(addr))
    }

    /// Desired UI bind from config (ignores CLI/env overrides). Always loopback.
    pub fn desired_ui_bind(&self) -> Result<SocketAddr> {
        Ok(force_loopback_socket(parse_ui_addr(&self.ui.host, self.ui.port)?))
    }

    /// Keep the control panel on loopback. Network sharing binds llama-server instead.
    pub fn keep_ui_private(&mut self) {
        let host = self.ui.host.trim();
        if host != DEFAULT_UI_HOST && host != "localhost" && host != "::1" {
            self.ui.host = DEFAULT_UI_HOST.into();
        }
    }

    /// Apply Network sharing listen scope to the managed llama-server bind + API keys.
    ///
    /// Fail closed: if sharing is on but the scope cannot be resolved (e.g. Tailscale
    /// missing), bind loopback so a previous `0.0.0.0` listen cannot linger.
    pub fn sync_llama_bind_from_network(&mut self) {
        self.network.migrate_api_keys();
        match self.network.resolve_listen_host() {
            Ok(host) => {
                self.server.host = host;
                if self.network.expose {
                    // First key used for local probes / chat Authorization header.
                    self.server.api_key = self
                        .network
                        .primary_api_key()
                        .unwrap_or("")
                        .to_string();
                } else {
                    self.server.api_key.clear();
                }
            }
            Err(_) => {
                self.server.host = "127.0.0.1".into();
                if self.network.expose {
                    self.server.api_key = self
                        .network
                        .primary_api_key()
                        .unwrap_or("")
                        .to_string();
                } else {
                    self.server.api_key.clear();
                }
            }
        }
    }

    /// Secrets that should be passed to llama-server when sharing is on.
    pub fn llama_api_keys(&self) -> Vec<String> {
        if self.network.expose {
            self.network.api_key_secrets()
        } else {
            Vec::new()
        }
    }

    /// Older builds exposed the UI; migrate those configs to expose llama instead.
    /// Always syncs the llama bind (fail closed to loopback when Share is off).
    pub fn migrate_network_expose_to_llama(&mut self) {
        self.network.migrate_api_keys();
        self.keep_ui_private();
        self.sync_llama_bind_from_network();
    }

    pub fn model_label(&self) -> String {
        match &self.model.source {
            ModelSource::HuggingFace(id) if id.trim().is_empty() => "No model selected".into(),
            ModelSource::Local(path) if path.as_os_str().is_empty() => "No model selected".into(),
            ModelSource::HuggingFace(id) => id.clone(),
            ModelSource::Local(path) => path.display().to_string(),
        }
    }

    pub fn effective_host(&self) -> String {
        // Network sharing owns the bind. `extra_args --host` is rejected / stripped.
        strip_brackets(self.server.host.trim()).to_string()
    }

    pub fn effective_port(&self) -> u16 {
        // Configure owns the port. `extra_args --port` is rejected / stripped.
        self.server.port
    }

    pub fn connect_host(&self) -> String {
        loopback_host(&self.effective_host())
    }

    /// True when Share is on and TLS material is attached for llama-server.
    pub fn uses_tls(&self) -> bool {
        self.network.expose
            && self.tls_cert_file.as_ref().is_some_and(|p| p.as_os_str().len() > 0)
            && self.tls_key_file.as_ref().is_some_and(|p| p.as_os_str().len() > 0)
    }

    pub fn scheme(&self) -> &'static str {
        if self.uses_tls() { "https" } else { "http" }
    }

    /// Browser-openable base URL for llama-server (never `0.0.0.0` / `::`).
    pub fn endpoint(&self) -> String {
        format!(
            "{}://{}",
            self.scheme(),
            format_authority(&self.connect_host(), self.effective_port())
        )
    }

    /// Human-readable listen description (bind address may be all-interfaces).
    pub fn listen_label(&self) -> String {
        let bind = self.effective_host();
        let port = self.effective_port();
        let connect = self.connect_host();
        let scheme = self.scheme();
        if bind == "0.0.0.0" || bind == "::" {
            format!("{scheme}://{bind}:{port} (open via {connect})")
        } else {
            format!("{scheme}://{}", format_authority(&bind, port))
        }
    }

    pub fn api_endpoint(&self) -> String {
        format!(
            "{}://{}/v1",
            self.scheme(),
            format_authority(&self.connect_host(), self.effective_port())
        )
    }

    /// Attach or clear self-signed TLS paths used when sharing.
    pub fn set_share_tls(&mut self, paths: Option<(PathBuf, PathBuf)>) {
        match paths {
            Some((cert, key)) => {
                self.tls_cert_file = Some(cert);
                self.tls_key_file = Some(key);
            }
            None => {
                self.tls_cert_file = None;
                self.tls_key_file = None;
            }
        }
    }

    pub fn remember_model(&mut self, source: ModelSource) {
        let empty = match &source {
            ModelSource::HuggingFace(id) => id.trim().is_empty(),
            ModelSource::Local(path) => path.as_os_str().is_empty(),
        };
        if empty {
            return;
        }
        self.recent_models.retain(|recent| recent != &source);
        self.recent_models.insert(0, source);
        self.recent_models.truncate(8);
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if let Err(error) = self.ui_addr() {
            errors.push(format!("ui listen address is invalid: {error:#}"));
        }
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
            errors.push(
                "extra --host is not allowed; the listen address is set only in Network sharing"
                    .into(),
            );
        }
        if has_extra_option(&self.server.extra_args, "--port") {
            errors.push(
                "extra --port is not allowed; set the port in Configure".into(),
            );
        }
        if has_extra_option(&self.server.extra_args, "--api-key") {
            errors.push(
                "extra --api-key is not allowed; manage keys in Network sharing when Share is on"
                    .into(),
            );
        }
        if self.network.expose {
            if let Err(error) = self.network.resolve_listen_host() {
                errors.push(error);
            }
            if self.llama_api_keys().is_empty() {
                errors.push("sharing is on but no API keys are configured".into());
            }
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
        if let Err(error) = normalize_cache_type(&self.runtime.cache_type_k) {
            errors.push(format!("cache_type_k: {error}"));
        }
        if let Err(error) = normalize_cache_type(&self.runtime.cache_type_v) {
            errors.push(format!("cache_type_v: {error}"));
        }
        match &self.model.source {
            ModelSource::HuggingFace(id) if id.trim().is_empty() => {
                errors.push("Select a model in the Models tab before starting".into());
            }
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
            ModelSource::Local(path) if path.as_os_str().is_empty() => {
                errors.push("Select a model in the Models tab before starting".into());
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

pub(crate) fn split_gguf_paths(path: &Path) -> Option<Vec<PathBuf>> {
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

fn parse_ui_addr(host: &str, port: u16) -> Result<SocketAddr> {
    if port == 0 {
        bail!("ui port must be between 1 and 65535");
    }
    let host = strip_brackets(host.trim());
    if host.is_empty() {
        bail!("ui host cannot be empty");
    }
    let ip: IpAddr = host.parse().with_context(|| {
        format!("ui host must be an IP address such as 127.0.0.1 or ::1 (got {host})")
    })?;
    Ok(SocketAddr::from((ip, port)))
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

fn force_loopback_socket(addr: SocketAddr) -> SocketAddr {
    if addr.ip().is_loopback() {
        return addr;
    }
    match addr {
        SocketAddr::V4(_) => SocketAddr::from(([127, 0, 0, 1], addr.port())),
        SocketAddr::V6(_) => SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], addr.port())),
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
        assert_eq!(
            RuntimePreset::matching(&cfg.runtime),
            Some(RuntimePreset::LowRam)
        );
        assert!(cfg.runtime.cpu_only);
        assert!(cfg.runtime.mmap);
        assert!(!cfg.runtime.fit);
        assert!(!cfg.runtime.repack);
        assert!(cfg.runtime.warmup);
        assert!(cfg.runtime.flash_attn);
        assert_eq!(cfg.runtime.cache_type_k, "q8_0");
        assert_eq!(cfg.runtime.cache_type_v, "q8_0");
        assert_eq!(cfg.runtime.context_size, 8192);
        assert_eq!(cfg.runtime.cache_ram_mib, 0);
        assert_eq!(cfg.runtime.context_checkpoints, 0);
    }

    #[test]
    fn gpu_fit_preset_enables_gpu_and_auto_fit() {
        let runtime = RuntimePreset::GpuFit.runtime();
        assert!(!runtime.cpu_only);
        assert!(runtime.fit);
        assert!(runtime.mmap);
        assert_eq!(
            RuntimePreset::matching(&runtime),
            Some(RuntimePreset::GpuFit)
        );
        assert_eq!(runtime.batch_size, RuntimeConfig::default().batch_size);
        assert_eq!(runtime.cache_type_k, "q8_0");
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
        assert_eq!(config.endpoint(), "http://127.0.0.1:8080");
        assert_eq!(config.api_endpoint(), "http://127.0.0.1:8080/v1");
        assert_eq!(
            config.listen_label(),
            "http://0.0.0.0:8080 (open via 127.0.0.1)"
        );
        config.server.host = "::".into();
        assert_eq!(config.endpoint(), "http://[::1]:8080");
        assert_eq!(config.api_endpoint(), "http://[::1]:8080/v1");
    }

    #[test]
    fn configure_owns_port_and_rejects_extra_bind_flags() {
        let mut config = Config::default();
        config.server.host = "127.0.0.1".into();
        config.server.port = 9090;
        assert_eq!(config.effective_port(), 9090);
        assert_eq!(config.endpoint(), "http://127.0.0.1:9090");
        assert_eq!(config.api_endpoint(), "http://127.0.0.1:9090/v1");
        assert!(config.validate().is_empty());

        config.server.extra_args = vec!["--port=8080".into()];
        assert!(
            config
                .validate()
                .iter()
                .any(|error| error.contains("extra --port"))
        );
        config.server.extra_args = vec!["--host".into(), "0.0.0.0".into()];
        assert!(
            config
                .validate()
                .iter()
                .any(|error| error.contains("extra --host"))
        );
        config.server.extra_args = vec!["--api-key".into(), "secret".into()];
        assert!(
            config
                .validate()
                .iter()
                .any(|error| error.contains("extra --api-key"))
        );
    }

    #[test]
    fn migrate_always_syncs_bind_when_share_off() {
        let mut config = Config::default();
        config.network.expose = false;
        config.server.host = "0.0.0.0".into();
        config.migrate_network_expose_to_llama();
        assert_eq!(config.server.host, "127.0.0.1");
    }

    #[test]
    fn sync_llama_bind_fails_closed_when_scope_unresolved() {
        let mut config = Config::default();
        config.network.expose = true;
        config.network.listen_scope = crate::network::ListenScope::Custom;
        config.network.listen_host.clear();
        config.server.host = "0.0.0.0".into();
        config.sync_llama_bind_from_network();
        assert_eq!(config.server.host, "127.0.0.1");
    }

    #[test]
    fn invalid_remote_settings_are_reported_before_launch() {
        let mut config = Config::default();
        config.model.source = ModelSource::HuggingFace("owner/model/extra".into());
        config.model.estimated_size_gib = f64::NAN;
        config.server.extra_args = vec!["--threads".into(), "not-a-number".into()];
        let errors = config.validate();
        assert!(errors.iter().any(|error| error.contains("owner/model")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("estimated file size"))
        );
    }

    #[test]
    fn example_config_matches_defaults() {
        let example: Config =
            toml::from_str(include_str!("../tinyinference.example.toml")).unwrap();
        assert_eq!(example, Config::default());
    }

    #[test]
    fn ui_bind_comes_from_config_host_and_port() {
        let mut config = Config::default();
        config.ui.host = "0.0.0.0".into();
        config.ui.port = 4000;
        assert_eq!(config.ui_addr().unwrap(), "0.0.0.0:4000".parse().unwrap());
    }

    #[test]
    fn resolve_ui_bind_prefers_cli_then_env_then_config() {
        let mut config = Config::default();
        config.ui.port = 4000;
        let cli = "192.168.1.10:5555".parse().unwrap();
        // Non-loopback binds are forced back to loopback (same port) for UI privacy.
        assert_eq!(
            Config::resolve_ui_bind_with_env(Some(cli), Some("10.0.0.1:9".into()), &config)
                .unwrap(),
            "127.0.0.1:5555".parse().unwrap()
        );
        assert_eq!(
            Config::resolve_ui_bind_with_env(None, Some("10.0.0.1:9".into()), &config).unwrap(),
            "127.0.0.1:9".parse().unwrap()
        );
        assert_eq!(
            Config::resolve_ui_bind_with_env(None, None, &config).unwrap(),
            "127.0.0.1:4000".parse().unwrap()
        );
    }

    #[test]
    fn resolve_ui_bind_stays_loopback_when_network_expose_enabled() {
        let mut config = Config::default();
        config.ui.port = 3920;
        config.network.expose = true;
        config.network.listen_scope = crate::network::ListenScope::All;
        assert_eq!(
            Config::resolve_ui_bind_with_env(None, None, &config).unwrap(),
            "127.0.0.1:3920".parse().unwrap()
        );
        let cli = "127.0.0.1:3921".parse().unwrap();
        assert_eq!(
            Config::resolve_ui_bind_with_env(Some(cli), None, &config).unwrap(),
            cli
        );
    }

    #[test]
    fn sync_llama_bind_follows_network_scope() {
        let mut config = Config::default();
        config.network.expose = true;
        config.network.listen_scope = crate::network::ListenScope::Custom;
        config.network.listen_host = "192.168.1.50".into();
        config.network.access_token = "secret-token".into();
        config.sync_llama_bind_from_network();
        assert_eq!(config.server.host, "192.168.1.50");
        assert_eq!(config.server.api_key, "secret-token");
        assert_eq!(config.llama_api_keys(), vec!["secret-token".to_string()]);
        config.network.expose = false;
        config.sync_llama_bind_from_network();
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(config.server.api_key.is_empty());
        assert!(config.llama_api_keys().is_empty());
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
