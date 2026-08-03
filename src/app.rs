use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{
    cache,
    config::{
        CACHE_TYPES, Config, DEFAULT_MODEL, ModelSource, RuntimePreset, normalize_cache_type,
    },
    hub,
    server::{
        CommandSpec, PendingProbe, ProbeResult, ServerEvent, ServerMetrics, ServerProcess,
        SlotsSnapshot, parse_log_throughput, probe_async,
    },
    system::{Machine, ProcessMonitor, ProcessUsage, copy_to_clipboard, executable_exists},
};

const MAX_LOG_LINES: usize = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPickerGroup {
    Recent,
    Cached,
}

impl ModelPickerGroup {
    pub fn label(self) -> &'static str {
        match self {
            Self::Recent => "Recent",
            Self::Cached => "Cached on this machine",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPickerEntry {
    pub group: ModelPickerGroup,
    pub source: ModelSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerStatus {
    Stopped,
    Downloading,
    Starting,
    Ready,
    Stopping,
    Failed,
}

impl ServerStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Downloading => "downloading",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Stopping => "stopping",
            Self::Failed => "failed",
        }
    }
}

/// How long a transfer rate is averaged over, and how often the cache is read.
const RATE_WINDOW: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(250);
/// How often to rescan local Hugging Face / llama.cpp caches for GGUF models.
const DISCOVER_INTERVAL: Duration = Duration::from_secs(15);

/// How often to re-probe llama-server `/slots` + `/metrics` while running.
const METRICS_PROBE_INTERVAL: Duration = Duration::from_millis(250);
/// Ignore tiny intervals so a burst of probes does not spike tok/s.
const MIN_LIVE_RATE_INTERVAL: Duration = Duration::from_millis(100);
/// Clear dashboard tok/s back to unavailable after this long without a fresh sample.
const THROUGHPUT_STALE_AFTER: Duration = Duration::from_secs(10);

/// Live progress of the model fetch llama-server performs before it can load.
///
/// The bytes come from the cache directory, which is the only place a download
/// is visible; the size to measure them against comes from the repository
/// listing, matched to the file being written by its object id.
#[derive(Debug)]
pub struct Download {
    pub repo: String,
    /// Name of the file being fetched, once the listing identifies it.
    pub file: Option<String>,
    pub downloaded: u64,
    /// Real size of the download, from Hugging Face.
    pub total: Option<u64>,
    active: bool,
    estimated_size_gib: f64,
    oids: Vec<String>,
    files: Vec<hub::RemoteFile>,
    listing: Option<hub::PendingListing>,
    pub listing_error: Option<String>,
    scan: cache::CacheScan,
    pending_scan: Option<cache::PendingScan>,
    scan_ready: bool,
    samples: VecDeque<(Instant, u64)>,
    last_poll: Instant,
}

impl Download {
    pub(crate) fn new(
        repo: &str,
        estimated_size_gib: f64,
        listing: Option<hub::PendingListing>,
    ) -> Self {
        let downloaded = 0;
        Self {
            repo: repo.to_string(),
            file: None,
            downloaded,
            total: None,
            active: true,
            estimated_size_gib,
            oids: Vec::new(),
            files: Vec::new(),
            listing,
            listing_error: None,
            scan: cache::CacheScan::default(),
            pending_scan: Some(cache::scan_async(repo)),
            scan_ready: false,
            samples: VecDeque::from([(Instant::now(), downloaded)]),
            last_poll: Instant::now(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn listing_error(&self) -> Option<&str> {
        self.listing_error.as_deref()
    }

    /// Bytes per second over the recent window, once there is enough history.
    pub fn rate(&self) -> Option<f64> {
        let (first_at, first_bytes) = *self.samples.front()?;
        let (last_at, last_bytes) = *self.samples.back()?;
        let seconds = last_at.duration_since(first_at).as_secs_f64();
        (seconds >= 1.0).then(|| last_bytes.saturating_sub(first_bytes) as f64 / seconds)
    }

    pub fn fraction(&self) -> Option<f64> {
        self.total
            .filter(|total| *total > 0)
            .map(|total| (self.downloaded as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Time left at the current rate.
    pub fn eta(&self) -> Option<Duration> {
        let remaining = self.total?.saturating_sub(self.downloaded);
        let rate = self.rate().filter(|rate| *rate > 1.0)?;
        Some(Duration::from_secs_f64(remaining as f64 / rate))
    }

    /// Re-read the cache, and the listing if it has arrived.
    fn poll(&mut self) {
        if self.last_poll.elapsed() < POLL_INTERVAL {
            return;
        }
        self.last_poll = Instant::now();
        let listing_changed =
            if let Some(result) = self.listing.as_mut().and_then(hub::PendingListing::take) {
                self.listing = None;
                match result {
                    Ok(files) => self.files = files,
                    Err(error) => self.listing_error = Some(error),
                }
                true
            } else {
                false
            };

        let completed_scan = self
            .pending_scan
            .as_ref()
            .and_then(cache::PendingScan::take);
        let scan_changed = if let Some(scan) = completed_scan {
            self.pending_scan = None;
            let first_scan = !self.scan_ready;
            self.scan_ready = true;
            self.scan = scan;
            let scan = self.scan.clone();
            self.resolve_target(&scan);
            self.update_from_scan(first_scan, true);
            true
        } else {
            false
        };

        if listing_changed && !scan_changed && self.scan_ready {
            let scan = self.scan.clone();
            self.resolve_target(&scan);
            self.update_from_scan(false, false);
        }
        if scan_changed {
            self.pending_scan = Some(cache::scan_async(&self.repo));
        }
    }

    fn update_from_scan(&mut self, first_scan: bool, add_sample: bool) {
        let downloaded = if self.oids.is_empty() {
            self.scan.total_bytes()
        } else {
            self.scan.bytes_of(&self.oids)
        };
        let in_flight = self.scan.in_flight().next().is_some();
        let complete_by_estimate =
            !cache::looks_incomplete_scan(&self.scan, self.estimated_size_gib);
        self.active = if first_scan && !in_flight && complete_by_estimate {
            false
        } else if in_flight || downloaded > self.downloaded {
            true
        } else if self.total.is_some_and(|total| downloaded >= total)
            || (self.total.is_none() && complete_by_estimate)
        {
            false
        } else {
            self.active
        };
        self.downloaded = downloaded;

        if add_sample {
            let now = Instant::now();
            self.samples.push_back((now, downloaded));
            while self
                .samples
                .front()
                .is_some_and(|(at, _)| now.duration_since(*at) > RATE_WINDOW)
            {
                self.samples.pop_front();
            }
        }
    }

    /// Match the file being written to the repository listing, which turns the
    /// blob on disk into a name and an exact size.
    fn resolve_target(&mut self, scan: &cache::CacheScan) {
        if self.total.is_some() || self.files.is_empty() {
            return;
        }
        let oid = scan
            .in_flight()
            .find_map(|blob| {
                self.files
                    .iter()
                    .find(|file| file.oid == blob.oid && is_model_file(&file.path))
                    .map(|file| file.oid.clone())
            })
            .or_else(|| {
                scan.blobs.iter().find_map(|blob| {
                    self.files
                        .iter()
                        .find(|file| file.oid == blob.oid && is_model_file(&file.path))
                        .map(|file| file.oid.clone())
                })
            });
        let Some(oid) = oid else {
            return;
        };
        let Some(wanted) = self.files.iter().find(|file| file.oid == oid) else {
            return;
        };
        let family = hub::family(&self.files, &wanted.path);
        self.total = Some(family.iter().map(|file| file.size).sum());
        self.oids = family.iter().map(|file| file.oid.clone()).collect();
        self.file = Some(wanted.path.clone());
    }

    fn finish(&mut self) {
        self.active = false;
    }
}

fn is_model_file(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingField {
    SourceKind,
    Model,
    EstimatedSize,
    Executable,
    Host,
    Port,
    Context,
    Batch,
    MicroBatch,
    Parallel,
    CpuOnly,
    Mmap,
    Fit,
    Repack,
    Warmup,
    FlashAttn,
    CacheTypeK,
    CacheTypeV,
    CacheRam,
    Checkpoints,
    Mmproj,
    Jinja,
}

impl SettingField {
    pub const ALL: [Self; 22] = [
        Self::SourceKind,
        Self::Model,
        Self::EstimatedSize,
        Self::Executable,
        Self::Host,
        Self::Port,
        Self::Context,
        Self::Batch,
        Self::MicroBatch,
        Self::Parallel,
        Self::CpuOnly,
        Self::Mmap,
        Self::Fit,
        Self::Repack,
        Self::Warmup,
        Self::FlashAttn,
        Self::CacheTypeK,
        Self::CacheTypeV,
        Self::CacheRam,
        Self::Checkpoints,
        Self::Mmproj,
        Self::Jinja,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::SourceKind => "Model source",
            Self::Model => "Model",
            Self::EstimatedSize => "Mapped file size",
            Self::Executable => "Server executable",
            Self::Host => "Listen address",
            Self::Port => "Port",
            Self::Context => "Context",
            Self::Batch => "Batch",
            Self::MicroBatch => "Micro-batch",
            Self::Parallel => "Parallel slots",
            Self::CpuOnly => "CPU only",
            Self::Mmap => "Memory map",
            Self::Fit => "Auto-fit",
            Self::Repack => "Repack weights",
            Self::Warmup => "Warm up",
            Self::FlashAttn => "Flash attention",
            Self::CacheTypeK => "Key (K)",
            Self::CacheTypeV => "Value (V)",
            Self::CacheRam => "Prompt cache RAM",
            Self::Checkpoints => "Context checkpoints",
            Self::Mmproj => "Multimodal projector",
            Self::Jinja => "Jinja template",
        }
    }

    pub fn is_editable(self) -> bool {
        matches!(
            self,
            Self::Model
                | Self::EstimatedSize
                | Self::Executable
                | Self::Host
                | Self::Port
                | Self::Context
                | Self::Batch
                | Self::MicroBatch
                | Self::Parallel
                | Self::CacheTypeK
                | Self::CacheTypeV
                | Self::CacheRam
                | Self::Checkpoints
        )
    }

    pub fn is_toggle(self) -> bool {
        matches!(
            self,
            Self::SourceKind
                | Self::CpuOnly
                | Self::Mmap
                | Self::Fit
                | Self::Repack
                | Self::Warmup
                | Self::FlashAttn
                | Self::Mmproj
                | Self::Jinja
        )
    }

    pub fn choices(self) -> Option<&'static [&'static str]> {
        matches!(self, Self::CacheTypeK | Self::CacheTypeV).then_some(CACHE_TYPES)
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::SourceKind => "source_kind",
            Self::Model => "model",
            Self::EstimatedSize => "estimated_size",
            Self::Executable => "executable",
            Self::Host => "host",
            Self::Port => "port",
            Self::Context => "context",
            Self::Batch => "batch",
            Self::MicroBatch => "micro_batch",
            Self::Parallel => "parallel",
            Self::CpuOnly => "cpu_only",
            Self::Mmap => "mmap",
            Self::Fit => "fit",
            Self::Repack => "repack",
            Self::Warmup => "warmup",
            Self::FlashAttn => "flash_attn",
            Self::CacheTypeK => "cache_type_k",
            Self::CacheTypeV => "cache_type_v",
            Self::CacheRam => "cache_ram",
            Self::Checkpoints => "checkpoints",
            Self::Mmproj => "mmproj",
            Self::Jinja => "jinja",
        }
    }
}

/// Live Hugging Face listing used to resolve the real mapped GGUF size.
#[derive(Debug)]
struct RemoteModelSize {
    repo: String,
    listing: Option<hub::PendingListing>,
    pub bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct App {
    pub config: Config,
    pub config_path: PathBuf,
    pub machine: Machine,
    pub status: ServerStatus,
    pub status_detail: String,
    pub process: Option<ServerProcess>,
    running_config: Option<Config>,
    pub logs: VecDeque<String>,
    last_hf_model: String,
    last_local_model: PathBuf,
    pub endpoint_online: bool,
    pub download: Option<Download>,
    pub process_usage: Option<ProcessUsage>,
    pub server_metrics: Option<ServerMetrics>,
    process_monitor: ProcessMonitor,
    probe: Option<PendingProbe>,
    remote_model_size: Option<RemoteModelSize>,
    missing_server_prompt: bool,
    last_probe: Instant,
    last_stats_refresh: Instant,
    /// Previous `/slots` sample used to derive live generation tok/s.
    last_slots_at: Option<Instant>,
    last_slots_decoded: Option<u64>,
    live_generated_tps: Option<f64>,
    /// Last time gen/prompt tok/s was refreshed from logs, slots, or metrics.
    last_throughput_at: Option<Instant>,
    /// GGUF models found in the local Hugging Face hub / llama.cpp caches.
    discovered_models: Vec<cache::DiscoveredModel>,
    pending_discover: Option<cache::PendingDiscover>,
    last_discover: Instant,
}

impl App {
    pub fn new(mut config: Config, config_path: PathBuf) -> Self {
        let executable_found = executable_exists(&config.server.executable);
        let initial_model = config.model.source.clone();
        config.remember_model(initial_model);
        let (last_hf_model, last_local_model) = match &config.model.source {
            ModelSource::HuggingFace(id) => (id.clone(), PathBuf::new()),
            ModelSource::Local(path) => (DEFAULT_MODEL.into(), path.clone()),
        };
        let status_detail = if executable_found {
            "Ready to launch".into()
        } else {
            format!(
                "{} was not found; set its path in Configure",
                config.server.executable.display()
            )
        };
        let mut app = Self {
            config,
            config_path,
            machine: Machine::detect(),
            status: ServerStatus::Stopped,
            status_detail,
            process: None,
            running_config: None,
            logs: VecDeque::from(["tinyinference initialized".into()]),
            last_hf_model,
            last_local_model,
            endpoint_online: false,
            download: None,
            process_usage: None,
            server_metrics: None,
            process_monitor: ProcessMonitor::default(),
            probe: None,
            remote_model_size: None,
            missing_server_prompt: !executable_found,
            last_probe: Instant::now() - Duration::from_secs(2),
            last_stats_refresh: Instant::now() - Duration::from_secs(2),
            last_slots_at: None,
            last_slots_decoded: None,
            live_generated_tps: None,
            last_throughput_at: None,
            discovered_models: Vec::new(),
            pending_discover: Some(cache::discover_models_async()),
            last_discover: Instant::now(),
        };
        app.sanitize_unified_memory_presets();
        app.sync_remote_model_size();
        app
    }

    pub fn command(&self) -> CommandSpec {
        CommandSpec::from_config(&self.config)
    }

    pub fn displayed_config(&self) -> &Config {
        self.running_config.as_ref().unwrap_or(&self.config)
    }

    pub fn has_pending_changes(&self) -> bool {
        self.running_config
            .as_ref()
            .is_some_and(|running| running != &self.config)
    }

    pub fn should_prompt_for_server(&self) -> bool {
        self.missing_server_prompt
    }

    pub fn dismiss_server_prompt(&mut self) {
        self.missing_server_prompt = false;
    }

    pub fn open_server_configuration(&mut self) {
        self.missing_server_prompt = false;
        self.status_detail = "Set the llama-server executable path, then save".into();
    }

    pub fn setting_value(&self, field: SettingField) -> String {
        match field {
            SettingField::SourceKind => match self.config.model.source {
                ModelSource::HuggingFace(_) => "Hugging Face".into(),
                ModelSource::Local(_) => "Local GGUF".into(),
            },
            SettingField::Model => match &self.config.model.source {
                ModelSource::HuggingFace(id) if id.trim().is_empty() => {
                    "<enter owner/model>".into()
                }
                ModelSource::Local(path) if path.as_os_str().is_empty() => {
                    "<enter path to .gguf>".into()
                }
                _ => self.config.model_label(),
            },
            SettingField::EstimatedSize => match self.mapped_model_gib() {
                Some(size) => match &self.config.model.source {
                    ModelSource::Local(_) => format!("{size:.1} GiB  (from file)"),
                    ModelSource::HuggingFace(_) => format!("{size:.1} GiB  (from Hugging Face)"),
                },
                None if matches!(self.config.model.source, ModelSource::Local(_)) => {
                    "auto from .gguf".into()
                }
                None if self.remote_model_size_loading() => "fetching from Hugging Face…".into(),
                None => "unavailable".into(),
            },
            SettingField::Executable => {
                let path = &self.config.server.executable;
                if path.components().count() == 1 && !path.is_absolute() {
                    format!("{}  (from PATH)", path.display())
                } else {
                    path.display().to_string()
                }
            }
            SettingField::Host => self.config.effective_host(),
            SettingField::Port => self.config.effective_port().to_string(),
            SettingField::Context => format!("{} tokens", self.config.runtime.context_size),
            SettingField::Batch => self.config.runtime.batch_size.to_string(),
            SettingField::MicroBatch => self.config.runtime.micro_batch_size.to_string(),
            SettingField::Parallel => self.config.runtime.parallel.to_string(),
            SettingField::CpuOnly => on_off(self.config.runtime.cpu_only),
            SettingField::Mmap => on_off(self.config.runtime.mmap),
            SettingField::Fit => on_off(self.config.runtime.fit),
            SettingField::Repack => on_off(self.config.runtime.repack),
            SettingField::Warmup => on_off(self.config.runtime.warmup),
            SettingField::FlashAttn => on_off(self.config.runtime.flash_attn),
            SettingField::CacheTypeK => self.config.runtime.cache_type_k.clone(),
            SettingField::CacheTypeV => self.config.runtime.cache_type_v.clone(),
            SettingField::CacheRam => format!("{} MiB", self.config.runtime.cache_ram_mib),
            SettingField::Checkpoints => self.config.runtime.context_checkpoints.to_string(),
            SettingField::Mmproj => on_off(self.config.runtime.multimodal_projector),
            SettingField::Jinja => on_off(self.config.runtime.jinja),
        }
    }

    pub fn setting_raw_value(&self, field: SettingField) -> String {
        match field {
            SettingField::SourceKind => match self.config.model.source {
                ModelSource::HuggingFace(_) => "hugging_face".into(),
                ModelSource::Local(_) => "local".into(),
            },
            SettingField::Model => self.config.model_label(),
            SettingField::EstimatedSize => self
                .mapped_model_gib()
                .map(|size| format!("{size:.1}"))
                .unwrap_or_default(),
            SettingField::Executable => self.config.server.executable.display().to_string(),
            SettingField::Host => self.config.effective_host(),
            SettingField::Port => self.config.effective_port().to_string(),
            SettingField::Context => self.config.runtime.context_size.to_string(),
            SettingField::Batch => self.config.runtime.batch_size.to_string(),
            SettingField::MicroBatch => self.config.runtime.micro_batch_size.to_string(),
            SettingField::Parallel => self.config.runtime.parallel.to_string(),
            SettingField::CpuOnly => self.config.runtime.cpu_only.to_string(),
            SettingField::Mmap => self.config.runtime.mmap.to_string(),
            SettingField::Fit => self.config.runtime.fit.to_string(),
            SettingField::Repack => self.config.runtime.repack.to_string(),
            SettingField::Warmup => self.config.runtime.warmup.to_string(),
            SettingField::FlashAttn => self.config.runtime.flash_attn.to_string(),
            SettingField::CacheTypeK => self.config.runtime.cache_type_k.clone(),
            SettingField::CacheTypeV => self.config.runtime.cache_type_v.clone(),
            SettingField::CacheRam => self.config.runtime.cache_ram_mib.to_string(),
            SettingField::Checkpoints => self.config.runtime.context_checkpoints.to_string(),
            SettingField::Mmproj => self.config.runtime.multimodal_projector.to_string(),
            SettingField::Jinja => self.config.runtime.jinja.to_string(),
        }
    }

    pub fn setting_label(&self, field: SettingField) -> &'static str {
        match (field, &self.config.model.source) {
            (SettingField::Model, ModelSource::HuggingFace(_)) => "Model repository",
            (SettingField::Model, ModelSource::Local(_)) => "GGUF file path",
            (SettingField::EstimatedSize, _) => "Mapped file size",
            (SettingField::Executable, _) => "llama-server path",
            _ => field.label(),
        }
    }

    pub fn setting_is_editable(&self, field: SettingField) -> bool {
        field.is_editable() && field != SettingField::EstimatedSize
    }

    /// Size of the model file(s) mmap will map, when known.
    pub fn mapped_model_gib(&self) -> Option<f64> {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        if let Some(size) = self.config.local_model_size_gib() {
            return Some(size);
        }
        if let Some(total) = self.active_download().and_then(|download| download.total) {
            return Some(total as f64 / GIB);
        }
        self.remote_model_size
            .as_ref()
            .and_then(|remote| remote.bytes)
            .map(|bytes| bytes as f64 / GIB)
    }

    pub fn remote_model_size_loading(&self) -> bool {
        matches!(
            &self.remote_model_size,
            Some(RemoteModelSize {
                listing: Some(_),
                bytes: None,
                error: None,
                ..
            })
        )
    }

    pub fn remote_model_size_error(&self) -> Option<&str> {
        self.remote_model_size
            .as_ref()
            .and_then(|remote| remote.error.as_deref())
    }

    pub fn setting_hint(&self, field: SettingField) -> &'static str {
        match (field, &self.config.model.source) {
            (SettingField::SourceKind, _) => "Switch between Hugging Face and a local GGUF file.",
            (SettingField::Model, ModelSource::HuggingFace(_)) => {
                "Enter owner/model, or pick a recent / autodiscovered model above."
            }
            (SettingField::Model, ModelSource::Local(_)) => {
                "Enter a full .gguf path, or pick a recent / autodiscovered model above."
            }
            (SettingField::EstimatedSize, ModelSource::Local(_)) => {
                "Calculated automatically from the GGUF file."
            }
            (SettingField::EstimatedSize, ModelSource::HuggingFace(_)) => {
                "Fetched from the Hugging Face file listing for this repository."
            }
            (SettingField::Executable, _) => {
                "Enter llama-server if it is on PATH, or enter its full executable path."
            }
            (SettingField::Host, _) => "127.0.0.1 keeps the server local to this machine.",
            (SettingField::Port, _) => "Port llama-server listens on.",
            (SettingField::Context, _) => "Token count; 8k is accepted.",
            (SettingField::Batch, _) => "Prompt batch size.",
            (SettingField::MicroBatch, _) => "Must be no larger than the batch size.",
            (SettingField::Parallel, _) => "Number of simultaneous server slots.",
            (SettingField::CpuOnly, _) => "On forces all model layers onto the CPU.",
            (SettingField::Mmap, _) => "On keeps model weights file-backed and demand-paged.",
            (SettingField::Fit, _) => "Off prevents llama.cpp from changing the requested profile.",
            (SettingField::Repack, _) => "Off avoids a separate repacked weight copy.",
            (SettingField::Warmup, _) => {
                "On touches model pages at startup for smoother first-token behavior."
            }
            (SettingField::FlashAttn, _) => {
                "On uses flash attention to cut attention memory; keep on with quantized KV."
            }
            (SettingField::CacheTypeK, _) => "Attention key half of the context scratchpad.",
            (SettingField::CacheTypeV, _) => "Attention value half of the context scratchpad.",
            (SettingField::CacheRam, _) => "Host prompt-cache limit in MiB; zero disables it.",
            (SettingField::Checkpoints, _) => "Saved context states per slot; zero disables them.",
            (SettingField::Mmproj, _) => "Leave off for text-only models.",
            (SettingField::Jinja, _) => "Uses the model's Jinja chat template.",
        }
    }

    pub fn tick(&mut self) {
        self.poll_remote_model_size();
        self.poll_discovered_models();

        let new_logs = self
            .process
            .as_ref()
            .map(|process| process.drain_logs().collect::<Vec<_>>())
            .unwrap_or_default();
        for event in new_logs {
            let ServerEvent::Log(line) = event;
            self.observe_startup_line(&line);
            self.observe_throughput_line(&line);
            self.push_log(line);
        }

        let exit = match self.process.as_mut() {
            Some(process) => process.try_wait(),
            None => Ok(None),
        };
        match exit {
            Ok(Some(status)) => {
                let tail_logs = if let Some(process) = self.process.as_mut() {
                    process.finish_output();
                    process.drain_logs().collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                for event in tail_logs {
                    let ServerEvent::Log(line) = event;
                    self.push_log(line);
                }
                self.process = None;
                self.running_config = None;
                self.endpoint_online = false;
                self.download = None;
                self.process_usage = None;
                self.server_metrics = None;
                self.clear_live_throughput();
                self.probe = None;
                self.status = if status.success() {
                    ServerStatus::Stopped
                } else {
                    ServerStatus::Failed
                };
                self.status_detail = match status.code() {
                    Some(code) => format!("llama-server exited with code {code}"),
                    None => "llama-server was terminated".into(),
                };
                self.push_log(self.status_detail.clone());
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
            }
            Ok(None) => {}
        }

        // Downloads are polled far more often than the endpoint, so the
        // progress figures move while a multi-gigabyte fetch is running.
        if self.process.is_some()
            && !self.endpoint_online
            && let Some(download) = self.download.as_mut()
        {
            download.poll();
            if download.is_active() {
                self.status = ServerStatus::Downloading;
            } else if self.status == ServerStatus::Downloading {
                self.status = ServerStatus::Starting;
                self.status_detail = "Model is on disk; loading weights".into();
            }
        }

        if self.process.is_some() {
            if let Some(result) = self.probe.as_ref().and_then(PendingProbe::take) {
                self.probe = None;
                self.apply_probe_result(result);
            }
            if self.probe.is_none() && self.last_probe.elapsed() >= METRICS_PROBE_INTERVAL {
                if let Some(config) = self.running_config.clone() {
                    self.probe = Some(probe_async(&config, true));
                }
                self.last_probe = Instant::now();
            }
        }

        if self.process.is_some() && self.last_stats_refresh.elapsed() >= Duration::from_secs(1) {
            let process_id = self.process.as_ref().map(ServerProcess::id);
            self.process_usage = process_id.and_then(|pid| self.process_monitor.refresh(pid));
            self.last_stats_refresh = Instant::now();
        }

        self.expire_stale_throughput();
    }

    fn apply_probe_result(&mut self, result: ProbeResult) {
        self.endpoint_online = result.endpoint_online;
        if let Some(slots) = result.slots {
            self.update_live_throughput(slots);
        }
        if result.metrics_requested {
            // Keep the last good sample when a scrape times out mid-decode.
            if let Some(metrics) = result.metrics {
                self.merge_server_metrics(metrics);
            }
        }
        self.publish_live_rates();
        if self.endpoint_online {
            self.download = None;
            self.status = ServerStatus::Ready;
            if let Some(config) = &self.running_config {
                self.status_detail = format!("Listening at {}", config.endpoint());
            }
        } else if self.status != ServerStatus::Downloading {
            // Process is still up; a timed-out task-queue probe is not a restart.
            self.status = ServerStatus::Starting;
        }
    }

    fn clear_live_throughput(&mut self) {
        self.last_slots_at = None;
        self.last_slots_decoded = None;
        self.live_generated_tps = None;
        self.last_throughput_at = None;
        if let Some(metrics) = self.server_metrics.as_mut() {
            metrics.generated_tokens_per_second = None;
            metrics.prompt_tokens_per_second = None;
        }
    }

    fn touch_throughput(&mut self) {
        self.last_throughput_at = Some(Instant::now());
    }

    fn expire_stale_throughput(&mut self) {
        let Some(last) = self.last_throughput_at else {
            return;
        };
        if last.elapsed() < THROUGHPUT_STALE_AFTER {
            return;
        }
        self.live_generated_tps = None;
        self.last_throughput_at = None;
        if let Some(metrics) = self.server_metrics.as_mut() {
            metrics.generated_tokens_per_second = None;
            metrics.prompt_tokens_per_second = None;
        }
    }

    fn update_live_throughput(&mut self, slots: SlotsSnapshot) {
        let now = Instant::now();
        if slots.requests_processing == 0 {
            self.last_slots_at = Some(now);
            self.last_slots_decoded = Some(0);
            // Idle: drop the live rate; stale timer will clear the displayed value.
            self.live_generated_tps = None;
            return;
        }

        if let (Some(prev_at), Some(prev_decoded)) = (self.last_slots_at, self.last_slots_decoded) {
            let dt = now.saturating_duration_since(prev_at);
            if dt >= MIN_LIVE_RATE_INTERVAL {
                if slots.decoded_tokens >= prev_decoded {
                    let delta = slots.decoded_tokens - prev_decoded;
                    if delta > 0 {
                        self.live_generated_tps = Some(delta as f64 / dt.as_secs_f64());
                        self.touch_throughput();
                    }
                    // delta == 0 between slow tokens: keep the previous live rate
                    // until THROUGHPUT_STALE_AFTER expires it.
                } else {
                    // Slot reset for a new request — re-baseline without a spike.
                    self.live_generated_tps = None;
                }
            }
        }

        self.last_slots_at = Some(now);
        self.last_slots_decoded = Some(slots.decoded_tokens);

        let metrics = self
            .server_metrics
            .get_or_insert_with(ServerMetrics::default);
        metrics.requests_processing = Some(slots.requests_processing as f64);
    }

    fn merge_server_metrics(&mut self, metrics: ServerMetrics) {
        let merged = self
            .server_metrics
            .get_or_insert_with(ServerMetrics::default);
        if metrics.prompt_tokens.is_some() {
            merged.prompt_tokens = metrics.prompt_tokens;
        }
        if metrics.generated_tokens.is_some() {
            merged.generated_tokens = metrics.generated_tokens;
        }
        let mut touched = false;
        if let Some(rate) = metrics.prompt_tokens_per_second.filter(|rate| *rate > 0.0) {
            merged.prompt_tokens_per_second = Some(rate);
            touched = true;
        }
        // Prefer live slot/log-derived gen tok/s while generating; accept
        // prometheus averages when idle or before the first live sample.
        if self.live_generated_tps.is_none()
            && let Some(rate) = metrics
                .generated_tokens_per_second
                .filter(|rate| *rate > 0.0)
        {
            merged.generated_tokens_per_second = Some(rate);
            touched = true;
        }
        if metrics.requests_processing.is_some() {
            merged.requests_processing = metrics.requests_processing;
        }
        if metrics.requests_deferred.is_some() {
            merged.requests_deferred = metrics.requests_deferred;
        }
        if touched {
            self.touch_throughput();
        }
    }

    fn publish_live_rates(&mut self) {
        let Some(live) = self.live_generated_tps else {
            return;
        };
        let metrics = self
            .server_metrics
            .get_or_insert_with(ServerMetrics::default);
        metrics.generated_tokens_per_second = Some(live);
    }

    pub fn start(&mut self) {
        if self.process.is_some() {
            return;
        }
        if let Err(errors) = self.validate_for_launch() {
            self.status = ServerStatus::Failed;
            self.status_detail = errors.clone();
            self.push_log(format!("[start blocked] {errors}"));
            return;
        }
        let launch_config = self.config.clone();
        let display = CommandSpec::from_config(&launch_config).display();
        self.push_log(format!("$ {display}"));
        match ServerProcess::start(&launch_config) {
            Ok(process) => {
                let pid = process.id();
                self.download = watch_download(&launch_config);
                self.process = Some(process);
                self.running_config = Some(launch_config);
                if self.active_download().is_some() {
                    self.status = ServerStatus::Downloading;
                    self.status_detail = "Fetching the model from Hugging Face".into();
                    self.push_log("model is not in the local cache; downloading".into());
                } else {
                    self.status = ServerStatus::Starting;
                    self.status_detail = format!("Waking llama-server (PID {pid})");
                    self.push_log(self.status_detail.clone());
                }
                self.endpoint_online = false;
                self.probe = None;
                self.last_probe = Instant::now() - Duration::from_secs(2);
                self.last_stats_refresh = Instant::now() - Duration::from_secs(2);
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
                self.push_log(format!("[launch failed] {error:#}"));
            }
        }
    }

    /// The download llama-server is running right now, if any.
    pub fn active_download(&self) -> Option<&Download> {
        self.download
            .as_ref()
            .filter(|download| download.is_active())
    }

    /// llama-server loading the weights means any download is over. The tick
    /// that follows reports the change, as it does for a download that the
    /// cache shows finishing.
    fn observe_startup_line(&mut self, line: &str) {
        if let Some(download) = self.download.as_mut()
            && cache::is_model_load_line(line)
        {
            download.finish();
        }
    }

    /// Prefer tok/s printed by llama-server itself — same source as the Logs tab.
    fn observe_throughput_line(&mut self, line: &str) {
        let parsed = parse_log_throughput(line);
        if parsed.generated_tokens_per_second.is_none() && parsed.prompt_tokens_per_second.is_none()
        {
            return;
        }
        let metrics = self
            .server_metrics
            .get_or_insert_with(ServerMetrics::default);
        if let Some(rate) = parsed.generated_tokens_per_second {
            self.live_generated_tps = Some(rate);
            metrics.generated_tokens_per_second = Some(rate);
        }
        if let Some(rate) = parsed.prompt_tokens_per_second {
            metrics.prompt_tokens_per_second = Some(rate);
        }
        self.touch_throughput();
    }

    pub fn stop(&mut self) {
        let Some(mut process) = self.process.take() else {
            return;
        };
        self.running_config = None;
        self.status = ServerStatus::Stopping;
        let stop_result = process.stop();
        let tail_logs = process.drain_logs().collect::<Vec<_>>();
        for event in tail_logs {
            let ServerEvent::Log(line) = event;
            self.push_log(line);
        }
        match stop_result {
            Ok(()) => {
                self.status = ServerStatus::Stopped;
                self.status_detail = "Stopped by user".into();
                self.push_log("llama-server stopped".into());
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
                self.push_log(format!("[stop failed] {error:#}"));
            }
        }
        self.endpoint_online = false;
        self.probe = None;
        self.download = None;
        self.process_usage = None;
        self.server_metrics = None;
        self.clear_live_throughput();
    }

    pub fn restart(&mut self) {
        if self.process.is_some() {
            self.stop();
        }
        self.start();
    }

    pub fn shutdown(&mut self) {
        if self.process.is_some() {
            self.stop();
        }
    }

    fn validate_for_launch(&self) -> std::result::Result<(), String> {
        let mut errors = self.config.validate();
        if !executable_exists(&self.config.server.executable) {
            errors.push(format!(
                "server executable was not found: {}",
                self.config.server.executable.display()
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    pub fn save(&mut self) {
        match self.config.save(&self.config_path) {
            Ok(()) => {
                self.status_detail = format!("Saved {}", self.config_path.display());
                self.push_log(self.status_detail.clone());
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
            }
        }
    }

    pub fn reset_to_defaults(&mut self) {
        let recent_models = self.config.recent_models.clone();
        self.config = Config::default();
        self.config.recent_models = recent_models;
        self.config.remember_model(self.config.model.source.clone());
        self.last_hf_model = DEFAULT_MODEL.into();
        self.last_local_model = PathBuf::new();
        self.missing_server_prompt = !executable_exists(&self.config.server.executable);
        self.sync_remote_model_size();
        self.status_detail = if self.process.is_some() {
            "Restored defaults; restart to apply, save to persist".into()
        } else {
            "Restored defaults; save to persist".into()
        };
        self.push_log("configuration reset to defaults".into());
    }

    pub fn apply_runtime_preset(&mut self, preset: RuntimePreset) -> Result<(), String> {
        self.apply_runtime_preset_with(preset, crate::system::likely_unified_memory())
    }

    fn apply_runtime_preset_with(
        &mut self,
        preset: RuntimePreset,
        unified_memory: bool,
    ) -> Result<(), String> {
        if unified_memory && preset.blocked_on_unified_memory() {
            return Err(
                "GPU + CPU spill is for discrete GPUs only. On unified memory it can exhaust the shared RAM pool and crash the system. Use Max RAM efficiency instead."
                    .into(),
            );
        }
        self.config.runtime = preset.runtime();
        self.status_detail = if self.process.is_some() {
            format!(
                "Applied {}; restart to apply, save to persist",
                preset.label()
            )
        } else {
            format!("Applied {}; save to persist", preset.label())
        };
        self.push_log(format!("applied runtime preset {}", preset.id()));
        Ok(())
    }

    /// Drop a saved GPU-spill profile on unified memory before it can be launched again.
    fn sanitize_unified_memory_presets(&mut self) {
        if !crate::system::likely_unified_memory() {
            return;
        }
        if RuntimePreset::matching(&self.config.runtime) != Some(RuntimePreset::GpuFit) {
            return;
        }
        self.config.runtime = RuntimePreset::LowRam.runtime();
        self.status_detail =
            "GPU + CPU spill was reset to Max RAM efficiency on unified memory (shared RAM can be exhausted)."
                .into();
        self.push_log("reset gpu_fit preset to max RAM efficiency on unified memory".into());
    }

    pub fn active_runtime_preset(&self) -> Option<RuntimePreset> {
        RuntimePreset::matching(&self.config.runtime)
    }

    pub fn set_field(&mut self, field: SettingField, raw: &str) -> std::result::Result<(), String> {
        self.apply_field_value(field, raw.trim())?;
        if field == SettingField::Model {
            self.remember_current_model();
        }
        if matches!(field, SettingField::Model | SettingField::SourceKind) {
            self.sync_remote_model_size();
        }
        self.mark_setting_changed(field);
        Ok(())
    }

    fn apply_field_value(
        &mut self,
        field: SettingField,
        raw: &str,
    ) -> std::result::Result<(), String> {
        let value = trim_wrapping_quotes(raw.trim());
        match field {
            SettingField::Model => {
                if value.is_empty() {
                    return Err(match &self.config.model.source {
                        ModelSource::HuggingFace(_) => "Repository cannot be empty.".into(),
                        ModelSource::Local(_) => "GGUF file path cannot be empty.".into(),
                    });
                }
                match &mut self.config.model.source {
                    ModelSource::HuggingFace(id) => {
                        *id = value.into();
                        self.last_hf_model = id.clone();
                    }
                    ModelSource::Local(path) => {
                        *path = PathBuf::from(value);
                        self.last_local_model = path.clone();
                    }
                }
            }
            SettingField::EstimatedSize => {
                return Err("Mapped file size is detected automatically.".into());
            }
            SettingField::Executable => {
                if value.is_empty() {
                    return Err("Enter llama-server or its full executable path.".into());
                }
                self.config.server.executable = value.into();
            }
            SettingField::Host => {
                if value.is_empty() {
                    return Err("Listen address cannot be empty.".into());
                }
                self.config.server.host = value.into();
            }
            SettingField::Port => {
                self.config.server.port = parse_bounded_u32(value, "port", 1, 65_535)? as u16;
            }
            SettingField::Context => {
                self.config.runtime.context_size = parse_token_count(value, 1, 1_048_576)?;
            }
            SettingField::Batch => {
                let batch = parse_bounded_u32(value, "batch size", 1, 4096)?;
                if batch < self.config.runtime.micro_batch_size {
                    return Err(format!(
                        "Batch must be at least the micro-batch size ({}).",
                        self.config.runtime.micro_batch_size
                    ));
                }
                self.config.runtime.batch_size = batch;
            }
            SettingField::MicroBatch => {
                self.config.runtime.micro_batch_size = parse_bounded_u32(
                    value,
                    "micro-batch size",
                    1,
                    self.config.runtime.batch_size,
                )?;
            }
            SettingField::Parallel => {
                self.config.runtime.parallel =
                    parse_bounded_u32(value, "parallel slots", 1, 64)? as u16;
            }
            SettingField::CacheRam => {
                self.config.runtime.cache_ram_mib =
                    parse_bounded_u32(value, "prompt cache RAM", 0, 131_072)?;
            }
            SettingField::Checkpoints => {
                self.config.runtime.context_checkpoints =
                    parse_bounded_u32(value, "context checkpoints", 0, 256)?;
            }
            SettingField::CacheTypeK => {
                self.config.runtime.cache_type_k = normalize_cache_type(value)?;
            }
            SettingField::CacheTypeV => {
                self.config.runtime.cache_type_v = normalize_cache_type(value)?;
            }
            SettingField::CpuOnly
            | SettingField::Mmap
            | SettingField::Fit
            | SettingField::Repack
            | SettingField::Warmup
            | SettingField::FlashAttn
            | SettingField::Mmproj
            | SettingField::Jinja => {
                let on = parse_bool(value)?;
                match field {
                    SettingField::CpuOnly => self.config.runtime.cpu_only = on,
                    SettingField::Mmap => self.config.runtime.mmap = on,
                    SettingField::Fit => self.config.runtime.fit = on,
                    SettingField::Repack => self.config.runtime.repack = on,
                    SettingField::Warmup => self.config.runtime.warmup = on,
                    SettingField::FlashAttn => self.config.runtime.flash_attn = on,
                    SettingField::Mmproj => self.config.runtime.multimodal_projector = on,
                    SettingField::Jinja => self.config.runtime.jinja = on,
                    _ => unreachable!(),
                }
            }
            SettingField::SourceKind => {
                let want_local = matches!(
                    value.to_ascii_lowercase().as_str(),
                    "local" | "local_gguf" | "gguf"
                );
                let is_local = matches!(self.config.model.source, ModelSource::Local(_));
                if want_local != is_local {
                    self.adjust_field(SettingField::SourceKind, 1);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn toggle_field(&mut self, field: SettingField) -> std::result::Result<(), String> {
        if !field.is_toggle() {
            return Err("This setting is not a toggle.".into());
        }
        self.adjust_field(field, 1);
        Ok(())
    }

    pub fn adjust_field(&mut self, field: SettingField, direction: i32) {
        let positive = direction > 0;
        match field {
            SettingField::SourceKind => {
                self.config.model.source = match &self.config.model.source {
                    ModelSource::HuggingFace(value) => {
                        self.last_hf_model = value.clone();
                        ModelSource::Local(self.last_local_model.clone())
                    }
                    ModelSource::Local(value) => {
                        self.last_local_model = value.clone();
                        ModelSource::HuggingFace(self.last_hf_model.clone())
                    }
                };
            }
            SettingField::Model | SettingField::Executable | SettingField::Host => {
                return;
            }
            SettingField::EstimatedSize => {
                return;
            }
            SettingField::Port => {
                self.config.server.port =
                    adjust_u32(self.config.server.port.into(), direction, 1, 65_535) as u16;
            }
            SettingField::Context => {
                self.config.runtime.context_size = adjust_u32(
                    self.config.runtime.context_size,
                    direction * 1024,
                    1024,
                    1_048_576,
                );
            }
            SettingField::Batch => {
                self.config.runtime.batch_size =
                    adjust_power_of_two(self.config.runtime.batch_size, positive, 1, 4096);
                self.config.runtime.micro_batch_size = self
                    .config
                    .runtime
                    .micro_batch_size
                    .min(self.config.runtime.batch_size);
            }
            SettingField::MicroBatch => {
                self.config.runtime.micro_batch_size = adjust_power_of_two(
                    self.config.runtime.micro_batch_size,
                    positive,
                    1,
                    self.config.runtime.batch_size,
                );
            }
            SettingField::Parallel => {
                self.config.runtime.parallel =
                    adjust_u32(self.config.runtime.parallel.into(), direction, 1, 64) as u16;
            }
            SettingField::CpuOnly => self.config.runtime.cpu_only = !self.config.runtime.cpu_only,
            SettingField::Mmap => self.config.runtime.mmap = !self.config.runtime.mmap,
            SettingField::Fit => self.config.runtime.fit = !self.config.runtime.fit,
            SettingField::Repack => self.config.runtime.repack = !self.config.runtime.repack,
            SettingField::Warmup => self.config.runtime.warmup = !self.config.runtime.warmup,
            SettingField::FlashAttn => {
                self.config.runtime.flash_attn = !self.config.runtime.flash_attn
            }
            SettingField::CacheTypeK | SettingField::CacheTypeV => {
                let current = match field {
                    SettingField::CacheTypeK => self.config.runtime.cache_type_k.as_str(),
                    _ => self.config.runtime.cache_type_v.as_str(),
                };
                let index = CACHE_TYPES
                    .iter()
                    .position(|item| *item == current)
                    .unwrap_or(0);
                let next = if positive {
                    CACHE_TYPES[(index + 1) % CACHE_TYPES.len()]
                } else {
                    CACHE_TYPES[(index + CACHE_TYPES.len() - 1) % CACHE_TYPES.len()]
                };
                match field {
                    SettingField::CacheTypeK => {
                        self.config.runtime.cache_type_k = (*next).into();
                    }
                    _ => self.config.runtime.cache_type_v = (*next).into(),
                }
            }
            SettingField::CacheRam => {
                self.config.runtime.cache_ram_mib = adjust_u32(
                    self.config.runtime.cache_ram_mib,
                    direction * 256,
                    0,
                    131_072,
                );
            }
            SettingField::Checkpoints => {
                self.config.runtime.context_checkpoints =
                    adjust_u32(self.config.runtime.context_checkpoints, direction, 0, 256);
            }
            SettingField::Mmproj => {
                self.config.runtime.multimodal_projector =
                    !self.config.runtime.multimodal_projector;
            }
            SettingField::Jinja => self.config.runtime.jinja = !self.config.runtime.jinja,
        }
        if field == SettingField::SourceKind {
            self.sync_remote_model_size();
        }
        self.mark_setting_changed(field);
    }

    fn mark_setting_changed(&mut self, field: SettingField) {
        self.status_detail = if self.process.is_some() {
            format!("Changed {}; restart to apply", self.setting_label(field))
        } else {
            format!("Changed {}; save to persist", self.setting_label(field))
        };
    }

    fn remember_current_model(&mut self) {
        let source = self.config.model.source.clone();
        self.config.remember_model(source);
    }

    /// Recent models first, then autodiscovered cache entries not already listed.
    pub fn model_picker_entries(&self) -> Vec<ModelPickerEntry> {
        let mut entries = Vec::new();
        for source in &self.config.recent_models {
            entries.push(ModelPickerEntry {
                group: ModelPickerGroup::Recent,
                source: source.clone(),
            });
        }
        for discovered in &self.discovered_models {
            let source = match &discovered.source {
                cache::DiscoveredSource::HuggingFace(repo) => {
                    ModelSource::HuggingFace(repo.clone())
                }
                cache::DiscoveredSource::Local(path) => ModelSource::Local(path.clone()),
            };
            if entries.iter().any(|entry| entry.source == source) {
                continue;
            }
            entries.push(ModelPickerEntry {
                group: ModelPickerGroup::Cached,
                source,
            });
        }
        entries
    }

    pub fn select_recent_model(&mut self, index: usize) -> std::result::Result<(), String> {
        let selected = self
            .model_picker_entries()
            .get(index)
            .map(|entry| entry.source.clone())
            .ok_or_else(|| "That model is no longer available.".to_string())?;
        match &selected {
            ModelSource::HuggingFace(id) => self.last_hf_model = id.clone(),
            ModelSource::Local(path) => self.last_local_model = path.clone(),
        }
        self.config.model.source = selected;
        self.sync_remote_model_size();
        self.mark_setting_changed(SettingField::Model);
        Ok(())
    }

    fn poll_discovered_models(&mut self) {
        if let Some(pending) = self.pending_discover.as_ref()
            && let Some(models) = pending.take()
        {
            self.discovered_models = models;
            self.pending_discover = None;
            self.last_discover = Instant::now();
        }
        if self.pending_discover.is_none() && self.last_discover.elapsed() >= DISCOVER_INTERVAL {
            self.pending_discover = Some(cache::discover_models_async());
            self.last_discover = Instant::now();
        }
    }

    fn sync_remote_model_size(&mut self) {
        let ModelSource::HuggingFace(repo) = &self.config.model.source else {
            self.remote_model_size = None;
            return;
        };
        if self.remote_model_size.as_ref().is_some_and(|remote| {
            remote.repo == *repo && (remote.bytes.is_some() || remote.listing.is_some())
        }) {
            return;
        }
        self.remote_model_size = Some(RemoteModelSize {
            repo: repo.clone(),
            listing: Some(hub::list_files_async(repo)),
            bytes: None,
            error: None,
        });
    }

    fn poll_remote_model_size(&mut self) {
        let Some(remote) = self.remote_model_size.as_mut() else {
            return;
        };
        let Some(result) = remote.listing.as_mut().and_then(hub::PendingListing::take) else {
            return;
        };
        remote.listing = None;
        let repo = remote.repo.clone();
        let resolved = match result {
            Ok(files) => match hub::primary_gguf_bytes(&files, &repo) {
                Some(bytes) => Ok(bytes),
                None => Err("No GGUF files found in the repository listing.".into()),
            },
            Err(error) => Err(error),
        };
        match resolved {
            Ok(bytes) => {
                if let Some(remote) = self.remote_model_size.as_mut() {
                    remote.bytes = Some(bytes);
                    remote.error = None;
                }
                self.config.model.estimated_size_gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            }
            Err(error) => {
                if let Some(remote) = self.remote_model_size.as_mut() {
                    remote.bytes = None;
                    remote.error = Some(error);
                }
            }
        }
    }

    pub fn copy_endpoint(&mut self) -> String {
        let endpoint = self.displayed_config().api_endpoint();
        self.copy_text("API endpoint", &endpoint);
        endpoint
    }

    pub fn copy_command(&mut self) -> String {
        let command = CommandSpec::from_config(self.displayed_config()).display();
        self.copy_text("launch command", &command);
        command
    }

    fn copy_text(&mut self, label: &str, text: &str) {
        self.status_detail = match copy_to_clipboard(text) {
            Ok(()) => format!("Copied {label}"),
            Err(error) => format!("Could not copy {label}: {error}"),
        };
    }

    fn push_log(&mut self, line: String) {
        if self.logs.len() == MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(line);
    }
}

/// Watch a Hugging Face launch for a download.
///
/// The cache is inspected in the background, so the indicator can correct
/// itself quickly for an already-cached model without blocking the interface.
/// llama-server may also re-fetch a model that looks cached, which the watch
/// picks up from growth on disk. Local models are never downloaded.
fn watch_download(config: &Config) -> Option<Download> {
    let ModelSource::HuggingFace(repo) = &config.model.source else {
        return None;
    };
    Some(Download::new(
        repo,
        config.model.estimated_size_gib,
        Some(hub::list_files_async(repo)),
    ))
}

fn on_off(value: bool) -> String {
    if value { "on" } else { "off" }.into()
}

fn parse_bool(value: &str) -> std::result::Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Ok(true),
        "false" | "0" | "off" | "no" => Ok(false),
        _ => Err("Enter true or false.".into()),
    }
}

fn adjust_u32(value: u32, delta: i32, min: u32, max: u32) -> u32 {
    if delta >= 0 {
        value.saturating_add(delta as u32).min(max)
    } else {
        value.saturating_sub(delta.unsigned_abs()).max(min)
    }
}

fn adjust_power_of_two(value: u32, increase: bool, min: u32, max: u32) -> u32 {
    if increase {
        value.saturating_mul(2).min(max)
    } else {
        (value / 2).max(min)
    }
}

fn parse_bounded_u32(
    value: &str,
    label: &str,
    min: u32,
    max: u32,
) -> std::result::Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("Enter {label} as a whole number."))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{label} must be between {min} and {max}."));
    }
    Ok(parsed)
}

fn parse_token_count(value: &str, min: u32, max: u32) -> std::result::Result<u32, String> {
    let lower = value.trim().to_ascii_lowercase();
    let (number, multiplier) = if let Some(number) = lower.strip_suffix('k') {
        (number, 1024.0)
    } else if let Some(number) = lower.strip_suffix('m') {
        (number, 1024.0 * 1024.0)
    } else {
        (lower.as_str(), 1.0)
    };
    let scaled = number
        .parse::<f64>()
        .map_err(|_| "Enter a token count such as 8192 or 8k.".to_string())?
        * multiplier;
    if !scaled.is_finite() || scaled.fract() != 0.0 {
        return Err("Token count must resolve to a whole number.".into());
    }
    if scaled < min as f64 || scaled > max as f64 {
        return Err(format!("Token count must be between {min} and {max}."));
    }
    Ok(scaled as u32)
}

fn trim_wrapping_quotes(value: &str) -> &str {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let quoted = (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'');
        if quoted {
            return &value[1..value.len() - 1];
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_adjustment_is_bounded() {
        assert_eq!(adjust_power_of_two(8, true, 1, 16), 16);
        assert_eq!(adjust_power_of_two(16, true, 1, 16), 16);
        assert_eq!(adjust_power_of_two(1, false, 1, 16), 1);
    }

    #[test]
    fn app_starts_stopped() {
        let app = App::new(Config::default(), "test.toml".into());
        assert_eq!(app.status, ServerStatus::Stopped);
    }

    #[test]
    fn recent_model_can_be_selected_by_index() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let current = app.config.model.source.clone();
        let local = ModelSource::Local("models/small.gguf".into());
        app.config.recent_models = vec![current, local.clone()];
        app.select_recent_model(1).unwrap();
        assert_eq!(app.config.model.source, local);
        assert!(app.select_recent_model(9).is_err());
    }

    #[test]
    fn picker_appends_autodiscovered_models_after_recent() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let recent = app.config.model.source.clone();
        app.config.recent_models = vec![recent.clone()];
        app.discovered_models = vec![
            cache::DiscoveredModel {
                source: cache::DiscoveredSource::HuggingFace("owner/cached-GGUF".into()),
                bytes: 100,
            },
            cache::DiscoveredModel {
                source: cache::DiscoveredSource::HuggingFace(match &recent {
                    ModelSource::HuggingFace(id) => id.clone(),
                    ModelSource::Local(_) => "other/unused".into(),
                }),
                bytes: 50,
            },
        ];
        let entries = app.model_picker_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].group, ModelPickerGroup::Recent);
        assert_eq!(entries[0].source, recent);
        assert_eq!(entries[1].group, ModelPickerGroup::Cached);
        assert_eq!(
            entries[1].source,
            ModelSource::HuggingFace("owner/cached-GGUF".into())
        );
        app.select_recent_model(1).unwrap();
        assert_eq!(
            app.config.model.source,
            ModelSource::HuggingFace("owner/cached-GGUF".into())
        );
    }

    #[test]
    fn missing_server_prompts_and_opens_configuration() {
        let mut config = Config::default();
        config.server.executable = "__tinyinference_missing_server__".into();
        let mut app = App::new(config, "test.toml".into());
        assert!(app.should_prompt_for_server());
        app.open_server_configuration();
        assert!(!app.should_prompt_for_server());
        assert!(app.status_detail.contains("executable"));
    }

    #[test]
    fn detected_server_does_not_prompt() {
        let mut config = Config::default();
        config.server.executable = std::env::current_exe().unwrap();
        let app = App::new(config, "test.toml".into());
        assert!(!app.should_prompt_for_server());
    }

    #[test]
    fn running_configuration_stays_stable_until_restart() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.running_config = Some(app.config.clone());
        app.config.server.port = 9090;
        assert_eq!(app.displayed_config().server.port, 8080);
        assert!(app.has_pending_changes());
    }

    #[test]
    fn exact_port_can_be_entered() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.dismiss_server_prompt();
        app.set_field(SettingField::Port, "4242").unwrap();
        assert_eq!(app.config.server.port, 4242);
    }

    #[test]
    fn reset_to_defaults_restores_settings_and_keeps_recent_models() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let local = ModelSource::Local("models/small.gguf".into());
        app.config.recent_models.push(local.clone());
        app.set_field(SettingField::Port, "4242").unwrap();
        app.set_field(SettingField::Context, "4096").unwrap();
        app.reset_to_defaults();
        assert_eq!(app.config.server.port, Config::default().server.port);
        assert_eq!(
            app.config.runtime.context_size,
            Config::default().runtime.context_size
        );
        assert!(app.config.recent_models.contains(&local));
    }

    #[test]
    fn apply_runtime_preset_only_changes_runtime() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.set_field(SettingField::Port, "4242").unwrap();
        assert_eq!(app.active_runtime_preset(), Some(RuntimePreset::LowRam));
        app.apply_runtime_preset_with(RuntimePreset::GpuFit, false)
            .unwrap();
        assert_eq!(app.active_runtime_preset(), Some(RuntimePreset::GpuFit));
        assert!(!app.config.runtime.cpu_only);
        assert!(app.config.runtime.fit);
        assert_eq!(app.config.server.port, 4242);
        app.set_field(SettingField::Context, "4096").unwrap();
        assert_eq!(app.active_runtime_preset(), None);
    }

    #[test]
    fn gpu_fit_preset_blocked_on_unified_memory() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let error = app
            .apply_runtime_preset_with(RuntimePreset::GpuFit, true)
            .unwrap_err();
        assert!(error.contains("unified memory"));
        assert_eq!(app.active_runtime_preset(), Some(RuntimePreset::LowRam));
    }

    #[test]
    fn context_accepts_k_suffix() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.set_field(SettingField::Context, "16k").unwrap();
        assert_eq!(app.config.runtime.context_size, 16 * 1024);
    }

    #[test]
    fn invalid_field_value_is_rejected() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let error = app.set_field(SettingField::Port, "70000").unwrap_err();
        assert!(error.contains("between"));
        assert_eq!(app.config.server.port, 8080);
    }

    #[test]
    fn switching_to_local_model_requests_a_real_path() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.adjust_field(SettingField::SourceKind, 1);
        assert!(matches!(
            app.config.model.source,
            ModelSource::Local(ref path) if path.as_os_str().is_empty()
        ));
        assert_eq!(
            app.setting_value(SettingField::Model),
            "<enter path to .gguf>"
        );
        assert_eq!(app.setting_label(SettingField::Model), "GGUF file path");
        assert_eq!(
            app.setting_value(SettingField::EstimatedSize),
            "auto from .gguf"
        );
        assert!(!app.setting_is_editable(SettingField::EstimatedSize));
    }

    #[test]
    fn hugging_face_mapped_size_comes_from_the_listing() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.remote_model_size = Some(RemoteModelSize {
            repo: DEFAULT_MODEL.into(),
            listing: None,
            bytes: Some(63_454_123_008),
            error: None,
        });
        let size = app.mapped_model_gib().unwrap();
        assert!((size - 59.1).abs() < 0.05);
        assert!(
            app.setting_value(SettingField::EstimatedSize)
                .contains("from Hugging Face")
        );
        assert!(!app.setting_is_editable(SettingField::EstimatedSize));
    }

    #[test]
    fn model_source_switch_preserves_both_values() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.adjust_field(SettingField::SourceKind, 1);
        app.set_field(SettingField::Model, r"C:\models\custom.gguf")
            .unwrap();
        app.adjust_field(SettingField::SourceKind, 1);
        assert_eq!(
            app.setting_value(SettingField::Model),
            "ggml-org/gpt-oss-120b-GGUF"
        );
        app.adjust_field(SettingField::SourceKind, 1);
        assert_eq!(
            app.setting_value(SettingField::Model),
            r"C:\models\custom.gguf"
        );
    }

    fn remote(path: &str, oid: &str, size: u64) -> hub::RemoteFile {
        hub::RemoteFile {
            path: path.into(),
            oid: oid.into(),
            size,
        }
    }

    fn download_of(files: Vec<hub::RemoteFile>) -> Download {
        let mut download = Download::new("owner/model", 59.1, None);
        download.files = files;
        download
    }

    #[test]
    fn only_hugging_face_launches_are_watched_for_a_download() {
        let mut config = Config::default();
        config.model.source = ModelSource::Local("model.gguf".into());
        assert!(watch_download(&config).is_none());
    }

    #[test]
    fn the_file_being_written_gives_the_real_name_and_size() {
        let mut download = download_of(vec![
            remote("model-Q4.gguf", "aaa", 400),
            remote("model-Q8.gguf", "bbb", 800),
        ]);
        download.resolve_target(&cache::CacheScan {
            blobs: vec![cache::CachedBlob {
                oid: "bbb".into(),
                bytes: 200,
                in_flight: true,
            }],
            flat_bytes: 0,
        });
        assert_eq!(download.file.as_deref(), Some("model-Q8.gguf"));
        assert_eq!(download.total, Some(800));
        assert_eq!(download.oids, ["bbb"]);
    }

    #[test]
    fn metadata_blobs_do_not_become_download_targets() {
        let mut download = download_of(vec![remote(".gitattributes", "meta", 1_000)]);
        download.resolve_target(&cache::CacheScan {
            blobs: vec![cache::CachedBlob {
                oid: "meta".into(),
                bytes: 1_000,
                in_flight: true,
            }],
            flat_bytes: 0,
        });
        assert_eq!(download.file, None);
        assert_eq!(download.total, None);
    }

    #[test]
    fn a_split_model_is_measured_across_every_shard() {
        let mut download = download_of(vec![
            remote("m-00001-of-00002.gguf", "aaa", 10),
            remote("m-00002-of-00002.gguf", "bbb", 20),
            remote("unrelated.gguf", "ccc", 5000),
        ]);
        download.resolve_target(&cache::CacheScan {
            blobs: vec![cache::CachedBlob {
                oid: "aaa".into(),
                bytes: 10,
                in_flight: true,
            }],
            flat_bytes: 0,
        });
        assert_eq!(download.total, Some(30));
        assert_eq!(download.oids, ["aaa", "bbb"]);
    }

    #[test]
    fn progress_rate_and_estimate_come_from_the_measured_bytes() {
        let mut download = download_of(Vec::new());
        download.total = Some(1_000);
        download.downloaded = 250;
        let now = Instant::now();
        download.samples = VecDeque::from([(now - Duration::from_secs(4), 50), (now, 250)]);
        assert_eq!(download.fraction(), Some(0.25));
        assert_eq!(download.rate(), Some(50.0));
        assert_eq!(download.eta(), Some(Duration::from_secs(15)));
    }

    #[test]
    fn a_rate_needs_more_than_one_sample() {
        let mut download = download_of(Vec::new());
        download.samples = VecDeque::from([(Instant::now(), 10)]);
        assert_eq!(download.rate(), None);
        assert_eq!(download.eta(), None);
        assert_eq!(download.fraction(), None);
    }

    #[test]
    fn loading_the_weights_ends_the_download() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.download = Some(Download::new(DEFAULT_MODEL, 59.1, None));
        assert!(app.active_download().is_some());

        app.observe_startup_line("srv    load_model: loading model");
        assert!(app.active_download().is_none());
    }

    #[test]
    fn quoted_executable_path_is_unwrapped() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.set_field(
            SettingField::Executable,
            r#""C:\Program Files\llama\llama-server.exe""#,
        )
        .unwrap();
        assert_eq!(
            app.config.server.executable,
            PathBuf::from(r"C:\Program Files\llama\llama-server.exe")
        );
    }
}
