use std::{
    collections::VecDeque,
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{
    cache,
    config::{
        CACHE_TYPES, Config, DEFAULT_MODEL, ModelSource, RuntimePreset, normalize_cache_type,
    },
    fetch::{self, FetchEvent},
    hub,
    instance::ManagedServer,
    network::{
        self, CatalogCache, DiscoveredPeer, HealthCache, InferenceMode, ListenCandidate,
        ListenScope, NetworkDiscovery, RemoteHealth, ShareUrl,
    },
    server::{
        CommandSpec, PendingProbe, PendingThinkingProbe, ProbeResult, ServerEvent, ServerMetrics,
        ServerProcess, SlotsSnapshot, parse_log_throughput,
        probe_async, thinking_support_async,
    },
    system::{Machine, ProcessMonitor, ProcessUsage, copy_to_clipboard, executable_exists},
};

/// Id used for the primary (first) managed llama-server slot.
pub const PRIMARY_SERVER_ID: &str = "main";

const MAX_LOG_LINES: usize = 2_000;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct NetworkUpdate {
    pub expose: Option<bool>,
    pub listen_scope: Option<String>,
    pub listen_host: Option<String>,
    pub regenerate_token: Option<bool>,
    pub inference_mode: Option<String>,
    pub remote_base: Option<String>,
    pub remote_token: Option<String>,
    pub device_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NetworkUpdateResult {
    pub restart_required: bool,
    /// Full secret shown once after create / regenerate (legacy field name).
    pub access_token: Option<String>,
    pub revealed_api_key_id: Option<String>,
}

impl NetworkUpdateResult {
    fn from_restart(restart_required: bool) -> Self {
        Self {
            restart_required,
            access_token: None,
            revealed_api_key_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerSummary {
    pub id: String,
    pub model: String,
    pub status: ServerStatus,
    pub status_label: &'static str,
    pub endpoint: String,
    pub api_endpoint: String,
    pub port: u16,
    pub ready: bool,
    pub pid: Option<u32>,
    pub thinking_supported: bool,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct ServerLookup<'a> {
    pub id: &'a str,
    pub config: &'a Config,
    pub ready: bool,
    pub thinking_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPickerEntry {
    pub source: ModelSource,
    pub recent: bool,
    pub on_disk: bool,
    pub bytes: u64,
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
    pub(crate) fn poll(&mut self) {
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

/// Managed library download started from the Models tab (not llama-server).
pub struct LibraryFetch {
    pub repo: String,
    pub file: Option<String>,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub error: Option<String>,
    pub done: bool,
    pending: Option<fetch::PendingFetch>,
    samples: VecDeque<(Instant, u64)>,
}

impl LibraryFetch {
    fn new(repo: &str, pending: fetch::PendingFetch) -> Self {
        Self {
            repo: repo.to_string(),
            file: None,
            downloaded: 0,
            total: None,
            error: None,
            done: false,
            pending: Some(pending),
            samples: VecDeque::from([(Instant::now(), 0)]),
        }
    }

    pub fn is_active(&self) -> bool {
        !self.done && self.error.is_none()
    }

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

    pub fn eta(&self) -> Option<Duration> {
        let remaining = self.total?.saturating_sub(self.downloaded);
        let rate = self.rate().filter(|rate| *rate > 1.0)?;
        Some(Duration::from_secs_f64(remaining as f64 / rate))
    }

    fn poll(&mut self) {
        let mut finished = false;
        let mut failed = None;
        loop {
            let event = self.pending.as_ref().and_then(fetch::PendingFetch::take);
            let Some(event) = event else {
                break;
            };
            match event {
                FetchEvent::Started { file, total, .. } => {
                    self.file = Some(file);
                    self.total = Some(total);
                }
                FetchEvent::Progress {
                    downloaded,
                    total,
                    file,
                } => {
                    self.file = Some(file);
                    self.total = Some(total);
                    self.downloaded = downloaded;
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
                FetchEvent::Finished { bytes } => {
                    self.downloaded = bytes;
                    if self.total.is_none() {
                        self.total = Some(bytes);
                    }
                    finished = true;
                }
                FetchEvent::Error(error) => {
                    failed = Some(error);
                }
            }
        }
        if let Some(error) = failed {
            self.error = Some(error);
            self.done = true;
            self.pending = None;
        } else if finished {
            self.done = true;
            self.pending = None;
        }
    }

    fn cancel(&mut self) {
        if let Some(pending) = self.pending.as_ref() {
            pending.cancel();
        }
    }
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host.trim(), "127.0.0.1" | "localhost" | "::1")
}

fn configs_differ_ignoring_instance_port(running: &Config, desired: &Config) -> bool {
    let mut running = running.clone();
    running.server.port = desired.server.port;
    &running != desired
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
    /// Models-tab managed download into the local hub cache.
    pub library_fetch: Option<LibraryFetch>,
    /// Raises the desktop window, when there is one. Installed by `desktop::run`
    /// so a second launch can surface the running instance instead of starting
    /// a rival server. Boxed rather than typed so `app` stays windowing-agnostic.
    focus_hook: Option<Box<dyn Fn() + Send + Sync>>,
    /// Address the HTTP server actually bound at process start.
    listen_addr: Option<SocketAddr>,
    discovery: NetworkDiscovery,
    remote_health: HealthCache,
    remote_catalog: CatalogCache,
    /// Whether the loaded model supports controllable thinking / reasoning effort.
    thinking_supported: Option<bool>,
    thinking_probe_for: Option<String>,
    pending_thinking_probe: Option<PendingThinkingProbe>,
    /// Additional llama-server processes beyond the primary slot (`process`).
    pub extra_servers: Vec<ManagedServer>,
    /// Chat / dashboard focus: [`PRIMARY_SERVER_ID`] or an extra server id.
    pub active_server_id: String,
    next_server_seq: u64,
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
        let before_migrate = config.clone();
        config.migrate_network_expose_to_llama();
        config.ui.normalize_fonts();
        if config != before_migrate {
            let _ = config.save(&config_path);
        }
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
            listen_addr: None,
            discovery: NetworkDiscovery::new(),
            remote_health: HealthCache::default(),
            remote_catalog: CatalogCache::default(),
            thinking_supported: None,
            thinking_probe_for: None,
            pending_thinking_probe: None,
            last_throughput_at: None,
            discovered_models: Vec::new(),
            pending_discover: Some(cache::discover_models_async()),
            last_discover: Instant::now(),
            library_fetch: None,
            focus_hook: None,
            extra_servers: Vec::new(),
            active_server_id: PRIMARY_SERVER_ID.into(),
            next_server_seq: 1,
        };
        app.sanitize_unified_memory_presets();
        app.sync_remote_model_size();
        let _ = app.sync_share_tls();
        app
    }

    pub fn command(&self) -> CommandSpec {
        CommandSpec::from_config(&self.config)
    }

    /// Register the callback that raises the desktop window.
    pub fn set_focus_hook(&mut self, hook: Box<dyn Fn() + Send + Sync>) {
        self.focus_hook = Some(hook);
    }

    /// Ask the desktop window to come forward. Returns false when this instance
    /// has no window (headless), which is not an error — the caller still knows
    /// an instance is alive and can print its address instead.
    pub fn request_focus(&self) -> bool {
        match &self.focus_hook {
            Some(hook) => {
                hook();
                true
            }
            None => false,
        }
    }

    pub fn set_listen_addr(&mut self, addr: SocketAddr) {
        self.listen_addr = Some(addr);
        self.sync_mdns_advertise();
    }

    pub fn listen_addr(&self) -> Option<SocketAddr> {
        self.listen_addr
    }

    pub fn skill_store(&self) -> crate::skills::SkillStore {
        crate::skills::SkillStore::new(&self.config_path)
    }

    pub fn list_user_skills(&self) -> Result<Vec<crate::skills::UserSkill>, String> {
        self.skill_store()
            .list()
            .map_err(|error| error.to_string())
    }

    pub fn enabled_user_skills(&self) -> Vec<crate::skills::UserSkill> {
        self.skill_store().enabled_skills().unwrap_or_default()
    }

    pub fn create_user_skill(
        &self,
        upsert: crate::skills::SkillUpsert,
    ) -> Result<crate::skills::UserSkill, String> {
        self.skill_store()
            .create(upsert)
            .map_err(|error| error.to_string())
    }

    pub fn import_user_skill(
        &self,
        filename: Option<&str>,
        content: &str,
    ) -> Result<crate::skills::UserSkill, String> {
        self.skill_store()
            .import_markdown(filename, content)
            .map_err(|error| error.to_string())
    }

    pub fn update_user_skill(
        &self,
        id: &str,
        upsert: crate::skills::SkillUpsert,
    ) -> Result<crate::skills::UserSkill, String> {
        self.skill_store()
            .update(id, upsert)
            .map_err(|error| error.to_string())
    }

    pub fn delete_user_skill(&self, id: &str) -> Result<(), String> {
        self.skill_store()
            .delete(id)
            .map_err(|error| error.to_string())
    }

    /// True when Share is on and the desired llama bind is beyond loopback.
    pub fn listening_exposed(&self) -> bool {
        if !self.config.network.expose {
            return false;
        }
        !is_loopback_host(&self.config.effective_host())
    }

    /// Ports of managed servers that are up while Share is actually bound off-loopback.
    pub fn shareable_running_ports(&self) -> Vec<u16> {
        if !self.listening_exposed() {
            return Vec::new();
        }
        let mut ports: Vec<u16> = self
            .server_summaries()
            .into_iter()
            .filter(|s| {
                s.ready
                    || s.status == ServerStatus::Starting
                    || s.status == ServerStatus::Downloading
            })
            .map(|s| s.port)
            .collect();
        ports.sort_unstable();
        ports.dedup();
        ports
    }

    /// Running llama needs a restart to pick up a new share bind / API keys.
    pub fn network_restart_required(&self) -> bool {
        let mut configs = Vec::new();
        if let Some(config) = &self.running_config {
            configs.push(config);
        }
        for server in &self.extra_servers {
            if server.is_running() {
                configs.push(&server.running_config);
            }
        }
        configs.iter().any(|config| self.keys_or_host_diverged(config))
    }

    fn keys_or_host_diverged(&self, running: &Config) -> bool {
        if !self.config.network.expose {
            let still_shared = !is_loopback_host(&running.effective_host());
            return still_shared || !running.llama_api_keys().is_empty();
        }
        let Ok(desired_host) = self.config.network.resolve_listen_host() else {
            return true;
        };
        let mut desired = self.config.llama_api_keys();
        let mut actual = running.llama_api_keys();
        desired.sort();
        actual.sort();
        running.effective_host() != desired_host || desired != actual
    }

    pub fn sync_mdns_advertise(&mut self) {
        // Advertise only while a shared llama is actually running.
        let ports = self.shareable_running_ports();
        let advertise = self.config.network.should_advertise_mdns() && !ports.is_empty();
        self.discovery.sync_advertise(
            advertise,
            &self.config.network.resolved_device_name(),
            &ports,
        );
        if let Some(error) = self.discovery.last_error() {
            let line = format!("[network] {error}");
            if self.logs.back() != Some(&line) {
                self.push_log(line);
            }
        }
    }

    pub fn mdns_error(&self) -> Option<String> {
        self.discovery.last_error()
    }

    pub fn discovery_advertising(&self) -> bool {
        self.discovery.is_advertising()
    }

    pub fn discovery_hint(&self) -> String {
        self.discovery.discovery_hint()
    }

    pub fn discovered_peers(&self) -> Vec<DiscoveredPeer> {
        let own_fullname = self.discovery.advertised_fullname();
        let (lan, ts) = network::shareable_ipv4_addrs();
        let local_ips: std::collections::HashSet<String> = lan
            .into_iter()
            .chain(ts)
            .map(|ip| ip.to_string())
            .collect();
        self.discovery
            .discovered_peers()
            .into_iter()
            .filter(|peer| {
                if own_fullname.as_deref() == Some(peer.fullname.as_str()) {
                    return false;
                }
                !local_ips.contains(&peer.host)
            })
            .collect()
    }

    pub fn network_share_urls(&self) -> Vec<ShareUrl> {
        let network = &self.config.network;
        if !network.expose {
            return Vec::new();
        }
        // Only advertise hosts/ports that match the live fail-closed bind.
        let bind_host = self.config.effective_host();
        if is_loopback_host(&bind_host) {
            return Vec::new();
        }
        let ports = self.shareable_running_ports();
        if ports.is_empty() {
            return Vec::new();
        }
        let scheme = self.config.scheme();

        let mut urls = Vec::new();
        for port in ports {
            match network.listen_scope {
                ListenScope::All => {
                    urls.extend(network::lan_share_urls(port, scheme));
                    urls.extend(network::tailscale_urls(port, scheme));
                }
                ListenScope::Tailscale => urls.extend(network::tailscale_urls(port, scheme)),
                ListenScope::Custom => {
                    if let Ok(ip) = bind_host.parse::<std::net::Ipv4Addr>() {
                        let kind = if network::is_tailscale_cg_nat(ip) {
                            "tailscale"
                        } else {
                            "custom"
                        };
                        let label = if kind == "tailscale" {
                            format!("Tailscale ({ip})")
                        } else {
                            format!("Custom ({ip})")
                        };
                        urls.push(network::share_url_for(kind, label, ip, port, scheme));
                    } else {
                        urls.push(network::share_url_host(
                            "custom",
                            format!("Custom ({bind_host})"),
                            &bind_host,
                            port,
                            scheme,
                        ));
                    }
                }
            }
        }
        urls
    }

    /// Ensure self-signed TLS material exists when Share is on; clear when off.
    pub fn sync_share_tls(&mut self) -> Result<(), String> {
        if !self.config.network.expose {
            self.config.set_share_tls(None);
            return Ok(());
        }
        let config_dir = self
            .config_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let mut extra_ips = Vec::new();
        if let Ok(host) = self.config.network.resolve_listen_host()
            && let Ok(ip) = host.parse::<std::net::IpAddr>()
        {
            extra_ips.push(ip);
        }
        let paths = crate::tls::ensure_self_signed(config_dir, &extra_ips)
            .map_err(|error| format!("could not prepare share TLS certificate: {error:#}"))?;
        self.config
            .set_share_tls(Some((paths.cert_file, paths.key_file)));
        Ok(())
    }

    pub fn listen_candidates(&self) -> Vec<ListenCandidate> {
        network::listen_candidates()
    }

    pub fn listen_bind_error(&self) -> Option<String> {
        if !self.config.network.expose {
            return None;
        }
        self.config.network.resolve_listen_host().err()
    }

    pub fn remote_health(&self) -> Option<RemoteHealth> {
        let remote = self.config.network.active_remote()?;
        let base = remote.base.trim();
        if base.is_empty() {
            return None;
        }
        Some(self.remote_health.probe(base, remote.token.trim()))
    }

    /// Cached peer health only — safe for high-frequency `/api/state` polls.
    pub fn remote_health_cached(&self) -> Option<RemoteHealth> {
        let remote = self.config.network.active_remote()?;
        let base = remote.base.trim();
        if base.is_empty() {
            return None;
        }
        self.remote_health.peek(base, remote.token.trim())
    }

    /// Probe (cached) health for every saved linked LLM — used by the manager UI.
    pub fn remote_health_for(&self, base: &str, token: &str) -> RemoteHealth {
        self.remote_health.probe(base.trim(), token.trim())
    }

    /// Models available on the active linked host (primary + sibling ports).
    pub fn remote_model_catalog(&self) -> Vec<network::RemoteModelOption> {
        let Some(remote) = self.config.network.active_remote() else {
            return Vec::new();
        };
        let base = remote.base.trim();
        if base.is_empty() {
            return Vec::new();
        }
        let mut extra_ports = Vec::new();
        if let Some((_, host, _)) = network::split_openai_base(base) {
            for peer in self.discovered_peers() {
                if peer.host.eq_ignore_ascii_case(&host) {
                    extra_ports.extend(peer.ports.iter().copied());
                    extra_ports.push(peer.port);
                }
            }
        }
        self.remote_catalog
            .probe(base, remote.token.trim(), &extra_ports)
    }

    /// Cached catalog only — safe for high-frequency polls after a warm probe.
    pub fn remote_model_catalog_cached(&self) -> Vec<network::RemoteModelOption> {
        let Some(remote) = self.config.network.active_remote() else {
            return Vec::new();
        };
        self.remote_catalog
            .peek(remote.base.trim(), remote.token.trim())
            .unwrap_or_default()
    }

    pub fn apply_network_update(
        &mut self,
        update: NetworkUpdate,
    ) -> Result<NetworkUpdateResult, String> {
        let mut token_revealed: Option<String> = None;

        let mut revealed_id: Option<String> = None;
        if let Some(expose) = update.expose {
            self.config.network.expose = expose;
            if expose {
                if self.config.network.ensure_token() {
                    token_revealed = self.config.network.primary_api_key().map(str::to_string);
                    revealed_id = self.config.network.api_keys.first().map(|k| k.id.clone());
                }
            }
        }

        if let Some(scope) = update.listen_scope.as_deref() {
            let Some(parsed) = ListenScope::parse(scope) else {
                return Err(
                    "listen_scope must be \"all\", \"tailscale\", or \"custom\"".into(),
                );
            };
            self.config.network.listen_scope = parsed;
        }

        if let Some(host) = update.listen_host {
            self.config.network.listen_host = host.trim().to_string();
        }

        // Validate the chosen scope can resolve before persisting a bad Tailscale-only setup
        // as the only option — still allow saving so the UI can show the error + fix path.
        if self.config.network.expose {
            if let Err(error) = self.config.network.resolve_listen_host() {
                // Soft: keep config, surface error via listen_bind_error in API state.
                self.push_log(format!("network listen warning: {error}"));
            }
        }

        if update.regenerate_token.unwrap_or(false) {
            self.config.network.regenerate_token();
            token_revealed = self.config.network.primary_api_key().map(str::to_string);
            revealed_id = self.config.network.api_keys.first().map(|k| k.id.clone());
        }

        if let Some(mode) = update.inference_mode.as_deref() {
            let Some(parsed) = InferenceMode::parse(mode) else {
                return Err("inference_mode must be \"local\" or \"remote\"".into());
            };
            self.config.network.migrate_remotes();
            if parsed == InferenceMode::Remote && self.config.network.remotes.is_empty() {
                return Err("Save a linked LLM before switching chat to Linked LLM.".into());
            }
            self.config.network.inference_mode = parsed;
        }

        // Legacy single-link fields upsert into the remotes manager list.
        if update.remote_base.is_some() || update.remote_token.is_some() {
            self.config.network.migrate_remotes();
            let base = update
                .remote_base
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .or_else(|| {
                    self.config
                        .network
                        .active_remote()
                        .map(|remote| remote.base.clone())
                })
                .unwrap_or_default();
            if base.trim().is_empty() {
                let id = self.config.network.active_remote_id.clone();
                if !id.is_empty() {
                    let _ = self.config.network.delete_remote(&id);
                }
            } else {
                let token = update.remote_token.as_deref();
                let id = self
                    .config
                    .network
                    .active_remote()
                    .map(|remote| remote.id.clone());
                let name = self
                    .config
                    .network
                    .active_remote()
                    .map(|remote| remote.name.clone())
                    .unwrap_or_default();
                self.config.network.upsert_remote(
                    id.as_deref(),
                    &name,
                    &base,
                    token,
                    true,
                )?;
            }
        }

        if let Some(name) = update.device_name {
            self.config.network.device_name = name.trim().to_string();
        }

        self.config.network.migrate_remotes();
        self.config.keep_ui_private();
        self.config.sync_llama_bind_from_network();
        self.sync_share_tls()?;
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        self.sync_mdns_advertise();
        let restart_required = self.network_restart_required();
        self.push_log(format!(
            "network updated (expose={}, scope={}, mode={}, remotes={}, llama={}:{}, tls={})",
            self.config.network.expose,
            self.config.network.listen_scope.as_str(),
            self.config.network.inference_mode.as_str(),
            self.config.network.remotes.len(),
            self.config.effective_host(),
            self.config.effective_port(),
            self.config.uses_tls()
        ));

        Ok(NetworkUpdateResult {
            restart_required,
            access_token: token_revealed,
            revealed_api_key_id: revealed_id,
        })
    }

    pub fn create_linked_remote(
        &mut self,
        name: &str,
        base: &str,
        token: &str,
        activate: bool,
    ) -> Result<NetworkUpdateResult, String> {
        self.config
            .network
            .upsert_remote(None, name, base, Some(token), activate)?;
        self.persist_network_manager("linked LLM added")
    }

    pub fn update_linked_remote(
        &mut self,
        id: &str,
        name: Option<&str>,
        base: Option<&str>,
        token: Option<&str>,
    ) -> Result<NetworkUpdateResult, String> {
        self.config.network.migrate_remotes();
        let current = self
            .config
            .network
            .remotes
            .iter()
            .find(|remote| remote.id == id)
            .cloned()
            .ok_or_else(|| "Linked LLM not found.".to_string())?;
        self.config.network.upsert_remote(
            Some(id),
            name.unwrap_or(&current.name),
            base.unwrap_or(&current.base),
            token,
            false,
        )?;
        self.persist_network_manager("linked LLM updated")
    }

    pub fn delete_linked_remote(&mut self, id: &str) -> Result<NetworkUpdateResult, String> {
        self.config.network.delete_remote(id)?;
        self.persist_network_manager("linked LLM removed")
    }

    pub fn activate_linked_remote(&mut self, id: &str) -> Result<NetworkUpdateResult, String> {
        self.config.network.set_active_remote(id)?;
        self.persist_network_manager("linked LLM activated")
    }

    fn persist_network_manager(&mut self, action: &str) -> Result<NetworkUpdateResult, String> {
        self.config.network.migrate_remotes();
        self.config.keep_ui_private();
        self.config.sync_llama_bind_from_network();
        self.sync_share_tls()?;
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        self.sync_mdns_advertise();
        self.push_log(format!(
            "network {action} (mode={}, remotes={})",
            self.config.network.inference_mode.as_str(),
            self.config.network.remotes.len()
        ));
        Ok(NetworkUpdateResult::from_restart(
            self.network_restart_required(),
        ))
    }

    pub fn create_api_key(&mut self, name: &str) -> Result<NetworkUpdateResult, String> {
        let key = self.config.network.create_api_key(name)?;
        self.config.sync_llama_bind_from_network();
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        self.push_log(format!("api key created: {}", key.name));
        Ok(NetworkUpdateResult {
            restart_required: self.network_restart_required(),
            access_token: Some(key.secret),
            revealed_api_key_id: Some(key.id),
        })
    }

    pub fn rename_api_key(&mut self, id: &str, name: &str) -> Result<NetworkUpdateResult, String> {
        self.config.network.rename_api_key(id, name)?;
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        Ok(NetworkUpdateResult::from_restart(false))
    }

    pub fn regenerate_api_key(&mut self, id: &str) -> Result<NetworkUpdateResult, String> {
        let key = self.config.network.regenerate_api_key(id)?;
        self.config.sync_llama_bind_from_network();
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        self.push_log(format!("api key regenerated: {}", key.name));
        Ok(NetworkUpdateResult {
            restart_required: self.network_restart_required(),
            access_token: Some(key.secret),
            revealed_api_key_id: Some(key.id),
        })
    }

    pub fn delete_api_key(&mut self, id: &str) -> Result<NetworkUpdateResult, String> {
        let name = self
            .config
            .network
            .api_keys
            .iter()
            .find(|key| key.id == id)
            .map(|key| key.name.clone())
            .unwrap_or_else(|| id.to_string());
        self.config.network.delete_api_key(id)?;
        self.config.sync_llama_bind_from_network();
        self.config
            .save(&self.config_path)
            .map_err(|error| format!("could not save config: {error:#}"))?;
        self.push_log(format!("api key deleted: {name}"));
        Ok(NetworkUpdateResult {
            restart_required: self.network_restart_required(),
            access_token: None,
            revealed_api_key_id: None,
        })
    }

    pub fn displayed_config(&self) -> &Config {
        if self.active_server_id != PRIMARY_SERVER_ID
            && let Some(extra) = self.extra_servers.iter().find(|s| s.id == self.active_server_id)
        {
            return &extra.running_config;
        }
        self.running_config.as_ref().unwrap_or(&self.config)
    }

    pub fn has_pending_changes(&self) -> bool {
        if self.active_server_id != PRIMARY_SERVER_ID {
            return self
                .extra_servers
                .iter()
                .find(|s| s.id == self.active_server_id)
                .is_some_and(|s| {
                    // Extra instances intentionally use an allocated port.
                    configs_differ_ignoring_instance_port(&s.running_config, &self.config)
                });
        }
        self.running_config
            .as_ref()
            .is_some_and(|running| running != &self.config)
    }

    pub fn used_ports(&self) -> Vec<u16> {
        let mut ports = Vec::new();
        if let Some(running) = &self.running_config {
            ports.push(running.effective_port());
        }
        for server in &self.extra_servers {
            if server.is_running() {
                ports.push(server.port());
            }
        }
        ports
    }

    pub fn allocate_port(&self) -> u16 {
        let used = self.used_ports();
        let mut port = self.config.server.port.max(1);
        while used.contains(&port) {
            port = port.saturating_add(1);
            if port == 0 {
                port = 8080;
            }
        }
        port
    }

    pub fn select_server(&mut self, id: &str) -> Result<(), String> {
        if id == PRIMARY_SERVER_ID {
            self.active_server_id = PRIMARY_SERVER_ID.into();
            return Ok(());
        }
        if self.extra_servers.iter().any(|s| s.id == id) {
            self.active_server_id = id.to_string();
            Ok(())
        } else {
            Err(format!("Unknown server id: {id}"))
        }
    }

    pub fn running_server_count(&self) -> usize {
        let primary = usize::from(self.process.is_some());
        primary
            + self
                .extra_servers
                .iter()
                .filter(|s| s.is_running())
                .count()
    }

    /// Snapshot of every managed llama-server (primary + extras) for UI/API.
    pub fn server_summaries(&self) -> Vec<ServerSummary> {
        let mut out = Vec::new();
        if self.process.is_some() || self.running_config.is_some() {
            let config = self.running_config.as_ref().unwrap_or(&self.config);
            out.push(ServerSummary {
                id: PRIMARY_SERVER_ID.into(),
                model: config.model_label(),
                status: self.status,
                status_label: self.status.label(),
                endpoint: config.endpoint(),
                api_endpoint: config.api_endpoint(),
                port: config.effective_port(),
                ready: self.status == ServerStatus::Ready && self.endpoint_online,
                pid: self.process.as_ref().map(ServerProcess::id),
                // Fail closed until /props probe finishes — don't flash the Think control.
                thinking_supported: self.thinking_supported.unwrap_or(false),
                active: self.active_server_id == PRIMARY_SERVER_ID,
            });
        }
        for server in &self.extra_servers {
            if !server.is_running() && server.status == ServerStatus::Stopped {
                continue;
            }
            out.push(ServerSummary {
                id: server.id.clone(),
                model: server.model_label(),
                status: server.status,
                status_label: server.status.label(),
                endpoint: server.running_config.endpoint(),
                api_endpoint: server.running_config.api_endpoint(),
                port: server.port(),
                ready: server.status == ServerStatus::Ready && server.endpoint_online,
                pid: server.process.as_ref().map(ServerProcess::id),
                thinking_supported: server.thinking_supported_flag(),
                active: self.active_server_id == server.id,
            });
        }
        out
    }

    pub fn server_by_id(&self, id: &str) -> Option<ServerLookup<'_>> {
        if id == PRIMARY_SERVER_ID {
            let config = self.running_config.as_ref()?;
            return Some(ServerLookup {
                id: PRIMARY_SERVER_ID,
                config,
                ready: self.process.is_some()
                    && self.status == ServerStatus::Ready
                    && self.endpoint_online,
                thinking_supported: self.thinking_supported.unwrap_or(false),
            });
        }
        self.extra_servers.iter().find(|s| s.id == id).map(|s| ServerLookup {
            id: s.id.as_str(),
            config: &s.running_config,
            ready: s.status == ServerStatus::Ready && s.endpoint_online,
            thinking_supported: s.thinking_supported_flag(),
        })
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
                ModelSource::Local(path) if path.as_os_str().is_empty() => {
                    "<enter path to .gguf>".into()
                }
                ModelSource::HuggingFace(id) if id.trim().is_empty() => {
                    "<add a model in Models>".into()
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
            SettingField::Model => match &self.config.model.source {
                ModelSource::Local(path) if path.as_os_str().is_empty() => String::new(),
                ModelSource::HuggingFace(id) if id.trim().is_empty() => String::new(),
                ModelSource::HuggingFace(id) => id.clone(),
                ModelSource::Local(path) => path.display().to_string(),
            },
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
                "Use Add model on the Models tab, or pick one from the library."
            }
            (SettingField::Model, ModelSource::Local(_)) => {
                "Use Add model on the Models tab, or pick a local GGUF from the library."
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
            (SettingField::Host, _) => {
                "Set by Devices (loopback when Share is off). Not editable here."
            }
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
        self.poll_library_fetch();

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
                self.clear_thinking_support();
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
            self.poll_thinking_support();
        }

        if self.process.is_some() && self.last_stats_refresh.elapsed() >= Duration::from_secs(1) {
            let process_id = self.process.as_ref().map(ServerProcess::id);
            self.process_usage = process_id.and_then(|pid| self.process_monitor.refresh(pid));
            self.last_stats_refresh = Instant::now();
        }

        self.expire_stale_throughput();

        // Additional concurrent llama-server instances.
        let mut extra_logs = Vec::new();
        for server in &mut self.extra_servers {
            extra_logs.extend(server.tick());
        }
        for line in extra_logs {
            self.observe_startup_line(&line);
            self.push_log(line);
        }
        self.extra_servers
            .retain(|s| s.is_running() || s.status == ServerStatus::Failed);
        if self.active_server_id != PRIMARY_SERVER_ID
            && !self
                .extra_servers
                .iter()
                .any(|s| s.id == self.active_server_id)
        {
            self.active_server_id = if self.process.is_some() {
                PRIMARY_SERVER_ID.into()
            } else {
                self.extra_servers
                    .iter()
                    .find(|s| s.is_running())
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| PRIMARY_SERVER_ID.into())
            };
        }

        // Keep mDNS in sync when a server becomes Ready / exits without going
        // through the start/stop helpers.
        self.sync_mdns_advertise();
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
                self.status_detail = format!("Listening on {}", config.listen_label());
            }
            self.ensure_thinking_probe();
        } else if self.status != ServerStatus::Downloading {
            // Process is still up; a timed-out task-queue probe is not a restart.
            self.status = ServerStatus::Starting;
        }
    }

    fn clear_thinking_support(&mut self) {
        self.thinking_supported = None;
        self.thinking_probe_for = None;
        self.pending_thinking_probe = None;
    }

    fn ensure_thinking_probe(&mut self) {
        let label = self.displayed_config().model_label();
        if self.thinking_probe_for.as_deref() == Some(label.as_str())
            && (self.thinking_supported.is_some() || self.pending_thinking_probe.is_some())
        {
            return;
        }
        self.thinking_supported = None;
        self.thinking_probe_for = Some(label);
        if let Some(config) = self.running_config.clone() {
            self.pending_thinking_probe = Some(thinking_support_async(&config));
        }
    }

    fn poll_thinking_support(&mut self) {
        if let Some(result) = self
            .pending_thinking_probe
            .as_ref()
            .and_then(PendingThinkingProbe::take)
        {
            self.pending_thinking_probe = None;
            // /props unavailable → treat as unsupported (no model-name allowlist).
            self.thinking_supported = Some(result.unwrap_or(false));
        } else if self.status == ServerStatus::Ready {
            self.ensure_thinking_probe();
        }
    }

    pub fn thinking_supported(&self) -> bool {
        if self.config.network.inference_mode == crate::network::InferenceMode::Remote {
            // Remote peers aren't probed via local /props yet — hide until we can.
            return false;
        }
        if self.active_server_id != PRIMARY_SERVER_ID
            && let Some(extra) = self
                .extra_servers
                .iter()
                .find(|s| s.id == self.active_server_id)
        {
            return extra.thinking_supported_flag();
        }
        // Unknown / still probing → hide the control.
        self.thinking_supported.unwrap_or(false)
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
        if let Err(errors) = self.validate_for_launch() {
            self.status = ServerStatus::Failed;
            self.status_detail = errors.clone();
            self.push_log(format!("[start blocked] {errors}"));
            return;
        }
        self.config.keep_ui_private();
        self.config.sync_llama_bind_from_network();
        if let Err(error) = self.sync_share_tls() {
            self.status = ServerStatus::Failed;
            self.status_detail = error.clone();
            self.push_log(format!("[start blocked] {error}"));
            return;
        }
        let mut launch_config = self.config.clone();
        // Additional instances get the next free port; primary keeps the configured port.
        if self.process.is_some() {
            launch_config.server.port = self.allocate_port();
        } else if self.used_ports().contains(&launch_config.effective_port()) {
            launch_config.server.port = self.allocate_port();
        }
        let display = CommandSpec::from_config(&launch_config).display();
        self.push_log(format!("$ {display}"));
        match ServerProcess::start(&launch_config) {
            Ok(process) => {
                let pid = process.id();
                let download = watch_download(&launch_config);
                if self.process.is_none() {
                    self.download = download;
                    self.process = Some(process);
                    self.running_config = Some(launch_config);
                    self.active_server_id = PRIMARY_SERVER_ID.into();
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
                } else {
                    self.next_server_seq += 1;
                    let id = format!("srv-{}", self.next_server_seq);
                    let port = launch_config.effective_port();
                    let server = ManagedServer::new_launching(id.clone(), launch_config, process, download);
                    self.push_log(format!(
                        "started additional llama-server {} on :{} (PID {pid})",
                        id, port
                    ));
                    self.active_server_id = id;
                    self.status_detail = server.status_detail.clone();
                    self.extra_servers.push(server);
                }
                self.sync_mdns_advertise();
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

    /// Managed Models-tab download, if one is in progress or just finished with an error.
    pub fn active_library_fetch(&self) -> Option<&LibraryFetch> {
        self.library_fetch.as_ref().filter(|fetch| {
            fetch.is_active() || fetch.error.is_some() || (fetch.done && fetch.error.is_none())
        })
    }

    /// Add a model to the library and optionally start downloading it.
    pub fn add_model(
        &mut self,
        kind: &str,
        value: &str,
        download: bool,
    ) -> std::result::Result<(), String> {
        let source = match kind {
            "hugging_face" | "huggingface" | "hf" => {
                let repo = hub::normalize_repo_id(value).map_err(|error| format!("{error:#}"))?;
                self.last_hf_model = repo.clone();
                ModelSource::HuggingFace(repo)
            }
            "local" => {
                let path = PathBuf::from(value.trim());
                if path.as_os_str().is_empty() {
                    return Err("Enter a full .gguf path.".into());
                }
                self.last_local_model = path.clone();
                ModelSource::Local(path)
            }
            _ => return Err("Choose Hugging Face or Local GGUF.".into()),
        };
        self.config.model.source = source.clone();
        self.config.remember_model(source.clone());
        self.sync_remote_model_size();
        self.mark_setting_changed(SettingField::Model);
        if download {
            match &source {
                ModelSource::HuggingFace(repo) => self.start_library_download(repo)?,
                ModelSource::Local(_) => {}
            }
        }
        self.persist_config();
        Ok(())
    }

    pub fn start_library_download_for_index(
        &mut self,
        index: usize,
    ) -> std::result::Result<(), String> {
        let source = self
            .library_entries()
            .get(index)
            .map(|entry| entry.source.clone())
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;
        let ModelSource::HuggingFace(repo) = source else {
            return Err("Only Hugging Face models can be downloaded into the cache.".into());
        };
        self.start_library_download(&repo)
    }

    pub fn start_library_download_by_label(
        &mut self,
        label: &str,
    ) -> std::result::Result<(), String> {
        let source = self
            .library_entries()
            .into_iter()
            .find(|entry| source_label(&entry.source) == label)
            .map(|entry| entry.source)
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;
        let ModelSource::HuggingFace(repo) = source else {
            return Err("Only Hugging Face models can be downloaded into the cache.".into());
        };
        self.start_library_download(&repo)
    }

    pub fn start_library_download(&mut self, repo: &str) -> std::result::Result<(), String> {
        if self
            .library_fetch
            .as_ref()
            .is_some_and(|fetch| fetch.is_active())
        {
            return Err("A download is already in progress.".into());
        }
        if self.process.is_some() {
            return Err("Stop the server before downloading a model into the cache.".into());
        }
        let repo = hub::normalize_repo_id(repo).map_err(|error| format!("{error:#}"))?;
        if cache::has_local_files(&ModelSource::HuggingFace(repo.clone()))
            && !cache::looks_incomplete(&repo, self.config.model.estimated_size_gib)
        {
            self.status_detail = format!("{repo} is already on disk");
            return Ok(());
        }
        let pending = fetch::fetch_primary_gguf_async(&repo);
        self.library_fetch = Some(LibraryFetch::new(&repo, pending));
        self.status_detail = format!("Downloading {repo}");
        self.push_log(format!(
            "downloading {repo} into the local Hugging Face cache"
        ));
        Ok(())
    }

    pub fn cancel_library_download(&mut self) -> std::result::Result<(), String> {
        let Some(fetch) = self.library_fetch.as_mut() else {
            return Err("No download is in progress.".into());
        };
        if !fetch.is_active() {
            return Err("No download is in progress.".into());
        }
        let repo = fetch.repo.clone();
        fetch.cancel();
        self.status_detail = format!("Cancelling download of {repo}");
        self.push_log(format!("cancelling download of {repo}"));
        Ok(())
    }

    fn poll_library_fetch(&mut self) {
        let Some(fetch) = self.library_fetch.as_mut() else {
            return;
        };
        let was_active = fetch.is_active();
        fetch.poll();
        if was_active && fetch.done {
            if let Some(error) = fetch.error.clone() {
                self.status_detail = format!("Download failed: {error}");
                self.push_log(format!("download failed: {error}"));
            } else {
                let repo = fetch.repo.clone();
                let gib = fetch.downloaded as f64 / (1024.0 * 1024.0 * 1024.0);
                self.status_detail = format!("Downloaded {repo} ({gib:.2} GiB)");
                self.push_log(format!("downloaded {repo} ({gib:.2} GiB)"));
                self.pending_discover = Some(cache::discover_models_async());
                self.last_discover = Instant::now();
                self.sync_remote_model_size();
            }
        }
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
        self.stop_server(None);
    }

    pub fn stop_server(&mut self, id: Option<&str>) {
        let target = id
            .map(str::to_string)
            .unwrap_or_else(|| self.active_server_id.clone());
        if target == PRIMARY_SERVER_ID {
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
            self.clear_thinking_support();
            if self.active_server_id == PRIMARY_SERVER_ID {
                if let Some(extra) = self.extra_servers.iter().find(|s| s.is_running()) {
                    self.active_server_id = extra.id.clone();
                }
            }
            self.sync_mdns_advertise();
            return;
        }

        if let Some(index) = self.extra_servers.iter().position(|s| s.id == target) {
            let mut server = self.extra_servers.remove(index);
            for line in server.stop() {
                self.push_log(line);
            }
            self.status_detail = server.status_detail;
            if self.active_server_id == target {
                self.active_server_id = if self.process.is_some() {
                    PRIMARY_SERVER_ID.into()
                } else if let Some(extra) = self.extra_servers.iter().find(|s| s.is_running()) {
                    extra.id.clone()
                } else {
                    PRIMARY_SERVER_ID.into()
                };
            }
            self.sync_mdns_advertise();
        }
    }

    pub fn restart(&mut self) {
        let active = self.active_server_id.clone();
        self.stop_server(Some(&active));
        // Restart always launches from the current config template into a free slot.
        self.start();
    }

    pub fn shutdown(&mut self) {
        while self.process.is_some() {
            self.stop_server(Some(PRIMARY_SERVER_ID));
        }
        let ids: Vec<_> = self.extra_servers.iter().map(|s| s.id.clone()).collect();
        for id in ids {
            self.stop_server(Some(&id));
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

    pub fn set_ui_theme(&mut self, theme: crate::config::UiTheme) {
        if self.config.ui.theme == theme {
            return;
        }
        self.config.ui.theme = theme;
        self.persist_config();
        self.push_log(format!("ui theme set to {}", theme.as_str()));
    }

    pub fn set_ui_appearance(
        &mut self,
        theme: Option<crate::config::UiTheme>,
        font_body: Option<String>,
        font_display: Option<String>,
        font_mono: Option<String>,
        font_scale: Option<crate::config::UiFontScale>,
    ) -> Result<(), String> {
        let mut changed = false;
        if let Some(theme) = theme {
            if self.config.ui.theme != theme {
                self.config.ui.theme = theme;
                changed = true;
            }
        }
        if let Some(font_body) = font_body {
            let id = font_body.trim().to_ascii_lowercase();
            if !crate::config::UI_FONT_BODY_IDS.contains(&id.as_str()) {
                return Err(format!(
                    "font_body must be one of: {}",
                    crate::config::UI_FONT_BODY_IDS.join(", ")
                ));
            }
            if self.config.ui.font_body != id {
                self.config.ui.font_body = id;
                changed = true;
            }
        }
        if let Some(font_display) = font_display {
            let id = font_display.trim().to_ascii_lowercase();
            if !crate::config::UI_FONT_DISPLAY_IDS.contains(&id.as_str()) {
                return Err(format!(
                    "font_display must be one of: {}",
                    crate::config::UI_FONT_DISPLAY_IDS.join(", ")
                ));
            }
            if self.config.ui.font_display != id {
                self.config.ui.font_display = id;
                changed = true;
            }
        }
        if let Some(font_mono) = font_mono {
            let id = font_mono.trim().to_ascii_lowercase();
            if !crate::config::UI_FONT_MONO_IDS.contains(&id.as_str()) {
                return Err(format!(
                    "font_mono must be one of: {}",
                    crate::config::UI_FONT_MONO_IDS.join(", ")
                ));
            }
            if self.config.ui.font_mono != id {
                self.config.ui.font_mono = id;
                changed = true;
            }
        }
        if let Some(font_scale) = font_scale {
            if self.config.ui.font_scale != font_scale {
                self.config.ui.font_scale = font_scale;
                changed = true;
            }
        }
        if changed {
            self.persist_config();
            self.push_log("ui appearance updated".into());
        }
        Ok(())
    }

    pub fn reset_ui_appearance(&mut self) {
        let before = self.config.ui.clone();
        self.config.ui.reset_appearance();
        if self.config.ui != before {
            self.persist_config();
            self.push_log("ui appearance reset to defaults".into());
        }
    }

    /// Persist config without replacing the current status line (used after library edits).
    fn persist_config(&mut self) {
        if let Err(error) = self.config.save(&self.config_path) {
            self.status = ServerStatus::Failed;
            self.status_detail = error.to_string();
            self.push_log(format!("failed to save config: {error}"));
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
        self.persist_config();
        self.status_detail = if self.process.is_some() {
            "Restored defaults; restart to apply".into()
        } else {
            "Restored defaults".into()
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
        self.persist_config();
        self.status_detail = if self.process.is_some() {
            format!("Applied {}; restart to apply", preset.label())
        } else {
            format!("Applied {}", preset.label())
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
                return Err(
                    "Listen address is controlled in Devices — change the scope there."
                        .into(),
                );
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
        self.persist_config();
        self.status_detail = if self.process.is_some() {
            format!("Changed {}; restart to apply", self.setting_label(field))
        } else {
            format!("Changed {}", self.setting_label(field))
        };
    }

    fn remember_current_model(&mut self) {
        let source = self.config.model.source.clone();
        self.config.remember_model(source);
    }

    /// Explicitly managed library entries (`recent_models` only).
    pub fn library_entries(&self) -> Vec<ModelPickerEntry> {
        self.config
            .recent_models
            .iter()
            .filter(|source| match source {
                ModelSource::HuggingFace(id) => !id.trim().is_empty(),
                ModelSource::Local(path) => !path.as_os_str().is_empty(),
            })
            .cloned()
            .map(|source| self.entry_for(source, true))
            .collect()
    }

    /// Autodiscovered cache models that are not already in the library.
    pub fn available_entries(&self) -> Vec<ModelPickerEntry> {
        self.discovered_models
            .iter()
            .filter_map(|discovered| {
                let source = match &discovered.source {
                    cache::DiscoveredSource::HuggingFace(repo) => {
                        ModelSource::HuggingFace(repo.clone())
                    }
                    cache::DiscoveredSource::Local(path) => ModelSource::Local(path.clone()),
                };
                if self
                    .config
                    .recent_models
                    .iter()
                    .any(|recent| sources_equivalent(recent, &source))
                {
                    return None;
                }
                Some(ModelPickerEntry {
                    source,
                    recent: false,
                    on_disk: true,
                    bytes: discovered.bytes,
                })
            })
            .collect()
    }

    /// Back-compat alias used by older call sites/tests.
    pub fn model_picker_entries(&self) -> Vec<ModelPickerEntry> {
        self.library_entries()
    }

    fn entry_for(&self, source: ModelSource, recent: bool) -> ModelPickerEntry {
        let discovered_bytes = self
            .discovered_models
            .iter()
            .find(|model| match (&model.source, &source) {
                (cache::DiscoveredSource::HuggingFace(left), ModelSource::HuggingFace(right)) => {
                    left == right
                }
                (cache::DiscoveredSource::Local(left), ModelSource::Local(right)) => left == right,
                _ => false,
            })
            .map(|model| model.bytes)
            .unwrap_or(0);
        let on_disk = cache::has_local_files(&source);
        let bytes = if discovered_bytes > 0 {
            discovered_bytes
        } else if on_disk {
            cache::local_bytes(&source)
        } else {
            0
        };
        ModelPickerEntry {
            source,
            recent,
            on_disk,
            bytes,
        }
    }

    pub fn select_recent_model(&mut self, index: usize) -> std::result::Result<(), String> {
        let selected = self
            .library_entries()
            .get(index)
            .map(|entry| entry.source.clone())
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;
        self.activate_model(selected);
        self.persist_config();
        Ok(())
    }

    pub fn select_library_model_by_label(
        &mut self,
        label: &str,
    ) -> std::result::Result<(), String> {
        let selected = self
            .library_entries()
            .into_iter()
            .find(|entry| source_label(&entry.source) == label)
            .map(|entry| entry.source)
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;
        self.activate_model(selected);
        self.persist_config();
        Ok(())
    }

    pub fn import_available_model(&mut self, label: &str) -> std::result::Result<(), String> {
        let source = self
            .available_entries()
            .into_iter()
            .find(|entry| source_label(&entry.source) == label)
            .map(|entry| entry.source)
            .ok_or_else(|| "That on-disk model is no longer available.".to_string())?;
        self.config.remember_model(source.clone());
        self.activate_model(source);
        self.persist_config();
        Ok(())
    }

    pub fn can_delete_picker_model(&self, index: usize) -> bool {
        if self.process.is_some() {
            return false;
        }
        self.library_entries().get(index).is_some()
    }

    pub fn delete_picker_model(&mut self, index: usize) -> std::result::Result<(), String> {
        let label = self
            .library_entries()
            .get(index)
            .map(|entry| source_label(&entry.source))
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;
        self.delete_library_model_by_label(&label)
    }

    pub fn delete_library_model_by_label(
        &mut self,
        label: &str,
    ) -> std::result::Result<(), String> {
        if self.process.is_some() {
            return Err("Stop the server before removing a model.".into());
        }
        let source = self
            .library_entries()
            .into_iter()
            .find(|entry| source_label(&entry.source) == label)
            .map(|entry| entry.source)
            .ok_or_else(|| "That model is no longer in your library.".to_string())?;

        let mut freed_gib = 0.0;
        if cache::has_local_files(&source) {
            let report = cache::delete_local_files(&source)?;
            freed_gib = report.freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            self.push_log(format!(
                "deleted local cache for {label} ({} path{}, {freed_gib:.2} GiB)",
                report.removed_paths.len(),
                if report.removed_paths.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }

        let was_active = sources_equivalent(&self.config.model.source, &source);
        self.config
            .recent_models
            .retain(|recent| !sources_equivalent(recent, &source));
        self.discovered_models.retain(|model| match &model.source {
            cache::DiscoveredSource::HuggingFace(repo) => {
                !sources_equivalent(&ModelSource::HuggingFace(repo.clone()), &source)
            }
            cache::DiscoveredSource::Local(path) => {
                !sources_equivalent(&ModelSource::Local(path.clone()), &source)
            }
        });
        if was_active {
            self.activate_fallback_model();
        }
        self.pending_discover = Some(cache::discover_models_async());
        self.last_discover = Instant::now();
        self.status_detail = if freed_gib > 0.0 {
            format!("Removed {label} ({freed_gib:.2} GiB freed)")
        } else {
            format!("Removed {label}")
        };
        self.push_log(format!("removed {label} from the model library"));
        self.persist_config();
        Ok(())
    }

    pub fn delete_available_model_by_label(
        &mut self,
        label: &str,
    ) -> std::result::Result<(), String> {
        if self.process.is_some() {
            return Err("Stop the server before deleting cached files.".into());
        }
        let source = self
            .available_entries()
            .into_iter()
            .find(|entry| source_label(&entry.source) == label)
            .map(|entry| entry.source)
            .ok_or_else(|| "That on-disk model is no longer available.".to_string())?;
        let report = cache::delete_local_files(&source)?;
        let gib = report.freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        self.discovered_models.retain(|model| match &model.source {
            cache::DiscoveredSource::HuggingFace(repo) => {
                !sources_equivalent(&ModelSource::HuggingFace(repo.clone()), &source)
            }
            cache::DiscoveredSource::Local(path) => {
                !sources_equivalent(&ModelSource::Local(path.clone()), &source)
            }
        });
        self.pending_discover = Some(cache::discover_models_async());
        self.last_discover = Instant::now();
        self.status_detail = format!("Deleted cached files for {label} ({gib:.2} GiB)");
        self.push_log(format!("deleted cached files for {label} ({gib:.2} GiB)"));
        self.persist_config();
        Ok(())
    }

    fn activate_model(&mut self, source: ModelSource) {
        match &source {
            ModelSource::HuggingFace(id) => self.last_hf_model = id.clone(),
            ModelSource::Local(path) => self.last_local_model = path.clone(),
        }
        self.config.model.source = source;
        self.sync_remote_model_size();
        self.mark_setting_changed(SettingField::Model);
    }

    fn activate_fallback_model(&mut self) {
        if let Some(fallback) = self.config.recent_models.first().cloned() {
            self.activate_model(fallback);
            return;
        }
        // Library is empty — do not resurrect the deleted default model.
        self.config.model.source = ModelSource::HuggingFace(String::new());
        self.remote_model_size = None;
        self.mark_setting_changed(SettingField::Model);
    }

    pub fn has_active_model(&self) -> bool {
        match &self.config.model.source {
            ModelSource::HuggingFace(id) => !id.trim().is_empty(),
            ModelSource::Local(path) => !path.as_os_str().is_empty(),
        }
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

    pub fn copy_logs(&mut self) -> String {
        let logs = self.logs.iter().cloned().collect::<Vec<_>>().join("\n");
        self.copy_text("logs", &logs);
        logs
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

fn source_label(source: &ModelSource) -> String {
    match source {
        ModelSource::HuggingFace(id) => id.clone(),
        ModelSource::Local(path) => path.display().to_string(),
    }
}

fn sources_equivalent(left: &ModelSource, right: &ModelSource) -> bool {
    match (left, right) {
        (ModelSource::Local(a), ModelSource::Local(b)) => a == b,
        (ModelSource::HuggingFace(a), ModelSource::HuggingFace(b)) => {
            fn strip(value: &str) -> &str {
                value.split(':').next().unwrap_or(value)
            }
            a == b || strip(a) == strip(b)
        }
        _ => false,
    }
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
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

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
    fn delete_removes_model_from_library_and_does_not_resurrect_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.gguf");
        std::fs::write(&path, vec![0; 32]).unwrap();
        let mut app = App::new(Config::default(), dir.path().join("config.toml"));
        let local = ModelSource::Local(path.clone());
        app.config.model.source = local.clone();
        app.config.recent_models = vec![local.clone()];
        app.discovered_models.clear();
        app.delete_library_model_by_label(&path.display().to_string())
            .unwrap();
        assert!(!path.exists());
        assert!(app.config.recent_models.is_empty());
        assert!(app.library_entries().is_empty());
        assert!(!app.has_active_model());
        assert!(app.status_detail.contains("Removed"));
        let reloaded = Config::load(&app.config_path).unwrap();
        assert!(reloaded.recent_models.is_empty());
    }

    #[test]
    fn delete_falls_back_to_remaining_library_model() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(Config::default(), dir.path().join("config.toml"));
        let missing = ModelSource::HuggingFace("tinyinference/never-cached".into());
        let other = ModelSource::HuggingFace("owner/other-GGUF".into());
        app.config.model.source = missing.clone();
        app.config.recent_models = vec![missing.clone(), other.clone()];
        app.delete_library_model_by_label("tinyinference/never-cached")
            .unwrap();
        assert!(!app.config.recent_models.contains(&missing));
        assert_eq!(app.config.model.source, other);
        assert_eq!(app.library_entries().len(), 1);
    }

    #[test]
    fn available_models_exclude_library_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(Config::default(), dir.path().join("config.toml"));
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
        assert_eq!(app.library_entries().len(), 1);
        let available = app.available_entries();
        assert_eq!(available.len(), 1);
        assert_eq!(
            available[0].source,
            ModelSource::HuggingFace("owner/cached-GGUF".into())
        );
        app.import_available_model("owner/cached-GGUF").unwrap();
        assert!(
            app.library_entries()
                .iter()
                .any(|entry| entry.source == ModelSource::HuggingFace("owner/cached-GGUF".into()))
        );
        assert!(app.available_entries().is_empty());
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
    fn allocate_port_skips_used_ports() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.running_config = Some(app.config.clone());
        assert_eq!(app.allocate_port(), 8081);
        app.config.server.port = 9090;
        assert_eq!(app.allocate_port(), 9090);
    }

    #[test]
    fn extra_server_allocated_port_is_not_pending_restart() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let mut extra_config = app.config.clone();
        extra_config.server.port = 8081;
        app.extra_servers
            .push(crate::instance::ManagedServer::stub_ready(
                "extra-1".into(),
                extra_config,
            ));
        app.active_server_id = "extra-1".into();
        assert!(!app.has_pending_changes());
        app.config.runtime.context_size = 4096;
        assert!(app.has_pending_changes());
    }

    #[test]
    fn share_urls_require_live_non_loopback_bind() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.config.network.expose = true;
        app.config.network.listen_scope = ListenScope::Custom;
        app.config.network.listen_host = "192.168.1.50".into();
        app.config.network.ensure_token();
        // Fail closed: unresolved/custom not synced yet → loopback → no URLs.
        app.config.server.host = "127.0.0.1".into();
        assert!(app.network_share_urls().is_empty());
        app.config.sync_llama_bind_from_network();
        app.running_config = Some(app.config.clone());
        app.status = ServerStatus::Ready;
        app.endpoint_online = true;
        assert!(!app.network_share_urls().is_empty());
        assert!(
            app.network_share_urls()
                .iter()
                .all(|url| url.api_base.contains("192.168.1.50"))
        );
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
    fn focus_is_only_reported_once_a_window_installs_a_hook() {
        let mut app = App::new(Config::default(), "test.toml".into());
        // Headless: an instance exists, but there is nothing to raise.
        assert!(!app.request_focus());

        let raised = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&raised);
        app.set_focus_hook(Box::new(move || flag.store(true, Ordering::SeqCst)));

        assert!(app.request_focus());
        assert!(raised.load(Ordering::SeqCst));
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
