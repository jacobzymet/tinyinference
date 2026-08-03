use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::{
    app::{App, Download, ServerStatus, SettingField},
    config::RuntimePreset,
    server::ServerProcess,
};

pub type SharedApp = Arc<Mutex<App>>;

pub async fn serve(app: SharedApp, addr: SocketAddr) -> anyhow::Result<()> {
    let tick_app = Arc::clone(&app);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            if let Ok(mut app) = tick_app.lock() {
                app.tick();
            }
        }
    });

    let router = Router::new()
        .route("/", get(index))
        .route("/api/state", get(state))
        .route("/api/start", post(start))
        .route("/api/stop", post(stop))
        .route("/api/restart", post(restart))
        .route("/api/save", post(save))
        .route("/api/reset", post(reset))
        .route("/api/presets/{preset}", post(apply_preset))
        .route("/api/settings", post(update_setting))
        .route("/api/settings/{field}/toggle", post(toggle_setting))
        .route("/api/settings/model/select", post(select_model))
        .route("/api/copy/endpoint", post(copy_endpoint))
        .route("/api/copy/command", post(copy_command))
        .route("/api/dismiss-prompt", post(dismiss_prompt))
        .route("/api/configure-server", post(configure_server))
        .with_state(app);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn state(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    Ok(Json(AppState::from_app(&app)))
}

async fn start(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.start())
}

async fn stop(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.stop())
}

async fn restart(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.restart())
}

async fn save(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.save())
}

async fn reset(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.reset_to_defaults())
}

async fn apply_preset(
    State(app): State<SharedApp>,
    Path(preset): Path<RuntimePreset>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.apply_runtime_preset(preset)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn update_setting(
    State(app): State<SharedApp>,
    Json(body): Json<SettingUpdate>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.set_field(body.field, &body.value)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn toggle_setting(
    State(app): State<SharedApp>,
    Path(field): Path<SettingField>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.toggle_field(field).map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn select_model(
    State(app): State<SharedApp>,
    Json(body): Json<SelectRecentModel>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.select_recent_model(body.index)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn copy_endpoint(State(app): State<SharedApp>) -> Result<Json<CopyResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let value = app.copy_endpoint();
    Ok(Json(CopyResponse {
        value,
        state: AppState::from_app(&app),
    }))
}

async fn copy_command(State(app): State<SharedApp>) -> Result<Json<CopyResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let value = app.copy_command();
    Ok(Json(CopyResponse {
        value,
        state: AppState::from_app(&app),
    }))
}

async fn dismiss_prompt(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.dismiss_server_prompt())
}

async fn configure_server(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.open_server_configuration())
}

fn with_app(app: SharedApp, action: impl FnOnce(&mut App)) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    action(&mut app);
    Ok(Json(AppState::from_app(&app)))
}

#[derive(Debug, Deserialize)]
struct SettingUpdate {
    field: SettingField,
    value: String,
}

#[derive(Debug, Deserialize)]
struct SelectRecentModel {
    index: usize,
}

#[derive(Debug, Serialize)]
struct CopyResponse {
    value: String,
    state: AppState,
}

#[derive(Debug, Serialize)]
struct AppState {
    status: ServerStatus,
    status_label: &'static str,
    status_detail: String,
    running: bool,
    endpoint_online: bool,
    pending_changes: bool,
    missing_server: bool,
    model: String,
    endpoint: String,
    api_endpoint: String,
    command: String,
    config_path: String,
    pid: Option<u32>,
    machine: MachineState,
    mapped_size: MappedSizeState,
    download: Option<DownloadState>,
    process: Option<ProcessState>,
    metrics: Option<MetricsState>,
    recent_models: Vec<RecentModelState>,
    settings: Vec<SettingState>,
    cache_type_help: Vec<ChoiceHelp>,
    presets: Vec<PresetState>,
    active_preset: Option<&'static str>,
    logs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PresetState {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    warning: Option<&'static str>,
    available: bool,
}

#[derive(Debug, Serialize)]
struct MappedSizeState {
    gib: Option<f64>,
    display: String,
    detail: String,
    loading: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecentModelState {
    index: usize,
    group: &'static str,
    kind: &'static str,
    label: String,
    selected: bool,
}

#[derive(Debug, Serialize)]
struct MachineState {
    cpu_name: String,
    logical_cpus: usize,
    physical_cpus: usize,
    available_gib: f64,
    total_gib: f64,
    runtime_summary: String,
    access_summary: String,
}

#[derive(Debug, Serialize)]
struct DownloadState {
    repo: String,
    file: Option<String>,
    downloaded: u64,
    total: Option<u64>,
    fraction: Option<f64>,
    rate: Option<f64>,
    eta_seconds: Option<u64>,
    listing_error: Option<String>,
    summary: String,
    progress: String,
}

#[derive(Debug, Serialize)]
struct ProcessState {
    cpu_percent: f32,
    resident_memory_gib: f64,
    virtual_memory_gib: f64,
    uptime_seconds: u64,
}

#[derive(Debug, Serialize)]
struct MetricsState {
    prompt_tokens: Option<f64>,
    generated_tokens: Option<f64>,
    prompt_tokens_per_second: Option<f64>,
    generated_tokens_per_second: Option<f64>,
    requests_processing: Option<f64>,
    requests_deferred: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ChoiceHelp {
    value: &'static str,
    description: &'static str,
}

#[derive(Debug, Serialize)]
struct SettingState {
    id: &'static str,
    field: SettingField,
    label: &'static str,
    value: String,
    raw: String,
    hint: &'static str,
    editable: bool,
    toggle: bool,
    choices: Option<Vec<&'static str>>,
}

impl AppState {
    fn from_app(app: &App) -> Self {
        let config = app.displayed_config();
        let memory = app.machine.memory_profile(config);
        let download = app.active_download().map(DownloadState::from_download);
        let settings = SettingField::ALL
            .iter()
            .copied()
            .filter(|field| *field != SettingField::EstimatedSize)
            .map(|field| SettingState {
                id: field.id(),
                field,
                label: app.setting_label(field),
                value: app.setting_value(field),
                raw: app.setting_raw_value(field),
                hint: app.setting_hint(field),
                editable: app.setting_is_editable(field),
                toggle: field.is_toggle(),
                choices: field.choices().map(|items| items.to_vec()),
            })
            .collect();
        let cache_type_help = crate::config::CACHE_TYPE_DESCRIPTIONS
            .iter()
            .map(|(value, description)| ChoiceHelp { value, description })
            .collect();
        let unified_memory = crate::system::likely_unified_memory();
        let presets = RuntimePreset::ALL
            .iter()
            .copied()
            .map(|preset| PresetState {
                id: preset.id(),
                label: preset.label(),
                description: preset.description(),
                warning: preset.warning(),
                available: !(unified_memory && preset.blocked_on_unified_memory()),
            })
            .collect();
        let active_preset = app.active_runtime_preset().map(RuntimePreset::id);
        let mapped_size = MappedSizeState::from_app(app);
        let recent_models = app
            .model_picker_entries()
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let (kind, label) = match &entry.source {
                    crate::config::ModelSource::HuggingFace(id) => ("Hugging Face", id.clone()),
                    crate::config::ModelSource::Local(path) => {
                        ("Local GGUF", path.display().to_string())
                    }
                };
                RecentModelState {
                    index,
                    group: entry.group.label(),
                    kind,
                    label,
                    selected: entry.source == app.config.model.source,
                }
            })
            .collect();

        Self {
            status: app.status,
            status_label: app.status.label(),
            status_detail: app.status_detail.clone(),
            running: app.process.is_some(),
            endpoint_online: app.endpoint_online,
            pending_changes: app.has_pending_changes(),
            missing_server: app.should_prompt_for_server(),
            model: config.model_label(),
            endpoint: config.endpoint(),
            api_endpoint: config.api_endpoint(),
            command: crate::server::CommandSpec::from_config(config).display(),
            config_path: app.config_path.display().to_string(),
            pid: app.process.as_ref().map(ServerProcess::id),
            machine: MachineState {
                cpu_name: app.machine.cpu_name.clone(),
                logical_cpus: app.machine.logical_cpus,
                physical_cpus: app.machine.physical_cpus,
                available_gib: memory.available_gib,
                total_gib: memory.total_gib,
                runtime_summary: runtime_summary(config),
                access_summary: access_summary(config),
            },
            mapped_size,
            download,
            process: app.process_usage.as_ref().map(|usage| ProcessState {
                cpu_percent: usage.cpu_percent,
                resident_memory_gib: usage.resident_memory_gib,
                virtual_memory_gib: usage.virtual_memory_gib,
                uptime_seconds: usage.uptime_seconds,
            }),
            metrics: app.server_metrics.as_ref().map(|metrics| MetricsState {
                prompt_tokens: metrics.prompt_tokens,
                generated_tokens: metrics.generated_tokens,
                prompt_tokens_per_second: metrics.prompt_tokens_per_second,
                generated_tokens_per_second: metrics.generated_tokens_per_second,
                requests_processing: metrics.requests_processing,
                requests_deferred: metrics.requests_deferred,
            }),
            recent_models,
            settings,
            cache_type_help,
            presets,
            active_preset,
            logs: app.logs.iter().cloned().collect(),
        }
    }
}

impl MappedSizeState {
    fn from_app(app: &App) -> Self {
        let loading = app.remote_model_size_loading();
        let error = app.remote_model_size_error().map(str::to_string);
        match app.mapped_model_gib() {
            Some(gib) => {
                let source = match &app.config.model.source {
                    crate::config::ModelSource::Local(_) => {
                        "Measured from the local GGUF file on disk"
                    }
                    crate::config::ModelSource::HuggingFace(_) => {
                        "Fetched from the Hugging Face repository listing"
                    }
                };
                Self {
                    gib: Some(gib),
                    display: format!("{gib:.1} GiB"),
                    detail: format!(
                        "{source}. mmap maps this from disk; it does not need to fit in RAM."
                    ),
                    loading: false,
                    error: None,
                }
            }
            None if loading => Self {
                gib: None,
                display: "Fetching…".into(),
                detail: "Asking Hugging Face for the GGUF file size.".into(),
                loading: true,
                error: None,
            },
            None => Self {
                gib: None,
                display: "Unavailable".into(),
                detail: error
                    .clone()
                    .unwrap_or_else(|| "Model file size could not be determined.".into()),
                loading: false,
                error,
            },
        }
    }
}

impl DownloadState {
    fn from_download(download: &Download) -> Self {
        Self {
            repo: download.repo.clone(),
            file: download.file.clone(),
            downloaded: download.downloaded,
            total: download.total,
            fraction: download.fraction(),
            rate: download.rate(),
            eta_seconds: download.eta().map(|eta| eta.as_secs()),
            listing_error: download.listing_error().map(str::to_string),
            summary: download_summary(download),
            progress: download_progress(download),
        }
    }
}

fn runtime_summary(config: &crate::config::Config) -> String {
    format!(
        "{} · {} context · batch {} · {} {}",
        if config.runtime.cpu_only {
            "CPU"
        } else {
            "GPU offload"
        },
        format_tokens(config.runtime.context_size),
        config.runtime.batch_size,
        config.runtime.parallel,
        if config.runtime.parallel == 1 {
            "slot"
        } else {
            "slots"
        },
    )
}

fn access_summary(config: &crate::config::Config) -> String {
    let mut parts = vec![
        if config.runtime.mmap {
            "mmap"
        } else {
            "loaded"
        },
        if config.runtime.warmup {
            "warmup"
        } else {
            "no warmup"
        },
        if config.runtime.repack {
            "repack"
        } else {
            "no repack"
        },
    ];
    let model = config.model_label().to_ascii_lowercase();
    if model.contains("gpt-oss-120b") {
        parts.push("MoE · 5.1B active/token");
    } else if model.contains("gpt-oss-20b") {
        parts.push("MoE · 3.6B active/token");
    }
    parts.join(" · ")
}

fn download_summary(download: &Download) -> String {
    match &download.file {
        Some(file) => format!("Downloading {file} · {}", download.repo),
        None => format!("Downloading {}", download.repo),
    }
}

fn download_progress(download: &Download) -> String {
    let mut parts = Vec::new();
    if download.listing_error().is_some() {
        parts.push("progress unavailable".into());
        if download.downloaded == 0 {
            parts.push("preparing".into());
        } else {
            parts.push(format!("{} fetched", format_bytes(download.downloaded)));
        }
    } else {
        match (download.fraction(), download.total) {
            (Some(fraction), Some(total)) => {
                parts.push(format!("{:.0}%", fraction * 100.0));
                parts.push(format!(
                    "{} of {}",
                    format_bytes(download.downloaded),
                    format_bytes(total)
                ));
            }
            _ if download.downloaded == 0 => parts.push("preparing".into()),
            _ => parts.push(format!("{} fetched", format_bytes(download.downloaded))),
        }
    }
    if let Some(rate) = download.rate() {
        parts.push(format!("{}/s", format_bytes(rate.round() as u64)));
    }
    if let Some(eta) = download.eta() {
        parts.push(format!("{} left", format_eta(eta.as_secs())));
    }
    parts.join("  ·  ")
}

fn format_tokens(tokens: u32) -> String {
    if tokens >= 1024 && tokens.is_multiple_of(1024) {
        format!("{}k", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

fn format_eta(seconds: u64) -> String {
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format_duration(seconds)
    }
}

fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [(&str, f64); 4] = [
        ("TiB", 1024.0 * 1024.0 * 1024.0 * 1024.0),
        ("GiB", 1024.0 * 1024.0 * 1024.0),
        ("MiB", 1024.0 * 1024.0),
        ("KiB", 1024.0),
    ];
    for (unit, scale) in UNITS {
        if bytes as f64 >= scale {
            return format!("{:.1} {unit}", bytes as f64 / scale);
        }
    }
    format!("{bytes} B")
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn lock() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "application state is unavailable".into(),
        }
    }

    fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

const INDEX_HTML: &str = include_str!("index.html");
