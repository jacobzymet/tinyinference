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
    logs: Vec<String>,
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
        let mapped_size = MappedSizeState::from_app(app);
        let recent_models = app
            .config
            .recent_models
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let (kind, label) = match source {
                    crate::config::ModelSource::HuggingFace(id) => ("Hugging Face", id.clone()),
                    crate::config::ModelSource::Local(path) => {
                        ("Local GGUF", path.display().to_string())
                    }
                };
                RecentModelState {
                    index,
                    kind,
                    label,
                    selected: source == &app.config.model.source,
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

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tinyinference</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --ti-canvas: #0f1418;
      --ti-surface: #182227;
      --ti-ink: #dce4e2;
      --ti-muted: #78858b;
      --ti-ice: #84b8d0;
      --ti-coral: #d78374;
      --ti-mint: #89bfa4;
    }
    body {
      background: var(--ti-canvas);
      color: var(--ti-ink);
      min-height: 100vh;
    }
    .navbar, .card, .modal-content, .form-control, .form-select, .nav-tabs .nav-link {
      background: var(--ti-surface);
      color: var(--ti-ink);
      border-color: #243036;
    }
    .navbar-brand { color: var(--ti-ink) !important; letter-spacing: 0.02em; }
    .navbar-brand span { color: var(--ti-ice); }
    .text-muted { color: var(--ti-muted) !important; }
    .nav-tabs { border-color: #243036; }
    .nav-tabs .nav-link { color: var(--ti-muted); }
    .nav-tabs .nav-link.active {
      background: var(--ti-canvas);
      color: var(--ti-ice);
      border-color: #243036 #243036 var(--ti-canvas);
    }
    .form-control:focus, .form-select:focus {
      background: var(--ti-surface);
      color: var(--ti-ink);
      border-color: var(--ti-ice);
      box-shadow: 0 0 0 0.2rem rgba(132, 184, 208, 0.2);
    }
    .form-control:disabled, .form-select:disabled {
      background: #12191d;
      color: var(--ti-ink);
      opacity: 1;
      border-color: #243036;
    }
    .form-check-input:checked { background-color: var(--ti-ice); border-color: var(--ti-ice); }
    .btn-ice { background: var(--ti-ice); color: #0f1418; border: none; }
    .btn-ice:hover { background: #9ec9db; color: #0f1418; }
    .btn-outline-ice { color: var(--ti-ice); border-color: var(--ti-ice); }
    .btn-outline-ice:hover { background: var(--ti-ice); color: #0f1418; }
    .btn-coral { background: var(--ti-coral); color: #0f1418; border: none; }
    .status-ready { color: var(--ti-mint); }
    .status-failed, .status-stopping { color: var(--ti-coral); }
    .status-downloading, .status-starting { color: var(--ti-ice); }
    .status-stopped { color: var(--ti-muted); }
    .log-view {
      background: #0b1013;
      border: 1px solid #243036;
      border-radius: 0.375rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.82rem;
      height: 28rem;
      overflow: auto;
      white-space: pre-wrap;
      padding: 1rem;
    }
    .metric-label { color: var(--ti-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .setting-hint { font-size: 0.8rem; color: var(--ti-muted); }
    .progress { background: #0b1013; height: 0.65rem; }
    .progress-bar { background: var(--ti-ice); }
    a { color: var(--ti-ice); }
    .external-icon { width: 0.9em; height: 0.9em; margin-left: 0.35em; vertical-align: -0.1em; }
    .btn .external-icon { margin-left: 0.45em; }
    .size-panel {
      background: #12191d;
      border: 1px solid #243036;
      border-radius: 0.375rem;
      padding: 1rem 1.1rem;
    }
    .size-panel .size-value {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--ti-ink);
      letter-spacing: 0.01em;
    }
    .size-panel .size-value.is-muted { color: var(--ti-muted); font-size: 1.15rem; font-weight: 500; }
    .navbar-tagline {
      color: var(--ti-muted);
      font-size: 0.92rem;
      letter-spacing: 0.02em;
    }
    .navbar-tagline strong { color: var(--ti-ink); font-weight: 600; }
    @media (max-width: 767.98px) {
      .navbar-tagline { display: none; }
    }
  </style>
</head>
<body>
  <nav class="navbar border-bottom border-secondary-subtle px-3 py-3">
    <div class="container-xxl d-flex justify-content-between align-items-center gap-3">
      <div class="d-flex align-items-baseline gap-3 min-w-0">
        <a class="navbar-brand mb-0 h1" href="#">tiny<span>inference</span></a>
        <span class="navbar-tagline text-truncate">Own your inference — on <strong>your</strong> machine, not someone else's cloud.</span>
      </div>
      <div class="d-flex align-items-center gap-3 flex-shrink-0">
        <span id="statusBadge" class="status-stopped">· stopped</span>
        <button id="btnStartStop" class="btn btn-ice btn-sm">Start</button>
        <button id="btnRestart" class="btn btn-outline-ice btn-sm">Restart</button>
      </div>
    </div>
  </nav>

  <main class="container-xxl py-4">
    <p id="statusDetail" class="text-muted mb-4">Loading…</p>

    <ul class="nav nav-tabs mb-4" role="tablist">
      <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#dashboard" type="button">Dashboard</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#configure" type="button">Configure</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#logs" type="button">Logs</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#stats" type="button">Stats</button></li>
    </ul>

    <div class="tab-content">
      <section class="tab-pane fade show active" id="dashboard">
        <div class="row g-4">
          <div class="col-lg-8">
            <div class="card p-4">
              <h2 id="modelLabel" class="h4 mb-3">—</h2>
              <div id="downloadBlock" class="d-none mb-3">
                <div class="d-flex justify-content-between mb-1">
                  <span id="downloadSummary" class="text-info-emphasis" style="color:var(--ti-ice)!important"></span>
                  <span id="downloadProgress" class="text-muted"></span>
                </div>
                <div class="progress"><div id="downloadBar" class="progress-bar" style="width:0%"></div></div>
              </div>
              <dl class="row mb-0">
                <dt class="col-sm-3 text-muted">Model size</dt><dd class="col-sm-9" id="storageLine">—</dd>
                <dt class="col-sm-3 text-muted">RAM</dt><dd class="col-sm-9" id="ramLine">—</dd>
                <dt class="col-sm-3 text-muted">Runtime</dt><dd class="col-sm-9" id="runtimeLine">—</dd>
                <dt class="col-sm-3 text-muted">Access</dt><dd class="col-sm-9" id="accessLine">—</dd>
                <dt class="col-sm-3 text-muted">Endpoint</dt>
                <dd class="col-sm-9">
                  <a id="endpointLink" class="icon-link" href="#" target="_blank" rel="noopener noreferrer">
                    <span id="endpointLabel">—</span>
                    <svg xmlns="http://www.w3.org/2000/svg" class="external-icon bi" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" focusable="false">
                      <path fill-rule="evenodd" d="M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5"/>
                      <path fill-rule="evenodd" d="M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0z"/>
                    </svg>
                    <span class="visually-hidden">(opens in a new tab)</span>
                  </a>
                </dd>
              </dl>
            </div>
          </div>
          <div class="col-lg-4">
            <div class="card p-4 h-100">
              <div class="metric-label mb-2">Quick actions</div>
              <div class="d-grid gap-2">
                <a id="openServerUi" class="btn btn-outline-ice" href="#" target="_blank" rel="noopener noreferrer">
                  Open llama-server UI
                  <svg xmlns="http://www.w3.org/2000/svg" class="external-icon bi" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" focusable="false">
                    <path fill-rule="evenodd" d="M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5"/>
                    <path fill-rule="evenodd" d="M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0z"/>
                  </svg>
                  <span class="visually-hidden">(opens in a new tab)</span>
                </a>
                <button class="btn btn-outline-ice" id="btnCopyEndpoint">Copy /v1 URL</button>
                <button class="btn btn-outline-ice" id="btnCopyCommand">Copy launch command</button>
                <button class="btn btn-outline-light" id="btnSaveDash">Save config</button>
              </div>
              <hr class="border-secondary">
              <div class="small text-muted" id="machineLine">—</div>
              <div class="small text-muted mt-2" id="configPathLine">—</div>
            </div>
          </div>
        </div>
      </section>

      <section class="tab-pane fade" id="configure">
        <div class="card p-4">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h2 class="h5 mb-0">Settings</h2>
            <div class="d-flex gap-2">
              <button class="btn btn-outline-light btn-sm" id="btnReset">Reset to defaults</button>
              <button class="btn btn-ice btn-sm" id="btnSave">Save</button>
            </div>
          </div>
          <div id="recentModelsBlock" class="mb-4 d-none">
            <label class="form-label mb-1" for="recentModels">Recent models</label>
            <select class="form-select" id="recentModels"></select>
            <div class="setting-hint mt-1">Choose a previously used Hugging Face repo or local GGUF path.</div>
          </div>
          <div class="size-panel mb-4">
            <div class="metric-label mb-1">Model file size</div>
            <div id="mappedSizeValue" class="size-value is-muted">—</div>
            <div id="mappedSizeDetail" class="setting-hint mt-2 mb-0">—</div>
          </div>
          <div id="settingsForm" class="row g-3"></div>
          <p id="settingsError" class="text-danger mt-3 mb-0 d-none"></p>
        </div>
      </section>

      <section class="tab-pane fade" id="logs">
        <div class="card p-3">
          <div id="logView" class="log-view"></div>
        </div>
      </section>

      <section class="tab-pane fade" id="stats">
        <div class="row g-3" id="statsGrid"></div>
      </section>
    </div>
  </main>

  <div class="modal fade" id="missingServerModal" tabindex="-1" data-bs-backdrop="static">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header border-secondary">
          <h5 class="modal-title">llama-server not found</h5>
        </div>
        <div class="modal-body">
          Set the executable path before starting inference.
        </div>
        <div class="modal-footer border-secondary">
          <button type="button" class="btn btn-outline-light" id="btnDismissPrompt">Later</button>
          <button type="button" class="btn btn-ice" id="btnConfigureServer">Configure</button>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    const missingModal = new bootstrap.Modal('#missingServerModal');
    let state = null;
    let settingsDirty = false;
    let prompted = false;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Request failed');
      return data;
    }

    function statusClass(status) {
      return 'status-' + status;
    }

    function formatBytes(bytes) {
      const units = [['TiB', 1024**4], ['GiB', 1024**3], ['MiB', 1024**2], ['KiB', 1024]];
      for (const [unit, scale] of units) {
        if (bytes >= scale) return (bytes / scale).toFixed(1) + ' ' + unit;
      }
      return bytes + ' B';
    }

    function formatDuration(seconds) {
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      if (h > 0) return h + 'h ' + m + 'm';
      if (m > 0) return m + 'm ' + s + 's';
      return s + 's';
    }

    function metric(value, suffix = '') {
      if (value === null || value === undefined) return 'unavailable';
      if (Number.isInteger(value)) return value + suffix;
      return value.toFixed(2) + suffix;
    }

    function applyState(next) {
      state = next;
      const badge = document.getElementById('statusBadge');
      badge.className = statusClass(next.status);
      badge.textContent = '· ' + next.status_label;

      let detail = next.status_detail;
      if (next.download) detail = next.download.summary;
      else if (next.pending_changes) detail = 'Settings changed · restart to apply';
      document.getElementById('statusDetail').textContent = detail;

      document.getElementById('btnStartStop').textContent = next.running ? 'Stop' : 'Start';
      document.getElementById('btnStartStop').className = next.running ? 'btn btn-coral btn-sm' : 'btn btn-ice btn-sm';

      document.getElementById('modelLabel').textContent = next.model;
      document.getElementById('storageLine').textContent =
        next.mapped_size.gib != null
          ? next.mapped_size.display + ' mapped from disk · does not need to fit in RAM'
          : next.mapped_size.display + ' · ' + next.mapped_size.detail;

      const sizeValue = document.getElementById('mappedSizeValue');
      sizeValue.textContent = next.mapped_size.display;
      sizeValue.classList.toggle('is-muted', next.mapped_size.gib == null);
      document.getElementById('mappedSizeDetail').textContent = next.mapped_size.detail;
      document.getElementById('ramLine').textContent =
        next.machine.available_gib.toFixed(1) + ' GiB available · ' + next.machine.total_gib.toFixed(1) + ' GiB total';
      document.getElementById('runtimeLine').textContent = next.machine.runtime_summary;
      document.getElementById('accessLine').textContent = next.machine.access_summary;
      const endpoint = document.getElementById('endpointLink');
      const openUi = document.getElementById('openServerUi');
      document.getElementById('endpointLabel').textContent = next.endpoint;
      endpoint.href = next.endpoint;
      endpoint.style.color = next.endpoint_online ? 'var(--ti-mint)' : 'var(--ti-muted)';
      openUi.href = next.endpoint;
      openUi.classList.toggle('disabled', !next.running);
      openUi.setAttribute('aria-disabled', next.running ? 'false' : 'true');

      const downloadBlock = document.getElementById('downloadBlock');
      if (next.download) {
        downloadBlock.classList.remove('d-none');
        document.getElementById('downloadSummary').textContent = next.download.summary;
        document.getElementById('downloadProgress').textContent = next.download.progress;
        const pct = next.download.fraction != null ? (next.download.fraction * 100) : 0;
        document.getElementById('downloadBar').style.width = pct + '%';
      } else {
        downloadBlock.classList.add('d-none');
      }

      document.getElementById('machineLine').textContent =
        next.machine.cpu_name +
        ' · ' + next.machine.physical_cpus + ' physical / ' +
        next.machine.logical_cpus + ' logical · llama-server --threads ' +
        next.machine.physical_cpus;
      document.getElementById('configPathLine').textContent = 'Config: ' + next.config_path;

      renderRecentModels(next);
      renderSettings(next);
      renderLogs(next);
      renderStats(next);

      if (next.missing_server && !prompted) {
        prompted = true;
        missingModal.show();
      }
      if (!next.missing_server) {
        missingModal.hide();
      }
    }

    function renderRecentModels(next) {
      const block = document.getElementById('recentModelsBlock');
      const select = document.getElementById('recentModels');
      if (document.activeElement === select) return;
      if (!next.recent_models.length) {
        block.classList.add('d-none');
        select.innerHTML = '';
        return;
      }
      block.classList.remove('d-none');
      const selected = next.recent_models.find((model) => model.selected);
      select.innerHTML = next.recent_models.map((model) =>
        `<option value="${model.index}">${model.kind} · ${escapeHtml(model.label)}</option>`
      ).join('');
      select.value = String(selected ? selected.index : next.recent_models[0].index);
    }

    function escapeHtml(value) {
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function renderSettings(next) {
      const form = document.getElementById('settingsForm');
      // Polling rebuilds this DOM; skip while the user is interacting so focus
      // and in-progress edits are not blown away every second.
      if (settingsDirty || form.contains(document.activeElement)) return;
      form.innerHTML = '';
      for (const setting of next.settings) {
        const col = document.createElement('div');
        col.className = 'col-md-6';
        const label = document.createElement('label');
        label.className = 'form-label mb-1';
        label.textContent = setting.label;
        label.setAttribute('for', 'setting-' + setting.id);
        col.appendChild(label);

        if (setting.field === 'source_kind') {
          const select = document.createElement('select');
          select.className = 'form-select';
          select.id = 'setting-' + setting.id;
          select.innerHTML = '<option value="hugging_face">Hugging Face</option><option value="local">Local GGUF</option>';
          select.value = setting.raw;
          select.addEventListener('change', () => updateSetting(setting.field, select.value));
          col.appendChild(select);
        } else if (setting.choices && setting.choices.length) {
          const select = document.createElement('select');
          select.className = 'form-select';
          select.id = 'setting-' + setting.id;
          select.innerHTML = setting.choices.map((choice) =>
            `<option value="${choice}">${choice}</option>`
          ).join('');
          select.value = setting.raw;
          select.disabled = !setting.editable;
          select.addEventListener('change', () => updateSetting(setting.field, select.value));
          col.appendChild(select);
        } else if (setting.toggle) {
          const wrap = document.createElement('div');
          wrap.className = 'form-check form-switch mt-1';
          const input = document.createElement('input');
          input.className = 'form-check-input';
          input.type = 'checkbox';
          input.id = 'setting-' + setting.id;
          input.checked = setting.raw === 'true';
          input.addEventListener('change', () => api('/api/settings/' + setting.field + '/toggle', { method: 'POST' }).then(applyState).catch(showSettingsError));
          const checkLabel = document.createElement('label');
          checkLabel.className = 'form-check-label';
          checkLabel.setAttribute('for', input.id);
          checkLabel.textContent = setting.value;
          wrap.appendChild(input);
          wrap.appendChild(checkLabel);
          col.appendChild(wrap);
        } else {
          const input = document.createElement('input');
          input.className = 'form-control';
          input.id = 'setting-' + setting.id;
          input.value = setting.raw;
          input.disabled = !setting.editable;
          input.addEventListener('input', () => { settingsDirty = true; });
          input.addEventListener('change', () => {
            if (!setting.editable) return;
            updateSetting(setting.field, input.value).finally(() => { settingsDirty = false; });
          });
          col.appendChild(input);
        }

        const hint = document.createElement('div');
        hint.className = 'setting-hint mt-1';
        hint.textContent = setting.hint;
        col.appendChild(hint);
        form.appendChild(col);
      }
    }

    function renderLogs(next) {
      const view = document.getElementById('logView');
      const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 24;
      view.textContent = next.logs.join('\n');
      if (atBottom) view.scrollTop = view.scrollHeight;
    }

    function renderStats(next) {
      const grid = document.getElementById('statsGrid');
      const cards = [
        ['Endpoint', next.endpoint_online ? 'online' : 'offline'],
        ['PID', next.pid != null ? String(next.pid) : '—'],
        ['Uptime', next.process ? formatDuration(next.process.uptime_seconds) : '—'],
        ['CPU', next.process ? next.process.cpu_percent.toFixed(1) + '%' : '—'],
        ['RSS', next.process ? next.process.resident_memory_gib.toFixed(2) + ' GiB' : '—'],
        ['VIRT', next.process ? next.process.virtual_memory_gib.toFixed(2) + ' GiB' : '—'],
        ['Prompt tokens', next.metrics ? metric(next.metrics.prompt_tokens) : 'unavailable'],
        ['Generated tokens', next.metrics ? metric(next.metrics.generated_tokens) : 'unavailable'],
        ['Prompt tok/s', next.metrics ? metric(next.metrics.prompt_tokens_per_second) : 'unavailable'],
        ['Gen tok/s', next.metrics ? metric(next.metrics.generated_tokens_per_second) : 'unavailable'],
        ['Requests', next.metrics ? metric(next.metrics.requests_processing) : 'unavailable'],
        ['Deferred', next.metrics ? metric(next.metrics.requests_deferred) : 'unavailable'],
      ];
      grid.innerHTML = cards.map(([label, value]) =>
        `<div class="col-6 col-md-4 col-xl-3"><div class="card p-3 h-100"><div class="metric-label">${label}</div><div class="fs-5 mt-1">${value}</div></div></div>`
      ).join('');
    }

    async function updateSetting(field, value) {
      try {
        document.getElementById('settingsError').classList.add('d-none');
        const next = await api('/api/settings', {
          method: 'POST',
          body: JSON.stringify({ field, value }),
        });
        applyState(next);
      } catch (error) {
        showSettingsError(error);
      }
    }

    function showSettingsError(error) {
      const el = document.getElementById('settingsError');
      el.textContent = error.message || String(error);
      el.classList.remove('d-none');
    }

    async function refresh() {
      try {
        const next = await api('/api/state');
        applyState(next);
      } catch (error) {
        document.getElementById('statusDetail').textContent = error.message;
      }
    }

    document.getElementById('btnStartStop').addEventListener('click', async () => {
      applyState(await api(state?.running ? '/api/stop' : '/api/start', { method: 'POST' }));
    });
    document.getElementById('btnRestart').addEventListener('click', async () => {
      applyState(await api('/api/restart', { method: 'POST' }));
    });
    document.getElementById('btnReset').addEventListener('click', async () => {
      settingsDirty = false;
      applyState(await api('/api/reset', { method: 'POST' }));
    });
    document.getElementById('btnSave').addEventListener('click', async () => {
      settingsDirty = false;
      applyState(await api('/api/save', { method: 'POST' }));
    });
    document.getElementById('btnSaveDash').addEventListener('click', async () => {
      applyState(await api('/api/save', { method: 'POST' }));
    });
    document.getElementById('recentModels').addEventListener('change', async (event) => {
      settingsDirty = false;
      try {
        document.getElementById('settingsError').classList.add('d-none');
        applyState(await api('/api/settings/model/select', {
          method: 'POST',
          body: JSON.stringify({ index: Number(event.target.value) }),
        }));
      } catch (error) {
        showSettingsError(error);
      }
    });
    document.getElementById('btnCopyEndpoint').addEventListener('click', async () => {
      const data = await api('/api/copy/endpoint', { method: 'POST' });
      try { await navigator.clipboard.writeText(data.value); } catch (_) {}
      applyState(data.state);
    });
    document.getElementById('btnCopyCommand').addEventListener('click', async () => {
      const data = await api('/api/copy/command', { method: 'POST' });
      try { await navigator.clipboard.writeText(data.value); } catch (_) {}
      applyState(data.state);
    });
    document.getElementById('btnDismissPrompt').addEventListener('click', async () => {
      prompted = true;
      applyState(await api('/api/dismiss-prompt', { method: 'POST' }));
    });
    document.getElementById('btnConfigureServer').addEventListener('click', async () => {
      prompted = true;
      applyState(await api('/api/configure-server', { method: 'POST' }));
      const tab = document.querySelector('[data-bs-target="#configure"]');
      bootstrap.Tab.getOrCreateInstance(tab).show();
      setTimeout(() => document.getElementById('setting-executable')?.focus(), 200);
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"##;
