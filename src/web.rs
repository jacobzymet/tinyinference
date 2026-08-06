use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::{
    agent::{self, AgentRequest},
    app::{
        App, Download, LibraryFetch, NetworkUpdate, PRIMARY_SERVER_ID, ServerStatus, ServerSummary,
        SettingField,
    },
    chat,
    config::RuntimePreset,
    network::{
        ApiKeyPublic, DiscoveredPeer, InferenceMode, LinkedRemotePublic, ListenCandidate,
        RemoteHealth, ShareUrl, mask_token,
    },
    server::ServerProcess,
};

pub type SharedApp = Arc<Mutex<App>>;

/// Marker returned by `POST /api/focus`, used by a second launch to tell a
/// running tinyinference apart from an unrelated program on the same port.
pub const INSTANCE_MARKER: &str = "tinyinference";

pub async fn serve(app: SharedApp, listener: TcpListener) -> anyhow::Result<()> {
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

    let api = Router::new()
        .route("/api/chat/completions", post(chat_completions))
        .route("/api/state", get(state))
        .route("/api/ui/theme", post(set_ui_theme))
        .route("/api/ui/appearance", post(set_ui_appearance))
        .route("/api/ui/appearance/reset", post(reset_ui_appearance))
        .route("/api/network", get(network_state).post(update_network))
        .route("/api/network/keys", post(create_api_key))
        .route("/api/network/keys/{id}", axum::routing::delete(delete_api_key).patch(rename_api_key))
        .route("/api/network/keys/{id}/regenerate", post(regenerate_api_key))
        .route("/api/network/remotes", post(create_linked_remote))
        .route(
            "/api/network/remotes/{id}",
            axum::routing::patch(update_linked_remote).delete(delete_linked_remote),
        )
        .route("/api/network/remotes/{id}/activate", post(activate_linked_remote))
        .route("/api/start", post(start))
        .route("/api/stop", post(stop))
        .route("/api/restart", post(restart))
        .route("/api/servers/select", post(select_server))
        .route("/api/save", post(save))
        .route("/api/reset", post(reset))
        .route("/api/presets/{preset}", post(apply_preset))
        .route("/api/settings", post(update_setting))
        .route("/api/settings/{field}/toggle", post(toggle_setting))
        .route("/api/settings/model/select", post(select_model))
        .route("/api/settings/model/delete", post(delete_model))
        .route("/api/models/add", post(add_model))
        .route("/api/models/select", post(select_model_by_id))
        .route("/api/models/delete", post(delete_model_by_id))
        .route("/api/models/import", post(import_model))
        .route("/api/models/download", post(download_model))
        .route("/api/models/download/cancel", post(cancel_download_model))
        .route("/api/copy/endpoint", post(copy_endpoint))
        .route("/api/copy/command", post(copy_command))
        .route("/api/copy/logs", post(copy_logs))
        .route("/api/dismiss-prompt", post(dismiss_prompt))
        .route("/api/configure-server", post(configure_server))
        .route("/api/focus", post(focus))
        .route("/api/skills", get(list_skills).post(create_skill))
        .route("/api/skills/import", post(import_skill))
        .route(
            "/api/skills/{id}",
            axum::routing::patch(update_skill).delete(delete_skill),
        );

    let router = Router::new()
        .route("/", get(index))
        .route("/chat", get(chat_page))
        .route("/orb.js", get(orb_script))
        .route("/ti.png", get(app_icon_png))
        .route("/ti-transparent-bg-white.png", get(ui_mark_white))
        .route("/ti-transparent-bg-black.png", get(ui_mark_black))
        .route("/favicon.ico", get(app_icon_png))
        .merge(api)
        .with_state(app);

    axum::serve(listener, router.into_make_service())
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

async fn chat_page() -> Html<&'static str> {
    Html(CHAT_HTML)
}

async fn orb_script() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        ORB_JS,
    )
}

/// App / favicon icon (solid `ti.png`).
async fn app_icon_png() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/png")],
        APP_ICON_PNG,
    )
}

/// Dark-UI wordmark mark (white glyphs on transparent).
async fn ui_mark_white() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/png")],
        UI_MARK_WHITE_PNG,
    )
}

/// Light-UI wordmark mark (black glyphs on transparent).
async fn ui_mark_black() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/png")],
        UI_MARK_BLACK_PNG,
    )
}

async fn set_ui_theme(
    State(app): State<SharedApp>,
    Json(body): Json<ThemeRequest>,
) -> Result<Json<AppState>, ApiError> {
    let theme = crate::config::UiTheme::parse(&body.theme).ok_or_else(|| {
        ApiError::bad_request("theme must be \"dark\", \"light\", or \"system\"".into())
    })?;
    with_app(app, |app| app.set_ui_theme(theme))
}

async fn set_ui_appearance(
    State(app): State<SharedApp>,
    Json(body): Json<AppearanceRequest>,
) -> Result<Json<AppState>, ApiError> {
    let theme = match body.theme.as_deref() {
        None => None,
        Some(value) => Some(crate::config::UiTheme::parse(value).ok_or_else(|| {
            ApiError::bad_request("theme must be \"dark\", \"light\", or \"system\"".into())
        })?),
    };
    let font_scale = match body.font_scale.as_deref() {
        None => None,
        Some(value) => Some(crate::config::UiFontScale::parse(value).ok_or_else(|| {
            ApiError::bad_request("font_scale must be compact, default, or large".into())
        })?),
    };
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.set_ui_appearance(
        theme,
        body.font_body,
        body.font_display,
        body.font_mono,
        font_scale,
    )
    .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn reset_ui_appearance(
    State(app): State<SharedApp>,
) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.reset_ui_appearance())
}

async fn chat_completions(
    State(app): State<SharedApp>,
    Json(mut body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let server_id = body
        .as_object_mut()
        .and_then(|obj| obj.remove("server_id"))
        .and_then(|v| v.as_str().map(str::to_string));

    let (remote, user_skills) = {
        let app = app.lock().map_err(|_| ApiError::lock())?;
        let user_skills = app.enabled_user_skills();
        let network = &app.config.network;
        let remote = if network.inference_mode == InferenceMode::Remote {
            let Some(active) = network.active_remote() else {
                return Err(ApiError::bad_request(
                    "Remote inference is selected but no linked LLM is configured.".into(),
                ));
            };
            let Some(url) = network.remote_chat_url() else {
                return Err(ApiError::bad_request(
                    "Remote inference is selected but no remote URL is configured.".into(),
                ));
            };
            Some((url, active.token.clone()))
        } else {
            None
        };
        (remote, user_skills)
    };

    let stream = if let Some((chat_url, token)) = remote {
        // Linked OpenAI-compatible LLM. Agent capabilities still run on this machine;
        // only model tokens go to the remote `/v1` base.
        let api_base = chat_url
            .trim_end_matches('/')
            .strip_suffix("/chat/completions")
            .unwrap_or(chat_url.trim_end_matches('/'))
            .to_string();
        let key = (!token.trim().is_empty()).then_some(token.as_str());
        match serde_json::from_value::<AgentRequest>(body.clone()) {
            Ok(request) if agent::should_run_agent(&request, &user_skills) => {
                if request.messages.is_empty() {
                    return Err(ApiError::bad_request("messages must not be empty".into()));
                }
                agent::stream_agent(&api_base, key, request, user_skills)
            }
            _ => {
                if let Some(messages) = body.get_mut("messages").and_then(|v| v.as_array_mut()) {
                    agent::inject_skill_catalog_into_messages(messages, &user_skills);
                }
                chat::stream_remote_completion(&chat_url, &token, body)
            }
        }
    } else {
        let (api_base, api_key) = {
            let app = app.lock().map_err(|_| ApiError::lock())?;
            let id = server_id
                .as_deref()
                .unwrap_or(app.active_server_id.as_str());
            let Some(lookup) = app.server_by_id(id) else {
                return Err(ApiError::bad_request(
                    "The selected model server is not running. Start it from the dashboard first."
                        .into(),
                ));
            };
            if !lookup.ready {
                return Err(ApiError::bad_request(
                    "The selected model server is not ready yet.".into(),
                ));
            }
            (
                lookup.config.api_endpoint(),
                lookup.config.server.api_key.clone(),
            )
        };
        let key_owned = api_key;
        let api_key = (!key_owned.trim().is_empty()).then_some(key_owned.as_str());

        // Agent mode runs a tool loop server-side; plain chat still streams through.
        // Strip tinyinference-only fields before proxying to llama-server.
        match serde_json::from_value::<AgentRequest>(body.clone()) {
            Ok(request) if agent::should_run_agent(&request, &user_skills) => {
                if request.messages.is_empty() {
                    return Err(ApiError::bad_request("messages must not be empty".into()));
                }
                agent::stream_agent(&api_base, api_key, request, user_skills)
            }
            _ => {
                let mut upstream = body;
                if let Some(object) = upstream.as_object_mut() {
                    object.remove("agent");
                    object.remove("skills");
                    if let Some(messages) = object.get_mut("messages").and_then(|v| v.as_array_mut())
                    {
                        agent::inject_skill_catalog_into_messages(messages, &user_skills);
                    }
                }
                chat::stream_completion(&api_base, api_key, upstream)
            }
        }
    };

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    Ok(response)
}

async fn network_state(State(app): State<SharedApp>) -> Result<Json<NetworkState>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    Ok(Json(NetworkState::from_app(&app)))
}

async fn update_network(
    State(app): State<SharedApp>,
    Json(update): Json<NetworkUpdate>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .apply_network_update(update)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

#[derive(Debug, Deserialize)]
struct CreateApiKeyBody {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RenameApiKeyBody {
    name: String,
}

async fn create_api_key(
    State(app): State<SharedApp>,
    Json(body): Json<CreateApiKeyBody>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .create_api_key(&body.name)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn rename_api_key(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
    Json(body): Json<RenameApiKeyBody>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .rename_api_key(&id, &body.name)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn regenerate_api_key(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .regenerate_api_key(&id)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn delete_api_key(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app.delete_api_key(&id).map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

#[derive(Debug, Deserialize)]
struct CreateLinkedRemoteBody {
    #[serde(default)]
    name: String,
    base: String,
    #[serde(default)]
    token: String,
    #[serde(default = "default_true")]
    activate: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct UpdateLinkedRemoteBody {
    name: Option<String>,
    base: Option<String>,
    token: Option<String>,
}

async fn create_linked_remote(
    State(app): State<SharedApp>,
    Json(body): Json<CreateLinkedRemoteBody>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .create_linked_remote(&body.name, &body.base, &body.token, body.activate)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn update_linked_remote(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
    Json(body): Json<UpdateLinkedRemoteBody>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .update_linked_remote(
            &id,
            body.name.as_deref(),
            body.base.as_deref(),
            body.token.as_deref(),
        )
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn delete_linked_remote(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .delete_linked_remote(&id)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

async fn activate_linked_remote(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
) -> Result<Json<NetworkMutationResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let result = app
        .activate_linked_remote(&id)
        .map_err(ApiError::bad_request)?;
    Ok(Json(NetworkMutationResponse::from_result(&app, result)))
}

#[derive(Debug, Serialize)]
struct SkillsState {
    skills: Vec<crate::skills::UserSkillPublic>,
}

#[derive(Debug, Deserialize)]
struct ImportSkillBody {
    content: String,
    #[serde(default)]
    filename: Option<String>,
}

async fn list_skills(State(app): State<SharedApp>) -> Result<Json<SkillsState>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    let skills = app
        .list_user_skills()
        .map_err(ApiError::bad_request)?
        .into_iter()
        .map(|skill| skill.to_public())
        .collect();
    Ok(Json(SkillsState { skills }))
}

async fn create_skill(
    State(app): State<SharedApp>,
    Json(body): Json<crate::skills::SkillUpsert>,
) -> Result<Json<crate::skills::UserSkillPublic>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    let skill = app
        .create_user_skill(body)
        .map_err(ApiError::bad_request)?;
    Ok(Json(skill.to_public()))
}

async fn import_skill(
    State(app): State<SharedApp>,
    Json(body): Json<ImportSkillBody>,
) -> Result<Json<crate::skills::UserSkillPublic>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    let skill = app
        .import_user_skill(body.filename.as_deref(), &body.content)
        .map_err(ApiError::bad_request)?;
    Ok(Json(skill.to_public()))
}

async fn update_skill(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
    Json(body): Json<crate::skills::SkillUpsert>,
) -> Result<Json<crate::skills::UserSkillPublic>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    let skill = app
        .update_user_skill(&id, body)
        .map_err(ApiError::bad_request)?;
    Ok(Json(skill.to_public()))
}

async fn delete_skill(
    State(app): State<SharedApp>,
    Path(id): Path<String>,
) -> Result<Json<SkillsState>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    app.delete_user_skill(&id).map_err(ApiError::bad_request)?;
    let skills = app
        .list_user_skills()
        .map_err(ApiError::bad_request)?
        .into_iter()
        .map(|skill| skill.to_public())
        .collect();
    Ok(Json(SkillsState { skills }))
}

async fn state(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    Ok(Json(AppState::from_app(&app)))
}

async fn start(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    with_app(app, |app| app.start())
}

#[derive(Debug, Default, Deserialize)]
struct ServerIdBody {
    id: Option<String>,
}

async fn stop(
    State(app): State<SharedApp>,
    body: Option<Json<ServerIdBody>>,
) -> Result<Json<AppState>, ApiError> {
    let id = body.and_then(|Json(b)| b.id);
    with_app(app, |app| app.stop_server(id.as_deref()))
}

async fn restart(
    State(app): State<SharedApp>,
    body: Option<Json<ServerIdBody>>,
) -> Result<Json<AppState>, ApiError> {
    let id = body.and_then(|Json(b)| b.id);
    with_app(app, move |app| {
        if let Some(id) = id.as_deref() {
            let _ = app.select_server(id);
        }
        app.restart();
    })
}

async fn select_server(
    State(app): State<SharedApp>,
    Json(body): Json<ServerIdBody>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let id = body.id.as_deref().unwrap_or(PRIMARY_SERVER_ID);
    app.select_server(id).map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
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

async fn delete_model(
    State(app): State<SharedApp>,
    Json(body): Json<SelectRecentModel>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.delete_picker_model(body.index)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn select_model_by_id(
    State(app): State<SharedApp>,
    Json(body): Json<ModelIdRequest>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.select_library_model_by_label(&body.id)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn delete_model_by_id(
    State(app): State<SharedApp>,
    Json(body): Json<ModelIdRequest>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    match body.scope.as_deref() {
        Some("available") => app
            .delete_available_model_by_label(&body.id)
            .map_err(ApiError::bad_request)?,
        _ => app
            .delete_library_model_by_label(&body.id)
            .map_err(ApiError::bad_request)?,
    }
    Ok(Json(AppState::from_app(&app)))
}

async fn import_model(
    State(app): State<SharedApp>,
    Json(body): Json<ModelIdRequest>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.import_available_model(&body.id)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn add_model(
    State(app): State<SharedApp>,
    Json(body): Json<AddModelRequest>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.add_model(&body.kind, &body.value, body.download)
        .map_err(ApiError::bad_request)?;
    Ok(Json(AppState::from_app(&app)))
}

async fn download_model(
    State(app): State<SharedApp>,
    Json(body): Json<ModelIdRequest>,
) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    if body.id.is_empty() {
        app.start_library_download_for_index(body.index.unwrap_or(0))
            .map_err(ApiError::bad_request)?;
    } else {
        app.start_library_download_by_label(&body.id)
            .map_err(ApiError::bad_request)?;
    }
    Ok(Json(AppState::from_app(&app)))
}

async fn cancel_download_model(State(app): State<SharedApp>) -> Result<Json<AppState>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    app.cancel_library_download()
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

async fn copy_logs(State(app): State<SharedApp>) -> Result<Json<CopyResponse>, ApiError> {
    let mut app = app.lock().map_err(|_| ApiError::lock())?;
    let value = app.copy_logs();
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

/// Identify this instance and raise its window.
///
/// A second launch that finds the port taken posts here: a tinyinference reply
/// means "already running, come to the front"; anything else means the port
/// belongs to an unrelated program.
async fn focus(State(app): State<SharedApp>) -> Result<Json<InstanceInfo>, ApiError> {
    let app = app.lock().map_err(|_| ApiError::lock())?;
    Ok(Json(InstanceInfo {
        app: INSTANCE_MARKER,
        version: env!("CARGO_PKG_VERSION"),
        focused: app.request_focus(),
    }))
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

#[derive(Debug, Deserialize)]
struct ModelIdRequest {
    #[serde(default)]
    id: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddModelRequest {
    kind: String,
    value: String,
    #[serde(default)]
    download: bool,
}

#[derive(Debug, Serialize)]
struct InstanceInfo {
    app: &'static str,
    version: &'static str,
    focused: bool,
}

#[derive(Debug, Serialize)]
struct CopyResponse {
    value: String,
    state: AppState,
}

#[derive(Debug, Deserialize)]
struct ThemeRequest {
    theme: String,
}

#[derive(Debug, Deserialize)]
struct AppearanceRequest {
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    font_body: Option<String>,
    #[serde(default)]
    font_display: Option<String>,
    #[serde(default)]
    font_mono: Option<String>,
    #[serde(default)]
    font_scale: Option<String>,
}

#[derive(Debug, Serialize)]
struct AppState {
    status: ServerStatus,
    status_label: &'static str,
    status_detail: String,
    theme: &'static str,
    font_body: String,
    font_display: String,
    font_mono: String,
    font_scale: &'static str,
    running: bool,
    endpoint_online: bool,
    thinking_supported: bool,
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
    library_download: Option<DownloadState>,
    process: Option<ProcessState>,
    metrics: Option<MetricsState>,
    recent_models: Vec<RecentModelState>,
    library: Vec<RecentModelState>,
    available: Vec<RecentModelState>,
    has_active_model: bool,
    settings: Vec<SettingState>,
    cache_type_help: Vec<ChoiceHelp>,
    presets: Vec<PresetState>,
    active_preset: Option<&'static str>,
    logs: Vec<String>,
    network: NetworkSummary,
    servers: Vec<ServerSummary>,
    active_server_id: String,
    running_server_count: usize,
}

#[derive(Debug, Serialize)]
struct NetworkSummary {
    expose: bool,
    listening_exposed: bool,
    restart_required: bool,
    /// Resolved advertise / display name for this PC.
    device_name: String,
    inference_mode: &'static str,
    remote_base: String,
    remote_label: String,
    remote_name: String,
    active_remote_id: String,
    remote_count: usize,
    /// At least one linked LLM is saved in config.
    remote_saved: bool,
    /// Chat is currently routed to the active linked LLM.
    via_remote: bool,
    remote_ok: bool,
    remote_model: Option<String>,
    remote_status: Option<String>,
}

#[derive(Debug, Serialize)]
struct NetworkState {
    expose: bool,
    listening_exposed: bool,
    restart_required: bool,
    listen_scope: &'static str,
    listen_scope_label: &'static str,
    listen_scope_hint: &'static str,
    listen_scope_detail: &'static str,
    listen_host: String,
    listen_bind_host: Option<String>,
    listen_bind_error: Option<String>,
    listen_candidates: Vec<ListenCandidate>,
    tailscale_available: bool,
    access_token_set: bool,
    access_token_masked: String,
    api_keys: Vec<ApiKeyPublic>,
    /// Custom name from config (empty → use hostname). Bound to the Device name field.
    device_name_custom: String,
    /// Resolved advertise / display name (custom or hostname).
    device_name: String,
    inference_mode: &'static str,
    remotes: Vec<LinkedRemotePublic>,
    active_remote_id: String,
    remote_base: String,
    remote_name: String,
    /// At least one linked LLM is saved in config.
    remote_saved: bool,
    remote_token_set: bool,
    remote_token_masked: String,
    share_urls: Vec<ShareUrl>,
    peers: Vec<DiscoveredPeer>,
    mdns_error: Option<String>,
    advertising: bool,
    discovery_hint: String,
    remote_health: Option<RemoteHealth>,
    llama_binds_loopback: bool,
    llama_endpoint: Option<String>,
}

#[derive(Debug, Serialize)]
struct NetworkMutationResponse {
    restart_required: bool,
    access_token: Option<String>,
    revealed_api_key_id: Option<String>,
    network: NetworkState,
    state: AppState,
}

impl NetworkMutationResponse {
    fn from_result(app: &App, result: crate::app::NetworkUpdateResult) -> Self {
        Self {
            restart_required: result.restart_required,
            access_token: result.access_token,
            revealed_api_key_id: result.revealed_api_key_id,
            network: NetworkState::from_app(app),
            state: AppState::from_app(app),
        }
    }
}

impl NetworkState {
    fn from_app(app: &App) -> Self {
        let network = &app.config.network;
        let host = app.config.effective_host();
        let llama_loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
        let scope = network.listen_scope;
        let candidates = app.listen_candidates();
        let tailscale_available = candidates.iter().any(|c| c.kind == "tailscale");
        let listen_bind_host = network.resolve_listen_host().ok();
        let primary = network.primary_api_key().unwrap_or("");
        let active_id = network
            .active_remote()
            .map(|remote| remote.id.clone())
            .unwrap_or_default();
        let remotes = network
            .remotes
            .iter()
            .map(|remote| LinkedRemotePublic {
                id: remote.id.clone(),
                name: remote.name.clone(),
                base: remote.base.clone(),
                token_set: !remote.token.trim().is_empty(),
                token_masked: mask_token(&remote.token),
                active: remote.id == active_id
                    && network.inference_mode == InferenceMode::Remote,
                health: Some(app.remote_health_for(&remote.base, &remote.token)),
            })
            .collect::<Vec<_>>();
        let active = network.active_remote();
        Self {
            expose: network.expose,
            listening_exposed: app.listening_exposed(),
            restart_required: app.network_restart_required(),
            listen_scope: scope.as_str(),
            listen_scope_label: scope.novice_label(),
            listen_scope_hint: scope.novice_hint(),
            listen_scope_detail: scope.technical_detail(),
            listen_host: network.listen_host.clone(),
            listen_bind_host,
            listen_bind_error: app.listen_bind_error(),
            listen_candidates: candidates,
            tailscale_available,
            access_token_set: !primary.is_empty(),
            access_token_masked: mask_token(primary),
            api_keys: network.public_api_keys(),
            device_name_custom: network.device_name.clone(),
            device_name: network.resolved_device_name(),
            inference_mode: network.inference_mode.as_str(),
            remotes,
            active_remote_id: active_id,
            remote_base: active.map(|r| r.base.clone()).unwrap_or_default(),
            remote_name: active.map(|r| r.name.clone()).unwrap_or_default(),
            remote_saved: !network.remotes.is_empty(),
            remote_token_set: active.is_some_and(|r| !r.token.trim().is_empty()),
            remote_token_masked: mask_token(active.map(|r| r.token.as_str()).unwrap_or("")),
            share_urls: app.network_share_urls(),
            peers: app.discovered_peers(),
            mdns_error: app.mdns_error(),
            advertising: app.discovery_advertising(),
            discovery_hint: app.discovery_hint(),
            remote_health: app.remote_health(),
            llama_binds_loopback: llama_loopback,
            llama_endpoint: if network.expose {
                Some(format!(
                    "{}:{}",
                    app.config.effective_host(),
                    app.config.effective_port()
                ))
            } else {
                None
            },
        }
    }
}

impl NetworkSummary {
    fn from_app(app: &App) -> Self {
        let network = &app.config.network;
        let remote_saved = !network.remotes.is_empty();
        let active = network.active_remote();
        let via_remote = network.inference_mode == InferenceMode::Remote && active.is_some();
        let remote_label = active
            .map(|remote| {
                remote
                    .base
                    .trim()
                    .trim_start_matches("http://")
                    .trim_start_matches("https://")
                    .trim_end_matches('/')
                    .to_string()
            })
            .unwrap_or_default();
        let remote_name = active
            .map(|remote| remote.name.clone())
            .unwrap_or_default();
        // Warm/cached probe for the active link so Dash can show reachability
        // even when chat is still on This PC. Cache keeps this cheap on polls.
        let health = if remote_saved { app.remote_health() } else { None };
        Self {
            expose: network.expose,
            listening_exposed: app.listening_exposed(),
            restart_required: app.network_restart_required(),
            device_name: network.resolved_device_name(),
            inference_mode: network.inference_mode.as_str(),
            remote_base: active.map(|r| r.base.clone()).unwrap_or_default(),
            remote_label,
            remote_name,
            active_remote_id: active.map(|r| r.id.clone()).unwrap_or_default(),
            remote_count: network.remotes.len(),
            remote_saved,
            via_remote,
            // Stay false until a probe succeeds — do not optimistically mark remotes ready.
            remote_ok: health.as_ref().is_some_and(|h| h.ok),
            remote_model: health.as_ref().and_then(|h| h.model.clone()),
            remote_status: health.as_ref().and_then(|h| h.status.clone()),
        }
    }
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

#[derive(Debug, Clone, Serialize)]
struct RecentModelState {
    index: usize,
    id: String,
    kind: &'static str,
    label: String,
    selected: bool,
    recent: bool,
    on_disk: bool,
    deletable: bool,
    downloadable: bool,
    downloading: bool,
    bytes: u64,
    size_display: Option<String>,
    status: &'static str,
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
        let library_download = app
            .library_fetch
            .as_ref()
            .filter(|fetch| fetch.is_active() || fetch.error.is_some())
            .map(DownloadState::from_library_fetch);
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
        let server_running = app.running_server_count() > 0;
        let servers = app.server_summaries();
        let active_extra = app
            .extra_servers
            .iter()
            .find(|s| s.id == app.active_server_id);
        let fetch_repo = app
            .library_fetch
            .as_ref()
            .filter(|fetch| fetch.is_active())
            .map(|fetch| fetch.repo.clone());
        let fetch_total = app.library_fetch.as_ref().and_then(|fetch| {
            if fetch.is_active() || fetch.done {
                fetch.total
            } else {
                None
            }
        });
        let mapped_bytes = app
            .mapped_model_gib()
            .map(|gib| (gib * 1024.0 * 1024.0 * 1024.0) as u64);
        let map_entry = |index: usize, entry: crate::app::ModelPickerEntry| {
            let (kind, label) = match &entry.source {
                crate::config::ModelSource::HuggingFace(id) => ("Hugging Face", id.clone()),
                crate::config::ModelSource::Local(path) => {
                    ("Local GGUF", path.display().to_string())
                }
            };
            let downloading = fetch_repo.as_ref().is_some_and(|repo| {
                matches!(
                    &entry.source,
                    crate::config::ModelSource::HuggingFace(id)
                        if repos_match(id, repo)
                )
            });
            let size_bytes = if downloading {
                fetch_total
                    .filter(|total| *total > 0)
                    .unwrap_or(entry.bytes)
            } else if entry.bytes > 0 {
                entry.bytes
            } else if entry.source == app.config.model.source {
                mapped_bytes.unwrap_or(0)
            } else {
                0
            };
            let size_display = format_size_label(size_bytes);
            let status = if downloading {
                "Downloading"
            } else if entry.on_disk {
                "Ready"
            } else {
                "Not downloaded"
            };
            let is_hf = matches!(&entry.source, crate::config::ModelSource::HuggingFace(_));
            RecentModelState {
                index,
                id: label.clone(),
                kind,
                label,
                selected: app.has_active_model() && entry.source == app.config.model.source,
                recent: entry.recent,
                on_disk: entry.on_disk,
                deletable: !server_running && !downloading,
                downloadable: is_hf && !server_running && !entry.on_disk && fetch_repo.is_none(),
                downloading,
                bytes: size_bytes,
                size_display,
                status,
            }
        };
        let library: Vec<RecentModelState> = app
            .library_entries()
            .into_iter()
            .enumerate()
            .map(|(index, entry)| map_entry(index, entry))
            .collect();
        let available: Vec<RecentModelState> = app
            .available_entries()
            .into_iter()
            .enumerate()
            .map(|(index, entry)| map_entry(index, entry))
            .collect();
        let recent_models = library.clone();

        let (
            status,
            status_label,
            endpoint_online,
            pid,
            process,
            metrics,
            download_state,
        ) = if let Some(extra) = active_extra {
            (
                extra.status,
                extra.status.label(),
                extra.endpoint_online,
                extra.process.as_ref().map(ServerProcess::id),
                extra.process_usage.as_ref().map(|usage| ProcessState {
                    cpu_percent: usage.cpu_percent,
                    resident_memory_gib: usage.resident_memory_gib,
                    virtual_memory_gib: usage.virtual_memory_gib,
                    uptime_seconds: usage.uptime_seconds,
                }),
                extra.server_metrics.as_ref().map(|metrics| MetricsState {
                    prompt_tokens: metrics.prompt_tokens,
                    generated_tokens: metrics.generated_tokens,
                    prompt_tokens_per_second: metrics.prompt_tokens_per_second,
                    generated_tokens_per_second: metrics.generated_tokens_per_second,
                    requests_processing: metrics.requests_processing,
                    requests_deferred: metrics.requests_deferred,
                }),
                extra
                    .download
                    .as_ref()
                    .filter(|d| d.is_active())
                    .map(DownloadState::from_download),
            )
        } else {
            (
                app.status,
                app.status.label(),
                app.endpoint_online,
                app.process.as_ref().map(ServerProcess::id),
                app.process_usage.as_ref().map(|usage| ProcessState {
                    cpu_percent: usage.cpu_percent,
                    resident_memory_gib: usage.resident_memory_gib,
                    virtual_memory_gib: usage.virtual_memory_gib,
                    uptime_seconds: usage.uptime_seconds,
                }),
                app.server_metrics.as_ref().map(|metrics| MetricsState {
                    prompt_tokens: metrics.prompt_tokens,
                    generated_tokens: metrics.generated_tokens,
                    prompt_tokens_per_second: metrics.prompt_tokens_per_second,
                    generated_tokens_per_second: metrics.generated_tokens_per_second,
                    requests_processing: metrics.requests_processing,
                    requests_deferred: metrics.requests_deferred,
                }),
                download.or_else(|| {
                    app.library_fetch
                        .as_ref()
                        .filter(|fetch| fetch.is_active())
                        .map(DownloadState::from_library_fetch)
                }),
            )
        };

        Self {
            status,
            status_label,
            status_detail: if let Some(extra) = active_extra {
                extra.status_detail.clone()
            } else {
                app.status_detail.clone()
            },
            theme: app.config.ui.theme.as_str(),
            font_body: app.config.ui.font_body.clone(),
            font_display: app.config.ui.font_display.clone(),
            font_mono: app.config.ui.font_mono.clone(),
            font_scale: app.config.ui.font_scale.as_str(),
            running: server_running,
            endpoint_online,
            thinking_supported: app.thinking_supported(),
            pending_changes: app.has_pending_changes(),
            missing_server: app.should_prompt_for_server(),
            model: config.model_label(),
            endpoint: config.endpoint(),
            api_endpoint: config.api_endpoint(),
            command: crate::server::CommandSpec::from_config(config).display(),
            config_path: app.config_path.display().to_string(),
            pid,
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
            download: download_state,
            library_download,
            process,
            metrics,
            recent_models,
            library,
            available,
            has_active_model: app.has_active_model(),
            settings,
            cache_type_help,
            presets,
            active_preset,
            logs: app.logs.iter().cloned().collect(),
            network: NetworkSummary::from_app(app),
            servers,
            active_server_id: app.active_server_id.clone(),
            running_server_count: app.running_server_count(),
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

    fn from_library_fetch(fetch: &LibraryFetch) -> Self {
        Self {
            repo: fetch.repo.clone(),
            file: fetch.file.clone(),
            downloaded: fetch.downloaded,
            total: fetch.total,
            fraction: fetch.fraction(),
            rate: fetch.rate(),
            eta_seconds: fetch.eta().map(|eta| eta.as_secs()),
            listing_error: fetch.error.clone(),
            summary: match &fetch.file {
                Some(file) => format!("Downloading {file} · {}", fetch.repo),
                None => format!("Downloading {}", fetch.repo),
            },
            progress: library_download_progress(fetch),
        }
    }
}

fn library_download_progress(fetch: &LibraryFetch) -> String {
    let downloaded = format_bytes(fetch.downloaded);
    match (fetch.total, fetch.rate(), fetch.eta()) {
        (Some(total), Some(rate), Some(eta)) => format!(
            "{downloaded} / {} · {}/s · ETA {}",
            format_bytes(total),
            format_bytes(rate as u64),
            format_eta(eta.as_secs())
        ),
        (Some(total), Some(rate), None) => {
            format!(
                "{downloaded} / {} · {}/s",
                format_bytes(total),
                format_bytes(rate as u64)
            )
        }
        (Some(total), None, _) => format!("{downloaded} / {}", format_bytes(total)),
        _ => downloaded,
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

fn repos_match(left: &str, right: &str) -> bool {
    fn strip(value: &str) -> &str {
        value.split(':').next().unwrap_or(value)
    }
    left == right || strip(left) == strip(right)
}

fn format_size_label(bytes: u64) -> Option<String> {
    if bytes == 0 {
        return None;
    }
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gib >= 0.01 {
        Some(format!("{gib:.2} GiB"))
    } else {
        let mib = bytes as f64 / (1024.0 * 1024.0);
        Some(format!("{mib:.1} MiB"))
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
const CHAT_HTML: &str = include_str!("chat.html");
const ORB_JS: &str = include_str!("orb.js");
const APP_ICON_PNG: &[u8] = include_bytes!("../assets/ti.png");
const UI_MARK_WHITE_PNG: &[u8] = include_bytes!("../assets/ti-transparent-bg-white.png");
const UI_MARK_BLACK_PNG: &[u8] = include_bytes!("../assets/ti-transparent-bg-black.png");
