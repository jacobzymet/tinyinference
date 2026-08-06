//! TLS share proxy in front of local llama-server.
//!
//! When Share is on, llama binds loopback HTTP without API keys. This module
//! terminates TLS on the public listen address, authenticates Bearer keys, and
//! records connected-client activity for the Devices UI.

use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, HeaderName, Method, StatusCode, Uri, header},
    response::{IntoResponse, Response},
};
use axum_server::tls_rustls::RustlsConfig;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use futures_util::StreamExt as _;
use serde::Serialize;

const IDLE_PRUNE: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientActivityState {
    Active,
    Idle,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectedClientPublic {
    pub id: String,
    pub remote_addr: String,
    pub key_id: String,
    pub key_name: String,
    pub local_port: u16,
    pub model: String,
    pub last_path: String,
    pub last_method: String,
    pub state: ClientActivityState,
    pub started_at: u64,
    pub last_seen: u64,
    pub tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShareActivitySummary {
    pub active_count: usize,
    pub idle_count: usize,
    pub tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone)]
struct ConnectedClient {
    id: String,
    remote_addr: String,
    key_id: String,
    key_name: String,
    local_port: u16,
    model: String,
    last_path: String,
    last_method: String,
    active_requests: u32,
    last_seen: Instant,
    started_at: u64,
    tokens_per_second: Option<f64>,
}

impl ConnectedClient {
    fn state(&self) -> ClientActivityState {
        if self.active_requests > 0 {
            ClientActivityState::Active
        } else {
            ClientActivityState::Idle
        }
    }

    fn to_public(&self) -> ConnectedClientPublic {
        ConnectedClientPublic {
            id: self.id.clone(),
            remote_addr: self.remote_addr.clone(),
            key_id: self.key_id.clone(),
            key_name: self.key_name.clone(),
            local_port: self.local_port,
            model: self.model.clone(),
            last_path: self.last_path.clone(),
            last_method: self.last_method.clone(),
            state: self.state(),
            started_at: self.started_at,
            last_seen: unix_secs(self.last_seen),
            tokens_per_second: if self.active_requests > 0 {
                self.tokens_per_second
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Default)]
struct ShareActivityStore {
    clients: HashMap<String, ConnectedClient>,
}

impl ShareActivityStore {
    fn begin_request(
        &mut self,
        remote_addr: &str,
        key_id: &str,
        key_name: &str,
        local_port: u16,
        model: &str,
        method: &str,
        path: &str,
    ) -> String {
        let id = format!("{key_id}|{remote_addr}|{local_port}");
        let now = Instant::now();
        let entry = self.clients.entry(id.clone()).or_insert_with(|| ConnectedClient {
            id: id.clone(),
            remote_addr: remote_addr.to_string(),
            key_id: key_id.to_string(),
            key_name: key_name.to_string(),
            local_port,
            model: model.to_string(),
            last_path: path.to_string(),
            last_method: method.to_string(),
            active_requests: 0,
            last_seen: now,
            started_at: unix_secs(now),
            tokens_per_second: None,
        });
        entry.key_name = key_name.to_string();
        entry.model = model.to_string();
        entry.last_path = path.to_string();
        entry.last_method = method.to_string();
        entry.active_requests = entry.active_requests.saturating_add(1);
        entry.last_seen = now;
        id
    }

    fn end_request(&mut self, id: &str) {
        let Some(entry) = self.clients.get_mut(id) else {
            return;
        };
        entry.active_requests = entry.active_requests.saturating_sub(1);
        entry.last_seen = Instant::now();
    }

    fn set_port_tps(&mut self, port: u16, tps: Option<f64>) {
        for client in self.clients.values_mut() {
            if client.local_port == port {
                client.tokens_per_second = tps;
            }
        }
    }

    fn snapshot(&mut self) -> (Vec<ConnectedClientPublic>, ShareActivitySummary) {
        let cutoff = Instant::now() - IDLE_PRUNE;
        self.clients
            .retain(|_, client| client.active_requests > 0 || client.last_seen >= cutoff);

        let mut clients: Vec<_> = self.clients.values().map(ConnectedClient::to_public).collect();
        clients.sort_by(|a, b| {
            b.last_seen
                .cmp(&a.last_seen)
                .then_with(|| a.key_name.cmp(&b.key_name))
                .then_with(|| a.remote_addr.cmp(&b.remote_addr))
        });

        let active_count = clients
            .iter()
            .filter(|c| c.state == ClientActivityState::Active)
            .count();
        let idle_count = clients.len().saturating_sub(active_count);
        let tokens_per_second = clients
            .iter()
            .filter_map(|c| c.tokens_per_second)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        (
            clients,
            ShareActivitySummary {
                active_count,
                idle_count,
                tokens_per_second,
            },
        )
    }
}

#[derive(Clone)]
struct ProxyState {
    local_port: u16,
    model: Arc<Mutex<String>>,
    keys: Arc<Mutex<Vec<(String, String, String)>>>,
    activity: Arc<Mutex<ShareActivityStore>>,
    client: Client<HttpConnector, Full<Bytes>>,
}

struct ProxyRunner {
    bind_host: String,
    cert: PathBuf,
    key: PathBuf,
    model: Arc<Mutex<String>>,
    handle: axum_server::Handle,
}

/// Manages one TLS listener per shared llama port.
pub struct ShareProxyManager {
    activity: Arc<Mutex<ShareActivityStore>>,
    keys: Arc<Mutex<Vec<(String, String, String)>>>,
    runners: HashMap<u16, ProxyRunner>,
    last_error: Option<String>,
}

impl Default for ShareProxyManager {
    fn default() -> Self {
        Self {
            activity: Arc::new(Mutex::new(ShareActivityStore::default())),
            keys: Arc::new(Mutex::new(Vec::new())),
            runners: HashMap::new(),
            last_error: None,
        }
    }
}

impl ShareProxyManager {
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn set_keys(&self, keys: Vec<(String, String, String)>) {
        if let Ok(mut guard) = self.keys.lock() {
            *guard = keys;
        }
    }

    pub fn update_port_tps(&self, port: u16, tps: Option<f64>) {
        if let Ok(mut store) = self.activity.lock() {
            store.set_port_tps(port, tps);
        }
    }

    pub fn snapshot(&self) -> (Vec<ConnectedClientPublic>, ShareActivitySummary) {
        match self.activity.lock() {
            Ok(mut store) => store.snapshot(),
            Err(_) => (
                Vec::new(),
                ShareActivitySummary {
                    active_count: 0,
                    idle_count: 0,
                    tokens_per_second: None,
                },
            ),
        }
    }

    pub fn shutdown_all(&mut self) {
        for (_, runner) in self.runners.drain() {
            runner.handle.shutdown();
        }
        if let Ok(mut store) = self.activity.lock() {
            store.clients.clear();
        }
        self.last_error = None;
    }

    /// Ensure listeners match the desired public bind + running share ports.
    pub fn sync(
        &mut self,
        expose: bool,
        bind_host: Option<&str>,
        ports: &[(u16, String)],
        cert: Option<&PathBuf>,
        key: Option<&PathBuf>,
        keys: Vec<(String, String, String)>,
    ) {
        self.set_keys(keys);

        if !expose || bind_host.is_none() || cert.is_none() || key.is_none() || ports.is_empty() {
            self.shutdown_all();
            return;
        }
        let bind_host = bind_host.unwrap().to_string();
        let cert = cert.unwrap().clone();
        let key = key.unwrap().clone();

        if is_loopback_host(&bind_host) {
            self.shutdown_all();
            self.last_error = Some("Share proxy will not bind on loopback".into());
            return;
        }

        let desired: HashMap<u16, &String> = ports.iter().map(|(p, m)| (*p, m)).collect();

        let stale: Vec<u16> = self
            .runners
            .iter()
            .filter(|(port, runner)| {
                !desired.contains_key(port)
                    || runner.bind_host != bind_host
                    || runner.cert != cert
                    || runner.key != key
            })
            .map(|(port, _)| *port)
            .collect();
        for port in stale {
            if let Some(runner) = self.runners.remove(&port) {
                runner.handle.shutdown();
            }
        }

        for (port, model) in ports {
            if let Some(runner) = self.runners.get_mut(port) {
                if let Ok(mut guard) = runner.model.lock() {
                    *guard = model.clone();
                }
                continue;
            }
            match self.spawn_listener(&bind_host, *port, model, &cert, &key) {
                Ok(runner) => {
                    self.runners.insert(*port, runner);
                    self.last_error = None;
                }
                Err(error) => {
                    self.last_error = Some(format!("share proxy :{port}: {error}"));
                }
            }
        }
    }

    fn spawn_listener(
        &self,
        bind_host: &str,
        port: u16,
        model: &str,
        cert: &PathBuf,
        key: &PathBuf,
    ) -> Result<ProxyRunner, String> {
        let addr = resolve_bind_addr(bind_host, port)?;
        let handle = axum_server::Handle::new();
        let serve_handle = handle.clone();
        let activity = Arc::clone(&self.activity);
        let keys = Arc::clone(&self.keys);
        let model_arc = Arc::new(Mutex::new(model.to_string()));
        let model_for_state = Arc::clone(&model_arc);
        let cert_path = cert.clone();
        let key_path = key.clone();
        let cert_for_spawn = cert_path.clone();
        let key_for_spawn = key_path.clone();
        let bind_host = bind_host.to_string();

        tokio::spawn(async move {
            let tls = match RustlsConfig::from_pem_file(&cert_for_spawn, &key_for_spawn).await {
                Ok(config) => config,
                Err(error) => {
                    eprintln!("[share-proxy] TLS config :{port}: {error}");
                    return;
                }
            };
            let client = Client::builder(TokioExecutor::new()).build_http();
            let state = ProxyState {
                local_port: port,
                model: model_for_state,
                keys,
                activity,
                client,
            };
            let app = Router::new().fallback(proxy_request).with_state(state);
            if let Err(error) = axum_server::bind_rustls(addr, tls)
                .handle(serve_handle)
                .serve(app.into_make_service_with_connect_info::<SocketAddr>())
                .await
            {
                eprintln!("[share-proxy] listener {addr}: {error}");
            }
        });

        Ok(ProxyRunner {
            bind_host,
            cert: cert_path,
            key: key_path,
            model: model_arc,
            handle,
        })
    }
}

struct ActivityEnd {
    activity: Arc<Mutex<ShareActivityStore>>,
    id: String,
}

impl Drop for ActivityEnd {
    fn drop(&mut self) {
        if let Ok(mut store) = self.activity.lock() {
            store.end_request(&self.id);
        }
    }
}

async fn proxy_request(
    State(state): State<ProxyState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    let method = request.method().clone();
    let path = request
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    let Some((key_id, key_name)) = authorize(request.headers(), &state.keys) else {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "Unauthorized",
        )
            .into_response();
    };

    let model = state
        .model
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| "model".into());
    let remote = peer.ip().to_string();
    let activity_id = {
        let Ok(mut store) = state.activity.lock() else {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        };
        store.begin_request(
            &remote,
            &key_id,
            &key_name,
            state.local_port,
            &model,
            method.as_str(),
            &path,
        )
    };

    match forward(&state, method, &path, request, activity_id).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            format!("share proxy upstream error: {error}"),
        )
            .into_response(),
    }
}

fn authorize(
    headers: &HeaderMap,
    keys: &Arc<Mutex<Vec<(String, String, String)>>>,
) -> Option<(String, String)> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))?
        .trim();
    if token.is_empty() {
        return None;
    }
    let Ok(guard) = keys.lock() else {
        return None;
    };
    guard
        .iter()
        .find(|(_, _, secret)| secret == token)
        .map(|(id, name, _)| (id.clone(), name.clone()))
}

async fn forward(
    state: &ProxyState,
    method: Method,
    path: &str,
    request: Request,
    activity_id: String,
) -> Result<Response, String> {
    let end = Arc::new(ActivityEnd {
        activity: Arc::clone(&state.activity),
        id: activity_id,
    });

    let (parts, body) = request.into_parts();
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return Err(error.to_string()),
    };

    let upstream = format!("http://127.0.0.1:{}{path}", state.local_port);
    let uri: Uri = upstream.parse().map_err(|error| format!("{error}"))?;

    let mut builder = hyper::Request::builder().method(method).uri(uri);
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop(name) || name == header::HOST || name == header::AUTHORIZATION {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder = builder.header(header::HOST, format!("127.0.0.1:{}", state.local_port));

    let upstream_req = builder
        .body(Full::new(bytes))
        .map_err(|error| error.to_string())?;

    let upstream_res = match state.client.request(upstream_req).await {
        Ok(response) => response,
        Err(error) => return Err(error.to_string()),
    };

    let (up_parts, up_body) = upstream_res.into_parts();
    let mut response = Response::builder().status(up_parts.status);
    for (name, value) in up_parts.headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        response = response.header(name, value);
    }

    let end_hold = Arc::clone(&end);
    let stream = up_body.into_data_stream().map(move |chunk| {
        let _keep_active = &end_hold;
        match chunk {
            Ok(bytes) => Ok::<_, std::io::Error>(bytes),
            Err(error) => Err(std::io::Error::other(error.to_string())),
        }
    });
    // Drop the local Arc so only the response body stream keeps activity Active.
    drop(end);
    let body = Body::from_stream(stream);
    response.body(body).map_err(|error| error.to_string())
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn resolve_bind_addr(host: &str, port: u16) -> Result<SocketAddr, String> {
    let ip: IpAddr = host
        .parse()
        .map_err(|_| format!("invalid share bind host: {host}"))?;
    Ok(SocketAddr::new(ip, port))
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
        || host
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

fn unix_secs(at: Instant) -> u64 {
    let now_wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let elapsed = Instant::now().saturating_duration_since(at).as_secs();
    now_wall.saturating_sub(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_tracks_active_and_prunes_idle() {
        let mut store = ShareActivityStore::default();
        let id = store.begin_request(
            "192.168.1.10",
            "key1",
            "Phone",
            8080,
            "model-a",
            "POST",
            "/v1/chat/completions",
        );
        let (clients, summary) = store.snapshot();
        assert_eq!(clients.len(), 1);
        assert_eq!(summary.active_count, 1);
        assert_eq!(clients[0].key_name, "Phone");
        assert_eq!(clients[0].state, ClientActivityState::Active);

        store.end_request(&id);
        let (clients, summary) = store.snapshot();
        assert_eq!(summary.active_count, 0);
        assert_eq!(summary.idle_count, 1);
        assert_eq!(clients[0].state, ClientActivityState::Idle);
    }
}
