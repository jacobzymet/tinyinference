//! LAN / Tailscale exposure, access tokens, share URLs, and mDNS discovery.

use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use local_ip_address::list_afinet_netifas;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};

pub const SERVICE_TYPE: &str = "_tinyinference._tcp.local.";
const TOKEN_BYTES: usize = 24;
const PEER_TTL: Duration = Duration::from_secs(90);
const HEALTH_CACHE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InferenceMode {
    #[default]
    Local,
    Remote,
}

impl InferenceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "local" => Some(Self::Local),
            "remote" => Some(Self::Remote),
            _ => None,
        }
    }
}

/// Where managed llama-server should listen when LLM sharing is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ListenScope {
    /// All IPv4 interfaces (`0.0.0.0`) — LAN + Tailscale + others.
    #[default]
    All,
    /// First Tailscale CGNAT address only (`100.64/10`).
    Tailscale,
    /// Explicit host/IP in [`NetworkConfig::listen_host`].
    Custom,
}

impl ListenScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Tailscale => "tailscale",
            Self::Custom => "custom",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "all" | "lan" | "any" => Some(Self::All),
            "tailscale" | "ts" => Some(Self::Tailscale),
            "custom" | "host" | "address" => Some(Self::Custom),
            _ => None,
        }
    }

    pub fn novice_label(self) -> &'static str {
        match self {
            Self::All => "Wi‑Fi / LAN and Tailscale",
            Self::Tailscale => "Tailscale only",
            Self::Custom => "A specific address",
        }
    }

    pub fn novice_hint(self) -> &'static str {
        match self {
            Self::All => {
                "Easiest option. Other machines on your home/office network or Tailscale can call this computer’s OpenAI-compatible API (with the API key)."
            }
            Self::Tailscale => {
                "More private. Only devices on your Tailscale network can reach the LLM API — not random devices on café Wi‑Fi. Tailscale must be running."
            }
            Self::Custom => {
                "Advanced: bind llama-server to one IP you choose (for example a single LAN or Tailscale address)."
            }
        }
    }

    pub fn technical_detail(self) -> &'static str {
        match self {
            Self::All => "Listens on 0.0.0.0 (all interfaces).",
            Self::Tailscale => "Listens on your Tailscale IP (100.x) only.",
            Self::Custom => "Listens on the address you select below.",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NetworkConfig {
    pub expose: bool,
    /// Used when [`Self::expose`] is true.
    pub listen_scope: ListenScope,
    /// Host/IP for [`ListenScope::Custom`] (and remembered Tailscale pin if set).
    pub listen_host: String,
    pub access_token: String,
    pub inference_mode: InferenceMode,
    pub remote_base: String,
    pub remote_token: String,
    pub device_name: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            expose: false,
            listen_scope: ListenScope::All,
            listen_host: String::new(),
            access_token: String::new(),
            inference_mode: InferenceMode::Local,
            remote_base: String::new(),
            remote_token: String::new(),
            device_name: String::new(),
        }
    }
}

impl NetworkConfig {
    pub fn ensure_token(&mut self) -> bool {
        if self.access_token.trim().is_empty() {
            self.access_token = generate_access_token();
            true
        } else {
            false
        }
    }

    pub fn regenerate_token(&mut self) {
        self.access_token = generate_access_token();
    }

    pub fn resolved_device_name(&self) -> String {
        let trimmed = self.device_name.trim();
        if !trimmed.is_empty() {
            return sanitize_instance_name(trimmed);
        }
        sanitize_instance_name(&default_hostname())
    }

    /// Control-panel `/api` stays local; auth for shared LLMs is llama `--api-key`.
    pub fn inbound_auth_required(&self) -> bool {
        false
    }

    /// Host string llama-server should bind to (without port).
    pub fn resolve_listen_host(&self) -> Result<String, String> {
        if !self.expose {
            return Ok("127.0.0.1".into());
        }
        match self.listen_scope {
            ListenScope::All => Ok("0.0.0.0".into()),
            ListenScope::Tailscale => {
                let (_, ts) = shareable_ipv4_addrs();
                ts.first()
                    .map(|ip| ip.to_string())
                    .ok_or_else(|| {
                        "Tailscale only is selected, but no Tailscale IP (100.x) was found. Start Tailscale, or choose Wi‑Fi / LAN and Tailscale.".into()
                    })
            }
            ListenScope::Custom => {
                let host = self.listen_host.trim();
                if host.is_empty() {
                    return Err(
                        "Pick a specific address to listen on, or choose another sharing option."
                            .into(),
                    );
                }
                if host.parse::<IpAddr>().is_err()
                    && host != "localhost"
                    && !host.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == ':')
                {
                    return Err(format!("Listen address looks invalid: {host}"));
                }
                Ok(host.to_string())
            }
        }
    }

    pub fn should_advertise_mdns(&self) -> bool {
        // mDNS is a LAN discovery tool; only advertise when we actually listen for LAN.
        self.expose && self.listen_scope == ListenScope::All
    }

    /// OpenAI-compatible chat completions URL on a linked LLM (`…/v1/chat/completions`).
    pub fn remote_chat_url(&self) -> Option<String> {
        let base = self.normalize_remote_base()?;
        if self.inference_mode != InferenceMode::Remote {
            return None;
        }
        Some(format!("{base}/chat/completions"))
    }

    /// Normalize a remote OpenAI base to `http://host:port/v1` (no trailing slash).
    pub fn normalize_remote_base(&self) -> Option<String> {
        let mut base = self.remote_base.trim().trim_end_matches('/').to_string();
        if base.is_empty() {
            return None;
        }
        if !base.contains("://") {
            base = format!("http://{base}");
        }
        // Accept pasted roots without `/v1`.
        if !base.ends_with("/v1") {
            base = format!("{base}/v1");
        }
        Some(base)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ListenCandidate {
    pub kind: &'static str,
    pub address: String,
    pub label: String,
}

pub fn listen_candidates() -> Vec<ListenCandidate> {
    let (lan, ts) = shareable_ipv4_addrs();
    let mut out = Vec::new();
    for ip in ts {
        out.push(ListenCandidate {
            kind: "tailscale",
            address: ip.to_string(),
            label: format!("Tailscale · {ip}"),
        });
    }
    for ip in lan {
        out.push(ListenCandidate {
            kind: "lan",
            address: ip.to_string(),
            label: format!("LAN · {ip}"),
        });
    }
    out
}

pub fn generate_access_token() -> String {
    let mut bytes = [0u8; TOKEN_BYTES];
    if getrandom::fill(&mut bytes).is_err() {
        // Extremely unlikely; fall back to a time-based mix so expose still works.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id() as u128;
        let mixed = nanos ^ (pid << 64) ^ 0x9e37_79b9_7f4a_7c15;
        return format!("{mixed:048x}");
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn mask_token(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.len() <= 8 {
        return "••••••••".into();
    }
    format!("{}…{}", &trimmed[..4], &trimmed[trimmed.len() - 4..])
}

pub fn default_hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .or_else(|_| std::env::var("NAME"))
        .unwrap_or_else(|_| "tinyinference".into())
}

fn sanitize_instance_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "tinyinference".into()
    } else {
        trimmed.chars().take(63).collect()
    }
}

/// Prefer Tailscale CGNAT (`100.64.0.0/10`), then other non-loopback IPv4.
pub fn shareable_ipv4_addrs() -> (Vec<Ipv4Addr>, Vec<Ipv4Addr>) {
    let mut lan = Vec::new();
    let mut tailscale = Vec::new();
    let Ok(ifs) = list_afinet_netifas() else {
        return (lan, tailscale);
    };
    for (_name, addr) in ifs {
        let IpAddr::V4(ip) = addr else { continue };
        if ip.is_loopback() || ip.is_unspecified() || ip.is_link_local() {
            continue;
        }
        if is_tailscale_cg_nat(ip) {
            if !tailscale.contains(&ip) {
                tailscale.push(ip);
            }
        } else if !lan.contains(&ip) {
            lan.push(ip);
        }
    }
    (lan, tailscale)
}

pub fn is_tailscale_cg_nat(ip: Ipv4Addr) -> bool {
    // 100.64.0.0/10
    let o = ip.octets();
    o[0] == 100 && (o[1] & 0xC0) == 64
}

#[derive(Debug, Clone, Serialize)]
pub struct ShareUrl {
    pub kind: &'static str,
    pub label: String,
    /// OpenAI-compatible API base, e.g. `http://192.168.1.10:8080/v1`.
    pub api_base: String,
}

pub fn lan_share_urls(port: u16, token: &str) -> Vec<ShareUrl> {
    let (lan, _) = shareable_ipv4_addrs();
    lan.into_iter()
        .map(|ip| share_url_for("lan", format!("LAN ({ip})"), ip, port, token))
        .collect()
}

pub fn tailscale_urls(port: u16, token: &str) -> Vec<ShareUrl> {
    let (_, ts) = shareable_ipv4_addrs();
    ts.into_iter()
        .map(|ip| share_url_for("tailscale", format!("Tailscale ({ip})"), ip, port, token))
        .collect()
}

pub fn share_url_for(
    kind: &'static str,
    label: String,
    ip: Ipv4Addr,
    port: u16,
    token: &str,
) -> ShareUrl {
    share_url_host(kind, label, &ip.to_string(), port, token)
}

pub fn share_url_host(
    kind: &'static str,
    label: String,
    host: &str,
    port: u16,
    _token: &str,
) -> ShareUrl {
    let authority = format!("{host}:{port}");
    ShareUrl {
        kind,
        label,
        api_base: format!("http://{authority}/v1"),
    }
}

pub fn extract_request_token(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(value) = headers.get(axum::http::header::AUTHORIZATION)
        && let Ok(raw) = value.to_str()
    {
        let trimmed = raw.trim();
        if let Some(rest) = trimmed.strip_prefix("Bearer ").or_else(|| trimmed.strip_prefix("bearer "))
        {
            let token = rest.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    if let Some(value) = headers.get("x-tinyinference-token")
        && let Ok(raw) = value.to_str()
    {
        let token = raw.trim();
        if !token.is_empty() {
            return Some(token.to_string());
        }
    }
    None
}

pub fn token_matches(expected: &str, provided: &str) -> bool {
    let expected = expected.as_bytes();
    let provided = provided.as_bytes();
    if expected.len() != provided.len() {
        return false;
    }
    // Constant-time-ish compare for short secrets.
    let mut diff = 0u8;
    for (a, b) in expected.iter().zip(provided.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

pub fn is_loopback_addr(addr: SocketAddr) -> bool {
    addr.ip().is_loopback()
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredPeer {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub base_url: String,
    pub fullname: String,
}

#[derive(Debug, Default)]
struct PeerStore {
    peers: HashMap<String, (DiscoveredPeer, Instant)>,
}

impl PeerStore {
    fn upsert(&mut self, peer: DiscoveredPeer) {
        let key = peer.fullname.clone();
        self.peers.insert(key, (peer, Instant::now()));
    }

    fn remove(&mut self, fullname: &str) {
        self.peers.remove(fullname);
    }

    fn list(&mut self) -> Vec<DiscoveredPeer> {
        let now = Instant::now();
        self.peers
            .retain(|_, (_, seen)| now.duration_since(*seen) < PEER_TTL);
        let mut peers: Vec<_> = self.peers.values().map(|(p, _)| p.clone()).collect();
        peers.sort_by(|a, b| a.name.cmp(&b.name));
        peers
    }
}

/// Shared mDNS advertise + browse lifecycle for the process.
pub struct NetworkDiscovery {
    peers: Arc<Mutex<PeerStore>>,
    daemon: Mutex<Option<ServiceDaemon>>,
    advertised_fullname: Mutex<Option<String>>,
}

impl NetworkDiscovery {
    pub fn new() -> Self {
        let peers = Arc::new(Mutex::new(PeerStore::default()));
        let discovery = Self {
            peers: Arc::clone(&peers),
            daemon: Mutex::new(None),
            advertised_fullname: Mutex::new(None),
        };
        discovery.ensure_daemon_and_browse();
        discovery
    }

    fn ensure_daemon_and_browse(&self) {
        let mut guard = self.daemon.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return;
        }
        let Ok(daemon) = ServiceDaemon::new() else {
            return;
        };
        if let Ok(receiver) = daemon.browse(SERVICE_TYPE) {
            let peers = Arc::clone(&self.peers);
            thread::spawn(move || {
                while let Ok(event) = receiver.recv() {
                    match event {
                        ServiceEvent::ServiceResolved(info) => {
                            if let Some(peer) = peer_from_resolved(&info) {
                                if let Ok(mut store) = peers.lock() {
                                    store.upsert(peer);
                                }
                            }
                        }
                        ServiceEvent::ServiceRemoved(_, fullname) => {
                            if let Ok(mut store) = peers.lock() {
                                store.remove(&fullname);
                            }
                        }
                        _ => {}
                    }
                }
            });
        }
        *guard = Some(daemon);
    }

    pub fn discovered_peers(&self) -> Vec<DiscoveredPeer> {
        self.ensure_daemon_and_browse();
        self.peers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list()
    }

    pub fn sync_advertise(&self, expose: bool, device_name: &str, port: u16) {
        self.ensure_daemon_and_browse();
        let daemon_guard = self.daemon.lock().unwrap_or_else(|e| e.into_inner());
        let Some(daemon) = daemon_guard.as_ref() else {
            return;
        };
        let mut name_guard = self
            .advertised_fullname
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if !expose {
            if let Some(fullname) = name_guard.take() {
                let _ = daemon.unregister(&fullname);
            }
            return;
        }

        let (lan, ts) = shareable_ipv4_addrs();
        let ip = ts.first().copied().or_else(|| lan.first().copied());
        let Some(ip) = ip else {
            return;
        };

        let instance = sanitize_instance_name(device_name);
        let host_name = format!("{ip}.local.");
        let properties = [("path", "/v1"), ("ver", "1")];
        let Ok(service) = ServiceInfo::new(
            SERVICE_TYPE,
            &instance,
            &host_name,
            &ip.to_string(),
            port,
            &properties[..],
        ) else {
            return;
        };
        let fullname = service.get_fullname().to_string();
        if name_guard.as_deref() == Some(fullname.as_str()) {
            return;
        }
        if let Some(previous) = name_guard.take() {
            let _ = daemon.unregister(&previous);
        }
        if daemon.register(service).is_ok() {
            *name_guard = Some(fullname);
        }
    }
}

fn peer_from_resolved(info: &mdns_sd::ResolvedService) -> Option<DiscoveredPeer> {
    if !info.is_valid() {
        return None;
    }
    let port = info.port;
    if port == 0 {
        return None;
    }
    let host = info
        .get_addresses_v4()
        .into_iter()
        .next()
        .map(|ip| ip.to_string())
        .or_else(|| {
            let host = info.host.trim_end_matches('.');
            if host.is_empty() {
                None
            } else {
                Some(host.to_string())
            }
        })?;
    let short_name = info
        .fullname
        .strip_suffix(&format!(".{SERVICE_TYPE}"))
        .unwrap_or(info.fullname.as_str())
        .trim_end_matches('.')
        .to_string();
    Some(DiscoveredPeer {
        name: short_name,
        host: host.clone(),
        port,
        base_url: format!("http://{host}:{port}/v1"),
        fullname: info.fullname.clone(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct RemoteHealth {
    pub ok: bool,
    pub model: Option<String>,
    pub status: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Default)]
pub struct HealthCache {
    last: Mutex<Option<(String, Instant, RemoteHealth)>>,
}

impl HealthCache {
    pub fn peek(&self, base: &str, token: &str) -> Option<RemoteHealth> {
        let key = format!("{base}|{token}");
        let Ok(guard) = self.last.lock() else {
            return None;
        };
        match guard.as_ref() {
            Some((cached_key, at, health))
                if cached_key == &key && at.elapsed() < HEALTH_CACHE =>
            {
                Some(health.clone())
            }
            _ => None,
        }
    }

    pub fn probe(&self, base: &str, token: &str) -> RemoteHealth {
        let key = format!("{base}|{token}");
        if let Some(health) = self.peek(base, token) {
            return health;
        }
        let health = probe_remote_state(base, token);
        if let Ok(mut guard) = self.last.lock() {
            *guard = Some((key, Instant::now(), health.clone()));
        }
        health
    }
}

fn probe_remote_state(base: &str, token: &str) -> RemoteHealth {
    let mut base = base.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return RemoteHealth {
            ok: false,
            model: None,
            status: None,
            error: Some("No remote URL configured".into()),
        };
    }
    if !base.contains("://") {
        base = format!("http://{base}");
    }
    if !base.ends_with("/v1") {
        base = format!("{base}/v1");
    }
    let url = format!("{base}/models");
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(2)))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut request = agent.get(&url);
    if !token.trim().is_empty() {
        request = request.header("Authorization", &format!("Bearer {}", token.trim()));
    }
    match request.call() {
        Ok(mut response) => {
            if response.status() != 200 {
                return RemoteHealth {
                    ok: false,
                    model: None,
                    status: None,
                    error: Some(format!("Remote responded with {}", response.status())),
                };
            }
            match response.body_mut().read_json::<serde_json::Value>() {
                Ok(body) => {
                    let model = body
                        .get("data")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|m| m.get("id"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    RemoteHealth {
                        ok: true,
                        model,
                        status: Some("ready".into()),
                        error: None,
                    }
                }
                Err(error) => RemoteHealth {
                    ok: false,
                    model: None,
                    status: None,
                    error: Some(error.to_string()),
                },
            }
        }
        Err(error) => RemoteHealth {
            ok: false,
            model: None,
            status: None,
            error: Some(error.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tailscale_range_detection() {
        assert!(is_tailscale_cg_nat(Ipv4Addr::new(100, 64, 0, 1)));
        assert!(is_tailscale_cg_nat(Ipv4Addr::new(100, 100, 1, 2)));
        assert!(is_tailscale_cg_nat(Ipv4Addr::new(100, 127, 255, 255)));
        assert!(!is_tailscale_cg_nat(Ipv4Addr::new(100, 63, 0, 1)));
        assert!(!is_tailscale_cg_nat(Ipv4Addr::new(100, 128, 0, 1)));
        assert!(!is_tailscale_cg_nat(Ipv4Addr::new(192, 168, 1, 1)));
    }

    #[test]
    fn token_roundtrip_mask() {
        let token = generate_access_token();
        assert!(token.len() >= 32);
        let masked = mask_token(&token);
        assert!(masked.contains('…'));
        assert!(!masked.contains(&token));
    }

    #[test]
    fn listen_scope_resolves_all_and_custom() {
        let mut cfg = NetworkConfig::default();
        cfg.expose = true;
        cfg.listen_scope = ListenScope::All;
        assert_eq!(cfg.resolve_listen_host().unwrap(), "0.0.0.0");

        cfg.listen_scope = ListenScope::Custom;
        cfg.listen_host = "10.0.0.8".into();
        assert_eq!(cfg.resolve_listen_host().unwrap(), "10.0.0.8");

        cfg.expose = false;
        assert_eq!(cfg.resolve_listen_host().unwrap(), "127.0.0.1");
    }
}
