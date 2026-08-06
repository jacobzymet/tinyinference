//! LAN / Tailscale exposure, API keys, share URLs, and mDNS discovery.

use std::{
    collections::HashMap,
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, UdpSocket},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use local_ip_address::list_afinet_netifas;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};

pub const SERVICE_TYPE: &str = "_tinyinference._tcp.local.";
/// UDP broadcast discovery port (mDNS fallback — more reliable on Windows Wi‑Fi).
pub const BEACON_PORT: u16 = 39217;
const BEACON_MAGIC: &[u8] = b"TI1\n";
const BEACON_INTERVAL: Duration = Duration::from_secs(2);
const SUBNET_SCAN_INTERVAL: Duration = Duration::from_secs(15);
const SCAN_PORTS: &[u16] = &[8080, 8081, 8090, 3000];
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
    All,
    /// First Tailscale CGNAT address only (`100.64/10`). Safer default.
    #[default]
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
                "Broadest reach: every interface on this PC. Prefer Tailscale only unless you need LAN."
            }
            Self::Tailscale => {
                "Narrowest common option: only Tailscale peers. Tailscale must be running."
            }
            Self::Custom => "Bind to exactly one IP you choose below — nothing else.",
        }
    }

    pub fn technical_detail(self) -> &'static str {
        match self {
            Self::All => {
                "Binds llama-server to 0.0.0.0 (all interfaces) with a self-signed TLS cert."
            }
            Self::Tailscale => {
                "Binds llama-server to your Tailscale IP (100.x) only with a self-signed TLS cert."
            }
            Self::Custom => {
                "Binds llama-server to the address you select with a self-signed TLS cert."
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    pub secret: String,
    /// Unix seconds when the key was created or last regenerated.
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyPublic {
    pub id: String,
    pub name: String,
    pub secret_masked: String,
    pub created_at: u64,
}

impl ApiKey {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: generate_key_id(),
            name: sanitize_key_name(name),
            secret: generate_access_token(),
            created_at: unix_now(),
        }
    }

    pub fn to_public(&self) -> ApiKeyPublic {
        ApiKeyPublic {
            id: self.id.clone(),
            name: self.name.clone(),
            secret_masked: mask_token(&self.secret),
            created_at: self.created_at,
        }
    }
}

/// A saved OpenAI-compatible LLM hosted on another device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinkedRemote {
    pub id: String,
    pub name: String,
    pub base: String,
    #[serde(default)]
    pub token: String,
}

impl LinkedRemote {
    pub fn new(name: impl Into<String>, base: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            id: generate_remote_id(),
            name: sanitize_key_name(name),
            base: base.into(),
            token: token.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkedRemotePublic {
    pub id: String,
    pub name: String,
    pub base: String,
    pub token_set: bool,
    pub token_masked: String,
    pub active: bool,
    pub health: Option<RemoteHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NetworkConfig {
    pub expose: bool,
    /// Used when [`Self::expose`] is true.
    pub listen_scope: ListenScope,
    /// Host/IP for [`ListenScope::Custom`] (and remembered Tailscale pin if set).
    pub listen_host: String,
    /// Legacy single token — migrated into [`Self::api_keys`] on load.
    #[serde(default)]
    pub access_token: String,
    /// Named API keys accepted by shared llama-server (`--api-key`, repeatable).
    #[serde(default)]
    pub api_keys: Vec<ApiKey>,
    pub inference_mode: InferenceMode,
    /// Named linked LLMs (multi-device manager). Preferred over legacy fields.
    #[serde(default)]
    pub remotes: Vec<LinkedRemote>,
    /// Which linked LLM chat uses when [`Self::inference_mode`] is Remote.
    #[serde(default)]
    pub active_remote_id: String,
    /// Legacy single remote — migrated into [`Self::remotes`] on load.
    #[serde(default)]
    pub remote_base: String,
    #[serde(default)]
    pub remote_token: String,
    pub device_name: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            expose: false,
            listen_scope: ListenScope::Tailscale,
            listen_host: String::new(),
            access_token: String::new(),
            api_keys: Vec::new(),
            inference_mode: InferenceMode::Local,
            remotes: Vec::new(),
            active_remote_id: String::new(),
            remote_base: String::new(),
            remote_token: String::new(),
            device_name: String::new(),
        }
    }
}

impl NetworkConfig {
    /// Move legacy `access_token` into `api_keys` when needed.
    pub fn migrate_api_keys(&mut self) {
        let legacy = self.access_token.trim();
        if !legacy.is_empty()
            && !self
                .api_keys
                .iter()
                .any(|key| key.secret.trim() == legacy)
        {
            self.api_keys.insert(
                0,
                ApiKey {
                    id: generate_key_id(),
                    name: "Default".into(),
                    secret: legacy.to_string(),
                    created_at: unix_now(),
                },
            );
        }
        // Keep legacy field in sync with the first key for older readers.
        self.access_token = self
            .api_keys
            .first()
            .map(|key| key.secret.clone())
            .unwrap_or_default();
    }

    /// Ensure at least one key exists when sharing; returns a newly created secret if any.
    pub fn ensure_token(&mut self) -> bool {
        self.migrate_api_keys();
        if self.api_keys.is_empty() {
            let key = ApiKey::new("Default");
            self.access_token = key.secret.clone();
            self.api_keys.push(key);
            true
        } else {
            false
        }
    }

    pub fn regenerate_token(&mut self) {
        self.migrate_api_keys();
        if let Some(key) = self.api_keys.first_mut() {
            key.secret = generate_access_token();
            key.created_at = unix_now();
            self.access_token = key.secret.clone();
        } else {
            let _ = self.ensure_token();
        }
    }

    pub fn api_key_secrets(&self) -> Vec<String> {
        self.api_keys
            .iter()
            .map(|key| key.secret.trim().to_string())
            .filter(|secret| !secret.is_empty())
            .collect()
    }

    pub fn primary_api_key(&self) -> Option<&str> {
        self.api_keys
            .iter()
            .map(|key| key.secret.trim())
            .find(|secret| !secret.is_empty())
    }

    pub fn public_api_keys(&self) -> Vec<ApiKeyPublic> {
        self.api_keys.iter().map(ApiKey::to_public).collect()
    }

    pub fn create_api_key(&mut self, name: &str) -> Result<ApiKey, String> {
        self.migrate_api_keys();
        let cleaned = sanitize_key_name(name);
        if cleaned.is_empty() {
            return Err("Give the API key a name.".into());
        }
        if self.api_keys.len() >= 32 {
            return Err("Maximum of 32 API keys reached.".into());
        }
        let key = ApiKey::new(cleaned);
        self.api_keys.push(key.clone());
        self.access_token = self
            .api_keys
            .first()
            .map(|k| k.secret.clone())
            .unwrap_or_default();
        Ok(key)
    }

    pub fn rename_api_key(&mut self, id: &str, name: &str) -> Result<(), String> {
        let cleaned = sanitize_key_name(name);
        if cleaned.is_empty() {
            return Err("Give the API key a name.".into());
        }
        let key = self
            .api_keys
            .iter_mut()
            .find(|key| key.id == id)
            .ok_or_else(|| "API key not found.".to_string())?;
        key.name = cleaned;
        Ok(())
    }

    pub fn regenerate_api_key(&mut self, id: &str) -> Result<ApiKey, String> {
        let key = self
            .api_keys
            .iter_mut()
            .find(|key| key.id == id)
            .ok_or_else(|| "API key not found.".to_string())?;
        key.secret = generate_access_token();
        key.created_at = unix_now();
        let updated = key.clone();
        if self.api_keys.first().map(|k| k.id.as_str()) == Some(id) {
            self.access_token = updated.secret.clone();
        }
        Ok(updated)
    }

    pub fn delete_api_key(&mut self, id: &str) -> Result<(), String> {
        if !self.api_keys.iter().any(|key| key.id == id) {
            return Err("API key not found.".into());
        }
        if self.expose && self.api_keys.len() <= 1 {
            return Err("Keep at least one API key while sharing is enabled.".into());
        }
        self.api_keys.retain(|key| key.id != id);
        self.access_token = self
            .api_keys
            .first()
            .map(|key| key.secret.clone())
            .unwrap_or_default();
        Ok(())
    }

    pub fn resolved_device_name(&self) -> String {
        let trimmed = self.device_name.trim();
        if !trimmed.is_empty() {
            return sanitize_instance_name(trimmed);
        }
        sanitize_instance_name(&default_hostname())
    }

    /// Fold legacy `remote_base` / `remote_token` into `remotes`, and keep the
    /// active id + mirrored legacy fields consistent.
    pub fn migrate_remotes(&mut self) {
        if self.remotes.is_empty() {
            let base = self.remote_base.trim();
            if !base.is_empty() {
                let mut remote = LinkedRemote::new(
                    label_from_remote_base(base),
                    base,
                    self.remote_token.clone(),
                );
                if let Some(normalized) = normalize_openai_base(&remote.base) {
                    remote.base = normalized;
                }
                self.active_remote_id = remote.id.clone();
                self.remotes.push(remote);
            }
        }

        self.remotes.retain(|remote| !remote.base.trim().is_empty());
        for remote in &mut self.remotes {
            if remote.id.trim().is_empty() {
                remote.id = generate_remote_id();
            }
            if remote.name.trim().is_empty() {
                remote.name = label_from_remote_base(&remote.base);
            } else {
                remote.name = sanitize_key_name(&remote.name);
            }
            if let Some(normalized) = normalize_openai_base(&remote.base) {
                remote.base = normalized;
            }
        }

        if self.active_remote_id.trim().is_empty()
            || !self
                .remotes
                .iter()
                .any(|remote| remote.id == self.active_remote_id)
        {
            self.active_remote_id = self
                .remotes
                .first()
                .map(|remote| remote.id.clone())
                .unwrap_or_default();
        }

        self.sync_legacy_remote_fields();
    }

    fn sync_legacy_remote_fields(&mut self) {
        if let Some(active) = self.active_remote().cloned() {
            self.remote_base = active.base;
            self.remote_token = active.token;
        } else {
            self.remote_base.clear();
            self.remote_token.clear();
        }
    }

    pub fn active_remote(&self) -> Option<&LinkedRemote> {
        if self.remotes.is_empty() {
            return None;
        }
        self.remotes
            .iter()
            .find(|remote| remote.id == self.active_remote_id)
            .or_else(|| self.remotes.first())
    }

    pub fn active_remote_mut(&mut self) -> Option<&mut LinkedRemote> {
        if self.remotes.is_empty() {
            return None;
        }
        let id = if self
            .remotes
            .iter()
            .any(|remote| remote.id == self.active_remote_id)
        {
            self.active_remote_id.clone()
        } else {
            self.remotes[0].id.clone()
        };
        self.active_remote_id = id.clone();
        self.remotes.iter_mut().find(|remote| remote.id == id)
    }

    pub fn set_active_remote(&mut self, id: &str) -> Result<(), String> {
        self.migrate_remotes();
        if !self.remotes.iter().any(|remote| remote.id == id) {
            return Err("Linked LLM not found.".into());
        }
        self.active_remote_id = id.to_string();
        self.inference_mode = InferenceMode::Remote;
        self.sync_legacy_remote_fields();
        Ok(())
    }

    pub fn upsert_remote(
        &mut self,
        id: Option<&str>,
        name: &str,
        base: &str,
        token: Option<&str>,
        activate: bool,
    ) -> Result<LinkedRemote, String> {
        self.migrate_remotes();
        let Some(normalized) = normalize_openai_base(base) else {
            return Err("Enter an API base URL (usually ending in /v1).".into());
        };
        let cleaned_name = {
            let n = sanitize_key_name(name);
            if n.is_empty() {
                label_from_remote_base(&normalized)
            } else {
                n
            }
        };

        // Create path: reuse an existing link with the same base instead of duplicating.
        let target_id = id.map(str::to_string).or_else(|| {
            self.remotes
                .iter()
                .find(|remote| normalize_openai_base(&remote.base).as_deref() == Some(normalized.as_str()))
                .map(|remote| remote.id.clone())
        });

        if let Some(id) = target_id {
            let remote = self
                .remotes
                .iter_mut()
                .find(|remote| remote.id == id)
                .ok_or_else(|| "Linked LLM not found.".to_string())?;
            remote.name = cleaned_name;
            remote.base = normalized;
            if let Some(token) = token {
                // Empty string clears; omit (None) keeps existing.
                remote.token = token.to_string();
            }
            let updated = remote.clone();
            if activate {
                self.active_remote_id = updated.id.clone();
                self.inference_mode = InferenceMode::Remote;
            }
            self.sync_legacy_remote_fields();
            return Ok(updated);
        }

        if self.remotes.len() >= 32 {
            return Err("Maximum of 32 linked LLMs reached.".into());
        }
        let remote = LinkedRemote::new(cleaned_name, normalized, token.unwrap_or(""));
        let created = remote.clone();
        self.remotes.push(remote);
        if activate || self.remotes.len() == 1 {
            self.active_remote_id = created.id.clone();
            if activate {
                self.inference_mode = InferenceMode::Remote;
            }
        }
        self.sync_legacy_remote_fields();
        Ok(created)
    }

    pub fn delete_remote(&mut self, id: &str) -> Result<(), String> {
        self.migrate_remotes();
        if !self.remotes.iter().any(|remote| remote.id == id) {
            return Err("Linked LLM not found.".into());
        }
        self.remotes.retain(|remote| remote.id != id);
        if self.active_remote_id == id {
            self.active_remote_id = self
                .remotes
                .first()
                .map(|remote| remote.id.clone())
                .unwrap_or_default();
            if self.remotes.is_empty() {
                self.inference_mode = InferenceMode::Local;
            }
        }
        self.sync_legacy_remote_fields();
        Ok(())
    }

    /// Host string for the public share listen scope (without port).
    ///
    /// Prefer [`Self::proxy_bind_hosts`] for the TLS share proxy — binding
    /// `0.0.0.0` conflicts with loopback llama on the same port on Windows.
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

    /// Concrete addresses the share proxy should bind (never `0.0.0.0`).
    ///
    /// llama stays on `127.0.0.1:<port>`; the proxy must use real NIC IPs so
    /// both can share the port number without an exclusive-bind clash.
    pub fn proxy_bind_hosts(&self) -> Result<Vec<String>, String> {
        let host = self.resolve_listen_host()?;
        if host == "0.0.0.0" || host == "::" {
            let (lan, ts) = shareable_ipv4_addrs();
            let hosts: Vec<String> = lan
                .into_iter()
                .chain(ts)
                .map(|ip| ip.to_string())
                .collect();
            if hosts.is_empty() {
                return Err(
                    "No LAN or Tailscale address found to bind the share proxy.".into(),
                );
            }
            return Ok(hosts);
        }
        if host == "127.0.0.1" || host == "localhost" || host == "::1" {
            return Err("Share proxy will not bind on loopback".into());
        }
        Ok(vec![host])
    }

    pub fn should_advertise_mdns(&self) -> bool {
        // mDNS is a LAN discovery tool; only advertise when we actually listen for LAN.
        self.expose && self.listen_scope == ListenScope::All
    }

    /// OpenAI-compatible chat completions URL on a linked LLM (`…/v1/chat/completions`).
    pub fn remote_chat_url(&self) -> Option<String> {
        if self.inference_mode != InferenceMode::Remote {
            return None;
        }
        let base = self.active_remote()?.base.trim();
        let base = normalize_openai_base(base)?;
        Some(format!("{base}/chat/completions"))
    }

    /// Normalize the active remote OpenAI base to `http://host:port/v1` (no trailing slash).
    pub fn normalize_remote_base(&self) -> Option<String> {
        normalize_openai_base(
            self.active_remote()
                .map(|remote| remote.base.as_str())
                .unwrap_or(self.remote_base.as_str()),
        )
    }
}

/// Normalize a remote OpenAI base to `http://host:port/v1` (no trailing slash).
pub fn normalize_openai_base(raw: &str) -> Option<String> {
    let mut base = raw.trim().trim_end_matches('/').to_string();
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

fn label_from_remote_base(base: &str) -> String {
    let trimmed = base
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let host = trimmed.split('/').next().unwrap_or(trimmed);
    let short = host.split(':').next().unwrap_or(host);
    let label = sanitize_key_name(short);
    if label.is_empty() {
        "Linked LLM".into()
    } else {
        label
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

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn generate_key_id() -> String {
    let mut bytes = [0u8; 8];
    if getrandom::fill(&mut bytes).is_err() {
        return format!("key-{}", unix_now());
    }
    format!(
        "key-{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    )
}

fn generate_remote_id() -> String {
    let mut bytes = [0u8; 8];
    if getrandom::fill(&mut bytes).is_err() {
        return format!("remote-{}", unix_now());
    }
    format!(
        "remote-{}",
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    )
}

fn sanitize_key_name(raw: impl Into<String>) -> String {
    let name = raw.into();
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    cleaned.chars().take(64).collect()
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
///
/// LAN addresses from virtual NICs (WSL, Hyper-V, Docker, …) are sorted last so
/// mDNS / share URL helpers prefer Wi‑Fi and Ethernet.
pub fn shareable_ipv4_addrs() -> (Vec<Ipv4Addr>, Vec<Ipv4Addr>) {
    let mut lan: Vec<(Ipv4Addr, bool, u8)> = Vec::new();
    let mut tailscale = Vec::new();
    let Ok(ifs) = list_afinet_netifas() else {
        return (Vec::new(), tailscale);
    };
    for (name, addr) in ifs {
        let IpAddr::V4(ip) = addr else { continue };
        if ip.is_loopback() || ip.is_unspecified() || ip.is_link_local() {
            continue;
        }
        if is_tailscale_cg_nat(ip) {
            if !tailscale.contains(&ip) {
                tailscale.push(ip);
            }
            continue;
        }
        if lan.iter().any(|(existing, _, _)| *existing == ip) {
            continue;
        }
        let virtual_iface = is_virtual_iface(&name);
        lan.push((ip, virtual_iface, lan_range_rank(ip)));
    }
    lan.sort_by_key(|(_, virtual_iface, rank)| (*virtual_iface, *rank));
    (lan.into_iter().map(|(ip, _, _)| ip).collect(), tailscale)
}

fn is_virtual_iface(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("vethernet")
        || n.contains("docker")
        || n.contains("wsl")
        || n.contains("hyper-v")
        || n.contains("vmware")
        || n.contains("virtualbox")
        || n.contains("vbox")
        || n.contains("virbr")
        || n.starts_with("br-")
        || n.contains("veth")
        || n.contains("utun")
        || n.contains("tun")
        || n.contains("tap")
        || n.contains("bridge")
}

fn lan_range_rank(ip: Ipv4Addr) -> u8 {
    let o = ip.octets();
    if o[0] == 192 && o[1] == 168 {
        0
    } else if o[0] == 10 {
        1
    } else if o[0] == 172 && (16..=31).contains(&o[1]) {
        // Often Docker / WSL / Hyper-V — keep, but prefer after RFC1918 home ranges.
        3
    } else {
        2
    }
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
    /// OpenAI-compatible API base, e.g. `https://192.168.1.10:8080/v1`.
    pub api_base: String,
}

pub fn lan_share_urls(port: u16, scheme: &str) -> Vec<ShareUrl> {
    let (lan, _) = shareable_ipv4_addrs();
    lan.into_iter()
        .map(|ip| share_url_for("lan", format!("LAN ({ip})"), ip, port, scheme))
        .collect()
}

pub fn tailscale_urls(port: u16, scheme: &str) -> Vec<ShareUrl> {
    let (_, ts) = shareable_ipv4_addrs();
    ts.into_iter()
        .map(|ip| share_url_for("tailscale", format!("Tailscale ({ip})"), ip, port, scheme))
        .collect()
}

pub fn share_url_for(
    kind: &'static str,
    label: String,
    ip: Ipv4Addr,
    port: u16,
    scheme: &str,
) -> ShareUrl {
    share_url_host(kind, label, &ip.to_string(), port, scheme)
}

pub fn share_url_host(
    kind: &'static str,
    label: String,
    host: &str,
    port: u16,
    scheme: &str,
) -> ShareUrl {
    let authority = format!("{host}:{port}");
    let scheme = if scheme == "https" { "https" } else { "http" };
    ShareUrl {
        kind,
        label,
        api_base: format!("{scheme}://{authority}/v1"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredPeer {
    pub name: String,
    pub host: String,
    pub port: u16,
    /// Additional shared llama ports on this peer (multi-LLM hosts).
    #[serde(default)]
    pub ports: Vec<u16>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdvertisedService {
    fullname: String,
    port: u16,
    ips: Vec<Ipv4Addr>,
}

#[derive(Debug, Clone)]
struct BeaconOut {
    name: String,
    port: u16,
    ports: Vec<u16>,
    ips: Vec<Ipv4Addr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeaconPayload {
    v: u8,
    /// `discover` asks peers to announce; `announce` (or omitted) advertises a share.
    #[serde(default)]
    t: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    port: u16,
    /// All shared llama ports on this device (multi-LLM hosts).
    #[serde(default)]
    ports: Vec<u16>,
    #[serde(default = "default_https_scheme")]
    scheme: String,
    #[serde(default)]
    ips: Vec<String>,
}

fn default_https_scheme() -> String {
    "https".into()
}

/// Shared mDNS + UDP-beacon advertise/browse lifecycle for the process.
pub struct NetworkDiscovery {
    peers: Arc<Mutex<PeerStore>>,
    daemon: Mutex<Option<ServiceDaemon>>,
    browse_started: Mutex<bool>,
    advertised: Mutex<Option<AdvertisedService>>,
    beacon_out: Arc<Mutex<Option<BeaconOut>>>,
    last_error: Arc<Mutex<Option<String>>>,
}

impl NetworkDiscovery {
    pub fn new() -> Self {
        ensure_windows_discovery_firewall();
        let peers = Arc::new(Mutex::new(PeerStore::default()));
        let beacon_out = Arc::new(Mutex::new(None));
        let last_error = Arc::new(Mutex::new(None));
        spawn_beacon_listener(
            Arc::clone(&peers),
            Arc::clone(&beacon_out),
            Arc::clone(&last_error),
        );
        spawn_beacon_broadcaster(Arc::clone(&beacon_out));
        spawn_subnet_scanner(Arc::clone(&peers));
        let discovery = Self {
            peers,
            daemon: Mutex::new(None),
            browse_started: Mutex::new(false),
            advertised: Mutex::new(None),
            beacon_out,
            last_error,
        };
        discovery.ensure_daemon_and_browse();
        discovery
    }

    pub fn discovery_hint(&self) -> String {
        let (lan, _) = shareable_ipv4_addrs();
        if lan.is_empty() {
            return "No Wi‑Fi/Ethernet IPv4 address found on this PC.".into();
        }
        let ips = lan
            .iter()
            .take(3)
            .map(|ip| ip.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        #[cfg(windows)]
        {
            format!(
                "Scanning via mDNS + UDP {BEACON_PORT} + subnet probe. This PC LAN: {ips}. If nothing appears: set Wi‑Fi to Private, allow tinyinference in Windows Firewall, disable router AP/client isolation, and rebuild tinyinference on the other PC too. Tailscale-only peers won’t show — paste their Endpoints URL into Linked LLM."
            )
        }
        #[cfg(not(windows))]
        {
            format!(
                "Scanning via mDNS + UDP {BEACON_PORT} + subnet probe. This PC LAN: {ips}. Peers must share the same Wi‑Fi/LAN (not Tailscale-only), with Share on and a current tinyinference build. Or paste their Endpoints URL into Linked LLM."
            )
        }
    }

    pub fn last_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn advertised_fullname(&self) -> Option<String> {
        self.advertised
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|adv| adv.fullname.clone())
    }

    pub fn is_advertising(&self) -> bool {
        self.advertised
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
            || self
                .beacon_out
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
    }

    fn set_error(&self, message: impl Into<String>) {
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(message.into());
    }

    fn clear_error(&self) {
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    fn ensure_daemon_and_browse(&self) {
        {
            let mut guard = self.daemon.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                match ServiceDaemon::new() {
                    Ok(daemon) => *guard = Some(daemon),
                    Err(error) => {
                        self.set_error(format!("mDNS unavailable: {error}"));
                        return;
                    }
                }
            }
        }

        let mut browse_started = self
            .browse_started
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *browse_started {
            return;
        }
        let daemon_guard = self.daemon.lock().unwrap_or_else(|e| e.into_inner());
        let Some(daemon) = daemon_guard.as_ref() else {
            return;
        };
        match daemon.browse(SERVICE_TYPE) {
            Ok(receiver) => {
                *browse_started = true;
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
            Err(error) => {
                self.set_error(format!("mDNS browse failed: {error}"));
            }
        }
    }

    pub fn discovered_peers(&self) -> Vec<DiscoveredPeer> {
        self.ensure_daemon_and_browse();
        self.peers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list()
    }

    pub fn sync_advertise(&self, expose: bool, device_name: &str, ports: &[u16]) {
        self.ensure_daemon_and_browse();

        let (lan, _ts) = shareable_ipv4_addrs();
        let ips = lan;
        let instance = sanitize_instance_name(device_name);
        let mut ports = ports.to_vec();
        ports.retain(|p| *p != 0);
        ports.sort_unstable();
        ports.dedup();
        let port = ports.first().copied().unwrap_or(0);

        // UDP beacon works even when mDNS is broken (common on Windows Wi‑Fi).
        {
            let mut beacon = self.beacon_out.lock().unwrap_or_else(|e| e.into_inner());
            if expose && port != 0 && !ips.is_empty() {
                *beacon = Some(BeaconOut {
                    name: instance.clone(),
                    port,
                    ports: ports.clone(),
                    ips: ips.clone(),
                });
            } else {
                *beacon = None;
            }
        }

        let daemon_guard = self.daemon.lock().unwrap_or_else(|e| e.into_inner());
        let Some(daemon) = daemon_guard.as_ref() else {
            if expose && ips.is_empty() {
                self.set_error(
                    "LAN discovery: no Wi‑Fi/Ethernet IPv4 found to advertise".to_string(),
                );
            }
            return;
        };
        let mut advertised = self.advertised.lock().unwrap_or_else(|e| e.into_inner());

        if !expose || port == 0 {
            if let Some(previous) = advertised.take() {
                let _ = daemon.unregister(&previous.fullname);
            }
            return;
        }

        if ips.is_empty() {
            self.set_error(
                "LAN discovery: no Wi‑Fi/Ethernet IPv4 found to advertise".to_string(),
            );
            if let Some(previous) = advertised.take() {
                let _ = daemon.unregister(&previous.fullname);
            }
            return;
        }

        let host_name = format!("{instance}.local.");
        let ip_addrs: Vec<IpAddr> = ips.iter().copied().map(IpAddr::V4).collect();
        let properties = [("path", "/v1"), ("ver", "1"), ("scheme", "https")];
        let Ok(service) = ServiceInfo::new(
            SERVICE_TYPE,
            &instance,
            &host_name,
            ip_addrs.as_slice(),
            port,
            &properties[..],
        ) else {
            self.set_error("mDNS advertise failed: invalid service info".to_string());
            return;
        };
        let service = service.enable_addr_auto();
        let fullname = service.get_fullname().to_string();
        let next = AdvertisedService {
            fullname: fullname.clone(),
            port,
            ips: ips.clone(),
        };
        if advertised.as_ref() == Some(&next) {
            return;
        }
        if let Some(previous) = advertised.take() {
            let _ = daemon.unregister(&previous.fullname);
        }
        match daemon.register(service) {
            Ok(()) => {
                *advertised = Some(next);
                self.clear_error();
            }
            Err(error) => {
                self.set_error(format!("mDNS advertise failed: {error}"));
            }
        }
    }
}

fn encode_beacon(payload: &BeaconPayload) -> Option<Vec<u8>> {
    let body = serde_json::to_vec(payload).ok()?;
    let mut packet = Vec::with_capacity(BEACON_MAGIC.len() + body.len());
    packet.extend_from_slice(BEACON_MAGIC);
    packet.extend_from_slice(&body);
    Some(packet)
}

fn announce_payload(out: &BeaconOut) -> BeaconPayload {
    BeaconPayload {
        v: 1,
        t: "announce".into(),
        name: out.name.clone(),
        port: out.port,
        ports: out.ports.clone(),
        scheme: "https".into(),
        ips: out.ips.iter().map(|ip| ip.to_string()).collect(),
    }
}

fn broadcast_packet(socket: &UdpSocket, packet: &[u8], ips: &[Ipv4Addr]) {
    let _ = socket.send_to(packet, ("255.255.255.255", BEACON_PORT));
    for ip in ips {
        let _ = socket.send_to(packet, (guess_broadcast(*ip), BEACON_PORT));
    }
}

fn spawn_beacon_listener(
    peers: Arc<Mutex<PeerStore>>,
    beacon_out: Arc<Mutex<Option<BeaconOut>>>,
    last_error: Arc<Mutex<Option<String>>>,
) {
    thread::Builder::new()
        .name("tinyinference-beacon-rx".into())
        .spawn(move || {
            let socket = match UdpSocket::bind(("0.0.0.0", BEACON_PORT)) {
                Ok(socket) => socket,
                Err(error) => {
                    *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!(
                        "LAN beacon listen failed on UDP {BEACON_PORT}: {error}. On Windows, allow tinyinference through the firewall (Private network)."
                    ));
                    return;
                }
            };
            let _ = socket.set_broadcast(true);
            let _ = socket.set_read_timeout(Some(Duration::from_secs(1)));
            let mut buf = [0u8; 2048];
            loop {
                match socket.recv_from(&mut buf) {
                    Ok((n, from)) => {
                        let Some(payload) = decode_beacon(&buf[..n]) else {
                            continue;
                        };
                        if payload.v != 1 {
                            continue;
                        }
                        if payload.t == "discover" {
                            // Actively reply so late joiners find us even if they missed announces.
                            if let Some(out) = beacon_out
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .clone()
                            {
                                if let Some(packet) = encode_beacon(&announce_payload(&out)) {
                                    let _ = socket.send_to(&packet, from);
                                }
                            }
                            continue;
                        }
                        if let Some(peer) = peer_from_announce(&payload, from) {
                            if let Ok(mut store) = peers.lock() {
                                store.upsert(peer);
                            }
                        }
                    }
                    Err(error)
                        if error.kind() == ErrorKind::TimedOut
                            || error.kind() == ErrorKind::WouldBlock => {}
                    Err(_) => break,
                }
            }
        })
        .ok();
}

fn spawn_beacon_broadcaster(beacon_out: Arc<Mutex<Option<BeaconOut>>>) {
    thread::Builder::new()
        .name("tinyinference-beacon-tx".into())
        .spawn(move || {
            let Ok(socket) = UdpSocket::bind(("0.0.0.0", 0)) else {
                return;
            };
            let _ = socket.set_broadcast(true);
            loop {
                thread::sleep(BEACON_INTERVAL);
                // Always ask the LAN who is sharing — multicast/mDNS often dies on Windows Wi‑Fi.
                let (lan, _) = shareable_ipv4_addrs();
                if let Some(packet) = encode_beacon(&BeaconPayload {
                    v: 1,
                    t: "discover".into(),
                    name: String::new(),
                    port: 0,
                    ports: Vec::new(),
                    scheme: "https".into(),
                    ips: Vec::new(),
                }) {
                    broadcast_packet(&socket, &packet, &lan);
                }
                let Some(out) = beacon_out
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                else {
                    continue;
                };
                if let Some(packet) = encode_beacon(&announce_payload(&out)) {
                    broadcast_packet(&socket, &packet, &out.ips);
                }
            }
        })
        .ok();
}

fn spawn_subnet_scanner(peers: Arc<Mutex<PeerStore>>) {
    thread::Builder::new()
        .name("tinyinference-lan-scan".into())
        .spawn(move || {
            loop {
                thread::sleep(SUBNET_SCAN_INTERVAL);
                let found = scan_local_subnets_for_peers();
                if found.is_empty() {
                    continue;
                }
                if let Ok(mut store) = peers.lock() {
                    for peer in found {
                        store.upsert(peer);
                    }
                }
            }
        })
        .ok();
}

fn scan_local_subnets_for_peers() -> Vec<DiscoveredPeer> {
    use std::sync::mpsc;
    let (lan, _) = shareable_ipv4_addrs();
    let local: std::collections::HashSet<_> = lan.iter().copied().collect();
    let mut targets = Vec::new();
    for ip in lan.iter().take(2) {
        let o = ip.octets();
        // Home/office /24s only — avoid blasting large enterprise prefixes.
        if !(o[0] == 192 && o[1] == 168) && o[0] != 10 {
            continue;
        }
        for host in 1..=254u8 {
            let target = Ipv4Addr::new(o[0], o[1], o[2], host);
            if local.contains(&target) {
                continue;
            }
            for &port in SCAN_PORTS {
                targets.push((target, port));
            }
        }
    }
    if targets.is_empty() {
        return Vec::new();
    }
    let queue = Arc::new(Mutex::new(targets));
    let (tx, rx) = mpsc::channel();
    let workers = 32usize;
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let queue = Arc::clone(&queue);
        let tx = tx.clone();
        handles.push(thread::spawn(move || loop {
            let next = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let Some((ip, port)) = next else {
                break;
            };
            if let Some(peer) = probe_openai_compatible_peer(ip, port) {
                let _ = tx.send(peer);
            }
        }));
    }
    drop(tx);
    let mut out = Vec::new();
    while let Ok(peer) = rx.recv() {
        out.push(peer);
    }
    for handle in handles {
        let _ = handle.join();
    }
    out
}

fn probe_openai_compatible_peer(ip: Ipv4Addr, port: u16) -> Option<DiscoveredPeer> {
    let addr = SocketAddr::from((ip, port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(120)).ok()?;
    let tls = ureq::tls::TlsConfig::builder()
        .disable_verification(true)
        .build();
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(700)))
        .tls_config(tls)
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let url = format!("https://{ip}:{port}/v1/models");
    let response = agent.get(&url).call().ok()?;
    let status = response.status();
    // llama-server / tinyinference share: 200 with keyless, or 401 when a key is required.
    if status != 200 && status != 401 {
        return None;
    }
    Some(DiscoveredPeer {
        name: format!("lan-{ip}"),
        host: ip.to_string(),
        port,
        ports: vec![port],
        base_url: format!("https://{ip}:{port}/v1"),
        fullname: format!("scan:{ip}:{port}"),
    })
}

/// Windows Firewall silently drops custom UDP discovery ports unless allowed.
/// Best-effort: succeeds without elevation on some setups, otherwise the UI hint covers it.
fn ensure_windows_discovery_firewall() {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use std::process::Command;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let exe = exe.to_string_lossy().replace('"', "");
        let name = "tinyinference LAN discovery";
        let _ = Command::new("netsh")
            .creation_flags(CREATE_NO_WINDOW)
            .args([
                "advfirewall",
                "firewall",
                "delete",
                "rule",
                &format!("name={name}"),
            ])
            .output();
        let _ = Command::new("netsh")
            .creation_flags(CREATE_NO_WINDOW)
            .args([
                "advfirewall",
                "firewall",
                "add",
                "rule",
                &format!("name={name}"),
                "dir=in",
                "action=allow",
                &format!("program={exe}"),
                "protocol=UDP",
                &format!("localport={BEACON_PORT}"),
                "profile=private,domain",
                "enable=yes",
            ])
            .output();
        // Also allow the binary inbound on private networks (covers llama share TCP).
        let _ = Command::new("netsh")
            .creation_flags(CREATE_NO_WINDOW)
            .args([
                "advfirewall",
                "firewall",
                "add",
                "rule",
                "name=tinyinference private inbound",
                "dir=in",
                "action=allow",
                &format!("program={exe}"),
                "profile=private,domain",
                "enable=yes",
            ])
            .output();
    }
}

fn guess_broadcast(ip: Ipv4Addr) -> Ipv4Addr {
    // Home/office LANs are almost always /24; good enough for discovery beacons.
    let o = ip.octets();
    Ipv4Addr::new(o[0], o[1], o[2], 255)
}

fn decode_beacon(bytes: &[u8]) -> Option<BeaconPayload> {
    let body = bytes.strip_prefix(BEACON_MAGIC)?;
    serde_json::from_slice(body).ok()
}

fn peer_from_announce(payload: &BeaconPayload, from: SocketAddr) -> Option<DiscoveredPeer> {
    if payload.port == 0 || payload.name.trim().is_empty() {
        return None;
    }
    if !payload.t.is_empty() && payload.t != "announce" {
        return None;
    }
    let mut ips: Vec<Ipv4Addr> = payload
        .ips
        .iter()
        .filter_map(|ip| ip.parse().ok())
        .collect();
    if ips.is_empty() {
        if let SocketAddr::V4(v4) = from {
            ips.push(*v4.ip());
        }
    }
    ips.retain(|ip| !ip.is_loopback() && !is_tailscale_cg_nat(*ip));
    ips.sort_by_key(|ip| lan_range_rank(*ip));
    let host = ips.first()?.to_string();
    let scheme = if payload.scheme == "http" {
        "http"
    } else {
        "https"
    };
    let name = sanitize_instance_name(&payload.name);
    let mut ports = payload.ports.clone();
    ports.push(payload.port);
    ports.retain(|p| *p != 0);
    ports.sort_unstable();
    ports.dedup();
    Some(DiscoveredPeer {
        name: name.clone(),
        host: host.clone(),
        port: payload.port,
        ports,
        base_url: format!("{scheme}://{host}:{}/v1", payload.port),
        fullname: format!("beacon:{name}:{}", payload.port),
    })
}

fn peer_from_resolved(info: &mdns_sd::ResolvedService) -> Option<DiscoveredPeer> {
    // Do not use is_valid() — on Windows, resolves often arrive with an empty
    // address set while host is still usable, and we'd drop every peer.
    let port = info.port;
    if port == 0 || info.fullname.trim().is_empty() {
        return None;
    }
    // Prefer a real LAN address over Tailscale/CGNAT when both are published.
    let mut v4: Vec<Ipv4Addr> = info.get_addresses_v4().into_iter().collect();
    v4.sort_by_key(|ip| (is_tailscale_cg_nat(*ip), lan_range_rank(*ip)));
    let host = v4
        .first()
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
    // Shared llama always uses HTTPS (+ self-signed cert). TXT may say so explicitly.
    let scheme = info
        .get_property_val_str("scheme")
        .filter(|value| *value == "http" || *value == "https")
        .unwrap_or("https");
    Some(DiscoveredPeer {
        name: short_name,
        host: host.clone(),
        port,
        ports: vec![port],
        base_url: format!("{scheme}://{host}:{port}/v1"),
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

/// One selectable model on a linked host (may span multiple llama ports).
#[derive(Debug, Clone, Serialize)]
pub struct RemoteModelOption {
    pub id: String,
    pub model: String,
    pub base: String,
    pub port: u16,
    pub ready: bool,
    pub label: String,
}

#[derive(Debug, Default)]
pub struct HealthCache {
    last: Mutex<Option<(String, Instant, RemoteHealth)>>,
}

#[derive(Debug, Default)]
pub struct CatalogCache {
    last: Mutex<Option<(String, Instant, Vec<RemoteModelOption>)>>,
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

impl CatalogCache {
    pub fn peek(&self, base: &str, token: &str) -> Option<Vec<RemoteModelOption>> {
        let key = format!("{base}|{token}");
        let Ok(guard) = self.last.lock() else {
            return None;
        };
        match guard.as_ref() {
            Some((cached_key, at, catalog))
                if cached_key == &key && at.elapsed() < HEALTH_CACHE =>
            {
                Some(catalog.clone())
            }
            _ => None,
        }
    }

    pub fn probe(&self, base: &str, token: &str, extra_ports: &[u16]) -> Vec<RemoteModelOption> {
        let key = format!("{base}|{token}");
        if let Some(catalog) = self.peek(base, token) {
            return catalog;
        }
        let catalog = probe_remote_catalog(base, token, extra_ports);
        if let Ok(mut guard) = self.last.lock() {
            *guard = Some((key, Instant::now(), catalog.clone()));
        }
        catalog
    }
}

fn probe_remote_state(base: &str, token: &str) -> RemoteHealth {
    match fetch_remote_models(base, token) {
        Ok(models) => RemoteHealth {
            ok: true,
            model: models.first().cloned(),
            status: Some("ready".into()),
            error: None,
        },
        Err(error) => RemoteHealth {
            ok: false,
            model: None,
            status: None,
            error: Some(error),
        },
    }
}

fn fetch_remote_models(base: &str, token: &str) -> Result<Vec<String>, String> {
    fetch_remote_models_with_timeout(base, token, Duration::from_secs(2))
}

fn fetch_remote_models_with_timeout(
    base: &str,
    token: &str,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let Some(base) = normalize_openai_base(base) else {
        return Err("No remote URL configured".into());
    };
    let url = format!("{base}/models");
    let agent = crate::chat::llm_http_agent(timeout);
    let mut request = agent.get(&url);
    if !token.trim().is_empty() {
        request = request.header("Authorization", &format!("Bearer {}", token.trim()));
    }
    let mut response = request.call().map_err(|error| error.to_string())?;
    if response.status() != 200 {
        return Err(format!("Remote responded with {}", response.status()));
    }
    let body = response
        .body_mut()
        .read_json::<serde_json::Value>()
        .map_err(|error| error.to_string())?;
    let models = body
        .get("data")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if models.is_empty() {
        Err("Remote /models returned no models".into())
    } else {
        Ok(models)
    }
}

/// Probe the linked base and sibling ports on the same host for every ready model.
pub fn probe_remote_catalog(base: &str, token: &str, extra_ports: &[u16]) -> Vec<RemoteModelOption> {
    let Some(primary) = normalize_openai_base(base) else {
        return Vec::new();
    };
    let Some((scheme, host, primary_port)) = split_openai_base(&primary) else {
        return Vec::new();
    };

    // Prefer advertised/discovered ports; only sweep a small neighborhood + common set.
    let mut ports = Vec::new();
    ports.push(primary_port);
    for port in extra_ports {
        ports.push(*port);
    }
    for port in SCAN_PORTS {
        ports.push(*port);
    }
    for delta in 1..=4u16 {
        ports.push(primary_port.saturating_add(delta));
        if primary_port > delta {
            ports.push(primary_port - delta);
        }
    }
    ports.sort_unstable();
    ports.dedup();

    // Fast fail on closed ports so a multi-port scan stays snappy.
    let sibling_timeout = Duration::from_millis(450);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for port in ports {
        let candidate = format!("{scheme}://{host}:{port}/v1");
        let timeout = if port == primary_port {
            Duration::from_secs(2)
        } else {
            sibling_timeout
        };
        let Ok(models) = fetch_remote_models_with_timeout(&candidate, token, timeout) else {
            continue;
        };
        for model in models {
            let id = format!("remote|{candidate}|{model}");
            if !seen.insert(id.clone()) {
                continue;
            }
            out.push(RemoteModelOption {
                id,
                model: model.clone(),
                base: candidate.clone(),
                port,
                ready: true,
                label: model,
            });
        }
    }
    out.sort_by(|a, b| a.port.cmp(&b.port).then(a.model.cmp(&b.model)));
    let multi = out.len() > 1;
    if multi {
        for item in &mut out {
            item.label = format!("{} · :{}", item.model, item.port);
        }
    }
    out
}

pub fn split_openai_base(base: &str) -> Option<(String, String, u16)> {
    let base = normalize_openai_base(base)?;
    let url = base.trim_end_matches("/v1");
    let without_scheme = url.split("://").nth(1)?;
    let scheme = url.split("://").next()?.to_string();
    let authority = without_scheme.split('/').next()?;
    let (host, port) = if let Some((h, p)) = authority.rsplit_once(':') {
        // Handle IPv6 [addr]:port lightly — tinyinference share URLs are IPv4 today.
        let port = p.parse().ok()?;
        (h.to_string(), port)
    } else {
        let port = if scheme == "https" { 443 } else { 80 };
        (authority.to_string(), port)
    };
    Some((scheme, host, port))
}

/// True when `candidate` is an OpenAI base on the same host as `linked`.
pub fn remote_base_same_host(linked: &str, candidate: &str) -> bool {
    let Some((_, host_a, _)) = split_openai_base(linked) else {
        return false;
    };
    let Some((_, host_b, _)) = split_openai_base(candidate) else {
        return false;
    };
    host_a.eq_ignore_ascii_case(&host_b)
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
    fn proxy_bind_hosts_expands_all_interfaces() {
        let mut cfg = NetworkConfig::default();
        cfg.expose = true;
        cfg.listen_scope = ListenScope::All;
        let hosts = cfg.proxy_bind_hosts().unwrap_or_default();
        // Machine-dependent, but must never ask the proxy to bind wildcard.
        assert!(!hosts.iter().any(|h| h == "0.0.0.0" || h == "::"));
        cfg.listen_scope = ListenScope::Custom;
        cfg.listen_host = "10.0.0.187".into();
        assert_eq!(cfg.proxy_bind_hosts().unwrap(), vec!["10.0.0.187".to_string()]);
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

    #[test]
    fn beacon_payload_roundtrip() {
        let payload = announce_payload(&BeaconOut {
            name: "desk".into(),
            port: 8080,
            ports: vec![8080, 8081],
            ips: vec![Ipv4Addr::new(192, 168, 1, 20)],
        });
        let packet = encode_beacon(&payload).unwrap();
        let decoded = decode_beacon(&packet).unwrap();
        let peer = peer_from_announce(&decoded, "192.168.1.20:9".parse().unwrap()).unwrap();
        assert_eq!(peer.name, "desk");
        assert_eq!(peer.port, 8080);
        assert_eq!(peer.ports, vec![8080, 8081]);
        assert_eq!(peer.base_url, "https://192.168.1.20:8080/v1");
    }

    #[test]
    fn api_key_crud_and_legacy_migration() {
        let mut cfg = NetworkConfig::default();
        cfg.access_token = "legacy-secret".into();
        cfg.migrate_api_keys();
        assert_eq!(cfg.api_keys.len(), 1);
        assert_eq!(cfg.api_keys[0].secret, "legacy-secret");
        assert_eq!(cfg.api_keys[0].name, "Default");

        let created = cfg.create_api_key("Phone").unwrap();
        assert_eq!(created.name, "Phone");
        assert_eq!(cfg.api_keys.len(), 2);

        cfg.rename_api_key(&created.id, "Tablet").unwrap();
        assert_eq!(cfg.api_keys[1].name, "Tablet");

        let regenerated = cfg.regenerate_api_key(&created.id).unwrap();
        assert_ne!(regenerated.secret, created.secret);

        cfg.expose = true;
        let first_id = cfg.api_keys[0].id.clone();
        let second_id = cfg.api_keys[1].id.clone();
        assert!(cfg.delete_api_key(&first_id).is_ok());
        assert!(cfg.delete_api_key(&second_id).is_err());
    }
}
