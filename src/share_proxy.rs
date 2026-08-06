//! Host-side “Connected now” activity.
//!
//! Shared clients talk to llama-server directly (TLS + API keys). This module
//! observes established TCP peers on share ports and pairs them with live
//! tok/s from llama logs — no reverse proxy in the data path.

use std::{
    collections::HashMap,
    net::IpAddr,
    process::Command,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

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
    local_port: u16,
    model: String,
    first_seen: Instant,
    last_seen: Instant,
    tokens_per_second: Option<f64>,
    busy: bool,
}

impl ConnectedClient {
    fn state(&self) -> ClientActivityState {
        if self.busy {
            ClientActivityState::Active
        } else {
            ClientActivityState::Idle
        }
    }

    fn to_public(&self) -> ConnectedClientPublic {
        ConnectedClientPublic {
            id: self.id.clone(),
            remote_addr: self.remote_addr.clone(),
            key_id: String::new(),
            key_name: "Caller".into(),
            local_port: self.local_port,
            model: self.model.clone(),
            last_path: "/v1/*".into(),
            last_method: "TCP".into(),
            state: self.state(),
            started_at: unix_secs(self.first_seen),
            last_seen: unix_secs(self.last_seen),
            tokens_per_second: if self.busy {
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
    port_tps: HashMap<u16, Option<f64>>,
}

/// Observes inbound TCP peers on shared llama ports.
pub struct ShareProxyManager {
    activity: Mutex<ShareActivityStore>,
    last_error: Option<String>,
}

impl Default for ShareProxyManager {
    fn default() -> Self {
        Self {
            activity: Mutex::new(ShareActivityStore::default()),
            last_error: None,
        }
    }
}

impl ShareProxyManager {
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn update_port_tps(&self, port: u16, tps: Option<f64>) {
        if let Ok(mut store) = self.activity.lock() {
            store.port_tps.insert(port, tps);
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
        if let Ok(mut store) = self.activity.lock() {
            store.clients.clear();
            store.port_tps.clear();
        }
        self.last_error = None;
    }

    /// Refresh connected peers for the given public share ports.
    pub fn sync(&mut self, expose: bool, ports: &[(u16, String)]) {
        if !expose || ports.is_empty() {
            self.shutdown_all();
            return;
        }

        let public_ports: Vec<u16> = ports.iter().map(|(p, _)| *p).collect();
        let models: HashMap<u16, String> = ports.iter().cloned().collect();

        match list_established_peers(&public_ports) {
            Ok(peers) => {
                self.last_error = None;
                if let Ok(mut store) = self.activity.lock() {
                    store.reconcile(peers, &models);
                }
            }
            Err(error) => {
                self.last_error = Some(format!("could not list connections: {error}"));
            }
        }
    }
}

impl ShareActivityStore {
    fn reconcile(&mut self, peers: Vec<(IpAddr, u16)>, models: &HashMap<u16, String>) {
        let now = Instant::now();
        let mut seen = std::collections::HashSet::new();
        for (ip, local_port) in peers {
            if ip.is_loopback() {
                continue;
            }
            let remote = ip.to_string();
            let id = format!("{remote}|{local_port}");
            seen.insert(id.clone());
            let model = models
                .get(&local_port)
                .cloned()
                .unwrap_or_else(|| format!(":{local_port}"));
            let tps = self.port_tps.get(&local_port).copied().flatten();
            let busy = tps.is_some_and(|rate| rate > 0.0);
            let entry = self.clients.entry(id.clone()).or_insert_with(|| ConnectedClient {
                id: id.clone(),
                remote_addr: remote.clone(),
                local_port,
                model: model.clone(),
                first_seen: now,
                last_seen: now,
                tokens_per_second: tps,
                busy,
            });
            entry.remote_addr = remote;
            entry.local_port = local_port;
            entry.model = model;
            entry.last_seen = now;
            entry.tokens_per_second = tps;
            entry.busy = busy;
        }

        let cutoff = now - IDLE_PRUNE;
        self.clients.retain(|id, client| {
            seen.contains(id) || (client.last_seen >= cutoff && !client.remote_addr.is_empty())
        });
        // Drop peers that disappeared from the socket table immediately.
        self.clients.retain(|id, _| seen.contains(id));
    }

    fn snapshot(&mut self) -> (Vec<ConnectedClientPublic>, ShareActivitySummary) {
        let mut clients: Vec<_> = self.clients.values().map(ConnectedClient::to_public).collect();
        clients.sort_by(|a, b| {
            b.last_seen
                .cmp(&a.last_seen)
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

fn list_established_peers(local_ports: &[u16]) -> Result<Vec<(IpAddr, u16)>, String> {
    if local_ports.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(windows)]
    {
        list_peers_netstat(local_ports)
    }
    #[cfg(not(windows))]
    {
        list_peers_netstat(local_ports)
    }
}

fn list_peers_netstat(local_ports: &[u16]) -> Result<Vec<(IpAddr, u16)>, String> {
    let output = Command::new("netstat")
        .args(["-ano", "-p", "TCP"])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() && output.stdout.is_empty() {
        return Err("netstat failed".into());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let wanted: std::collections::HashSet<u16> = local_ports.iter().copied().collect();
    let mut peers = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("TCP") {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 4 {
            continue;
        }
        let state = cols[3];
        if !state.eq_ignore_ascii_case("ESTABLISHED") {
            continue;
        }
        let Some((local_ip, local_port)) = split_ip_port(cols[1]) else {
            continue;
        };
        if !wanted.contains(&local_port) {
            continue;
        }
        let Some((remote_ip, _)) = split_ip_port(cols[2]) else {
            continue;
        };
        if remote_ip.is_unspecified() || remote_ip.is_loopback() {
            continue;
        }
        // Ignore our own listen-side weirdness; require a real peer.
        let _ = local_ip;
        peers.push((remote_ip, local_port));
    }
    peers.sort_by_key(|(ip, port)| (ip.to_string(), *port));
    peers.dedup();
    Ok(peers)
}

fn split_ip_port(value: &str) -> Option<(IpAddr, u16)> {
    // netstat uses 10.0.0.1:8080 or [fe80::1]:8080
    if let Some(rest) = value.strip_prefix('[') {
        let (ip, port) = rest.split_once("]:")?;
        return Some((ip.parse().ok()?, port.parse().ok()?));
    }
    let (ip, port) = value.rsplit_once(':')?;
    Some((ip.parse().ok()?, port.parse().ok()?))
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
    fn split_ip_port_parses_v4() {
        let (ip, port) = split_ip_port("10.0.0.151:52344").unwrap();
        assert_eq!(ip.to_string(), "10.0.0.151");
        assert_eq!(port, 52344);
    }
}
