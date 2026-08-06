//! One managed llama-server process (model + port + lifecycle).

use std::time::{Duration, Instant};

use crate::{
    app::{Download, ServerStatus},
    config::Config,
    server::{
        PendingProbe, PendingThinkingProbe, ProbeResult, ServerEvent, ServerMetrics, ServerProcess,
        SlotsSnapshot, parse_log_throughput, probe_async, thinking_support_async,
    },
    system::{ProcessMonitor, ProcessUsage},
};

const METRICS_PROBE_INTERVAL: Duration = Duration::from_millis(250);
const THROUGHPUT_STALE_AFTER: Duration = Duration::from_secs(10);
const MIN_LIVE_RATE_INTERVAL: Duration = Duration::from_millis(100);

/// Runtime state for a single llama-server child.
pub struct ManagedServer {
    pub id: String,
    pub process: Option<ServerProcess>,
    pub running_config: Config,
    pub status: ServerStatus,
    pub status_detail: String,
    pub endpoint_online: bool,
    pub download: Option<Download>,
    pub process_usage: Option<ProcessUsage>,
    pub server_metrics: Option<ServerMetrics>,
    process_monitor: ProcessMonitor,
    probe: Option<PendingProbe>,
    last_probe: Instant,
    last_stats_refresh: Instant,
    last_slots_at: Option<Instant>,
    last_slots_decoded: Option<u64>,
    live_generated_tps: Option<f64>,
    last_throughput_at: Option<Instant>,
    thinking_supported: Option<bool>,
    thinking_probe_for: Option<String>,
    pending_thinking_probe: Option<PendingThinkingProbe>,
}

impl ManagedServer {
    #[cfg(test)]
    pub fn stub_ready(id: String, running_config: Config) -> Self {
        Self {
            id,
            process: None,
            running_config,
            status: ServerStatus::Ready,
            status_detail: String::new(),
            endpoint_online: true,
            download: None,
            process_usage: None,
            server_metrics: None,
            process_monitor: ProcessMonitor::default(),
            probe: None,
            last_probe: Instant::now(),
            last_stats_refresh: Instant::now(),
            last_slots_at: None,
            last_slots_decoded: None,
            live_generated_tps: None,
            last_throughput_at: None,
            thinking_supported: None,
            thinking_probe_for: None,
            pending_thinking_probe: None,
        }
    }

    pub fn new_launching(
        id: String,
        running_config: Config,
        process: ServerProcess,
        download: Option<Download>,
    ) -> Self {
        let pid = process.id();
        let (status, status_detail) = if download.as_ref().is_some_and(|d| d.is_active()) {
            (
                ServerStatus::Downloading,
                "Fetching the model from Hugging Face".into(),
            )
        } else {
            (
                ServerStatus::Starting,
                format!("Waking llama-server (PID {pid})"),
            )
        };
        Self {
            id,
            process: Some(process),
            running_config,
            status,
            status_detail,
            endpoint_online: false,
            download,
            process_usage: None,
            server_metrics: None,
            process_monitor: ProcessMonitor::default(),
            probe: None,
            last_probe: Instant::now() - Duration::from_secs(2),
            last_stats_refresh: Instant::now() - Duration::from_secs(2),
            last_slots_at: None,
            last_slots_decoded: None,
            live_generated_tps: None,
            last_throughput_at: None,
            thinking_supported: None,
            thinking_probe_for: None,
            pending_thinking_probe: None,
        }
    }

    pub fn model_label(&self) -> String {
        self.running_config.model_label()
    }

    pub fn port(&self) -> u16 {
        self.running_config.effective_port()
    }

    pub fn is_running(&self) -> bool {
        self.process.is_some()
    }

    pub fn thinking_supported_flag(&self) -> bool {
        // Fail closed while the /props probe is still in flight.
        self.thinking_supported.unwrap_or(false)
    }

    pub fn tick(&mut self) -> Vec<String> {
        let mut log_lines = Vec::new();
        if let Some(process) = self.process.as_ref() {
            for event in process.drain_logs() {
                let ServerEvent::Log(line) = event;
                log_lines.push(line);
            }
        }
        for line in &log_lines {
            self.observe_throughput_line(line);
        }

        let exit = match self.process.as_mut() {
            Some(process) => process.try_wait(),
            None => Ok(None),
        };
        match exit {
            Ok(Some(status)) => {
                if let Some(process) = self.process.as_mut() {
                    process.finish_output();
                    for event in process.drain_logs() {
                        let ServerEvent::Log(line) = event;
                        log_lines.push(line);
                    }
                }
                self.process = None;
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
                log_lines.push(format!("[{}] {}", self.port(), self.status_detail));
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
            }
            Ok(None) => {}
        }

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
                self.probe = Some(probe_async(&self.running_config, true));
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
        log_lines
    }

    fn apply_probe_result(&mut self, result: ProbeResult) {
        self.endpoint_online = result.endpoint_online;
        if let Some(slots) = result.slots {
            self.update_live_throughput(slots);
        }
        if result.metrics_requested {
            if let Some(metrics) = result.metrics {
                self.merge_server_metrics(metrics);
            }
        }
        self.publish_live_rates();
        if self.endpoint_online {
            self.download = None;
            self.status = ServerStatus::Ready;
            self.status_detail = format!("Listening on {}", self.running_config.listen_label());
            self.ensure_thinking_probe();
        } else if self.status != ServerStatus::Downloading {
            self.status = ServerStatus::Starting;
        }
    }

    fn clear_thinking_support(&mut self) {
        self.thinking_supported = None;
        self.thinking_probe_for = None;
        self.pending_thinking_probe = None;
    }

    fn ensure_thinking_probe(&mut self) {
        let label = self.running_config.model_label();
        if self.thinking_probe_for.as_deref() == Some(label.as_str())
            && (self.thinking_supported.is_some() || self.pending_thinking_probe.is_some())
        {
            return;
        }
        self.thinking_supported = None;
        self.thinking_probe_for = Some(label);
        self.pending_thinking_probe = Some(thinking_support_async(&self.running_config));
    }

    fn poll_thinking_support(&mut self) {
        if let Some(result) = self
            .pending_thinking_probe
            .as_ref()
            .and_then(PendingThinkingProbe::take)
        {
            self.pending_thinking_probe = None;
            self.thinking_supported = Some(result.unwrap_or(false));
        } else if self.status == ServerStatus::Ready {
            self.ensure_thinking_probe();
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
            self.live_generated_tps = None;
            return;
        }

        if let (Some(prev_at), Some(prev_decoded)) = (self.last_slots_at, self.last_slots_decoded)
        {
            let dt = now.saturating_duration_since(prev_at);
            if dt >= MIN_LIVE_RATE_INTERVAL {
                if slots.decoded_tokens >= prev_decoded {
                    let delta = slots.decoded_tokens - prev_decoded;
                    if delta > 0 {
                        self.live_generated_tps = Some(delta as f64 / dt.as_secs_f64());
                        self.touch_throughput();
                    }
                } else {
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

    fn observe_throughput_line(&mut self, line: &str) {
        let parsed = parse_log_throughput(line);
        if parsed.generated_tokens_per_second.is_none()
            && parsed.prompt_tokens_per_second.is_none()
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

    pub fn stop(&mut self) -> Vec<String> {
        let Some(mut process) = self.process.take() else {
            return Vec::new();
        };
        self.status = ServerStatus::Stopping;
        let stop_result = process.stop();
        let mut lines = process
            .drain_logs()
            .map(|ServerEvent::Log(line)| line)
            .collect::<Vec<_>>();
        match stop_result {
            Ok(()) => {
                self.status = ServerStatus::Stopped;
                self.status_detail = "Stopped by user".into();
                lines.push(format!("[{}] llama-server stopped", self.port()));
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
                lines.push(format!("[{}] [stop failed] {error:#}", self.port()));
            }
        }
        self.endpoint_online = false;
        self.probe = None;
        self.download = None;
        self.process_usage = None;
        self.server_metrics = None;
        self.clear_live_throughput();
        self.clear_thinking_support();
        lines
    }
}
