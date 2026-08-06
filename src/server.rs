use std::{
    ffi::OsString,
    io::{BufRead, BufReader, Read as _, Write},
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, bail};

use crate::{
    config::{Config, ModelSource},
    system::recommended_threads,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
}

impl CommandSpec {
    pub fn from_config(config: &Config) -> Self {
        Self::from_config_with_threads(config, recommended_threads())
    }

    pub fn from_config_with_threads(config: &Config, threads: usize) -> Self {
        let mut args = Vec::<OsString>::new();
        match &config.model.source {
            ModelSource::HuggingFace(id) => push_pair(&mut args, "-hf", id.as_str()),
            ModelSource::Local(path) => {
                args.push("-m".into());
                args.push(path.as_os_str().into());
            }
        }

        // Advanced extras first; managed flags are stripped so Configure / Network win.
        let filtered_extras = filter_managed_extra_args(&config.server.extra_args);
        args.extend(filtered_extras.iter().map(OsString::from));

        if config.runtime.cpu_only {
            push_pair(&mut args, "--device", "none");
            push_pair(&mut args, "--n-gpu-layers", "0");
        } else if !config.runtime.fit {
            // --fit refuses to run when n_gpu_layers is already set by the user.
            // Leave layer count unset so fit can choose how many layers fit in VRAM.
            push_pair(&mut args, "--n-gpu-layers", "999");
        }

        push_pair(
            &mut args,
            "--fit",
            if config.runtime.fit { "on" } else { "off" },
        );
        args.push(
            if config.runtime.mmap {
                "--mmap"
            } else {
                "--no-mmap"
            }
            .into(),
        );
        args.push(
            if config.runtime.repack {
                "--repack"
            } else {
                "--no-repack"
            }
            .into(),
        );
        args.push(
            if config.runtime.warmup {
                "--warmup"
            } else {
                "--no-warmup"
            }
            .into(),
        );
        push_pair(
            &mut args,
            "--flash-attn",
            if config.runtime.flash_attn {
                "on"
            } else {
                "off"
            },
        );
        push_pair(
            &mut args,
            "--cache-type-k",
            config.runtime.cache_type_k.as_str(),
        );
        push_pair(
            &mut args,
            "--cache-type-v",
            config.runtime.cache_type_v.as_str(),
        );
        push_pair(
            &mut args,
            "--ctx-size",
            config.runtime.context_size.to_string(),
        );
        push_pair(
            &mut args,
            "--batch-size",
            config.runtime.batch_size.to_string(),
        );
        push_pair(
            &mut args,
            "--ubatch-size",
            config.runtime.micro_batch_size.to_string(),
        );
        push_pair(&mut args, "--parallel", config.runtime.parallel.to_string());
        push_pair(
            &mut args,
            "--cache-ram",
            config.runtime.cache_ram_mib.to_string(),
        );
        push_pair(
            &mut args,
            "--ctx-checkpoints",
            config.runtime.context_checkpoints.to_string(),
        );
        if !config.runtime.multimodal_projector {
            args.push("--no-mmproj".into());
        }
        if config.runtime.jinja {
            args.push("--jinja".into());
        }
        // Prefer physical cores; skip when extras already set --threads.
        if !has_threads_override(&filtered_extras) {
            push_pair(&mut args, "--threads", threads.max(1).to_string());
        }
        // Local-only observability. When sharing, prefer an inference-only surface.
        if config.network.expose {
            args.push("--no-webui".into());
            args.push("--no-slots".into());
        } else {
            args.push("--metrics".into());
        }
        push_pair(&mut args, "--host", config.effective_host());
        push_pair(
            &mut args,
            "--port",
            config.effective_port().to_string(),
        );
        if config.uses_tls() {
            if let (Some(cert), Some(key)) = (&config.tls_cert_file, &config.tls_key_file) {
                push_pair(&mut args, "--ssl-cert-file", cert.as_os_str());
                push_pair(&mut args, "--ssl-key-file", key.as_os_str());
            }
        }
        for key in config.llama_api_keys() {
            if !key.trim().is_empty() {
                push_pair(&mut args, "--api-key", key);
            }
        }

        Self {
            program: config.server.executable.as_os_str().into(),
            args,
        }
    }

    pub fn display(&self) -> String {
        std::iter::once(shell_quote(&self.program))
            .chain(self.args.iter().map(shell_quote))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn push_pair(args: &mut Vec<OsString>, flag: impl Into<OsString>, value: impl Into<OsString>) {
    args.push(flag.into());
    args.push(value.into());
}

fn has_threads_override(extra_args: &[String]) -> bool {
    has_extra_option(extra_args, "--threads")
}

/// Drop flags owned by Configure / Network so `extra_args` cannot bypass them.
fn filter_managed_extra_args(extra_args: &[String]) -> Vec<String> {
    const PAIRED: &[&str] = &[
        "--host",
        "--port",
        "--api-key",
        "--ssl-cert-file",
        "--ssl-key-file",
        "--device",
        "--n-gpu-layers",
        "--fit",
        "--flash-attn",
        "-fa",
        "--cache-type-k",
        "-ctk",
        "--cache-type-v",
        "-ctv",
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--parallel",
        "--cache-ram",
        "--ctx-checkpoints",
    ];
    const FLAGS: &[&str] = &[
        "--mmap",
        "--no-mmap",
        "--repack",
        "--no-repack",
        "--warmup",
        "--no-warmup",
        "--no-mmproj",
        "--jinja",
        "--metrics",
        "--no-webui",
        "--no-ui",
        "--no-slots",
    ];

    let mut out = Vec::with_capacity(extra_args.len());
    let mut index = 0;
    while index < extra_args.len() {
        let argument = &extra_args[index];
        if FLAGS.iter().any(|flag| argument == flag) {
            index += 1;
            continue;
        }
        if let Some(option) = PAIRED.iter().find(|option| {
            *argument == **option || argument.starts_with(&format!("{option}="))
        }) {
            if *argument == *option {
                index += 2;
            } else {
                index += 1;
            }
            let _ = option;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out
}

fn has_extra_option(extra_args: &[String], option: &str) -> bool {
    let prefix = format!("{option}=");
    extra_args
        .iter()
        .any(|argument| argument == option || argument.starts_with(&prefix))
}

fn shell_quote(value: &OsString) -> String {
    let text = value.to_string_lossy();
    if text
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./:\\@".contains(c))
    {
        text.into_owned()
    } else {
        format!("\"{}\"", text.replace('"', "\\\""))
    }
}

#[derive(Debug)]
pub enum ServerEvent {
    Log(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerMetrics {
    pub prompt_tokens: Option<f64>,
    pub generated_tokens: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    pub generated_tokens_per_second: Option<f64>,
    pub requests_processing: Option<f64>,
    pub requests_deferred: Option<f64>,
}

/// Live slot counters from llama-server `/slots`.
///
/// Prometheus `predicted_tokens_seconds` only updates when a request finishes,
/// so mid-generation tok/s has to be derived from `n_decoded` deltas instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotsSnapshot {
    pub decoded_tokens: u64,
    pub requests_processing: u64,
}

#[derive(Debug)]
pub struct ProbeResult {
    pub endpoint_online: bool,
    pub metrics_requested: bool,
    pub metrics: Option<ServerMetrics>,
    pub slots: Option<SlotsSnapshot>,
}

#[derive(Debug)]
pub struct PendingThinkingProbe {
    receiver: Receiver<Option<bool>>,
}

impl PendingThinkingProbe {
    pub fn take(&self) -> Option<Option<bool>> {
        self.receiver.try_recv().ok()
    }
}

pub fn thinking_support_async(config: &Config) -> PendingThinkingProbe {
    let (sender, receiver) = mpsc::channel();
    let config = config.clone();
    thread::spawn(move || {
        let _ = sender.send(fetch_thinking_support(&config));
    });
    PendingThinkingProbe { receiver }
}

#[derive(Debug)]
pub struct PendingProbe {
    receiver: Receiver<ProbeResult>,
}

impl PendingProbe {
    pub fn take(&self) -> Option<ProbeResult> {
        self.receiver.try_recv().ok()
    }
}

pub fn probe_async(config: &Config, metrics_requested: bool) -> PendingProbe {
    let (sender, receiver) = mpsc::channel();
    let config = config.clone();
    thread::spawn(move || {
        // `/slots` and `/metrics` are served from the llama-server task queue and
        // only run between decode steps. On slow CPU hosts a single token can take
        // longer than a short HTTP timeout, so wait generously and prefer `/slots`
        // (it carries live `n_decoded`) over a separate `/health` round-trip.
        let slots = fetch_slots(&config);
        // While tokens are actively decoding, skip `/metrics` — its tok/s gauges
        // only update when a request finishes, and a second task-queue wait just
        // delays the next live `/slots` sample. Still scrape during prompt eval
        // (`n_decoded == 0`) and when idle so totals/prompt rates stay fresh.
        let metrics = if metrics_requested {
            match &slots {
                Some(snapshot)
                    if snapshot.requests_processing > 0 && snapshot.decoded_tokens > 0 =>
                {
                    None
                }
                _ => fetch_metrics(&config),
            }
        } else {
            None
        };
        let endpoint_online = slots.is_some() || metrics.is_some() || endpoint_healthy(&config);
        let _ = sender.send(ProbeResult {
            endpoint_online,
            metrics_requested,
            metrics,
            slots,
        });
    });
    PendingProbe { receiver }
}

#[derive(Debug)]
pub struct ServerProcess {
    child: Child,
    receiver: Receiver<ServerEvent>,
    readers: Vec<JoinHandle<()>>,
}

impl ServerProcess {
    pub fn start(config: &Config) -> Result<Self> {
        let errors = config.validate();
        if !errors.is_empty() {
            bail!("configuration is invalid: {}", errors.join("; "));
        }
        let spec = CommandSpec::from_config(config);
        Self::start_spec(&spec)
    }

    fn start_spec(spec: &CommandSpec) -> Result<Self> {
        let mut child = Command::new(&spec.program)
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "could not start {} — install llama.cpp or set the executable in Configure",
                    spec.program.to_string_lossy()
                )
            })?;

        let (sender, receiver) = mpsc::channel();
        let mut readers = Vec::with_capacity(2);
        if let Some(stdout) = child.stdout.take() {
            readers.push(stream_lines(stdout, sender.clone()));
        }
        if let Some(stderr) = child.stderr.take() {
            readers.push(stream_lines(stderr, sender));
        }
        Ok(Self {
            child,
            receiver,
            readers,
        })
    }

    pub fn drain_logs(&self) -> impl Iterator<Item = ServerEvent> + '_ {
        self.receiver.try_iter()
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child
            .try_wait()
            .context("could not inspect llama-server")
    }

    pub fn stop(&mut self) -> Result<()> {
        if self.child.try_wait()?.is_some() {
            self.finish_output();
            return Ok(());
        }
        self.child.kill().context("could not stop llama-server")?;
        self.child.wait().context("could not reap llama-server")?;
        self.finish_output();
        Ok(())
    }

    pub fn finish_output(&mut self) {
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        self.finish_output();
    }
}

fn stream_lines<R: std::io::Read + Send + 'static>(
    reader: R,
    sender: Sender<ServerEvent>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => {
                    if sender.send(ServerEvent::Log(line)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(ServerEvent::Log(format!("[stream error] {error}")));
                    break;
                }
            }
        }
    })
}

pub fn endpoint_healthy(config: &Config) -> bool {
    http_get(config, "/health", Duration::from_millis(500)).is_some_and(|(status, _)| status == 200)
}

/// Whether the loaded model/template supports thinking, from llama-server `GET /props`.
///
/// Evidence is capability flags and the chat template itself (`<think>` tags,
/// `enable_thinking`, `reasoning_effort`, …) — not a hardcoded model-name list.
pub fn fetch_thinking_support(config: &Config) -> Option<bool> {
    let (status, body) = http_get(config, "/props", Duration::from_secs(2))?;
    if status != 200 {
        return None;
    }
    Some(thinking_support_from_props(&body))
}

fn thinking_support_from_props(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    if let Some(caps) = value.get("chat_template_caps").and_then(|v| v.as_object()) {
        const FLAGS: &[&str] = &[
            "supports_thinking",
            "enable_thinking",
            "supports_enable_thinking",
            "supports_reasoning_effort",
            "reasoning_budget",
            "supports_preserve_reasoning",
        ];
        if FLAGS
            .iter()
            .any(|key| caps.get(*key).and_then(|v| v.as_bool()) == Some(true))
        {
            return true;
        }
    }
    // Inspect only the chat template string — not the whole /props JSON dump
    // (which can mention thinking elsewhere without the template using it).
    value
        .get("chat_template")
        .and_then(|v| v.as_str())
        .is_some_and(template_suggests_thinking)
}

fn template_suggests_thinking(template: &str) -> bool {
    let lower = template.to_ascii_lowercase();
    lower.contains("<think>")
        || lower.contains("</think>")
        || lower.contains("<thinking>")
        || lower.contains("</thinking>")
        || lower.contains("enable_thinking")
        || lower.contains("reasoning_effort")
        || lower.contains("thinking_budget")
        || lower.contains("reasoning_content")
        || lower.contains("redacted_thinking")
}

#[cfg(test)]
mod thinking_support_tests {
    use super::thinking_support_from_props;

    #[test]
    fn props_detect_think_tags_and_controls_in_template() {
        let with_tags =
            r#"{"chat_template":"User: {{message}}\nAssistant: <think>maybe</think>"}"#;
        assert!(thinking_support_from_props(with_tags));

        let controlled =
            r#"{"chat_template":"{% if enable_thinking %}...{% endif %}","chat_template_caps":{}}"#;
        assert!(thinking_support_from_props(controlled));

        let caps = r#"{"chat_template_caps":{"supports_reasoning_effort":true}}"#;
        assert!(thinking_support_from_props(caps));

        let plain = r#"{"chat_template":"{{ messages }}","chat_template_caps":{}}"#;
        assert!(!thinking_support_from_props(plain));

        // Mentions outside chat_template must not count.
        let noise = r#"{"model_path":"/models/foo-thinking.gguf","chat_template":"{{ messages }}"}"#;
        assert!(!thinking_support_from_props(noise));
    }
}

/// Timeout for task-queue endpoints (`/slots`, `/metrics`). These only answer
/// between `llama_decode` steps, so short timeouts falsely look "offline".
const TASK_QUEUE_HTTP_TIMEOUT: Duration = Duration::from_secs(5);

pub fn fetch_metrics(config: &Config) -> Option<ServerMetrics> {
    let (status, body) = http_get(config, "/metrics", TASK_QUEUE_HTTP_TIMEOUT)?;
    (status == 200).then(|| ServerMetrics {
        prompt_tokens: metric_value(&body, "llamacpp:prompt_tokens_total"),
        generated_tokens: metric_value(&body, "llamacpp:tokens_predicted_total"),
        prompt_tokens_per_second: metric_value(&body, "llamacpp:prompt_tokens_seconds"),
        generated_tokens_per_second: metric_value(&body, "llamacpp:predicted_tokens_seconds"),
        requests_processing: metric_value(&body, "llamacpp:requests_processing"),
        requests_deferred: metric_value(&body, "llamacpp:requests_deferred"),
    })
}

pub fn fetch_slots(config: &Config) -> Option<SlotsSnapshot> {
    let (status, body) = http_get(config, "/slots", TASK_QUEUE_HTTP_TIMEOUT)?;
    if status != 200 {
        return None;
    }
    parse_slots_snapshot(&body)
}

fn parse_slots_snapshot(body: &str) -> Option<SlotsSnapshot> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let slots = value.as_array()?;
    let mut decoded_tokens = 0_u64;
    let mut requests_processing = 0_u64;
    for slot in slots {
        if slot.get("is_processing").and_then(|flag| flag.as_bool()) != Some(true) {
            continue;
        }
        requests_processing += 1;
        let decoded = slot
            .pointer("/next_token/n_decoded")
            .and_then(json_u64)
            .unwrap_or(0);
        decoded_tokens = decoded_tokens.saturating_add(decoded);
    }
    Some(SlotsSnapshot {
        decoded_tokens,
        requests_processing,
    })
}

fn json_u64(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
        .or_else(|| value.as_f64().and_then(|n| (n >= 0.0).then_some(n as u64)))
}

fn http_get(config: &Config, path: &str, timeout: Duration) -> Option<(u16, String)> {
    if config.uses_tls() {
        return https_get(config, path, timeout);
    }
    let host = config.connect_host();
    let port = config.effective_port();
    let address = host
        .parse::<IpAddr>()
        .ok()
        .map(|ip| SocketAddr::new(ip, port))
        .or_else(|| (host.as_str(), port).to_socket_addrs().ok()?.next());
    let mut stream =
        address.and_then(|address| TcpStream::connect_timeout(&address, timeout).ok())?;
    if stream.set_read_timeout(Some(timeout)).is_err()
        || stream.set_write_timeout(Some(timeout)).is_err()
    {
        return None;
    }
    let host_header = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let auth_header = {
        let key = config.server.api_key.trim();
        if key.is_empty() {
            String::new()
        } else {
            format!("Authorization: Bearer {key}\r\n")
        }
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\n{auth_header}Connection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    (&mut stream)
        .take(256 * 1024)
        .read_to_string(&mut response)
        .ok()?;
    let status = response
        .lines()
        .next()?
        .split_ascii_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or_default()
        .to_string();
    Some((status, body))
}

/// Probes against our own self-signed llama TLS (verification intentionally off).
fn https_get(config: &Config, path: &str, timeout: Duration) -> Option<(u16, String)> {
    let url = format!(
        "{}{}",
        config.endpoint().trim_end_matches('/'),
        path
    );
    let tls = ureq::tls::TlsConfig::builder()
        .disable_verification(true)
        .build();
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .tls_config(tls)
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut request = agent.get(&url);
    let key = config.server.api_key.trim();
    if !key.is_empty() {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let mut response = request.call().ok()?;
    let status = u16::from(response.status());
    let body = response.body_mut().read_to_string().ok()?;
    Some((status, body))
}

fn metric_value(body: &str, name: &str) -> Option<f64> {
    body.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut fields = line.split_ascii_whitespace();
            let metric = fields.next()?.split('{').next()?;
            if metric == name {
                fields.next()?.parse().ok()
            } else {
                None
            }
        })
}

/// Throughput scraped from llama-server log lines (same numbers the console shows).
///
/// HTTP `/metrics` gauges only update when a request finishes, and `/slots` waits
/// on the decode task queue. Logs are emitted from inside the decode loop, so
/// they update as soon as llama.cpp itself prints them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogThroughput {
    pub generated_tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
}

pub fn parse_log_throughput(line: &str) -> LogThroughput {
    let mut throughput = LogThroughput::default();

    // "n_decoded = 123, tg = 12.34 t/s, tg_3s = 15.67 t/s"
    // Prefer the recent window when present — that is the live feel users see.
    if let Some(rate) = rate_after_marker(line, "tg_3s =") {
        throughput.generated_tokens_per_second = Some(rate);
    } else if let Some(rate) =
        rate_after_marker(line, ", tg =").or_else(|| rate_after_marker(line, " tg ="))
    {
        throughput.generated_tokens_per_second = Some(rate);
    }

    // Mid-prompt: "prompt processing, n_tokens = ... / 12.34 tokens per second"
    if line.contains("prompt processing")
        && let Some(rate) = rate_before_unit(line, "tokens per second")
    {
        throughput.prompt_tokens_per_second = Some(rate);
    }

    // Final timings after a request completes.
    if line.contains("prompt eval time =")
        && let Some(rate) = rate_before_unit(line, "tokens per second")
    {
        throughput.prompt_tokens_per_second = Some(rate);
    } else if line.contains("eval time =")
        && !line.contains("prompt eval")
        && let Some(rate) = rate_before_unit(line, "tokens per second")
    {
        throughput.generated_tokens_per_second = Some(rate);
    }

    throughput
}

fn rate_after_marker(line: &str, marker: &str) -> Option<f64> {
    let rest = line.split_once(marker)?.1.trim_start();
    let token = rest.split_whitespace().next()?;
    token
        .parse()
        .ok()
        .filter(|rate: &f64| rate.is_finite() && *rate > 0.0)
}

fn rate_before_unit(line: &str, unit: &str) -> Option<f64> {
    let head = line.rsplit_once(unit)?.0.trim_end();
    let token = head.split_whitespace().next_back()?;
    // Strip a trailing comma from formats like "(12.34,"
    let token = token.trim_end_matches(',');
    token
        .parse()
        .ok()
        .filter(|rate: &f64| rate.is_finite() && *rate > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(config: &Config) -> Vec<String> {
        CommandSpec::from_config_with_threads(config, 8)
            .args
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn default_command_contains_low_memory_flags() {
        let actual = args(&Config::default());
        assert_eq!(
            actual,
            [
                "-hf",
                "ggml-org/gpt-oss-120b-GGUF",
                "--device",
                "none",
                "--n-gpu-layers",
                "0",
                "--fit",
                "off",
                "--mmap",
                "--no-repack",
                "--warmup",
                "--flash-attn",
                "on",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--ctx-size",
                "8192",
                "--batch-size",
                "8",
                "--ubatch-size",
                "8",
                "--parallel",
                "1",
                "--cache-ram",
                "0",
                "--ctx-checkpoints",
                "0",
                "--no-mmproj",
                "--jinja",
                "--threads",
                "8",
                "--metrics",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
            ]
        );
    }

    #[test]
    fn local_model_uses_model_flag() {
        let mut config = Config::default();
        config.model.source = ModelSource::Local("model.gguf".into());
        let actual = args(&config);
        assert_eq!(&actual[..2], &["-m", "model.gguf"]);
    }

    #[test]
    fn gpu_fit_preset_command_uses_gpu_and_fit() {
        let mut config = Config::default();
        config.runtime = crate::config::RuntimePreset::GpuFit.runtime();
        let actual = args(&config);
        assert!(!actual.iter().any(|argument| argument == "--device"));
        // Fit must own n_gpu_layers; setting it to 999 makes llama.cpp abort auto-fit.
        assert!(!actual.iter().any(|argument| argument == "--n-gpu-layers"));
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--fit" && window[1] == "on")
        );
        assert!(actual.iter().any(|argument| argument == "--mmap"));
    }

    #[test]
    fn auto_threads_use_physical_core_count() {
        let actual = args(&Config::default());
        let threads = actual
            .windows(2)
            .find_map(|window| (window[0] == "--threads").then_some(window[1].as_str()));
        assert_eq!(threads, Some("8"));
    }

    #[test]
    fn extra_args_threads_override_skips_auto_threads() {
        let mut config = Config::default();
        config.server.extra_args = vec!["--threads".into(), "12".into()];
        let actual = args(&config);
        assert_eq!(
            actual
                .iter()
                .filter(|argument| *argument == "--threads")
                .count(),
            1
        );
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--threads" && window[1] == "12")
        );
        assert_eq!(
            &actual[actual.len() - 4..],
            &["--host", "127.0.0.1", "--port", "8080"]
        );
    }

    #[test]
    fn configure_and_network_win_over_extra_runtime_flags() {
        let mut config = Config::default();
        config.network.expose = true;
        config.network.listen_scope = crate::network::ListenScope::Custom;
        config.network.listen_host = "100.64.1.2".into();
        config.network.ensure_token();
        config.sync_llama_bind_from_network();
        let dir = tempfile::tempdir().unwrap();
        let paths = crate::tls::ensure_self_signed(dir.path(), &[]).unwrap();
        config.set_share_tls(Some((paths.cert_file, paths.key_file)));
        config.server.extra_args = vec![
            "--host".into(),
            "0.0.0.0".into(),
            "--port".into(),
            "9999".into(),
            "--api-key".into(),
            "evil".into(),
            "--no-mmap".into(),
            "--flash-attn".into(),
            "off".into(),
            "--threads".into(),
            "4".into(),
        ];
        let actual = args(&config);
        // Share proxy owns the public bind/TLS/keys; llama stays on loopback HTTP.
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--host" && window[1] == "127.0.0.1")
        );
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--port" && window[1] == "8080")
        );
        assert!(!actual.iter().any(|argument| argument == "0.0.0.0"));
        assert!(!actual.iter().any(|argument| argument == "100.64.1.2"));
        assert!(!actual.iter().any(|argument| argument == "evil"));
        assert!(!actual.iter().any(|argument| argument == "9999"));
        assert!(actual.iter().any(|argument| argument == "--mmap"));
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--flash-attn" && window[1] == "on")
        );
        assert!(!actual.iter().any(|argument| *argument == "--api-key"));
        assert!(actual.iter().any(|argument| *argument == "--no-webui"));
        assert!(actual.iter().any(|argument| *argument == "--no-slots"));
        assert!(!actual.iter().any(|argument| *argument == "--metrics"));
        assert!(!actual.iter().any(|argument| *argument == "--ssl-cert-file"));
        assert!(!actual.iter().any(|argument| *argument == "--ssl-key-file"));
        assert!(
            actual
                .windows(2)
                .any(|window| window[0] == "--threads" && window[1] == "4")
        );
    }

    #[test]
    fn process_output_and_exit_are_observed() {
        use std::time::{Duration, Instant};

        let spec = CommandSpec {
            program: std::env::current_exe().unwrap().into_os_string(),
            args: vec!["--help".into()],
        };
        let mut process = ServerProcess::start_spec(&spec).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let status = loop {
            if let Some(status) = process.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "child process did not exit");
            std::thread::sleep(Duration::from_millis(10));
        };
        process.finish_output();
        let logs = process.drain_logs().collect::<Vec<_>>();
        assert!(status.success());
        assert!(!logs.is_empty(), "child output was not captured");
    }

    #[test]
    fn health_requires_http_200() {
        assert!(probe_test_health("200 OK"));
        assert!(!probe_test_health("503 Service Unavailable"));
    }

    #[test]
    fn health_uses_the_configured_port() {
        assert!(probe_test_health("200 OK"));
    }

    #[test]
    fn prometheus_metrics_are_parsed() {
        let body = "\
llamacpp:prompt_tokens_total 128
llamacpp:tokens_predicted_total 42
llamacpp:predicted_tokens_seconds 3.5
llamacpp:requests_processing 1
";
        assert_eq!(
            metric_value(body, "llamacpp:tokens_predicted_total"),
            Some(42.0)
        );
        assert_eq!(
            metric_value(body, "llamacpp:predicted_tokens_seconds"),
            Some(3.5)
        );
        assert_eq!(metric_value(body, "missing"), None);
    }

    #[test]
    fn slots_snapshot_sums_live_decoded_tokens() {
        let body = r#"[
            {"id":0,"is_processing":true,"next_token":{"n_decoded":12}},
            {"id":1,"is_processing":false,"next_token":{"n_decoded":99}},
            {"id":2,"is_processing":true,"next_token":{"n_decoded":3}}
        ]"#;
        assert_eq!(
            parse_slots_snapshot(body),
            Some(SlotsSnapshot {
                decoded_tokens: 15,
                requests_processing: 2,
            })
        );
    }

    #[test]
    fn log_throughput_prefers_recent_window() {
        let line = "slot   print_timings_tg: id  0 | task 3 | n_decoded =    240, tg =  18.50 t/s, tg_3s =  21.25 t/s";
        assert_eq!(
            parse_log_throughput(line),
            LogThroughput {
                generated_tokens_per_second: Some(21.25),
                prompt_tokens_per_second: None,
            }
        );
    }

    #[test]
    fn log_throughput_reads_prompt_and_final_eval() {
        let prompt = "slot print_timings_pp: prompt processing, n_tokens = 512, progress = 0.50, t = 4.00 s / 128.00 tokens per second";
        assert_eq!(
            parse_log_throughput(prompt).prompt_tokens_per_second,
            Some(128.0)
        );
        let eval = "slot print_timings:        eval time =   1000.00 ms /   50 tokens (   20.00 ms per token,    50.00 tokens per second)";
        assert_eq!(
            parse_log_throughput(eval).generated_tokens_per_second,
            Some(50.0)
        );
    }

    fn probe_test_health(status: &str) -> bool {
        probe_test_health_with_extra_args(status, Vec::new())
    }

    fn probe_test_health_with_extra_args(status: &str, mut extra_args: Vec<String>) -> bool {
        use std::io::Read;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let status = status.to_string();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 256];
            let read = stream.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..read]).starts_with("GET /health HTTP/1.1"));
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let mut config = Config::default();
        config.server.host = "127.0.0.1".into();
        if extra_args.iter().any(|argument| argument == "PORT") {
            extra_args = extra_args
                .into_iter()
                .map(|argument| {
                    if argument == "PORT" {
                        port.to_string()
                    } else {
                        argument
                    }
                })
                .collect();
            config.server.port = 1;
        } else {
            config.server.port = port;
        }
        config.server.extra_args = extra_args;
        let healthy = endpoint_healthy(&config);
        server.join().unwrap();
        healthy
    }
}
