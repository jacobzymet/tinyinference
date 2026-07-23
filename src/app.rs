use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::{
    config::{Config, DEFAULT_MODEL, ModelSource},
    server::{
        CommandSpec, ServerEvent, ServerMetrics, ServerProcess, endpoint_healthy, fetch_metrics,
    },
    system::{Machine, ProcessMonitor, ProcessUsage, copy_to_clipboard, executable_exists},
};

const MAX_LOG_LINES: usize = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Dashboard,
    Configure,
    Logs,
    Stats,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Ready,
    Stopping,
    Failed,
}

impl ServerStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Stopping => "stopping",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    CacheRam,
    Checkpoints,
    Mmproj,
    Jinja,
}

impl SettingField {
    pub const ALL: [Self; 19] = [
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
                | Self::Host
                | Self::Port
                | Self::Context
                | Self::Batch
                | Self::MicroBatch
                | Self::Parallel
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
                | Self::Mmproj
                | Self::Jinja
        )
    }
}

#[derive(Debug, Clone)]
pub struct Editor {
    pub field: SettingField,
    pub value: String,
    pub cursor: usize,
}

#[derive(Debug)]
pub struct App {
    pub config: Config,
    pub config_path: PathBuf,
    pub machine: Machine,
    pub view: View,
    pub status: ServerStatus,
    pub status_detail: String,
    pub process: Option<ServerProcess>,
    running_config: Option<Config>,
    pub logs: VecDeque<String>,
    pub log_offset: usize,
    pub setting_index: usize,
    pub editor: Option<Editor>,
    pub editor_error: Option<String>,
    last_hf_model: String,
    last_local_model: PathBuf,
    pub should_quit: bool,
    pub endpoint_online: bool,
    pub startup_frame: usize,
    pub process_usage: Option<ProcessUsage>,
    pub server_metrics: Option<ServerMetrics>,
    process_monitor: ProcessMonitor,
    missing_server_prompt: bool,
    last_probe: Instant,
    last_stats_refresh: Instant,
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
        Self {
            config,
            config_path,
            machine: Machine::detect(),
            view: View::Dashboard,
            status: ServerStatus::Stopped,
            status_detail,
            process: None,
            running_config: None,
            logs: VecDeque::from(["tinyinference initialized".into()]),
            log_offset: 0,
            setting_index: 0,
            editor: None,
            editor_error: None,
            last_hf_model,
            last_local_model,
            should_quit: false,
            endpoint_online: false,
            startup_frame: 0,
            process_usage: None,
            server_metrics: None,
            process_monitor: ProcessMonitor::default(),
            missing_server_prompt: !executable_found,
            last_probe: Instant::now() - Duration::from_secs(2),
            last_stats_refresh: Instant::now() - Duration::from_secs(2),
        }
    }

    pub fn command(&self) -> CommandSpec {
        CommandSpec::from_config(&self.config)
    }

    pub fn displayed_config(&self) -> &Config {
        self.running_config.as_ref().unwrap_or(&self.config)
    }

    pub fn has_pending_changes(&self) -> bool {
        self.running_config
            .as_ref()
            .is_some_and(|running| running != &self.config)
    }

    pub fn should_prompt_for_server(&self) -> bool {
        self.missing_server_prompt
    }

    pub fn dismiss_server_prompt(&mut self) {
        self.missing_server_prompt = false;
    }

    pub fn selected_field(&self) -> SettingField {
        SettingField::ALL[self.setting_index]
    }

    pub fn setting_value(&self, field: SettingField) -> String {
        match field {
            SettingField::SourceKind => match self.config.model.source {
                ModelSource::HuggingFace(_) => "Hugging Face".into(),
                ModelSource::Local(_) => "Local GGUF".into(),
            },
            SettingField::Model => match &self.config.model.source {
                ModelSource::HuggingFace(id) if id.trim().is_empty() => {
                    "<enter owner/model>".into()
                }
                ModelSource::Local(path) if path.as_os_str().is_empty() => {
                    "<enter path to .gguf>".into()
                }
                _ => self.config.model_label(),
            },
            SettingField::EstimatedSize => match &self.config.model.source {
                ModelSource::Local(_) => self
                    .config
                    .local_model_size_gib()
                    .map(|size| format!("{size:.1} GiB  (from file)"))
                    .unwrap_or_else(|| "auto from .gguf".into()),
                ModelSource::HuggingFace(_) => {
                    format!("{:.1} GiB", self.config.model.estimated_size_gib)
                }
            },
            SettingField::Executable => {
                let path = &self.config.server.executable;
                if path.components().count() == 1 && !path.is_absolute() {
                    format!("{}  (from PATH)", path.display())
                } else {
                    path.display().to_string()
                }
            }
            SettingField::Host => self.config.server.host.clone(),
            SettingField::Port => self.config.server.port.to_string(),
            SettingField::Context => format!("{} tokens", self.config.runtime.context_size),
            SettingField::Batch => self.config.runtime.batch_size.to_string(),
            SettingField::MicroBatch => self.config.runtime.micro_batch_size.to_string(),
            SettingField::Parallel => self.config.runtime.parallel.to_string(),
            SettingField::CpuOnly => on_off(self.config.runtime.cpu_only),
            SettingField::Mmap => on_off(self.config.runtime.mmap),
            SettingField::Fit => on_off(self.config.runtime.fit),
            SettingField::Repack => on_off(self.config.runtime.repack),
            SettingField::Warmup => on_off(self.config.runtime.warmup),
            SettingField::CacheRam => format!("{} MiB", self.config.runtime.cache_ram_mib),
            SettingField::Checkpoints => self.config.runtime.context_checkpoints.to_string(),
            SettingField::Mmproj => on_off(self.config.runtime.multimodal_projector),
            SettingField::Jinja => on_off(self.config.runtime.jinja),
        }
    }

    pub fn setting_label(&self, field: SettingField) -> &'static str {
        match (field, &self.config.model.source) {
            (SettingField::Model, ModelSource::HuggingFace(_)) => "Model repository",
            (SettingField::Model, ModelSource::Local(_)) => "GGUF file path",
            (SettingField::EstimatedSize, ModelSource::Local(_)) => "Mapped file size",
            (SettingField::EstimatedSize, ModelSource::HuggingFace(_)) => "Estimated file size",
            (SettingField::Executable, _) => "llama-server path",
            _ => field.label(),
        }
    }

    pub fn setting_is_editable(&self, field: SettingField) -> bool {
        field.is_editable()
            && !matches!(
                (field, &self.config.model.source),
                (SettingField::EstimatedSize, ModelSource::Local(_))
            )
    }

    pub fn setting_hint(&self, field: SettingField) -> &'static str {
        match (field, &self.config.model.source) {
            (SettingField::SourceKind, _) => {
                "Use \u{2190}\u{2192} to switch source; then edit the field below."
            }
            (SettingField::Model, ModelSource::HuggingFace(_)) => {
                "Enter owner/model, or press r to cycle recent models."
            }
            (SettingField::Model, ModelSource::Local(_)) => {
                "Enter a full .gguf path, or press r to cycle recent models."
            }
            (SettingField::EstimatedSize, ModelSource::Local(_)) => {
                "Calculated automatically from the GGUF file."
            }
            (SettingField::EstimatedSize, ModelSource::HuggingFace(_)) => {
                "Display-only estimate for the remote model mapping."
            }
            (SettingField::Executable, _) => {
                "Enter llama-server if it is on PATH, or enter its full executable path."
            }
            (SettingField::Host, _) => "127.0.0.1 keeps the server local to this machine.",
            (SettingField::Port, _) => "Press Enter for an exact port; arrows adjust by one.",
            (SettingField::Context, _) => "Press Enter for an exact token count; 8k is accepted.",
            (SettingField::Batch, _) => "Prompt batch size. Enter an exact value or use arrows.",
            (SettingField::MicroBatch, _) => "Must be no larger than the batch size.",
            (SettingField::Parallel, _) => "Number of simultaneous server slots.",
            (SettingField::CpuOnly, _) => "On forces all model layers onto the CPU.",
            (SettingField::Mmap, _) => "On keeps model weights file-backed and demand-paged.",
            (SettingField::Fit, _) => "Off prevents llama.cpp from changing the requested profile.",
            (SettingField::Repack, _) => "Off avoids a separate repacked weight copy.",
            (SettingField::Warmup, _) => {
                "Off avoids touching model pages before the first request."
            }
            (SettingField::CacheRam, _) => "Host prompt-cache limit in MiB; zero disables it.",
            (SettingField::Checkpoints, _) => "Saved context states per slot; zero disables them.",
            (SettingField::Mmproj, _) => "Leave off for text-only models.",
            (SettingField::Jinja, _) => "Uses the model's Jinja chat template.",
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }
        if self.missing_server_prompt {
            match key.code {
                KeyCode::Enter | KeyCode::Char('c') => self.open_server_configuration(),
                KeyCode::Esc => self.dismiss_server_prompt(),
                KeyCode::Char('q') => self.should_quit = true,
                _ => {}
            }
            return;
        }
        if self.editor.is_some() && key.code == KeyCode::Enter {
            self.commit_editor();
            return;
        }
        if let Some(editor) = self.editor.as_mut() {
            match key.code {
                KeyCode::Esc => {
                    self.editor = None;
                    self.editor_error = None;
                }
                KeyCode::Backspace => {
                    self.editor_error = None;
                    if editor.cursor > 0 {
                        let previous = previous_char_boundary(&editor.value, editor.cursor);
                        editor.value.drain(previous..editor.cursor);
                        editor.cursor = previous;
                    }
                }
                KeyCode::Delete => {
                    self.editor_error = None;
                    if editor.cursor < editor.value.len() {
                        let next = next_char_boundary(&editor.value, editor.cursor);
                        editor.value.drain(editor.cursor..next);
                    }
                }
                KeyCode::Left => {
                    editor.cursor = previous_char_boundary(&editor.value, editor.cursor);
                }
                KeyCode::Right => {
                    editor.cursor = next_char_boundary(&editor.value, editor.cursor);
                }
                KeyCode::Home => editor.cursor = 0,
                KeyCode::End => editor.cursor = editor.value.len(),
                KeyCode::Char(c)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.editor_error = None;
                    editor.value.insert(editor.cursor, c);
                    editor.cursor += c.len_utf8();
                }
                _ => {}
            }
            return;
        }

        match self.view {
            View::Dashboard => self.handle_dashboard_key(key.code),
            View::Configure => self.handle_config_key(key.code),
            View::Logs => self.handle_logs_key(key.code),
            View::Stats => self.handle_stats_key(key.code),
            View::Help => {
                if matches!(
                    key.code,
                    KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q')
                ) {
                    self.view = View::Dashboard;
                }
            }
        }
    }

    fn handle_dashboard_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('s') => {
                if self.process.is_some() {
                    self.stop();
                } else {
                    self.start();
                }
            }
            KeyCode::Char('r') => self.restart(),
            KeyCode::Char('c') => self.view = View::Configure,
            KeyCode::Char('l') => self.view = View::Logs,
            KeyCode::Char('t') => {
                self.view = View::Stats;
                self.last_stats_refresh = Instant::now() - Duration::from_secs(2);
            }
            KeyCode::Char('y') => self.copy_endpoint(),
            KeyCode::Char('Y') => self.copy_command(),
            KeyCode::Char('?') => self.view = View::Help,
            _ => {}
        }
    }

    fn open_server_configuration(&mut self) {
        self.missing_server_prompt = false;
        self.view = View::Configure;
        self.setting_index = SettingField::ALL
            .iter()
            .position(|field| *field == SettingField::Executable)
            .unwrap_or(0);
    }

    fn handle_config_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc | KeyCode::Char('c') => self.view = View::Dashboard,
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => {
                self.setting_index = self.setting_index.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.setting_index = (self.setting_index + 1).min(SettingField::ALL.len() - 1);
            }
            KeyCode::Left | KeyCode::Char('h') => self.adjust_selected(-1),
            KeyCode::Right | KeyCode::Char('l') => self.adjust_selected(1),
            KeyCode::Char(' ') if self.selected_field().is_toggle() => self.adjust_selected(1),
            KeyCode::Enter => {
                if self.setting_is_editable(self.selected_field()) {
                    self.begin_edit();
                } else {
                    self.adjust_selected(1);
                }
            }
            KeyCode::Char('s') => self.save(),
            KeyCode::Char('r') if self.selected_field() == SettingField::Model => {
                self.select_next_recent_model();
            }
            _ => {}
        }
    }

    fn handle_logs_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc | KeyCode::Char('l') => self.view = View::Dashboard,
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => {
                self.log_offset = (self.log_offset + 1).min(self.logs.len().saturating_sub(1));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.log_offset = self.log_offset.saturating_sub(1);
            }
            KeyCode::PageUp => {
                self.log_offset = (self.log_offset + 10).min(self.logs.len().saturating_sub(1));
            }
            KeyCode::PageDown => self.log_offset = self.log_offset.saturating_sub(10),
            KeyCode::Home => self.log_offset = self.logs.len().saturating_sub(1),
            KeyCode::End => self.log_offset = 0,
            _ => {}
        }
    }

    fn handle_stats_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc | KeyCode::Char('t') => self.view = View::Dashboard,
            KeyCode::Char('y') => self.copy_endpoint(),
            KeyCode::Char('Y') => self.copy_command(),
            KeyCode::Char('q') => self.should_quit = true,
            _ => {}
        }
    }

    pub fn tick(&mut self) {
        if self.status == ServerStatus::Starting {
            self.startup_frame = self.startup_frame.wrapping_add(1);
        }

        let new_logs = self
            .process
            .as_ref()
            .map(|process| process.drain_logs().collect::<Vec<_>>())
            .unwrap_or_default();
        for event in new_logs {
            let ServerEvent::Log(line) = event;
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
                self.process_usage = None;
                self.server_metrics = None;
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

        if self.process.is_some() && self.last_probe.elapsed() >= Duration::from_secs(1) {
            self.endpoint_online = self.running_config.as_ref().is_some_and(endpoint_healthy);
            self.last_probe = Instant::now();
            self.status = if self.endpoint_online {
                ServerStatus::Ready
            } else {
                ServerStatus::Starting
            };
            if self.endpoint_online {
                self.startup_frame = 0;
                if let Some(config) = &self.running_config {
                    self.status_detail = format!("Listening at {}", config.endpoint());
                }
            }
        }

        if self.view == View::Stats
            && self.process.is_some()
            && self.last_stats_refresh.elapsed() >= Duration::from_secs(1)
        {
            let process_id = self.process.as_ref().map(ServerProcess::id);
            self.process_usage = process_id.and_then(|pid| self.process_monitor.refresh(pid));
            if !self.endpoint_online {
                self.server_metrics = None;
            } else if let Some(metrics) = self.running_config.as_ref().and_then(fetch_metrics) {
                self.server_metrics = Some(metrics);
            }
            self.last_stats_refresh = Instant::now();
        }
    }

    pub fn start(&mut self) {
        if self.process.is_some() {
            return;
        }
        if let Err(errors) = self.validate_for_launch() {
            self.status = ServerStatus::Failed;
            self.status_detail = errors;
            return;
        }
        let launch_config = self.config.clone();
        let display = CommandSpec::from_config(&launch_config).display();
        self.push_log(format!("$ {display}"));
        match ServerProcess::start(&launch_config) {
            Ok(process) => {
                let pid = process.id();
                self.process = Some(process);
                self.running_config = Some(launch_config);
                self.status = ServerStatus::Starting;
                self.status_detail = format!("Waking llama-server (PID {pid})");
                self.endpoint_online = false;
                self.startup_frame = 0;
                self.last_probe = Instant::now() - Duration::from_secs(2);
                self.last_stats_refresh = Instant::now() - Duration::from_secs(2);
            }
            Err(error) => {
                self.status = ServerStatus::Failed;
                self.status_detail = error.to_string();
                self.push_log(format!("[launch failed] {error:#}"));
            }
        }
    }

    pub fn stop(&mut self) {
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
        self.process_usage = None;
        self.server_metrics = None;
    }

    pub fn restart(&mut self) {
        if self.process.is_some() {
            self.stop();
        }
        self.start();
    }

    pub fn shutdown(&mut self) {
        if self.process.is_some() {
            self.stop();
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

    fn save(&mut self) {
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

    fn begin_edit(&mut self) {
        let field = self.selected_field();
        if !self.setting_is_editable(field) {
            return;
        }
        let value = match field {
            SettingField::Model => self.config.model_label(),
            SettingField::EstimatedSize => self.config.model.estimated_size_gib.to_string(),
            SettingField::Executable => self.config.server.executable.display().to_string(),
            SettingField::Host => self.config.server.host.clone(),
            SettingField::Port => self.config.server.port.to_string(),
            SettingField::Context => self.config.runtime.context_size.to_string(),
            SettingField::Batch => self.config.runtime.batch_size.to_string(),
            SettingField::MicroBatch => self.config.runtime.micro_batch_size.to_string(),
            SettingField::Parallel => self.config.runtime.parallel.to_string(),
            SettingField::CacheRam => self.config.runtime.cache_ram_mib.to_string(),
            SettingField::Checkpoints => self.config.runtime.context_checkpoints.to_string(),
            _ => return,
        };
        self.editor_error = None;
        self.editor = Some(Editor {
            cursor: value.len(),
            field,
            value,
        });
    }

    fn commit_editor(&mut self) {
        let Some(editor) = self.editor.clone() else {
            return;
        };
        match self.apply_editor_value(editor.field, editor.value.trim()) {
            Ok(()) => {
                if editor.field == SettingField::Model {
                    self.remember_current_model();
                }
                self.editor = None;
                self.editor_error = None;
                self.mark_setting_changed(editor.field);
            }
            Err(error) => {
                self.editor_error = Some(error);
            }
        }
    }

    fn apply_editor_value(
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
                let size = value
                    .parse::<f64>()
                    .map_err(|_| "Enter a size in GiB, such as 59.1.".to_string())?;
                if !size.is_finite() || !(0.1..=100_000.0).contains(&size) {
                    return Err("File size must be between 0.1 and 100000 GiB.".into());
                }
                self.config.model.estimated_size_gib = size;
            }
            SettingField::Executable => {
                if value.is_empty() {
                    return Err("Enter llama-server or its full executable path.".into());
                }
                self.config.server.executable = value.into();
            }
            SettingField::Host => {
                if value.is_empty() {
                    return Err("Listen address cannot be empty.".into());
                }
                self.config.server.host = value.into();
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
            _ => return Err("This setting is changed with the arrow keys.".into()),
        }
        Ok(())
    }

    fn adjust_selected(&mut self, direction: i32) {
        let field = self.selected_field();
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
                if matches!(&self.config.model.source, ModelSource::Local(_)) {
                    return;
                }
                self.config.model.estimated_size_gib = adjust_f64(
                    self.config.model.estimated_size_gib,
                    direction as f64,
                    0.1,
                    999.0,
                );
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
        self.mark_setting_changed(field);
    }

    fn mark_setting_changed(&mut self, field: SettingField) {
        self.status_detail = if self.process.is_some() {
            format!("Changed {}; restart to apply", self.setting_label(field))
        } else {
            format!("Changed {}; press s to save", self.setting_label(field))
        };
    }

    fn remember_current_model(&mut self) {
        let source = self.config.model.source.clone();
        self.config.remember_model(source);
    }

    fn select_next_recent_model(&mut self) {
        if self.config.recent_models.len() < 2 {
            self.status_detail = "No other recent models yet".into();
            return;
        }
        let current = &self.config.model.source;
        let next_index = self
            .config
            .recent_models
            .iter()
            .position(|recent| recent == current)
            .map(|index| (index + 1) % self.config.recent_models.len())
            .unwrap_or(0);
        let selected = self.config.recent_models[next_index].clone();
        match &selected {
            ModelSource::HuggingFace(id) => self.last_hf_model = id.clone(),
            ModelSource::Local(path) => self.last_local_model = path.clone(),
        }
        self.config.model.source = selected;
        self.mark_setting_changed(SettingField::Model);
    }

    fn copy_endpoint(&mut self) {
        let endpoint = self.displayed_config().api_endpoint();
        self.copy_text("API endpoint", &endpoint);
    }

    fn copy_command(&mut self) {
        let command = CommandSpec::from_config(self.displayed_config()).display();
        self.copy_text("launch command", &command);
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

fn on_off(value: bool) -> String {
    if value { "on" } else { "off" }.into()
}

fn adjust_u32(value: u32, delta: i32, min: u32, max: u32) -> u32 {
    if delta >= 0 {
        value.saturating_add(delta as u32).min(max)
    } else {
        value.saturating_sub(delta.unsigned_abs()).max(min)
    }
}

fn adjust_f64(value: f64, delta: f64, min: f64, max: f64) -> f64 {
    (value + delta).clamp(min, max)
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

fn previous_char_boundary(value: &str, cursor: usize) -> usize {
    value[..cursor]
        .char_indices()
        .next_back()
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn next_char_boundary(value: &str, cursor: usize) -> usize {
    value[cursor..]
        .char_indices()
        .nth(1)
        .map(|(index, _)| cursor + index)
        .unwrap_or(value.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_adjustment_is_bounded() {
        assert_eq!(adjust_power_of_two(8, true, 1, 16), 16);
        assert_eq!(adjust_power_of_two(16, true, 1, 16), 16);
        assert_eq!(adjust_power_of_two(1, false, 1, 16), 1);
    }

    #[test]
    fn text_cursor_moves_across_unicode_boundaries() {
        let value = "aλb";
        assert_eq!(next_char_boundary(value, 1), 3);
        assert_eq!(previous_char_boundary(value, 3), 1);
    }

    #[test]
    fn app_starts_with_dashboard() {
        let app = App::new(Config::default(), "test.toml".into());
        assert_eq!(app.view, View::Dashboard);
        assert_eq!(app.status, ServerStatus::Stopped);
    }

    #[test]
    fn recent_model_shortcut_cycles_saved_models() {
        let mut app = App::new(Config::default(), "test.toml".into());
        let current = app.config.model.source.clone();
        let local = ModelSource::Local("models/small.gguf".into());
        app.config.recent_models = vec![current, local.clone()];
        app.select_next_recent_model();
        assert_eq!(app.config.model.source, local);
    }

    #[test]
    fn missing_server_prompts_and_opens_the_right_setting() {
        let mut config = Config::default();
        config.server.executable = "__tinyinference_missing_server__".into();
        let mut app = App::new(config, "test.toml".into());
        assert!(app.should_prompt_for_server());
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.view, View::Configure);
        assert_eq!(app.selected_field(), SettingField::Executable);
        assert!(!app.should_prompt_for_server());
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
    fn control_c_quits_while_editor_is_open() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.setting_index = SettingField::ALL
            .iter()
            .position(|field| *field == SettingField::Port)
            .unwrap();
        app.begin_edit();
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit);
    }

    #[test]
    fn exact_port_can_be_entered() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.dismiss_server_prompt();
        app.view = View::Configure;
        app.setting_index = SettingField::ALL
            .iter()
            .position(|field| *field == SettingField::Port)
            .unwrap();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let editor = app.editor.as_mut().unwrap();
        editor.value = "4242".into();
        editor.cursor = editor.value.len();
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.config.server.port, 4242);
        assert!(app.editor.is_none());
    }

    #[test]
    fn context_accepts_k_suffix() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.apply_editor_value(SettingField::Context, "16k")
            .unwrap();
        assert_eq!(app.config.runtime.context_size, 16 * 1024);
    }

    #[test]
    fn invalid_editor_value_keeps_editor_open() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.setting_index = SettingField::ALL
            .iter()
            .position(|field| *field == SettingField::Port)
            .unwrap();
        app.begin_edit();
        app.editor.as_mut().unwrap().value = "70000".into();
        app.commit_editor();
        assert!(app.editor.is_some());
        assert!(app.editor_error.as_deref().unwrap().contains("between"));
        assert_eq!(app.config.server.port, 8080);
    }

    #[test]
    fn switching_to_local_model_requests_a_real_path() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.adjust_selected(1);
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
    fn model_source_switch_preserves_both_values() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.adjust_selected(1);
        app.apply_editor_value(SettingField::Model, r"C:\models\custom.gguf")
            .unwrap();
        app.adjust_selected(1);
        assert_eq!(
            app.setting_value(SettingField::Model),
            "ggml-org/gpt-oss-120b-GGUF"
        );
        app.adjust_selected(1);
        assert_eq!(
            app.setting_value(SettingField::Model),
            r"C:\models\custom.gguf"
        );
    }

    #[test]
    fn quoted_executable_path_is_unwrapped() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.apply_editor_value(
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
