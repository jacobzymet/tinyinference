use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::config::Config;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone)]
pub struct Machine {
    pub total_memory_gib: f64,
    pub available_memory_gib: f64,
    pub cpu_name: String,
    pub logical_cpus: usize,
    pub physical_cpus: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Size of the read-only model mapping, not a resident-RAM requirement.
    pub mapped_model_gib: f64,
    pub available_gib: f64,
    pub total_gib: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ProcessUsage {
    pub cpu_percent: f32,
    pub resident_memory_gib: f64,
    pub virtual_memory_gib: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Default)]
pub struct ProcessMonitor {
    system: System,
}

impl ProcessMonitor {
    pub fn refresh(&mut self, process_id: u32) -> Option<ProcessUsage> {
        let pid = Pid::from_u32(process_id);
        self.system
            .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        self.system.process(pid).map(|process| ProcessUsage {
            cpu_percent: process.cpu_usage(),
            resident_memory_gib: process.memory() as f64 / GIB,
            virtual_memory_gib: process.virtual_memory() as f64 / GIB,
            uptime_seconds: process.run_time(),
        })
    }
}

impl Machine {
    pub fn detect() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let logical_cpus = system.cpus().len().max(1);
        let physical_cpus = System::physical_core_count()
            .filter(|count| *count > 0)
            .unwrap_or(logical_cpus);
        let cpu_name = system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().trim().to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Unknown CPU".into());
        Self {
            total_memory_gib: system.total_memory() as f64 / GIB,
            available_memory_gib: system.available_memory() as f64 / GIB,
            logical_cpus,
            physical_cpus,
            cpu_name,
        }
    }

    pub fn memory_profile(&self, config: &Config) -> MemoryProfile {
        let mapped_model_gib = config
            .local_model_size_gib()
            .unwrap_or(config.model.estimated_size_gib);
        MemoryProfile {
            mapped_model_gib,
            available_gib: self.available_memory_gib,
            total_gib: self.total_memory_gib,
        }
    }
}

/// Apple Silicon and similar share one RAM pool for CPU and GPU.
pub fn likely_unified_memory() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Thread count for llama-server: physical cores when known, otherwise logical.
pub fn recommended_threads() -> usize {
    System::physical_core_count()
        .filter(|count| *count > 0)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|count| count.get())
        })
        .unwrap_or(1)
        .max(1)
}

pub fn executable_exists(path: &Path) -> bool {
    if path.components().count() > 1 || path.is_absolute() {
        return is_executable_file(path);
    }
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    let names = {
        let default = vec![path.to_path_buf()];
        #[cfg(windows)]
        {
            if path.extension().is_none() {
                let extensions =
                    std::env::var_os("PATHEXT").unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into());
                extensions
                    .to_string_lossy()
                    .split(';')
                    .filter(|ext| !ext.is_empty())
                    .map(|ext| path.with_extension(ext.trim_start_matches('.')))
                    .collect()
            } else {
                default
            }
        }
        #[cfg(not(windows))]
        {
            default
        }
    };
    std::env::split_paths(&paths)
        .any(|dir| names.iter().any(|name| is_executable_file(&dir.join(name))))
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(windows)]
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    pipe_to_clipboard("clip.exe", &[], text)
}

#[cfg(target_os = "macos")]
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    pipe_to_clipboard("pbcopy", &[], text)
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let candidates: [(&str, &[&str]); 3] = [
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
    ];
    for (program, args) in candidates {
        if pipe_to_clipboard(program, args, text).is_ok() {
            return Ok(());
        }
    }
    bail!("clipboard unavailable; install wl-copy, xclip, or xsel")
}

#[cfg(not(any(windows, unix)))]
pub fn copy_to_clipboard(_text: &str) -> Result<()> {
    bail!("clipboard is not supported on this platform")
}

fn pipe_to_clipboard(program: &str, args: &[&str], text: &str) -> Result<()> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("could not start {program}"))?;
    child
        .stdin
        .as_mut()
        .context("clipboard input was unavailable")?
        .write_all(text.as_bytes())
        .context("could not write to the clipboard")?;
    let status = child.wait().context("clipboard command did not finish")?;
    if !status.success() {
        bail!("{program} exited unsuccessfully");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_model_is_not_treated_as_required_ram() {
        let machine = Machine {
            total_memory_gib: 16.0,
            available_memory_gib: 12.0,
            cpu_name: "test".into(),
            logical_cpus: 8,
            physical_cpus: 4,
        };
        let mut config = Config::default();
        config.model.estimated_size_gib = 500.0;
        let profile = machine.memory_profile(&config);
        assert_eq!(profile.mapped_model_gib, 500.0);
        assert_eq!(profile.available_gib, 12.0);
        assert_eq!(profile.total_gib, 16.0);
    }

    #[test]
    fn recommended_threads_is_at_least_one() {
        assert!(recommended_threads() >= 1);
    }

    #[test]
    fn directories_are_not_reported_as_executables() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!executable_exists(directory.path()));
    }
}
