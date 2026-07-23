use std::{
    ffi::OsString,
    io::{BufRead, BufReader, Write},
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, bail};

use crate::config::{Config, ModelSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
}

impl CommandSpec {
    pub fn from_config(config: &Config) -> Self {
        let mut args = Vec::<OsString>::new();
        match &config.model.source {
            ModelSource::HuggingFace(id) => push_pair(&mut args, "-hf", id.as_str()),
            ModelSource::Local(path) => {
                args.push("-m".into());
                args.push(path.as_os_str().into());
            }
        }

        if config.runtime.cpu_only {
            push_pair(&mut args, "--device", "none");
            push_pair(&mut args, "--n-gpu-layers", "0");
        } else {
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
        push_pair(&mut args, "--host", config.server.host.as_str());
        push_pair(&mut args, "--port", config.server.port.to_string());
        args.extend(config.server.extra_args.iter().map(OsString::from));

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
    let timeout = Duration::from_millis(250);
    let host = config.server.host.as_str();
    let address = host
        .parse::<IpAddr>()
        .ok()
        .map(|ip| SocketAddr::new(ip, config.server.port))
        .or_else(|| (host, config.server.port).to_socket_addrs().ok()?.next());
    let Some(mut stream) =
        address.and_then(|address| TcpStream::connect_timeout(&address, timeout).ok())
    else {
        return false;
    };
    if stream.set_read_timeout(Some(timeout)).is_err()
        || stream.set_write_timeout(Some(timeout)).is_err()
    {
        return false;
    }
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        config.server.host, config.server.port
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut status_line = String::new();
    let Ok(read) = BufReader::new(&mut stream).read_line(&mut status_line) else {
        return false;
    };
    read > 0
        && (status_line.starts_with("HTTP/1.1 200 ") || status_line.starts_with("HTTP/1.0 200 "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(config: &Config) -> Vec<String> {
        CommandSpec::from_config(config)
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
                "--no-warmup",
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
    fn extra_args_are_last_so_advanced_users_can_override() {
        let mut config = Config::default();
        config.server.extra_args = vec!["--threads".into(), "12".into()];
        let actual = args(&config);
        assert_eq!(&actual[actual.len() - 2..], &["--threads", "12"]);
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

    fn probe_test_health(status: &str) -> bool {
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
        config.server.port = port;
        let healthy = endpoint_healthy(&config);
        server.join().unwrap();
        healthy
    }
}
