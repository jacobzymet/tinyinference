use std::{
    io::ErrorKind,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use tinyinference::{
    app::App, config::Config, server::CommandSpec, system::open_in_browser, web,
};
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(
    name = "tinyinference",
    version,
    about = "A minimal control panel for llama.cpp"
)]
struct Cli {
    /// Use a specific TOML configuration file
    #[arg(long, value_name = "PATH")]
    config: Option<std::path::PathBuf>,

    /// Print the resolved llama-server command and exit
    #[arg(long)]
    print_command: bool,

    /// Launch llama-server immediately when the control panel starts
    #[arg(long)]
    start: bool,

    /// Address for the tinyinference web server (overrides config/env)
    #[arg(long, value_name = "ADDR")]
    bind: Option<SocketAddr>,

    /// Serve the control panel in the browser instead of a native window
    #[arg(long)]
    no_window: bool,

    /// Open the control panel in the default browser (implies --no-window)
    #[arg(long)]
    open: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let use_window = windowed(&cli);
    let open_in_default_browser = cli.open;
    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let config = Config::load(&config_path)?;

    if cli.print_command {
        let mut config = config;
        config.migrate_network_expose_to_llama();
        if config.network.expose {
            let dir = config_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            if let Ok(paths) = tinyinference::tls::ensure_self_signed(dir, &[]) {
                config.set_share_tls(Some((paths.cert_file, paths.key_file)));
            }
        }
        println!("{}", CommandSpec::from_config(&config).display());
        return Ok(());
    }

    let bind = Config::resolve_ui_bind(cli.bind, &config)?;
    let url = format!("http://{bind}");

    // The desktop event loop must own the main thread, so the async runtime is
    // built by hand and the server runs on it in the background.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("could not start the async runtime")?;

    // Claim the port before doing any work. The port doubles as the
    // single-instance lock: if it is already held by another tinyinference,
    // hand the user back to that one instead of racing it.
    let listener = match runtime.block_on(TcpListener::bind(bind)) {
        Ok(listener) => listener,
        Err(error) if error.kind() == ErrorKind::AddrInUse => {
            return greet_running_instance(&url, bind);
        }
        Err(error) => return Err(error).with_context(|| format!("could not bind {bind}")),
    };

    let mut app = App::new(config, config_path);
    app.set_listen_addr(bind);
    if cli.start {
        app.start();
    }

    let shared = Arc::new(Mutex::new(app));
    let server = {
        let shared = Arc::clone(&shared);
        runtime.spawn(async move { web::serve(shared, listener).await })
    };

    if use_window {
        println!("tinyinference running on {url}");
        return run_window(&url, Arc::clone(&shared));
    }

    println!("tinyinference listening on {url}");
    if open_in_default_browser {
        let _ = open_in_browser(&url);
    }
    let result = runtime.block_on(async { server.await? });
    if let Ok(mut app) = shared.lock() {
        app.shutdown();
    }
    result
}

/// How long to wait for a running instance to answer `/api/focus`.
const FOCUS_TIMEOUT: Duration = Duration::from_secs(5);

/// The address is taken. If tinyinference is what holds it, bring that instance
/// forward and exit quietly; otherwise report the clash.
fn greet_running_instance(url: &str, bind: SocketAddr) -> Result<()> {
    match focus_running_instance(url) {
        Some(true) => println!("tinyinference is already running — raised its window."),
        Some(false) => println!("tinyinference is already running on {url}"),
        None => bail!(
            "{bind} is already in use by another program — pass --bind to choose a different address"
        ),
    }
    Ok(())
}

/// Ask whoever holds the port to identify itself and come forward.
///
/// `Some(focused)` means it was tinyinference; the flag says whether it had a
/// window to raise. `None` means the port belongs to something else.
fn focus_running_instance(url: &str) -> Option<bool> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(FOCUS_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();
    let mut response = agent.post(format!("{url}/api/focus")).send_empty().ok()?;
    if response.status() != 200 {
        return None;
    }
    let body = response.body_mut().read_to_string().ok()?;
    let info: serde_json::Value = serde_json::from_str(&body).ok()?;
    if info.get("app").and_then(|app| app.as_str()) != Some(web::INSTANCE_MARKER) {
        return None;
    }
    Some(
        info.get("focused")
            .and_then(|focused| focused.as_bool())
            .unwrap_or(false),
    )
}

/// Whether to host the control panel in a native window.
///
/// `--open` hands the panel to the browser, so it implies `--no-window`.
#[cfg(feature = "desktop")]
fn windowed(cli: &Cli) -> bool {
    !cli.no_window && !cli.open
}

#[cfg(not(feature = "desktop"))]
fn windowed(_cli: &Cli) -> bool {
    false
}

#[cfg(feature = "desktop")]
fn run_window(url: &str, shared: Arc<Mutex<App>>) -> Result<()> {
    tinyinference::desktop::run(url, shared)
}

#[cfg(not(feature = "desktop"))]
fn run_window(_url: &str, _shared: Arc<Mutex<App>>) -> Result<()> {
    unreachable!("the desktop feature is disabled")
}
