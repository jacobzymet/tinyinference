use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use clap::Parser;
use tinyinference::{app::App, config::Config, server::CommandSpec, web};

#[derive(Debug, Parser)]
#[command(
    name = "tinyinference",
    version,
    about = "A minimal web control plane for llama.cpp"
)]
struct Cli {
    /// Use a specific TOML configuration file
    #[arg(long, value_name = "PATH")]
    config: Option<std::path::PathBuf>,

    /// Print the resolved llama-server command and exit
    #[arg(long)]
    print_command: bool,

    /// Launch llama-server immediately when the UI starts
    #[arg(long)]
    start: bool,

    /// Address for the tinyinference web UI
    #[arg(long, default_value = "127.0.0.1:3920")]
    bind: SocketAddr,

    /// Open the web UI in the default browser
    #[arg(long)]
    open: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let config = Config::load(&config_path)?;

    if cli.print_command {
        println!("{}", CommandSpec::from_config(&config).display());
        return Ok(());
    }

    let mut app = App::new(config, config_path);
    if cli.start {
        app.start();
    }

    let shared = Arc::new(Mutex::new(app));
    let url = format!("http://{}", cli.bind);
    println!("tinyinference listening on {url}");

    if cli.open {
        let _ = open_browser(&url);
    }

    let result = web::serve(Arc::clone(&shared), cli.bind).await;
    if let Ok(mut app) = shared.lock() {
        app.shutdown();
    }
    result
}

fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }
    Ok(())
}
