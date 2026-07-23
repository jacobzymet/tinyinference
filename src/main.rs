use std::{path::PathBuf, time::Duration};

use anyhow::Result;
use clap::Parser;
use ratatui::crossterm::event::{self, Event};
use tinyinference::{app::App, config::Config, server::CommandSpec, ui};

#[derive(Debug, Parser)]
#[command(
    name = "tinyinference",
    version,
    about = "A minimal terminal control plane for llama.cpp"
)]
struct Cli {
    /// Use a specific TOML configuration file
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Print the resolved llama-server command and exit
    #[arg(long)]
    print_command: bool,

    /// Launch llama-server immediately when the interface opens
    #[arg(long)]
    start: bool,
}

fn main() -> Result<()> {
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

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app);
    app.shutdown();
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<()> {
    while !app.should_quit {
        app.tick();
        terminal.draw(|frame| ui::render(frame, app))?;
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            && key.kind == event::KeyEventKind::Press
        {
            app.handle_key(key);
        }
    }
    Ok(())
}
