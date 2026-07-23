use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Row, Table, Wrap},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{
    app::{App, ServerStatus, SettingField, View},
    config::Config,
};

const CANVAS: Color = Color::Rgb(11, 16, 20);
const SURFACE: Color = Color::Rgb(21, 30, 35);
const INK: Color = Color::Rgb(220, 228, 226);
const MUTED: Color = Color::Rgb(120, 133, 139);
const ICE: Color = Color::Rgb(132, 184, 208);
const CORAL: Color = Color::Rgb(215, 131, 116);
const MINT: Color = Color::Rgb(137, 191, 164);

pub fn render(frame: &mut Frame, app: &mut App) {
    frame.render_widget(
        Block::default().style(Style::default().bg(CANVAS)),
        frame.area(),
    );
    let area = frame.area().inner(Margin {
        horizontal: if frame.area().width > 72 { 3 } else { 1 },
        vertical: 1,
    });
    match app.view {
        View::Dashboard => dashboard(frame, area, app),
        View::Configure => configure(frame, area, app),
        View::Logs => logs(frame, area, app),
        View::Help => {
            dashboard(frame, area, app);
            help_overlay(frame);
        }
    }
    if app.should_prompt_for_server() {
        missing_server_overlay(frame);
    }
    if app.editor.is_some() {
        editor_overlay(frame, app);
    }
}

fn missing_server_overlay(frame: &mut Frame) {
    let area = centered_rect(60, 34, frame.area());
    frame.render_widget(Clear, area);
    let block = dialog(" llama-server not found ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "Set the executable path before starting inference.",
                Style::default().fg(INK),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("enter", Style::default().fg(ICE).bold()),
                Span::styled(" configure  \u{00b7}  ", Style::default().fg(MUTED)),
                Span::styled("esc", Style::default().fg(ICE).bold()),
                Span::styled(" later  \u{00b7}  ", Style::default().fg(MUTED)),
                Span::styled("q", Style::default().fg(ICE).bold()),
                Span::styled(" quit", Style::default().fg(MUTED)),
            ]),
        ]),
        inner,
    );
}

fn dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let config = app.displayed_config();
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .split(area);

    header(frame, rows[0], app);
    let (notice, notice_style) = if app.has_pending_changes() {
        (
            "Settings changed \u{00b7} restart to apply",
            Style::default().fg(ICE),
        )
    } else {
        (app.status_detail.as_str(), status_detail_style(app.status))
    };
    frame.render_widget(Paragraph::new(notice).style(notice_style), rows[1]);

    frame.render_widget(
        Paragraph::new(config.model_label())
            .style(Style::default().fg(INK).add_modifier(Modifier::BOLD))
            .wrap(Wrap { trim: true }),
        rows[2],
    );

    let memory = app.machine.memory_profile(config);
    info_line(
        frame,
        rows[3],
        "storage",
        format!(
            "{:.1} GiB mapped from disk · not required to fit in RAM",
            memory.mapped_model_gib
        ),
        ICE,
    );
    info_line(
        frame,
        rows[4],
        "ram",
        format!(
            "{:.1} GiB available · {:.1} GiB total",
            memory.available_gib, memory.total_gib
        ),
        INK,
    );
    info_line(
        frame,
        rows[5],
        "runtime",
        format!(
            "{} · {} context · batch {} · {} {}",
            if config.runtime.cpu_only {
                "CPU"
            } else {
                "GPU offload"
            },
            format_tokens(config.runtime.context_size),
            config.runtime.batch_size,
            config.runtime.parallel,
            plural(config.runtime.parallel, "slot", "slots"),
        ),
        INK,
    );
    info_line(frame, rows[6], "access", access_summary(config), MUTED);
    info_line(
        frame,
        rows[7],
        "endpoint",
        config.endpoint(),
        if app.endpoint_online { MINT } else { MUTED },
    );

    footer(
        frame,
        rows[9],
        &[
            (
                "s",
                if app.process.is_some() {
                    "stop"
                } else {
                    "start"
                },
            ),
            ("c", "configure"),
            ("l", "logs"),
            ("?", "help"),
            ("q", "quit"),
        ],
    );
}

fn header(frame: &mut Frame, area: Rect, app: &App) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("tiny", Style::default().fg(INK).bold()),
            Span::styled("inference", Style::default().fg(ICE).bold()),
        ])),
        area,
    );
    let color = status_color(app.status);
    frame.render_widget(
        Paragraph::new(format!("{} {}", status_marker(app), app.status.label()))
            .style(Style::default().fg(color))
            .alignment(Alignment::Right),
        area,
    );
}

fn status_marker(app: &App) -> &'static str {
    match app.status {
        ServerStatus::Starting => ["·", "✧", "✦", "✧"][(app.startup_frame / 2) % 4],
        ServerStatus::Ready => "●",
        ServerStatus::Stopping | ServerStatus::Stopped | ServerStatus::Failed => "·",
    }
}

fn info_line(frame: &mut Frame, area: Rect, label: &str, value: String, color: Color) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!("{label:<10}"), Style::default().fg(MUTED)),
            Span::styled(value, Style::default().fg(color)),
        ])),
        area,
    );
}

fn access_summary(config: &Config) -> String {
    let mut parts = vec![
        if config.runtime.mmap {
            "mmap"
        } else {
            "loaded"
        },
        if config.runtime.warmup {
            "warmup"
        } else {
            "no warmup"
        },
        if config.runtime.repack {
            "repack"
        } else {
            "no repack"
        },
    ];
    let model = config.model_label().to_ascii_lowercase();
    if model.contains("gpt-oss-120b") {
        parts.push("MoE · 5.1B active/token");
    } else if model.contains("gpt-oss-20b") {
        parts.push("MoE · 3.6B active/token");
    }
    parts.join(" · ")
}

fn configure(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Min(8),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(2),
    ])
    .split(area);
    section_header(
        frame,
        rows[0],
        "configure",
        "enter exact  ·  \u{2190}\u{2192} adjust  ·  space toggle",
    );
    frame.render_widget(
        Paragraph::new(format!("saved to  {}", app.config_path.display()))
            .style(Style::default().fg(MUTED)),
        rows[1],
    );

    let settings = SettingField::ALL.iter().map(|field| {
        Row::new(vec![
            app.setting_label(*field).to_string(),
            app.setting_value(*field),
        ])
        .style(Style::default().fg(INK))
    });
    let table = Table::new(settings, [Constraint::Length(24), Constraint::Min(20)])
        .row_highlight_style(Style::default().fg(ICE).bg(SURFACE).bold())
        .highlight_symbol("› ");
    let mut state = ratatui::widgets::TableState::default();
    state.select(Some(app.setting_index));
    frame.render_stateful_widget(table, rows[2], &mut state);

    frame.render_widget(
        Paragraph::new(app.setting_hint(app.selected_field())).style(Style::default().fg(ICE)),
        rows[3],
    );
    frame.render_widget(
        Paragraph::new(app.status_detail.as_str()).style(status_detail_style(app.status)),
        rows[4],
    );
    footer(
        frame,
        rows[5],
        &[("enter", "set"), ("s", "save"), ("esc", "back")],
    );
}

fn logs(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(4),
        Constraint::Length(2),
    ])
    .split(area);
    section_header(frame, rows[0], "logs", &format!("{} lines", app.logs.len()));
    let visible = rows[1].height as usize;
    let end = app.logs.len().saturating_sub(app.log_offset);
    let start = end.saturating_sub(visible);
    let items = app
        .logs
        .iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|line| {
            let color = if line.contains("error") || line.contains("failed") {
                CORAL
            } else if line.starts_with('$') {
                ICE
            } else {
                INK
            };
            ListItem::new(Line::from(Span::styled(
                line.as_str(),
                Style::default().fg(color),
            )))
        });
    let mut state = ListState::default();
    frame.render_stateful_widget(List::new(items), rows[1], &mut state);
    footer(
        frame,
        rows[2],
        &[("↑↓", "scroll"), ("home/end", "jump"), ("esc", "back")],
    );
}

fn help_overlay(frame: &mut Frame) {
    let area = centered_rect(58, 62, frame.area());
    frame.render_widget(Clear, area);
    let block = dialog(" keys ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let text = Text::from(vec![
        help_line("s", "start or stop the server"),
        help_line("c", "configure the launch profile"),
        help_line("l", "read server output"),
        help_line("r", "restart the server"),
        help_line("q", "quit and stop the managed server"),
        Line::from(""),
        Line::from(Span::styled(
            "? or esc closes this",
            Style::default().fg(MUTED),
        )),
    ]);
    frame.render_widget(Paragraph::new(text), inner);
}

fn editor_overlay(frame: &mut Frame, app: &App) {
    let Some(editor) = app.editor.as_ref() else {
        return;
    };
    let area = centered_rect(72, 32, frame.area());
    frame.render_widget(Clear, area);
    let title = format!(" {} ", app.setting_label(editor.field).to_lowercase());
    let block = dialog(&title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(inner);
    let (visible_value, cursor_column) = editor_window(&editor.value, editor.cursor, rows[0].width);
    frame.render_widget(
        Paragraph::new(visible_value).style(Style::default().fg(INK).bg(CANVAS)),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(app.editor_error.as_deref().unwrap_or("")).style(Style::default().fg(CORAL)),
        rows[1],
    );
    frame.render_widget(
        Paragraph::new(app.setting_hint(editor.field)).style(Style::default().fg(MUTED)),
        rows[2],
    );
    frame.render_widget(
        Paragraph::new("enter apply  ·  esc cancel").style(Style::default().fg(MUTED)),
        rows[4],
    );
    let cursor_x = rows[0].x + cursor_column;
    frame.set_cursor_position((cursor_x, rows[0].y));
}

fn editor_window(value: &str, cursor: usize, width: u16) -> (&str, u16) {
    let available = width.saturating_sub(1) as usize;
    let before_cursor = &value[..cursor];
    let cursor_width = UnicodeWidthStr::width(before_cursor);
    if cursor_width <= available {
        return (value, cursor_width as u16);
    }

    let target = cursor_width - available;
    let mut removed = 0;
    let mut start = 0;
    for (index, character) in before_cursor.char_indices() {
        if removed >= target {
            start = index;
            break;
        }
        removed += character.width().unwrap_or(0);
        start = index + character.len_utf8();
    }
    let visible_cursor = UnicodeWidthStr::width(&value[start..cursor]).min(available);
    (&value[start..], visible_cursor as u16)
}

fn section_header(frame: &mut Frame, area: Rect, title: &str, note: &str) {
    frame.render_widget(
        Paragraph::new(title).style(Style::default().fg(INK).bold()),
        area,
    );
    frame.render_widget(
        Paragraph::new(note)
            .style(Style::default().fg(MUTED))
            .alignment(Alignment::Right),
        area,
    );
}

fn footer(frame: &mut Frame, area: Rect, keys: &[(&str, &str)]) {
    let mut spans = Vec::new();
    for (index, (key, label)) in keys.iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(MUTED)));
        }
        spans.push(Span::styled(*key, Style::default().fg(ICE).bold()));
        spans.push(Span::styled(
            format!(" {label}"),
            Style::default().fg(MUTED),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn dialog<'a>(title: &'a str) -> Block<'a> {
    Block::default()
        .title(title)
        .title_style(Style::default().fg(ICE).bold())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(MUTED))
        .style(Style::default().bg(SURFACE))
        .padding(ratatui::widgets::Padding::horizontal(2))
}

fn help_line<'a>(key: &'a str, description: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{key:>5}  "), Style::default().fg(ICE).bold()),
        Span::styled(description, Style::default().fg(INK)),
    ])
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn status_color(status: ServerStatus) -> Color {
    match status {
        ServerStatus::Ready => MINT,
        ServerStatus::Starting | ServerStatus::Stopping => ICE,
        ServerStatus::Failed => CORAL,
        ServerStatus::Stopped => MUTED,
    }
}

fn status_detail_style(status: ServerStatus) -> Style {
    Style::default().fg(if status == ServerStatus::Failed {
        CORAL
    } else {
        MUTED
    })
}

fn format_tokens(tokens: u32) -> String {
    if tokens >= 1024 && tokens % 1024 == 0 {
        format!("{}K", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

fn plural(value: u16, singular: &'static str, plural: &'static str) -> &'static str {
    if value == 1 { singular } else { plural }
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend};

    use super::*;
    use crate::config::{Config, ModelSource};

    fn rendered_dashboard() -> String {
        let backend = TestBackend::new(100, 26);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default(), "test.toml".into());
        app.dismiss_server_prompt();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>()
    }

    #[test]
    fn dashboard_describes_disk_mapping_without_ram_requirement() {
        let text = rendered_dashboard();
        assert!(text.contains("59.1 GiB mapped from disk"));
        assert!(text.contains("not required to fit in RAM"));
        assert!(text.contains("MoE · 5.1B active/token"));
        assert!(!text.contains("GiB needed"));
        assert!(!text.contains("MEMORY RUNWAY"));
        assert!(!text.contains("LAUNCH COMMAND"));
    }

    #[test]
    fn dashboard_keeps_core_controls_visible() {
        let text = rendered_dashboard();
        assert!(text.contains("ggml-org/gpt-oss-120b-GGUF"));
        assert!(text.contains("configure"));
        assert!(text.contains("logs"));
    }

    #[test]
    fn starting_status_uses_a_pulsing_star() {
        let mut app = App::new(Config::default(), "test.toml".into());
        app.status = ServerStatus::Starting;
        app.startup_frame = 4;
        assert_eq!(status_marker(&app), "✦");
    }

    #[test]
    fn editor_window_keeps_the_cursor_visible_for_long_paths() {
        let value = r"C:\very\long\model\directory\gpt-oss-120b.gguf";
        let (visible, cursor) = editor_window(value, value.len(), 16);
        assert!(visible.ends_with("120b.gguf"));
        assert!(!visible.starts_with("C:\\"));
        assert!(cursor < 16);
    }

    #[test]
    fn editor_window_counts_wide_characters() {
        let value = "models/\u{6a21}\u{578b}/large.gguf";
        let (visible, cursor) = editor_window(value, value.len(), 12);
        assert!(visible.ends_with("large.gguf"));
        assert!(cursor < 12);
    }

    #[test]
    fn local_models_do_not_claim_gpt_oss_active_parameter_count() {
        let backend = TestBackend::new(100, 26);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut config = Config::default();
        config.model.source = ModelSource::Local("other-model.gguf".into());
        let mut app = App::new(config, "test.toml".into());
        app.dismiss_server_prompt();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!text.contains("active/token"));
    }

    #[test]
    fn configure_screen_explains_repo_paths_and_server_lookup() {
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(Config::default(), "test.toml".into());
        app.dismiss_server_prompt();
        app.view = View::Configure;
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("Model repository"));
        assert!(text.contains("llama-server path"));
        assert!(text.contains("llama-server  (from PATH)"));
        assert!(text.contains("switch source"));

        app.handle_key(ratatui::crossterm::event::KeyEvent::new(
            ratatui::crossterm::event::KeyCode::Right,
            ratatui::crossterm::event::KeyModifiers::NONE,
        ));
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("GGUF file path"));
        assert!(text.contains("<enter path to .gguf>"));
    }

    #[test]
    fn missing_server_prompt_explains_the_next_action() {
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut config = Config::default();
        config.server.executable = "__tinyinference_missing_server__".into();
        let mut app = App::new(config, "test.toml".into());
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("llama-server not found"));
        assert!(text.contains("Set the executable path"));
        assert!(text.contains("enter"));
        assert!(text.contains("configure"));
    }
}
