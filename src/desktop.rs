//! Native window hosting the control panel.
//!
//! The control panel is served by the same axum instance as everything else;
//! this module just points a platform webview (WebView2 / WKWebView /
//! WebKitGTK) at it so tinyinference launches as an app rather than a URL.
//!
//! The window is deliberately a *single-document* host: it shows the control
//! panel and nothing else. Every other destination — the chat page, the
//! llama-server UI, any outbound link — is handed to the user's real browser,
//! where they have their tabs, history, and extensions.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use tao::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::{Icon, WindowBuilder},
};
use wry::{NewWindowResponse, WebViewBuilder};

use crate::{app::App, system::open_in_browser};

const MIN_WIDTH: f64 = 720.0;
const MIN_HEIGHT: f64 = 520.0;
const DEFAULT_WIDTH: f64 = 980.0;
const DEFAULT_HEIGHT: f64 = 700.0;

/// Window / taskbar icon. Windows looks best around 32px; we downscale the
/// bundled 800px asset rather than shipping a second file.
fn app_icon() -> Option<Icon> {
    let bytes = include_bytes!("../assets/ti.png");
    let image = image::load_from_memory(bytes).ok()?.into_rgba8();
    let resized = image::imageops::resize(
        &image,
        32,
        32,
        image::imageops::FilterType::Lanczos3,
    );
    let (width, height) = resized.dimensions();
    Icon::from_rgba(resized.into_raw(), width, height).ok()
}

/// True when `url` is the control panel document itself, and so may be shown
/// in the native window.
///
/// Anything else under the same origin — `/chat` above all — counts as
/// external: same server, different surface. Fragment and query variants of
/// the root are kept in-window because the stage tabs navigate by hash.
fn is_control_panel(url: &str, base: &str) -> bool {
    match url.strip_prefix(base) {
        Some(rest) => {
            rest.is_empty() || rest == "/" || rest.starts_with("/#") || rest.starts_with("/?")
        }
        None => false,
    }
}

/// Send a URL to the default browser, ignoring non-web schemes so a stray
/// `about:blank` or `javascript:` navigation never reaches a shell command.
fn hand_off(url: &str) {
    if url.starts_with("http://") || url.starts_with("https://") {
        let _ = open_in_browser(url);
    }
}

/// Messages sent to the window from outside the event loop.
#[derive(Debug, Clone, Copy)]
pub enum UserEvent {
    /// A second launch asked this instance to come forward.
    Focus,
}

/// Open the control panel window and run the event loop until it closes.
///
/// Never returns: `tao`'s event loop exits the process, so the managed
/// `llama-server` is shut down from inside the handler rather than after.
pub fn run(url: &str, app: Arc<Mutex<App>>) -> Result<()> {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();

    // Let a second launch raise this window instead of starting a rival server.
    // The proxy is Send + Sync, so the web handler can poke it from a request.
    let proxy = event_loop.create_proxy();
    if let Ok(mut app) = app.lock() {
        app.set_focus_hook(Box::new(move || {
            let _ = proxy.send_event(UserEvent::Focus);
        }));
    }

    let window = WindowBuilder::new()
        .with_title("tinyinference")
        .with_inner_size(LogicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT))
        .with_min_inner_size(LogicalSize::new(MIN_WIDTH, MIN_HEIGHT))
        .with_window_icon(app_icon())
        .build(&event_loop)
        .context("could not create the tinyinference window")?;

    let base = url.trim_end_matches('/').to_string();
    let nav_base = base.clone();
    let popup_base = base.clone();

    let webview = WebViewBuilder::new()
        .with_url(url)
        .with_navigation_handler(move |target| {
            if is_control_panel(&target, &nav_base) {
                return true;
            }
            hand_off(&target);
            false
        })
        .with_new_window_req_handler(move |target, _features| {
            // `target="_blank"` links (llama-server UI, the endpoint link).
            if !is_control_panel(&target, &popup_base) {
                hand_off(&target);
            }
            NewWindowResponse::Deny
        })
        .build(&window)
        .context("could not create the control panel webview")?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        // Keep the webview alive for as long as the window it is drawn into.
        let _ = &webview;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                // The loop exits the process, so stop llama-server here.
                if let Ok(mut app) = app.lock() {
                    app.shutdown();
                }
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(UserEvent::Focus) => {
                // A minimized window has to be restored before it can take focus.
                window.set_minimized(false);
                window.set_visible(true);
                window.set_focus();
            }
            _ => {}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::is_control_panel;

    const BASE: &str = "http://127.0.0.1:3920";

    #[test]
    fn control_panel_root_stays_in_the_window() {
        assert!(is_control_panel("http://127.0.0.1:3920", BASE));
        assert!(is_control_panel("http://127.0.0.1:3920/", BASE));
        assert!(is_control_panel("http://127.0.0.1:3920/#dashboard", BASE));
        assert!(is_control_panel("http://127.0.0.1:3920/?x=1", BASE));
    }

    #[test]
    fn chat_and_outbound_links_go_to_the_browser() {
        assert!(!is_control_panel("http://127.0.0.1:3920/chat", BASE));
        assert!(!is_control_panel("http://127.0.0.1:8080/", BASE));
        assert!(!is_control_panel("https://huggingface.co/", BASE));
    }

    /// A different host that merely starts with the same text must not slip
    /// through the prefix check.
    #[test]
    fn lookalike_origins_are_external() {
        assert!(!is_control_panel("http://127.0.0.1:39201/", BASE));
        assert!(!is_control_panel("http://127.0.0.1:3920.evil.test/", BASE));
    }
}
