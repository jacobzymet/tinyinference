//! Compile-time embed: the release binary ships alone — no HTML/JS/PNG sidecars.

pub const ADMIN_HTML: &str = include_str!("../ui/index.html");
pub const CHAT_HTML: &str = include_str!("../ui/chat.html");
pub const ORB_JS: &str = include_str!("../ui/orb.js");
pub const HIGHLIGHT_JS: &str = include_str!("../ui/vendor/highlight.min.js");
pub const APP_ICON_PNG: &[u8] = include_bytes!("../../assets/ti.png");
pub const UI_MARK_WHITE_PNG: &[u8] = include_bytes!("../../assets/ti-transparent-bg-white.png");
pub const UI_MARK_BLACK_PNG: &[u8] = include_bytes!("../../assets/ti-transparent-bg-black.png");
