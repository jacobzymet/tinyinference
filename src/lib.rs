pub mod agent;
pub mod app;
pub mod config;
pub mod inference;
pub mod network;
pub mod system;
pub mod update;
pub mod web;

// Stable crate-root paths (pre-reorg import style).
pub use agent::{chat, skills};
pub use inference::{cache, fetch, hub, instance, server, share_proxy, tls};
