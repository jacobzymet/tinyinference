//! Streaming proxy between the tinyinference chat UI and llama-server's
//! OpenAI-compatible `/v1/chat/completions` endpoint.
//!
//! `ureq` is blocking, so the request/response round trip runs on a plain
//! thread (matching the pattern already used for downloads in `fetch.rs` and
//! probes in `server.rs`) and forwards raw bytes to the async response body
//! through a channel.

use std::{io::Read, thread, time::Duration};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Generation on CPU can be very slow (see README), so allow a generous
/// window for the whole streamed response rather than a short request timeout.
pub(crate) const REQUEST_TIMEOUT: Duration = Duration::from_secs(60 * 60);
pub(crate) const CHANNEL_CAPACITY: usize = 32;

pub type ChatStream = ReceiverStream<Result<Vec<u8>, std::io::Error>>;

/// Proxy a chat completion request to `api_base` (e.g. `http://127.0.0.1:8080/v1`),
/// forcing `stream: true`, and relay the upstream SSE bytes to the caller as
/// they arrive. Upstream failures are turned into an `event: error` SSE frame
/// instead of failing the HTTP response, since headers are already committed
/// by the time a mid-stream error can happen.
pub fn stream_completion(api_base: &str, mut payload: serde_json::Value) -> ChatStream {
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));
    thread::spawn(move || {
        if let Some(object) = payload.as_object_mut() {
            object.insert("stream".into(), serde_json::json!(true));
            object
                .entry("model")
                .or_insert_with(|| serde_json::json!("local"));
        }

        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();

        match agent.post(&url).send_json(&payload) {
            Ok(mut response) => {
                if response.status() != 200 {
                    let status = response.status();
                    let body = response.body_mut().read_to_string().unwrap_or_default();
                    let _ = tx.blocking_send(Ok(sse_error(&format!(
                        "llama-server responded with {status}: {body}"
                    ))));
                    return;
                }
                let mut reader = response.body_mut().as_reader();
                let mut buffer = [0u8; 8192];
                loop {
                    match reader.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(read) => {
                            if tx.blocking_send(Ok(buffer[..read].to_vec())).is_err() {
                                // Client disconnected; stop reading from llama-server.
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = tx.blocking_send(Ok(sse_error(&error.to_string())));
                            break;
                        }
                    }
                }
            }
            Err(error) => {
                let _ = tx.blocking_send(Ok(sse_error(&error.to_string())));
            }
        }
    });
    ReceiverStream::new(rx)
}

pub(crate) fn sse_error(message: &str) -> Vec<u8> {
    let payload = serde_json::json!({ "error": message });
    format!("event: error\ndata: {payload}\n\n").into_bytes()
}
