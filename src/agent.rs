//! Agent mode: skill-augmented chat with a simple tool-call loop.
//!
//! Local models rarely implement OpenAI `tools` reliably, so we use an
//! explicit XML+JSON protocol the model is instructed to emit:
//!
//! ```text
//! <tool_call>
//! {"name":"web_search","arguments":{"query":"…"}}
//! </tool_call>
//! ```
//!
//! The server executes the skill, feeds `<tool_result>` back, and continues
//! until the model answers without a tool call (or hits the round limit).

use std::{
    io::{BufRead, BufReader},
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::chat::{sse_error, ChatStream, CHANNEL_CAPACITY, REQUEST_TIMEOUT};

const MAX_AGENT_ROUNDS: usize = 4;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_SEARCH_RESULTS: usize = 6;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentRequest {
    pub messages: Vec<Value>,
    #[serde(default)]
    pub agent: bool,
    #[serde(default)]
    pub skills: AgentSkills,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentSkills {
    #[serde(default)]
    pub web_search: bool,
}

impl AgentSkills {
    pub fn any_enabled(&self) -> bool {
        self.web_search
    }
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    arguments: Value,
}

/// Run the agent loop on a background thread and stream SSE to the client.
pub fn stream_agent(api_base: &str, mut request: AgentRequest) -> ChatStream {
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let api_base = api_base.trim_end_matches('/').to_string();
    thread::spawn(move || {
        if let Err(error) = run_agent_loop(&api_base, &mut request, &tx) {
            let _ = tx.blocking_send(Ok(sse_error(&error)));
        }
    });
    ReceiverStream::new(rx)
}

fn run_agent_loop(
    api_base: &str,
    request: &mut AgentRequest,
    tx: &mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
) -> Result<(), String> {
    inject_agent_system_prompt(&mut request.messages, &request.skills);
    let _ = tx.blocking_send(Ok(sse_agent(json!({
        "phase": "status",
        "message": "Agent mode"
    }))));

    for round in 0..MAX_AGENT_ROUNDS {
        let _ = tx.blocking_send(Ok(sse_agent(json!({
            "phase": "status",
            "message": if round == 0 { "Thinking…" } else { "Continuing…" }
        }))));

        // Stream tokens live from llama-server. Tool-call rounds suppress the
        // XML once detected; final answers appear token-by-token in the UI.
        let reply = stream_once(api_base, &request.messages, tx)?;
        let visible = strip_think_blocks(&reply);

        if let Some(call) = extract_tool_call(&visible) {
            if !skill_allowed(&call.name, &request.skills) {
                return Err(format!("Skill '{}' is not enabled.", call.name));
            }

            let _ = tx.blocking_send(Ok(sse_agent(json!({
                "phase": "content_clear",
            }))));
            let _ = tx.blocking_send(Ok(sse_agent(json!({
                "phase": "tool_call",
                "name": call.name,
                "arguments": call.arguments,
            }))));

            let result = execute_tool(&call)?;
            let preview = result.chars().take(240).collect::<String>();
            let _ = tx.blocking_send(Ok(sse_agent(json!({
                "phase": "tool_result",
                "name": call.name,
                "ok": true,
                "preview": preview,
            }))));

            request.messages.push(json!({
                "role": "assistant",
                "content": reply,
            }));
            request.messages.push(json!({
                "role": "user",
                "content": format!(
                    "<tool_result name=\"{}\">\n{}\n</tool_result>\n\nUse these results to answer the user. If you need another search, emit another tool_call. Otherwise reply normally without a tool_call.",
                    call.name, result
                ),
            }));
            continue;
        }

        let _ = tx.blocking_send(Ok(b"data: [DONE]\n\n".to_vec()));
        return Ok(());
    }

    Err("Agent stopped after too many tool rounds.".into())
}

fn skill_allowed(name: &str, skills: &AgentSkills) -> bool {
    match name {
        "web_search" => skills.web_search,
        _ => false,
    }
}

fn inject_agent_system_prompt(messages: &mut Vec<Value>, skills: &AgentSkills) {
    let block = agent_system_block(skills);
    if block.is_empty() {
        return;
    }
    if let Some(first) = messages.first_mut() {
        if first.get("role").and_then(|r| r.as_str()) == Some("system") {
            if let Some(content) = first.get("content").and_then(|c| c.as_str()) {
                let merged = format!("{content}\n\n{block}");
                first
                    .as_object_mut()
                    .unwrap()
                    .insert("content".into(), Value::String(merged));
                return;
            }
        }
    }
    messages.insert(0, json!({ "role": "system", "content": block }));
}

fn agent_system_block(skills: &AgentSkills) -> String {
    let mut lines: Vec<String> = vec![
        "You are running in agent mode with callable skills.".into(),
        "When a skill would help, emit EXACTLY one tool call in this form and nothing else after it:".into(),
        "<tool_call>".into(),
        r#"{"name":"SKILL_NAME","arguments":{...}}"#.into(),
        "</tool_call>".into(),
        "Do not invent skill results. Wait for a <tool_result> message.".into(),
        "You may call skills multiple times in sequence: after each <tool_result>, either emit another tool_call (for a follow-up search or different query) or reply normally with no tool_call.".into(),
        "Prefer a follow-up tool_call when the first result is incomplete, stale, or misses what the user asked.".into(),
        "If you can answer without a skill, reply normally with no tool_call.".into(),
    ];
    if skills.web_search {
        lines.push(
            "Available skill: web_search — search the public web via DuckDuckGo. Arguments: {\"query\":\"search terms\"}. You can run several searches one after another when that helps."
                .into(),
        );
    }
    lines.join("\n")
}

/// Stream one chat completion from llama-server, forwarding content deltas to
/// the client as they arrive. Returns the full assistant text.
///
/// Once a `<tool_call>` marker appears, further deltas are withheld and a
/// `content_clear` agent event is sent so the UI does not keep tool XML.
fn stream_once(
    api_base: &str,
    messages: &[Value],
    tx: &mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
) -> Result<String, String> {
    let url = format!("{api_base}/chat/completions");
    let payload = json!({
        "model": "local",
        "stream": true,
        "messages": messages,
    });

    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();

    let mut response = agent
        .post(&url)
        .send_json(&payload)
        .map_err(|error| error.to_string())?;

    if response.status() != 200 {
        let status = response.status();
        let body = response.body_mut().read_to_string().unwrap_or_default();
        return Err(format!("llama-server responded with {status}: {body}"));
    }

    let reader = response.body_mut().as_reader();
    let mut lines = BufReader::new(reader);
    let mut content = String::new();
    let mut forwarding = true;
    let mut line = String::new();

    loop {
        line.clear();
        let read = lines.read_line(&mut line).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() || !trimmed.starts_with("data:") {
            continue;
        }
        let data = trimmed[5..].trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let Some(delta) = value
            .pointer("/choices/0/delta/content")
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        if delta.is_empty() {
            continue;
        }

        content.push_str(delta);

        if forwarding && content.contains("<tool_call>") {
            forwarding = false;
            let _ = tx.blocking_send(Ok(sse_agent(json!({ "phase": "content_clear" }))));
            continue;
        }
        if !forwarding {
            continue;
        }

        let frame = json!({
            "choices": [{ "delta": { "content": delta }, "index": 0 }]
        });
        if tx
            .blocking_send(Ok(format!("data: {frame}\n\n").into_bytes()))
            .is_err()
        {
            // Client disconnected; still drain? Stop early — llama will cancel
            // when we drop the reader.
            break;
        }
    }

    if content.is_empty() {
        return Err("Model returned no message content.".into());
    }
    Ok(content)
}

fn sse_agent(payload: Value) -> Vec<u8> {
    format!("event: agent\ndata: {payload}\n\n").into_bytes()
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = text.to_string();
    for (open, close) in [("<think>", "</think>"), ("<thinking>", "</thinking>")] {
        while let Some(start) = out.find(open) {
            if let Some(rel) = out[start + open.len()..].find(close) {
                let end = start + open.len() + rel + close.len();
                out.replace_range(start..end, "");
            } else {
                out.replace_range(start.., "");
                break;
            }
        }
    }
    out
}

fn extract_tool_call(text: &str) -> Option<ToolCall> {
    let start = text.rfind("<tool_call>")?;
    let after = start + "<tool_call>".len();
    let end = text[after..].find("</tool_call>")? + after;
    let raw = text[after..end].trim();
    let value: Value = serde_json::from_str(raw).ok()?;
    let name = value.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = value.get("arguments").cloned().unwrap_or_else(|| json!({}));
    Some(ToolCall { name, arguments })
}

fn execute_tool(call: &ToolCall) -> Result<String, String> {
    match call.name.as_str() {
        "web_search" => {
            let query = call
                .arguments
                .get("query")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "web_search requires a non-empty \"query\" string.".to_string())?;
            duckduckgo_search(query)
        }
        other => Err(format!("Unknown skill '{other}'.")),
    }
}

#[derive(Debug)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

fn duckduckgo_search(query: &str) -> Result<String, String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(SEARCH_TIMEOUT))
        .user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        .build()
        .new_agent();

    // HTML endpoint — no API key. Prefer POST; some regions treat GET more harshly.
    let mut response = agent
        .post("https://html.duckduckgo.com/html/")
        .header("Accept", "text/html")
        .send_form([("q", query), ("b", "")])
        .or_else(|_| {
            agent
                .get("https://html.duckduckgo.com/html/")
                .query("q", query)
                .header("Accept", "text/html")
                .call()
        })
        .map_err(|error| format!("DuckDuckGo request failed: {error}"))?;

    if response.status() != 200 && response.status() != 202 {
        return Err(format!(
            "DuckDuckGo returned HTTP {}",
            response.status()
        ));
    }

    let html = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("Failed to read DuckDuckGo body: {error}"))?;

    if html.contains("anomaly.js") || html.contains("Please complete the captcha") {
        return Err("DuckDuckGo blocked the search request (captcha / bot check).".into());
    }

    let hits = parse_ddg_html(&html);
    if hits.is_empty() {
        // Fallback: Instant Answer API (sparser, but keyless and structured).
        return duckduckgo_instant_answer(query);
    }

    let mut out = format!("Web search results for {query:?} (DuckDuckGo):\n");
    for (index, hit) in hits.iter().take(MAX_SEARCH_RESULTS).enumerate() {
        out.push_str(&format!(
            "\n{}. {}\n   URL: {}\n   {}\n",
            index + 1,
            hit.title,
            hit.url,
            hit.snippet
        ));
    }
    Ok(out)
}

fn duckduckgo_instant_answer(query: &str) -> Result<String, String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(SEARCH_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();

    let mut response = agent
        .get("https://api.duckduckgo.com/")
        .query("q", query)
        .query("format", "json")
        .query("no_html", "1")
        .query("skip_disambig", "1")
        .call()
        .map_err(|error| format!("DuckDuckGo Instant Answer failed: {error}"))?;

    let body: Value = response
        .body_mut()
        .read_json()
        .map_err(|error| format!("Invalid Instant Answer JSON: {error}"))?;

    let mut lines = vec![format!("Web search results for {query:?} (DuckDuckGo Instant Answer):")];
    if let Some(text) = body.get("AbstractText").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            let url = body
                .get("AbstractURL")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            lines.push(format!("\nSummary: {text}"));
            if !url.is_empty() {
                lines.push(format!("Source: {url}"));
            }
        }
    }

    let mut count = 0usize;
    if let Some(topics) = body.get("RelatedTopics").and_then(|v| v.as_array()) {
        for topic in topics {
            if count >= MAX_SEARCH_RESULTS {
                break;
            }
            if let Some(text) = topic.get("Text").and_then(|v| v.as_str()) {
                let url = topic
                    .get("FirstURL")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                count += 1;
                lines.push(format!("\n{count}. {text}"));
                if !url.is_empty() {
                    lines.push(format!("   URL: {url}"));
                }
            } else if let Some(nested) = topic.get("Topics").and_then(|v| v.as_array()) {
                for item in nested {
                    if count >= MAX_SEARCH_RESULTS {
                        break;
                    }
                    if let Some(text) = item.get("Text").and_then(|v| v.as_str()) {
                        let url = item
                            .get("FirstURL")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        count += 1;
                        lines.push(format!("\n{count}. {text}"));
                        if !url.is_empty() {
                            lines.push(format!("   URL: {url}"));
                        }
                    }
                }
            }
        }
    }

    if lines.len() == 1 {
        lines.push("\nNo results found.".into());
    }
    Ok(lines.join("\n"))
}

fn parse_ddg_html(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut rest = html;
    while let Some(idx) = rest.find("result__a") {
        rest = &rest[idx..];
        let href = match attr_after(rest, "href=\"") {
            Some(v) => v,
            None => {
                rest = &rest[1..];
                continue;
            }
        };
        let title_html = match between(rest, ">", "</a>") {
            Some(v) => v,
            None => {
                rest = &rest[1..];
                continue;
            }
        };
        let title = collapse_ws(&strip_tags(title_html));
        let url = decode_ddg_href(&href);
        // Snippet often follows shortly after in result__snippet
        let snippet = rest
            .find("result__snippet")
            .and_then(|s| {
                let slice = &rest[s..];
                // stop looking too far ahead so we don't steal the next result's snippet
                let window = &slice[..slice.len().min(1200)];
                between(window, ">", "</").map(strip_tags).map(|s| collapse_ws(&s))
            })
            .unwrap_or_default();

        if !title.is_empty() && !url.is_empty() {
            hits.push(SearchHit {
                title,
                url,
                snippet,
            });
        }
        if hits.len() >= MAX_SEARCH_RESULTS {
            break;
        }
        rest = &rest[1..];
    }
    hits
}

fn attr_after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let start = s.find(key)? + key.len();
    let end = s[start..].find('"')? + start;
    Some(&s[start..end])
}

fn between<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = s.find(open)? + open.len();
    let end = s[start..].find(close)? + start;
    Some(&s[start..end])
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    html_unescape(&out)
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

fn decode_ddg_href(href: &str) -> String {
    let full = if let Some(rest) = href.strip_prefix("//") {
        format!("https://{rest}")
    } else {
        href.to_string()
    };
    if let Some(idx) = full.find("uddg=") {
        let start = idx + "uddg=".len();
        let end = full[start..]
            .find('&')
            .map(|i| start + i)
            .unwrap_or(full.len());
        return percent_decode(&full[start..end]);
    }
    full
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (from_hex(bytes[i + 1]), from_hex(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_call_json() {
        let text = r#"Sure.
<tool_call>
{"name":"web_search","arguments":{"query":"rust async"}}
</tool_call>"#;
        let call = extract_tool_call(text).unwrap();
        assert_eq!(call.name, "web_search");
        assert_eq!(call.arguments["query"], "rust async");
    }

    #[test]
    fn strips_think_before_tool_parse() {
        let text = r#"<think>plan</think>
<tool_call>
{"name":"web_search","arguments":{"query":"hi"}}
</tool_call>"#;
        let visible = strip_think_blocks(text);
        let call = extract_tool_call(&visible).unwrap();
        assert_eq!(call.arguments["query"], "hi");
    }

    #[test]
    fn decodes_ddg_redirect() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath&rut=x";
        assert_eq!(decode_ddg_href(href), "https://example.com/path");
    }
}
