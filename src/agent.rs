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
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    chat::{CHANNEL_CAPACITY, ChatStream, REQUEST_TIMEOUT, sse_error},
    skills::UserSkill,
};

const MAX_AGENT_ROUNDS: usize = 6;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(25);
const PAGE_TIMEOUT: Duration = Duration::from_secs(12);
const MAX_SEARCH_RESULTS: usize = 6;
const MAX_PAGE_BYTES: u64 = 1_500_000;
/// Default extract size for the dedicated fetch_url capability.
const FETCH_URL_MAX_CHARS: usize = 8_000;

#[derive(Debug, Clone, Deserialize)]
pub struct AgentRequest {
    pub messages: Vec<Value>,
    #[serde(default)]
    pub agent: bool,
    #[serde(default)]
    pub skills: AgentSkills,
    /// Forwarded to llama-server for reasoning / thinking models.
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(default)]
    pub thinking_budget_tokens: Option<i64>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// How many result pages to open after DuckDuckGo returns links.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchDepth {
    /// Snippets / titles only — no page fetches.
    Off,
    /// Balanced default: a few pages with mid-size extracts.
    #[default]
    Auto,
    Light,
    Standard,
    Deep,
}

impl WebSearchDepth {
    /// `(pages_to_fetch, max_chars_per_page)`, or `None` when scraping is off.
    fn scrape_plan(self) -> Option<(usize, usize)> {
        match self {
            Self::Off => None,
            Self::Auto => Some((3, 2800)),
            Self::Light => Some((2, 1600)),
            Self::Standard => Some((4, 3200)),
            Self::Deep => Some((6, 4800)),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Light => "light",
            Self::Standard => "standard",
            Self::Deep => "deep",
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentSkills {
    #[serde(default)]
    pub web_search: bool,
    #[serde(default)]
    pub web_search_depth: WebSearchDepth,
    /// Fetch readable text from a single user-provided URL (not search).
    #[serde(default)]
    pub fetch_url: bool,
}

impl AgentSkills {
    pub fn any_enabled(&self) -> bool {
        self.web_search || self.fetch_url
    }
}

#[derive(Debug, Clone)]
struct ToolCall {
    name: String,
    arguments: Value,
}

/// Run the agent loop on a background thread and stream SSE to the client.
pub fn stream_agent(
    api_base: &str,
    api_key: Option<&str>,
    mut request: AgentRequest,
    user_skills: Vec<UserSkill>,
) -> ChatStream {
    let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
    let api_base = api_base.trim_end_matches('/').to_string();
    let api_key = api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string);
    thread::spawn(move || {
        if let Err(error) = run_agent_loop(
            &api_base,
            api_key.as_deref(),
            &mut request,
            &user_skills,
            &tx,
        ) {
            let _ = tx.blocking_send(Ok(sse_error(&error)));
        }
    });
    ReceiverStream::new(rx)
}

fn run_agent_loop(
    api_base: &str,
    api_key: Option<&str>,
    request: &mut AgentRequest,
    user_skills: &[UserSkill],
    tx: &mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
) -> Result<(), String> {
    inject_agent_system_prompt(&mut request.messages, &request.skills, user_skills);
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
        let reply = stream_once(api_base, api_key, request, tx)?;
        let visible = strip_think_blocks(&reply);

        if let Some(call) = extract_tool_call(&visible) {
            if !capability_allowed(&call.name, &request.skills, user_skills) {
                return Err(format!("Capability '{}' is not enabled.", call.name));
            }

            let _ = tx.blocking_send(Ok(sse_agent(json!({
                "phase": "content_clear",
            }))));
            let _ = tx.blocking_send(Ok(sse_agent(json!({
                "phase": "tool_call",
                "name": call.name,
                "arguments": call.arguments,
            }))));

            let result = execute_tool(&call, &request.skills, user_skills)?;
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
            let follow_up = if call.name == "activate_skill" || call.name == "read_skill" {
                "Follow the activated skill instructions for the user's request. You may activate another skill, call web_search, or fetch_url if needed. Otherwise reply normally without a tool_call."
            } else {
                "Use these results to answer the user. If you need another search, to fetch a specific URL, or to activate a skill, emit another tool_call. Otherwise reply normally without a tool_call."
            };
            request.messages.push(json!({
                "role": "user",
                "content": format!(
                    "<tool_result name=\"{}\">\n{}\n</tool_result>\n\n{follow_up}",
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

fn capability_allowed(name: &str, skills: &AgentSkills, user_skills: &[UserSkill]) -> bool {
    match name {
        "web_search" => skills.web_search,
        "fetch_url" => skills.fetch_url,
        "activate_skill" | "read_skill" => !user_skills.is_empty(),
        _ => false,
    }
}

/// True when the request should enter the agent tool loop.
pub fn should_run_agent(request: &AgentRequest, user_skills: &[UserSkill]) -> bool {
    request.agent && (request.skills.any_enabled() || !user_skills.is_empty())
}

fn inject_agent_system_prompt(
    messages: &mut Vec<Value>,
    skills: &AgentSkills,
    user_skills: &[UserSkill],
) {
    let block = agent_system_block(skills, user_skills);
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

fn agent_system_block(skills: &AgentSkills, user_skills: &[UserSkill]) -> String {
    let mut lines: Vec<String> = vec![
        "You are running in agent mode with callable capabilities.".into(),
        "When a capability would help, emit EXACTLY one tool call in this form and nothing else after it:".into(),
        "<tool_call>".into(),
        r#"{"name":"CAPABILITY_NAME","arguments":{...}}"#.into(),
        "</tool_call>".into(),
        "Do not invent capability results. Wait for a <tool_result> message.".into(),
        "You may call capabilities multiple times in sequence: after each <tool_result>, either emit another tool_call or reply normally with no tool_call.".into(),
        "Prefer a follow-up tool_call when the first result is incomplete, stale, or misses what the user asked.".into(),
        "If you can answer without a capability, reply normally with no tool_call.".into(),
    ];
    if skills.web_search {
        let depth = skills.web_search_depth;
        let depth_note = match depth.scrape_plan() {
            None => {
                "Returns titles, URLs, and snippets only (page fetch depth is off).".to_string()
            }
            Some((pages, chars)) => format!(
                "Also opens up to {pages} result pages and extracts ~{chars} characters of text each (depth: {}).",
                depth.label()
            ),
        };
        lines.push(format!(
            "Available capability: web_search — search the public web via DuckDuckGo, then dig into sources. Use this when the user wants you to look something up and has not given a specific URL. {depth_note} Arguments: {{\"query\":\"search terms\"}}."
        ));
    }
    if skills.fetch_url {
        lines.push(format!(
            "Available capability: fetch_url — open one specific http(s) URL and extract readable page text (~{FETCH_URL_MAX_CHARS} characters). Prefer this over web_search when the user (or a prior result) already gives a concrete URL to view. Arguments: {{\"url\":\"https://…\"}}."
        ));
    }
    if !user_skills.is_empty() {
        lines.push(
            "Available capability: activate_skill — load a skill's full SKILL.md instructions into context. Arguments: {\"name\":\"skill name or id\"}. Only activate skills that match the user's request. read_skill is an alias.".into(),
        );
        lines.push(crate::skills::user_skills_catalog_block(user_skills));
    }
    lines.join("\n")
}

/// Stage-1 only: put the skill catalog (names/descriptions) into the system prompt.
pub fn inject_skill_catalog_into_messages(messages: &mut Vec<Value>, user_skills: &[UserSkill]) {
    let block = crate::skills::user_skills_catalog_block(user_skills);
    if block.is_empty() {
        return;
    }
    let note = format!(
        "{block}\n\nTo load full skill instructions you need agent mode (toggle Agent or @agent), which exposes activate_skill."
    );
    if let Some(first) = messages.first_mut() {
        if first.get("role").and_then(|r| r.as_str()) == Some("system") {
            if let Some(content) = first.get("content").and_then(|c| c.as_str()) {
                let merged = format!("{content}\n\n{note}");
                first
                    .as_object_mut()
                    .unwrap()
                    .insert("content".into(), Value::String(merged));
                return;
            }
        }
    }
    messages.insert(0, json!({ "role": "system", "content": note }));
}

/// Stream one chat completion from llama-server, forwarding content deltas to
/// the client as they arrive. Returns the full assistant text.
///
/// Once a `<tool_call>` marker appears, further deltas are withheld and a
/// `content_clear` agent event is sent so the UI does not keep tool XML.
fn stream_once(
    api_base: &str,
    api_key: Option<&str>,
    request: &AgentRequest,
    tx: &mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
) -> Result<String, String> {
    let url = format!("{api_base}/chat/completions");
    let mut payload = json!({
        "model": "local",
        "stream": true,
        "messages": request.messages,
    });
    if let Some(object) = payload.as_object_mut() {
        if let Some(kwargs) = &request.chat_template_kwargs {
            object.insert("chat_template_kwargs".into(), kwargs.clone());
        }
        if let Some(budget) = request.thinking_budget_tokens {
            object.insert("thinking_budget_tokens".into(), json!(budget));
        }
        if let Some(effort) = &request.reasoning_effort {
            object.insert("reasoning_effort".into(), json!(effort));
        }
    }

    let agent = crate::chat::llm_http_agent(REQUEST_TIMEOUT);

    let mut request_builder = agent.post(&url);
    if let Some(key) = api_key.map(str::trim).filter(|key| !key.is_empty()) {
        request_builder = request_builder.header("Authorization", &format!("Bearer {key}"));
    }
    let mut response = request_builder
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
        let read = lines
            .read_line(&mut line)
            .map_err(|error| error.to_string())?;
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

fn execute_tool(
    call: &ToolCall,
    skills: &AgentSkills,
    user_skills: &[UserSkill],
) -> Result<String, String> {
    match call.name.as_str() {
        "web_search" => {
            let query = call
                .arguments
                .get("query")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "web_search requires a non-empty \"query\" string.".to_string())?;
            duckduckgo_search(query, skills.web_search_depth)
        }
        "fetch_url" => {
            let url = call
                .arguments
                .get("url")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "fetch_url requires a non-empty \"url\" string.".to_string())?;
            fetch_single_url(url)
        }
        "activate_skill" | "read_skill" => {
            let key = call
                .arguments
                .get("name")
                .or_else(|| call.arguments.get("id"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    "activate_skill requires a non-empty \"name\" (or \"id\") string.".to_string()
                })?;
            let skill = crate::skills::find_skill(user_skills, key).ok_or_else(|| {
                format!("Unknown skill '{key}'. Use a name or id from the available skills list.")
            })?;
            Ok(skill.full_instructions())
        }
        other => Err(format!("Unknown capability '{other}'.")),
    }
}

fn fetch_single_url(url: &str) -> Result<String, String> {
    if !scrapeable_url(url) {
        return Err(
            "fetch_url only supports http(s) pages (not files like PDF, images, or archives)."
                .into(),
        );
    }
    let agent = search_http_agent(PAGE_TIMEOUT);
    let text = fetch_page_text(&agent, url, FETCH_URL_MAX_CHARS)?;
    if text.trim().is_empty() {
        return Err(format!("Fetched {url} but extracted no readable text."));
    }
    Ok(format!(
        "Fetched page text from {url} (up to {FETCH_URL_MAX_CHARS} characters):\n{text}"
    ))
}

#[derive(Debug)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

fn search_http_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        .build()
        .new_agent()
}

fn duckduckgo_search(query: &str, depth: WebSearchDepth) -> Result<String, String> {
    let agent = search_http_agent(SEARCH_TIMEOUT);

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
        return Err(format!("DuckDuckGo returned HTTP {}", response.status()));
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
        return duckduckgo_instant_answer(query, depth);
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
    append_scraped_pages(&mut out, &hits, depth);
    Ok(out)
}

fn duckduckgo_instant_answer(query: &str, depth: WebSearchDepth) -> Result<String, String> {
    let agent = search_http_agent(SEARCH_TIMEOUT);

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

    let mut lines = vec![format!(
        "Web search results for {query:?} (DuckDuckGo Instant Answer):"
    )];
    let mut hits = Vec::new();
    if let Some(text) = body.get("AbstractText").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            let url = body
                .get("AbstractURL")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            lines.push(format!("\nSummary: {text}"));
            if !url.is_empty() {
                lines.push(format!("Source: {url}"));
                hits.push(SearchHit {
                    title: "Abstract".into(),
                    url: url.to_string(),
                    snippet: text.to_string(),
                });
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
                let url = topic.get("FirstURL").and_then(|v| v.as_str()).unwrap_or("");
                count += 1;
                lines.push(format!("\n{count}. {text}"));
                if !url.is_empty() {
                    lines.push(format!("   URL: {url}"));
                    hits.push(SearchHit {
                        title: format!("Related {count}"),
                        url: url.to_string(),
                        snippet: text.to_string(),
                    });
                }
            } else if let Some(nested) = topic.get("Topics").and_then(|v| v.as_array()) {
                for item in nested {
                    if count >= MAX_SEARCH_RESULTS {
                        break;
                    }
                    if let Some(text) = item.get("Text").and_then(|v| v.as_str()) {
                        let url = item.get("FirstURL").and_then(|v| v.as_str()).unwrap_or("");
                        count += 1;
                        lines.push(format!("\n{count}. {text}"));
                        if !url.is_empty() {
                            lines.push(format!("   URL: {url}"));
                            hits.push(SearchHit {
                                title: format!("Related {count}"),
                                url: url.to_string(),
                                snippet: text.to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    if lines.len() == 1 {
        lines.push("\nNo results found.".into());
    }
    let mut out = lines.join("\n");
    append_scraped_pages(&mut out, &hits, depth);
    Ok(out)
}

fn append_scraped_pages(out: &mut String, hits: &[SearchHit], depth: WebSearchDepth) {
    let Some((page_count, max_chars)) = depth.scrape_plan() else {
        return;
    };
    let targets: Vec<&SearchHit> = hits
        .iter()
        .filter(|hit| scrapeable_url(&hit.url))
        .take(page_count)
        .collect();
    if targets.is_empty() {
        return;
    }

    out.push_str(&format!(
        "\n--- Fetched page text (depth: {}, up to {} pages) ---\n",
        depth.label(),
        page_count
    ));

    let agent = search_http_agent(PAGE_TIMEOUT);
    for (index, hit) in targets.iter().enumerate() {
        match fetch_page_text(&agent, &hit.url, max_chars) {
            Ok(text) if !text.trim().is_empty() => {
                out.push_str(&format!(
                    "\n[{}] {} ({})\n{}\n",
                    index + 1,
                    hit.title,
                    hit.url,
                    text
                ));
            }
            Ok(_) => {
                out.push_str(&format!(
                    "\n[{}] {} ({})\n(no extractable text)\n",
                    index + 1,
                    hit.title,
                    hit.url
                ));
            }
            Err(error) => {
                out.push_str(&format!(
                    "\n[{}] {} ({})\n(fetch failed: {error})\n",
                    index + 1,
                    hit.title,
                    hit.url
                ));
            }
        }
    }
}

fn scrapeable_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return false;
    }
    let path = lower.split('?').next().unwrap_or(&lower);
    const SKIP_EXT: &[&str] = &[
        ".pdf", ".zip", ".gz", ".tgz", ".rar", ".7z", ".exe", ".dmg", ".apk", ".mp3", ".mp4",
        ".mov", ".avi", ".mkv", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".css",
        ".js", ".mjs", ".json", ".xml", ".rss", ".atom", ".woff", ".woff2", ".ttf",
    ];
    !SKIP_EXT.iter().any(|ext| path.ends_with(ext))
}

fn fetch_page_text(agent: &ureq::Agent, url: &str, max_chars: usize) -> Result<String, String> {
    let mut response = agent
        .get(url)
        .header(
            "Accept",
            "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        )
        .call()
        .map_err(|error| format!("{error}"))?;

    let status = response.status();
    if status != 200 && status != 203 && status != 206 {
        return Err(format!("HTTP {status}"));
    }

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !content_type.is_empty()
        && !(content_type.contains("text/html")
            || content_type.contains("application/xhtml")
            || content_type.contains("text/plain")
            || content_type.contains("text/xml")
            || content_type.contains("application/xml"))
    {
        return Err(format!("unsupported content-type ({content_type})"));
    }

    let html = response
        .body_mut()
        .with_config()
        .limit(MAX_PAGE_BYTES)
        .read_to_string()
        .map_err(|error| format!("read failed: {error}"))?;

    Ok(html_to_readable_text(&html, max_chars))
}

fn html_to_readable_text(html: &str, max_chars: usize) -> String {
    let mut cleaned = remove_tag_blocks(html, "script");
    cleaned = remove_tag_blocks(&cleaned, "style");
    cleaned = remove_tag_blocks(&cleaned, "noscript");
    cleaned = remove_tag_blocks(&cleaned, "svg");
    cleaned = remove_tag_blocks(&cleaned, "template");

    for marker in [
        "</p>",
        "</div>",
        "</section>",
        "</article>",
        "</li>",
        "</tr>",
        "</h1>",
        "</h2>",
        "</h3>",
        "</h4>",
        "</h5>",
        "</h6>",
        "</blockquote>",
        "<br>",
        "<br/>",
        "<br />",
        "<hr>",
        "<hr/>",
        "<hr />",
    ] {
        cleaned = cleaned.replace(marker, "\n");
        cleaned = cleaned.replace(&marker.to_ascii_uppercase(), "\n");
    }

    let text = strip_tags(&cleaned);
    let mut lines = Vec::new();
    for line in text.lines() {
        let collapsed = collapse_ws(line);
        if collapsed.is_empty() {
            if lines.last().is_some_and(|prev: &String| !prev.is_empty()) {
                lines.push(String::new());
            }
            continue;
        }
        // Drop ultra-common chrome crumbs.
        let lower = collapsed.to_ascii_lowercase();
        if lower == "skip to content"
            || lower == "skip to main content"
            || lower == "advertisement"
            || lower.starts_with("cookie") && lower.len() < 80
        {
            continue;
        }
        lines.push(collapsed);
    }

    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }

    let joined = lines.join("\n");
    truncate_chars(&joined, max_chars)
}

fn remove_tag_blocks(html: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut rest = html;
    let mut out = String::with_capacity(html.len());
    while let Some(start) = find_ignore_ascii_case(rest, &open) {
        out.push_str(&rest[..start]);
        let after_open = &rest[start..];
        match find_ignore_ascii_case(after_open, &close) {
            Some(rel) => rest = &after_open[rel + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

fn find_ignore_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let n = needle.as_bytes();
    let h = haystack.as_bytes();
    if h.len() < n.len() {
        return None;
    }
    'outer: for i in 0..=(h.len() - n.len()) {
        for (a, b) in h[i..i + n.len()].iter().zip(n.iter()) {
            if a.to_ascii_lowercase() != b.to_ascii_lowercase() {
                continue 'outer;
            }
        }
        return Some(i);
    }
    None
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
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
                between(window, ">", "</")
                    .map(strip_tags)
                    .map(|s| collapse_ws(&s))
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

    #[test]
    fn extracts_readable_text_from_html() {
        let html = r#"<html><head><style>.x{color:red}</style><script>evil()</script></head>
        <body><nav>Skip to content</nav><article><h1>Hello</h1><p>World <b>there</b>.</p></article></body></html>"#;
        let text = html_to_readable_text(html, 500);
        assert!(text.contains("Hello"));
        assert!(text.contains("World there."));
        assert!(!text.contains("evil"));
        assert!(!text.contains("color:red"));
    }

    #[test]
    fn scrape_plan_auto_fetches_pages() {
        assert!(WebSearchDepth::Off.scrape_plan().is_none());
        assert_eq!(WebSearchDepth::Auto.scrape_plan(), Some((3, 2800)));
        assert_eq!(WebSearchDepth::Deep.scrape_plan(), Some((6, 4800)));
    }

    #[test]
    fn skips_binary_result_urls() {
        assert!(scrapeable_url("https://example.com/story"));
        assert!(!scrapeable_url("https://example.com/file.pdf"));
        assert!(!scrapeable_url("ftp://example.com/a"));
    }

    #[test]
    fn fetch_url_capability_gated() {
        let off = AgentSkills::default();
        let on = AgentSkills {
            fetch_url: true,
            ..AgentSkills::default()
        };
        assert!(!capability_allowed("fetch_url", &off, &[]));
        assert!(capability_allowed("fetch_url", &on, &[]));
        assert!(on.any_enabled());
        assert!(!off.any_enabled());
    }

    #[test]
    fn fetch_url_rejects_bad_targets() {
        let err = fetch_single_url("ftp://example.com/a").unwrap_err();
        assert!(err.contains("http(s)"));
        let err = fetch_single_url("https://example.com/doc.pdf").unwrap_err();
        assert!(err.contains("http(s)") || err.contains("PDF"));
    }
}
