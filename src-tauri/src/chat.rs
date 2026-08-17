/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Built-in chat console backend.
//!
//! Streams completions from the running llama-server's OpenAI-compatible
//! `/v1/chat/completions` endpoint (SSE) on a worker thread, emitting
//! `chat-delta` events per token and a final `chat-done` with measured decode
//! speed. Routing through Rust keeps the webview free of CORS concerns and all
//! HTTP in one place.
//!
//! Note: the server's root URL serves no web page (LM Studio's llama-server
//! build ships API routes only, so `GET /` is a JSON 404) — this console *is*
//! the UI for talking to the model.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::Emitter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatParams {
    pub temperature: Option<f64>,
    pub top_k: Option<u32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct ChatDelta {
    id: u64,
    content: String,
    /// True when this delta is reasoning/thinking text (reasoning models emit
    /// `reasoning_content` before the final answer).
    reasoning: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ChatDone {
    id: u64,
    tokens: u64,
    decode_tok_s: f64,
    stopped: bool,
    /// Why the server stopped generating, straight from the stream's
    /// `finish_reason`: `"stop"` (hit EOS — a real, complete answer) or
    /// `"length"` (ran into `max_tokens`, or filled the context). `None` means
    /// the stream ended without ever saying why, which is itself a symptom —
    /// the connection dropped or the server died mid-answer.
    ///
    /// Without this the UI cannot tell a finished answer from a guillotined
    /// one: both just stop arriving. Never drop it on the floor.
    finish: Option<String>,
    error: Option<String>,
}

/// Cancel flag for the in-flight generation (one at a time, like the server).
#[derive(Default)]
pub struct ChatState {
    cancel: Mutex<Option<Arc<AtomicBool>>>,
}

impl ChatState {
    /// Arm a fresh cancel flag, cancelling any previous generation.
    fn arm(&self) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        let mut guard = self.cancel.lock().unwrap();
        if let Some(old) = guard.replace(flag.clone()) {
            old.store(true, Ordering::Relaxed);
        }
        flag
    }

    pub fn cancel(&self) {
        if let Some(flag) = self.cancel.lock().unwrap().as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

/// Ask the running model for a short title for a session.
///
/// Deliberately *not* routed through `start_stream`: that path is single-flight
/// and arming it would cancel whatever the user is generating. This is a plain
/// blocking request with its own connection and a tight token budget.
pub fn generate_title(base_url: &str, transcript: &str) -> Result<String, String> {
    let prompt = format!(
        "Summarise this conversation as a title of at most 6 words.\n\
         Reply with the title only — no quotes, no punctuation at the end, no \
         preamble, no explanation.\n\n{transcript}"
    );
    let body = json!({
        "messages": [{ "role": "user", "content": prompt }],
        "stream": false,
        "temperature": 0.3,
        // Naming a chat does not need reasoning, and letting a reasoning model
        // think here is actively harmful: measured on qwen3:14b, thinking ran
        // to 1.2-4k tokens and regularly blew past the cap, returning an empty
        // title. Qwen-family templates honour this switch and drop to zero
        // reasoning tokens; models that ignore it are covered by the generous
        // cap and the reasoning_content fallback below.
        "chat_template_kwargs": { "enable_thinking": false },
        "max_tokens": 2000,
    });

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(90))
        .build();
    let text = agent
        .post(&format!("{base_url}/v1/chat/completions"))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("title request failed: {e}"))?
        .into_string()
        .map_err(|e| e.to_string())?;

    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let msg = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));
    let pick = |key: &str| {
        msg.and_then(|m| m.get(key))
            .and_then(|c| c.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    // A model that ignored `enable_thinking` and then ran out of budget leaves
    // content empty with everything in reasoning_content — still salvageable.
    let raw = pick("content")
        .or_else(|| pick("reasoning_content"))
        .unwrap_or("");
    let title = clean_title(raw);
    if title.is_empty() {
        return Err("model returned no usable title".into());
    }
    Ok(title)
}

/// Take a model's reply and squeeze a usable title out of it. Models pad with
/// quotes, `Title:` prefixes, trailing periods, and reasoning blocks even when
/// told not to — so the last non-empty line is the safest thing to trust.
fn clean_title(raw: &str) -> String {
    let without_think = match (raw.find("</think>"), raw.rfind("</think>")) {
        (Some(_), Some(end)) => &raw[end + "</think>".len()..],
        _ => raw,
    };
    let line = without_think
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .unwrap_or("");
    let line = line
        .trim_start_matches("**")
        .trim_end_matches("**")
        .trim()
        .trim_start_matches(['"', '\'', '“'])
        .trim_end_matches(['"', '\'', '”', '.']);
    let line = line
        .strip_prefix("Title:")
        .or_else(|| line.strip_prefix("title:"))
        .unwrap_or(line)
        .trim();
    // Cap length so a model that ignores the word limit cannot blow up the rail.
    let mut out: String = line.chars().take(70).collect();
    if let Some(i) = out.rfind(char::is_whitespace) {
        if out.chars().count() == 70 {
            out.truncate(i);
        }
    }
    out.trim().to_string()
}

/// Start a streaming generation on a worker thread. Deltas and completion are
/// delivered as window events so the UI stays responsive.
pub fn start_stream(
    window: tauri::Window,
    state: &ChatState,
    base_url: String,
    id: u64,
    messages: Vec<ChatMessage>,
    params: ChatParams,
) {
    let cancel = state.arm();
    std::thread::spawn(move || {
        let done = run_stream(&window, &base_url, id, &messages, &params, &cancel);
        let _ = window.emit("chat-done", done);
    });
}

fn run_stream(
    window: &tauri::Window,
    base_url: &str,
    id: u64,
    messages: &[ChatMessage],
    params: &ChatParams,
    cancel: &AtomicBool,
) -> ChatDone {
    let mut body = json!({
        "messages": messages,
        "stream": true,
    });
    if let Some(v) = params.temperature {
        body["temperature"] = json!(v);
    }
    if let Some(v) = params.top_k {
        body["top_k"] = json!(v);
    }
    if let Some(v) = params.top_p {
        body["top_p"] = json!(v);
    }
    if let Some(v) = params.min_p {
        body["min_p"] = json!(v);
    }
    if let Some(v) = params.max_tokens {
        body["max_tokens"] = json!(v);
    }

    // Per-read timeout only — an overall timeout would cap long generations.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(120))
        .build();

    let resp = match agent
        .post(&format!("{base_url}/v1/chat/completions"))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
    {
        Ok(r) => r,
        Err(e) => {
            return ChatDone {
                id,
                tokens: 0,
                decode_tok_s: 0.0,
                stopped: false,
                finish: None,
                error: Some(format!("request failed: {e}")),
            }
        }
    };

    let reader = BufReader::new(resp.into_reader());
    let mut tokens: u64 = 0;
    let mut first_token: Option<Instant> = None;
    let mut last_token = Instant::now();
    let mut stopped = false;
    let mut finish: Option<String> = None;
    let mut read_err: Option<String> = None;

    for line in reader.lines() {
        if cancel.load(Ordering::Relaxed) {
            stopped = true;
            break; // dropping the reader closes the connection
        }
        // A mid-stream read failure (read timeout, server crash, reset socket)
        // used to `break` silently and report a clean finish, so a truncated
        // answer looked complete. Surface it instead.
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                read_err = Some(format!("stream interrupted after {tokens} tokens: {e}"));
                break;
            }
        };
        let Some(payload) = sse_payload(&line) else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        if let Some(reason) = extract_finish(payload) {
            finish = Some(reason);
        }
        if let Some((content, reasoning)) = extract_delta(payload) {
            tokens += 1;
            let now = Instant::now();
            first_token.get_or_insert(now);
            last_token = now;
            let _ = window.emit(
                "chat-delta",
                ChatDelta {
                    id,
                    content,
                    reasoning,
                },
            );
        }
    }

    // Decode rate over the generation span (first token → last token).
    let decode_tok_s = match first_token {
        Some(first) if tokens > 1 => {
            let secs = last_token.duration_since(first).as_secs_f64();
            if secs > 0.0 {
                (tokens - 1) as f64 / secs
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    ChatDone {
        id,
        tokens,
        decode_tok_s,
        stopped,
        finish,
        error: read_err,
    }
}

/// Extract the payload of an SSE `data:` line, if this line is one.
fn sse_payload(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim)
}

/// Pull `finish_reason` out of a streaming chunk. It rides on the final chunk,
/// whose `delta` is empty — so this is checked separately from `extract_delta`.
fn extract_finish(payload: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    v.get("choices")?
        .get(0)?
        .get("finish_reason")?
        .as_str()
        .map(str::to_string)
}

/// Pull the text delta out of a streaming chunk: `delta.content` (answer) or
/// `delta.reasoning_content` (thinking, on reasoning models). Role-only and
/// finish chunks carry neither. Returns (text, is_reasoning).
fn extract_delta(payload: &str) -> Option<(String, bool)> {
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    let delta = v.get("choices")?.get(0)?.get("delta")?;
    let take = |key: &str| {
        delta
            .get(key)
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    if let Some(content) = take("content") {
        return Some((content, false));
    }
    if let Some(thinking) = take("reasoning_content") {
        return Some((thinking, true));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_up_model_titles() {
        // Plain answer.
        assert_eq!(clean_title("Todo app design"), "Todo app design");
        // Padding models add despite instructions.
        assert_eq!(clean_title("\"Todo app design\""), "Todo app design");
        assert_eq!(clean_title("Title: Todo app design."), "Todo app design");
        assert_eq!(clean_title("**Todo app design**"), "Todo app design");
        // Reasoning models emit a think block, then the title.
        assert_eq!(
            clean_title("<think>weighing options</think>
Todo app design"),
            "Todo app design"
        );
        // Preamble lines before the answer: trust the last one.
        assert_eq!(
            clean_title("Sure, here is a title:
Todo app design"),
            "Todo app design"
        );
        // Runaway output is capped without cutting mid-word.
        let long = clean_title(&"word ".repeat(40));
        assert!(long.chars().count() <= 70);
        assert!(!long.ends_with("wor"));
        // Nothing usable.
        assert_eq!(clean_title("   "), "");
    }

    #[test]
    fn parses_sse_data_lines() {
        assert_eq!(sse_payload("data: {\"x\":1}"), Some("{\"x\":1}"));
        assert_eq!(sse_payload("data:[DONE]"), Some("[DONE]"));
        assert_eq!(sse_payload(": comment"), None);
        assert_eq!(sse_payload(""), None);
    }

    #[test]
    fn extracts_content_delta() {
        let chunk = r#"{"choices":[{"delta":{"content":"Hel"},"index":0}]}"#;
        assert_eq!(extract_delta(chunk), Some(("Hel".to_string(), false)));
        // Reasoning models stream thinking under reasoning_content.
        let think = r#"{"choices":[{"delta":{"reasoning_content":"hmm"},"index":0}]}"#;
        assert_eq!(extract_delta(think), Some(("hmm".to_string(), true)));
        // Role-only first chunk carries no content.
        let role = r#"{"choices":[{"delta":{"role":"assistant"},"index":0}]}"#;
        assert_eq!(extract_delta(role), None);
        // Finish chunk with empty delta.
        let fin = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(extract_delta(fin), None);
    }

    #[test]
    fn extracts_finish_reason() {
        // The reason rides on a chunk whose delta is empty — the one case
        // extract_delta ignores, which is exactly why it needs its own parse.
        let cut = r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#;
        assert_eq!(extract_finish(cut), Some("length".to_string()));
        assert_eq!(extract_delta(cut), None);

        let done = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(extract_finish(done), Some("stop".to_string()));

        // Ordinary content chunks carry no reason (it is JSON null until the end).
        let mid = r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
        assert_eq!(extract_finish(mid), None);
        assert_eq!(extract_delta(mid), Some(("hi".to_string(), false)));
    }

    /// Real end-to-end chat stream against the 4B model. Ignored by default;
    /// run with: cargo test -- --ignored --nocapture chat_real_stream
    #[test]
    #[ignore]
    fn chat_real_stream() {
        use crate::llama::{LlamaManager, LlamaServerConfig};

        let Some(home) = dirs::home_dir() else { return };
        let model = home.join(".lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf");
        if !model.is_file() {
            eprintln!("model not present, skipping");
            return;
        }

        let mgr = LlamaManager::new();
        mgr.start(LlamaServerConfig {
            model_path: model.to_string_lossy().into_owned(),
            n_gpu_layers: Some(999),
            ctx_size: Some(4096),
            port: 8141,
            binary_path: None,
            flash_attn: false,
            cache_type_k: None,
            cache_type_v: None,
            context_shift: false,
            extra_args: vec![],
            ..Default::default()
        })
        .expect("start");
        for _ in 0..60 {
            if mgr.status().health == "ok" {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        assert_eq!(mgr.status().health, "ok");

        // Drive the same request path the command uses (minus the window).
        let body = json!({
            "messages": [ChatMessage { role: "user".into(), content: "Say hello in five words.".into() }],
            "stream": true, "max_tokens": 32,
        });
        let resp = ureq::post("http://127.0.0.1:8141/v1/chat/completions")
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .expect("chat request");
        let reader = BufReader::new(resp.into_reader());
        let mut text = String::new();
        let mut chunks = 0;
        for line in reader.lines().map_while(Result::ok) {
            let Some(p) = sse_payload(&line) else { continue };
            if p == "[DONE]" {
                break;
            }
            if let Some((c, _reasoning)) = extract_delta(p) {
                chunks += 1;
                text.push_str(&c);
            }
        }
        println!("streamed {chunks} chunks: {text:?}");
        assert!(chunks > 0, "should stream at least one delta");
        assert!(!text.trim().is_empty(), "should produce text");

        mgr.stop().expect("stop");
    }
}
