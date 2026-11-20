//! Gemini CLI backend: headless `gemini --output-format stream-json`.

use serde_json::Value;
use uuid::Uuid;

use crate::{
    agent::{
        AgentSessionState, CliTurnCommand, StreamOutcome, StreamParser, describe_tool_call,
        merged_instructions,
    },
    codex::CodexEvent,
    config::GeminiConfig,
    models::{SessionRecord, TurnRequest},
};

#[derive(Clone)]
pub struct GeminiRunner {
    config: GeminiConfig,
}

impl GeminiRunner {
    pub fn new(config: GeminiConfig) -> Self {
        Self { config }
    }

    pub fn models(&self) -> &[String] {
        &self.config.models
    }

    pub fn default_model(&self) -> Option<&str> {
        self.config.default_model.as_deref()
    }

    pub(crate) fn build_command(
        &self,
        session: &SessionRecord,
        state: &AgentSessionState,
        request: &TurnRequest,
    ) -> CliTurnCommand {
        let new_session_id = Uuid::now_v7().to_string();
        CliTurnCommand {
            program: self.config.binary.clone(),
            args: build_gemini_args(&self.config, session, state, &new_session_id),
            current_dir: session.cwd.clone(),
            stdin: gemini_stdin(session, request),
        }
    }
}

/// Gemini has no system-prompt flag, so instructions are prepended to the
/// prompt it reads from stdin (stdin that is not a TTY selects headless mode).
fn gemini_stdin(session: &SessionRecord, request: &TurnRequest) -> String {
    let prompt = request.prompt.trim();
    match merged_instructions(session, request) {
        Some(instructions) => {
            format!("<instructions>\n{instructions}\n</instructions>\n\n{prompt}")
        }
        None => prompt.to_string(),
    }
}

pub(crate) fn build_gemini_args(
    config: &GeminiConfig,
    session: &SessionRecord,
    state: &AgentSessionState,
    new_session_id: &str,
) -> Vec<String> {
    let mut args = vec!["--output-format".to_string(), "stream-json".to_string()];
    if config.skip_trust {
        args.push("--skip-trust".to_string());
    }
    match state.thread_id.as_deref() {
        Some(thread_id) => push_pair(&mut args, "--resume", thread_id),
        None => push_pair(&mut args, "--session-id", new_session_id),
    }
    if let Some(model) = state.model.as_deref().or(config.default_model.as_deref()) {
        push_pair(&mut args, "--model", model);
    }
    let approval_mode = config
        .approval_mode
        .as_deref()
        .unwrap_or_else(|| gemini_approval_mode_for_sandbox(&session.sandbox_mode));
    push_pair(&mut args, "--approval-mode", approval_mode);
    for dir in &session.add_dirs {
        push_pair(
            &mut args,
            "--include-directories",
            &dir.display().to_string(),
        );
    }
    args.extend(config.extra_args.iter().cloned());
    args
}

fn push_pair(args: &mut Vec<String>, flag: &str, value: &str) {
    args.push(flag.to_string());
    args.push(value.to_string());
}

fn gemini_approval_mode_for_sandbox(sandbox_mode: &str) -> &'static str {
    match sandbox_mode {
        "danger-full-access" => "yolo",
        "workspace-write" => "auto_edit",
        _ => "default",
    }
}

/// Parser for Gemini CLI `stream-json` events
/// (`init`, `message`, `tool_use`, `tool_result`, `error`, `result`).
#[derive(Debug, Default)]
pub(crate) struct GeminiStreamParser {
    session_id: Option<String>,
    segment: String,
    last_text: String,
    last_error: Option<String>,
    outcome: Option<StreamOutcome>,
}

impl StreamParser for GeminiStreamParser {
    fn parse_line(&mut self, line: &str) -> Vec<CodexEvent> {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            tracing::debug!("ignoring non-json gemini stdout line: {line}");
            return Vec::new();
        };
        let mut events = Vec::new();
        match value.get("type").and_then(Value::as_str) {
            Some("init") => {
                if let Some(session_id) = value.get("session_id").and_then(Value::as_str) {
                    self.session_id = Some(session_id.to_string());
                    events.push(CodexEvent::ThreadStarted(session_id.to_string()));
                }
            }
            Some("message") if value.get("role").and_then(Value::as_str) == Some("assistant") => {
                let content = value
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if value.get("delta").and_then(Value::as_bool) == Some(true) {
                    self.segment.push_str(content);
                } else {
                    self.segment = content.to_string();
                }
                if !self.segment.trim().is_empty() {
                    events.push(CodexEvent::AssistantText(self.segment.clone()));
                }
            }
            Some("tool_use") => {
                // Text after the last tool call is the answer; keep earlier
                // narration only as a fallback.
                if !self.segment.trim().is_empty() {
                    self.last_text = std::mem::take(&mut self.segment);
                }
                self.segment.clear();
                let name = value
                    .get("tool_name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                let parameters = value.get("parameters").unwrap_or(&Value::Null);
                events.push(CodexEvent::Progress(describe_tool_call(name, parameters)));
            }
            Some("tool_result") if value.get("status").and_then(Value::as_str) == Some("error") => {
                let message = value
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("tool error");
                events.push(CodexEvent::Progress(format!("Tool failed: {message}")));
            }
            Some("error") => {
                if let Some(message) = value.get("message").and_then(Value::as_str) {
                    self.last_error = Some(message.to_string());
                    events.push(CodexEvent::Progress(format!("⚠️ {message}")));
                }
            }
            Some("result") => {
                if value.get("status").and_then(Value::as_str) == Some("success") {
                    let final_text = if self.segment.trim().is_empty() {
                        self.last_text.trim().to_string()
                    } else {
                        self.segment.trim().to_string()
                    };
                    if !final_text.is_empty() {
                        events.push(CodexEvent::AssistantText(final_text.clone()));
                    }
                    self.outcome = Some(StreamOutcome::Success(final_text));
                } else {
                    let message = value
                        .get("error")
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or_else(|| self.last_error.clone())
                        .unwrap_or_else(|| "Gemini reported an error".to_string());
                    self.outcome = Some(StreamOutcome::Failed(message));
                }
            }
            _ => {}
        }
        events
    }

    fn thread_id(&self) -> Option<String> {
        self.session_id.clone()
    }

    fn outcome(&self) -> Option<StreamOutcome> {
        self.outcome.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SearchMode;
    use std::path::PathBuf;

    const TOOLS_FIXTURE: &str = include_str!("../tests/fixtures/gemini_stream_tools.jsonl");
    const ERROR_FIXTURE: &str = include_str!("../tests/fixtures/gemini_stream_error.jsonl");

    fn parse_fixture(fixture: &str) -> (GeminiStreamParser, Vec<CodexEvent>) {
        let mut parser = GeminiStreamParser::default();
        let events = fixture
            .lines()
            .flat_map(|line| parser.parse_line(line))
            .collect();
        (parser, events)
    }

    fn session(sandbox: &str) -> SessionRecord {
        SessionRecord {
            id: 9,
            key: crate::models::SessionKey {
                chat_id: 1,
                thread_id: 0,
            },
            session_title: None,
            codex_thread_id: None,
            force_fresh_thread: false,
            updated_at: "2026-11-21T00:00:00Z".to_string(),
            cwd: PathBuf::from("/srv/helen"),
            model: Some("gpt-6-astra".to_string()),
            reasoning_effort: None,
            service_tier: None,
            session_prompt: None,
            sandbox_mode: sandbox.to_string(),
            approval_policy: "never".to_string(),
            search_mode: SearchMode::Disabled,
            add_dirs: vec![PathBuf::from("/srv/extra")],
            busy: false,
        }
    }

    fn config() -> GeminiConfig {
        toml::from_str("binary = \"gemini\"").unwrap()
    }

    #[test]
    fn parses_tool_turn() {
        let (parser, events) = parse_fixture(TOOLS_FIXTURE);
        assert_eq!(
            parser.thread_id().as_deref(),
            Some("7d4c2b7e-1d3a-4c55-9a0e-2b8f5e0c1a11")
        );
        assert!(matches!(&events[0], CodexEvent::ThreadStarted(_)));
        assert!(events.iter().any(
            |event| matches!(event, CodexEvent::Progress(text) if text == "Running `echo hi`")
        ));
        assert!(events.iter().any(
            |event| matches!(event, CodexEvent::Progress(text) if text.starts_with("Tool failed"))
        ));
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Success(
                "The command printed hi.".to_string()
            ))
        );
        assert!(matches!(
            events.last(),
            Some(CodexEvent::AssistantText(text)) if text == "The command printed hi."
        ));
    }

    #[test]
    fn reports_error_result() {
        let (parser, events) = parse_fixture(ERROR_FIXTURE);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, CodexEvent::Progress(text) if text.contains("quota")))
        );
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Failed(
                "Invalid session identifier \"0199\".".to_string()
            ))
        );
    }

    #[test]
    fn error_result_without_payload_uses_last_error_event() {
        let mut parser = GeminiStreamParser::default();
        parser.parse_line(r#"{"type":"error","severity":"error","message":"Quota exhausted"}"#);
        parser.parse_line(r#"{"type":"result","status":"error","stats":{}}"#);
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Failed("Quota exhausted".to_string()))
        );
    }

    #[test]
    fn builds_args_for_new_and_resumed_sessions() {
        let args = build_gemini_args(
            &config(),
            &session("danger-full-access"),
            &AgentSessionState::default(),
            "new-id",
        )
        .join(" ");
        assert!(args.starts_with("--output-format stream-json --skip-trust"));
        assert!(args.contains("--session-id new-id"));
        assert!(args.contains("--approval-mode yolo"));
        assert!(args.contains("--include-directories /srv/extra"));
        assert!(!args.contains("--model"));

        let state = AgentSessionState {
            thread_id: Some("abc".to_string()),
            model: Some("gemini-2.5-flash".to_string()),
        };
        let args = build_gemini_args(&config(), &session("read-only"), &state, "new-id").join(" ");
        assert!(args.contains("--resume abc"));
        assert!(!args.contains("--session-id"));
        assert!(args.contains("--model gemini-2.5-flash"));
        assert!(args.contains("--approval-mode default"));
    }

    #[test]
    fn prepends_instructions_to_stdin_prompt() {
        let mut session = session("workspace-write");
        session.session_prompt = Some("Be brief.".to_string());
        let request = TurnRequest {
            session_key: session.key,
            from_user_id: 1,
            prompt: " hello ".to_string(),
            runtime_instructions: None,
            attachments: vec![],
            review_mode: None,
            override_search_mode: None,
            guest_query_id: None,
            guest_inline_message_id: None,
        };
        assert_eq!(
            gemini_stdin(&session, &request),
            "<instructions>\nBe brief.\n</instructions>\n\nhello"
        );
        session.session_prompt = None;
        assert_eq!(gemini_stdin(&session, &request), "hello");
    }
}
