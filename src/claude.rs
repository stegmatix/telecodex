//! Claude Code CLI backend: `claude -p --output-format stream-json`.

use serde_json::Value;
use uuid::Uuid;

use crate::{
    agent::{
        AgentSessionState, CliTurnCommand, StreamOutcome, StreamParser, describe_tool_call,
        merged_instructions,
    },
    codex::CodexEvent,
    config::ClaudeConfig,
    models::{SessionRecord, TurnRequest},
};

#[derive(Clone)]
pub struct ClaudeRunner {
    config: ClaudeConfig,
}

impl ClaudeRunner {
    pub fn new(config: ClaudeConfig) -> Self {
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
            args: build_claude_args(&self.config, session, state, request, &new_session_id),
            current_dir: session.cwd.clone(),
            stdin: request.prompt.trim().to_string(),
        }
    }
}

pub(crate) fn build_claude_args(
    config: &ClaudeConfig,
    session: &SessionRecord,
    state: &AgentSessionState,
    request: &TurnRequest,
    new_session_id: &str,
) -> Vec<String> {
    let mut args = [
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    match state.thread_id.as_deref() {
        Some(thread_id) => push_pair(&mut args, "--resume", thread_id),
        None => push_pair(&mut args, "--session-id", new_session_id),
    }
    if let Some(model) = state.model.as_deref().or(config.default_model.as_deref()) {
        push_pair(&mut args, "--model", model);
    }
    if let Some(effort) = session
        .reasoning_effort
        .as_deref()
        .and_then(claude_effort_level)
    {
        push_pair(&mut args, "--effort", effort);
    }
    let permission_mode = config
        .permission_mode
        .as_deref()
        .unwrap_or_else(|| claude_permission_mode_for_sandbox(&session.sandbox_mode));
    push_pair(&mut args, "--permission-mode", permission_mode);
    if let Some(instructions) = merged_instructions(session, request) {
        push_pair(&mut args, "--append-system-prompt", &instructions);
    }
    for dir in &session.add_dirs {
        push_pair(&mut args, "--add-dir", &dir.display().to_string());
    }
    args.extend(config.extra_args.iter().cloned());
    args
}

fn push_pair(args: &mut Vec<String>, flag: &str, value: &str) {
    args.push(flag.to_string());
    args.push(value.to_string());
}

fn claude_permission_mode_for_sandbox(sandbox_mode: &str) -> &'static str {
    match sandbox_mode {
        "danger-full-access" => "bypassPermissions",
        "workspace-write" => "acceptEdits",
        _ => "default",
    }
}

/// Maps Telecodex `/think` levels onto `claude --effort` values.
fn claude_effort_level(level: &str) -> Option<&'static str> {
    match level {
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" => Some("max"),
        _ => None,
    }
}

/// Parser for Claude Code `stream-json` (with `--include-partial-messages`).
#[derive(Debug, Default)]
pub(crate) struct ClaudeStreamParser {
    session_id: Option<String>,
    announced: bool,
    message_text: String,
    last_text: String,
    outcome: Option<StreamOutcome>,
}

impl StreamParser for ClaudeStreamParser {
    fn parse_line(&mut self, line: &str) -> Vec<CodexEvent> {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            tracing::debug!("ignoring non-json claude stdout line: {line}");
            return Vec::new();
        };
        let mut events = Vec::new();
        if let Some(session_id) = value.get("session_id").and_then(Value::as_str) {
            self.session_id = Some(session_id.to_string());
            if !self.announced {
                self.announced = true;
                events.push(CodexEvent::ThreadStarted(session_id.to_string()));
            }
        }
        // Sub-agent traffic carries a parent tool id; only the top-level
        // conversation is shown to the Telegram user.
        let is_top_level = value
            .get("parent_tool_use_id")
            .is_none_or(serde_json::Value::is_null);
        match value.get("type").and_then(Value::as_str) {
            Some("stream_event") if is_top_level => {
                let event = value.get("event").unwrap_or(&Value::Null);
                match event.get("type").and_then(Value::as_str) {
                    Some("message_start") => self.message_text.clear(),
                    Some("content_block_delta") => {
                        let delta = event.get("delta").unwrap_or(&Value::Null);
                        if delta.get("type").and_then(Value::as_str) == Some("text_delta") {
                            if let Some(text) = delta.get("text").and_then(Value::as_str) {
                                self.message_text.push_str(text);
                                events.push(CodexEvent::AssistantText(self.message_text.clone()));
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some("assistant") if is_top_level => {
                let content = value
                    .get("message")
                    .and_then(|message| message.get("content"))
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let text = content
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<String>();
                if !text.trim().is_empty() {
                    self.message_text = text.clone();
                    self.last_text = text.clone();
                    events.push(CodexEvent::AssistantText(text));
                }
                for block in content
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
                {
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let input = block.get("input").unwrap_or(&Value::Null);
                    events.push(CodexEvent::Progress(describe_tool_call(name, input)));
                }
            }
            Some("result") => {
                let failed = value.get("is_error").and_then(Value::as_bool) == Some(true)
                    || value
                        .get("subtype")
                        .and_then(Value::as_str)
                        .is_some_and(|subtype| subtype != "success");
                let result_text = value
                    .get("result")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_string);
                if failed {
                    let message = result_text
                        .or_else(|| claude_error_list(&value))
                        .or_else(|| {
                            value
                                .get("subtype")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .unwrap_or_else(|| "Claude reported an error".to_string());
                    self.outcome = Some(StreamOutcome::Failed(message));
                } else {
                    let final_text = result_text.unwrap_or_else(|| self.last_text.clone());
                    if !final_text.is_empty() {
                        events.push(CodexEvent::AssistantText(final_text.clone()));
                    }
                    self.outcome = Some(StreamOutcome::Success(final_text));
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

fn claude_error_list(value: &Value) -> Option<String> {
    let errors = value
        .get("errors")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SearchMode;
    use std::path::PathBuf;

    const SIMPLE_FIXTURE: &str = include_str!("../tests/fixtures/claude_stream_simple.jsonl");
    const TOOLS_FIXTURE: &str = include_str!("../tests/fixtures/claude_stream_tools.jsonl");
    const ERROR_FIXTURE: &str = include_str!("../tests/fixtures/claude_stream_error.jsonl");

    fn parse_fixture(fixture: &str) -> (ClaudeStreamParser, Vec<CodexEvent>) {
        let mut parser = ClaudeStreamParser::default();
        let events = fixture
            .lines()
            .flat_map(|line| parser.parse_line(line))
            .collect();
        (parser, events)
    }

    fn session() -> SessionRecord {
        SessionRecord {
            id: 7,
            key: crate::models::SessionKey {
                chat_id: 1,
                thread_id: 0,
            },
            session_title: None,
            codex_thread_id: Some("codex-thread".to_string()),
            force_fresh_thread: false,
            updated_at: "2026-11-21T00:00:00Z".to_string(),
            cwd: PathBuf::from("/srv/helen"),
            model: Some("gpt-6-astra".to_string()),
            reasoning_effort: Some("minimal".to_string()),
            service_tier: None,
            session_prompt: Some("Be brief.".to_string()),
            sandbox_mode: "danger-full-access".to_string(),
            approval_policy: "never".to_string(),
            search_mode: SearchMode::Disabled,
            add_dirs: vec![PathBuf::from("/srv/extra")],
            busy: false,
        }
    }

    fn request() -> TurnRequest {
        TurnRequest {
            session_key: session().key,
            from_user_id: 1,
            prompt: "--looks like a flag".to_string(),
            runtime_instructions: Some("Save files in /out.".to_string()),
            attachments: vec![],
            review_mode: None,
            override_search_mode: None,
            guest_query_id: None,
            guest_inline_message_id: None,
        }
    }

    fn config() -> ClaudeConfig {
        toml::from_str("binary = \"claude\"").unwrap()
    }

    #[test]
    fn parses_real_text_turn() {
        let (parser, events) = parse_fixture(SIMPLE_FIXTURE);
        assert_eq!(
            parser.thread_id().as_deref(),
            Some("2208f70e-6dd7-45e6-973b-642e417edeb0")
        );
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Success("pong".to_string()))
        );
        assert!(
            matches!(&events[0], CodexEvent::ThreadStarted(id) if id == "2208f70e-6dd7-45e6-973b-642e417edeb0")
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CodexEvent::ThreadStarted(_)))
                .count(),
            1
        );
        assert!(matches!(events.last(), Some(CodexEvent::AssistantText(text)) if text == "pong"));
    }

    #[test]
    fn parses_tool_turn_and_ignores_subagent_text() {
        let (parser, events) = parse_fixture(TOOLS_FIXTURE);
        let progress = events
            .iter()
            .filter_map(|event| match event {
                CodexEvent::Progress(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(progress, vec!["Running `echo hi`", "Task"]);
        assert!(!events.iter().any(
            |event| matches!(event, CodexEvent::AssistantText(text) if text.contains("subagent"))
        ));
        assert!(events.iter().any(
            |event| matches!(event, CodexEvent::AssistantText(text) if text == "Let me check.")
        ));
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Success(
                "The command printed: hi".to_string()
            ))
        );
    }

    #[test]
    fn reports_error_results() {
        let (parser, _) = parse_fixture(ERROR_FIXTURE);
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Failed(
                "No conversation found with session ID: 0199".to_string()
            ))
        );
    }

    #[test]
    fn builds_new_session_args() {
        let args = build_claude_args(
            &config(),
            &session(),
            &AgentSessionState::default(),
            &request(),
            "new-id",
        );
        let joined = args.join(" ");
        assert!(joined.starts_with("-p --output-format stream-json --verbose"));
        assert!(joined.contains("--session-id new-id"));
        assert!(!joined.contains("--resume"));
        assert!(joined.contains("--effort low"));
        assert!(joined.contains("--permission-mode bypassPermissions"));
        assert!(joined.contains("--append-system-prompt Be brief.\n\nSave files in /out."));
        assert!(joined.contains("--add-dir /srv/extra"));
        assert!(!joined.contains("--model"), "codex model must not leak");
        assert!(
            !joined.contains("looks like a flag"),
            "prompt goes to stdin"
        );
    }

    #[test]
    fn builds_resume_args_with_vendor_model() {
        let state = AgentSessionState {
            thread_id: Some("abc".to_string()),
            model: Some("opus".to_string()),
        };
        let mut config = config();
        config.permission_mode = Some("auto".to_string());
        config.extra_args = vec!["--max-budget-usd".to_string(), "5".to_string()];
        let args = build_claude_args(&config, &session(), &state, &request(), "new-id");
        let joined = args.join(" ");
        assert!(joined.contains("--resume abc"));
        assert!(!joined.contains("--session-id"));
        assert!(joined.contains("--model opus"));
        assert!(joined.contains("--permission-mode auto"));
        assert!(joined.ends_with("--max-budget-usd 5"));
    }

    #[test]
    fn command_sends_prompt_on_stdin_in_session_cwd() {
        let runner = ClaudeRunner::new(config());
        let command = runner.build_command(&session(), &AgentSessionState::default(), &request());
        assert_eq!(command.stdin, "--looks like a flag");
        assert_eq!(command.current_dir, PathBuf::from("/srv/helen"));
    }

    /// Live check against the installed CLI: `cargo test live_claude -- --ignored`.
    #[tokio::test]
    #[ignore = "spawns the real claude CLI and spends a few tokens"]
    async fn live_claude_new_then_resumed_turn() {
        use crate::agent::{AgentRunners, AgentVendor};
        let workspace = tempfile::tempdir().unwrap();
        let mut session = session();
        session.cwd = workspace.path().to_path_buf();
        session.add_dirs.clear();
        session.session_prompt = None;
        session.sandbox_mode = "read-only".to_string();
        session.reasoning_effort = None;
        let mut runners = AgentRunners::for_tests();
        let mut claude_config = config();
        claude_config.default_model = Some("haiku".to_string());
        claude_config.extra_args = vec!["--tools".to_string(), String::new()];
        runners.claude = ClaudeRunner::new(claude_config);

        let turn = |prompt: &str, state: AgentSessionState| {
            let mut request = request();
            request.prompt = prompt.to_string();
            request.runtime_instructions = None;
            let runners = runners.clone();
            let session = session.clone();
            async move {
                runners
                    .run_turn(
                        AgentVendor::Claude,
                        &session,
                        &state,
                        &request,
                        tokio_util::sync::CancellationToken::new(),
                        |_| async { Ok(crate::codex::CodexEventOutcome::None) },
                    )
                    .await
                    .unwrap()
            }
        };
        let first = turn("Reply with exactly: pong", AgentSessionState::default()).await;
        assert_eq!(first.assistant_text.trim(), "pong");
        let thread_id = first.thread_id.clone().expect("claude session id");
        let second = turn(
            "What exact word did you reply with last time? Reply with just that word.",
            AgentSessionState {
                thread_id: Some(thread_id.clone()),
                model: None,
            },
        )
        .await;
        assert_eq!(second.thread_id.as_deref(), Some(thread_id.as_str()));
        assert!(second.assistant_text.to_lowercase().contains("pong"));
    }
}
