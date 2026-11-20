//! Antigravity CLI backend: `agy --output-format stream-json --input-format stream-json`.
//!
//! Replaces the Gemini CLI, which Google no longer serves to personal accounts.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::{
    agent::{
        AgentSessionState, CliTurnCommand, StreamOutcome, StreamParser, describe_tool_call,
        merged_instructions,
    },
    codex::{CodexEvent, CommandSpec, spawnable_command},
    config::AntigravityConfig,
    models::{SessionRecord, TurnRequest},
};

const MODELS_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct AntigravityRunner {
    config: AntigravityConfig,
}

impl AntigravityRunner {
    pub fn new(config: AntigravityConfig) -> Self {
        Self { config }
    }

    pub fn models(&self) -> &[String] {
        &self.config.models
    }

    pub fn default_model(&self) -> Option<&str> {
        self.config.default_model.as_deref()
    }

    /// Live model ids from `agy models`.
    pub async fn read_models(&self) -> Result<Vec<String>> {
        let spec = CommandSpec {
            program: self.config.binary.clone(),
            args: vec!["models".to_string()],
            current_dir: None,
        };
        let mut command = spawnable_command(&spec);
        command
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        let output = tokio::time::timeout(MODELS_TIMEOUT, command.output())
            .await
            .context("agy models timed out")?
            .context("failed to run agy models")?;
        if !output.status.success() {
            bail!(
                "agy models exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let models = parse_model_list(&String::from_utf8_lossy(&output.stdout));
        if models.is_empty() {
            bail!("agy models returned no models");
        }
        Ok(models)
    }

    pub(crate) fn build_command(
        &self,
        session: &SessionRecord,
        state: &AgentSessionState,
        request: &TurnRequest,
    ) -> CliTurnCommand {
        CliTurnCommand {
            program: self.config.binary.clone(),
            args: build_antigravity_args(&self.config, session, state),
            current_dir: session.cwd.clone(),
            stdin: antigravity_stdin(session, request),
        }
    }
}

/// `agy models` prints `<id>\t<label>` rows after a progress line.
fn parse_model_list(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(id, _)| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect()
}

/// `agy` has no system-prompt flag and its `--print` value cannot come from
/// stdin, so the turn is sent as one NDJSON user message on stdin; session
/// instructions are prepended to the prompt.
fn antigravity_stdin(session: &SessionRecord, request: &TurnRequest) -> String {
    let prompt = request.prompt.trim();
    let content = match merged_instructions(session, request) {
        Some(instructions) => {
            format!("<instructions>\n{instructions}\n</instructions>\n\n{prompt}")
        }
        None => prompt.to_string(),
    };
    let message = json!({
        "event": "user",
        "message": {"role": "user", "content": content},
    });
    format!("{message}\n")
}

pub(crate) fn build_antigravity_args(
    config: &AntigravityConfig,
    session: &SessionRecord,
    state: &AgentSessionState,
) -> Vec<String> {
    let mut args = [
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    // agy assigns conversation ids itself; an unknown id starts a new one.
    if let Some(thread_id) = state.thread_id.as_deref() {
        push_pair(&mut args, "--conversation", thread_id);
    }
    if let Some(model) = state.model.as_deref().or(config.default_model.as_deref()) {
        push_pair(&mut args, "--model", model);
    }
    let permission_mode = config
        .permission_mode
        .as_deref()
        .unwrap_or_else(|| antigravity_permission_for_sandbox(&session.sandbox_mode));
    match permission_mode {
        "skip-permissions" => args.push("--dangerously-skip-permissions".to_string()),
        "default" => {}
        mode => push_pair(&mut args, "--mode", mode),
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

fn antigravity_permission_for_sandbox(sandbox_mode: &str) -> &'static str {
    match sandbox_mode {
        "danger-full-access" => "skip-permissions",
        "workspace-write" => "accept-edits",
        _ => "default",
    }
}

/// Parser for `agy` stream-json events (`init`, `step_update`, `result`).
#[derive(Debug, Default)]
pub(crate) struct AntigravityStreamParser {
    conversation_id: Option<String>,
    announced: bool,
    response_step: Option<i64>,
    segment: String,
    last_text: String,
    outcome: Option<StreamOutcome>,
}

impl AntigravityStreamParser {
    fn remember_conversation(&mut self, id: Option<&str>, events: &mut Vec<CodexEvent>) {
        let Some(id) = id.map(str::trim).filter(|id| !id.is_empty()) else {
            return;
        };
        self.conversation_id = Some(id.to_string());
        if !self.announced {
            self.announced = true;
            events.push(CodexEvent::ThreadStarted(id.to_string()));
        }
    }

    fn handle_step(&mut self, step: &Value, events: &mut Vec<CodexEvent>) {
        let state = step
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match step.get("step_type").and_then(Value::as_str) {
            Some("agent_response") => {
                let index = step.get("step_index").and_then(Value::as_i64);
                if index != self.response_step {
                    self.response_step = index;
                    if !self.segment.trim().is_empty() {
                        self.last_text = std::mem::take(&mut self.segment);
                    }
                    self.segment.clear();
                }
                if let Some(delta) = step.get("text_delta").and_then(Value::as_str) {
                    self.segment.push_str(delta);
                    if !self.segment.trim().is_empty() {
                        events.push(CodexEvent::AssistantText(
                            self.segment.trim_end().to_string(),
                        ));
                    }
                }
            }
            Some("tool") => {
                let info = step.get("tool_info").unwrap_or(&Value::Null);
                let name = step
                    .get("tool_name")
                    .or_else(|| info.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                match state {
                    "ACTIVE" => {
                        let parameters = info.get("parameters").unwrap_or(&Value::Null);
                        events.push(CodexEvent::Progress(describe_tool_call(name, parameters)));
                    }
                    "ERROR" => {
                        let message = info
                            .get("error")
                            .and_then(|error| error.get("message"))
                            .and_then(Value::as_str)
                            .and_then(|message| message.lines().next())
                            .unwrap_or("tool error");
                        events.push(CodexEvent::Progress(format!("{name} failed: {message}")));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn handle_result(&mut self, result: &Value, events: &mut Vec<CodexEvent>) {
        let status = result.get("status").and_then(Value::as_str).unwrap_or("");
        if status != "SUCCESS" {
            let message = result
                .get("error")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .unwrap_or("Antigravity reported an error");
            self.outcome = Some(StreamOutcome::Failed(message.to_string()));
            return;
        }
        let response = result
            .get("response")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if self.segment.trim().is_empty() {
                    self.last_text.trim().to_string()
                } else {
                    self.segment.trim().to_string()
                }
            });
        if response.is_empty() {
            if let Some(denied) = denied_actions(result) {
                self.outcome = Some(StreamOutcome::Failed(format!(
                    "Antigravity was not allowed to use {denied} in headless mode; set `[antigravity] permission_mode` or the session `/sandbox` to allow it"
                )));
                return;
            }
        } else {
            events.push(CodexEvent::AssistantText(response.clone()));
        }
        self.outcome = Some(StreamOutcome::Success(response));
    }
}

fn denied_actions(result: &Value) -> Option<String> {
    let names = result
        .get("denied_actions")?
        .as_array()?
        .iter()
        .filter_map(|action| {
            action
                .get("display_name")
                .or_else(|| action.get("action"))
                .and_then(Value::as_str)
        })
        .collect::<Vec<_>>();
    if names.is_empty() {
        None
    } else {
        Some(names.join(", "))
    }
}

impl StreamParser for AntigravityStreamParser {
    fn parse_line(&mut self, line: &str) -> Vec<CodexEvent> {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            tracing::debug!("ignoring non-json agy stdout line: {line}");
            return Vec::new();
        };
        let mut events = Vec::new();
        match value.get("event").and_then(Value::as_str) {
            Some("init") => {
                let id = value.get("conversation_id").and_then(Value::as_str);
                self.remember_conversation(id, &mut events);
            }
            Some("step_update") => {
                let step = value.get("step_update").unwrap_or(&Value::Null);
                let id = step.get("conversation_id").and_then(Value::as_str);
                self.remember_conversation(id, &mut events);
                self.handle_step(step, &mut events);
            }
            Some("result") => {
                let result = value.get("result").unwrap_or(&Value::Null);
                let id = result.get("conversation_id").and_then(Value::as_str);
                self.remember_conversation(id, &mut events);
                self.handle_result(result, &mut events);
            }
            _ => {}
        }
        events
    }

    fn thread_id(&self) -> Option<String> {
        self.conversation_id.clone()
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

    const SIMPLE_FIXTURE: &str = include_str!("../tests/fixtures/antigravity_stream_simple.jsonl");
    const TOOLS_FIXTURE: &str = include_str!("../tests/fixtures/antigravity_stream_tools.jsonl");
    const DENIED_FIXTURE: &str = include_str!("../tests/fixtures/antigravity_stream_denied.jsonl");
    const ERROR_FIXTURE: &str = include_str!("../tests/fixtures/antigravity_stream_error.jsonl");

    fn parse_fixture(fixture: &str) -> (AntigravityStreamParser, Vec<CodexEvent>) {
        let mut parser = AntigravityStreamParser::default();
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
            reasoning_effort: Some("xhigh".to_string()),
            service_tier: None,
            session_prompt: None,
            sandbox_mode: sandbox.to_string(),
            approval_policy: "never".to_string(),
            search_mode: SearchMode::Disabled,
            add_dirs: vec![PathBuf::from("/srv/extra")],
            busy: false,
        }
    }

    fn request(prompt: &str) -> TurnRequest {
        TurnRequest {
            session_key: session("read-only").key,
            from_user_id: 1,
            prompt: prompt.to_string(),
            runtime_instructions: None,
            attachments: vec![],
            review_mode: None,
            override_search_mode: None,
            guest_query_id: None,
            guest_inline_message_id: None,
        }
    }

    fn config() -> AntigravityConfig {
        toml::from_str("binary = \"agy\"").unwrap()
    }

    #[test]
    fn parses_real_text_turn() {
        let (parser, events) = parse_fixture(SIMPLE_FIXTURE);
        assert_eq!(
            parser.thread_id().as_deref(),
            Some("7f98293c-908a-4846-a790-9f9c5a66a0ee")
        );
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Success("pong".to_string()))
        );
        assert!(matches!(&events[0], CodexEvent::ThreadStarted(_)));
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
    fn parses_real_tool_turn() {
        let (parser, events) = parse_fixture(TOOLS_FIXTURE);
        assert!(events.iter().any(
            |event| matches!(event, CodexEvent::Progress(text) if text == "Running `echo hi`")
        ));
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Success(
                "The command printed `hi`.".to_string()
            ))
        );
    }

    #[test]
    fn empty_answer_after_denied_tool_is_a_failure() {
        let (parser, events) = parse_fixture(DENIED_FIXTURE);
        assert!(events.iter().any(
            |event| matches!(event, CodexEvent::Progress(text) if text.starts_with("run_command failed: permission check failed"))
        ));
        match parser.outcome() {
            Some(StreamOutcome::Failed(message)) => {
                assert!(message.contains("not allowed to use Bash"), "{message}");
            }
            other => panic!("unexpected outcome {other:?}"),
        }
    }

    #[test]
    fn reports_error_results() {
        let (parser, _) = parse_fixture(ERROR_FIXTURE);
        assert_eq!(
            parser.outcome(),
            Some(StreamOutcome::Failed(
                "Error: empty prompt. Usage: agy --print \"your prompt here\"".to_string()
            ))
        );
        assert_eq!(parser.thread_id(), None);
    }

    #[test]
    fn builds_args_for_new_and_resumed_conversations() {
        let args = build_antigravity_args(
            &config(),
            &session("danger-full-access"),
            &AgentSessionState::default(),
        )
        .join(" ");
        assert_eq!(
            args,
            "--output-format stream-json --input-format stream-json --dangerously-skip-permissions --add-dir /srv/extra"
        );
        assert!(!args.contains("--effort"), "effort comes from the model id");

        let state = AgentSessionState {
            thread_id: Some("abc".to_string()),
            model: Some("gemini-3.8-flash-low".to_string()),
        };
        let args = build_antigravity_args(&config(), &session("workspace-write"), &state).join(" ");
        assert!(args.contains("--conversation abc"));
        assert!(args.contains("--model gemini-3.8-flash-low"));
        assert!(args.contains("--mode accept-edits"));

        let args = build_antigravity_args(&config(), &session("read-only"), &state);
        assert!(!args.iter().any(|arg| arg == "--mode"));
        assert!(
            !args
                .iter()
                .any(|arg| arg == "--dangerously-skip-permissions")
        );

        let mut config = config();
        config.permission_mode = Some("plan".to_string());
        let args =
            build_antigravity_args(&config, &session("danger-full-access"), &state).join(" ");
        assert!(args.contains("--mode plan"));
        assert!(!args.contains("--dangerously-skip-permissions"));
    }

    #[test]
    fn sends_prompt_as_ndjson_user_message() {
        let mut session = session("read-only");
        session.session_prompt = Some("Be brief.".to_string());
        let stdin = antigravity_stdin(&session, &request(" --hello\n\"quoted\" "));
        assert!(stdin.ends_with('\n'));
        let value: Value = serde_json::from_str(stdin.trim_end()).unwrap();
        assert_eq!(value["event"], "user");
        assert_eq!(value["message"]["role"], "user");
        assert_eq!(
            value["message"]["content"],
            "<instructions>\nBe brief.\n</instructions>\n\n--hello\n\"quoted\""
        );
    }

    #[test]
    fn parses_model_list_output() {
        let stdout = "Fetching available models...\ngemini-4-neon-high\tGemini 4 Neon (High)\ngemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n";
        assert_eq!(
            parse_model_list(stdout),
            vec!["gemini-4-neon-high", "gemini-3.8-flash-low"]
        );
    }

    /// Live check against the installed CLI: `cargo test live_antigravity -- --ignored`.
    #[tokio::test]
    #[ignore = "spawns the real agy CLI and needs a signed-in account"]
    async fn live_antigravity_new_then_resumed_turn() {
        use crate::agent::{AgentRunners, AgentVendor};
        let workspace = tempfile::tempdir().unwrap();
        let mut session = session("read-only");
        session.cwd = workspace.path().to_path_buf();
        session.add_dirs.clear();
        let mut runners = AgentRunners::for_tests();
        let mut config = config();
        config.binary = std::env::var("TELECODEX_LIVE_AGY_BINARY")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("agy"));
        config.default_model = Some(
            std::env::var("TELECODEX_LIVE_AGY_MODEL")
                .unwrap_or_else(|_| "gemini-3.8-flash-low".to_string()),
        );
        let runner = AntigravityRunner::new(config);
        assert!(!runner.read_models().await.unwrap().is_empty());
        runners.antigravity = runner;

        let turn = |prompt: &str, state: AgentSessionState| {
            let request = request(prompt);
            let runners = runners.clone();
            let session = session.clone();
            async move {
                runners
                    .run_turn(
                        AgentVendor::Antigravity,
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
        assert_eq!(first.assistant_text.trim().to_lowercase(), "pong");
        let thread_id = first.thread_id.clone().expect("agy conversation id");
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
