//! Backend seam: one turn lifecycle shared by the Codex, Claude and Gemini CLIs.
//!
//! Every backend reports progress through [`CodexEvent`] and returns a
//! [`RunSummary`], so Telegram rendering and the `turns` table stay identical
//! regardless of which vendor produced the answer.

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result, anyhow, bail};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::Mutex,
};
use tokio_util::sync::CancellationToken;

use crate::{
    claude::ClaudeRunner,
    codex::{
        CodexEvent, CodexEventOutcome, CodexRunner, CommandSpec, RunSummary, spawnable_command,
        terminate_child,
    },
    gemini::GeminiRunner,
    models::{SessionRecord, TurnRequest},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AgentVendor {
    #[default]
    Codex,
    Claude,
    Gemini,
}

impl AgentVendor {
    pub const ALL: [AgentVendor; 3] = [Self::Codex, Self::Claude, Self::Gemini];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
            Self::Gemini => "gemini",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Codex => "Codex",
            Self::Claude => "Claude",
            Self::Gemini => "Gemini",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "codex" | "openai" => Some(Self::Codex),
            "claude" | "anthropic" => Some(Self::Claude),
            "gemini" | "google" => Some(Self::Gemini),
            _ => None,
        }
    }
}

/// Per-vendor conversation binding for one Telegram session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentSessionState {
    pub thread_id: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone)]
pub struct AgentRunners {
    pub codex: CodexRunner,
    pub claude: ClaudeRunner,
    pub gemini: GeminiRunner,
}

impl AgentRunners {
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            codex: CodexRunner::new(config.codex.binary.clone()),
            claude: ClaudeRunner::new(config.claude.clone()),
            gemini: GeminiRunner::new(config.gemini.clone()),
        }
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self {
            codex: CodexRunner::new(PathBuf::from("codex")),
            claude: ClaudeRunner::new(toml::from_str("").expect("claude defaults")),
            gemini: GeminiRunner::new(toml::from_str("").expect("gemini defaults")),
        }
    }

    pub async fn run_turn<F, Fut>(
        &self,
        vendor: AgentVendor,
        session: &SessionRecord,
        state: &AgentSessionState,
        request: &TurnRequest,
        cancel: CancellationToken,
        mut on_event: F,
    ) -> Result<RunSummary>
    where
        F: FnMut(CodexEvent) -> Fut,
        Fut: std::future::Future<Output = Result<CodexEventOutcome>>,
    {
        if vendor != AgentVendor::Codex && request.review_mode.is_some() {
            bail!("/review is only supported by Codex; switch with `/agent codex`");
        }
        match vendor {
            AgentVendor::Codex => {
                self.codex
                    .run_turn(session, request, cancel, on_event)
                    .await
            }
            AgentVendor::Claude => {
                let command = self.claude.build_command(session, state, request);
                run_cli_turn(
                    "claude",
                    command,
                    crate::claude::ClaudeStreamParser::default(),
                    cancel,
                    &mut on_event,
                )
                .await
            }
            AgentVendor::Gemini => {
                let command = self.gemini.build_command(session, state, request);
                run_cli_turn(
                    "gemini",
                    command,
                    crate::gemini::GeminiStreamParser::default(),
                    cancel,
                    &mut on_event,
                )
                .await
            }
        }
    }
}

/// Terminal state reported inside a vendor's JSON stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamOutcome {
    Success(String),
    Failed(String),
}

/// Incremental parser for one vendor's line-delimited JSON output.
pub(crate) trait StreamParser {
    fn parse_line(&mut self, line: &str) -> Vec<CodexEvent>;
    fn thread_id(&self) -> Option<String>;
    fn outcome(&self) -> Option<StreamOutcome>;
}

/// A fully resolved non-interactive CLI invocation; the prompt goes to stdin.
#[derive(Debug, Clone)]
pub(crate) struct CliTurnCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub current_dir: PathBuf,
    pub stdin: String,
}

const STDERR_TAIL_LIMIT: usize = 8 * 1024;

pub(crate) async fn run_cli_turn<P, F, Fut>(
    label: &str,
    command: CliTurnCommand,
    mut parser: P,
    cancel: CancellationToken,
    on_event: &mut F,
) -> Result<RunSummary>
where
    P: StreamParser,
    F: FnMut(CodexEvent) -> Fut,
    Fut: std::future::Future<Output = Result<CodexEventOutcome>>,
{
    let spec = CommandSpec {
        program: command.program.clone(),
        args: command.args.clone(),
        current_dir: Some(command.current_dir.clone()),
    };
    let mut process = spawnable_command(&spec);
    process.kill_on_drop(true);
    let mut child = process.spawn().with_context(|| {
        format!(
            "failed to spawn {label} command: {}",
            command.program.display()
        )
    })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("{label} stdin pipe unavailable"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("{label} stdout pipe unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("{label} stderr pipe unavailable"))?;

    let prompt = command.stdin;
    let stdin_task = tokio::spawn(async move {
        let _ = stdin.write_all(prompt.as_bytes()).await;
        let _ = stdin.shutdown().await;
    });
    let stderr_buffer = Arc::new(Mutex::new(String::new()));
    let stderr_buffer_task = stderr_buffer.clone();
    let stderr_label = label.to_string();
    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("{stderr_label} stderr: {line}");
            let mut buffer = stderr_buffer_task.lock().await;
            if !buffer.is_empty() {
                buffer.push('\n');
            }
            buffer.push_str(&line);
            if buffer.len() > STDERR_TAIL_LIMIT {
                let mut cut = buffer.len() - STDERR_TAIL_LIMIT;
                while !buffer.is_char_boundary(cut) {
                    cut += 1;
                }
                buffer.drain(..cut);
            }
        }
    });

    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut announced_thread: Option<String> = None;
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                terminate_child(&mut child).await;
                stdin_task.abort();
                let _ = stderr_task.await;
                bail!("{label} turn cancelled");
            }
            next_line = stdout_lines.next_line() => {
                let Some(line) = next_line.with_context(|| format!("reading {label} stdout failed"))? else {
                    break;
                };
                for event in parser.parse_line(&line) {
                    if let CodexEvent::ThreadStarted(thread_id) = &event {
                        announced_thread = Some(thread_id.clone());
                    }
                    let _ = on_event(event).await?;
                }
            }
        }
    }

    let status = child
        .wait()
        .await
        .with_context(|| format!("waiting for {label} failed"))?;
    let _ = stdin_task.await;
    let _ = stderr_task.await;
    let stderr_text = stderr_buffer.lock().await.trim().to_string();
    let thread_id = parser.thread_id().or(announced_thread);
    match parser.outcome() {
        Some(StreamOutcome::Failed(message)) => bail!("{label} turn failed: {message}"),
        Some(StreamOutcome::Success(assistant_text)) if status.success() => Ok(RunSummary {
            thread_id,
            assistant_text,
            stderr_text,
        }),
        _ if !status.success() => {
            if stderr_text.is_empty() {
                bail!("{label} exited with status {status}");
            }
            bail!("{label} exited with status {status}: {stderr_text}");
        }
        _ => bail!("{label} exited without reporting a final result"),
    }
}

/// Codex-style session prompt + per-turn runtime instructions in one block.
pub(crate) fn merged_instructions(
    session: &SessionRecord,
    request: &TurnRequest,
) -> Option<String> {
    let sections = [
        session.session_prompt.as_deref(),
        request.runtime_instructions.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .filter(|section| !section.is_empty())
    .collect::<Vec<_>>();
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

/// One-line progress label for a tool call reported by Claude or Gemini.
pub(crate) fn describe_tool_call(name: &str, input: &serde_json::Value) -> String {
    let detail = [
        "command",
        "file_path",
        "absolute_path",
        "path",
        "pattern",
        "url",
        "query",
    ]
    .into_iter()
    .find_map(|key| input.get(key).and_then(serde_json::Value::as_str))
    .map(|value| value.lines().next().unwrap_or_default().trim().to_string())
    .filter(|value| !value.is_empty());
    let text = match detail {
        Some(detail) if is_shell_tool(name) => format!("Running `{detail}`"),
        Some(detail) => format!("{name} `{detail}`"),
        None => name.to_string(),
    };
    truncate_chars(&text, 200)
}

fn is_shell_tool(name: &str) -> bool {
    matches!(name, "Bash" | "run_shell_command" | "shell")
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit - 1).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_vendor_names() {
        assert_eq!(AgentVendor::parse("Claude"), Some(AgentVendor::Claude));
        assert_eq!(AgentVendor::parse(" gemini "), Some(AgentVendor::Gemini));
        assert_eq!(AgentVendor::parse("codex"), Some(AgentVendor::Codex));
        assert_eq!(AgentVendor::parse("gpt"), None);
        for vendor in AgentVendor::ALL {
            assert_eq!(AgentVendor::parse(vendor.as_str()), Some(vendor));
        }
    }

    #[test]
    fn describes_tool_calls() {
        assert_eq!(
            describe_tool_call("Bash", &json!({"command": "ls -la\necho hi"})),
            "Running `ls -la`"
        );
        assert_eq!(
            describe_tool_call("Read", &json!({"file_path": "/tmp/a.txt"})),
            "Read `/tmp/a.txt`"
        );
        assert_eq!(describe_tool_call("TodoWrite", &json!({})), "TodoWrite");
    }
}
