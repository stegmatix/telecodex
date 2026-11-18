use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use tokio::process::Command;

use crate::models::AttachmentTranscript;

const DEFAULT_GROQ_MODEL: &str = "whisper-large-v3-turbo";

#[derive(Debug, Clone)]
pub enum TranscriptionBackend {
    Command { executable: PathBuf },
    Groq { api_key: String, model: String },
}

pub fn detect_transcription_backend() -> Option<TranscriptionBackend> {
    if let Some(executable) =
        std::env::var_os("TELECODEX_TRANSCRIBE_COMMAND").filter(|value| !value.is_empty())
    {
        return Some(TranscriptionBackend::Command {
            executable: executable.into(),
        });
    }
    if let Some(api_key) = std::env::var_os("GROQ_API_KEY") {
        let api_key = api_key.to_string_lossy().trim().to_string();
        if !api_key.is_empty() {
            let model = std::env::var("GROQ_TRANSCRIPTION_MODEL")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| DEFAULT_GROQ_MODEL.to_string());
            return Some(TranscriptionBackend::Groq { api_key, model });
        }
    }
    None
}

pub fn transcription_backend_label(backend: &TranscriptionBackend) -> &'static str {
    match backend {
        TranscriptionBackend::Command { .. } => "Helen ASR",
        TranscriptionBackend::Groq { .. } => "Groq",
    }
}

pub async fn transcribe_audio_file(
    backend: &TranscriptionBackend,
    source_path: PathBuf,
    _scratch_dir: PathBuf,
) -> Result<AttachmentTranscript> {
    let transcript = match backend {
        TranscriptionBackend::Command { executable } => {
            transcribe_audio_file_command(executable, source_path).await
        }
        TranscriptionBackend::Groq { api_key, model } => {
            transcribe_audio_file_groq(api_key, model, source_path).await
        }
    }?;
    tracing::info!(engine = %transcript.engine, "audio transcription completed");
    Ok(transcript)
}

#[derive(Deserialize)]
struct CommandTranscript {
    text: String,
    provider: String,
    model: String,
}

async fn transcribe_audio_file_command(
    executable: &Path,
    source_path: PathBuf,
) -> Result<AttachmentTranscript> {
    if !executable.is_absolute() {
        bail!("transcription command must be an absolute executable path");
    }
    let source_path = tokio::fs::canonicalize(source_path)
        .await
        .context("cannot resolve audio file")?;
    // The shared helper owns local deadlines, cancellation and cloud fallback.
    // Allow multi-hour recordings; dropping this future also terminates the helper.
    let output = tokio::time::timeout(
        Duration::from_secs(24 * 3600),
        Command::new(executable)
            .arg("--json")
            .arg(source_path)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .context("transcription command timed out")?
    .context("could not start transcription command")?;
    if !output.status.success() {
        // Do not expose captured stderr: providers may include private details.
        bail!("transcription command failed ({})", output.status);
    }
    let result: CommandTranscript =
        serde_json::from_slice(&output.stdout).context("invalid transcription command response")?;
    let text = result.text.trim().to_string();
    if text.is_empty() {
        bail!("transcript is empty");
    }
    let provider = match result.provider.as_str() {
        "samsung" => "Samsung Qwen",
        "groq" => "Groq",
        _ => bail!("unrecognized transcription provider"),
    };
    if result.model.is_empty()
        || result.model.len() > 128
        || result.model.chars().any(char::is_control)
    {
        bail!("invalid transcription model identity");
    }
    Ok(AttachmentTranscript {
        engine: format!("{provider} ({})", result.model),
        text,
    })
}

async fn transcribe_audio_file_groq(
    api_key: &str,
    model: &str,
    source_path: PathBuf,
) -> Result<AttachmentTranscript> {
    let file_name = normalized_upload_name(&source_path);
    let bytes = tokio::fs::read(&source_path)
        .await
        .with_context(|| format!("failed to read {}", source_path.display()))?;
    let part = Part::bytes(bytes)
        .file_name(file_name)
        .mime_str("application/octet-stream")
        .context("failed to build multipart upload")?;
    let form = Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", "text".to_string());

    let response = reqwest::Client::new()
        .post("https://api.groq.com/openai/v1/audio/transcriptions")
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await
        .context("groq transcription request failed")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read groq transcription response")?;
    if !status.is_success() {
        bail!("Groq API error {status}: {body}");
    }
    let text = body.trim().to_string();
    if text.is_empty() {
        bail!("transcript is empty");
    }
    Ok(AttachmentTranscript {
        engine: format!("Groq ({model})"),
        text,
    })
}

fn normalized_upload_name(source_path: &Path) -> String {
    let filename = source_path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("audio");
    let suffix = source_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());
    match suffix.as_deref() {
        Some("oga") | Some("opus") => format!(
            "{}.ogg",
            source_path
                .file_stem()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("audio")
        ),
        _ => filename.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_groq_upload_extension_for_telegram_audio() {
        assert_eq!(
            normalized_upload_name(Path::new("/tmp/voice.oga")),
            "voice.ogg"
        );
        assert_eq!(
            normalized_upload_name(Path::new("/tmp/voice.opus")),
            "voice.ogg"
        );
        assert_eq!(
            normalized_upload_name(Path::new("/tmp/voice.ogg")),
            "voice.ogg"
        );
    }

    #[cfg(unix)]
    fn command_fixture(body: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("transcribe helper.sh");
        std::fs::write(&executable, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let audio = root.path().join("voice ; $(false).oga");
        std::fs::write(&audio, b"audio").unwrap();
        (root, executable, audio)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_preserves_audio_argument_and_actual_provider() {
        for (provider, expected) in [("samsung", "Samsung Qwen"), ("groq", "Groq")] {
            let body = format!(
                r#"test "$#" = 2 && test "$1" = --json && test -f "$2" || exit 4
printf '%s' '{{"text":"  Пример речи  ","provider":"{provider}","model":"fixture"}}'"#
            );
            let (root, executable, audio) = command_fixture(&body);
            let result = transcribe_audio_file(
                &TranscriptionBackend::Command { executable },
                audio.clone(),
                root.path().into(),
            )
            .await
            .unwrap();
            assert_eq!(result.text, "Пример речи");
            assert_eq!(result.engine, format!("{expected} (fixture)"));
            assert!(audio.exists(), "the bridge still owns the attachment");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_errors_do_not_expose_private_stderr_or_claim_cloud_success() {
        let (root, executable, audio) = command_fixture("echo private-secret >&2\nexit 3");
        let error = transcribe_audio_file(
            &TranscriptionBackend::Command { executable },
            audio,
            root.path().into(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("command failed"));
        assert!(!format!("{error:#}").contains("private-secret"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_empty_malformed_or_unknown_command_results() {
        for response in [
            "not json",
            r#"{"text":"","provider":"samsung","model":"qwen"}"#,
            r#"{"text":"text","provider":"unknown","model":"model"}"#,
        ] {
            let (root, executable, audio) = command_fixture(&format!("printf '%s' '{response}'"));
            assert!(
                transcribe_audio_file(
                    &TranscriptionBackend::Command { executable },
                    audio,
                    root.path().into()
                )
                .await
                .is_err()
            );
        }
    }

    #[tokio::test]
    #[ignore = "requires an explicitly supplied public fixture and configured Helen helper"]
    async fn live_helen_transcription_command() {
        let command = std::env::var_os("TELECODEX_TRANSCRIBE_COMMAND").expect("helper required");
        let audio = std::env::var_os("TELECODEX_ASR_TEST_AUDIO").expect("public fixture required");
        let expected =
            std::env::var("TELECODEX_ASR_TEST_PROVIDER").expect("expected provider required");
        let result = transcribe_audio_file(
            &TranscriptionBackend::Command {
                executable: command.into(),
            },
            audio.into(),
            std::env::temp_dir(),
        )
        .await
        .unwrap();
        assert!(
            result.engine.starts_with(&expected),
            "unexpected provider: {}",
            result.engine
        );
        assert!(!result.text.is_empty());
        println!(
            "provider={} transcript_characters={}",
            result.engine,
            result.text.chars().count()
        );
    }
}
