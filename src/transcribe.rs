use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use reqwest::multipart::{Form, Part};

use crate::models::AttachmentTranscript;

const DEFAULT_GROQ_MODEL: &str = "whisper-large-v3-turbo";

#[derive(Debug, Clone)]
pub enum TranscriptionBackend {
    Groq { api_key: String, model: String },
}

pub fn detect_transcription_backend() -> Option<TranscriptionBackend> {
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
        TranscriptionBackend::Groq { .. } => "Groq",
    }
}

pub async fn transcribe_audio_file(
    backend: &TranscriptionBackend,
    source_path: PathBuf,
    _scratch_dir: PathBuf,
) -> Result<AttachmentTranscript> {
    match backend {
        TranscriptionBackend::Groq { api_key, model } => {
            transcribe_audio_file_groq(api_key, model, source_path).await
        }
    }
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
}
