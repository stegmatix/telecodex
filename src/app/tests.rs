use super::agents::{agent_switched_text, codex_only_text};
use super::*;
use crate::config::SearchMode;
use std::path::PathBuf;
use std::sync::{Mutex as StdMutex, OnceLock};
use tempfile::NamedTempFile;

fn sample_workspace() -> PathBuf {
    std::env::temp_dir()
        .join("telecodex-tests")
        .join("workspace")
}

fn sample_voice_file() -> PathBuf {
    std::env::temp_dir()
        .join("telecodex-tests")
        .join("attachments")
        .join("voice.ogg")
}

fn sample_turn_workspace() -> TurnWorkspace {
    let root = std::env::temp_dir().join("telecodex-tests").join("turn");
    let out_dir = root.join("out");
    TurnWorkspace { root, out_dir }
}

pub(super) fn sample_defaults() -> SessionDefaults {
    SessionDefaults {
        cwd: sample_workspace(),
        model: Some("gpt-5.5".to_string()),
        reasoning_effort: Some("xhigh".to_string()),
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
    }
}

fn sample_turn_request(session_key: SessionKey) -> TurnRequest {
    TurnRequest {
        session_key,
        from_user_id: 100,
        prompt: "hello".to_string(),
        runtime_instructions: None,
        attachments: vec![],
        review_mode: None,
        override_search_mode: None,
        guest_query_id: None,
        guest_inline_message_id: None,
    }
}

fn codex_home_test_lock() -> &'static StdMutex<()> {
    static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| StdMutex::new(()))
}

struct CodexHomeGuard(Option<std::ffi::OsString>);

impl CodexHomeGuard {
    fn set(path: &std::path::Path) -> Self {
        let previous = std::env::var_os("CODEX_HOME");
        unsafe {
            std::env::set_var("CODEX_HOME", path);
        }
        Self(previous)
    }
}

impl Drop for CodexHomeGuard {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => unsafe {
                std::env::set_var("CODEX_HOME", value);
            },
            None => unsafe {
                std::env::remove_var("CODEX_HOME");
            },
        }
    }
}

pub(super) fn sample_config(db_path: PathBuf, default_cwd: PathBuf) -> Config {
    Config {
        telegram: crate::config::TelegramConfig {
            bot_token: Some("test-token".to_string()),
            bot_token_env: None,
            api_base: "https://api.telegram.org".to_string(),
            use_message_drafts: false,
            primary_forum_chat_id: None,
            auto_create_topics: false,
            forum_sync_topics_per_poll: 2,
            stale_topic_days: None,
            stale_topic_action: crate::config::StaleTopicAction::None,
            completion_notify_usernames: vec![],
        },
        codex: crate::config::CodexConfig {
            binary: PathBuf::from("codex"),
            default_cwd: default_cwd.clone(),
            default_model: Some("gpt-5.4".to_string()),
            default_reasoning_effort: Some("medium".to_string()),
            default_sandbox: "workspace-write".to_string(),
            default_approval: "never".to_string(),
            default_search_mode: SearchMode::Disabled,
            default_add_dirs: vec![],
            seed_workspaces: vec![],
            import_desktop_history: true,
            import_cli_history: true,
        },
        claude: toml::from_str("").unwrap(),
        gemini: toml::from_str("").unwrap(),
        db_path,
        startup_admin_ids: vec![],
        poll_timeout_seconds: 30,
        edit_debounce_ms: 250,
        max_text_chunk: 3500,
        tmp_dir: None,
    }
}

#[test]
fn detects_stale_codex_thread_errors() {
    let error = anyhow::anyhow!("no rollout found for thread id 019abc | code -32600");

    assert!(should_reset_session_after_error(&error));
}

#[test]
fn formats_turn_completion_notification() {
    assert_eq!(turn_completion_notification_text(&[]), None);
    assert_eq!(
        turn_completion_notification_text(&["@sama".to_string()]).as_deref(),
        Some("Готово, @sama fyi ✅")
    );
    assert_eq!(
        turn_completion_notification_text(&["@sama".to_string(), "@reviewer".to_string()])
            .as_deref(),
        Some("Готово, @sama @reviewer fyi ✅")
    );
}

#[test]
fn builds_rate_limit_retry_keyboard() {
    let keyboard = rate_limit_retry_keyboard(42);

    assert_eq!(keyboard.inline_keyboard[0][0].text, "Retry");
    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data.as_deref(),
        Some("cmd:/retry 42")
    );
}

#[test]
fn detects_stale_codex_thread_errors_in_error_context() {
    let error = anyhow::anyhow!("codex turn failed")
        .context("no rollout found for thread id 019abc | code -32600");

    assert!(should_reset_session_after_error(&error));
}

#[test]
fn ignores_unrelated_invalid_request_errors() {
    let error = anyhow::anyhow!("json-rpc request rejected with code -32600");

    assert!(!should_reset_session_after_error(&error));
}

#[test]
fn validates_absolute_directories() {
    let cwd = std::env::current_dir().unwrap();
    assert!(validate_directory(cwd.to_str().unwrap()).is_ok());
    assert!(validate_directory("relative\\path").is_err());
}

#[test]
fn validates_sandbox_values() {
    assert!(ensure_sandbox_mode("read-only").is_ok());
    assert!(ensure_sandbox_mode("boom").is_err());
}

#[test]
fn enables_live_search_for_latest_queries() {
    assert_eq!(
        auto_search_mode_for_prompt("what's new in the world over the last day?"),
        Some(SearchMode::Live)
    );
    assert_eq!(auto_search_mode_for_prompt("explain this code"), None);
}

#[test]
fn truncates_live_updates_to_single_chunk() {
    let text = "line one\n\nline two\n\nline three";
    let truncated = truncate_for_live_update(text, 16);
    assert!(truncated.len() <= 16);
    assert!(!truncated.is_empty());
}

#[test]
fn hides_sessions_overview_body_when_keyboard_is_available() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Water meter".to_string()),
        codex_thread_id: Some("019ce152-99e8-7c30-b5b7-166e6aebd550".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };

    let body = format_sessions_overview(&[session.clone()], session.key, &chat);

    assert_eq!(body, "\u{2063}");
}

#[test]
fn builds_clickable_chat_sessions_keyboard() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Water meter".to_string()),
        codex_thread_id: Some("019ce152-99e8-7c30-b5b7-166e6aebd550".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };

    let keyboard = chat_sessions_keyboard(&session, &chat, std::slice::from_ref(&session)).unwrap();

    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data,
        Some("ses:323".to_string())
    );
    assert_eq!(keyboard.inline_keyboard[0][0].url, None);
}

#[test]
fn builds_topic_links_for_dashboard_root_sessions_keyboard() {
    let root_session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, None),
        session_title: Some("Dashboard".to_string()),
        codex_thread_id: None,
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let topic_session = crate::models::SessionRecord {
        id: 2,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Water meter".to_string()),
        codex_thread_id: Some("019ce152-99e8-7c30-b5b7-166e6aebd550".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };

    let keyboard =
        chat_sessions_keyboard(&root_session, &chat, std::slice::from_ref(&topic_session)).unwrap();

    assert_eq!(keyboard.inline_keyboard[0][0].callback_data, None);
    assert_eq!(
        keyboard.inline_keyboard[0][0].url,
        Some("https://t.me/varv_alarms_bot_chat/323?thread=323".to_string())
    );
}

#[test]
fn derives_private_topic_link_slug_from_bot_api_chat_id() {
    assert_eq!(private_topic_link_slug(-1001234567890), Some(1234567890));
    assert_eq!(private_topic_link_slug(275328656), None);
}

#[test]
fn session_environment_match_requires_same_title_and_cwd() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(10)),
        session_title: Some("Ops Alerts".to_string()),
        codex_thread_id: Some("019".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-14T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };

    let same = CodexEnvironmentSummary {
        cwd: environment_identity_for_cwd(&session.cwd),
        name: "Ops Alerts".to_string(),
        latest_thread_id: Some("thr-1".to_string()),
        updated_at: "2026-03-14T10:05:00Z".to_string(),
    };
    let different_title = CodexEnvironmentSummary {
        cwd: environment_identity_for_cwd(&session.cwd),
        name: "ops alerts".to_string(),
        latest_thread_id: Some("thr-2".to_string()),
        updated_at: "2026-03-14T10:06:00Z".to_string(),
    };

    assert!(session_matches_environment(&session, &same));
    assert!(!session_matches_environment(&session, &different_title));
}

#[test]
fn forum_sync_preserves_manual_codex_binding() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: Some("manual-thread".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let environment = crate::codex_history::CodexEnvironmentSummary {
        cwd: sample_workspace(),
        name: "kombez".to_string(),
        latest_thread_id: Some("latest-thread".to_string()),
        updated_at: "2026-03-13T10:00:00Z".to_string(),
    };

    assert_eq!(
        super::forum::environment_sync_thread_binding(&session, &environment),
        None
    );
}

#[test]
fn forum_sync_seeds_unbound_environment_session() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: None,
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let environment = crate::codex_history::CodexEnvironmentSummary {
        cwd: sample_workspace(),
        name: "kombez".to_string(),
        latest_thread_id: Some("latest-thread".to_string()),
        updated_at: "2026-03-13T10:00:00Z".to_string(),
    };

    assert_eq!(
        super::forum::environment_sync_thread_binding(&session, &environment),
        Some("latest-thread")
    );
}

#[test]
fn forum_sync_preserves_fresh_thread_request() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: None,
        force_fresh_thread: true,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let environment = crate::codex_history::CodexEnvironmentSummary {
        cwd: sample_workspace(),
        name: "kombez".to_string(),
        latest_thread_id: Some("latest-thread".to_string()),
        updated_at: "2026-03-13T10:00:00Z".to_string(),
    };

    assert_eq!(
        super::forum::environment_sync_thread_binding(&session, &environment),
        None
    );
}

#[test]
fn format_session_status_marks_unbound_codex_session() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Water meter".to_string()),
        codex_thread_id: None,
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };

    let status = format_session_status(&session, &chat);

    assert!(status.contains("**Current Telegram session:** Water meter"));
    assert!(status.contains("- codex session title: unbound"));
    assert!(!status.contains("- codex session title: Water meter"));
}

#[test]
fn format_session_status_marks_fresh_codex_session() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Water meter".to_string()),
        codex_thread_id: None,
        force_fresh_thread: true,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };

    let status = format_session_status(&session, &chat);

    assert!(status.contains("- codex session title: fresh"));
}

#[test]
fn history_page_cache_evicts_oldest_entry_when_size_limit_is_hit() {
    let ttl = Duration::from_secs(300);
    let base = Instant::now();
    let mut cache = HistoryPageCache::default();
    let page = HistoryPageData {
        thread_title: "Session".to_string(),
        pages: vec![crate::codex_history::CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "answer".to_string(),
            timestamp: "2026-03-13T09:00:00Z".to_string(),
        }],
    };
    let first = HistoryPageCacheKey {
        codex_thread_id: "thread-1".to_string(),
        message_id: 10,
    };
    let second = HistoryPageCacheKey {
        codex_thread_id: "thread-2".to_string(),
        message_id: 11,
    };
    let third = HistoryPageCacheKey {
        codex_thread_id: "thread-3".to_string(),
        message_id: 12,
    };

    cache.insert(first.clone(), page.clone(), base, ttl, 2);
    cache.insert(
        second.clone(),
        page.clone(),
        base + Duration::from_secs(1),
        ttl,
        2,
    );
    cache.insert(
        third.clone(),
        page.clone(),
        base + Duration::from_secs(2),
        ttl,
        2,
    );

    assert!(
        cache
            .get(&first, base + Duration::from_secs(2), ttl)
            .is_none()
    );
    assert_eq!(
        cache.get(&second, base + Duration::from_secs(2), ttl),
        Some(page.clone())
    );
    assert_eq!(
        cache.get(&third, base + Duration::from_secs(2), ttl),
        Some(page)
    );
}

#[test]
fn history_page_cache_expires_stale_entries() {
    let ttl = Duration::from_secs(60);
    let base = Instant::now();
    let mut cache = HistoryPageCache::default();
    let key = HistoryPageCacheKey {
        codex_thread_id: "thread-1".to_string(),
        message_id: 10,
    };

    cache.insert(
        key.clone(),
        HistoryPageData {
            thread_title: "Session".to_string(),
            pages: vec![crate::codex_history::CodexHistoryEntry {
                role: "assistant".to_string(),
                text: "answer".to_string(),
                timestamp: "2026-03-13T09:00:00Z".to_string(),
            }],
        },
        base,
        ttl,
        4,
    );

    assert!(
        cache
            .get(&key, base + Duration::from_secs(61), ttl)
            .is_none()
    );
    assert!(cache.entries.is_empty());
}

#[test]
fn picks_last_assistant_text_from_history() {
    let history = vec![
        crate::codex_history::CodexHistoryEntry {
            role: "user".to_string(),
            text: "first".to_string(),
            timestamp: "2026-03-13T09:00:00Z".to_string(),
        },
        crate::codex_history::CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "alpha".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        crate::codex_history::CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "beta".to_string(),
            timestamp: "2026-03-13T09:00:02Z".to_string(),
        },
    ];

    assert_eq!(latest_assistant_text_from_history(&history), Some("beta"));
}

#[test]
fn assistant_history_pages_keep_only_assistant_messages_and_start_latest() {
    let history = vec![
        crate::codex_history::CodexHistoryEntry {
            role: "user".to_string(),
            text: "u1".to_string(),
            timestamp: "2026-03-13T09:00:00Z".to_string(),
        },
        crate::codex_history::CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "a1".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        crate::codex_history::CodexHistoryEntry {
            role: "user".to_string(),
            text: "u2".to_string(),
            timestamp: "2026-03-13T09:00:02Z".to_string(),
        },
        crate::codex_history::CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "a2".to_string(),
            timestamp: "2026-03-13T09:00:03Z".to_string(),
        },
    ];

    let pages = assistant_history_pages(&history);

    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].text, "a2");
    assert_eq!(pages[1].text, "a1");
}

#[test]
fn history_callback_matches_only_current_session_binding() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: Some("019ce672-9445-7612-bc5e-c8243a0d1915".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };

    assert!(history_callback_matches_current_session(
        &session,
        "019ce672-9445-7612-bc5e-c8243a0d1915"
    ));
    assert!(!history_callback_matches_current_session(
        &session,
        "019ce672-9445-7612-bc5e-c8243a0d1916"
    ));
}

#[test]
fn history_callback_rejects_unbound_fresh_session() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: None,
        force_fresh_thread: true,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };

    assert!(!history_callback_matches_current_session(
        &session,
        "019ce672-9445-7612-bc5e-c8243a0d1915"
    ));
}

#[test]
fn formats_stale_history_page_for_rebound_topic() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("kombez".to_string()),
        codex_thread_id: Some("019ce672-9445-7612-bc5e-c8243a0d1916".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };

    let text = format_stale_history_page(&session, "019ce672-9445-7612-bc5e-c8243a0d1915");

    assert!(text.contains("This `/history` view is stale."));
    assert!(text.contains("019ce672"));
    assert!(text.contains("Run `/history` again"));
}

#[test]
fn builds_import_button_for_seed_environment() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(-1001234567890, Some(323)),
        session_title: Some("Current topic".to_string()),
        codex_thread_id: Some("019ce152-99e8-7c30-b5b7-166e6aebd550".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let chat = crate::telegram::Chat {
        id: -1001234567890,
        kind: "supergroup".to_string(),
        is_forum: Some(true),
        username: Some("varv_alarms_bot_chat".to_string()),
        title: Some("Codex chat".to_string()),
    };
    let environment = CodexEnvironmentSummary {
        cwd: sample_workspace().join("seeded"),
        name: "Seeded".to_string(),
        latest_thread_id: None,
        updated_at: String::new(),
    };

    let keyboard = environment_dashboard_keyboard(&chat, &session, &[environment], &[]).unwrap();
    let button = &keyboard.inline_keyboard[0][0];

    assert_eq!(button.url, None);
    assert!(
        button
            .callback_data
            .as_deref()
            .unwrap()
            .starts_with("env:cwd:")
    );
}

#[test]
fn builds_model_quick_commands_from_current_and_default() {
    let commands = model_quick_commands(&[], Some("gpt-5.4"), Some("gpt-5.5"));

    assert_eq!(
        commands,
        vec![
            vec!["/model gpt-5.4".to_string(), "/model gpt-5.5".to_string()],
            vec![
                "/model gpt-6-astra".to_string(),
                "/model gpt-5.6-sol".to_string()
            ],
            vec![
                "/model gpt-5.6-terra".to_string(),
                "/model gpt-5.6-luna".to_string()
            ],
            vec![
                "/model gpt-5.4-mini".to_string(),
                "/model gpt-5.3-codex".to_string()
            ],
            vec![
                "/model gpt-5.3-codex-spark".to_string(),
                "/model gpt-5.2".to_string()
            ],
            vec!["/model default".to_string()],
        ]
    );
}

#[test]
fn deduplicates_model_quick_commands_when_current_matches_default() {
    let commands = model_quick_commands(&[], Some("gpt-5.5"), Some("gpt-5.5"));

    assert_eq!(
        commands,
        vec![
            vec![
                "/model gpt-5.5".to_string(),
                "/model gpt-6-astra".to_string()
            ],
            vec![
                "/model gpt-5.6-sol".to_string(),
                "/model gpt-5.6-terra".to_string()
            ],
            vec![
                "/model gpt-5.6-luna".to_string(),
                "/model gpt-5.4".to_string()
            ],
            vec![
                "/model gpt-5.4-mini".to_string(),
                "/model gpt-5.3-codex".to_string()
            ],
            vec![
                "/model gpt-5.3-codex-spark".to_string(),
                "/model gpt-5.2".to_string()
            ],
            vec!["/model default".to_string()],
        ]
    );
}

#[test]
fn includes_catalog_models_in_model_quick_commands() {
    let commands = model_quick_commands(
        &[
            AvailableModel {
                id: "gpt-5.5".to_string(),
                display_name: Some("gpt-5.5".to_string()),
                description: None,
                is_default: true,
            },
            AvailableModel {
                id: "gpt-5.3-codex".to_string(),
                display_name: Some("gpt-5.3-codex".to_string()),
                description: None,
                is_default: false,
            },
        ],
        Some("gpt-5.5"),
        None,
    );

    assert_eq!(
        commands,
        vec![
            vec![
                "/model gpt-5.5".to_string(),
                "/model gpt-6-astra".to_string()
            ],
            vec![
                "/model gpt-5.6-sol".to_string(),
                "/model gpt-5.6-terra".to_string()
            ],
            vec![
                "/model gpt-5.6-luna".to_string(),
                "/model gpt-5.4".to_string()
            ],
            vec![
                "/model gpt-5.4-mini".to_string(),
                "/model gpt-5.3-codex".to_string()
            ],
            vec![
                "/model gpt-5.3-codex-spark".to_string(),
                "/model gpt-5.2".to_string()
            ],
            vec!["/model default".to_string()],
        ]
    );
}

#[test]
fn formats_model_help_text_from_catalog() {
    let text = format_model_help_text(
        "gpt-5.5",
        &[
            AvailableModel {
                id: "gpt-5.5".to_string(),
                display_name: Some("gpt-5.5".to_string()),
                description: None,
                is_default: true,
            },
            AvailableModel {
                id: "gpt-5.3-codex".to_string(),
                display_name: Some("gpt-5.3-codex".to_string()),
                description: None,
                is_default: false,
            },
        ],
    );

    assert!(text.contains("Current model: `gpt-5.5`"));
    assert_eq!(
        text,
        "Current model: `gpt-5.5`\nAvailable quick picks: `gpt-5.5`, `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`"
    );
}

#[test]
fn builds_clickable_codex_sessions_keyboard() {
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("Telecodex".to_string()),
        codex_thread_id: Some("019ce672-9445-7612-bc5e-c8243a0d1915".to_string()),
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let summaries = vec![CodexThreadSummary {
        id: "019ce672-9445-7612-bc5e-c8243a0d1915".to_string(),
        title: "Check OpenAI app server".to_string(),
        cwd: sample_workspace(),
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        source: crate::codex_history::CodexHistorySource::Desktop,
    }];

    let keyboard = codex_sessions_keyboard(&session, &summaries).expect("keyboard");

    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data,
        Some("cmd:/use 019ce672-9445-7612-bc5e-c8243a0d1915".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[1][0].callback_data,
        Some("cmd:/use latest".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[1][1].callback_data,
        Some("cmd:/clear".to_string())
    );
}

#[test]
fn formats_recent_codex_history_preview() {
    let preview = format_codex_history_preview_plain(&[
        CodexHistoryEntry {
            role: "user".to_string(),
            text: "weather".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "done".to_string(),
            timestamp: "2026-03-13T09:00:03Z".to_string(),
        },
    ]);

    assert!(preview.contains("**Recent Codex History**"));
    assert!(preview.contains("**You**\n│ weather"));
    assert!(preview.contains("**Codex**\n│ done"));
}

#[test]
fn merges_adjacent_history_entries_with_same_role() {
    let preview = format_codex_history_preview_plain(&[
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "first answer".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "second answer".to_string(),
            timestamp: "2026-03-13T09:00:02Z".to_string(),
        },
    ]);

    assert!(preview.contains("│ first answer\n│ second answer"));
    assert_eq!(preview.matches("**Codex**").count(), 1);
}

#[test]
fn deduplicates_adjacent_identical_history_entries() {
    let preview = format_codex_history_preview_plain(&[
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "same answer".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "same answer".to_string(),
            timestamp: "2026-03-13T09:00:02Z".to_string(),
        },
    ]);

    assert_eq!(preview.matches("same answer").count(), 1);
}

#[test]
fn formats_recent_codex_history_preview_as_html_blockquotes() {
    let preview = format_codex_history_preview_html(&[
        CodexHistoryEntry {
            role: "user".to_string(),
            text: "weather".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "done".to_string(),
            timestamp: "2026-03-13T09:00:03Z".to_string(),
        },
    ]);

    assert!(preview.contains("<b>Recent Codex History</b>"));
    assert!(preview.contains("<b>You</b>\n<blockquote>weather</blockquote>"));
    assert!(preview.contains("<b>Codex</b>\n<blockquote>done</blockquote>"));
}

#[test]
fn preserves_markdown_inside_history_html_blockquotes() {
    let preview = format_codex_history_preview_html(&[CodexHistoryEntry {
        role: "assistant".to_string(),
        text: "Then yes, **counting** is already in progress and there is `code`.".to_string(),
        timestamp: "2026-03-13T09:00:03Z".to_string(),
    }]);

    assert!(preview.contains(
            "<blockquote>Then yes, <b>counting</b> is already in progress and there is <code>code</code>.</blockquote>"
        ));
}

#[test]
fn normalizes_history_lines_that_already_use_quote_prefixes() {
    let preview = format_codex_history_preview_html(&[CodexHistoryEntry {
        role: "assistant".to_string(),
        text: "│ **Codex**\n│ [repo](/home/s/projects/repo)".to_string(),
        timestamp: "2026-03-13T09:00:03Z".to_string(),
    }]);

    assert!(preview.contains(
        "<blockquote><b>Codex</b>\nrepo (&#x2F;home&#x2F;s&#x2F;projects&#x2F;repo)</blockquote>"
    ));
}

#[test]
fn formats_codex_history_context_for_runtime() {
    let context = format_codex_history_context(&[
        CodexHistoryEntry {
            role: "user".to_string(),
            text: "I need a script".to_string(),
            timestamp: "2026-03-13T09:00:01Z".to_string(),
        },
        CodexHistoryEntry {
            role: "assistant".to_string(),
            text: "working on the script".to_string(),
            timestamp: "2026-03-13T09:00:03Z".to_string(),
        },
    ]);

    assert!(context.contains("Recent conversation context from the selected Codex session"));
    assert!(context.contains("User: I need a script"));
    assert!(context.contains("Assistant: working on the script"));
}

#[test]
fn keeps_audio_transcript_in_user_prompt_only() {
    let voice_path = sample_voice_file();
    let voice_path_display = voice_path.display().to_string();
    let workspace = sample_turn_workspace();
    let out_dir_display = workspace.out_dir.display().to_string();
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("Voice notes".to_string()),
        codex_thread_id: None,
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let request = TurnRequest {
        session_key: session.key,
        from_user_id: 100,
        prompt: "summarize".to_string(),
        runtime_instructions: None,
        attachments: vec![crate::models::LocalAttachment {
            path: voice_path.clone(),
            file_name: "voice.ogg".to_string(),
            mime_type: Some("audio/ogg".to_string()),
            kind: AttachmentKind::Voice,
            transcript: Some(crate::models::AttachmentTranscript {
                engine: "Handy Parakeet".to_string(),
                text: "Hello world".to_string(),
            }),
        }],
        review_mode: None,
        override_search_mode: None,
        guest_query_id: None,
        guest_inline_message_id: None,
    };

    let runtime_request = prepare_runtime_request(&session, &request, &workspace);

    assert_eq!(runtime_request.prompt, "summarize\n\nHello world");
    assert!(!runtime_request.prompt.contains(&format!(
        "Attached local files:\n- voice.ogg -> {voice_path_display}"
    )));
    assert!(
        !runtime_request
            .prompt
            .contains("If you generate final deliverable files for the user")
    );
    assert!(!runtime_request.prompt.contains(&voice_path_display));
    let runtime_instructions = runtime_request.runtime_instructions.unwrap();
    assert!(runtime_instructions.contains(&out_dir_display));
    assert!(!runtime_instructions.contains(&voice_path_display));
}

#[test]
fn keeps_non_transcribed_audio_paths_in_user_prompt() {
    let voice_path = sample_voice_file();
    let voice_path_display = voice_path.display().to_string();
    let workspace = sample_turn_workspace();
    let session = crate::models::SessionRecord {
        id: 1,
        key: SessionKey::new(1, Some(2)),
        session_title: Some("Voice notes".to_string()),
        codex_thread_id: None,
        force_fresh_thread: false,
        updated_at: "2026-03-13T10:00:00Z".to_string(),
        cwd: sample_workspace(),
        model: None,
        reasoning_effort: None,
        service_tier: None,
        session_prompt: None,
        sandbox_mode: "workspace-write".to_string(),
        approval_policy: "never".to_string(),
        search_mode: SearchMode::Disabled,
        add_dirs: vec![],
        busy: false,
    };
    let request = TurnRequest {
        session_key: session.key,
        from_user_id: 100,
        prompt: "Analyze the attached files.".to_string(),
        runtime_instructions: None,
        attachments: vec![crate::models::LocalAttachment {
            path: voice_path.clone(),
            file_name: "voice.ogg".to_string(),
            mime_type: Some("audio/ogg".to_string()),
            kind: AttachmentKind::Voice,
            transcript: None,
        }],
        review_mode: None,
        override_search_mode: None,
        guest_query_id: None,
        guest_inline_message_id: None,
    };

    let runtime_request = prepare_runtime_request(&session, &request, &workspace);

    assert!(
        runtime_request
            .prompt
            .contains("Local files for this turn:")
    );
    assert!(
        runtime_request
            .prompt
            .contains(&format!("voice.ogg -> {voice_path_display}"))
    );
    assert!(
        runtime_request
            .runtime_instructions
            .unwrap()
            .contains(&format!(
                "Attached local files:\n- voice.ogg -> {voice_path_display}"
            ))
    );
}

#[test]
fn parses_approval_callback_payloads() {
    assert_eq!(
        parse_approval_callback_data("apr:abc123:a"),
        Some(("abc123".to_string(), CodexApprovalDecision::Accept))
    );
    assert_eq!(
        parse_approval_callback_data("apr:abc123:s"),
        Some((
            "abc123".to_string(),
            CodexApprovalDecision::AcceptForSession
        ))
    );
    assert_eq!(parse_approval_callback_data("cmd:/help"), None);
}

#[test]
fn parses_history_callback_payloads() {
    assert_eq!(
        parse_history_callback_data("his:019ce672-9445-7612-bc5e-c8243a0d1915:7"),
        Some(("019ce672-9445-7612-bc5e-c8243a0d1915".to_string(), 7))
    );
    assert_eq!(parse_history_callback_data("his:bad"), None);
}

#[test]
fn builds_approval_keyboard_buttons() {
    let keyboard = approval_keyboard(
        "token123",
        &[
            CodexApprovalDecision::Accept,
            CodexApprovalDecision::Decline,
            CodexApprovalDecision::Cancel,
        ],
    )
    .expect("approval keyboard");

    assert_eq!(keyboard.inline_keyboard.len(), 2);
    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data,
        Some("apr:token123:a".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[0][1].callback_data,
        Some("apr:token123:d".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[1][0].callback_data,
        Some("apr:token123:c".to_string())
    );
}

#[test]
fn builds_history_keyboard_buttons() {
    let keyboard =
        history_keyboard("019ce672-9445-7612-bc5e-c8243a0d1915", 1, 3).expect("history keyboard");

    assert_eq!(keyboard.inline_keyboard.len(), 1);
    assert_eq!(keyboard.inline_keyboard[0].len(), 2);
    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data,
        Some("his:019ce672-9445-7612-bc5e-c8243a0d1915:0".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[0][1].callback_data,
        Some("his:019ce672-9445-7612-bc5e-c8243a0d1915:2".to_string())
    );
}

#[test]
fn history_keyboard_wraps_around() {
    let keyboard =
        history_keyboard("019ce672-9445-7612-bc5e-c8243a0d1915", 0, 3).expect("history keyboard");

    assert_eq!(
        keyboard.inline_keyboard[0][0].callback_data,
        Some("his:019ce672-9445-7612-bc5e-c8243a0d1915:2".to_string())
    );
    assert_eq!(
        keyboard.inline_keyboard[0][1].callback_data,
        Some("his:019ce672-9445-7612-bc5e-c8243a0d1915:1".to_string())
    );
}

#[test]
fn formats_history_page() {
    let entry = crate::codex_history::CodexHistoryEntry {
        role: "assistant".to_string(),
        text: "Done".to_string(),
        timestamp: "2026-03-13T09:00:01Z".to_string(),
    };

    let page = format_history_page(
        "kombez",
        "019ce672-9445-7612-bc5e-c8243a0d1915",
        1,
        3,
        &entry,
    );

    assert!(page.contains("**Session history**"));
    assert!(page.contains("message: `2/3`"));
    assert!(page.contains("role: `assistant`"));
    assert!(page.contains("Done"));
}

#[test]
fn derives_session_title_from_first_non_empty_line() {
    assert_eq!(
        derive_session_title_from_text("\n  Check OpenAI app server   \nsecond line"),
        Some("Check OpenAI app server".to_string())
    );
}

#[test]
fn truncates_long_session_titles() {
    let title = derive_session_title_from_text(
        "Check a very long session title so the Telegram layout stays readable and does not break",
    )
    .expect("title");
    assert!(title.ends_with('…'));
    assert!(title.chars().count() <= 48);
}

#[test]
fn detects_commands_that_use_session_context() {
    assert!(command_uses_session_context(&ParsedInput::Forward(
        "/help".to_string()
    )));
    assert!(command_uses_session_context(&ParsedInput::Bridge(
        BridgeCommand::Copy
    )));
    assert!(!command_uses_session_context(&ParsedInput::Bridge(
        BridgeCommand::Sessions
    )));
    assert!(!command_uses_session_context(&ParsedInput::Bridge(
        BridgeCommand::History
    )));
    assert!(!command_uses_session_context(&ParsedInput::Bridge(
        BridgeCommand::Status
    )));
    assert!(!command_uses_session_context(&ParsedInput::Bridge(
        BridgeCommand::RestartBot
    )));
}

#[test]
fn detects_commands_that_require_codex_auth() {
    assert!(!parsed_input_requires_codex_auth(&ParsedInput::Bridge(
        BridgeCommand::Status
    )));
    assert!(!parsed_input_requires_codex_auth(&ParsedInput::Bridge(
        BridgeCommand::History
    )));
    assert!(parsed_input_requires_codex_auth(&ParsedInput::Bridge(
        BridgeCommand::Review(crate::models::ReviewRequest {
            base: None,
            commit: None,
            uncommitted: true,
            title: None,
            prompt: None,
        })
    )));
    assert!(!parsed_input_requires_codex_auth(&ParsedInput::Bridge(
        BridgeCommand::Pwd
    )));
    assert!(!parsed_input_requires_codex_auth(&ParsedInput::Bridge(
        BridgeCommand::Login
    )));
}

#[tokio::test]
async fn upload_failure_marks_turn_failed_and_cleanup_still_runs() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path(), &[100], &sample_defaults()).unwrap();
    let session = store
        .ensure_session(SessionKey::new(1, Some(2)), 100, &sample_defaults())
        .unwrap();
    let turn_id = store
        .record_turn_started(session.id, &sample_turn_request(session.key))
        .unwrap();

    let attachment_dir = tempfile::tempdir().unwrap();
    let attachment_path = attachment_dir.path().join("input.txt");
    std::fs::write(&attachment_path, "payload").unwrap();
    let turn_root = attachment_dir.path().join("turn-root");
    std::fs::create_dir_all(&turn_root).unwrap();

    let attachment = LocalAttachment {
        path: attachment_path.clone(),
        file_name: "input.txt".to_string(),
        mime_type: Some("text/plain".to_string()),
        kind: AttachmentKind::Text,
        transcript: None,
    };
    let summary = crate::codex::RunSummary {
        thread_id: Some("thread-123".to_string()),
        assistant_text: "answer".to_string(),
        stderr_text: String::new(),
    };
    let failure_messages = Arc::new(StdMutex::new(Vec::<String>::new()));
    let failure_messages_sink = failure_messages.clone();

    let result = finalize_foreground_turn(
        ForegroundTurnSuccess {
            store: &store,
            session: &session,
            vendor: crate::agent::AgentVendor::Codex,
            turn_id,
            review_mode: false,
            summary: &summary,
        },
        || async { Err(anyhow!("upload failed")) },
        || async { Ok(()) },
        move |message| {
            let failure_messages_sink = failure_messages_sink.clone();
            async move {
                failure_messages_sink.lock().unwrap().push(message);
                Ok(())
            }
        },
    )
    .await;
    let result = finish_turn_cleanup(&[attachment], &turn_root, result);

    assert!(result.is_err());
    assert_eq!(
        store.turn_status(turn_id).unwrap().as_deref(),
        Some("failed")
    );
    assert!(!attachment_path.exists());
    assert!(!turn_root.exists());
    assert!(
        failure_messages
            .lock()
            .unwrap()
            .iter()
            .any(|message| message.contains("upload failed"))
    );
}

#[test]
fn rebinds_stale_session_to_latest_active_thread_for_same_cwd() {
    let _lock = codex_home_test_lock().lock().unwrap();
    let codex_home = tempfile::tempdir().unwrap();
    let workspace = codex_home.path().join("workspace");
    std::fs::create_dir_all(codex_home.path().join("sessions")).unwrap();
    std::fs::create_dir_all(codex_home.path().join("archived_sessions")).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(
        codex_home
            .path()
            .join("archived_sessions")
            .join("rollout-archive.jsonl"),
        format!(
            "{{\"timestamp\":\"2026-03-13T08:00:00Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"archived-thread\",\"timestamp\":\"2026-03-13T08:00:00Z\",\"cwd\":\"{}\",\"source\":\"exec\"}}}}\n",
            workspace.display()
        ),
    )
    .unwrap();
    std::fs::write(
        codex_home
            .path()
            .join("sessions")
            .join("rollout-active.jsonl"),
        format!(
            "{{\"timestamp\":\"2026-03-13T09:00:00Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"active-thread\",\"timestamp\":\"2026-03-13T09:00:00Z\",\"cwd\":\"{}\",\"source\":\"exec\"}}}}\n",
            workspace.display()
        ),
    )
    .unwrap();

    let _codex_home_guard = CodexHomeGuard::set(codex_home.path());

    let db = NamedTempFile::new().unwrap();
    let config = sample_config(db.path().to_path_buf(), workspace.clone());
    let store = Store::open(db.path(), &[100], &sample_defaults()).unwrap();
    let shared = Arc::new(AppShared {
        config,
        store,
        telegram: TelegramClient::new(
            "test-token".to_string(),
            "https://api.telegram.org".to_string(),
        ),
        agents: crate::agent::AgentRunners::for_tests(),
        bot_username: None,
        service_user_id: 0,
        transcription_backend: None,
        session_defaults: sample_defaults(),
        limits_cache: Mutex::new(None),
        history_page_cache: Mutex::new(HistoryPageCache::default()),
        pending_approvals: Mutex::new(HashMap::new()),
        pending_codex_login: Mutex::new(None),
        codex_login_backoff_until: Mutex::new(None),
        shutdown: CancellationToken::new(),
    });
    let session = shared
        .store
        .ensure_session(SessionKey::new(1, Some(52)), 100, &sample_defaults())
        .unwrap();
    shared
        .store
        .set_session_cwd(session.key, &workspace)
        .unwrap();
    shared
        .store
        .set_session_codex_thread(session.key, "archived-thread")
        .unwrap();
    let session = shared.store.get_session(session.key).unwrap().unwrap();

    let rebound = resolve_session_codex_binding_from_history(&shared, session).unwrap();

    assert_eq!(rebound.codex_thread_id.as_deref(), Some("active-thread"));
}

#[test]
fn keeps_truly_archived_session_unbound_when_no_active_replacement_exists() {
    let _lock = codex_home_test_lock().lock().unwrap();
    let codex_home = tempfile::tempdir().unwrap();
    let workspace = codex_home.path().join("workspace");
    std::fs::create_dir_all(codex_home.path().join("archived_sessions")).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(
        codex_home
            .path()
            .join("archived_sessions")
            .join("rollout-archive.jsonl"),
        format!(
            "{{\"timestamp\":\"2026-03-13T08:00:00Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"archived-thread\",\"timestamp\":\"2026-03-13T08:00:00Z\",\"cwd\":\"{}\",\"source\":\"exec\"}}}}\n",
            workspace.display()
        ),
    )
    .unwrap();

    let _codex_home_guard = CodexHomeGuard::set(codex_home.path());

    let db = NamedTempFile::new().unwrap();
    let config = sample_config(db.path().to_path_buf(), workspace.clone());
    let store = Store::open(db.path(), &[100], &sample_defaults()).unwrap();
    let shared = Arc::new(AppShared {
        config,
        store,
        telegram: TelegramClient::new(
            "test-token".to_string(),
            "https://api.telegram.org".to_string(),
        ),
        agents: crate::agent::AgentRunners::for_tests(),
        bot_username: None,
        service_user_id: 0,
        transcription_backend: None,
        session_defaults: sample_defaults(),
        limits_cache: Mutex::new(None),
        history_page_cache: Mutex::new(HistoryPageCache::default()),
        pending_approvals: Mutex::new(HashMap::new()),
        pending_codex_login: Mutex::new(None),
        codex_login_backoff_until: Mutex::new(None),
        shutdown: CancellationToken::new(),
    });
    let session = shared
        .store
        .ensure_session(SessionKey::new(1, Some(52)), 100, &sample_defaults())
        .unwrap();
    shared
        .store
        .set_session_cwd(session.key, &workspace)
        .unwrap();
    shared
        .store
        .set_session_codex_thread(session.key, "archived-thread")
        .unwrap();
    let session = shared.store.get_session(session.key).unwrap().unwrap();

    let rebound = resolve_session_codex_binding_from_history(&shared, session).unwrap();

    assert_eq!(rebound.codex_thread_id.as_deref(), Some("archived-thread"));
}

#[test]
fn offers_gpt6_without_catalog_and_deduplicates_live_model() {
    let fallback = model_quick_commands(&[], None, None);
    assert_eq!(fallback[0][0], "/model gpt-6-astra");
    let models = [AvailableModel {
        id: "gpt-6-astra".to_string(),
        display_name: Some("GPT-6 Astra".to_string()),
        description: None,
        is_default: true,
    }];
    let commands = model_quick_commands(&models, Some("gpt-6-astra"), Some("gpt-6-astra"));
    assert_eq!(
        commands
            .iter()
            .flatten()
            .filter(|command| command.as_str() == "/model gpt-6-astra")
            .count(),
        1
    );
    assert!(
        format_model_help_text("gpt-6-astra", &models)
            .contains("Current model: \u{0060}gpt-6-astra\u{0060}")
    );
}

#[test]
fn supports_gpt6_max_reasoning_effort() {
    assert_eq!(normalize_reasoning_effort(" MAX ").unwrap(), "max");
    assert!(normalize_reasoning_effort("unsupported").is_err());
}

#[test]
fn detects_stale_claude_and_gemini_sessions() {
    let claude = anyhow::anyhow!(
        "claude turn failed: No conversation found with session ID: 0199aaaa-0000-7000-8000-000000000000"
    );
    let gemini = anyhow::anyhow!("gemini turn failed: Invalid session identifier \"0199\".");
    assert!(should_reset_session_after_error(&claude));
    assert!(should_reset_session_after_error(&gemini));
}

#[test]
fn agent_texts_name_the_vendors() {
    use crate::agent::{AgentSessionState, AgentVendor};
    let switched = agent_switched_text(AgentVendor::Codex, AgentVendor::Claude);
    assert!(switched.contains("`codex` → `claude`"));
    assert!(switched.contains("/new"));
    assert!(
        codex_only_text("/limits", AgentVendor::Gemini)
            .contains("`/limits` is Codex-only, but the active agent is `gemini`")
    );
    let state = AgentSessionState {
        thread_id: Some("abc".to_string()),
        model: None,
    };
    let status = format_agent_status_line(AgentVendor::Claude, &state, Some("opus".to_string()));
    assert!(status.contains("**Agent:** `claude`"));
    assert!(status.contains("Claude session: `abc`"));
    assert!(status.contains("Claude model: `opus`"));
    assert_eq!(
        format_agent_status_line(AgentVendor::Codex, &state, None),
        "**Agent:** `codex`"
    );
}

/// Mirrors `TelecodexTurnStore` in Helen's `scripts/ops/magic_button_consumer.py`:
/// the voice button reads these tables directly, so every vendor must leave the
/// same `sessions`/`turns` footprint.
#[tokio::test]
async fn voice_consumer_queries_resolve_turns_for_every_vendor() {
    use crate::agent::AgentVendor;
    use rusqlite::{Connection, OpenFlags, params};

    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path(), &[100], &sample_defaults()).unwrap();
    let key = SessionKey::new(4242, None);
    store.ensure_session(key, 100, &sample_defaults()).unwrap();
    let reader = Connection::open_with_flags(
        tmp.path(),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .unwrap();
    reader.execute_batch("PRAGMA query_only = ON").unwrap();
    let anchor = |reader: &Connection| -> (i64, i64) {
        reader
            .query_row(
                "SELECT
                   (SELECT COALESCE(MAX(id), 0) FROM turns) AS turn_floor,
                   sessions.id
                 FROM sessions
                 WHERE sessions.chat_id = ?
                   AND sessions.thread_id = ?",
                params![key.chat_id, key.thread_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap()
    };
    let resolutions = |reader: &Connection, after: i64, session_id: i64, prompt: &str| {
        let mut stmt = reader
            .prepare(
                "SELECT id, status, assistant_text
                   FROM turns
                  WHERE id > ?
                    AND session_id = ?
                    AND from_user_id = ?
                    AND prompt = ?
                  ORDER BY id ASC",
            )
            .unwrap();
        stmt.query_map(params![after, session_id, 100, prompt], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap()
    };

    for (index, vendor) in AgentVendor::ALL.into_iter().enumerate() {
        store.set_agent_vendor(vendor).unwrap();
        // `/agent` and `/new` must never replace the sessions row the voice
        // consumer anchored on.
        let session = store.get_session(key).unwrap().unwrap();
        store.clear_agent_conversation(&session, vendor).unwrap();
        let (turn_floor, session_id) = anchor(&reader);
        assert_eq!(session_id, session.id);

        let transcript = format!("voice command {index}");
        let mut request = sample_turn_request(key);
        request.prompt = transcript.clone();
        let turn_id = store.record_turn_started(session.id, &request).unwrap();
        let pending = resolutions(&reader, turn_floor, session_id, &transcript);
        assert_eq!(pending, vec![(turn_id, "running".to_string(), None)]);

        let summary = crate::codex::RunSummary {
            thread_id: Some(format!("{}-session", vendor.as_str())),
            assistant_text: format!("{} answer", vendor.label()),
            stderr_text: String::new(),
        };
        finalize_foreground_turn(
            ForegroundTurnSuccess {
                store: &store,
                session: &session,
                vendor,
                turn_id,
                review_mode: false,
                summary: &summary,
            },
            || async { Ok(()) },
            || async { Ok(()) },
            |_| async { Ok(()) },
        )
        .await
        .unwrap();

        let resolved = resolutions(&reader, turn_floor, session_id, &transcript);
        assert_eq!(
            resolved,
            vec![(
                turn_id,
                "completed".to_string(),
                Some(format!("{} answer", vendor.label()))
            )]
        );
        let session = store.get_session(key).unwrap().unwrap();
        let state = store.agent_session_state(&session, vendor).unwrap();
        assert_eq!(
            state.thread_id.as_deref(),
            Some(format!("{}-session", vendor.as_str()).as_str())
        );
    }

    // Each vendor kept its own conversation, so switching back resumes it.
    let session = store.get_session(key).unwrap().unwrap();
    assert_eq!(session.codex_thread_id.as_deref(), Some("codex-session"));
    for vendor in [AgentVendor::Claude, AgentVendor::Gemini] {
        assert_eq!(
            store
                .agent_session_state(&session, vendor)
                .unwrap()
                .thread_id
                .as_deref(),
            Some(format!("{}-session", vendor.as_str()).as_str())
        );
    }

    // Failed vendor turns surface the statuses the consumer maps to errors.
    let (turn_floor, session_id) = anchor(&reader);
    let mut request = sample_turn_request(key);
    request.prompt = "broken".to_string();
    let turn_id = store.record_turn_started(session.id, &request).unwrap();
    store
        .record_turn_finished(turn_id, "cancelled", None)
        .unwrap();
    assert_eq!(
        resolutions(&reader, turn_floor, session_id, "broken"),
        vec![(turn_id, "cancelled".to_string(), None)]
    );
}

#[test]
fn agent_banner_names_non_codex_vendor() {
    use crate::agent::AgentVendor;
    assert_eq!(
        with_agent_banner(Some("limits".to_string()), AgentVendor::Codex, None).as_deref(),
        Some("limits")
    );
    assert_eq!(
        with_agent_banner(None, AgentVendor::Claude, Some("opus".to_string())).as_deref(),
        Some("🤖 Claude · `opus`")
    );
    assert_eq!(
        with_agent_banner(Some("⚡ fast".to_string()), AgentVendor::Gemini, None).as_deref(),
        Some("🤖 Gemini · `default`\n⚡ fast")
    );
}
