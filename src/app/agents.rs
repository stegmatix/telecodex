use super::*;

impl App {
    pub(super) fn active_agent_vendor(&self) -> Result<AgentVendor> {
        self.shared.store.agent_vendor()
    }

    /// `/agent [codex|claude|gemini]`: show or switch the global agent CLI.
    pub(super) async fn handle_agent_command(
        &self,
        user: &crate::models::UserRecord,
        message: &Message,
        requested: Option<AgentVendor>,
    ) -> Result<()> {
        let current = self.active_agent_vendor()?;
        let Some(requested) = requested else {
            self.send_command_help(
                message.chat.id,
                message.message_thread_id,
                &CommandHelp {
                    text: format!(
                        "Current agent: `{}`\n\nThe agent is global: it serves every chat, topic and the voice button.\n\nChoose one:",
                        current.as_str()
                    ),
                    quick_commands: vec![
                        AgentVendor::ALL
                            .iter()
                            .map(|vendor| format!("/agent {}", vendor.as_str()))
                            .collect(),
                    ],
                },
            )
            .await?;
            return Ok(());
        };
        ensure_admin(user)?;
        if requested == current {
            self.send_status(
                message.chat.id,
                message.message_thread_id,
                &format!("Agent is already `{}`.", current.as_str()),
            )
            .await?;
            return Ok(());
        }
        self.shared.store.set_agent_vendor(requested)?;
        self.shared.store.audit(
            Some(user.tg_user_id),
            "set_agent_vendor",
            serde_json::json!({ "from": current.as_str(), "to": requested.as_str() }),
        )?;
        self.send_status(
            message.chat.id,
            message.message_thread_id,
            &agent_switched_text(current, requested),
        )
        .await?;
        Ok(())
    }

    /// Replies and returns `true` when a Codex-only command runs under another vendor.
    pub(super) async fn reply_if_codex_only(
        &self,
        message: &Message,
        command: &str,
    ) -> Result<bool> {
        let vendor = self.active_agent_vendor()?;
        if vendor == AgentVendor::Codex {
            return Ok(false);
        }
        self.send_status(
            message.chat.id,
            message.message_thread_id,
            &codex_only_text(command, vendor),
        )
        .await?;
        Ok(true)
    }

    /// `/model` for Claude and Gemini: per-session model stored per vendor.
    pub(super) async fn handle_vendor_model_command(
        &self,
        vendor: AgentVendor,
        session: &crate::models::SessionRecord,
        message: &Message,
        model: Option<String>,
    ) -> Result<()> {
        let (models, default_model) = match vendor {
            AgentVendor::Claude => (
                self.shared.agents.claude.models(),
                self.shared.agents.claude.default_model(),
            ),
            AgentVendor::Gemini => (
                self.shared.agents.gemini.models(),
                self.shared.agents.gemini.default_model(),
            ),
            AgentVendor::Codex => bail!("Codex models are handled by the Codex model picker"),
        };
        let fallback = format!("{} CLI default", vendor.label());
        if let Some(model) = model {
            let next_model = if model == "-" || model.eq_ignore_ascii_case("default") {
                None
            } else {
                Some(model)
            };
            self.shared
                .store
                .set_agent_model(session, vendor, next_model.as_deref())?;
            let label = next_model.as_deref().or(default_model).unwrap_or(&fallback);
            self.send_status(
                message.chat.id,
                message.message_thread_id,
                &format!("{} model set to `{label}`.", vendor.label()),
            )
            .await?;
            return Ok(());
        }
        let state = self.shared.store.agent_session_state(session, vendor)?;
        let current = state
            .model
            .as_deref()
            .or(default_model)
            .unwrap_or(&fallback);
        let mut quick_commands = models
            .chunks(2)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|model| format!("/model {model}"))
                    .collect()
            })
            .collect::<Vec<Vec<String>>>();
        quick_commands.push(vec!["/model default".to_string()]);
        self.send_command_help(
            message.chat.id,
            message.message_thread_id,
            &CommandHelp {
                text: format!(
                    "Current {} model: `{current}`\n\nPick one or send `/model <name>` with any model the {} CLI accepts:",
                    vendor.label(),
                    vendor.label()
                ),
                quick_commands,
            },
        )
        .await?;
        Ok(())
    }
}

pub(super) fn agent_switched_text(from: AgentVendor, to: AgentVendor) -> String {
    format!(
        "Agent switched: `{}` → `{}`.\n\n{} does not see the {} conversation. Each chat continues {}'s own earlier conversation if there is one; send `/new` to start fresh. Switching back resumes the {} conversation.",
        from.as_str(),
        to.as_str(),
        to.label(),
        from.label(),
        to.label(),
        from.label()
    )
}

pub(super) fn codex_only_text(command: &str, vendor: AgentVendor) -> String {
    format!(
        "`{command}` is Codex-only, but the active agent is `{}`. Switch with `/agent codex` to use it.",
        vendor.as_str()
    )
}
