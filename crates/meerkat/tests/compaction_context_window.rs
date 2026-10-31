//! The compaction trigger against a model's context window, end to end
//! through the runtime-backed facade and a durable session store.
//!
//! The model double counts two request bytes a token, twice as dense as the
//! transcript estimator's four bytes a token (as JSON listings are), reports
//! that count for every request it answers, and refuses a request over its
//! context window with a context-length error, as a provider does. Every row
//! sets an explicit 100k-token compaction threshold.
//!
//! Each row sizes its payloads from the counts the double reports, so the
//! system prompt and framing every request carries do not move the shapes.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

#[cfg(all(
    feature = "session-store",
    feature = "session-compaction",
    not(target_arch = "wasm32")
))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use meerkat::surface::{
        build_runtime_backed_service, default_persistent_executor, materialize_session,
    };
    use meerkat::{
        AgentFactory, Config, CreateSessionRequest, FactoryAgentBuilder, PersistenceBundle,
        PersistentSessionService, Session,
    };
    use meerkat_client::types::LlmStream;
    use meerkat_client::{LlmClient, LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
    use meerkat_core::session_store::SessionStore;
    use meerkat_core::{Message, SessionBuildOptions};
    use meerkat_runtime::completion::CompletionOutcome;
    use meerkat_runtime::{Input, MeerkatMachine, PromptInput};
    use tokio::time::Duration;

    /// The compaction threshold every row configures explicitly.
    const THRESHOLD: u64 = 100_000;
    /// The context window of the model in the A and C rows.
    const WINDOW: u64 = 150_000;
    /// The bytes of the first run's prompt in the A and C rows: about 70k
    /// presented tokens, under the threshold.
    const FIRST_PROMPT_BYTES: usize = 140_000;
    /// The tool the double calls when asked to list.
    const LISTING_TOOL: &str = "list_records";
    /// The prompt the double answers with a call to the listing tool.
    const LIST_PROMPT: &str = "list every record";
    /// What the double's compaction summary says.
    const SUMMARY: &str = "summary of the earlier work";

    /// One request the double served.
    #[derive(Debug, Clone)]
    struct Seen {
        /// A compaction summary request.
        summary: bool,
        /// The request's tokens, at two request bytes a token.
        presented: u64,
        /// Refused for being over the window.
        refused: bool,
        /// The length of the listing result the request carries, if any.
        listing: Option<usize>,
    }

    /// A model with a `window`-token context window (see the module docs).
    /// It answers a compaction summary request with [`SUMMARY`], the
    /// [`LIST_PROMPT`] with a call to [`LISTING_TOOL`], a request that ends
    /// in that call's result with "done", and anything else with "ok".
    struct WindowedModel {
        window: u64,
        seen: Mutex<Vec<Seen>>,
    }

    impl WindowedModel {
        fn new(window: u64) -> Arc<Self> {
            Arc::new(Self {
                window,
                seen: Mutex::new(Vec::new()),
            })
        }

        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }

        fn seen_since(&self, from: usize) -> Vec<Seen> {
            self.seen()[from..].to_vec()
        }

        /// The tokens of the last ordinary request it answered.
        fn last_answered(&self) -> u64 {
            self.seen()
                .iter()
                .rev()
                .find(|seen| !seen.summary && !seen.refused)
                .map(|seen| seen.presented)
                .expect("an answered request")
        }
    }

    fn presented_tokens(request: &LlmRequest) -> u64 {
        let bytes = serde_json::to_vec(&request.messages).expect("request messages serialize");
        u64::try_from(bytes.len()).unwrap_or(u64::MAX) / 2
    }

    fn last_user_text(request: &LlmRequest) -> String {
        request
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.text_content()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn listing_carried(request: &LlmRequest) -> Option<usize> {
        request.messages.iter().find_map(|message| match message {
            Message::ToolResults { results, .. } => results
                .iter()
                .map(|result| result.text_content())
                .find(|text| text.starts_with("record-"))
                .map(|text| text.len()),
            _ => None,
        })
    }

    fn text(text: &str) -> LlmEvent {
        LlmEvent::TextDelta {
            delta: text.to_string(),
            meta: None,
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for WindowedModel {
        fn project_replay_messages(&self, messages: &[Message]) -> Result<Vec<Message>, LlmError> {
            Ok(messages.to_vec())
        }

        fn stream<'a>(&'a self, request: &'a LlmRequest) -> LlmStream<'a> {
            let presented = presented_tokens(request);
            let last_user = last_user_text(request);
            let summary = last_user.contains("CONTEXT COMPACTION");
            let refused = presented > self.window;
            self.seen.lock().unwrap().push(Seen {
                summary,
                presented,
                refused,
                listing: listing_carried(request),
            });
            if refused {
                return Box::pin(futures::stream::iter([Err(
                    LlmError::ContextLengthExceeded {
                        max: usize::try_from(self.window).unwrap_or(usize::MAX),
                        requested: usize::try_from(presented).unwrap_or(usize::MAX),
                    },
                )]));
            }
            let (first, stop) = if summary {
                (text(SUMMARY), meerkat_core::StopReason::EndTurn)
            } else if matches!(request.messages.last(), Some(Message::ToolResults { .. })) {
                (text("done"), meerkat_core::StopReason::EndTurn)
            } else if last_user == LIST_PROMPT {
                (
                    LlmEvent::ToolCallComplete {
                        id: "call-list".to_string(),
                        name: LISTING_TOOL.to_string(),
                        args: serde_json::json!({}),
                        meta: None,
                    },
                    meerkat_core::StopReason::ToolUse,
                )
            } else {
                (text("ok"), meerkat_core::StopReason::EndTurn)
            };
            let usage = LlmEvent::UsageUpdate {
                usage: meerkat_core::TurnUsage::host_declared(
                    meerkat_core::Provider::OpenAI,
                    &request.model,
                    meerkat_core::Usage {
                        input_tokens: presented,
                        output_tokens: 8,
                        ..meerkat_core::Usage::default()
                    },
                ),
            };
            Box::pin(futures::stream::iter(
                [
                    first,
                    usage,
                    LlmEvent::Done {
                        outcome: LlmDoneOutcome::success(stop),
                    },
                ]
                .into_iter()
                .map(Ok),
            ))
        }

        fn provider(&self) -> meerkat_core::Provider {
            meerkat_core::Provider::Other
        }

        async fn health_check(&self) -> Result<(), LlmError> {
            Ok(())
        }
    }

    /// [`LISTING_TOOL`]: a listing of the size the row sets before the call,
    /// short records without a character JSON escapes, so its request bytes
    /// are its length. Counts its calls.
    struct ListingTool {
        bytes: AtomicUsize,
        calls: AtomicUsize,
    }

    impl ListingTool {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                bytes: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
            })
        }

        fn set_listing(&self, bytes: usize) {
            self.bytes.store(bytes, Ordering::SeqCst);
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    fn listing(bytes: usize) -> String {
        let mut text = String::with_capacity(bytes + 32);
        let mut record = 0_usize;
        while text.len() < bytes {
            text.push_str(&format!("record-{record:07} active; "));
            record += 1;
        }
        text.truncate(bytes);
        text
    }

    #[async_trait::async_trait]
    impl meerkat_core::AgentToolDispatcher for ListingTool {
        fn tools(&self) -> Arc<[Arc<meerkat_core::ToolDef>]> {
            Arc::from([Arc::new(meerkat_core::ToolDef::new(
                LISTING_TOOL,
                "list every record",
                serde_json::json!({ "type": "object" }),
            ))])
        }

        async fn dispatch(
            &self,
            call: meerkat_core::ToolCallView<'_>,
        ) -> Result<meerkat_core::ToolDispatchOutcome, meerkat_core::ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(meerkat_core::ToolResult::new(
                call.id.to_string(),
                listing(self.bytes.load(Ordering::SeqCst)),
                false,
            )
            .into())
        }
    }

    /// A runtime-backed service over one HeadCanonical SQLite realm under
    /// `root`, with the explicit [`THRESHOLD`].
    fn build_service(
        root: &std::path::Path,
        model: &Arc<WindowedModel>,
    ) -> (
        Arc<PersistentSessionService<FactoryAgentBuilder>>,
        Arc<MeerkatMachine>,
    ) {
        let sqlite_path = root.join("sessions.sqlite3");
        let session_store: Arc<dyn SessionStore> = Arc::new(
            meerkat::SqliteSessionStore::open(sqlite_path.clone()).expect("open session sqlite"),
        );
        let runtime_store: Arc<dyn meerkat_runtime::RuntimeStore> = Arc::new(
            meerkat_runtime::store::SqliteRuntimeStore::new_head_canonical(sqlite_path)
                .expect("open runtime sqlite"),
        );
        let blob_store: Arc<dyn meerkat_core::BlobStore> =
            Arc::new(meerkat_store::FsBlobStore::new(root.join("blobs")));
        let bundle = PersistenceBundle::new(session_store, runtime_store, blob_store)
            .expect("construct runtime authority");

        let mut config = Config::default();
        config.compaction.auto_compact_threshold = THRESHOLD;
        config.compaction.auto_compact_threshold_explicit = true;
        let client: Arc<dyn LlmClient> = model.clone();
        let mut builder =
            FactoryAgentBuilder::new(AgentFactory::new(root.join("sessions")), config);
        builder.default_llm_client = Some(client);
        let (service, adapter) = build_runtime_backed_service(builder, 4, bundle)
            .expect("compose the runtime-backed service");
        (Arc::new(service), adapter)
    }

    /// End a host lifetime so a successor over the same realm can host the
    /// session: every runtime registration unregistered to terminal, and the
    /// service shut down.
    async fn stand_down(
        service: &Arc<PersistentSessionService<FactoryAgentBuilder>>,
        adapter: &Arc<MeerkatMachine>,
    ) {
        for registration in adapter.current_session_registration_witnesses().await {
            assert!(
                adapter
                    .unregister_session_registration_until_terminal_if_current(&registration)
                    .await
                    .expect("unregister the lifetime's runtime registration"),
                "the lifetime's current registration is torn down"
            );
        }
        service
            .try_shutdown()
            .await
            .expect("shut the lifetime's session service down");
    }

    fn create_request(tool: &Arc<ListingTool>) -> CreateSessionRequest {
        let tools: Arc<dyn meerkat_core::AgentToolDispatcher> = tool.clone();
        let build = SessionBuildOptions {
            external_tools: Some(tools),
            override_builtins: meerkat_core::ToolCategoryOverride::Disable,
            ..SessionBuildOptions::default()
        };
        CreateSessionRequest {
            injected_context: Vec::new(),
            model: "gpt-5.4".to_string(),
            prompt: meerkat_core::ContentInput::Text(String::new()),
            system_prompt: meerkat::SystemPromptOverride::Set(
                "compaction against the context window".to_string(),
            ),
            max_tokens: None,
            event_tx: None,
            initial_turn: meerkat_core::service::InitialTurnPolicy::Defer,
            deferred_prompt_policy: meerkat_core::service::DeferredPromptPolicy::Discard,
            build: Some(build),
            labels: None,
        }
    }

    async fn materialize(
        service: &Arc<PersistentSessionService<FactoryAgentBuilder>>,
        adapter: &Arc<MeerkatMachine>,
        session: Session,
        tool: &Arc<ListingTool>,
    ) {
        let service_for_executor = Arc::clone(service);
        let adapter_for_executor = Arc::clone(adapter);
        Box::pin(materialize_session(
            service,
            adapter,
            session,
            create_request(tool),
            move |session_id| {
                default_persistent_executor(service_for_executor, adapter_for_executor, session_id)
            },
        ))
        .await
        .expect("materialize session");
    }

    /// The outcome of one prompt, or why it did not resolve.
    async fn run_prompt(
        adapter: &Arc<MeerkatMachine>,
        session_id: &meerkat::SessionId,
        prompt: &str,
    ) -> Result<CompletionOutcome, String> {
        let (_outcome, handle) = adapter
            .accept_input_with_completion(session_id, Input::Prompt(PromptInput::new(prompt, None)))
            .await
            .map_err(|error| format!("accept_input failed: {error}"))?;
        let handle = handle.ok_or_else(|| "missing completion handle".to_string())?;
        match tokio::time::timeout(Duration::from_secs(60), handle.wait()).await {
            Err(_) => Err("prompt timed out".to_string()),
            Ok(Err(error)) => Err(format!("completion waiter failed: {error:?}")),
            Ok(Ok(outcome)) => Ok(outcome),
        }
    }

    fn completed(outcome: &Result<CompletionOutcome, String>) -> bool {
        matches!(outcome, Ok(CompletionOutcome::Completed(_)))
    }

    async fn committed(
        service: &Arc<PersistentSessionService<FactoryAgentBuilder>>,
        session_id: &meerkat::SessionId,
    ) -> Session {
        service
            .load_authoritative_session(session_id)
            .await
            .expect("authoritative load")
            .expect("the session exists")
    }

    fn user_texts(session: &Session) -> Vec<String> {
        session
            .messages()
            .iter()
            .filter_map(|message| match message {
                Message::User(user) => Some(user.text_content()),
                _ => None,
            })
            .collect()
    }

    fn has_compaction_summary(session: &Session) -> bool {
        session.messages().iter().any(|message| {
            matches!(
                message,
                Message::User(user)
                    if user.transcript_role
                        == meerkat_core::types::TranscriptUserRole::CompactionSummary
            )
        })
    }

    fn carries_listing(session: &Session) -> bool {
        session.messages().iter().any(|message| {
            matches!(
                message,
                Message::ToolResults { results, .. }
                    if results.iter().any(|result| result.text_content().starts_with("record-"))
            )
        })
    }

    /// The reported failure, prevented. The first run's request presents
    /// about 70k tokens, under the threshold. The second run calls a tool
    /// whose listing takes the next request past the 150k window, while the
    /// transcript estimate keeps the history and the forecast under the
    /// threshold and the last answered request's count lags by exactly the
    /// listing. The projection (that count plus the estimate of only what
    /// was appended since) crosses the threshold, so the rows before the run
    /// fold into a summary first and the request with the listing fits: the
    /// run completes, the tool ran once and its result reaches the model
    /// whole. Without the projection nothing fired before the request, and
    /// the provider refused it.
    #[tokio::test]
    async fn a_tool_result_that_crosses_the_window_compacts_before_the_next_request() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = WindowedModel::new(WINDOW);
        let tool = ListingTool::new();
        let (service, adapter) = build_service(temp.path(), &model);
        let session = Session::new();
        let session_id = session.id().clone();
        materialize(&service, &adapter, session, &tool).await;

        let first = run_prompt(&adapter, &session_id, &"a".repeat(FIRST_PROMPT_BYTES)).await;
        assert!(completed(&first), "the first run completes: {first:?}");
        let anchor = model.last_answered();
        assert!(
            anchor + 10_000 < THRESHOLD,
            "test setup: the first request presents under the threshold: {anchor}"
        );
        let history = meerkat_core::agent::compact::estimate_tokens(
            committed(&service, &session_id).await.messages(),
        )
        .expect("estimate the history");
        // A listing past the window once added to the anchor (two bytes a
        // token), and under the threshold once added to the history estimate
        // (four bytes a token), while the projection crosses it.
        let listing_size = 2 * (WINDOW - anchor) + 20_000;
        assert!(
            history + listing_size / 4 + 2_000 < THRESHOLD,
            "test setup: only the projection can fire: history {history} + listing {}",
            listing_size / 4
        );
        assert!(
            anchor + listing_size / 4 >= THRESHOLD,
            "test setup: the projection crosses the threshold"
        );
        let listing_bytes = usize::try_from(listing_size).expect("listing size");
        tool.set_listing(listing_bytes);

        let second = run_prompt(&adapter, &session_id, LIST_PROMPT).await;
        let seen = model.seen();
        assert!(
            completed(&second),
            "the run completes: {second:?}; requests: {seen:?}"
        );
        assert_eq!(tool.calls(), 1, "the tool ran once");
        assert!(
            !seen.iter().any(|seen| seen.refused),
            "no request was refused: {seen:?}"
        );
        assert_eq!(
            seen.iter().filter(|seen| seen.summary).count(),
            1,
            "one compaction: {seen:?}"
        );
        let summary_at = seen
            .iter()
            .position(|seen| seen.summary)
            .expect("a summary request");
        let listing_at = seen
            .iter()
            .position(|seen| seen.listing.is_some())
            .expect("a request carries the listing");
        assert!(
            summary_at < listing_at,
            "the compaction precedes the request carrying the listing: {seen:?}"
        );
        assert_eq!(
            seen[listing_at].listing,
            Some(listing_bytes),
            "the listing reaches the model whole"
        );

        let session = committed(&service, &session_id).await;
        assert!(has_compaction_summary(&session), "the compaction committed");
        let texts = user_texts(&session);
        assert!(
            !texts.iter().any(|text| text.starts_with("aaaa")),
            "the first run folded into the summary"
        );
        assert!(
            texts.iter().any(|text| text == LIST_PROMPT),
            "the run's prompt stays"
        );
        assert!(
            session.messages().iter().any(|message| matches!(
                message,
                Message::ToolResults { results, .. }
                    if results.iter().any(|result| result.text_content().len() == listing_bytes)
            )),
            "the run's result stays whole"
        );
    }

    /// Runs a session into a context refusal no fold inside the run can
    /// prevent: after a first run of about 70k tokens, a run calls the tool
    /// for a listing that is over the window by itself. That run compacts
    /// the rows before it and still sends a request over the window: it is
    /// refused, and the run fails. Returns the session id.
    async fn refuse_an_oversize_round(
        service: &Arc<PersistentSessionService<FactoryAgentBuilder>>,
        adapter: &Arc<MeerkatMachine>,
        model: &Arc<WindowedModel>,
        tool: &Arc<ListingTool>,
    ) -> meerkat::SessionId {
        let session = Session::new();
        let session_id = session.id().clone();
        materialize(service, adapter, session, tool).await;
        let first = run_prompt(adapter, &session_id, &"a".repeat(FIRST_PROMPT_BYTES)).await;
        assert!(completed(&first), "the first run completes: {first:?}");
        // 160k tokens: over the window with nothing else in the request.
        tool.set_listing(usize::try_from(2 * (WINDOW + 10_000)).expect("listing size"));

        let before = model.seen().len();
        let second = run_prompt(adapter, &session_id, LIST_PROMPT).await;
        let seen = model.seen_since(before);
        assert!(
            !completed(&second),
            "test setup: the run with the oversize listing fails: {second:?}; requests: {seen:?}"
        );
        assert!(
            seen.iter()
                .any(|seen| seen.refused && seen.listing.is_some()),
            "test setup: the provider refused the request carrying the listing: {seen:?}"
        );
        assert_eq!(tool.calls(), 1, "the tool ran once");
        session_id
    }

    /// The run after the refusal, `continue`, and one more: the run compacts
    /// the whole transcript before its first request, the round holding the
    /// oversize listing folds into the summary, and the run completes. Its
    /// summary request carries the listing too and is refused, so the summary
    /// is the mechanical one. The next run compacts no more.
    async fn assert_the_next_run_heals(
        service: &Arc<PersistentSessionService<FactoryAgentBuilder>>,
        adapter: &Arc<MeerkatMachine>,
        model: &Arc<WindowedModel>,
        session_id: &meerkat::SessionId,
    ) {
        let before = model.seen().len();
        let third = run_prompt(adapter, session_id, "continue").await;
        let seen = model.seen_since(before);
        assert!(
            completed(&third),
            "the run after the refusal completes: {third:?}; requests: {seen:?}"
        );
        assert!(
            seen.first().is_some_and(|seen| seen.summary),
            "the run compacts before its first request: {seen:?}"
        );
        assert!(
            seen.last()
                .is_some_and(|seen| !seen.summary && !seen.refused && seen.listing.is_none()),
            "the run's request fits once the listing is folded: {seen:?}"
        );
        let session = committed(service, session_id).await;
        assert!(has_compaction_summary(&session), "the compaction committed");
        assert!(
            !carries_listing(&session),
            "the oversize round folded into the summary"
        );
        assert!(
            user_texts(&session).iter().any(|text| text == "continue"),
            "the run's own prompt stays"
        );

        let before = model.seen().len();
        let fourth = run_prompt(adapter, session_id, "and again").await;
        let seen = model.seen_since(before);
        assert!(completed(&fourth), "{fourth:?}; requests: {seen:?}");
        assert!(
            !seen.iter().any(|seen| seen.summary),
            "no refusal since, no compaction: {seen:?}"
        );
    }

    /// The wedge, in the host that saw the refusal: without the refusal
    /// fact, the next run's first check saw the same counts as before the
    /// refused request (it reported no usage), sent the same oversize
    /// request, and was refused again, as was every run after it.
    #[tokio::test]
    async fn the_run_after_a_context_refusal_compacts_first() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = WindowedModel::new(WINDOW);
        let tool = ListingTool::new();
        let (service, adapter) = build_service(temp.path(), &model);
        let session_id = refuse_an_oversize_round(&service, &adapter, &model, &tool).await;

        assert_the_next_run_heals(&service, &adapter, &model, &session_id).await;
    }

    /// The wedge across a restart: the refusal fact rides the session's
    /// durable compaction cadence, so the agent a successor host rebuilds
    /// from the store compacts first too.
    #[tokio::test]
    async fn a_session_left_refused_compacts_first_after_a_restart() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model = WindowedModel::new(WINDOW);
        let tool = ListingTool::new();
        let session_id = {
            let (service, adapter) = build_service(temp.path(), &model);
            let session_id = refuse_an_oversize_round(&service, &adapter, &model, &tool).await;
            stand_down(&service, &adapter).await;
            session_id
        };

        let (service, adapter) = build_service(temp.path(), &model);
        let resume_source = committed(&service, &session_id).await;
        materialize(&service, &adapter, resume_source, &tool).await;
        assert_the_next_run_heals(&service, &adapter, &model, &session_id).await;
    }

    /// A restored agent projects its first request from the anchor its
    /// session persisted. The last answered request presented about 106k
    /// tokens, over the threshold, so the agent that answered it would
    /// compact before its next request on that count. After a restart the
    /// rebuilt agent has no count of its own, and the next prompt takes the
    /// request past the model's 120k window while the transcript estimate
    /// stays under the threshold: without the persisted anchor nothing fired
    /// and the provider refused the restored agent's first request. With it
    /// the earlier rows fold first, the prompt stays verbatim and the run
    /// completes.
    #[tokio::test]
    async fn a_restored_agent_projects_its_first_request_from_the_persisted_anchor() {
        const RESTORED_WINDOW: u64 = 120_000;
        let temp = tempfile::tempdir().expect("tempdir");
        let model = WindowedModel::new(RESTORED_WINDOW);
        let tool = ListingTool::new();
        let (session_id, anchor) = {
            let (service, adapter) = build_service(temp.path(), &model);
            let session = Session::new();
            let session_id = session.id().clone();
            materialize(&service, &adapter, session, &tool).await;
            let hello = run_prompt(&adapter, &session_id, "hi").await;
            assert!(completed(&hello), "{hello:?}");
            // The tokens every request carries before its prompt.
            let overhead = model.last_answered();
            let first = run_prompt(
                &adapter,
                &session_id,
                &"a".repeat(usize::try_from(2 * (THRESHOLD + 6_000 - overhead)).expect("size")),
            )
            .await;
            assert!(completed(&first), "{first:?}");
            let anchor = model.last_answered();
            assert!(
                anchor >= THRESHOLD && anchor + 8_000 < RESTORED_WINDOW,
                "test setup: the last answered request presents over the threshold and under \
                 the window: {anchor}"
            );
            assert!(
                !model.seen().iter().any(|seen| seen.summary),
                "test setup: nothing compacted before the restart"
            );
            stand_down(&service, &adapter).await;
            (session_id, anchor)
        };

        let (service, adapter) = build_service(temp.path(), &model);
        let resume_source = committed(&service, &session_id).await;
        let history = meerkat_core::agent::compact::estimate_tokens(resume_source.messages())
            .expect("estimate the history");
        // Past the window on top of the anchor, small enough for the rebuild
        // to keep it verbatim (the retention budget is threshold / 2 bytes),
        // and the history estimate stays under the threshold with it.
        let prompt_bytes = 2 * (RESTORED_WINDOW - anchor) + 8_000;
        assert!(
            prompt_bytes < 48_000,
            "test setup: the prompt fits the retention budget: {prompt_bytes}"
        );
        assert!(
            history + prompt_bytes / 4 + 1_000 < THRESHOLD,
            "test setup: the history estimate stays under the threshold: {history}"
        );
        materialize(&service, &adapter, resume_source, &tool).await;

        let prompt = "b".repeat(usize::try_from(prompt_bytes).expect("size"));
        let before = model.seen().len();
        let second = run_prompt(&adapter, &session_id, &prompt).await;
        let seen = model.seen_since(before);
        assert!(
            completed(&second),
            "the restored agent's first run completes: {second:?}; requests: {seen:?}"
        );
        assert!(
            seen.first().is_some_and(|seen| seen.summary),
            "the restored agent compacts before its first request: {seen:?}"
        );
        assert!(
            !seen.iter().any(|seen| seen.refused),
            "nothing is refused: {seen:?}"
        );
        let session = committed(&service, &session_id).await;
        assert!(has_compaction_summary(&session), "the compaction committed");
        let texts = user_texts(&session);
        assert!(
            texts.iter().any(|text| *text == prompt),
            "the restored run's prompt stays verbatim"
        );
        assert!(
            !texts.iter().any(|text| text.starts_with("aaaa")),
            "the earlier rows folded into the summary"
        );
    }
}
