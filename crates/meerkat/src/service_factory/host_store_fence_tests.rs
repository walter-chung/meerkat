//! A host store writer whose self-fence window closes dispatches nothing more:
//! a real native agent runs over the store checkpointer of a host store realm,
//! and the window closes while one of its round commits lands. A writer that
//! one timed-out renew stopped waits for its re-acquire instead: the turn goes
//! on once the re-acquire succeeds, and is refused as before when the
//! re-acquire fails, finds another writer's batch, or does not finish within
//! the self-fence window. Unrun until the candidate's validation.

use super::*;
use futures::stream;
use meerkat_client::{LlmDoneOutcome, LlmError, LlmEvent, LlmRequest};
use meerkat_core::host_store::{
    AcquireMode, CommitBatch, CommitId, FenceGrant, FenceTiming, FenceToken, HeadPage, HostStore,
    HostStoreError, Row, RowIndex, ScopeHead, ScopeKind, ScopeName, ScopeRef, ScopeVersion,
    VersionedHead,
};
use meerkat_core::service::{DeferredPromptPolicy, InitialTurnPolicy};
use meerkat_core::{
    AgentError, AgentToolDispatcher, Provider, ToolCallView, ToolDef, ToolDispatchOutcome,
    ToolError, ToolResult,
};
use meerkat_runtime::MeerkatMachine;
use meerkat_runtime::identifiers::LogicalRuntimeId;
use meerkat_runtime::input_state::InputLifecycleState;
use meerkat_runtime::store::spi::{SpiRealm, SpiRealmConfig, SpiRuntimeStore};
use meerkat_runtime::store::{InputStateRow, RuntimeStore};
use meerkat_session::PersistentSessionService;
use meerkat_store::{MemoryBlobStore, MemoryStore};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex as StdMutex, OnceLock, Weak};
use std::time::Duration;
use tempfile::TempDir;

const DEADLINE: Duration = Duration::from_secs(10);

/// A turn's deadline under paused time: a row whose re-acquire lists for
/// 30 s waits that long for the run's cleanup writes, at no wall clock cost.
const PAUSED_DEADLINE: Duration = Duration::from_secs(600);

/// How long a row waits for the writer to stop.
const STOP_DEADLINE: Duration = Duration::from_secs(30);

/// The host's renew interval on its 3 s lease leaves a self-fence window of
/// 2 s; the realm holds a renew to the interval (1 s).
const RENEW_INTERVAL: Duration = Duration::from_secs(1);

/// A re-acquire listing that answers within the 2 s window.
const LISTING_WITHIN_THE_WINDOW: Duration = Duration::from_millis(1500);

/// A re-acquire listing that answers long after the 2 s window.
const LISTING_PAST_THE_WINDOW: Duration = Duration::from_secs(30);

/// The scope another writer's batch creates in the foreign-batch row.
const FOREIGN_SCOPE: &str = "rt:another-writer";

/// Which step of the turn the writer stops during.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosesDuring {
    /// The round's first commit: the window is closed before any call of
    /// the round is dispatched.
    FirstCommit,
    /// The round's second commit: the window is closed before the next
    /// model call.
    SecondCommit,
    /// The first of the round's two calls, which run one at a time: the
    /// window is closed before the second call starts.
    FirstCall,
}

/// How the row stops the writer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Stop {
    /// Every renew from then on fails, the renew of each re-acquire
    /// included.
    #[default]
    FailingRenews,
    /// One renew never answers, and the realm's renew timeout gives up on
    /// it; later renews answer. What the re-acquire after it meets:
    TimedOutRenew(Reacquire),
}

/// What the re-acquire after a timed-out renew meets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reacquire {
    /// Its listing answers after this long; its acquire and its renew
    /// answer at once.
    ListsIn(Duration),
    /// Every acquire answers `Unavailable`.
    Fails,
    /// Another writer committed while this one was stopped: the listing
    /// shows that writer's batch.
    FindsForeignBatch,
}

/// Shared between the host, the model and the tools of one row.
#[derive(Default)]
struct FenceControl {
    /// How the row stops the writer.
    stop: Stop,
    /// Every renew from now on fails.
    fail_renews: AtomicBool,
    /// A renew failed since `fail_renews` was set.
    renew_failed: AtomicBool,
    /// The next host commit waits until the writer stopped, then lands.
    hold_next_commit: AtomicBool,
    /// The next renew never answers.
    hang_next_renew: AtomicBool,
    /// Every acquire from now on answers `Unavailable`.
    fail_acquires: AtomicBool,
    /// The next listing call answers after this long.
    list_delay: StdMutex<Option<Duration>>,
    /// The next listing finds a batch another writer committed.
    foreign_batch: AtomicBool,
    /// The realm, once it is open: a row waits on its own window.
    realm: OnceLock<Weak<SpiRealm>>,
}

impl FenceControl {
    /// Close the writer's window while the next host commit is in flight:
    /// the commit has passed the write path's own fence check when the host
    /// sees it, and the writer stops only from then on, so the commit lands
    /// and the window is closed by the time it answers.
    fn close_during_next_commit(&self) {
        self.hold_next_commit.store(true, Ordering::SeqCst);
    }

    /// Stop the writer with one renew that never answers, set up what the
    /// re-acquire after it meets, and wait until the realm's own window
    /// reads closed. A row that does not stop the writer this way returns at
    /// once.
    async fn stop_with_a_timed_out_renew(&self) {
        let Stop::TimedOutRenew(reacquire) = self.stop else {
            return;
        };
        match reacquire {
            Reacquire::ListsIn(delay) => {
                *self
                    .list_delay
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(delay);
            }
            Reacquire::Fails => self.fail_acquires.store(true, Ordering::SeqCst),
            Reacquire::FindsForeignBatch => self.foreign_batch.store(true, Ordering::SeqCst),
        }
        self.hang_next_renew.store(true, Ordering::SeqCst);
        let _ = tokio::time::timeout(STOP_DEADLINE, async {
            while !self.writer_stopped() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
    }

    /// Whether the realm's own window reads closed.
    fn writer_stopped(&self) -> bool {
        self.realm
            .get()
            .and_then(|realm| realm.upgrade())
            .is_some_and(|realm| !realm.lease_is_live())
    }
}

#[derive(Debug, Clone)]
struct StoredScope {
    version: ScopeVersion,
    commit_id: CommitId,
    head: Vec<u8>,
    start: u64,
    rows: BTreeMap<u64, Vec<u8>>,
    end: u64,
}

#[derive(Default)]
struct HostState {
    token: u64,
    scopes: BTreeMap<(ScopeKind, String), StoredScope>,
    landed: HashSet<String>,
}

/// An in-memory host store that meets its obligations at commit, with a
/// 1 s renew interval and a 3 s lease, whose renews, acquires, listings and
/// commits follow `control`.
struct FenceHost {
    state: StdMutex<HostState>,
    control: Arc<FenceControl>,
}

impl FenceHost {
    fn new(control: Arc<FenceControl>) -> Self {
        Self {
            state: StdMutex::new(HostState::default()),
            control,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, HostState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[async_trait]
impl HostStore for FenceHost {
    async fn acquire_fence(&self, _mode: AcquireMode) -> Result<FenceGrant, HostStoreError> {
        if self.control.fail_acquires.load(Ordering::SeqCst) {
            return Err(HostStoreError::Unavailable(
                "the row stops the acquires".to_string(),
            ));
        }
        let mut state = self.state();
        state.token += 1;
        Ok(FenceGrant {
            token: FenceToken::new(state.token),
            superseded_lease: Duration::ZERO,
        })
    }

    async fn renew_fence(&self, token: FenceToken) -> Result<(), HostStoreError> {
        if self.control.hang_next_renew.swap(false, Ordering::SeqCst) {
            // The realm's renew timeout gives up on this renew.
            std::future::pending::<()>().await;
        }
        if self.control.fail_renews.load(Ordering::SeqCst) {
            self.control.renew_failed.store(true, Ordering::SeqCst);
            return Err(HostStoreError::Unavailable(
                "the row stops the renews".to_string(),
            ));
        }
        let state = self.state();
        if token.get() != state.token {
            return Err(HostStoreError::FencedOut {
                current: FenceToken::new(state.token),
            });
        }
        Ok(())
    }

    fn fence_timing(&self) -> FenceTiming {
        FenceTiming::try_new(RENEW_INTERVAL, Duration::from_secs(3))
            .expect("a 1 s renew interval on a 3 s lease")
    }

    async fn load_head(&self, scope: &ScopeRef) -> Result<Option<VersionedHead>, HostStoreError> {
        Ok(self
            .state()
            .scopes
            .get(&(scope.kind, scope.name.as_str().to_string()))
            .map(|stored| VersionedHead {
                version: stored.version,
                commit_id: stored.commit_id.clone(),
                head: stored.head.clone(),
            }))
    }

    async fn list_heads(
        &self,
        kind: ScopeKind,
        after: Option<&ScopeName>,
        limit: usize,
    ) -> Result<HeadPage, HostStoreError> {
        let delay = self
            .control
            .list_delay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        if self.control.foreign_batch.swap(false, Ordering::SeqCst) {
            // Another writer's batch landed while this writer was stopped.
            self.state().scopes.insert(
                (ScopeKind::Session, FOREIGN_SCOPE.to_string()),
                StoredScope {
                    version: ScopeVersion::FIRST,
                    commit_id: CommitId::try_new("another-writers-batch")?,
                    head: Vec::new(),
                    start: 0,
                    rows: BTreeMap::new(),
                    end: 0,
                },
            );
        }
        let state = self.state();
        let mut heads = Vec::new();
        let mut more = false;
        for ((scope_kind, name), stored) in &state.scopes {
            if *scope_kind != kind || after.is_some_and(|after| name.as_str() <= after.as_str()) {
                continue;
            }
            if heads.len() == limit {
                more = true;
                break;
            }
            heads.push(ScopeHead {
                name: ScopeName::try_new(name.clone())?,
                head: VersionedHead {
                    version: stored.version,
                    commit_id: stored.commit_id.clone(),
                    head: stored.head.clone(),
                },
            });
        }
        let next = if more {
            heads.last().map(|head| head.name.clone())
        } else {
            None
        };
        Ok(HeadPage { heads, next })
    }

    async fn read_rows(
        &self,
        scope: &ScopeRef,
        from: RowIndex,
        limit: usize,
    ) -> Result<Vec<Row>, HostStoreError> {
        let state = self.state();
        let Some(stored) = state
            .scopes
            .get(&(scope.kind, scope.name.as_str().to_string()))
        else {
            return Ok(Vec::new());
        };
        stored
            .rows
            .range(from.get().max(stored.start)..)
            .take(limit)
            .map(|(_, row)| Row::try_new(row.clone()))
            .collect()
    }

    async fn commit(&self, batch: &CommitBatch) -> Result<(), HostStoreError> {
        if self.control.hold_next_commit.swap(false, Ordering::SeqCst) {
            // The writer passed its own fence check before this commit; it
            // stops while the commit is in flight.
            match self.control.stop {
                Stop::FailingRenews => {
                    self.control.fail_renews.store(true, Ordering::SeqCst);
                    let _ = tokio::time::timeout(Duration::from_secs(5), async {
                        while !self.control.renew_failed.load(Ordering::SeqCst) {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                    })
                    .await;
                }
                Stop::TimedOutRenew(_) => self.control.stop_with_a_timed_out_renew().await,
            }
        }
        let mut state = self.state();
        if state.landed.contains(batch.commit_id().as_str()) {
            return Ok(());
        }
        if batch.fence().get() < state.token {
            return Err(HostStoreError::FencedOut {
                current: FenceToken::new(state.token),
            });
        }
        for update in batch.updates() {
            let key = (update.scope.kind, update.scope.name.as_str().to_string());
            let current = state.scopes.get(&key).map(|stored| stored.version);
            if current != update.previous {
                return Err(HostStoreError::Conflict {
                    scope: update.scope.clone(),
                    current,
                });
            }
        }
        for update in batch.updates() {
            let key = (update.scope.kind, update.scope.name.as_str().to_string());
            if update.delete {
                state.scopes.remove(&key);
                continue;
            }
            let stored = state.scopes.entry(key).or_insert_with(|| StoredScope {
                version: update.next,
                commit_id: batch.commit_id().clone(),
                head: Vec::new(),
                start: 0,
                rows: BTreeMap::new(),
                end: 0,
            });
            if let Some(truncate) = update.truncate_after {
                stored.rows.retain(|index, _| *index < truncate.get());
                stored.end = stored.end.min(truncate.get());
            }
            if let Some(drop_before) = update.drop_before {
                stored.rows.retain(|index, _| *index >= drop_before.get());
                stored.start = stored.start.max(drop_before.get());
            }
            for row in &update.append {
                stored.rows.insert(stored.end, row.as_bytes().to_vec());
                stored.end += 1;
            }
            if let Some(head) = &update.head {
                stored.head = head.clone();
            }
            stored.version = update.next;
            stored.commit_id = batch.commit_id().clone();
        }
        state.landed.insert(batch.commit_id().as_str().to_string());
        Ok(())
    }
}

/// Asks for the row's calls of `probe` (one, or two when the window closes
/// during the first call), then answers with text. Counts its requests.
struct FenceModel {
    requests: AtomicUsize,
    control: Arc<FenceControl>,
    closes_during: ClosesDuring,
}

#[async_trait]
impl LlmClient for FenceModel {
    fn project_replay_messages(
        &self,
        messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::Message>, LlmError> {
        Ok(messages.to_vec())
    }

    fn stream<'a>(
        &'a self,
        request: &'a LlmRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = Result<LlmEvent, LlmError>> + Send + 'a>> {
        let first = self.requests.fetch_add(1, Ordering::SeqCst) == 0;
        let usage = meerkat_core::TurnUsage::host_declared(
            Provider::Anthropic,
            &request.model,
            Default::default(),
        );
        if first {
            if self.closes_during == ClosesDuring::FirstCommit {
                self.control.close_during_next_commit();
            }
            let mut events: Vec<Result<LlmEvent, LlmError>> =
                vec![Ok(LlmEvent::ToolCallComplete {
                    id: "probe-call".into(),
                    name: "probe".into(),
                    args: json!({}),
                    meta: None,
                })];
            if self.closes_during == ClosesDuring::FirstCall {
                events.push(Ok(LlmEvent::ToolCallComplete {
                    id: "probe-call-2".into(),
                    name: "probe".into(),
                    args: json!({}),
                    meta: None,
                }));
            }
            events.push(Ok(LlmEvent::UsageUpdate { usage }));
            events.push(Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::success(meerkat_core::StopReason::ToolUse),
            }));
            return Box::pin(stream::iter(events));
        }
        Box::pin(stream::iter(vec![
            Ok(LlmEvent::TextDelta {
                delta: "probed".into(),
                meta: None,
            }),
            Ok(LlmEvent::UsageUpdate { usage }),
            Ok(LlmEvent::Done {
                outcome: LlmDoneOutcome::success(meerkat_core::StopReason::EndTurn),
            }),
        ]))
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    async fn health_check(&self) -> Result<(), LlmError> {
        Ok(())
    }
}

/// One tool, `probe`. Counts its dispatches.
struct FenceTools {
    dispatched: AtomicUsize,
    control: Arc<FenceControl>,
    closes_during: ClosesDuring,
}

#[async_trait]
impl AgentToolDispatcher for FenceTools {
    fn tools(&self) -> Arc<[Arc<ToolDef>]> {
        vec![Arc::new(ToolDef::new(
            "probe",
            "a probe",
            json!({"type":"object"}),
        ))]
        .into()
    }

    async fn dispatch(&self, call: ToolCallView<'_>) -> Result<ToolDispatchOutcome, ToolError> {
        let first = self.dispatched.fetch_add(1, Ordering::SeqCst) == 0;
        match self.closes_during {
            ClosesDuring::SecondCommit => self.control.close_during_next_commit(),
            ClosesDuring::FirstCall if first => self.control.stop_with_a_timed_out_renew().await,
            ClosesDuring::FirstCommit | ClosesDuring::FirstCall => {}
        }
        Ok(ToolDispatchOutcome::sync_result(ToolResult::new(
            call.id.into(),
            "probed".into(),
            false,
        )))
    }
}

/// A real native agent through the factory, with the row's model, tools
/// and configuration.
struct FenceBuilder {
    factory: AgentFactory,
    client: Arc<FenceModel>,
    tools: Arc<FenceTools>,
    config: Config,
}

#[async_trait]
impl SessionAgentBuilder for FenceBuilder {
    type Agent = FactoryAgent;

    async fn build_agent(
        &self,
        request: &CreateSessionRequest,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Result<FactoryAgent, SessionError> {
        let mut config = AgentBuildConfig::from_create_session_request(request, event_tx);
        config.llm_client_override = Some(self.client.clone());
        config.tool_dispatcher_override = Some(self.tools.clone());
        let session_context = match request
            .build
            .as_ref()
            .map(|build| &build.runtime_build_mode)
        {
            Some(meerkat_core::RuntimeBuildMode::SessionOwned(bindings)) => {
                Some(Arc::clone(bindings.session_context()))
            }
            _ => None,
        };
        let factory = self.factory.clone();
        let agent_config = self.config.clone();
        let agent = meerkat_runtime::stack_relief::relieve_caller_stack(move || async move {
            factory.build_agent(config, &agent_config).await
        })
        .await
        .map_err(|error| SessionError::Agent(AgentError::InternalError(error.to_string())))?;
        Ok(FactoryAgent {
            agent,
            session_context,
            pending_head_canonical_boundary: None,
            acknowledged_head_canonical_boundary: None,
        })
    }
}

/// What a run whose writer stopped during one of its steps did.
struct FencedRun {
    failed: bool,
    error: String,
    model_requests: usize,
    dispatched: usize,
    lease_live_after: bool,
    poisoned_after: bool,
    /// The phase of each of the session's input rows after the turn (`None`
    /// for a row that does not decode), when the store answered.
    input_phases: Option<Vec<Option<InputLifecycleState>>>,
    /// The realm, kept open for a row that waits on it after the turn.
    realm: Arc<SpiRealm>,
}

/// Run one turn of a fresh session over a host store realm whose writer's
/// window closes during the round commit `closes_during` names, through
/// renews that fail from then on.
async fn run_with_window_closing_during(closes_during: ClosesDuring) -> FencedRun {
    run_with_stop(closes_during, Stop::FailingRenews).await
}

/// Run one turn of a fresh session over a host store realm whose writer
/// stops, the way `stop` names, during the step `closes_during` names.
async fn run_with_stop(closes_during: ClosesDuring, stop: Stop) -> FencedRun {
    let temp = TempDir::new().expect("temp");
    let control = Arc::new(FenceControl {
        stop,
        ..FenceControl::default()
    });
    let host: Arc<dyn HostStore> = Arc::new(FenceHost::new(Arc::clone(&control)));
    let realm = SpiRealm::open(host, SpiRealmConfig::default())
        .await
        .expect("the realm opens");
    let _ = control.realm.set(Arc::downgrade(&realm));
    let runtime_store = Arc::new(
        SpiRuntimeStore::open(Arc::clone(&realm))
            .await
            .expect("the runtime store opens"),
    );
    let store = Arc::new(MemoryStore::new());
    let blob_store = Arc::new(MemoryBlobStore::new());
    let machine = Arc::new(
        MeerkatMachine::persistent(runtime_store.clone(), blob_store.clone())
            .expect("canonical machine"),
    );
    let client = Arc::new(FenceModel {
        requests: AtomicUsize::new(0),
        control: Arc::clone(&control),
        closes_during,
    });
    let tools = Arc::new(FenceTools {
        dispatched: AtomicUsize::new(0),
        control: Arc::clone(&control),
        closes_during,
    });
    let mut config = Config::default();
    if closes_during == ClosesDuring::FirstCall {
        // One call at a time: the round's second call asks the fence once
        // the first one has run.
        config.tools.max_concurrent = 1;
    }
    let builder = FenceBuilder {
        factory: AgentFactory::new(temp.path().join("sessions")).session_store(store.clone()),
        client: client.clone(),
        tools: tools.clone(),
        config,
    };
    let service = Arc::new(
        PersistentSessionService::new(builder, 1, store.clone(), runtime_store.clone(), blob_store)
            .with_canonical_runtime_adapter(machine.clone()),
    );
    let request = CreateSessionRequest {
        model: "claude-sonnet-4-5".into(),
        prompt: "probe once".into(),
        injected_context: Vec::new(),
        system_prompt: meerkat_core::SystemPromptOverride::Inherit,
        max_tokens: None,
        event_tx: None,
        initial_turn: InitialTurnPolicy::RunImmediately,
        deferred_prompt_policy: DeferredPromptPolicy::Discard,
        build: None,
        labels: None,
    };
    let (request, initial_turn) =
        crate::surface::split_runtime_backed_eager_create_request(request);
    let created = Box::pin(crate::surface::materialize_session(
        &service,
        &machine,
        meerkat_core::Session::new(),
        request,
        {
            let service = service.clone();
            let machine = machine.clone();
            move |id| crate::surface::default_persistent_executor(service, machine, id)
        },
    ))
    .await
    .expect("materialize the session and its executor");
    let deadline = match stop {
        Stop::FailingRenews => DEADLINE,
        Stop::TimedOutRenew(_) => PAUSED_DEADLINE,
    };
    let turn = tokio::time::timeout(
        deadline,
        crate::surface::run_runtime_backed_initial_turn_with_machine(
            &service,
            &machine,
            &created.session_id,
            initial_turn.expect("initial turn"),
        ),
    )
    .await
    .expect("the turn ends");
    let input_phases: Option<Vec<Option<InputLifecycleState>>> = runtime_store
        .load_input_states(&LogicalRuntimeId::for_session(&created.session_id))
        .await
        .ok()
        .map(|rows| {
            rows.iter()
                .map(|row| match row {
                    InputStateRow::Decoded(stored) => Some(stored.seed.phase),
                    _ => None,
                })
                .collect()
        });
    FencedRun {
        failed: turn.is_err(),
        error: turn
            .err()
            .map(|error| format!("{error:?}"))
            .unwrap_or_default(),
        model_requests: client.requests.load(Ordering::SeqCst),
        dispatched: tools.dispatched.load(Ordering::SeqCst),
        lease_live_after: realm.lease_is_live(),
        poisoned_after: realm.realm_poisoned().is_some(),
        input_phases,
        realm,
    }
}

/// A turn whose writer one timed-out renew stopped at `site`, and whose
/// re-acquire then succeeded, ran to its end: the dispatch at `site` waited
/// for the re-acquire, and nothing ran twice.
fn assert_completed_once(run: &FencedRun, site: &str, calls: usize) {
    assert!(
        !run.failed,
        "{site}: the dispatch waits for the re-acquire, and the turn completes: {}",
        run.error
    );
    assert_eq!(
        run.model_requests, 2,
        "{site}: the tool round and the answer, each asked once"
    );
    assert_eq!(
        run.dispatched, calls,
        "{site}: each call of the round ran once"
    );
    assert!(
        run.lease_live_after,
        "{site}: the re-acquire resumed the writer"
    );
    assert!(!run.poisoned_after, "{site}: no other writer showed up");
    assert_eq!(
        run.input_phases,
        Some(vec![Some(InputLifecycleState::Consumed)]),
        "{site}: the turn's one input is consumed once, not stopped or queued again"
    );
}

/// A turn whose writer stopped while the round's second commit was in
/// flight, and whose re-acquire did not restore its window, made no model
/// call after the stop: the dispatch fence refused it, as before.
fn assert_refused_after_the_round(run: &FencedRun, row: &str) {
    assert!(run.failed, "{row}: the run stops at the closed window");
    assert!(
        run.error.contains("self-fence window closed"),
        "{row}: the dispatch fence refused it, not the write path: {}",
        run.error
    );
    assert_eq!(
        run.dispatched, 1,
        "{row}: the call ran once, before the writer stopped"
    );
    assert_eq!(run.model_requests, 1, "{row}: no model call after the stop");
}

/// A host store writer whose self-fence window has closed makes no model
/// call and dispatches no tool, through the real store checkpointer. Its
/// round commit passes the write path's own fence check, and the window
/// closes while that commit is in flight, so only the dispatch fence stands
/// between the commit and the next step:
/// - closed during the round's first commit, the round's call is never
///   dispatched, and no model call follows;
/// - closed during the round's second commit, the call ran once, while the
///   window was open, and the next model call is refused.
///
/// Every renew fails from then on, the renew of each re-acquire included,
/// so the writer never gets its window back.
#[tokio::test]
async fn a_writer_past_its_fence_window_makes_no_model_call_and_dispatches_no_tool() {
    let first = run_with_window_closing_during(ClosesDuring::FirstCommit).await;
    assert!(!first.lease_live_after, "the window closed");
    assert!(first.failed, "the run stops at the closed window");
    assert!(
        first.error.contains("self-fence window closed"),
        "the dispatch fence refused it, not the write path: {}",
        first.error
    );
    assert_eq!(first.dispatched, 0, "the round's call is never dispatched");
    assert_eq!(first.model_requests, 1, "no model call follows");

    let second = run_with_window_closing_during(ClosesDuring::SecondCommit).await;
    assert!(!second.lease_live_after, "the window closed");
    assert!(second.failed, "the run stops at the closed window");
    assert!(
        second.error.contains("self-fence window closed"),
        "the dispatch fence refused it, not the write path: {}",
        second.error
    );
    assert_eq!(
        second.dispatched, 1,
        "the call ran once, before the window closed"
    );
    assert_eq!(second.model_requests, 1, "the next model call is refused");
}

/// A writer that one timed-out renew stopped while the round's second
/// commit was in flight waits, at the next model call, for the re-acquire
/// that its next write would make. The re-acquire's listing answers 1.5 s
/// later, within the 2 s window, so the model call goes ahead and the turn
/// completes with its input consumed once. Before the fix the model call was
/// refused at once and the turn failed with the teardown class
/// (`DurableProjectionAuthorityUnknown`, "self-fence window closed ...
/// renew failed: true").
#[tokio::test(start_paused = true)]
async fn a_model_call_after_a_timed_out_renew_waits_for_the_reacquire() {
    let run = run_with_stop(
        ClosesDuring::SecondCommit,
        Stop::TimedOutRenew(Reacquire::ListsIn(LISTING_WITHIN_THE_WINDOW)),
    )
    .await;
    assert_completed_once(&run, "the model call after the round", 1);
}

/// A writer that one timed-out renew stopped while the round's first commit
/// was in flight waits, before it admits the round's call, for the
/// re-acquire; the call then runs once and the turn completes. Before the
/// fix the round was refused at once and the turn failed.
#[tokio::test(start_paused = true)]
async fn a_tool_round_after_a_timed_out_renew_waits_for_the_reacquire() {
    let run = run_with_stop(
        ClosesDuring::FirstCommit,
        Stop::TimedOutRenew(Reacquire::ListsIn(LISTING_WITHIN_THE_WINDOW)),
    )
    .await;
    assert_completed_once(&run, "the round after its first commit", 1);
}

/// A writer that one timed-out renew stopped while the round's first call
/// ran (two calls, one at a time) waits, right before the second call
/// starts, for the re-acquire; the second call then runs once and the turn
/// completes. Before the fix the second call was refused ("this writer's
/// self-fence window closed before the call started") and the turn failed.
#[tokio::test(start_paused = true)]
async fn a_tool_call_after_a_timed_out_renew_waits_for_the_reacquire() {
    let run = run_with_stop(
        ClosesDuring::FirstCall,
        Stop::TimedOutRenew(Reacquire::ListsIn(LISTING_WITHIN_THE_WINDOW)),
    )
    .await;
    assert_completed_once(&run, "the second call of the round", 2);
}

/// A re-acquire that fails (every acquire answers a store error) restores
/// nothing: the waiting model call is refused as before, the writer stays
/// stopped, and the realm is not poisoned.
#[tokio::test(start_paused = true)]
async fn a_failed_reacquire_still_refuses_the_dispatch() {
    let run = run_with_stop(
        ClosesDuring::SecondCommit,
        Stop::TimedOutRenew(Reacquire::Fails),
    )
    .await;
    assert_refused_after_the_round(&run, "a failed re-acquire");
    assert!(!run.lease_live_after, "the writer stays stopped");
    assert!(!run.poisoned_after, "a failed acquire poisons nothing");
}

/// A re-acquire whose listing shows a batch another writer committed while
/// this one was stopped poisons the realm: the waiting model call is
/// refused, and nothing more is dispatched.
#[tokio::test(start_paused = true)]
async fn a_foreign_batch_at_the_reacquire_refuses_the_dispatch() {
    let run = run_with_stop(
        ClosesDuring::SecondCommit,
        Stop::TimedOutRenew(Reacquire::FindsForeignBatch),
    )
    .await;
    assert_refused_after_the_round(&run, "another writer's batch");
    assert!(run.poisoned_after, "the foreign batch poisons the realm");
    assert!(!run.lease_live_after, "the writer never resumes");
}

/// The wait is bounded by the self-fence window: a re-acquire whose listing
/// takes 30 s is not waited for, and the model call is refused as before.
/// The re-acquire itself goes on and resumes the writer once its listing
/// answers; only the dispatch gave up.
#[tokio::test(start_paused = true)]
async fn a_reacquire_slower_than_the_window_refuses_the_dispatch() {
    let run = run_with_stop(
        ClosesDuring::SecondCommit,
        Stop::TimedOutRenew(Reacquire::ListsIn(LISTING_PAST_THE_WINDOW)),
    )
    .await;
    assert_refused_after_the_round(&run, "a re-acquire slower than the window");
    assert!(!run.poisoned_after, "a slow listing poisons nothing");
    tokio::time::sleep(LISTING_PAST_THE_WINDOW + Duration::from_secs(5)).await;
    assert!(
        run.realm.lease_is_live(),
        "the slow re-acquire resumes the writer once its listing answers"
    );
}
