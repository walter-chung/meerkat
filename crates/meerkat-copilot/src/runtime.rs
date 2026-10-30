use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration as StdDuration;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use futures::StreamExt;
use meerkat_auth_core::auth_oauth::{exchange_refresh_token, oauth_refresh_error};
use meerkat_auth_core::resolver::{
    LockedManagedStoreOAuthRefresh, ManagedStoreOAuthRefreshPreparationSlot,
    OAuthLoginCredentialAdmission, load_managed_store_tokens_with_lifecycle,
    prepare_managed_store_oauth_refresh_under_lock, resolve_oauth_login_credential_disposition,
    resolve_simple_secret,
};
use meerkat_core::auth::{
    PersistedAuthMode, PersistedTokens, ProviderAuthPersistence, RefreshError, RefreshFn,
    TokenStoreAccess,
};
use meerkat_core::provider_matrix::{AnthropicAuthMethod, GoogleAuthMethod, OpenAiAuthMethod};
use meerkat_core::{
    AuthCredentialIdentity, AuthError, AuthMetadata, CredentialSourceSpec,
    HttpAuthorizationReceipt, HttpAuthorizationRequest, HttpAuthorizationResponse,
    HttpAuthorizationResponseAction, HttpAuthorizer, Provider,
};
use meerkat_llm_core::provider_runtime::{
    NormalizedAuthMethod, ProviderAuthError, ProviderClientError, ResolverEnvironment,
    ValidatedBinding,
};
use parking_lot::{Mutex, RwLock};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::protocol::{
    CopilotBackendConfig, CopilotEndpoint, CopilotModelSnapshot, CopilotModelsEnvelope,
    CopilotTokenEndpoints, CopilotTokenEnvelope, CopilotTokenErrorEnvelope,
    DEFAULT_COPILOT_API_BASE,
};

const COPILOT_USER_URL: &str = "https://api.github.com/copilot_internal/user";
const CAPI_EXPIRY_SKEW_SECONDS: i64 = 300;
const MODEL_DISCOVERY_RETRY_SECONDS: u64 = 30;
const MODEL_DISCOVERY_REFRESH_SECONDS: u64 = 300;
const MODEL_DISCOVERY_MAX_RETRY_SECONDS: u64 = 3600;
const BEARER_SCHEME: &str = "Bearer";
const COPILOT_HTTP_TIMEOUT: StdDuration = StdDuration::from_secs(30);
const MAX_COPILOT_HTTP_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy)]
enum CopilotRequestIntent {
    ModelAccess,
    ConversationPanel,
}

impl CopilotRequestIntent {
    fn as_str(self) -> &'static str {
        match self {
            Self::ModelAccess => "model-access",
            Self::ConversationPanel => "conversation-panel",
        }
    }
}

struct CopilotHttpResponse {
    status: u16,
    body: Vec<u8>,
    retry_after: Option<StdDuration>,
}

#[derive(Debug, thiserror::Error)]
enum CopilotHttpTransportError {
    #[error("Copilot HTTP request failed")]
    Request(#[source] reqwest::Error),
    #[error("Copilot HTTP request timed out")]
    Timeout,
    #[error("Copilot HTTP response exceeded {MAX_COPILOT_HTTP_RESPONSE_BYTES} bytes")]
    ResponseTooLarge,
    /// The endpoint answered with a redirect, refused by its status alone
    /// (no `Location`, header or body is read).
    #[error(
        "Copilot HTTP endpoint answered with a redirect (status {status}); redirects are refused"
    )]
    RedirectRefused { status: u16 },
    #[error(transparent)]
    ClientUnavailable(meerkat_auth_core::auth_oauth::CredentialHttpClientUnavailable),
    #[cfg(test)]
    #[error("{0}")]
    Scripted(String),
}

#[derive(Debug, thiserror::Error)]
enum CopilotModelDiscoveryError {
    #[error("Copilot model discovery transport failed")]
    Transport(#[source] CopilotHttpTransportError),
    #[error("Copilot model discovery returned HTTP {status}: {body}")]
    Http {
        status: u16,
        body: String,
        retry_after: Option<StdDuration>,
    },
    #[error("invalid Copilot model-discovery request headers")]
    Header(#[source] ProviderAuthError),
    #[error("invalid Copilot model-discovery response")]
    Decode(#[source] serde_json::Error),
    #[error("invalid Copilot model-discovery protocol response")]
    Protocol(#[source] crate::protocol::CopilotProtocolError),
}

impl CopilotModelDiscoveryError {
    fn status(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } => Some(*status),
            Self::Transport(_) | Self::Header(_) | Self::Decode(_) | Self::Protocol(_) => None,
        }
    }

    fn retry_delay(&self) -> StdDuration {
        match self {
            Self::Http {
                retry_after: Some(delay),
                ..
            } => (*delay).min(StdDuration::from_secs(MODEL_DISCOVERY_MAX_RETRY_SECONDS)),
            Self::Transport(_)
            | Self::Header(_)
            | Self::Decode(_)
            | Self::Protocol(_)
            | Self::Http { .. } => StdDuration::from_secs(MODEL_DISCOVERY_RETRY_SECONDS),
        }
    }
}

#[async_trait]
trait CopilotHttpTransport: Send + Sync {
    async fn get(
        &self,
        url: &str,
        headers: HeaderMap,
    ) -> Result<CopilotHttpResponse, CopilotHttpTransportError>;

    async fn get_pat(
        &self,
        url: &str,
        headers: HeaderMap,
    ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
        self.get(url, headers).await
    }
}

/// Credential-bearing Copilot requests follow no redirects. A client build
/// failure is kept and every request fails with it.
struct ReqwestCopilotHttpTransport {
    client: Result<reqwest::Client, meerkat_auth_core::auth_oauth::CredentialHttpClientUnavailable>,
}

impl ReqwestCopilotHttpTransport {
    fn new() -> Self {
        Self {
            client: meerkat_auth_core::auth_oauth::credential_http_client(),
        }
    }
}

#[async_trait]
impl CopilotHttpTransport for ReqwestCopilotHttpTransport {
    async fn get(
        &self,
        url: &str,
        headers: HeaderMap,
    ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
        self.get_response(url, headers, false).await
    }

    async fn get_pat(
        &self,
        url: &str,
        headers: HeaderMap,
    ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
        self.get_response(url, headers, true).await
    }
}

impl ReqwestCopilotHttpTransport {
    async fn get_response(
        &self,
        url: &str,
        headers: HeaderMap,
        reject_before_body: bool,
    ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
        let mut rejection = None;
        let result = tokio::time::timeout(COPILOT_HTTP_TIMEOUT, async {
            let response = self
                .client
                .as_ref()
                .map_err(|error| CopilotHttpTransportError::ClientUnavailable(*error))?
                .get(url)
                .headers(headers)
                .send()
                .await
                .map_err(CopilotHttpTransportError::Request)?;
            if response.status().is_redirection() {
                return Err(CopilotHttpTransportError::RedirectRefused {
                    status: response.status().as_u16(),
                });
            }
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(StdDuration::from_secs);
            if matches!(status, 401 | 403) {
                if reject_before_body {
                    return Ok(CopilotHttpResponse {
                        status,
                        body: Vec::new(),
                        retry_after,
                    });
                }
                rejection = Some((status, retry_after));
            }
            if response
                .content_length()
                .is_some_and(|length| length > MAX_COPILOT_HTTP_RESPONSE_BYTES as u64)
            {
                return Err(CopilotHttpTransportError::ResponseTooLarge);
            }
            let mut stream = response.bytes_stream();
            let mut body = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(CopilotHttpTransportError::Request)?;
                if body.len().saturating_add(chunk.len()) > MAX_COPILOT_HTTP_RESPONSE_BYTES {
                    return Err(CopilotHttpTransportError::ResponseTooLarge);
                }
                body.extend_from_slice(&chunk);
            }
            Ok(CopilotHttpResponse {
                status,
                body,
                retry_after,
            })
        })
        .await
        .unwrap_or(Err(CopilotHttpTransportError::Timeout));
        match (result, rejection) {
            // A failed body cannot turn an observed credential rejection into
            // a transient discovery outage. Valid OAuth error bodies stay intact.
            (Err(_), Some((status, retry_after))) => Ok(CopilotHttpResponse {
                status,
                body: Vec::new(),
                retry_after,
            }),
            (result, _) => result,
        }
    }
}

pub struct CopilotRuntime {
    accounts: Mutex<BTreeMap<CopilotAccountCacheKey, Weak<CopilotAccountState>>>,
    transport: Arc<dyn CopilotHttpTransport>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CopilotAccountCacheKey {
    persistence_id: Option<meerkat_core::auth::ProviderAuthPersistenceId>,
    credential_identity: AuthCredentialIdentity,
}

#[derive(PartialEq, Eq)]
enum CopilotCredential {
    ManagedOauth,
    EnvironmentPat {
        source: CredentialSourceSpec,
        secret: String,
    },
}

impl CopilotRuntime {
    pub fn new() -> Self {
        Self {
            accounts: Mutex::new(BTreeMap::new()),
            transport: Arc::new(ReqwestCopilotHttpTransport::new()),
        }
    }

    pub async fn resolve(
        &self,
        binding: &ValidatedBinding,
        env: &ResolverEnvironment,
    ) -> Result<CopilotResolvedAuth, ProviderAuthError> {
        let is_pat = matches!(
            binding.auth(),
            NormalizedAuthMethod::OpenAi(OpenAiAuthMethod::GitHubCopilotPat)
                | NormalizedAuthMethod::Anthropic(AnthropicAuthMethod::GitHubCopilotPat)
                | NormalizedAuthMethod::Google(GoogleAuthMethod::GitHubCopilotPat)
        );
        let (persistence_id, credential) = if is_pat {
            if matches!(
                binding.credential_identity(),
                AuthCredentialIdentity::Account(_)
            ) || !matches!(
                binding.auth_profile().source,
                CredentialSourceSpec::Env { .. }
            ) {
                return Err(ProviderAuthError::SourceResolutionFailed(
                    "github_copilot_pat requires source.kind = env without credential_account"
                        .to_string(),
                ));
            }
            let secret =
                resolve_simple_secret(&binding.auth_profile().source, env, binding).await?;
            if secret.trim().is_empty() {
                return Err(ProviderAuthError::Auth(AuthError::MissingSecret));
            }
            (
                None,
                CopilotCredential::EnvironmentPat {
                    source: binding.auth_profile().source.clone(),
                    secret,
                },
            )
        } else {
            if !matches!(
                binding.credential_identity(),
                AuthCredentialIdentity::Account(_)
            ) {
                return Err(ProviderAuthError::SourceResolutionFailed(
                    "Copilot backend requires an account-scoped credential identity".to_string(),
                ));
            }
            if binding.auth().persisted_auth_mode() != Some(PersistedAuthMode::GithubCopilotOauth) {
                return Err(ProviderAuthError::SourceResolutionFailed(
                    "Copilot backend requires github_copilot_oauth or github_copilot_pat"
                        .to_string(),
                ));
            }
            if !matches!(
                binding.auth_profile().source,
                CredentialSourceSpec::ManagedStore
            ) {
                return Err(ProviderAuthError::SourceResolutionFailed(
                    "github_copilot_oauth requires source.kind = managed_store".to_string(),
                ));
            }
            let persistence = env
                .provider_auth_persistence()
                .filter(|_| env.auth_lease_handle.is_some())
                .ok_or_else(|| {
                    ProviderAuthError::SourceResolutionFailed(
                        "Copilot requires provider-auth persistence and AuthMachine authority"
                            .to_string(),
                    )
                })?;
            (
                Some(persistence.authority_id()),
                CopilotCredential::ManagedOauth,
            )
        };
        let mut config = CopilotBackendConfig::from_options(&binding.backend_profile().options)
            .map_err(|error| ProviderAuthError::SourceResolutionFailed(error.to_string()))?;
        if is_pat
            && binding
                .backend_profile()
                .options
                .get("integration_id")
                .is_none()
        {
            config.integration_id = "copilot-developer-cli".to_string();
        }
        let state = self.account_state(
            persistence_id,
            binding.credential_identity(),
            config,
            credential,
        )?;
        let authorizer = Arc::new(CopilotAuthorizer {
            state,
            binding: binding.clone(),
            env: env.clone(),
            transport: Arc::clone(&self.transport),
        });
        authorizer.prime().await?;
        let metadata =
            meerkat_auth_core::resolver::finalize_auth_metadata(binding, AuthMetadata::default())?;
        Ok(CopilotResolvedAuth {
            authorizer,
            metadata,
        })
    }

    fn account_state(
        &self,
        persistence_id: Option<meerkat_core::auth::ProviderAuthPersistenceId>,
        identity: &AuthCredentialIdentity,
        config: CopilotBackendConfig,
        credential: CopilotCredential,
    ) -> Result<Arc<CopilotAccountState>, ProviderAuthError> {
        let key = CopilotAccountCacheKey {
            persistence_id,
            credential_identity: identity.clone(),
        };
        let mut accounts = self.accounts.lock();
        accounts.retain(|_, state| state.strong_count() > 0);
        if let Some(existing) = accounts.get(&key).and_then(Weak::upgrade) {
            if existing.credential != credential {
                return Err(ProviderAuthError::SourceResolutionFailed(
                    "Copilot binding already uses a different credential source; restart to rotate the PAT"
                        .to_string(),
                ));
            }
            if existing.config != config {
                return Err(ProviderAuthError::SourceResolutionFailed(format!(
                    "Copilot routes sharing credential identity '{identity}' disagree on backend options"
                )));
            }
            return Ok(existing);
        }
        let state = Arc::new(CopilotAccountState {
            config,
            credential,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        accounts.insert(key, Arc::downgrade(&state));
        Ok(state)
    }

    pub fn resolve_route(
        &self,
        connection: &meerkat_llm_core::provider_runtime::ResolvedConnection,
        provider: Provider,
        model: &str,
    ) -> Option<CopilotRouteResolution> {
        let persistence_id = match connection.auth_lease.kind() {
            meerkat_core::ResolvedAuthKind::DynamicAuthorizer(authorizer) => {
                authorizer.persistence_authority_id()
            }
            _ => return None,
        };
        let key = CopilotAccountCacheKey {
            persistence_id,
            credential_identity: connection.credential_identity.clone(),
        };
        let Some(state) = self.accounts.lock().get(&key).and_then(Weak::upgrade) else {
            return None;
        };
        resolve_route_for_state(&state, provider, model)
    }
}

fn resolve_route_for_state(
    state: &Arc<CopilotAccountState>,
    provider: Provider,
    model: &str,
) -> Option<CopilotRouteResolution> {
    let cache = state.cache.read();
    let cached = cache.as_ref()?;
    let (access, capabilities, available_models) = match cached.models.as_ref() {
        Some(snapshot) => {
            let model = snapshot.model(model);
            let access = model.map_or(CopilotModelAccess::Unavailable, |model| {
                model
                    .route_for(provider)
                    .map_or(CopilotModelAccess::Unavailable, |endpoint| {
                        CopilotModelAccess::Available { endpoint }
                    })
            });
            (
                access,
                model.map(|model| model.capabilities.clone()),
                Some(Arc::clone(snapshot)),
            )
        }
        None => (CopilotModelAccess::Unknown, None, None),
    };
    Some(CopilotRouteResolution {
        access,
        api_base: cached.api_base.clone(),
        capabilities,
        available_models,
        state: Arc::downgrade(state),
        provider,
        model: model.to_string(),
    })
}

impl Default for CopilotRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct CopilotResolvedAuth {
    authorizer: Arc<CopilotAuthorizer>,
    metadata: AuthMetadata,
}

impl CopilotResolvedAuth {
    pub fn authorizer(&self) -> Arc<dyn HttpAuthorizer> {
        self.authorizer.clone()
    }

    pub fn concrete_authorizer(&self) -> Arc<CopilotAuthorizer> {
        self.authorizer.clone()
    }

    pub fn metadata(&self) -> &AuthMetadata {
        &self.metadata
    }

    pub fn model_snapshot(&self) -> Option<Arc<CopilotModelSnapshot>> {
        self.authorizer.model_snapshot()
    }

    pub fn route_for(&self, provider: Provider, model: &str) -> CopilotModelAccess {
        let Some(snapshot) = self.model_snapshot() else {
            return CopilotModelAccess::Unknown;
        };
        let Some(model_access) = snapshot.model(model) else {
            return CopilotModelAccess::Unavailable;
        };
        model_access
            .route_for(provider)
            .map_or(CopilotModelAccess::Unavailable, |endpoint| {
                CopilotModelAccess::Available { endpoint }
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopilotModelAccess {
    Available { endpoint: CopilotEndpoint },
    Unavailable,
    Unknown,
}

pub struct CopilotRouteResolution {
    pub access: CopilotModelAccess,
    pub api_base: String,
    pub capabilities: Option<crate::protocol::CopilotModelCapabilities>,
    available_models: Option<Arc<CopilotModelSnapshot>>,
    state: Weak<CopilotAccountState>,
    provider: Provider,
    model: String,
}

impl CopilotRouteResolution {
    pub fn bind_authorizer(&self, inner: Arc<dyn HttpAuthorizer>) -> Arc<dyn HttpAuthorizer> {
        Arc::new(CopilotRouteBoundAuthorizer {
            inner,
            state: self.state.clone(),
            provider: self.provider,
            model: self.model.clone(),
            expected: CopilotRouteFingerprint::from(self),
        })
    }

    pub fn unavailable_message(
        &self,
        provider: Provider,
        model: &str,
        endpoint: CopilotEndpoint,
    ) -> String {
        let Some(state) = self.state.upgrade() else {
            return "Copilot account does not expose the requested model endpoint".to_string();
        };
        let mut message = format!(
            "Copilot account does not expose {} for {} model '{model}'",
            endpoint.path(),
            provider.as_str(),
        );
        if let Some(snapshot) = self.available_models.as_ref() {
            let mut ids = snapshot.available_model_ids(provider).peekable();
            if ids.peek().is_some() {
                message.push_str("; available account models: ");
                for (index, id) in ids.enumerate() {
                    if index > 0 {
                        message.push_str(", ");
                    }
                    message.push_str(id);
                }
            }
        }
        if let CopilotCredential::EnvironmentPat { secret, .. } = &state.credential {
            message = message.replace(secret, "[redacted]");
        }
        message
    }
}

pub struct CopilotChatCompletionsClientSpec {
    provider: Provider,
    model: String,
    api_base: String,
    authorizer: Arc<dyn HttpAuthorizer>,
    supports_temperature: bool,
    supports_thinking: bool,
    supports_reasoning: bool,
    supports_image_input: bool,
    supports_image_tool_results: bool,
}

impl CopilotChatCompletionsClientSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Provider,
        model: String,
        api_base: String,
        authorizer: Arc<dyn HttpAuthorizer>,
        supports_temperature: bool,
        supports_thinking: bool,
        supports_reasoning: bool,
        supports_image_tool_results: bool,
    ) -> Self {
        Self {
            provider,
            model,
            api_base,
            authorizer,
            supports_temperature,
            supports_thinking,
            supports_reasoning,
            supports_image_input: true,
            supports_image_tool_results,
        }
    }

    #[must_use]
    pub fn with_image_input_support(mut self, supports_image_input: bool) -> Self {
        self.supports_image_input = supports_image_input;
        self
    }

    #[must_use]
    pub const fn supports_image_input(&self) -> bool {
        self.supports_image_input
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Provider,
        String,
        String,
        Arc<dyn HttpAuthorizer>,
        bool,
        bool,
        bool,
        bool,
    ) {
        (
            self.provider,
            self.model,
            self.api_base,
            self.authorizer,
            self.supports_temperature,
            self.supports_thinking,
            self.supports_reasoning,
            self.supports_image_tool_results,
        )
    }
}

pub trait CopilotChatCompletionsClientFactory: Send + Sync {
    fn build(
        &self,
        spec: CopilotChatCompletionsClientSpec,
    ) -> Result<Arc<dyn meerkat_llm_core::LlmClient>, ProviderClientError>;
}

pub fn capability_gated_client(
    client: Arc<dyn meerkat_llm_core::LlmClient>,
    capabilities: Option<crate::protocol::CopilotModelCapabilities>,
) -> Arc<dyn meerkat_llm_core::LlmClient> {
    match capabilities {
        Some(capabilities) => Arc::new(CopilotCapabilityGatedClient {
            inner: client,
            capabilities,
        }),
        None => client,
    }
}

pub type CopilotRouteClientFactory = Arc<
    dyn Fn(
            &CopilotRouteResolution,
            &meerkat_llm_core::provider_runtime::ResolvedConnection,
        ) -> Result<Arc<dyn meerkat_llm_core::LlmClient>, ProviderClientError>
        + Send
        + Sync,
>;

pub fn routed_client(
    runtime: Arc<CopilotRuntime>,
    connection: meerkat_llm_core::provider_runtime::ResolvedConnection,
    provider: Provider,
    model: String,
    factory: CopilotRouteClientFactory,
) -> Result<Arc<dyn meerkat_llm_core::LlmClient>, ProviderClientError> {
    let route = runtime
        .resolve_route(&connection, provider, &model)
        .ok_or_else(|| {
            ProviderClientError::ClientInit(
                "Copilot route was not initialized during auth resolution".to_string(),
            )
        })?;
    let fingerprint = CopilotRouteFingerprint::from(&route);
    let capabilities = route.capabilities.clone();
    let schema_compiler = factory(&route, &connection)?;
    let initial_client = capability_gated_client(Arc::clone(&schema_compiler), capabilities);
    let client = CopilotRoutedClient {
        runtime,
        connection,
        provider,
        model,
        factory,
        schema_compiler,
        prepared_route: RwLock::new(PreparedCopilotRoute {
            fingerprint,
            client: initial_client,
        }),
    };
    Ok(Arc::new(client))
}

#[derive(Clone, PartialEq, Eq)]
struct CopilotRouteFingerprint {
    access: CopilotModelAccess,
    api_base: String,
    capabilities: Option<crate::protocol::CopilotModelCapabilities>,
}

impl From<&CopilotRouteResolution> for CopilotRouteFingerprint {
    fn from(route: &CopilotRouteResolution) -> Self {
        Self {
            access: route.access,
            api_base: route.api_base.clone(),
            capabilities: route.capabilities.clone(),
        }
    }
}

#[derive(Clone)]
struct PreparedCopilotRoute {
    fingerprint: CopilotRouteFingerprint,
    client: Arc<dyn meerkat_llm_core::LlmClient>,
}

struct CopilotRouteBoundAuthorizer {
    inner: Arc<dyn HttpAuthorizer>,
    state: Weak<CopilotAccountState>,
    provider: Provider,
    model: String,
    expected: CopilotRouteFingerprint,
}

impl CopilotRouteBoundAuthorizer {
    fn ensure_current_route(&self) -> Result<(), AuthError> {
        let state = self.state.upgrade().ok_or_else(|| {
            AuthError::ResolveRequired("Copilot route state was retired".to_string())
        })?;
        let current = resolve_route_for_state(&state, self.provider, &self.model)
            .map(|route| CopilotRouteFingerprint::from(&route))
            .ok_or_else(|| {
                AuthError::ResolveRequired("Copilot route requires re-resolution".to_string())
            })?;
        if current != self.expected {
            return Err(AuthError::ResolveRequired(
                "Copilot API route changed while refreshing authorization".to_string(),
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl HttpAuthorizer for CopilotRouteBoundAuthorizer {
    fn observe_lease_publication(
        &self,
        publication: &meerkat_core::handles::AuthLeasePublication,
    ) -> Result<(), AuthError> {
        self.inner.observe_lease_publication(publication)
    }

    fn observe_response_status(
        &self,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<(), AuthError> {
        self.inner.observe_response_status(response)
    }

    fn redact_response_diagnostic(&self, diagnostic: &mut String) {
        self.inner.redact_response_diagnostic(diagnostic);
    }

    async fn prepare_request(&self) -> Result<(), AuthError> {
        self.inner.prepare_request().await?;
        self.ensure_current_route()
    }

    async fn authorize(&self, request: &mut HttpAuthorizationRequest<'_>) -> Result<(), AuthError> {
        self.inner.authorize(request).await?;
        self.ensure_current_route()
    }

    async fn authorize_with_receipt(
        &self,
        request: &mut HttpAuthorizationRequest<'_>,
    ) -> Result<HttpAuthorizationReceipt, AuthError> {
        let receipt = self.inner.authorize_with_receipt(request).await?;
        self.ensure_current_route()?;
        Ok(receipt)
    }

    async fn observe_response(
        &self,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<HttpAuthorizationResponseAction, AuthError> {
        self.inner.observe_response(response).await
    }

    async fn observe_response_with_receipt(
        &self,
        receipt: HttpAuthorizationReceipt,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<HttpAuthorizationResponseAction, AuthError> {
        self.inner
            .observe_response_with_receipt(receipt, response)
            .await
    }

    fn label(&self) -> &str {
        self.inner.label()
    }

    fn append_content_headers(
        &self,
        content: meerkat_core::HttpAuthorizationContent,
        headers: &mut Vec<(String, String)>,
    ) -> Result<(), AuthError> {
        self.inner.append_content_headers(content, headers)
    }

    fn persistence_authority_id(&self) -> Option<meerkat_core::auth::ProviderAuthPersistenceId> {
        self.inner.persistence_authority_id()
    }

    fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.inner.expires_at()
    }
}

struct CopilotRoutedClient {
    runtime: Arc<CopilotRuntime>,
    connection: meerkat_llm_core::provider_runtime::ResolvedConnection,
    provider: Provider,
    model: String,
    factory: CopilotRouteClientFactory,
    schema_compiler: Arc<dyn meerkat_llm_core::LlmClient>,
    prepared_route: RwLock<PreparedCopilotRoute>,
}

impl CopilotRoutedClient {
    fn sync_client(&self) -> Arc<dyn meerkat_llm_core::LlmClient> {
        Arc::clone(&self.prepared_route.read().client)
    }

    fn refresh_prepared_route(&self) -> Result<bool, ProviderClientError> {
        let route = self
            .runtime
            .resolve_route(&self.connection, self.provider, &self.model)
            .ok_or_else(|| {
                ProviderClientError::ClientInit(
                    "Copilot route is unavailable after authorization refresh".to_string(),
                )
            })?;
        let fingerprint = CopilotRouteFingerprint::from(&route);
        if self.prepared_route.read().fingerprint == fingerprint {
            return Ok(false);
        }
        let capabilities = route.capabilities.clone();
        let client =
            capability_gated_client((self.factory)(&route, &self.connection)?, capabilities);
        let mut prepared = self.prepared_route.write();
        if prepared.fingerprint == fingerprint {
            return Ok(false);
        }
        *prepared = PreparedCopilotRoute {
            fingerprint,
            client,
        };
        Ok(true)
    }

    fn client_error(error: ProviderClientError) -> meerkat_llm_core::LlmError {
        meerkat_llm_core::LlmError::InvalidRequest {
            message: error.to_string(),
        }
    }

    fn missing_route_witness() -> meerkat_llm_core::LlmError {
        meerkat_llm_core::LlmError::InvalidRequest {
            message: "Copilot routed requests require request-scoped route preparation".to_string(),
        }
    }

    fn ensure_prepared_route_current(
        &self,
        prepared: &PreparedCopilotRoute,
    ) -> Result<(), meerkat_llm_core::LlmError> {
        let current = self
            .runtime
            .resolve_route(&self.connection, self.provider, &self.model)
            .ok_or_else(|| meerkat_llm_core::LlmError::ModelNotFound {
                model: self.model.clone(),
            })?;
        if CopilotRouteFingerprint::from(&current) == prepared.fingerprint {
            return Ok(());
        }
        self.refresh_prepared_route().map_err(Self::client_error)?;
        Err(meerkat_llm_core::LlmError::AuthorizationRouteChanged {
            message: "Copilot route changed before request lowering completed".to_string(),
        })
    }
}

#[async_trait]
impl meerkat_llm_core::LlmClient for CopilotRoutedClient {
    /// The current route's account limit.
    fn output_token_limit(&self) -> Option<u32> {
        self.sync_client().output_token_limit()
    }

    async fn prepare_controller_credential(&self) -> Result<(), AuthError> {
        self.connection
            .resolved_authorizer()
            .ok_or(AuthError::HostOwnedUnavailable)?
            .prepare_request()
            .await
    }

    fn project_replay_request(
        &self,
        messages: &[meerkat_core::Message],
    ) -> Result<meerkat_llm_core::LlmReplayProjection, meerkat_llm_core::LlmError> {
        self.refresh_prepared_route().map_err(Self::client_error)?;
        let prepared = self.prepared_route.read().clone();
        let messages = prepared.client.project_replay_messages(messages)?;
        Ok(meerkat_llm_core::LlmReplayProjection::new(messages)
            .with_route_witness(meerkat_llm_core::LlmRequestRouteWitness::new(prepared)))
    }

    fn project_replay_messages(
        &self,
        messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::Message>, meerkat_llm_core::LlmError> {
        self.sync_client().project_replay_messages(messages)
    }

    fn request_pressure(
        &self,
        _request: &meerkat_llm_core::LlmRequest,
    ) -> Result<Option<meerkat_core::ProviderRequestPressure>, meerkat_llm_core::LlmError> {
        Err(Self::missing_route_witness())
    }

    fn prepared_request_pressure(
        &self,
        request: &meerkat_llm_core::PreparedLlmRequest,
    ) -> Result<Option<meerkat_core::ProviderRequestPressure>, meerkat_llm_core::LlmError> {
        let prepared = request
            .route_witness::<PreparedCopilotRoute>()
            .ok_or_else(Self::missing_route_witness)?;
        self.ensure_prepared_route_current(prepared)?;
        prepared.client.request_pressure(request.request())
    }

    fn authored_cache_breakpoints(
        &self,
        _request: &meerkat_llm_core::LlmRequest,
        _canonical_messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::ProviderCacheBreakpointClaim>, meerkat_llm_core::LlmError> {
        Err(Self::missing_route_witness())
    }

    fn prepared_cache_breakpoints(
        &self,
        request: &meerkat_llm_core::PreparedLlmRequest,
        canonical_messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::ProviderCacheBreakpointClaim>, meerkat_llm_core::LlmError> {
        let prepared = request
            .route_witness::<PreparedCopilotRoute>()
            .ok_or_else(Self::missing_route_witness)?;
        self.ensure_prepared_route_current(prepared)?;
        prepared
            .client
            .authored_cache_breakpoints(request.request(), canonical_messages)
    }

    fn stream<'a>(
        &'a self,
        _request: &'a meerkat_llm_core::LlmRequest,
    ) -> meerkat_llm_core::LlmStream<'a> {
        Box::pin(futures::stream::once(async {
            Err(Self::missing_route_witness())
        }))
    }

    fn stream_prepared<'a>(
        &'a self,
        request: &'a meerkat_llm_core::PreparedLlmRequest,
    ) -> meerkat_llm_core::LlmStream<'a> {
        let Some(prepared) = request.route_witness::<PreparedCopilotRoute>().cloned() else {
            return Box::pin(futures::stream::once(async {
                Err(Self::missing_route_witness())
            }));
        };
        Box::pin(async_stream::try_stream! {
            let authorizer = self
                .connection
                .resolved_authorizer()
                .ok_or_else(|| meerkat_llm_core::LlmError::AuthenticationFailed {
                    message: "Copilot route has no dynamic authorizer".to_string(),
                })?;
            authorizer
                .prepare_request()
                .await
                .map_err(meerkat_llm_core::LlmError::from_authorizer)?;
            self.ensure_prepared_route_current(&prepared)?;
            let client = Arc::clone(&prepared.client);
            // Keep the original replay-route witness and per-work companion
            // through delegation. The concrete provider checks its lowered
            // target again after this account/route preparation await.
            let mut stream = client.stream_prepared(request);
            let mut emitted = false;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                    emitted = true;
                    yield event;
                    }
                    Err(error @ meerkat_llm_core::LlmError::AuthorizationRouteChanged { .. }) => {
                    if emitted {
                        Err(meerkat_llm_core::LlmError::IncompleteResponse {
                            message: "Copilot authorization route changed after streaming began"
                                .to_string(),
                        })?;
                    }
                    authorizer
                        .prepare_request()
                        .await
                        .map_err(meerkat_llm_core::LlmError::from_authorizer)?;
                    self.refresh_prepared_route().map_err(Self::client_error)?;
                    Err(error)?;
                    }
                    Err(error) => Err(error)?,
                }
            }
        })
    }

    fn provider(&self) -> Provider {
        self.provider
    }

    async fn health_check(&self) -> Result<(), meerkat_llm_core::LlmError> {
        let authorizer = self.connection.resolved_authorizer().ok_or_else(|| {
            meerkat_llm_core::LlmError::AuthenticationFailed {
                message: "Copilot route has no dynamic authorizer".to_string(),
            }
        })?;
        authorizer.prepare_request().await.map_err(|error| {
            meerkat_llm_core::LlmError::AuthenticationFailed {
                message: error.to_string(),
            }
        })?;
        self.refresh_prepared_route().map_err(Self::client_error)?;
        self.sync_client().health_check().await
    }

    fn compile_schema(
        &self,
        output_schema: &meerkat_core::OutputSchema,
    ) -> Result<meerkat_core::schema::CompiledSchema, meerkat_core::schema::SchemaError> {
        self.schema_compiler.compile_schema(output_schema)
    }
}

struct CopilotCapabilityGatedClient {
    inner: Arc<dyn meerkat_llm_core::LlmClient>,
    capabilities: crate::protocol::CopilotModelCapabilities,
}

impl CopilotCapabilityGatedClient {
    fn validate_request(
        &self,
        request: &meerkat_llm_core::LlmRequest,
    ) -> Result<(), meerkat_llm_core::LlmError> {
        let supports = &self.capabilities.supports;
        if supports.streaming == Some(false) {
            return Err(meerkat_llm_core::LlmError::InvalidRequest {
                message: format!(
                    "Copilot account model '{}' does not support streaming",
                    request.model
                ),
            });
        }
        if supports.tool_calls == Some(false) && !request.tools.is_empty() {
            return Err(meerkat_llm_core::LlmError::InvalidRequest {
                message: format!(
                    "Copilot account model '{}' does not support tool calls",
                    request.model
                ),
            });
        }
        if supports.structured_outputs == Some(false)
            && request
                .provider_params
                .as_ref()
                .is_some_and(|tag| match tag {
                    meerkat_core::lifecycle::run_primitive::ProviderTag::Anthropic(tag) => {
                        tag.structured_output.is_some()
                    }
                    meerkat_core::lifecycle::run_primitive::ProviderTag::OpenAi(tag) => {
                        tag.structured_output.is_some()
                    }
                    meerkat_core::lifecycle::run_primitive::ProviderTag::Gemini(tag) => {
                        tag.structured_output.is_some()
                    }
                    meerkat_core::lifecycle::run_primitive::ProviderTag::Unknown { .. } => false,
                })
        {
            return Err(meerkat_llm_core::LlmError::InvalidRequest {
                message: format!(
                    "Copilot account model '{}' does not support structured outputs",
                    request.model
                ),
            });
        }
        if supports.vision == Some(false) && request.has_images() {
            return Err(meerkat_llm_core::LlmError::InvalidRequest {
                message: format!(
                    "Copilot account model '{}' does not support image input",
                    request.model
                ),
            });
        }
        if let Some(meerkat_core::lifecycle::run_primitive::ProviderTag::Anthropic(tag)) =
            request.provider_params.as_ref()
        {
            if supports.adaptive_thinking == Some(false)
                && matches!(
                    tag.thinking,
                    Some(meerkat_core::lifecycle::run_primitive::AnthropicThinkingConfig::Adaptive)
                )
            {
                return Err(meerkat_llm_core::LlmError::InvalidRequest {
                    message: format!(
                        "Copilot account model '{}' does not support adaptive thinking",
                        request.model
                    ),
                });
            }
            let thinking_budget = match tag.thinking {
                Some(
                    meerkat_core::lifecycle::run_primitive::AnthropicThinkingConfig::Enabled {
                        budget_tokens,
                    },
                ) => Some(budget_tokens),
                Some(
                    meerkat_core::lifecycle::run_primitive::AnthropicThinkingConfig::Adaptive
                    | meerkat_core::lifecycle::run_primitive::AnthropicThinkingConfig::BetweenTools
                    | meerkat_core::lifecycle::run_primitive::AnthropicThinkingConfig::Disabled,
                ) => None,
                None => tag.thinking_budget_tokens,
            };
            if let (Some(budget), Some(maximum)) = (thinking_budget, supports.max_thinking_budget)
                && budget > maximum
            {
                return Err(meerkat_llm_core::LlmError::InvalidRequest {
                    message: format!(
                        "Copilot account model '{}' allows at most {maximum} thinking tokens, requested {budget}",
                        request.model
                    ),
                });
            }
            if let (Some(budget), Some(minimum)) = (thinking_budget, supports.min_thinking_budget)
                && budget < minimum
            {
                return Err(meerkat_llm_core::LlmError::InvalidRequest {
                    message: format!(
                        "Copilot account model '{}' requires at least {minimum} thinking tokens, requested {budget}",
                        request.model
                    ),
                });
            }
        }
        if let Some(max_output_tokens) = self.capabilities.limits.max_output_tokens
            && request.max_tokens > max_output_tokens
        {
            return Err(meerkat_llm_core::LlmError::InvalidRequest {
                message: format!(
                    "Copilot account model '{}' allows at most {max_output_tokens} output tokens, requested {}",
                    request.model, request.max_tokens
                ),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl meerkat_llm_core::LlmClient for CopilotCapabilityGatedClient {
    async fn prepare_controller_credential(&self) -> Result<(), AuthError> {
        self.inner.prepare_controller_credential().await
    }

    /// The account model's own output limit, the one the gate enforces.
    fn output_token_limit(&self) -> Option<u32> {
        self.capabilities
            .limits
            .max_output_tokens
            .or_else(|| self.inner.output_token_limit())
    }

    fn project_replay_messages(
        &self,
        messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::Message>, meerkat_llm_core::LlmError> {
        self.inner.project_replay_messages(messages)
    }

    fn request_pressure(
        &self,
        request: &meerkat_llm_core::LlmRequest,
    ) -> Result<Option<meerkat_core::ProviderRequestPressure>, meerkat_llm_core::LlmError> {
        self.validate_request(request)?;
        let pressure = self.inner.request_pressure(request)?;
        if let Some(measured) = pressure.as_ref().and_then(|value| {
            value
                .provider_issued_input_tokens
                .map(|tokens| (tokens, value))
        }) {
            let prompt_limit = self
                .capabilities
                .limits
                .max_prompt_tokens
                .or_else(|| {
                    self.capabilities
                        .limits
                        .max_context_window_tokens
                        .map(|context| context.saturating_sub(request.max_tokens))
                })
                .map(u64::from);
            if let Some(limit) = prompt_limit
                && measured.0 > limit
            {
                return Err(meerkat_llm_core::LlmError::InvalidRequest {
                    message: format!(
                        "Copilot account model '{}' allows at most {limit} prompt tokens, measured {}",
                        request.model, measured.0
                    ),
                });
            }
        }
        Ok(pressure)
    }

    fn authored_cache_breakpoints(
        &self,
        request: &meerkat_llm_core::LlmRequest,
        canonical_messages: &[meerkat_core::Message],
    ) -> Result<Vec<meerkat_core::ProviderCacheBreakpointClaim>, meerkat_llm_core::LlmError> {
        self.inner
            .authored_cache_breakpoints(request, canonical_messages)
    }

    fn stream<'a>(
        &'a self,
        request: &'a meerkat_llm_core::LlmRequest,
    ) -> meerkat_llm_core::LlmStream<'a> {
        if let Err(error) = self.validate_request(request) {
            return Box::pin(futures::stream::once(async move { Err(error) }));
        }
        self.inner.stream(request)
    }

    fn provider(&self) -> Provider {
        self.inner.provider()
    }

    async fn health_check(&self) -> Result<(), meerkat_llm_core::LlmError> {
        self.inner.health_check().await
    }

    fn compile_schema(
        &self,
        output_schema: &meerkat_core::OutputSchema,
    ) -> Result<meerkat_core::schema::CompiledSchema, meerkat_core::schema::SchemaError> {
        self.inner.compile_schema(output_schema)
    }
}

#[derive(serde::Deserialize)]
struct CopilotUserEnvelope {
    endpoints: CopilotTokenEndpoints,
}

struct CopilotAccountState {
    config: CopilotBackendConfig,
    credential: CopilotCredential,
    pat_rejected: AtomicBool,
    publications: Mutex<
        Vec<(
            meerkat_core::handles::WeakGeneratedAuthLeaseHandle,
            meerkat_core::handles::AuthLeasePublication,
        )>,
    >,
    cache: RwLock<Option<CachedCopilotToken>>,
    refresh: Mutex<Option<Arc<CopilotRefreshState>>>,
    next_generation: AtomicU64,
}

struct CopilotRefreshState {
    source_generation: Option<u64>,
    outcome: Mutex<CopilotRefreshOutcome>,
    notify: tokio::sync::Notify,
}

#[derive(Clone)]
enum CopilotRefreshOutcome {
    Pending,
    Cancelled,
    Completed(Result<CachedTokenView, ProviderAuthError>),
}

struct CopilotRefreshFlight<'a> {
    state: &'a CopilotAccountState,
    flight: Arc<CopilotRefreshState>,
}

impl Drop for CopilotRefreshFlight<'_> {
    fn drop(&mut self) {
        let mut active = self.state.refresh.lock();
        let mut outcome = self.flight.outcome.lock();
        if matches!(*outcome, CopilotRefreshOutcome::Pending) {
            *outcome = CopilotRefreshOutcome::Cancelled;
        }
        if active
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &self.flight))
        {
            *active = None;
        }
        drop(outcome);
        drop(active);
        self.flight.notify.notify_waiters();
    }
}

struct CachedCopilotToken {
    token: String,
    lifetime: CopilotTokenLifetime,
    api_base: String,
    models: Option<Arc<CopilotModelSnapshot>>,
    model_discovery_refresh_at: Option<DateTime<Utc>>,
}

#[derive(Debug)]
enum CopilotTokenLifetime {
    ManagedOauth {
        generation: u64,
        source_generation: u64,
        source_expires_at: Option<DateTime<Utc>>,
        refresh_at: DateTime<Utc>,
    },
    EnvironmentPat,
}

impl CopilotTokenLifetime {
    fn generation(&self) -> Option<u64> {
        match self {
            Self::ManagedOauth { generation, .. } => Some(*generation),
            Self::EnvironmentPat => None,
        }
    }

    fn source_generation(&self) -> Option<u64> {
        match self {
            Self::ManagedOauth {
                source_generation, ..
            } => Some(*source_generation),
            Self::EnvironmentPat => None,
        }
    }

    fn fresh(&self, now: DateTime<Utc>) -> bool {
        match self {
            Self::ManagedOauth { refresh_at, .. } => now < *refresh_at,
            Self::EnvironmentPat => true,
        }
    }
}

impl std::fmt::Debug for CachedCopilotToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CachedCopilotToken")
            .field("token", &"<redacted>")
            .field("lifetime", &self.lifetime)
            .field("api_base", &self.api_base)
            .field("has_models", &self.models.is_some())
            .field(
                "model_discovery_refresh_at",
                &self.model_discovery_refresh_at,
            )
            .finish()
    }
}

pub struct CopilotAuthorizer {
    state: Arc<CopilotAccountState>,
    binding: ValidatedBinding,
    env: ResolverEnvironment,
    transport: Arc<dyn CopilotHttpTransport>,
}

impl CopilotAuthorizer {
    pub async fn prime(&self) -> Result<(), ProviderAuthError> {
        self.ensure_token().await.map(|_| ())
    }

    pub fn credential_identity(&self) -> &AuthCredentialIdentity {
        self.binding.credential_identity()
    }

    pub fn model_snapshot(&self) -> Option<Arc<CopilotModelSnapshot>> {
        self.state
            .cache
            .read()
            .as_ref()
            .and_then(|cached| cached.models.clone())
    }

    pub fn api_base(&self) -> Option<String> {
        self.state
            .cache
            .read()
            .as_ref()
            .map(|cached| cached.api_base.clone())
    }

    fn invalidate_derived_token_generation(&self, generation: u64) {
        let mut cache = self.state.cache.write();
        if cache
            .as_ref()
            .is_some_and(|cached| cached.lifetime.generation() == Some(generation))
        {
            *cache = None;
        }
    }

    fn reject_pat(&self) -> ProviderAuthError {
        self.state.pat_rejected.store(true, Ordering::Release);
        *self.state.cache.write() = None;
        let key = meerkat_core::handles::LeaseKey::from_credential_identity(
            self.binding.credential_identity(),
        );
        let mut publications = self.state.publications.lock();
        publications.retain(|(owner, _)| owner.upgrade().is_some());
        for (owner, publication) in publications.iter() {
            let Some(handle) = owner.upgrade() else {
                continue;
            };
            if let Err(error) = handle.mark_reauth_required_for_publication(&key, publication) {
                return ProviderAuthError::SourceResolutionFailed(format!(
                    "Copilot PAT rejection could not be published to AuthMachine: {error}"
                ));
            }
        }
        ProviderAuthError::Auth(AuthError::UserReauthRequired)
    }

    async fn ensure_token(&self) -> Result<CachedTokenView, ProviderAuthError> {
        loop {
            if self.state.pat_rejected.load(Ordering::Acquire) {
                return Err(self.reject_pat());
            }
            let source = match self.state.credential {
                CopilotCredential::ManagedOauth => {
                    let tokens = self.load_source_tokens().await?;
                    let generation = meerkat_core::tokens_lifecycle_published_generation(&tokens)
                        .ok_or_else(|| {
                        ProviderAuthError::SourceResolutionFailed(
                            "Copilot source credential has no AuthMachine publication generation"
                                .to_string(),
                        )
                    })?;
                    Some((tokens, generation))
                }
                CopilotCredential::EnvironmentPat { .. } => None,
            };
            let source_generation = source.as_ref().map(|(_, generation)| *generation);
            let now = (self.env.now)();
            if let Some(cached) = self.fresh_cached(source_generation, now)
                && !cached.model_discovery_due
            {
                return Ok(cached);
            }
            let (flight, leader) = {
                let mut active = self.state.refresh.lock();
                if let Some(flight) = active.as_ref() {
                    (Arc::clone(flight), false)
                } else {
                    let flight = Arc::new(CopilotRefreshState {
                        source_generation,
                        outcome: Mutex::new(CopilotRefreshOutcome::Pending),
                        notify: tokio::sync::Notify::new(),
                    });
                    *active = Some(Arc::clone(&flight));
                    (flight, true)
                }
            };
            if !leader {
                loop {
                    let notified = flight.notify.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    let outcome = flight.outcome.lock().clone();
                    match outcome {
                        CopilotRefreshOutcome::Pending => notified.await,
                        CopilotRefreshOutcome::Cancelled => break,
                        CopilotRefreshOutcome::Completed(result) => {
                            if self.state.pat_rejected.load(Ordering::Acquire) {
                                return Err(self.reject_pat());
                            }
                            if flight.source_generation == source_generation {
                                match result {
                                    Err(error) => {
                                        if source.is_some() {
                                            let current = self.load_source_tokens().await?;
                                            if meerkat_core::tokens_lifecycle_published_generation(
                                                &current,
                                            ) != source_generation
                                            {
                                                break;
                                            }
                                        }
                                        return Err(error);
                                    }
                                    Ok(cached) if source.is_none() => return Ok(cached),
                                    Ok(_) => {}
                                }
                            }
                            break;
                        }
                    }
                }
                continue;
            }
            let _flight = CopilotRefreshFlight {
                state: self.state.as_ref(),
                flight: Arc::clone(&flight),
            };
            let result = match source {
                Some((tokens, generation)) => {
                    self.refresh_token_as_leader(&tokens, generation).await
                }
                None => self.refresh_pat_models_as_leader().await,
            };
            let result = if self.state.pat_rejected.load(Ordering::Acquire) {
                Err(self.reject_pat())
            } else {
                result
            };
            *flight.outcome.lock() = CopilotRefreshOutcome::Completed(result.clone());
            return result;
        }
    }

    async fn refresh_pat_models_as_leader(&self) -> Result<CachedTokenView, ProviderAuthError> {
        if let Some(cached) = self.fresh_cached(None, (self.env.now)())
            && !cached.model_discovery_due
        {
            return Ok(cached);
        }
        let CopilotCredential::EnvironmentPat { secret, .. } = &self.state.credential else {
            return Err(ProviderAuthError::SourceResolutionFailed(
                "Copilot PAT discovery requires an environment credential".to_string(),
            ));
        };
        let api_base = match self.api_base() {
            Some(api_base) => api_base,
            None => self.lookup_pat_api_base(secret).await?,
        };
        let models = self.fetch_models(&api_base, secret).await;
        if models
            .as_ref()
            .is_err_and(|error| matches!(error.status(), Some(401 | 403)))
        {
            return Err(self.reject_pat());
        }
        let discovery_now = (self.env.now)();
        let mut cache = self.state.cache.write();
        let (models, delay) = match models {
            Ok(snapshot) => (
                Some(Arc::new(snapshot)),
                StdDuration::from_secs(MODEL_DISCOVERY_REFRESH_SECONDS),
            ),
            Err(error) => {
                // Authenticated error bodies and decode errors may echo credentials.
                tracing::warn!(
                    credential_identity = %self.binding.credential_identity(),
                    status = error.status(),
                    "Copilot PAT model discovery failed; retaining prior availability"
                );
                (
                    cache.as_ref().and_then(|cached| cached.models.clone()),
                    error.retry_delay(),
                )
            }
        };
        let view = CachedTokenView {
            generation: None,
            token: secret.clone(),
            api_base: api_base.clone(),
            model_discovery_due: false,
        };
        *cache = Some(CachedCopilotToken {
            token: secret.clone(),
            lifetime: CopilotTokenLifetime::EnvironmentPat,
            api_base,
            models,
            model_discovery_refresh_at: Some(model_discovery_deadline(discovery_now, delay)),
        });
        Ok(view)
    }

    async fn lookup_pat_api_base(&self, secret: &str) -> Result<String, ProviderAuthError> {
        let mut headers = HeaderMap::new();
        insert_header(
            &mut headers,
            "Authorization",
            &format!("{BEARER_SCHEME} {secret}"),
        )?;
        insert_header(&mut headers, "Accept", "application/vnd.github.v3+json")?;
        insert_header(&mut headers, "User-Agent", &self.state.config.user_agent)?;
        let response = self
            .transport
            .get_pat(COPILOT_USER_URL, headers)
            .await
            .map_err(|_| {
                ProviderAuthError::SourceResolutionFailed(
                    "Copilot account lookup failed".to_string(),
                )
            })?;
        if matches!(response.status, 401 | 403) {
            return Err(self.reject_pat());
        }
        if !(200..=299).contains(&response.status) {
            return Err(ProviderAuthError::SourceResolutionFailed(format!(
                "Copilot account lookup returned HTTP {}",
                response.status
            )));
        }
        let user: CopilotUserEnvelope = serde_json::from_slice(&response.body).map_err(|_| {
            ProviderAuthError::SourceResolutionFailed(
                "invalid Copilot account response".to_string(),
            )
        })?;
        resolve_pat_api_base(&user.endpoints)
    }

    async fn refresh_token_as_leader(
        &self,
        source: &PersistedTokens,
        source_generation: u64,
    ) -> Result<CachedTokenView, ProviderAuthError> {
        let now = (self.env.now)();
        if let Some(cached) = self.fresh_cached(Some(source_generation), now) {
            if cached.model_discovery_due {
                let models = self.fetch_models(&cached.api_base, &cached.token).await;
                let discovery_now = (self.env.now)();
                let remint = models
                    .as_ref()
                    .is_err_and(|error| error.status() == Some(401));
                let reauth = models
                    .as_ref()
                    .is_err_and(|error| error.status() == Some(403));
                {
                    let mut cache = self.state.cache.write();
                    if let Some(current) = cache.as_mut()
                        && current.lifetime.source_generation() == Some(source_generation)
                        && current.token == cached.token
                    {
                        match models {
                            Ok(snapshot) => {
                                current.models = Some(Arc::new(snapshot));
                                current.model_discovery_refresh_at =
                                    Some(model_discovery_deadline(
                                        discovery_now,
                                        StdDuration::from_secs(MODEL_DISCOVERY_REFRESH_SECONDS),
                                    ));
                            }
                            Err(error) if !remint && !reauth => {
                                let retry_delay = error.retry_delay();
                                tracing::warn!(
                                    credential_identity = %self.binding.credential_identity(),
                                    error = %error,
                                    "Copilot account model discovery retry failed"
                                );
                                current.model_discovery_refresh_at =
                                    Some(model_discovery_deadline(discovery_now, retry_delay));
                            }
                            Err(_) => {
                                *cache = None;
                            }
                        }
                    }
                }
                if reauth {
                    self.mark_account_reauth_required()?;
                    return Err(ProviderAuthError::Auth(AuthError::UserReauthRequired));
                }
                if remint {
                    return self.mint_and_cache_token(source, source_generation).await;
                }
            }
            return Ok(cached);
        }
        self.mint_and_cache_token(source, source_generation).await
    }

    async fn mint_and_cache_token(
        &self,
        source: &PersistedTokens,
        source_generation: u64,
    ) -> Result<CachedTokenView, ProviderAuthError> {
        let github_token = source
            .primary_secret
            .as_deref()
            .ok_or(ProviderAuthError::Auth(AuthError::MissingSecret))?;
        let mut token = self.mint_copilot_token(github_token).await?;
        let mut api_base = resolve_api_base(&token)?;
        let mut models_result = self.fetch_models(&api_base, &token.token).await;
        if models_result
            .as_ref()
            .is_err_and(|error| error.status() == Some(401))
        {
            token = self.mint_copilot_token(github_token).await?;
            api_base = resolve_api_base(&token)?;
            models_result = self.fetch_models(&api_base, &token.token).await;
        }
        if models_result
            .as_ref()
            .is_err_and(|error| matches!(error.status(), Some(401 | 403)))
        {
            self.mark_account_reauth_required()?;
            return Err(ProviderAuthError::Auth(AuthError::UserReauthRequired));
        }
        let now = (self.env.now)();
        let refresh_at = token_refresh_at(&token, now)?;
        let discovery_now = (self.env.now)();
        let (models, model_discovery_refresh_at) = match models_result {
            Ok(snapshot) => (
                Some(Arc::new(snapshot)),
                Some(model_discovery_deadline(
                    discovery_now,
                    StdDuration::from_secs(MODEL_DISCOVERY_REFRESH_SECONDS),
                )),
            ),
            Err(error) => {
                let retry_delay = error.retry_delay();
                tracing::warn!(
                    credential_identity = %self.binding.credential_identity(),
                    error = %error,
                    "Copilot authentication succeeded but account model discovery failed"
                );
                (
                    None,
                    Some(model_discovery_deadline(discovery_now, retry_delay)),
                )
            }
        };
        let generation = self.state.next_generation.fetch_add(1, Ordering::Relaxed);
        let view = CachedTokenView {
            generation: Some(generation),
            token: token.token.clone(),
            api_base: api_base.clone(),
            model_discovery_due: false,
        };
        *self.state.cache.write() = Some(CachedCopilotToken {
            token: token.token,
            lifetime: CopilotTokenLifetime::ManagedOauth {
                generation,
                source_generation,
                source_expires_at: source.expires_at,
                refresh_at,
            },
            api_base,
            models,
            model_discovery_refresh_at,
        });
        Ok(view)
    }

    fn fresh_cached(
        &self,
        source_generation: Option<u64>,
        now: DateTime<Utc>,
    ) -> Option<CachedTokenView> {
        let cache = self.state.cache.read();
        let cached = cache.as_ref()?;
        (cached.lifetime.source_generation() == source_generation && cached.lifetime.fresh(now))
            .then(|| CachedTokenView {
                generation: cached.lifetime.generation(),
                token: cached.token.clone(),
                api_base: cached.api_base.clone(),
                model_discovery_due: cached
                    .model_discovery_refresh_at
                    .is_some_and(|refresh_at| now >= refresh_at),
            })
    }

    async fn load_source_tokens(&self) -> Result<PersistedTokens, ProviderAuthError> {
        let mut managed =
            load_managed_store_tokens_with_lifecycle(&self.env, &self.binding).await?;
        match resolve_oauth_login_credential_disposition(
            &self.env,
            &self.binding,
            managed.tokens.primary_secret.is_some(),
        )? {
            OAuthLoginCredentialAdmission::UseCached => Ok(managed.tokens),
            OAuthLoginCredentialAdmission::BeginRefresh => {
                managed.release_prelock_lifecycle_guard();
                self.refresh_source_token(managed).await
            }
        }
    }

    async fn refresh_source_token(
        &self,
        managed: meerkat_auth_core::resolver::ManagedStoreTokens,
    ) -> Result<PersistedTokens, ProviderAuthError> {
        let persistence = self
            .env
            .provider_auth_persistence()
            .cloned()
            .ok_or_else(|| {
                ProviderAuthError::SourceResolutionFailed(
                    "Copilot source refresh requires provider-auth persistence".to_string(),
                )
            })?;
        let preparation_env = self.env.clone();
        let preparation_binding = self.binding.clone();
        let preparation: meerkat_auth_core::resolver::ManagedStoreOAuthRefreshPrepareFn =
            Box::new(move |locked_baseline, mode| {
                Box::pin(async move {
                    prepare_managed_store_oauth_refresh_under_lock(
                        &preparation_env,
                        &preparation_binding,
                        managed,
                        locked_baseline,
                        mode,
                    )
                    .await
                    .map_err(meerkat_auth_core::resolver::refresh_error_from_provider)
                })
            });
        refresh_github_tokens(
            persistence,
            preparation,
            self.binding.credential_identity().clone(),
            self.state.config.clone(),
            self.env.force_refresh,
        )
        .await
        .map_err(|error| ProviderAuthError::Auth(auth_error_from_refresh(error)))
    }

    fn mark_account_reauth_required(&self) -> Result<(), ProviderAuthError> {
        if let Some(auth_lease) = self.env.auth_lease_handle.as_ref() {
            let lease_key = meerkat_core::handles::LeaseKey::from_credential_identity(
                self.binding.credential_identity(),
            );
            auth_lease
                .mark_reauth_required(&lease_key)
                .map_err(|error| {
                    ProviderAuthError::SourceResolutionFailed(format!(
                        "Copilot source rejection could not be published to AuthMachine: {error}"
                    ))
                })?;
        }
        Ok(())
    }

    async fn mint_copilot_token(
        &self,
        github_token: &str,
    ) -> Result<CopilotTokenEnvelope, ProviderAuthError> {
        let endpoints = self
            .state
            .config
            .endpoints()
            .map_err(|error| ProviderAuthError::SourceResolutionFailed(error.to_string()))?;
        let mut headers = HeaderMap::new();
        insert_header(
            &mut headers,
            "Authorization",
            &format!("token {github_token}"),
        )?;
        insert_header(&mut headers, "Accept", "application/json")?;
        insert_header(
            &mut headers,
            "X-GitHub-Api-Version",
            &self.state.config.token_api_version,
        )?;
        insert_header(&mut headers, "User-Agent", &self.state.config.user_agent)?;
        insert_header(
            &mut headers,
            "Editor-Version",
            &self.state.config.editor_version,
        )?;
        insert_header(
            &mut headers,
            "Editor-Plugin-Version",
            &self.state.config.editor_plugin_version,
        )?;
        insert_header(
            &mut headers,
            "Copilot-Integration-Id",
            &self.state.config.integration_id,
        )?;
        let response = self
            .transport
            .get(endpoints.copilot_token_url.as_str(), headers)
            .await
            .map_err(|error| {
                ProviderAuthError::SourceResolutionFailed(format!(
                    "Copilot token exchange failed: {error}"
                ))
            })?;
        if !(200..=299).contains(&response.status) {
            let body = String::from_utf8_lossy(&response.body);
            if token_exchange_requires_account_action(response.status, &response.body) {
                self.mark_account_reauth_required()?;
                return Err(ProviderAuthError::Auth(AuthError::UserReauthRequired));
            }
            return Err(ProviderAuthError::SourceResolutionFailed(format!(
                "Copilot token exchange returned {}: {body}",
                response.status
            )));
        }
        let token: CopilotTokenEnvelope =
            serde_json::from_slice(&response.body).map_err(|error| {
                ProviderAuthError::SourceResolutionFailed(format!(
                    "invalid Copilot token response: {error}"
                ))
            })?;
        if token.token.trim().is_empty() {
            return Err(ProviderAuthError::SourceResolutionFailed(
                "Copilot token response contains an empty token".to_string(),
            ));
        }
        let now_epoch = (self.env.now)().timestamp().max(0) as u64;
        if token.expires_at <= now_epoch {
            return Err(ProviderAuthError::SourceResolutionFailed(
                "Copilot token response is already expired".to_string(),
            ));
        }
        Ok(token)
    }

    async fn fetch_models(
        &self,
        api_base: &str,
        token: &str,
    ) -> Result<CopilotModelSnapshot, CopilotModelDiscoveryError> {
        let url = format!("{}/models", api_base.trim_end_matches('/'));
        let headers = self
            .inference_headers(token, CopilotRequestIntent::ModelAccess)
            .map_err(CopilotModelDiscoveryError::Header)?;
        let response = match self.state.credential {
            CopilotCredential::EnvironmentPat { .. } => self.transport.get_pat(&url, headers).await,
            CopilotCredential::ManagedOauth => self.transport.get(&url, headers).await,
        }
        .map_err(CopilotModelDiscoveryError::Transport)?;
        if !(200..=299).contains(&response.status) {
            let body = match self.state.credential {
                CopilotCredential::EnvironmentPat { .. } => "[response body omitted]".to_string(),
                CopilotCredential::ManagedOauth => {
                    String::from_utf8_lossy(&response.body).into_owned()
                }
            };
            return Err(CopilotModelDiscoveryError::Http {
                status: response.status,
                body,
                retry_after: response.retry_after,
            });
        }
        let body: CopilotModelsEnvelope =
            serde_json::from_slice(&response.body).map_err(CopilotModelDiscoveryError::Decode)?;
        CopilotModelSnapshot::from_models(body.data).map_err(CopilotModelDiscoveryError::Protocol)
    }

    fn inference_headers(
        &self,
        token: &str,
        intent: CopilotRequestIntent,
    ) -> Result<HeaderMap, ProviderAuthError> {
        let config = &self.state.config;
        let mut headers = HeaderMap::new();
        insert_header(
            &mut headers,
            "Authorization",
            &format!("{BEARER_SCHEME} {token}"),
        )?;
        insert_header(
            &mut headers,
            "X-GitHub-Api-Version",
            &config.inference_api_version,
        )?;
        insert_header(&mut headers, "User-Agent", &config.user_agent)?;
        insert_header(&mut headers, "Editor-Version", &config.editor_version)?;
        insert_header(
            &mut headers,
            "Editor-Plugin-Version",
            &config.editor_plugin_version,
        )?;
        insert_header(
            &mut headers,
            "Copilot-Integration-Id",
            &config.integration_id,
        )?;
        insert_header(&mut headers, "OpenAI-Intent", intent.as_str())?;
        insert_header(
            &mut headers,
            "X-Request-Id",
            &meerkat_core::time_compat::new_uuid_v7().to_string(),
        )?;
        Ok(headers)
    }
}

#[async_trait]
impl HttpAuthorizer for CopilotAuthorizer {
    fn observe_lease_publication(
        &self,
        publication: &meerkat_core::handles::AuthLeasePublication,
    ) -> Result<(), AuthError> {
        if matches!(
            self.state.credential,
            CopilotCredential::EnvironmentPat { .. }
        ) && let Some(handle) = &self.env.auth_lease_handle
        {
            let key = meerkat_core::handles::LeaseKey::from_credential_identity(
                self.binding.credential_identity(),
            );
            let mut publications = self.state.publications.lock();
            publications.retain(|(owner, _)| owner.upgrade().is_some());
            if handle.is_current_publication(&key, publication) {
                if let Some((_, current)) = publications.iter_mut().find(|(owner, _)| {
                    owner
                        .upgrade()
                        .is_some_and(|owner| owner.same_owner(handle))
                }) {
                    *current = publication.clone();
                } else {
                    publications.push((handle.downgrade(), publication.clone()));
                }
            }
        }
        if self.state.pat_rejected.load(Ordering::Acquire) {
            return Err(auth_error_from_provider(self.reject_pat()));
        }
        Ok(())
    }

    fn observe_response_status(
        &self,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<(), AuthError> {
        if matches!(
            self.state.credential,
            CopilotCredential::EnvironmentPat { .. }
        ) && matches!(response.status, 401 | 403)
        {
            return Err(auth_error_from_provider(self.reject_pat()));
        }
        Ok(())
    }

    fn redact_response_diagnostic(&self, diagnostic: &mut String) {
        if let CopilotCredential::EnvironmentPat { secret, .. } = &self.state.credential
            && diagnostic.contains(secret.as_str())
        {
            *diagnostic = diagnostic.replace(secret, "[redacted]");
        }
    }

    async fn prepare_request(&self) -> Result<(), AuthError> {
        self.ensure_token()
            .await
            .map(|_| ())
            .map_err(auth_error_from_provider)
    }

    async fn authorize(&self, request: &mut HttpAuthorizationRequest<'_>) -> Result<(), AuthError> {
        self.authorize_attempt(request).await.map(|_| ())
    }

    async fn authorize_with_receipt(
        &self,
        request: &mut HttpAuthorizationRequest<'_>,
    ) -> Result<HttpAuthorizationReceipt, AuthError> {
        self.authorize_attempt(request).await.map(|generation| {
            generation.map_or_else(
                HttpAuthorizationReceipt::untracked,
                HttpAuthorizationReceipt::tracked,
            )
        })
    }

    async fn observe_response(
        &self,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<HttpAuthorizationResponseAction, AuthError> {
        self.observe_response_status(response)?;
        if response.status == 401 {
            return Err(AuthError::Other(
                "Copilot 401 observation requires its authorization receipt".to_string(),
            ));
        }
        Ok(HttpAuthorizationResponseAction::Propagate)
    }

    async fn observe_response_with_receipt(
        &self,
        receipt: HttpAuthorizationReceipt,
        response: &HttpAuthorizationResponse<'_>,
    ) -> Result<HttpAuthorizationResponseAction, AuthError> {
        if matches!(
            self.state.credential,
            CopilotCredential::EnvironmentPat { .. }
        ) {
            return self.observe_response(response).await;
        }
        if response.status == 401 {
            let generation = receipt.generation().ok_or_else(|| {
                AuthError::Other(
                    "Copilot 401 observation received an untracked authorization receipt"
                        .to_string(),
                )
            })?;
            self.invalidate_derived_token_generation(generation);
            return Ok(HttpAuthorizationResponseAction::RetryWithFreshAuthorization);
        }
        Ok(HttpAuthorizationResponseAction::Propagate)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn label(&self) -> &str {
        crate::GITHUB_COPILOT_AUTHORIZER_LABEL
    }

    fn append_content_headers(
        &self,
        content: meerkat_core::HttpAuthorizationContent,
        headers: &mut Vec<(String, String)>,
    ) -> Result<(), AuthError> {
        if content.has_images {
            headers.push(("Copilot-Vision-Request".to_string(), "true".to_string()));
        }
        Ok(())
    }

    fn persistence_authority_id(&self) -> Option<meerkat_core::auth::ProviderAuthPersistenceId> {
        if matches!(
            self.state.credential,
            CopilotCredential::EnvironmentPat { .. }
        ) {
            return None;
        }
        self.env
            .provider_auth_persistence()
            .map(meerkat_core::auth::ProviderAuthPersistence::authority_id)
    }

    fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.state
            .cache
            .read()
            .as_ref()
            .and_then(|cached| match cached.lifetime {
                CopilotTokenLifetime::ManagedOauth {
                    source_expires_at, ..
                } => source_expires_at,
                CopilotTokenLifetime::EnvironmentPat => None,
            })
    }
}

impl CopilotAuthorizer {
    async fn authorize_attempt(
        &self,
        request: &mut HttpAuthorizationRequest<'_>,
    ) -> Result<Option<u64>, AuthError> {
        let token = self
            .ensure_token()
            .await
            .map_err(auth_error_from_provider)?;
        if !url_is_within_api_base(request.url, &token.api_base) {
            return Err(AuthError::Other(
                "Copilot authorizer refused URL outside resolved API base".to_string(),
            ));
        }
        for (name, value) in self
            .inference_headers(&token.token, CopilotRequestIntent::ConversationPanel)
            .map_err(|error| AuthError::Other(error.to_string()))?
        {
            let Some(name) = name else {
                continue;
            };
            let value = value
                .to_str()
                .map_err(|error| AuthError::Other(error.to_string()))?;
            request.headers.push((name.to_string(), value.to_string()));
        }
        Ok(token.generation)
    }
}

fn auth_error_from_provider(error: ProviderAuthError) -> AuthError {
    match error {
        ProviderAuthError::Auth(error) => error,
        other => AuthError::RefreshFailed(other.to_string()),
    }
}

fn auth_error_from_refresh(error: RefreshError) -> AuthError {
    if let Some(access) = error.credential_access_error() {
        return access;
    }
    match error {
        RefreshError::StalePreparation => AuthError::StaleCredential,
        RefreshError::ReauthRequired(_)
        | RefreshError::Classified {
            disposition: meerkat_core::auth::RefreshFailureDisposition::ReauthRequired,
            ..
        }
        | RefreshError::DurableTerminalCommit {
            disposition: meerkat_core::auth::RefreshFailureDisposition::ReauthRequired,
            ..
        } => AuthError::UserReauthRequired,
        other => AuthError::RefreshFailed(other.to_string()),
    }
}

#[derive(Clone)]
struct CachedTokenView {
    generation: Option<u64>,
    token: String,
    api_base: String,
    model_discovery_due: bool,
}

fn resolve_api_base(token: &CopilotTokenEnvelope) -> Result<String, ProviderAuthError> {
    let candidate = token
        .endpoints
        .api
        .clone()
        .or_else(|| api_base_from_proxy_token(&token.token))
        .unwrap_or_else(|| DEFAULT_COPILOT_API_BASE.to_string());
    let url = url::Url::parse(&candidate).map_err(|error| {
        ProviderAuthError::SourceResolutionFailed(format!("invalid Copilot API endpoint: {error}"))
    })?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return Err(ProviderAuthError::SourceResolutionFailed(
            "Copilot API endpoint must be HTTPS without user information".to_string(),
        ));
    }

    Ok(candidate.trim_end_matches('/').to_string())
}

fn resolve_pat_api_base(endpoints: &CopilotTokenEndpoints) -> Result<String, ProviderAuthError> {
    let candidate = endpoints.api.as_deref().ok_or_else(|| {
        ProviderAuthError::SourceResolutionFailed(
            "Copilot account response has no API endpoint".to_string(),
        )
    })?;
    let url = url::Url::parse(candidate).map_err(|_| {
        ProviderAuthError::SourceResolutionFailed(
            "invalid Copilot account API endpoint".to_string(),
        )
    })?;
    let github_host = url
        .host_str()
        .is_some_and(|host| host == "githubcopilot.com" || host.ends_with(".githubcopilot.com"));
    if url.scheme() != "https"
        || !github_host
        || url.port_or_known_default() != Some(443)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ProviderAuthError::SourceResolutionFailed(
            "Copilot account API endpoint must be a GitHub-owned HTTPS endpoint".to_string(),
        ));
    }
    Ok(url.as_str().trim_end_matches('/').to_string())
}

fn url_is_within_api_base(request_url: &str, api_base: &str) -> bool {
    let (Ok(request), Ok(base)) = (url::Url::parse(request_url), url::Url::parse(api_base)) else {
        return false;
    };
    if request.scheme() != base.scheme()
        || request.host_str() != base.host_str()
        || request.port_or_known_default() != base.port_or_known_default()
        || !request.username().is_empty()
        || request.password().is_some()
    {
        return false;
    }
    let base_path = base.path().trim_end_matches('/');
    base_path.is_empty()
        || request.path() == base_path
        || request
            .path()
            .strip_prefix(base_path)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn api_base_from_proxy_token(token: &str) -> Option<String> {
    let proxy = token
        .split(';')
        .find_map(|part| part.strip_prefix("proxy-ep="))?;
    let host = proxy
        .strip_prefix("proxy.")
        .map_or_else(|| proxy.to_string(), |suffix| format!("api.{suffix}"));
    Some(format!("https://{host}"))
}

fn token_exchange_requires_account_action(status: u16, body: &[u8]) -> bool {
    if status == 401 {
        return true;
    }
    if status != 403 {
        return false;
    }
    let envelope = serde_json::from_slice::<CopilotTokenErrorEnvelope>(body).unwrap_or_default();
    envelope
        .error_details
        .and_then(|details| details.notification_id)
        .is_some()
}

fn token_refresh_at(
    token: &CopilotTokenEnvelope,
    now: DateTime<Utc>,
) -> Result<DateTime<Utc>, ProviderAuthError> {
    let expires_at = i64::try_from(token.expires_at)
        .ok()
        .and_then(|timestamp| DateTime::<Utc>::from_timestamp(timestamp, 0))
        .ok_or_else(|| {
            ProviderAuthError::SourceResolutionFailed(
                "Copilot token expiry is out of range".to_string(),
            )
        })?;
    let lifetime_seconds = expires_at.signed_duration_since(now).num_seconds();
    if lifetime_seconds <= 1 {
        return Err(ProviderAuthError::SourceResolutionFailed(
            "Copilot token expiry is not in the usable future".to_string(),
        ));
    }
    let refresh_seconds = i64::try_from(token.refresh_in).map_err(|_| {
        ProviderAuthError::SourceResolutionFailed(
            "Copilot token refresh interval is out of range".to_string(),
        )
    })?;
    let refresh_from_hint = now
        .checked_add_signed(Duration::seconds(refresh_seconds))
        .ok_or_else(|| {
            ProviderAuthError::SourceResolutionFailed(
                "Copilot token refresh deadline is out of range".to_string(),
            )
        })?;
    let expiry_skew_seconds =
        CAPI_EXPIRY_SKEW_SECONDS.min((lifetime_seconds / 5).clamp(1, CAPI_EXPIRY_SKEW_SECONDS));
    let refresh_before_expiry = expires_at
        .checked_sub_signed(Duration::seconds(expiry_skew_seconds))
        .ok_or_else(|| {
            ProviderAuthError::SourceResolutionFailed(
                "Copilot token expiry refresh deadline is out of range".to_string(),
            )
        })?;
    let minimum_future = now
        .checked_add_signed(Duration::seconds(1))
        .ok_or_else(|| {
            ProviderAuthError::SourceResolutionFailed(
                "Copilot minimum refresh deadline is out of range".to_string(),
            )
        })?;
    Ok(refresh_from_hint
        .min(refresh_before_expiry)
        .max(minimum_future)
        .min(expires_at))
}

fn model_discovery_deadline(now: DateTime<Utc>, delay: StdDuration) -> DateTime<Utc> {
    let seconds = delay.as_secs().min(MODEL_DISCOVERY_MAX_RETRY_SECONDS);
    let Ok(seconds) = i64::try_from(seconds) else {
        return DateTime::<Utc>::MAX_UTC;
    };
    let Some(delta) = Duration::try_seconds(seconds) else {
        return DateTime::<Utc>::MAX_UTC;
    };
    now.checked_add_signed(delta)
        .unwrap_or(DateTime::<Utc>::MAX_UTC)
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), ProviderAuthError> {
    let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
        ProviderAuthError::SourceResolutionFailed(format!(
            "invalid Copilot header name {name}: {error}"
        ))
    })?;
    let mut value = HeaderValue::from_str(value).map_err(|error| {
        ProviderAuthError::SourceResolutionFailed(format!("invalid Copilot header {name}: {error}"))
    })?;
    if name == reqwest::header::AUTHORIZATION {
        value.set_sensitive(true);
    }
    headers.insert(name, value);
    Ok(())
}

async fn refresh_github_tokens(
    persistence: ProviderAuthPersistence,
    prepare: meerkat_auth_core::resolver::ManagedStoreOAuthRefreshPrepareFn,
    credential_identity: AuthCredentialIdentity,
    config: CopilotBackendConfig,
    force_refresh: bool,
) -> Result<PersistedTokens, RefreshError> {
    // The GitHub refresh grant follows no redirects; without that client no
    // refresh begins.
    let http = meerkat_auth_core::auth_oauth::credential_http_client()
        .map_err(|error| RefreshError::Refresh(error.to_string()))?;
    let preparation = ManagedStoreOAuthRefreshPreparationSlot::new(prepare);
    let token_store = persistence.token_store();
    let key = meerkat_core::auth::TokenKey::from_credential_identity(&credential_identity);
    let refresh_key = key.clone();
    let preparation_for_refresh = preparation.clone();
    let refresh_fn: RefreshFn = Box::new(move || {
        let token_store = token_store.clone();
        let key = refresh_key.clone();
        let preparation = preparation_for_refresh.clone();
        let config = config.clone();
        let http = http.clone();
        Box::pin(async move {
            let current = token_store
                .load_for(&key, TokenStoreAccess::Unattended)
                .await
                .map_err(RefreshError::from_token_store)?
                .ok_or_else(|| {
                    RefreshError::Refresh(
                        "Copilot source credential disappeared before refresh".to_string(),
                    )
                })?;
            match preparation.claim_refresh_owner(current.clone()).await? {
                LockedManagedStoreOAuthRefresh::UseCached(cached) => Ok(cached),
                LockedManagedStoreOAuthRefresh::Refresh(transaction) => {
                    let refresh_token = match current.refresh_token.clone() {
                        Some(refresh_token) => refresh_token,
                        None => {
                            return Err(transaction.fail(RefreshError::Observed {
                            message: "GitHub credential has no refresh token".to_string(),
                            observation:
                                meerkat_core::RefreshFailureObservation::local_credential_unusable(),
                            }).await);
                        }
                    };
                    let mut endpoints = meerkat_auth_core::oauth_flow::oauth_provider_endpoints(
                        meerkat_core::OAuthProviderIdentity::GitHubCopilot,
                        "",
                    );
                    let configured = match config.endpoints() {
                        Ok(configured) => configured,
                        Err(error) => {
                            return Err(transaction
                                .fail(RefreshError::Refresh(error.to_string()))
                                .await);
                        }
                    };
                    endpoints.token_url = configured.oauth_token_url.to_string();
                    let result =
                        match exchange_refresh_token(&http, &endpoints, &refresh_token, None).await
                        {
                            Ok(result) => result,
                            Err(error) => {
                                return Err(transaction.fail(oauth_refresh_error(error)).await);
                            }
                        };
                    let now = Utc::now();
                    let expires_at = match result.expires_at_from(now) {
                        Ok(expires_at) => expires_at,
                        Err(error) => {
                            return Err(transaction
                                .fail(RefreshError::Refresh(error.to_string()))
                                .await);
                        }
                    };
                    let refreshed = PersistedTokens {
                        auth_mode: PersistedAuthMode::GithubCopilotOauth,
                        primary_secret: Some(result.access_token),
                        refresh_token: result.refresh_token.or(Some(refresh_token)),
                        id_token: result.id_token,
                        expires_at,
                        last_refresh: Some(now),
                        scopes: result
                            .scope
                            .as_deref()
                            .map(|scope| scope.split_whitespace().map(str::to_string).collect())
                            .unwrap_or_default(),
                        account_id: current.account_id,
                        metadata: serde_json::Value::Null,
                    };
                    transaction.commit(refreshed).await
                }
            }
        })
    });
    let refreshed = if force_refresh {
        persistence
            .refresh_coordinator()
            .with_forced_refresh(key.clone(), refresh_fn)
            .await
    } else {
        persistence
            .refresh_coordinator()
            .with_refresh(key.clone(), refresh_fn)
            .await
    }?;
    preparation
        .finish_coordinated_refresh(
            persistence.refresh_coordinator(),
            persistence.token_store(),
            key,
            refreshed,
        )
        .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    mod redirect_fixture {
        #![allow(clippy::unwrap_used)]
        /// A credential endpoint that answers every request with a `302` to a
        /// second listener, which counts the requests that reach it. The
        /// `Location` and the body carry canaries that must never be rendered.
        pub(super) const LOCATION_CANARY: &str = "redirect-location-secret-canary";
        pub(super) const BODY_CANARY: &str = "redirect-body-secret-canary";

        pub(super) async fn spawn_redirecting_endpoint()
        -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicUsize, Ordering};
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            async fn read_request(stream: &mut tokio::net::TcpStream) {
                let mut buffer = Vec::new();
                let mut chunk = [0_u8; 4096];
                loop {
                    let Ok(read) = stream.read(&mut chunk).await else {
                        return;
                    };
                    if read == 0 {
                        return;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&buffer);
                    if let Some(end) = text.find("\r\n\r\n") {
                        let length = text[..end]
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if buffer.len() >= end + 4 + length {
                            return;
                        }
                    }
                }
            }

            let hits = Arc::new(AtomicUsize::new(0));
            let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let target_addr = target.local_addr().unwrap();
            let counted = Arc::clone(&hits);
            tokio::spawn(async move {
                while let Ok((mut stream, _)) = target.accept().await {
                    counted.fetch_add(1, Ordering::SeqCst);
                    read_request(&mut stream).await;
                    let _ = stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await;
                }
            });
            let endpoint = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint_addr = endpoint.local_addr().unwrap();
            tokio::spawn(async move {
                while let Ok((mut stream, _)) = endpoint.accept().await {
                    read_request(&mut stream).await;
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nlocation: http://{target_addr}/token?leak={LOCATION_CANARY}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{BODY_CANARY}",
                        BODY_CANARY.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                }
            });
            (format!("http://{endpoint_addr}/token"), hits)
        }
    }

    #[tokio::test]
    async fn copilot_credential_requests_never_follow_a_redirect() {
        use redirect_fixture::{BODY_CANARY, LOCATION_CANARY, spawn_redirecting_endpoint};
        let (url, target_hits) = spawn_redirecting_endpoint().await;
        let error = match ReqwestCopilotHttpTransport::new()
            .get(&url, HeaderMap::new())
            .await
        {
            Err(error) => error,
            Ok(response) => panic!("a redirect was not refused: status {}", response.status),
        };
        assert!(
            matches!(
                error,
                CopilotHttpTransportError::RedirectRefused { status: 302 }
            ),
            "{error:?}"
        );
        let rendered = format!("{error} {error:?}");
        assert!(!rendered.contains(LOCATION_CANARY) && !rendered.contains(BODY_CANARY));
        assert_eq!(target_hits.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn copilot_rejection_status_survives_unreadable_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for status in [401, 403] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/models", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut head = Vec::new();
                let mut chunk = [0; 1024];
                while !head.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(read, 0);
                    head.extend_from_slice(&chunk[..read]);
                }
                socket.write_all(format!(
                    "HTTP/1.1 {status} Rejected\r\nContent-Length: 100\r\nConnection: close\r\n\r\ntruncated"
                ).as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
            let response = ReqwestCopilotHttpTransport::new()
                .get(&url, HeaderMap::new())
                .await
                .unwrap();
            assert_eq!(response.status, status);
            assert!(response.body.is_empty());
            server.await.unwrap();
        }
    }
    use meerkat_core::auth::TokenStore;
    use meerkat_core::provider_matrix::openai::{OpenAiAuthMethod, OpenAiBackendKind};
    use meerkat_core::{
        AuthBindingRef, AuthProfile, BackendProfile, BindingId, BindingOrigin, BindingPolicy,
        CredentialAccountId, CredentialAccountRef, CredentialSourceSpec, RealmId,
    };
    use meerkat_llm_core::provider_runtime::ProviderRuntimeCatalog;
    use std::collections::VecDeque;

    struct SpecAuthorizer;

    #[async_trait]
    impl HttpAuthorizer for SpecAuthorizer {
        async fn authorize(
            &self,
            _request: &mut HttpAuthorizationRequest<'_>,
        ) -> Result<(), AuthError> {
            Ok(())
        }

        fn label(&self) -> &'static str {
            "copilot-spec-test"
        }
    }

    #[test]
    fn chat_completions_spec_constructor_preserves_historical_image_input_support() {
        let spec = CopilotChatCompletionsClientSpec::new(
            Provider::OpenAI,
            "model".to_string(),
            "https://example.test".to_string(),
            Arc::new(SpecAuthorizer),
            true,
            true,
            true,
            false,
        );

        assert!(spec.supports_image_input());
        assert!(!spec.with_image_input_support(false).supports_image_input());
    }

    fn account_identity() -> AuthCredentialIdentity {
        AuthCredentialIdentity::Account(CredentialAccountRef {
            realm: RealmId::global(),
            account: CredentialAccountId::parse("github_copilot").expect("valid account id"),
        })
    }

    fn persistence_id() -> meerkat_core::auth::ProviderAuthPersistenceId {
        ProviderAuthPersistence::new(
            Arc::new(meerkat_auth_core::auth_store::EphemeralTokenStore::new()),
            Arc::new(meerkat_auth_core::auth_store::InMemoryCoordinator::new()),
        )
        .authority_id()
    }

    struct ScriptedTransport {
        responses: Mutex<VecDeque<CopilotHttpResponse>>,
        requests: Mutex<Vec<(String, HeaderMap)>>,
    }

    struct DelayedTransport {
        inner: ScriptedTransport,
    }

    struct CancelFirstTransport {
        inner: ScriptedTransport,
        calls: AtomicU64,
        first_started: tokio::sync::Notify,
    }

    #[async_trait]
    impl CopilotHttpTransport for DelayedTransport {
        async fn get(
            &self,
            url: &str,
            headers: HeaderMap,
        ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
            tokio::time::sleep(StdDuration::from_millis(10)).await;
            self.inner.get(url, headers).await
        }
    }

    #[async_trait]
    impl CopilotHttpTransport for CancelFirstTransport {
        async fn get(
            &self,
            url: &str,
            headers: HeaderMap,
        ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.first_started.notify_one();
                return futures::future::pending().await;
            }
            self.inner.get(url, headers).await
        }
    }

    struct NoopLlmClient;

    #[async_trait]
    impl meerkat_llm_core::LlmClient for NoopLlmClient {
        fn project_replay_messages(
            &self,
            messages: &[meerkat_core::Message],
        ) -> Result<Vec<meerkat_core::Message>, meerkat_llm_core::LlmError> {
            Ok(messages.to_vec())
        }

        fn stream<'a>(
            &'a self,
            _request: &'a meerkat_llm_core::LlmRequest,
        ) -> meerkat_llm_core::LlmStream<'a> {
            Box::pin(futures::stream::empty())
        }

        fn provider(&self) -> Provider {
            Provider::OpenAI
        }

        async fn health_check(&self) -> Result<(), meerkat_llm_core::LlmError> {
            Ok(())
        }
    }

    struct DispatchCountingClient {
        dispatches: Arc<AtomicU64>,
    }

    #[async_trait]
    impl meerkat_llm_core::LlmClient for DispatchCountingClient {
        fn project_replay_messages(
            &self,
            messages: &[meerkat_core::Message],
        ) -> Result<Vec<meerkat_core::Message>, meerkat_llm_core::LlmError> {
            Ok(messages.to_vec())
        }

        fn stream<'a>(
            &'a self,
            _request: &'a meerkat_llm_core::LlmRequest,
        ) -> meerkat_llm_core::LlmStream<'a> {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            Box::pin(futures::stream::empty())
        }

        fn provider(&self) -> Provider {
            Provider::OpenAI
        }

        async fn health_check(&self) -> Result<(), meerkat_llm_core::LlmError> {
            Ok(())
        }
    }

    impl ScriptedTransport {
        fn new(responses: impl IntoIterator<Item = CopilotHttpResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn request_count(&self) -> usize {
            self.requests.lock().len()
        }
    }

    #[async_trait]
    impl CopilotHttpTransport for ScriptedTransport {
        async fn get(
            &self,
            url: &str,
            headers: HeaderMap,
        ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
            self.requests.lock().push((url.to_string(), headers));
            self.responses.lock().pop_front().ok_or_else(|| {
                CopilotHttpTransportError::Scripted("unexpected Copilot HTTP request".to_string())
            })
        }
    }

    fn json_response(status: u16, body: serde_json::Value) -> CopilotHttpResponse {
        CopilotHttpResponse {
            status,
            body: serde_json::to_vec(&body).expect("serialize scripted response"),
            retry_after: None,
        }
    }

    fn token_response(token: &str) -> CopilotHttpResponse {
        json_response(
            200,
            serde_json::json!({
                "token": token,
                "expires_at": Utc::now().timestamp() + 3600,
                "refresh_in": 1800,
                "endpoints": {"api": "https://copilot.example.test"}
            }),
        )
    }

    fn models_response() -> CopilotHttpResponse {
        json_response(
            200,
            serde_json::json!({
                "data": [
                    {
                        "id": "gpt-test",
                        "supported_endpoints": ["/responses", "/chat/completions"]
                    },
                    {
                        "id": "claude-test",
                        "supported_endpoints": ["/v1/messages"]
                    },
                    {
                        "id": "gemini-test",
                        "supported_endpoints": ["/chat/completions"]
                    }
                ]
            }),
        )
    }

    fn validated_binding() -> (AuthBindingRef, ValidatedBinding) {
        let auth_binding = AuthBindingRef {
            realm: RealmId::global(),
            binding: BindingId::parse("copilot_openai").expect("valid binding"),
            profile: None,
            origin: BindingOrigin::Configured,
        };
        let backend = BackendProfile {
            id: "copilot".to_string(),
            provider: Provider::OpenAI,
            backend_kind: OpenAiBackendKind::Copilot.as_str().to_string(),
            base_url: None,
            options: serde_json::Value::Null,
            server: None,
        };
        let auth = AuthProfile {
            id: "copilot".to_string(),
            provider: Provider::OpenAI,
            auth_method: OpenAiAuthMethod::GitHubCopilotOauth.as_str().to_string(),
            source: CredentialSourceSpec::ManagedStore,
            constraints: Default::default(),
            metadata_defaults: Default::default(),
        };
        let validated = ProviderRuntimeCatalog::validate_binding_with_credential_identity(
            &auth_binding,
            account_identity(),
            &backend,
            &auth,
            &BindingPolicy::default(),
        )
        .expect("valid Copilot binding");
        (auth_binding, validated)
    }

    fn validated_pat_binding(provider: Provider) -> ValidatedBinding {
        let (auth_binding, oauth) = validated_binding();
        let mut backend = oauth.backend_profile().as_ref().clone();
        backend.provider = provider;
        let mut auth = oauth.auth_profile().as_ref().clone();
        auth.provider = provider;
        auth.auth_method = "github_copilot_pat".to_string();
        auth.source = CredentialSourceSpec::Env {
            env: "COPILOT_GITHUB_TOKEN".to_string(),
            fallback: Vec::new(),
        };
        ProviderRuntimeCatalog::validate_binding(
            &auth_binding,
            &backend,
            &auth,
            &BindingPolicy::default(),
        )
        .expect("valid environment-backed PAT binding")
    }

    fn pat_environment(secret: &'static str) -> ResolverEnvironment {
        ResolverEnvironment::testing().with_env_lookup(move |name| {
            (name == "COPILOT_GITHUB_TOKEN").then(|| secret.to_string())
        })
    }

    fn pat_account_response() -> CopilotHttpResponse {
        json_response(
            200,
            serde_json::json!({
                "endpoints": {"api": "https://api.business.githubcopilot.com/account/"}
            }),
        )
    }

    #[tokio::test]
    async fn pat_rejection_updates_published_binding_status() {
        use meerkat_core::handles::{AuthLeasePhase, LeaseKey};

        for status in [401, 403] {
            let transport = Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
            ]));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport,
            };
            let machine = meerkat_runtime::MeerkatMachine::ephemeral();
            let handle = machine.generated_auth_lease_handle();
            let binding = validated_pat_binding(Provider::OpenAI);
            let key = LeaseKey::from_credential_identity(binding.credential_identity());
            let env =
                pat_environment("synthetic-published-pat").with_auth_lease_handle(handle.clone());
            let resolved = runtime.resolve(&binding, &env).await.unwrap();
            let publication = handle.publish_lease(&key, u64::MAX).unwrap();
            resolved
                .authorizer
                .observe_lease_publication(&publication)
                .unwrap();
            for _ in 0..2 {
                let results = std::thread::scope(|scope| {
                    let attempts: [_; 2] =
                        std::array::from_fn(|_| {
                            scope.spawn(|| {
                        resolved.authorizer.observe_response_status(&HttpAuthorizationResponse {
                            method: "POST",
                            url: "https://api.business.githubcopilot.com/account/responses",
                            status,
                        })
                    })
                        });
                    attempts.map(|attempt| attempt.join().unwrap())
                });
                for result in results {
                    assert!(matches!(result, Err(AuthError::UserReauthRequired)));
                }
                let snapshot = handle.snapshot(&key);
                assert_eq!(snapshot.phase, Some(AuthLeasePhase::ReauthRequired));
                assert_eq!(
                    meerkat_core::project_published_auth_status(Utc::now(), None, &snapshot).phase,
                    meerkat_core::AuthStatusPhase::ReauthRequired
                );
            }
        }
    }

    #[tokio::test]
    async fn pat_publications_release_retired_runtime_custody() {
        use meerkat_core::handles::{AuthLeasePhase, LeaseKey};
        use meerkat_runtime::MeerkatMachine;
        use meerkat_runtime::meerkat_machine::StagedCustodyRelease;
        use meerkat_runtime::store::InMemoryRuntimeStore;

        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
            ])),
        };
        let store_a = Arc::new(InMemoryRuntimeStore::new());
        let store_b = Arc::new(InMemoryRuntimeStore::new());
        let staged_a = MeerkatMachine::persistent_governed_staged(store_a.clone(), None).unwrap();
        let staged_b = MeerkatMachine::persistent_governed_staged(store_b.clone(), None).unwrap();
        let handle_a = staged_a.generated_auth_lease_handle();
        let handle_b = staged_b.generated_auth_lease_handle();
        let retired_owner = Arc::downgrade(&handle_a.clone_handle());
        let binding = validated_pat_binding(Provider::OpenAI);
        let key = LeaseKey::from_credential_identity(binding.credential_identity());
        let env_a =
            pat_environment("synthetic-shared-owner-pat").with_auth_lease_handle(handle_a.clone());
        let env_b =
            pat_environment("synthetic-shared-owner-pat").with_auth_lease_handle(handle_b.clone());
        let resolved_a = runtime.resolve(&binding, &env_a).await.unwrap();
        let resolved_b = runtime.resolve(&binding, &env_b).await.unwrap();
        assert!(Arc::ptr_eq(
            &resolved_a.authorizer.state,
            &resolved_b.authorizer.state,
        ));
        resolved_a
            .authorizer
            .observe_lease_publication(&handle_a.publish_lease(&key, u64::MAX).unwrap())
            .unwrap();
        resolved_b
            .authorizer
            .observe_lease_publication(&handle_b.publish_lease(&key, u64::MAX).unwrap())
            .unwrap();

        drop(resolved_a);
        drop(env_a);
        drop(handle_a);
        let release = staged_a.release();
        let reopened = MeerkatMachine::persistent_governed_staged(store_a, None)
            .expect("retired publication must not retain the runtime's execution custody");
        assert_eq!(release, StagedCustodyRelease::Released);
        assert!(retired_owner.upgrade().is_none());
        assert_eq!(reopened.release(), StagedCustodyRelease::Released);

        assert!(matches!(
            resolved_b
                .authorizer
                .observe_response_status(&HttpAuthorizationResponse {
                    method: "POST",
                    url: "https://api.business.githubcopilot.com/account/responses",
                    status: 401,
                }),
            Err(AuthError::UserReauthRequired)
        ));
        assert_eq!(
            handle_b.snapshot(&key).phase,
            Some(AuthLeasePhase::ReauthRequired)
        );
        assert_eq!(resolved_b.authorizer.state.publications.lock().len(), 1);
        drop(resolved_b);
        drop(env_b);
        drop(handle_b);
        assert_eq!(staged_b.release(), StagedCustodyRelease::Released);
    }

    #[tokio::test]
    async fn pat_rejection_reports_generated_failure_and_retries_publication() {
        use meerkat_core::handles::{AuthLeasePhase, LeaseKey};

        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let authority = Arc::new(meerkat_runtime::handles::RuntimeAuthLeaseHandle::new());
        let handle =
            meerkat_runtime::protocol_auth_lease_lifecycle_publication::generated_auth_lease_handle(
                authority.clone(),
            )
            .unwrap();
        let binding = validated_pat_binding(Provider::OpenAI);
        let key = LeaseKey::from_credential_identity(binding.credential_identity());
        let env =
            pat_environment("synthetic-publication-failure").with_auth_lease_handle(handle.clone());
        let resolved = runtime.resolve(&binding, &env).await.unwrap();
        let publication = handle.publish_lease(&key, u64::MAX).unwrap();
        resolved
            .authorizer
            .observe_lease_publication(&publication)
            .unwrap();
        let published = handle.snapshot(&key);
        authority.fail_next_reauth_required_for_test(&key);

        assert!(matches!(
            resolved.authorizer.reject_pat(),
            ProviderAuthError::SourceResolutionFailed(_)
        ));
        assert_eq!(handle.snapshot(&key), published);
        assert!(resolved.model_snapshot().is_none());
        assert!(matches!(
            resolved.authorizer.prime().await,
            Err(ProviderAuthError::Auth(AuthError::UserReauthRequired))
        ));
        assert_eq!(
            handle.snapshot(&key).phase,
            Some(AuthLeasePhase::ReauthRequired)
        );
        assert_eq!(transport.request_count(), 2);
    }

    #[tokio::test]
    async fn pat_rejection_does_not_invalidate_replacement_publication() {
        use meerkat_core::handles::LeaseKey;

        for already_rejected in [false, true] {
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: Arc::new(ScriptedTransport::new([
                    pat_account_response(),
                    models_response(),
                ])),
            };
            let machine = meerkat_runtime::MeerkatMachine::ephemeral();
            let handle = machine.generated_auth_lease_handle();
            let binding = validated_pat_binding(Provider::OpenAI);
            let key = LeaseKey::from_credential_identity(binding.credential_identity());
            let env =
                pat_environment("synthetic-retired-pat").with_auth_lease_handle(handle.clone());
            let resolved = runtime.resolve(&binding, &env).await.unwrap();
            let publication = handle.publish_lease(&key, u64::MAX).unwrap();
            resolved
                .authorizer
                .observe_lease_publication(&publication)
                .unwrap();
            if already_rejected {
                assert!(matches!(
                    resolved.authorizer.reject_pat(),
                    ProviderAuthError::Auth(AuthError::UserReauthRequired)
                ));
            }
            handle.publish_lease(&key, u64::MAX).unwrap();
            let replacement = handle.snapshot(&key);
            assert!(matches!(
                resolved
                    .authorizer
                    .observe_response_status(&HttpAuthorizationResponse {
                        method: "POST",
                        url: "https://api.business.githubcopilot.com/account/responses",
                        status: 401,
                    }),
                Err(AuthError::UserReauthRequired)
            ));
            assert_eq!(handle.snapshot(&key), replacement);
            assert!(matches!(
                resolved.authorizer.prepare_request().await,
                Err(AuthError::UserReauthRequired)
            ));
            assert_eq!(handle.snapshot(&key), replacement);
            assert!(matches!(
                resolved.authorizer.observe_lease_publication(&publication),
                Err(AuthError::UserReauthRequired)
            ));
            assert_eq!(handle.snapshot(&key), replacement);
            let republication = handle.publish_lease(&key, u64::MAX).unwrap();
            assert!(matches!(
                resolved
                    .authorizer
                    .observe_lease_publication(&republication),
                Err(AuthError::UserReauthRequired)
            ));
            assert_eq!(
                handle.snapshot(&key).phase,
                Some(meerkat_core::handles::AuthLeasePhase::ReauthRequired)
            );
        }
    }

    #[tokio::test]
    async fn pat_rejection_uses_latest_shared_publication_after_late_callback() {
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
            ])),
        };
        let machine = meerkat_runtime::MeerkatMachine::ephemeral();
        let handle = machine.generated_auth_lease_handle();
        let binding = validated_pat_binding(Provider::OpenAI);
        let key = meerkat_core::handles::LeaseKey::from_credential_identity(
            binding.credential_identity(),
        );
        let env =
            pat_environment("synthetic-shared-publication").with_auth_lease_handle(handle.clone());
        let first = runtime.resolve(&binding, &env).await.unwrap();
        let second = runtime.resolve(&binding, &env).await.unwrap();
        let old = handle.publish_lease(&key, u64::MAX).unwrap();
        let current = handle.publish_lease(&key, u64::MAX).unwrap();
        second
            .authorizer
            .observe_lease_publication(&current)
            .unwrap();
        first.authorizer.observe_lease_publication(&old).unwrap();
        assert!(matches!(
            first.authorizer.reject_pat(),
            ProviderAuthError::Auth(AuthError::UserReauthRequired)
        ));
        assert_eq!(
            handle.snapshot(&key).phase,
            Some(meerkat_core::handles::AuthLeasePhase::ReauthRequired)
        );
    }

    #[tokio::test]
    async fn pat_model_metadata_is_redacted_only_in_diagnostics() {
        let secret = "synthetic-model-id-canary";
        let mut response = models_response();
        let mut body: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        body["data"][0]["id"] = secret.into();
        response.body = serde_json::to_vec(&body).unwrap();
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: Arc::new(ScriptedTransport::new([pat_account_response(), response])),
        };
        let resolved = runtime
            .resolve(
                &validated_pat_binding(Provider::OpenAI),
                &pat_environment(secret),
            )
            .await
            .unwrap();
        let route = resolve_route_for_state(
            &resolved.authorizer.state,
            Provider::OpenAI,
            "not-available",
        )
        .unwrap();
        assert!(resolved.model_snapshot().unwrap().model(secret).is_some());
        assert!(
            !route
                .unavailable_message(
                    Provider::OpenAI,
                    "not-available",
                    CopilotEndpoint::Responses,
                )
                .contains(secret)
        );
        drop(resolved);
        assert!(
            !route
                .unavailable_message(
                    Provider::OpenAI,
                    "not-available",
                    CopilotEndpoint::Responses,
                )
                .contains(secret)
        );
    }

    #[tokio::test]
    async fn pat_rejection_before_lease_publication_stays_terminal() {
        use meerkat_core::handles::{AuthLeasePhase, LeaseKey};
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
            ])),
        };
        let machine = meerkat_runtime::MeerkatMachine::ephemeral();
        let handle = machine.generated_auth_lease_handle();
        let binding = validated_pat_binding(Provider::OpenAI);
        let key = LeaseKey::from_credential_identity(binding.credential_identity());
        let env =
            pat_environment("synthetic-publication-race").with_auth_lease_handle(handle.clone());
        let resolved = runtime.resolve(&binding, &env).await.unwrap();
        assert!(matches!(
            resolved.authorizer.reject_pat(),
            ProviderAuthError::Auth(AuthError::UserReauthRequired)
        ));
        assert_eq!(handle.snapshot(&key).phase, None);
        let publication = handle.publish_lease(&key, u64::MAX).unwrap();
        assert!(matches!(
            resolved.authorizer.observe_lease_publication(&publication),
            Err(AuthError::UserReauthRequired)
        ));
        assert_eq!(
            handle.snapshot(&key).phase,
            Some(AuthLeasePhase::ReauthRequired)
        );
    }

    #[tokio::test]
    async fn pat_discovery_rejection_updates_published_binding_status() {
        use meerkat_core::handles::{AuthLeasePhase, LeaseKey};

        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
            json_response(403, serde_json::json!({})),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport,
        };
        let machine = meerkat_runtime::MeerkatMachine::ephemeral();
        let handle = machine.generated_auth_lease_handle();
        let binding = validated_pat_binding(Provider::OpenAI);
        let key = LeaseKey::from_credential_identity(binding.credential_identity());
        let clock = Arc::new(Mutex::new(Utc::now()));
        let mut env =
            pat_environment("synthetic-discovery-pat").with_auth_lease_handle(handle.clone());
        env.now = {
            let clock = clock.clone();
            Arc::new(move || *clock.lock())
        };
        let resolved = runtime.resolve(&binding, &env).await.unwrap();
        let publication = handle.publish_lease(&key, u64::MAX).unwrap();
        resolved
            .authorizer
            .observe_lease_publication(&publication)
            .unwrap();
        *clock.lock() += Duration::seconds(301);
        assert!(matches!(
            resolved.authorizer.prime().await,
            Err(ProviderAuthError::Auth(AuthError::UserReauthRequired))
        ));
        assert_eq!(
            handle.snapshot(&key).phase,
            Some(AuthLeasePhase::ReauthRequired)
        );
    }

    struct GatedTransport {
        inner: ScriptedTransport,
        gate: AtomicBool,
        release: tokio::sync::Semaphore,
    }

    #[async_trait]
    impl CopilotHttpTransport for GatedTransport {
        async fn get(
            &self,
            url: &str,
            headers: HeaderMap,
        ) -> Result<CopilotHttpResponse, CopilotHttpTransportError> {
            if self.gate.swap(false, Ordering::AcqRel) {
                self.release.acquire().await.unwrap().forget();
            }
            self.inner.get(url, headers).await
        }
    }

    #[tokio::test]
    async fn pat_waiters_share_zero_delay_discovery_fallback() {
        enum Completion {
            Shared,
            Cancelled,
            Rejected,
            Overtaken,
        }

        for completion in [
            Completion::Shared,
            Completion::Cancelled,
            Completion::Rejected,
            Completion::Overtaken,
        ] {
            let zero_retry = || {
                let mut response = json_response(429, serde_json::json!({}));
                response.retry_after = Some(StdDuration::ZERO);
                response
            };
            let transport = Arc::new(GatedTransport {
                inner: ScriptedTransport::new([
                    pat_account_response(),
                    models_response(),
                    zero_retry(),
                    zero_retry(),
                ]),
                gate: AtomicBool::new(false),
                release: tokio::sync::Semaphore::new(0),
            });
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let clock = Arc::new(Mutex::new(Utc::now()));
            let mut env = pat_environment("synthetic-shared-pat");
            env.now = {
                let clock = clock.clone();
                Arc::new(move || *clock.lock())
            };
            let resolved = runtime
                .resolve(&validated_pat_binding(Provider::OpenAI), &env)
                .await
                .unwrap();
            *clock.lock() += Duration::seconds(301);
            transport.gate.store(true, Ordering::Release);
            let mut leader = Box::pin(resolved.authorizer.prime());
            let mut waiter = Box::pin(resolved.authorizer.prime());
            assert!(futures::poll!(leader.as_mut()).is_pending());
            assert!(futures::poll!(waiter.as_mut()).is_pending());
            match completion {
                Completion::Cancelled => drop(leader),
                _ => {
                    transport.release.add_permits(1);
                    leader.await.unwrap();
                }
            }
            match completion {
                Completion::Rejected => {
                    assert!(matches!(
                        resolved.authorizer.reject_pat(),
                        ProviderAuthError::Auth(AuthError::UserReauthRequired)
                    ));
                    assert!(matches!(
                        waiter.await,
                        Err(ProviderAuthError::Auth(AuthError::UserReauthRequired))
                    ));
                    assert_eq!(transport.inner.request_count(), 3);
                    assert!(resolved.model_snapshot().is_none());
                }
                Completion::Overtaken => {
                    resolved.authorizer.prime().await.unwrap();
                    waiter.await.unwrap();
                    assert_eq!(transport.inner.request_count(), 4);
                }
                Completion::Shared | Completion::Cancelled => {
                    waiter.await.unwrap();
                    assert_eq!(transport.inner.request_count(), 3);
                    resolved.authorizer.prime().await.unwrap();
                    assert_eq!(transport.inner.request_count(), 4);
                }
            }
        }
    }

    #[tokio::test]
    async fn pat_rejection_does_not_wait_for_discovery_body() {
        use axum::{Router, body::Body, http::StatusCode, routing::get};

        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let app = Router::new().route(
                "/models",
                get(move || async move {
                    (
                        status,
                        Body::from_stream(futures::stream::pending::<
                            Result<String, std::convert::Infallible>,
                        >()),
                    )
                }),
            );
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: Arc::new(ScriptedTransport::new([
                    pat_account_response(),
                    models_response(),
                ])),
            };
            let mut resolved = runtime
                .resolve(
                    &validated_pat_binding(Provider::OpenAI),
                    &pat_environment("synthetic-stalled-pat"),
                )
                .await
                .unwrap();
            {
                let mut cache = resolved.authorizer.state.cache.write();
                let cached = cache.as_mut().unwrap();
                cached.api_base = format!("http://{address}");
                cached.model_discovery_refresh_at = Some(Utc::now() - Duration::seconds(1));
            }
            Arc::get_mut(&mut resolved.authorizer).unwrap().transport =
                Arc::new(ReqwestCopilotHttpTransport::new());
            let result =
                tokio::time::timeout(StdDuration::from_secs(1), resolved.authorizer.prime()).await;
            server.abort();
            assert!(matches!(
                result,
                Ok(Err(ProviderAuthError::Auth(AuthError::UserReauthRequired)))
            ));
            assert!(resolved.model_snapshot().is_none());
        }
    }

    #[tokio::test]
    async fn pat_discovers_and_authorizes_without_oauth_store_or_exchange() {
        for (provider, model) in [
            (Provider::OpenAI, "gpt-test"),
            (Provider::Anthropic, "claude-test"),
            (Provider::Gemini, "gemini-test"),
        ] {
            let transport = Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
            ]));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let binding = validated_pat_binding(provider);
            let resolved = runtime
                .resolve(&binding, &pat_environment("test-pat"))
                .await
                .expect("PAT requires no persistence or OAuth lifecycle");
            assert!(matches!(
                resolved.route_for(provider, model),
                CopilotModelAccess::Available { .. }
            ));
            let authorizer = resolved.authorizer();
            assert!(authorizer.persistence_authority_id().is_none());
            assert!(authorizer.expires_at().is_none());
            let connection = meerkat_llm_core::provider_runtime::ResolvedConnection {
                provider,
                backend: binding.backend(),
                backend_profile: binding.backend_profile().clone(),
                credential_identity: binding.credential_identity().clone(),
                auth_lease: Arc::new(
                    meerkat_llm_core::provider_runtime::DynamicLease::from_authorizer(
                        authorizer.clone(),
                        resolved.metadata().clone(),
                        crate::GITHUB_COPILOT_AUTHORIZER_LABEL,
                    ),
                ),
            };
            let route = runtime
                .resolve_route(&connection, provider, model)
                .expect("PAT route");
            assert!(matches!(route.access, CopilotModelAccess::Available { .. }));
            assert_eq!(
                route.api_base,
                "https://api.business.githubcopilot.com/account"
            );
            let mut headers = Vec::new();
            authorizer
                .authorize(&mut HttpAuthorizationRequest {
                    method: "POST",
                    url: "https://api.business.githubcopilot.com/account/responses",
                    headers: &mut headers,
                })
                .await
                .expect("direct PAT authorization");
            assert!(
                headers
                    .iter()
                    .any(|(name, value)| { name == "authorization" && value == "Bearer test-pat" })
            );
            assert!(headers.iter().any(|(name, value)| {
                name == "copilot-integration-id" && value == "copilot-developer-cli"
            }));
            let requests = transport.requests.lock();
            assert_eq!(requests.len(), 2, "account lookup and models only");
            assert_eq!(
                requests[0].0,
                "https://api.github.com/copilot_internal/user"
            );
            assert_eq!(
                requests[1].0,
                "https://api.business.githubcopilot.com/account/models"
            );
            assert_eq!(requests[0].1["authorization"], "Bearer test-pat");
            assert_eq!(
                requests[1].1["copilot-integration-id"],
                "copilot-developer-cli"
            );
            for (_, headers) in requests.iter() {
                assert!(headers["authorization"].is_sensitive());
                assert!(!format!("{headers:?}").contains("test-pat"));
            }
        }
    }

    #[tokio::test]
    async fn pat_route_bound_authorizer_protects_response_diagnostics_and_status() {
        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport,
        };
        let binding = validated_pat_binding(Provider::OpenAI);
        let resolved = runtime
            .resolve(&binding, &pat_environment("synthetic-pat-canary"))
            .await
            .unwrap();
        let route = resolve_route_for_state(
            &resolved.concrete_authorizer().state,
            Provider::OpenAI,
            "gpt-test",
        )
        .unwrap();
        let authorizer = route.bind_authorizer(resolved.authorizer());
        let mut diagnostic = "echoed synthetic-pat-canary in response".to_string();
        authorizer.redact_response_diagnostic(&mut diagnostic);
        assert_eq!(diagnostic, "echoed [redacted] in response");
        assert!(matches!(
            authorizer.observe_response_status(&HttpAuthorizationResponse {
                method: "POST",
                url: "https://api.business.githubcopilot.com/account/responses",
                status: 403,
            }),
            Err(AuthError::UserReauthRequired)
        ));
        assert!(matches!(
            authorizer.prepare_request().await,
            Err(AuthError::UserReauthRequired)
        ));
        assert!(resolved.model_snapshot().is_none());
    }

    #[tokio::test]
    async fn pat_missing_or_empty_secret_never_falls_back_to_github_tokens() {
        let transport = Arc::new(ScriptedTransport::new([]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        for env in [
            ResolverEnvironment::testing().with_env_lookup(|name| {
                matches!(name, "GITHUB_TOKEN" | "GH_TOKEN").then(|| "other-account".to_string())
            }),
            pat_environment(""),
            pat_environment(" \n "),
        ] {
            assert!(matches!(
                runtime
                    .resolve(&validated_pat_binding(Provider::OpenAI), &env)
                    .await,
                Err(ProviderAuthError::Auth(AuthError::MissingSecret))
            ));
        }
        assert_eq!(transport.request_count(), 0);
    }

    #[tokio::test]
    async fn pat_account_lookup_failure_does_not_echo_secret_or_attempt_exchange() {
        for status in [302, 400, 401, 403, 429, 500] {
            let transport = Arc::new(ScriptedTransport::new([json_response(
                status,
                serde_json::json!({"error": "echoed-test-pat"}),
            )]));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let error = runtime
                .resolve(
                    &validated_pat_binding(Provider::OpenAI),
                    &pat_environment("echoed-test-pat"),
                )
                .await
                .err()
                .expect("failed account lookup must fail closed");
            assert!(!format!("{error:?} {error}").contains("echoed-test-pat"));
            if matches!(status, 401 | 403) {
                assert!(matches!(
                    error,
                    ProviderAuthError::Auth(AuthError::UserReauthRequired)
                ));
            } else {
                assert!(error.to_string().contains("account lookup"));
            }
            assert_eq!(transport.request_count(), 1);
            assert_eq!(
                transport.requests.lock()[0].0,
                "https://api.github.com/copilot_internal/user"
            );
        }
    }

    #[tokio::test]
    async fn pat_rejection_is_terminal_and_foreign_origin_receives_no_secret() {
        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let resolved = runtime
            .resolve(
                &validated_pat_binding(Provider::OpenAI),
                &pat_environment("test-pat"),
            )
            .await
            .expect("resolve PAT");
        let authorizer = resolved.authorizer();
        let mut headers = Vec::new();
        let error = authorizer
            .authorize(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://other.example.test/test-pat",
                headers: &mut headers,
            })
            .await
            .expect_err("foreign origin must receive no credential");
        assert!(!format!("{error:?} {error}").contains("test-pat"));
        assert!(headers.is_empty());
        let receipt = authorizer
            .authorize_with_receipt(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://api.business.githubcopilot.com/account/responses",
                headers: &mut headers,
            })
            .await
            .expect("authorize");
        for status in [401, 403] {
            assert!(matches!(
                authorizer
                    .observe_response_with_receipt(
                        receipt,
                        &HttpAuthorizationResponse {
                            method: "POST",
                            url: "https://api.business.githubcopilot.com/account/responses",
                            status,
                        },
                    )
                    .await,
                Err(AuthError::UserReauthRequired)
            ));
            assert!(matches!(
                authorizer.prepare_request().await,
                Err(AuthError::UserReauthRequired)
            ));
        }
        assert_eq!(transport.request_count(), 2);
        assert!(resolved.model_snapshot().is_none());
    }

    #[tokio::test]
    async fn pat_same_binding_cannot_silently_switch_to_another_credential() {
        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let binding = validated_pat_binding(Provider::OpenAI);
        let first = runtime
            .resolve(&binding, &pat_environment("first-pat"))
            .await
            .expect("first credential");
        let second = runtime
            .resolve(&binding, &pat_environment("second-pat"))
            .await;
        assert!(second.is_err());
        assert!(first.model_snapshot().is_some());
        assert_eq!(transport.request_count(), 2);
    }

    #[tokio::test]
    async fn pat_refuses_managed_source_and_shared_oauth_account() {
        let transport = Arc::new(ScriptedTransport::new([]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let pat = validated_pat_binding(Provider::OpenAI);
        for (identity, source) in [
            (
                pat.credential_identity().clone(),
                CredentialSourceSpec::ManagedStore,
            ),
            (account_identity(), pat.auth_profile().source.clone()),
        ] {
            let mut auth = pat.auth_profile().as_ref().clone();
            auth.source = source;
            let binding = ProviderRuntimeCatalog::validate_binding_with_credential_identity(
                pat.auth_binding_ref(),
                identity,
                pat.backend_profile(),
                &auth,
                &BindingPolicy::default(),
            )
            .expect("catalog-valid auth method");
            assert!(
                runtime
                    .resolve(&binding, &pat_environment("test-pat"))
                    .await
                    .is_err()
            );
        }
        assert_eq!(transport.request_count(), 0);
    }

    #[tokio::test]
    async fn pat_concurrent_discovery_and_model_refresh_use_one_flight() {
        let transport = Arc::new(DelayedTransport {
            inner: ScriptedTransport::new([
                pat_account_response(),
                models_response(),
                models_response(),
            ]),
        });
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let binding = validated_pat_binding(Provider::OpenAI);
        let clock = Arc::new(Mutex::new(Utc::now()));
        let mut env = pat_environment("test-pat");
        env.now = {
            let clock = clock.clone();
            Arc::new(move || *clock.lock())
        };
        let (first, second) = tokio::join!(
            runtime.resolve(&binding, &env),
            runtime.resolve(&binding, &env)
        );
        let first = first
            .expect("first concurrent resolve")
            .concrete_authorizer();
        let second = second
            .expect("second concurrent resolve")
            .concrete_authorizer();
        assert_eq!(transport.inner.request_count(), 2);
        *clock.lock() += Duration::minutes(6);
        let (first_result, second_result) = tokio::join!(first.prime(), second.prime());
        first_result.expect("first refresh");
        second_result.expect("second refresh");
        assert_eq!(transport.inner.request_count(), 3);
        assert!(
            transport
                .inner
                .requests
                .lock()
                .iter()
                .skip(1)
                .all(|(url, headers)| {
                    url == "https://api.business.githubcopilot.com/account/models"
                        && headers["authorization"] == "Bearer test-pat"
                })
        );
    }

    #[tokio::test]
    async fn pat_account_response_requires_a_trusted_api_endpoint() {
        for api in [
            serde_json::Value::Null,
            serde_json::json!(""),
            serde_json::json!("echoed-test-pat"),
            serde_json::json!("http://api.githubcopilot.com"),
            serde_json::json!("https://githubcopilot.com.attacker.test"),
            serde_json::json!("https://notgithubcopilot.com"),
            serde_json::json!("https://127.0.0.1"),
            serde_json::json!("https://api.githubcopilot.com:8443"),
            serde_json::json!("https://echoed-test-pat@api.githubcopilot.com"),
            serde_json::json!("https://user:echoed-test-pat@api.githubcopilot.com"),
            serde_json::json!("https://api.githubcopilot.com?token=echoed-test-pat"),
            serde_json::json!("https://api.githubcopilot.com#echoed-test-pat"),
            serde_json::json!({"unexpected": "echoed-test-pat"}),
        ] {
            let transport = Arc::new(ScriptedTransport::new([json_response(
                200,
                serde_json::json!({"endpoints": {"api": api}}),
            )]));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let error = runtime
                .resolve(
                    &validated_pat_binding(Provider::OpenAI),
                    &pat_environment("echoed-test-pat"),
                )
                .await
                .err()
                .expect("untrusted account endpoint must prevent authentication");
            assert!(error.to_string().contains("account"));
            assert!(!format!("{error:?} {error}").contains("echoed-test-pat"));
            assert_eq!(transport.request_count(), 1);
            assert_eq!(
                transport.requests.lock()[0].0,
                "https://api.github.com/copilot_internal/user"
            );
        }
    }

    #[tokio::test]
    async fn pat_preserves_explicit_integration_id_with_account_routing() {
        let pat = validated_pat_binding(Provider::OpenAI);
        let mut backend = pat.backend_profile().as_ref().clone();
        backend.options = serde_json::json!({"integration_id": "custom-integration"});
        let binding = ProviderRuntimeCatalog::validate_binding(
            pat.auth_binding_ref(),
            &backend,
            pat.auth_profile(),
            &BindingPolicy::default(),
        )
        .expect("explicit integration");
        let transport = Arc::new(ScriptedTransport::new([
            pat_account_response(),
            models_response(),
        ]));
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let resolved = runtime
            .resolve(&binding, &pat_environment("test-pat"))
            .await
            .expect("PAT binding");
        let mut headers = Vec::new();
        resolved
            .authorizer()
            .authorize(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://api.business.githubcopilot.com/account/responses",
                headers: &mut headers,
            })
            .await
            .expect("account-routed inference");
        assert!(headers.iter().any(|(name, value)| {
            name == "copilot-integration-id" && value == "custom-integration"
        }));
        assert_eq!(
            transport.requests.lock()[1].1["copilot-integration-id"],
            "custom-integration"
        );
    }

    #[tokio::test]
    async fn pat_initial_model_failure_keeps_unknown_availability_and_delays_retry() {
        for failure in [
            Some(json_response(
                503,
                serde_json::json!({"error": "echoed-test-pat"}),
            )),
            Some(json_response(
                200,
                serde_json::json!({"data": "echoed-test-pat"}),
            )),
            None,
        ] {
            let transport = Arc::new(ScriptedTransport::new(
                std::iter::once(pat_account_response()).chain(failure),
            ));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let clock = Arc::new(Mutex::new(Utc::now()));
            let mut env = pat_environment("echoed-test-pat");
            env.now = {
                let clock = clock.clone();
                Arc::new(move || *clock.lock())
            };
            let resolved = runtime
                .resolve(&validated_pat_binding(Provider::OpenAI), &env)
                .await
                .expect("trusted account survives optional discovery failure");
            assert_eq!(
                resolved.route_for(Provider::OpenAI, "gpt-test"),
                CopilotModelAccess::Unknown
            );
            assert!(resolved.model_snapshot().is_none());
            let authorizer = resolved.concrete_authorizer();
            let route = resolve_route_for_state(&authorizer.state, Provider::OpenAI, "gpt-test")
                .expect("trusted route without model metadata");
            assert_eq!(route.access, CopilotModelAccess::Unknown);
            assert!(route.available_models.is_none());
            assert_eq!(
                route.api_base,
                "https://api.business.githubcopilot.com/account"
            );
            assert!(!format!("{:?}", authorizer.state.cache.read()).contains("echoed-test-pat"));
            transport.responses.lock().push_back(models_response());
            for _ in 0..3 {
                let mut headers = Vec::new();
                authorizer
                    .authorize(&mut HttpAuthorizationRequest {
                        method: "POST",
                        url: "https://api.business.githubcopilot.com/account/responses",
                        headers: &mut headers,
                    })
                    .await
                    .expect("inference remains authorized");
            }
            assert_eq!(transport.request_count(), 2);
            *clock.lock() += Duration::seconds(29);
            authorizer.prime().await.expect("before retry deadline");
            assert_eq!(transport.request_count(), 2);
            *clock.lock() += Duration::seconds(1);
            authorizer.prime().await.expect("retry succeeds");
            assert_eq!(transport.request_count(), 3);
            assert!(matches!(
                resolved.route_for(Provider::OpenAI, "gpt-test"),
                CopilotModelAccess::Available { .. }
            ));
        }
    }

    #[tokio::test]
    async fn pat_transient_refresh_preserves_snapshot_and_honors_retry_after() {
        for (retry_after, delay) in [(None, 30), (Some(90), 90), (Some(7200), 3600)] {
            let mut failure = json_response(429, serde_json::json!({"error": "echoed-test-pat"}));
            failure.retry_after = retry_after.map(StdDuration::from_secs);
            let transport = Arc::new(ScriptedTransport::new([
                pat_account_response(),
                models_response(),
                failure,
                json_response(200, serde_json::json!({"data": []})),
            ]));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let clock = Arc::new(Mutex::new(Utc::now()));
            let mut env = pat_environment("echoed-test-pat");
            env.now = {
                let clock = clock.clone();
                Arc::new(move || *clock.lock())
            };
            let resolved = runtime
                .resolve(&validated_pat_binding(Provider::OpenAI), &env)
                .await
                .expect("initial discovery");
            let snapshot = resolved.model_snapshot().expect("initial snapshot");
            let authorizer = resolved.concrete_authorizer();
            *clock.lock() += Duration::seconds(301);
            authorizer
                .prime()
                .await
                .expect("transient refresh keeps authentication");
            assert!(Arc::ptr_eq(
                &snapshot,
                &resolved.model_snapshot().expect("retained snapshot")
            ));
            assert!(matches!(
                resolved.route_for(Provider::OpenAI, "gpt-test"),
                CopilotModelAccess::Available { .. }
            ));
            for _ in 0..3 {
                authorizer.prime().await.expect("retry remains scheduled");
            }
            *clock.lock() += Duration::seconds(delay - 1);
            authorizer.prime().await.expect("before retry deadline");
            assert_eq!(transport.request_count(), 3);
            *clock.lock() += Duration::seconds(1);
            authorizer.prime().await.expect("model retry");
            assert_eq!(
                resolved.route_for(Provider::OpenAI, "gpt-test"),
                CopilotModelAccess::Unavailable
            );
            assert_eq!(
                transport.request_count(),
                4,
                "no repeat account lookup or OAuth exchange"
            );
        }
    }

    #[tokio::test]
    async fn pat_model_rejection_never_uses_transient_fallback() {
        for (status, refresh) in [(401, false), (403, true)] {
            let transport = Arc::new(ScriptedTransport::new(
                std::iter::once(pat_account_response())
                    .chain(refresh.then(models_response))
                    .chain(std::iter::once(json_response(
                        status,
                        serde_json::json!({"error": "echoed-test-pat"}),
                    ))),
            ));
            let runtime = CopilotRuntime {
                accounts: Mutex::new(BTreeMap::new()),
                transport: transport.clone(),
            };
            let binding = validated_pat_binding(Provider::OpenAI);
            let clock = Arc::new(Mutex::new(Utc::now()));
            let mut env = pat_environment("echoed-test-pat");
            env.now = {
                let clock = clock.clone();
                Arc::new(move || *clock.lock())
            };
            let result = runtime.resolve(&binding, &env).await;
            if refresh {
                let resolved = result.expect("initial successful discovery");
                *clock.lock() += Duration::seconds(301);
                for _ in 0..2 {
                    assert!(matches!(
                        resolved.authorizer().prepare_request().await,
                        Err(AuthError::UserReauthRequired)
                    ));
                }
                assert!(resolved.model_snapshot().is_none());
                assert_eq!(transport.request_count(), 3);
            } else {
                assert!(matches!(
                    result,
                    Err(ProviderAuthError::Auth(AuthError::UserReauthRequired))
                ));
                assert_eq!(transport.request_count(), 2);
            }
        }
    }

    #[tokio::test]
    async fn pat_concurrent_transient_discovery_shares_fallback_and_retry() {
        let transport = Arc::new(DelayedTransport {
            inner: ScriptedTransport::new([
                pat_account_response(),
                json_response(503, serde_json::json!({"message": "temporary"})),
                models_response(),
            ]),
        });
        let runtime = CopilotRuntime {
            accounts: Mutex::new(BTreeMap::new()),
            transport: transport.clone(),
        };
        let binding = validated_pat_binding(Provider::OpenAI);
        let clock = Arc::new(Mutex::new(Utc::now()));
        let mut env = pat_environment("test-pat");
        env.now = {
            let clock = clock.clone();
            Arc::new(move || *clock.lock())
        };
        let (first, second) = tokio::join!(
            runtime.resolve(&binding, &env),
            runtime.resolve(&binding, &env)
        );
        let first = first.expect("first resolve");
        let second = second.expect("second resolve");
        assert!(first.model_snapshot().is_none());
        assert!(second.model_snapshot().is_none());
        assert_eq!(transport.inner.request_count(), 2);
        *clock.lock() += Duration::seconds(30);
        let first = first.concrete_authorizer();
        let second = second.concrete_authorizer();
        let (a, b) = tokio::join!(first.prime(), second.prime());
        a.expect("first retry");
        b.expect("second retry");
        assert_eq!(transport.inner.request_count(), 3);
        assert!(Arc::ptr_eq(
            &first.model_snapshot().expect("first snapshot"),
            &second.model_snapshot().expect("second snapshot")
        ));
    }

    async fn resolver_environment(
        auth_binding: &AuthBindingRef,
    ) -> (
        ResolverEnvironment,
        meerkat_core::handles::GeneratedAuthLeaseHandle,
    ) {
        resolver_environment_with_token(auth_binding, "github-source-token").await
    }

    async fn resolver_environment_with_token(
        _auth_binding: &AuthBindingRef,
        github_token: &str,
    ) -> (
        ResolverEnvironment,
        meerkat_core::handles::GeneratedAuthLeaseHandle,
    ) {
        let machine = meerkat_runtime::MeerkatMachine::ephemeral();
        let auth_lease = machine.generated_auth_lease_handle();
        let store = Arc::new(meerkat_auth_core::auth_store::EphemeralTokenStore::new());
        let persistence = ProviderAuthPersistence::new(
            store.clone(),
            Arc::new(meerkat_auth_core::auth_store::InMemoryCoordinator::new()),
        );
        let identity = account_identity();
        let tokens = PersistedTokens {
            auth_mode: PersistedAuthMode::GithubCopilotOauth,
            primary_secret: Some(github_token.to_string()),
            refresh_token: None,
            id_token: None,
            expires_at: None,
            last_refresh: Some(Utc::now()),
            scopes: vec!["read:user".to_string()],
            account_id: None,
            metadata: serde_json::Value::Null,
        };
        let transition = meerkat_core::publish_token_lifecycle_acquired_for_identity(
            &auth_lease,
            &identity,
            &tokens,
        )
        .expect("publish source lifecycle");
        let marked = meerkat_core::mark_tokens_lifecycle_published_for_transition(
            &meerkat_core::auth::TokenKey::from_credential_identity(&identity),
            &tokens,
            &transition,
        )
        .expect("mark source token");
        store
            .save(
                &meerkat_core::auth::TokenKey::from_credential_identity(&identity),
                &marked,
            )
            .await
            .expect("save source token");
        (
            ResolverEnvironment::testing()
                .with_provider_auth_persistence(persistence)
                .with_auth_lease_handle(auth_lease.clone()),
            auth_lease,
        )
    }

    #[test]
    fn authorization_scope_requires_exact_origin_and_path_boundary() {
        assert!(url_is_within_api_base(
            "https://api.githubcopilot.com/chat/completions",
            "https://api.githubcopilot.com"
        ));
        assert!(url_is_within_api_base(
            "https://example.test/copilot/responses",
            "https://example.test/copilot"
        ));
        assert!(!url_is_within_api_base(
            "https://api.githubcopilot.com.evil.test/chat/completions",
            "https://api.githubcopilot.com"
        ));
        assert!(!url_is_within_api_base(
            "https://example.test/copilot-evil/responses",
            "https://example.test/copilot"
        ));
        assert!(!url_is_within_api_base(
            "https://example.test:444/copilot/responses",
            "https://example.test/copilot"
        ));
    }

    #[test]
    fn proxy_endpoint_is_derived_without_leaking_token_material() {
        assert_eq!(
            api_base_from_proxy_token("tid=abc;proxy-ep=proxy.individual.githubcopilot.com;ol=1"),
            Some("https://api.individual.githubcopilot.com".to_string())
        );
        assert_eq!(
            api_base_from_proxy_token("proxy-ep=enterprise.example.test"),
            Some("https://enterprise.example.test".to_string())
        );
    }

    #[test]
    fn refresh_hint_never_outlives_expiry_skew() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).expect("valid timestamp");
        let token = CopilotTokenEnvelope {
            token: "derived".to_string(),
            expires_at: 1_800_003_600,
            refresh_in: 3_500,
            endpoints: Default::default(),
            sku: None,
            individual: None,
        };
        assert_eq!(
            token_refresh_at(&token, now).expect("refresh instant"),
            DateTime::from_timestamp(1_800_003_300, 0).expect("valid timestamp")
        );
    }

    #[test]
    fn short_lived_token_refresh_deadline_remains_in_the_future() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).expect("valid timestamp");
        let token = CopilotTokenEnvelope {
            token: "derived".to_string(),
            expires_at: 1_800_000_300,
            refresh_in: 300,
            endpoints: Default::default(),
            sku: None,
            individual: None,
        };

        assert_eq!(
            token_refresh_at(&token, now).expect("refresh instant"),
            DateTime::from_timestamp(1_800_000_240, 0).expect("valid timestamp")
        );
    }

    #[test]
    fn hostile_refresh_interval_fails_without_datetime_overflow() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).expect("valid timestamp");
        let token = CopilotTokenEnvelope {
            token: "derived".to_string(),
            expires_at: 1_800_003_600,
            refresh_in: u64::MAX,
            endpoints: Default::default(),
            sku: None,
            individual: None,
        };

        assert!(token_refresh_at(&token, now).is_err());
    }

    #[test]
    fn hostile_retry_after_is_clamped_before_datetime_conversion() {
        let now = DateTime::from_timestamp(1_800_000_000, 0).expect("valid timestamp");
        let error = CopilotModelDiscoveryError::Http {
            status: 429,
            body: "busy".to_string(),
            retry_after: Some(StdDuration::from_secs(u64::MAX)),
        };

        let delay = error.retry_delay();
        assert_eq!(
            delay,
            StdDuration::from_secs(MODEL_DISCOVERY_MAX_RETRY_SECONDS)
        );
        assert_eq!(
            model_discovery_deadline(now, delay),
            now + Duration::seconds(MODEL_DISCOVERY_MAX_RETRY_SECONDS as i64)
        );
    }

    #[test]
    fn terminal_auth_failures_preserve_reauth_classification() {
        assert_eq!(
            auth_error_from_provider(ProviderAuthError::Auth(AuthError::UserReauthRequired)),
            AuthError::UserReauthRequired
        );
        assert_eq!(
            auth_error_from_refresh(RefreshError::ReauthRequired("invalid grant".to_string())),
            AuthError::UserReauthRequired
        );
    }

    #[test]
    fn routes_with_one_account_share_derived_state() {
        let runtime = CopilotRuntime::new();
        let identity = account_identity();
        let persistence_id = persistence_id();
        let first = runtime
            .account_state(
                Some(persistence_id),
                &identity,
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("first route");
        let second = runtime
            .account_state(
                Some(persistence_id),
                &identity,
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("second route");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn equal_account_keys_from_distinct_persistence_authorities_do_not_share_state() {
        let runtime = CopilotRuntime::new();
        let identity = account_identity();
        let first = runtime
            .account_state(
                Some(persistence_id()),
                &identity,
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("first persistence authority");
        let second = runtime
            .account_state(
                Some(persistence_id()),
                &identity,
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("second persistence authority");

        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn routed_client_requires_reprojection_after_route_change() {
        let runtime = Arc::new(CopilotRuntime::new());
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let persistence_id = env
            .provider_auth_persistence()
            .expect("test persistence")
            .authority_id();
        let state = runtime
            .account_state(
                Some(persistence_id),
                binding.credential_identity(),
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("account state");
        let snapshot = CopilotModelSnapshot::from_models(vec![crate::CopilotModel {
            id: "gpt-test".to_string(),
            vendor: None,
            name: None,
            version: None,
            model_picker_enabled: None,
            capabilities: crate::CopilotModelCapabilities::default(),
            policy: None,
            supported_endpoints: vec![CopilotEndpoint::Responses],
        }])
        .expect("model snapshot");
        *state.cache.write() = Some(CachedCopilotToken {
            token: "derived".to_string(),
            lifetime: CopilotTokenLifetime::ManagedOauth {
                generation: 1,
                source_generation: 1,
                source_expires_at: None,
                refresh_at: Utc::now() + Duration::hours(1),
            },
            api_base: "https://first.example.test".to_string(),
            models: Some(Arc::new(snapshot)),
            model_discovery_refresh_at: None,
        });
        let authorizer = Arc::new(CopilotAuthorizer {
            state: Arc::clone(&state),
            binding: binding.clone(),
            env,
            transport: Arc::new(ScriptedTransport::new([])),
        });
        let connection = meerkat_llm_core::provider_runtime::ResolvedConnection {
            provider: Provider::OpenAI,
            backend: binding.backend(),
            backend_profile: binding.backend_profile().clone(),
            credential_identity: binding.credential_identity().clone(),
            auth_lease: Arc::new(
                meerkat_llm_core::provider_runtime::DynamicLease::from_authorizer(
                    authorizer,
                    AuthMetadata::default(),
                    crate::GITHUB_COPILOT_AUTHORIZER_LABEL,
                ),
            ),
        };
        runtime.accounts.lock().insert(
            CopilotAccountCacheKey {
                persistence_id: Some(persistence_id),
                credential_identity: binding.credential_identity().clone(),
            },
            Arc::downgrade(&state),
        );
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_for_factory = Arc::clone(&observed);
        let dispatches = Arc::new(AtomicU64::new(0));
        let dispatches_for_factory = Arc::clone(&dispatches);
        let factory: CopilotRouteClientFactory = Arc::new(move |route, _| {
            observed_for_factory.lock().push(route.api_base.clone());
            Ok(Arc::new(DispatchCountingClient {
                dispatches: Arc::clone(&dispatches_for_factory),
            }))
        });
        let client = routed_client(
            Arc::clone(&runtime),
            connection,
            Provider::OpenAI,
            "gpt-test".to_string(),
            factory,
        )
        .expect("routed client");

        let first_projection = client
            .project_replay_request(&[])
            .expect("first route projection");
        let first_request = meerkat_llm_core::PreparedLlmRequest::from_projection(
            meerkat_llm_core::LlmRequest::new("gpt-test", Vec::new()),
            first_projection,
        );
        state.cache.write().as_mut().expect("cache").api_base =
            "https://second.example.test".to_string();
        assert!(matches!(
            client.prepared_request_pressure(&first_request),
            Err(meerkat_llm_core::LlmError::AuthorizationRouteChanged { .. })
        ));
        let first_attempt = client
            .stream_prepared(&first_request)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            first_attempt.as_slice(),
            [Err(
                meerkat_llm_core::LlmError::AuthorizationRouteChanged { .. }
            )]
        ));
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            0,
            "a stale request route must be rejected before provider dispatch"
        );
        let second_projection = client
            .project_replay_request(&[])
            .expect("second route projection");
        let second_request = meerkat_llm_core::PreparedLlmRequest::from_projection(
            meerkat_llm_core::LlmRequest::new("gpt-test", Vec::new()),
            second_projection,
        );
        let second_attempt = client
            .stream_prepared(&second_request)
            .collect::<Vec<_>>()
            .await;
        assert!(
            second_attempt.is_empty(),
            "unexpected routed events after reprojection: {second_attempt:?}"
        );
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(
            observed.lock().as_slice(),
            ["https://first.example.test", "https://second.example.test"]
        );
    }

    /// Records what the routed client hands the provider adapter and compiles
    /// schemas with a marker, standing in for the route's adapter client.
    struct StructuredOutputRecordingClient {
        seen: Arc<Mutex<Vec<meerkat_llm_core::LlmRequest>>>,
    }

    #[async_trait]
    impl meerkat_llm_core::LlmClient for StructuredOutputRecordingClient {
        fn project_replay_messages(
            &self,
            messages: &[meerkat_core::Message],
        ) -> Result<Vec<meerkat_core::Message>, meerkat_llm_core::LlmError> {
            Ok(messages.to_vec())
        }

        fn stream<'a>(
            &'a self,
            request: &'a meerkat_llm_core::LlmRequest,
        ) -> meerkat_llm_core::LlmStream<'a> {
            self.seen.lock().push(request.clone());
            Box::pin(futures::stream::empty())
        }

        fn provider(&self) -> Provider {
            Provider::OpenAI
        }

        async fn health_check(&self) -> Result<(), meerkat_llm_core::LlmError> {
            Ok(())
        }

        fn compile_schema(
            &self,
            output_schema: &meerkat_core::OutputSchema,
        ) -> Result<meerkat_core::schema::CompiledSchema, meerkat_core::schema::SchemaError>
        {
            let mut schema = output_schema.schema.as_value().clone();
            schema["x-compiled-by"] = serde_json::json!("route-adapter");
            Ok(meerkat_core::schema::CompiledSchema {
                schema,
                warnings: Vec::new(),
            })
        }
    }

    /// Structured output on every Copilot route: the routed client (and the
    /// account capability gate in front of the adapter) hands the adapter the
    /// exact request the agent loop composed, schema section and native
    /// schema slot included, and the section shows the schema the route's
    /// adapter compiles. The adapters' lowering of those requests is pinned in
    /// the provider crates.
    #[tokio::test]
    async fn routed_client_passes_structured_output_requests_through_unchanged() {
        use meerkat_core::lifecycle::run_primitive::{OpenAiProviderTag, ProviderTag};
        use meerkat_core::structured_output::{
            OUTPUT_SCHEMA_INSTRUCTIONS_OPEN, project_output_schema_instructions,
            render_output_schema_instructions,
        };

        let runtime = Arc::new(CopilotRuntime::new());
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let persistence_id = env
            .provider_auth_persistence()
            .expect("test persistence")
            .authority_id();
        let state = runtime
            .account_state(
                Some(persistence_id),
                binding.credential_identity(),
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("account state");
        let snapshot = CopilotModelSnapshot::from_models(vec![crate::CopilotModel {
            id: "gpt-test".to_string(),
            vendor: None,
            name: None,
            version: None,
            model_picker_enabled: None,
            capabilities: crate::CopilotModelCapabilities {
                supports: crate::CopilotModelSupports {
                    tool_calls: Some(true),
                    streaming: Some(true),
                    structured_outputs: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            },
            policy: None,
            supported_endpoints: vec![CopilotEndpoint::Responses],
        }])
        .expect("model snapshot");
        *state.cache.write() = Some(CachedCopilotToken {
            token: "derived".to_string(),
            lifetime: CopilotTokenLifetime::ManagedOauth {
                generation: 1,
                source_generation: 1,
                source_expires_at: None,
                refresh_at: Utc::now() + Duration::hours(1),
            },
            api_base: "https://copilot.example.test".to_string(),
            models: Some(Arc::new(snapshot)),
            model_discovery_refresh_at: None,
        });
        let authorizer = Arc::new(CopilotAuthorizer {
            state: Arc::clone(&state),
            binding: binding.clone(),
            env,
            transport: Arc::new(ScriptedTransport::new([])),
        });
        let connection = meerkat_llm_core::provider_runtime::ResolvedConnection {
            provider: Provider::OpenAI,
            backend: binding.backend(),
            backend_profile: binding.backend_profile().clone(),
            credential_identity: binding.credential_identity().clone(),
            auth_lease: Arc::new(
                meerkat_llm_core::provider_runtime::DynamicLease::from_authorizer(
                    authorizer,
                    AuthMetadata::default(),
                    crate::GITHUB_COPILOT_AUTHORIZER_LABEL,
                ),
            ),
        };
        runtime.accounts.lock().insert(
            CopilotAccountCacheKey {
                persistence_id: Some(persistence_id),
                credential_identity: binding.credential_identity().clone(),
            },
            Arc::downgrade(&state),
        );
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_for_factory = Arc::clone(&seen);
        let factory: CopilotRouteClientFactory = Arc::new(move |_, _| {
            Ok(Arc::new(StructuredOutputRecordingClient {
                seen: Arc::clone(&seen_for_factory),
            }))
        });
        let client = routed_client(
            Arc::clone(&runtime),
            connection,
            Provider::OpenAI,
            "gpt-test".to_string(),
            factory,
        )
        .expect("routed client");

        let schema = meerkat_core::OutputSchema::new(serde_json::json!({
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"]
        }))
        .expect("valid schema");
        let compiled = client.compile_schema(&schema).expect("compile");
        assert_eq!(
            compiled.schema["x-compiled-by"], "route-adapter",
            "the routed client compiles with the route's adapter"
        );
        let section = render_output_schema_instructions(&compiled.schema);
        let mut messages = vec![
            meerkat_core::Message::System(meerkat_core::SystemMessage::new("You are a reviewer.")),
            meerkat_core::Message::User(meerkat_core::UserMessage::text("review")),
        ];
        project_output_schema_instructions(&mut messages, &section);
        let tool = Arc::new(meerkat_core::ToolDef {
            audience: Default::default(),
            name: "lookup".into(),
            description: "returns a fixed observation".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            provenance: None,
        });
        let mut main_turn =
            meerkat_llm_core::LlmRequest::new("gpt-test", messages.clone()).with_tools(vec![tool]);
        main_turn.provider_params = Some(ProviderTag::OpenAi(OpenAiProviderTag::default()));
        let mut extraction =
            meerkat_llm_core::LlmRequest::new("gpt-test", messages).with_temperature(0.0);
        extraction.provider_params = Some(ProviderTag::OpenAi(OpenAiProviderTag {
            structured_output: Some(schema.clone()),
            ..Default::default()
        }));

        for request in [&main_turn, &extraction] {
            let projection = client
                .project_replay_request(&request.messages)
                .expect("route projection");
            let prepared =
                meerkat_llm_core::PreparedLlmRequest::from_projection(request.clone(), projection);
            let events = client.stream_prepared(&prepared).collect::<Vec<_>>().await;
            assert!(events.is_empty(), "unexpected routed events: {events:?}");
        }
        let seen = seen.lock().clone();
        assert_eq!(seen.len(), 2, "both requests reached the adapter");
        for (sent, received) in [&main_turn, &extraction].into_iter().zip(&seen) {
            assert_eq!(
                serde_json::to_value(received).expect("serialize"),
                serde_json::to_value(sent).expect("serialize"),
                "the routed client hands the adapter the request unchanged"
            );
            assert_eq!(
                serde_json::to_string(&received.messages)
                    .expect("serialize")
                    .matches(OUTPUT_SCHEMA_INSTRUCTIONS_OPEN)
                    .count(),
                1,
                "the section reaches the adapter exactly once"
            );
        }
        let Some(ProviderTag::OpenAi(tag)) = seen[1].provider_params.as_ref() else {
            panic!("extraction keeps its provider tag");
        };
        assert!(
            tag.structured_output.is_some(),
            "the extraction request keeps its native schema slot"
        );
    }

    #[test]
    fn account_capability_gate_rejects_excess_output_budget() {
        let client = capability_gated_client(
            Arc::new(NoopLlmClient),
            Some(crate::CopilotModelCapabilities {
                limits: crate::CopilotModelLimits {
                    max_output_tokens: Some(16),
                    ..Default::default()
                },
                supports: Default::default(),
            }),
        );
        let request = meerkat_llm_core::LlmRequest::new("gpt-test", Vec::new()).with_max_tokens(17);

        assert!(matches!(
            client.request_pressure(&request),
            Err(meerkat_llm_core::LlmError::InvalidRequest { .. })
        ));
        assert_eq!(
            client.output_token_limit(),
            Some(16),
            "the limit the gate enforces is stated to the factory"
        );
        assert_eq!(
            capability_gated_client(Arc::new(NoopLlmClient), None).output_token_limit(),
            None
        );
    }

    #[test]
    fn shared_account_rejects_conflicting_backend_options() {
        let runtime = CopilotRuntime::new();
        let identity = account_identity();
        let persistence_id = persistence_id();
        let _first = runtime
            .account_state(
                Some(persistence_id),
                &identity,
                CopilotBackendConfig::default(),
                CopilotCredential::ManagedOauth,
            )
            .expect("first route");
        let conflicting = CopilotBackendConfig {
            integration_id: "different-client".to_string(),
            ..CopilotBackendConfig::default()
        };
        assert!(
            runtime
                .account_state(
                    Some(persistence_id),
                    &identity,
                    conflicting,
                    CopilotCredential::ManagedOauth
                )
                .is_err()
        );
    }

    #[test]
    fn token_exchange_403_distinguishes_entitlement_from_rate_limit() {
        assert!(token_exchange_requires_account_action(
            403,
            br#"{"error_details":{"notification_id":"no_copilot_access"}}"#
        ));
        assert!(!token_exchange_requires_account_action(
            403,
            br#"{"message":"API rate limit exceeded for user"}"#
        ));
        assert!(token_exchange_requires_account_action(401, b""));
        assert!(!token_exchange_requires_account_action(500, b""));
    }

    #[tokio::test]
    async fn route_scoped_credential_identity_is_rejected() {
        let (auth_binding, account_binding) = validated_binding();
        let route_binding = ProviderRuntimeCatalog::validate_binding(
            &auth_binding,
            account_binding.backend_profile(),
            account_binding.auth_profile(),
            &BindingPolicy::default(),
        )
        .expect("otherwise valid route-scoped binding");
        let (env, _) = resolver_environment(&auth_binding).await;
        let runtime = CopilotRuntime::new();

        let error = runtime
            .resolve(&route_binding, &env)
            .await
            .err()
            .expect("Copilot credentials must be account scoped");

        assert!(error.to_string().contains("account-scoped"));
    }

    #[tokio::test]
    async fn model_discovery_401_remints_before_retrying_discovery() {
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(ScriptedTransport::new([
            token_response("derived-one"),
            json_response(401, serde_json::json!({"message": "expired"})),
            token_response("derived-two"),
            models_response(),
        ]));
        let state = Arc::new(CopilotAccountState {
            credential: CopilotCredential::ManagedOauth,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            config: CopilotBackendConfig::default(),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        let authorizer = CopilotAuthorizer {
            state: Arc::clone(&state),
            binding,
            env,
            transport: transport.clone(),
        };

        authorizer.prime().await.expect("401 remint succeeds");

        assert_eq!(transport.request_count(), 4);
        assert_eq!(
            state
                .cache
                .read()
                .as_ref()
                .map(|cache| cache.token.as_str()),
            Some("derived-two")
        );
    }

    #[tokio::test]
    async fn concurrent_authorization_uses_one_derived_token_flight() {
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(DelayedTransport {
            inner: ScriptedTransport::new([token_response("derived"), models_response()]),
        });
        let state = Arc::new(CopilotAccountState {
            credential: CopilotCredential::ManagedOauth,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            config: CopilotBackendConfig::default(),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        let first = CopilotAuthorizer {
            state: Arc::clone(&state),
            binding: binding.clone(),
            env: env.clone(),
            transport: transport.clone(),
        };
        let second = CopilotAuthorizer {
            state,
            binding,
            env,
            transport: transport.clone(),
        };

        let (first_result, second_result) = tokio::join!(first.prime(), second.prime());

        first_result.expect("first authorization");
        second_result.expect("second authorization");
        assert_eq!(transport.inner.request_count(), 2);
    }

    #[tokio::test]
    async fn concurrent_authorization_shares_one_failed_derived_token_flight() {
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(DelayedTransport {
            inner: ScriptedTransport::new([json_response(
                500,
                serde_json::json!({"message": "temporary failure"}),
            )]),
        });
        let state = Arc::new(CopilotAccountState {
            credential: CopilotCredential::ManagedOauth,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            config: CopilotBackendConfig::default(),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        let first = CopilotAuthorizer {
            state: Arc::clone(&state),
            binding: binding.clone(),
            env: env.clone(),
            transport: transport.clone(),
        };
        let second = CopilotAuthorizer {
            state,
            binding,
            env,
            transport: transport.clone(),
        };

        let (first_result, second_result) = tokio::join!(first.prime(), second.prime());

        assert!(first_result.is_err());
        assert!(second_result.is_err());
        assert_eq!(
            transport.inner.request_count(),
            1,
            "concurrent waiters must observe the leader's typed failure"
        );
    }

    #[tokio::test]
    async fn oauth_waiter_reloads_source_generation_after_shared_refresh() {
        for old_flight_fails in [false, true] {
            let (auth_binding, binding) = validated_binding();
            let (env, auth_lease) = resolver_environment(&auth_binding).await;
            let mut responses = if old_flight_fails {
                vec![json_response(500, serde_json::json!({}))]
            } else {
                vec![token_response("old-derived"), models_response()]
            };
            responses.extend([token_response("current-derived"), models_response()]);
            let transport = Arc::new(GatedTransport {
                inner: ScriptedTransport::new(responses),
                gate: AtomicBool::new(true),
                release: tokio::sync::Semaphore::new(0),
            });
            let authorizer = CopilotAuthorizer {
                state: Arc::new(CopilotAccountState {
                    credential: CopilotCredential::ManagedOauth,
                    pat_rejected: AtomicBool::new(false),
                    publications: Mutex::new(Vec::new()),
                    config: CopilotBackendConfig::default(),
                    cache: RwLock::new(None),
                    refresh: Mutex::new(None),
                    next_generation: AtomicU64::new(1),
                }),
                binding: binding.clone(),
                env: env.clone(),
                transport: transport.clone(),
            };
            let mut leader = Box::pin(authorizer.ensure_token());
            let mut waiter = Box::pin(authorizer.ensure_token());
            assert!(futures::poll!(leader.as_mut()).is_pending());
            assert!(futures::poll!(waiter.as_mut()).is_pending());

            let store = env.provider_auth_persistence().unwrap().token_store();
            let identity = binding.credential_identity();
            let key = meerkat_core::auth::TokenKey::from_credential_identity(identity);
            let mut tokens = store.load(&key).await.unwrap().unwrap();
            tokens.primary_secret = Some("replacement-github-source".to_string());
            let transition = meerkat_core::publish_token_lifecycle_acquired_for_identity(
                &auth_lease,
                identity,
                &tokens,
            )
            .unwrap();
            let tokens = meerkat_core::mark_tokens_lifecycle_published_for_transition(
                &key,
                &tokens,
                &transition,
            )
            .unwrap();
            store.save(&key, &tokens).await.unwrap();

            transport.release.add_permits(1);
            assert_eq!(leader.await.is_err(), old_flight_fails);
            let current = waiter.await.expect("waiter uses the replacement source");
            assert_eq!(current.token, "current-derived");
            let requests = transport.inner.requests.lock();
            let new_mint = if old_flight_fails { 1 } else { 2 };
            assert_eq!(requests.len(), new_mint + 2);
            assert_eq!(
                requests[new_mint].1.get("authorization").unwrap(),
                "token replacement-github-source"
            );
            assert_eq!(
                auth_lease
                    .snapshot(&meerkat_core::handles::LeaseKey::from_credential_identity(
                        identity
                    ))
                    .generation,
                transition.generation()
            );
        }
    }

    #[tokio::test]
    async fn cancelled_refresh_leader_releases_waiters_for_a_new_flight() {
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(CancelFirstTransport {
            inner: ScriptedTransport::new([token_response("derived"), models_response()]),
            calls: AtomicU64::new(0),
            first_started: tokio::sync::Notify::new(),
        });
        let state = Arc::new(CopilotAccountState {
            credential: CopilotCredential::ManagedOauth,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            config: CopilotBackendConfig::default(),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        let first = CopilotAuthorizer {
            state: Arc::clone(&state),
            binding: binding.clone(),
            env: env.clone(),
            transport: transport.clone(),
        };
        let second = CopilotAuthorizer {
            state,
            binding,
            env,
            transport: transport.clone(),
        };

        let first_started = transport.first_started.notified();
        let leader = tokio::spawn(async move { first.prime().await });
        first_started.await;
        leader.abort();
        assert!(
            leader
                .await
                .expect_err("leader must be cancelled")
                .is_cancelled()
        );

        second
            .prime()
            .await
            .expect("a waiter can lead the replacement flight");
        assert_eq!(transport.inner.request_count(), 2);
    }

    #[tokio::test]
    async fn mint_discovery_cache_and_derived_401_preserve_source_lifecycle() {
        let (auth_binding, binding) = validated_binding();
        let (env, auth_lease) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(ScriptedTransport::new([
            token_response("derived-one"),
            models_response(),
            token_response("derived-two"),
            models_response(),
        ]));
        let authorizer = CopilotAuthorizer {
            state: Arc::new(CopilotAccountState {
                credential: CopilotCredential::ManagedOauth,
                pat_rejected: AtomicBool::new(false),
                publications: Mutex::new(Vec::new()),
                config: CopilotBackendConfig::default(),
                cache: RwLock::new(None),
                refresh: Mutex::new(None),
                next_generation: AtomicU64::new(1),
            }),
            binding,
            env,
            transport: transport.clone(),
        };

        authorizer
            .prime()
            .await
            .expect("initial mint and discovery");
        assert_eq!(transport.request_count(), 2);
        assert!(matches!(
            authorizer
                .model_snapshot()
                .and_then(|snapshot| snapshot.model("gpt-test").cloned())
                .and_then(|model| model.route_for(Provider::OpenAI)),
            Some(CopilotEndpoint::Responses)
        ));

        let mut headers = Vec::new();
        let receipt = authorizer
            .authorize_with_receipt(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://copilot.example.test/responses",
                headers: &mut headers,
            })
            .await
            .expect("cached authorization");
        assert_eq!(transport.request_count(), 2);
        assert!(headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("authorization") && value == "Bearer derived-one"
        }));
        assert!(headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("openai-intent") && value == "conversation-panel"
        }));
        assert!(
            headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
        );

        assert_eq!(
            authorizer
                .observe_response_with_receipt(
                    receipt,
                    &HttpAuthorizationResponse {
                        method: "POST",
                        url: "https://copilot.example.test/responses",
                        status: 401,
                    },
                )
                .await
                .expect("401 observation"),
            HttpAuthorizationResponseAction::RetryWithFreshAuthorization
        );
        let mut refreshed_headers = Vec::new();
        let refreshed_receipt = authorizer
            .authorize_with_receipt(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://copilot.example.test/responses",
                headers: &mut refreshed_headers,
            })
            .await
            .expect("remint derived token");
        assert_eq!(transport.request_count(), 4);
        assert!(refreshed_headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("authorization") && value == "Bearer derived-two"
        }));
        assert_ne!(receipt, refreshed_receipt);
        authorizer
            .observe_response_with_receipt(
                receipt,
                &HttpAuthorizationResponse {
                    method: "POST",
                    url: "https://copilot.example.test/responses",
                    status: 401,
                },
            )
            .await
            .expect("delayed stale 401 observation");
        let mut still_fresh_headers = Vec::new();
        authorizer
            .authorize_with_receipt(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://copilot.example.test/responses",
                headers: &mut still_fresh_headers,
            })
            .await
            .expect("newer derived token survives stale 401");
        assert_eq!(transport.request_count(), 4);
        assert_eq!(
            auth_lease
                .snapshot(&meerkat_core::handles::LeaseKey::from_credential_identity(
                    &account_identity(),
                ))
                .phase,
            Some(meerkat_core::handles::AuthLeasePhase::Valid)
        );

        let requests = transport.requests.lock();
        let (_, mint_headers) = &requests[0];
        assert_eq!(
            mint_headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("token github-source-token")
        );
        assert!(mint_headers.contains_key("editor-version"));
        assert!(mint_headers.contains_key("copilot-integration-id"));
        let (_, model_headers) = &requests[1];
        assert_eq!(
            model_headers
                .get("openai-intent")
                .and_then(|value| value.to_str().ok()),
            Some("model-access")
        );
    }

    #[tokio::test]
    async fn failed_model_discovery_retries_without_reminting_derived_token() {
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment(&auth_binding).await;
        let transport = Arc::new(ScriptedTransport::new([
            token_response("derived"),
            json_response(503, serde_json::json!({"message": "temporary"})),
            models_response(),
        ]));
        let state = Arc::new(CopilotAccountState {
            credential: CopilotCredential::ManagedOauth,
            pat_rejected: AtomicBool::new(false),
            publications: Mutex::new(Vec::new()),
            config: CopilotBackendConfig::default(),
            cache: RwLock::new(None),
            refresh: Mutex::new(None),
            next_generation: AtomicU64::new(1),
        });
        let authorizer = CopilotAuthorizer {
            state: Arc::clone(&state),
            binding,
            env,
            transport: transport.clone(),
        };
        authorizer
            .prime()
            .await
            .expect("token mint survives discovery failure");
        assert_eq!(transport.request_count(), 2);
        state
            .cache
            .write()
            .as_mut()
            .expect("derived cache")
            .model_discovery_refresh_at = Some(Utc::now() - Duration::seconds(1));

        let mut headers = Vec::new();
        authorizer
            .authorize(&mut HttpAuthorizationRequest {
                method: "POST",
                url: "https://copilot.example.test/responses",
                headers: &mut headers,
            })
            .await
            .expect("discovery retry succeeds");

        assert_eq!(
            transport.request_count(),
            3,
            "discovery retry must reuse the derived token"
        );
        assert!(
            authorizer
                .model_snapshot()
                .is_some_and(|snapshot| snapshot.model("gpt-test").is_some())
        );
    }

    #[tokio::test]
    #[ignore = "requires a live COPILOT_GITHUB_TOKEN with Copilot entitlement"]
    async fn live_pat_account_and_model_discovery() {
        assert!(
            std::env::var("COPILOT_GITHUB_TOKEN").is_ok_and(|token| !token.trim().is_empty()),
            "COPILOT_GITHUB_TOKEN is required"
        );
        let machine = meerkat_runtime::MeerkatMachine::ephemeral();
        let handle = machine.generated_auth_lease_handle();
        let env = ResolverEnvironment::testing()
            .with_env_lookup(|name| std::env::var(name).ok())
            .with_auth_lease_handle(handle.clone());
        let runtime = CopilotRuntime::new();
        let binding = validated_pat_binding(Provider::OpenAI);
        let result = runtime.resolve(&binding, &env).await;
        assert!(result.is_ok(), "live PAT account resolution failed");
        let resolved = result.unwrap();
        assert!(
            resolved
                .model_snapshot()
                .is_some_and(|snapshot| snapshot.models().next().is_some()),
            "live PAT model discovery returned no usable model snapshot"
        );
        let key = meerkat_core::handles::LeaseKey::from_credential_identity(
            binding.credential_identity(),
        );
        let publication = handle.publish_lease(&key, u64::MAX).unwrap();
        assert!(
            resolved
                .authorizer
                .observe_lease_publication(&publication)
                .is_ok(),
            "live PAT lifecycle publication failed"
        );
        assert!(
            resolved.authorizer.prime().await.is_ok(),
            "live PAT cached credential preparation failed"
        );
        assert_eq!(
            handle.snapshot(&key).phase,
            Some(meerkat_core::handles::AuthLeasePhase::Valid)
        );
    }

    #[tokio::test]
    #[ignore = "requires a live COPILOT_GITHUB_TOKEN with Copilot entitlement"]
    async fn live_token_exchange_and_model_discovery() {
        let github_token =
            std::env::var("COPILOT_GITHUB_TOKEN").expect("COPILOT_GITHUB_TOKEN is required");
        let (auth_binding, binding) = validated_binding();
        let (env, _) = resolver_environment_with_token(&auth_binding, &github_token).await;
        let authorizer = CopilotAuthorizer {
            state: Arc::new(CopilotAccountState {
                credential: CopilotCredential::ManagedOauth,
                pat_rejected: AtomicBool::new(false),
                publications: Mutex::new(Vec::new()),
                config: CopilotBackendConfig::default(),
                cache: RwLock::new(None),
                refresh: Mutex::new(None),
                next_generation: AtomicU64::new(1),
            }),
            binding,
            env,
            transport: Arc::new(ReqwestCopilotHttpTransport::new()),
        };
        authorizer
            .prime()
            .await
            .expect("live Copilot token exchange");
        assert!(
            authorizer
                .model_snapshot()
                .is_some_and(|snapshot| snapshot.models().next().is_some()),
            "live Copilot model discovery returned no usable model snapshot"
        );
    }

    /// Structural pin for credential clients built inside a function (the
    /// refresh and setup paths): the production part of this file builds no
    /// HTTP client other than `credential_http_client`, so a revert to a
    /// default client fails here.
    #[test]
    #[allow(clippy::unwrap_used)]
    fn production_code_builds_only_the_redirect_free_credential_client() {
        let source = include_str!("runtime.rs");
        let production = &source[..source.find("\nmod tests {").unwrap()];
        for forbidden in [
            concat!("Client", "::new()"),
            concat!("Client", "::default()"),
            concat!("Client", "::builder()"),
            concat!("ReqwestCopilotHttpTransport", "::default()"),
        ] {
            assert!(!production.contains(forbidden), "{forbidden}");
        }
        assert!(production.contains("credential_http_client()"));
    }
    // The N1 common trait is the predecessor of these regression tests. The
    // actual Copilot runtime and authorizer remain the credential/route owners;
    // ScriptedTransport replaces only the external token/model HTTP transport.
    mod controller_maintenance {
        use super::*;

        struct Fixture {
            client: Arc<dyn meerkat_llm_core::LlmClient>,
            authorizer: Arc<CopilotAuthorizer>,
            auth_lease: meerkat_core::handles::GeneratedAuthLeaseHandle,
            transport: Arc<ScriptedTransport>,
            factory_routes: Arc<Mutex<Vec<String>>>,
            dispatches: Arc<AtomicU64>,
        }

        impl Fixture {
            async fn new(capability_wrapper: bool) -> Self {
                let (auth_binding, binding) = validated_binding();
                let (env, auth_lease) = resolver_environment(&auth_binding).await;
                let transport = Arc::new(ScriptedTransport::new([
                    token_response("maintenance-derived-one"),
                    models_response(),
                    token_response("maintenance-derived-two"),
                    models_response(),
                ]));
                let runtime = Arc::new(CopilotRuntime {
                    accounts: Mutex::new(BTreeMap::new()),
                    transport: transport.clone(),
                });
                let resolved = runtime.resolve(&binding, &env).await.unwrap();
                let authorizer = resolved.concrete_authorizer();
                let connection = meerkat_llm_core::provider_runtime::ResolvedConnection {
                    provider: Provider::OpenAI,
                    backend: binding.backend(),
                    backend_profile: binding.backend_profile().clone(),
                    credential_identity: binding.credential_identity().clone(),
                    auth_lease: Arc::new(
                        meerkat_llm_core::provider_runtime::DynamicLease::from_authorizer(
                            resolved.authorizer(),
                            resolved.metadata().clone(),
                            crate::GITHUB_COPILOT_AUTHORIZER_LABEL,
                        ),
                    ),
                };
                let factory_routes = Arc::new(Mutex::new(Vec::new()));
                let routes = Arc::clone(&factory_routes);
                let dispatches = Arc::new(AtomicU64::new(0));
                let child_dispatches = Arc::clone(&dispatches);
                let factory: CopilotRouteClientFactory = Arc::new(move |route, _| {
                    routes.lock().push(route.api_base.clone());
                    Ok(Arc::new(DispatchCountingClient {
                        dispatches: Arc::clone(&child_dispatches),
                    }))
                });
                let routed = routed_client(
                    runtime,
                    connection,
                    Provider::OpenAI,
                    "gpt-test".to_string(),
                    factory,
                )
                .unwrap();
                // Exercise the capability wrapper against a real maintained
                // inner route, independently of its position in a provider stack.
                let client = if capability_wrapper {
                    capability_gated_client(
                        routed,
                        Some(crate::CopilotModelCapabilities::default()),
                    )
                } else {
                    routed
                };
                Self {
                    client,
                    authorizer,
                    auth_lease,
                    transport,
                    factory_routes,
                    dispatches,
                }
            }

            fn source_snapshot(&self) -> meerkat_core::handles::AuthLeaseSnapshot {
                self.auth_lease.snapshot(
                    &meerkat_core::handles::LeaseKey::from_credential_identity(&account_identity()),
                )
            }

            fn assert_route_untouched(&self) {
                assert_eq!(self.client.provider(), Provider::OpenAI);
                assert_eq!(
                    self.factory_routes.lock().as_slice(),
                    ["https://copilot.example.test"],
                    "request-free maintenance must not reconstruct/reselect a route"
                );
                assert_eq!(self.dispatches.load(Ordering::SeqCst), 0);
                assert_eq!(
                    self.authorizer.api_base().as_deref(),
                    Some("https://copilot.example.test")
                );
                assert!(
                    self.authorizer
                        .model_snapshot()
                        .unwrap()
                        .model("gpt-test")
                        .unwrap()
                        .route_for(Provider::OpenAI)
                        == Some(CopilotEndpoint::Responses)
                );
            }
        }

        async fn remint(capability_wrapper: bool) {
            let fixture = Fixture::new(capability_wrapper).await;
            let source = fixture.source_snapshot();
            assert_eq!(
                source.phase,
                Some(meerkat_core::handles::AuthLeasePhase::Valid)
            );
            assert_eq!(fixture.transport.request_count(), 2);
            fixture
                .client
                .prepare_controller_credential()
                .await
                .unwrap();
            assert_eq!(
                fixture.transport.request_count(),
                2,
                "fresh maintenance is cached"
            );

            let mut headers = Vec::new();
            let first = fixture
                .authorizer
                .authorize_with_receipt(&mut HttpAuthorizationRequest {
                    method: "POST",
                    url: "https://copilot.example.test/responses",
                    headers: &mut headers,
                })
                .await
                .unwrap();
            assert!(
                headers
                    .iter()
                    .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                        && value == "Bearer maintenance-derived-one")
            );
            assert_eq!(
                fixture
                    .authorizer
                    .observe_response_with_receipt(
                        first,
                        &HttpAuthorizationResponse {
                            method: "POST",
                            url: "https://copilot.example.test/responses",
                            status: 401,
                        },
                    )
                    .await
                    .unwrap(),
                HttpAuthorizationResponseAction::RetryWithFreshAuthorization
            );
            assert!(
                fixture.authorizer.api_base().is_none(),
                "actual receipt observation invalidates the derived cache"
            );

            fixture
                .client
                .prepare_controller_credential()
                .await
                .unwrap();
            assert_eq!(
                fixture.transport.request_count(),
                4,
                "the real authorizer must remint and discover before any model request"
            );
            let mut headers = Vec::new();
            let second = fixture
                .authorizer
                .authorize_with_receipt(&mut HttpAuthorizationRequest {
                    method: "POST",
                    url: "https://copilot.example.test/responses",
                    headers: &mut headers,
                })
                .await
                .unwrap();
            assert_ne!(first, second);
            assert!(
                headers
                    .iter()
                    .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                        && value == "Bearer maintenance-derived-two")
            );
            assert_eq!(
                fixture.transport.request_count(),
                4,
                "the positive readback cannot rescue missing preparation"
            );
            assert_eq!(
                fixture.source_snapshot(),
                source,
                "derived Copilot maintenance cannot invent a source credential generation"
            );
            fixture.assert_route_untouched();
            let requests = fixture.transport.requests.lock();
            for index in [0, 2] {
                assert_eq!(
                    requests[index].1.get("authorization").unwrap(),
                    "token github-source-token"
                );
            }
            for index in [1, 3] {
                assert_eq!(
                    requests[index].1.get("openai-intent").unwrap(),
                    "model-access"
                );
            }
        }

        async fn reauth(capability_wrapper: bool) {
            let fixture = Fixture::new(capability_wrapper).await;
            let key =
                meerkat_core::handles::LeaseKey::from_credential_identity(&account_identity());
            fixture.auth_lease.mark_reauth_required(&key).unwrap();
            let refused = fixture.source_snapshot();
            assert_eq!(
                refused.phase,
                Some(meerkat_core::handles::AuthLeasePhase::ReauthRequired)
            );
            let result = fixture.client.prepare_controller_credential().await;
            assert!(
                matches!(result, Err(AuthError::UserReauthRequired)),
                "{result:?}"
            );
            assert_eq!(fixture.source_snapshot(), refused);
            assert_eq!(
                fixture.transport.request_count(),
                2,
                "the generated reauth decision must arrive without extra token/model transport"
            );
            fixture.assert_route_untouched();
        }

        #[tokio::test]
        async fn routed_maintenance_uses_retained_real_authorizer() {
            remint(false).await;
        }

        #[tokio::test]
        async fn capability_maintenance_forwards_to_real_inner_route() {
            remint(true).await;
        }

        #[tokio::test]
        async fn routed_maintenance_preserves_typed_reauth() {
            reauth(false).await;
        }

        #[tokio::test]
        async fn capability_maintenance_preserves_typed_reauth() {
            reauth(true).await;
        }
    }
}

#[cfg(test)]
#[test]
fn ce_stale_refresh_remains_stale_credential() {
    assert!(matches!(
        auth_error_from_refresh(RefreshError::StalePreparation),
        AuthError::StaleCredential
    ));
}
