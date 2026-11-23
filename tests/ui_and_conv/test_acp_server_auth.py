import time
from unittest.mock import patch

import acp
import pytest
from pydantic import SecretStr

from kimi_cli.acp.server import ACPServer
from kimi_cli.auth.oauth import OAuthToken
from kimi_cli.config import Config, LLMModel, LLMProvider, OAuthRef


@pytest.fixture
def server() -> ACPServer:
    """Create an ACPServer instance with mocked auth methods."""
    s = ACPServer()
    s._auth_methods = [
        acp.schema.AuthMethod(
            id="login",
            name="Test Login",
            description="Test description",
            field_meta={
                "terminal-auth": {
                    "type": "terminal",
                    "args": ["kimi", "login"],
                    "env": {},
                }
            },
        )
    ]
    return s


def _make_token(
    access_token: str = "valid_token_123",
    refresh_token: str = "refresh_123",
    expires_at: float | None = None,
) -> OAuthToken:
    if expires_at is None:
        expires_at = time.time() + 3600  # 1 hour from now
    return OAuthToken(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        scope="",
        token_type="Bearer",
    )


def _custom_config(api_key: str = "sk-test-static-key") -> Config:
    """Config whose default model runs on a custom provider with a static key."""
    return Config(
        default_model="zhipu/glm-5.3",
        models={
            "zhipu/glm-5.3": LLMModel(provider="zhipu", model="glm-5.3", max_context_size=128000)
        },
        providers={
            "zhipu": LLMProvider(
                type="openai_legacy",
                base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                api_key=SecretStr(api_key),
            )
        },
    )


def _managed_config() -> Config:
    """Config whose default model runs on the OAuth-backed managed provider."""
    return Config(
        default_model="kimi-code/k2",
        models={
            "kimi-code/k2": LLMModel(
                provider="managed:kimi-code", model="k2", max_context_size=262144
            )
        },
        providers={
            "managed:kimi-code": LLMProvider(
                type="kimi",
                base_url="https://api.kimi.com/coding",
                api_key=SecretStr(""),
                oauth=OAuthRef(storage="file", key="oauth/kimi-code"),
            )
        },
    )


_PROVIDER_ENV_VARS = (
    "KIMI_BASE_URL",
    "KIMI_API_KEY",
    "KIMI_MODEL_NAME",
    "KIMI_MODEL_MAX_CONTEXT_SIZE",
    "KIMI_MODEL_CAPABILITIES",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


@pytest.fixture
def clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient provider env overrides so key assertions are deterministic."""
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_check_auth_raises_when_no_token(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when no token exists."""
    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=_managed_config())

        assert exc_info.value.code == -32000  # AUTH_REQUIRED error code


def test_check_auth_raises_when_token_has_no_access_token(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when token has no access_token."""
    token = _make_token(access_token="")

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=_managed_config())

        assert exc_info.value.code == -32000


def test_check_auth_passes_when_valid_token(server: ACPServer) -> None:
    """Test that _check_auth passes when a valid token exists."""
    token = _make_token()

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        # Should not raise
        server._check_auth(config=_managed_config())


def test_check_auth_raises_when_token_expired_without_refresh(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when token expired and no refresh token."""
    token = _make_token(
        expires_at=time.time() - 100,  # expired 100 seconds ago
        refresh_token="",
    )

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=_managed_config())

        assert exc_info.value.code == -32000


def test_check_auth_passes_when_token_expired_but_has_refresh(server: ACPServer) -> None:
    """Test that _check_auth passes when token expired but refresh token is available.

    The background refresh mechanism will handle renewal.
    """
    token = _make_token(
        expires_at=time.time() - 100,  # expired
        refresh_token="refresh_123",
    )

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        # Should not raise — background refresh will handle it
        server._check_auth(config=_managed_config())


def test_check_auth_passes_when_expires_at_is_zero(server: ACPServer) -> None:
    """Test that expires_at=0 (no expiry info from server) is treated as valid.

    OAuthToken.from_dict() sets expires_at=0.0 when the response has no
    expires_at field. The code uses ``token.expires_at and ...`` so 0.0
    (falsy) skips the expiry check entirely.
    """
    token = _make_token(expires_at=0.0)

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        # Should not raise
        server._check_auth(config=_managed_config())


# ---------------------------------------------------------------------------
# authenticate() must agree with _check_auth()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_rejects_expired_token_without_refresh(server: ACPServer) -> None:
    """authenticate('login') must reject an expired token with no refresh token,
    the same way _check_auth() does. Otherwise the client gets a false-success
    from authenticate, then immediately fails on new_session.
    """
    token = _make_token(
        expires_at=time.time() - 100,
        refresh_token="",
    )

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        with pytest.raises(acp.RequestError) as exc_info:
            await server.authenticate(method_id="login")

        assert exc_info.value.code == -32000


@pytest.mark.asyncio
async def test_authenticate_accepts_valid_token(server: ACPServer) -> None:
    """authenticate('login') should succeed for a valid, non-expired token."""
    token = _make_token()

    with patch("kimi_cli.acp.server.load_tokens", return_value=token):
        result = await server.authenticate(method_id="login")

    assert result is not None


# ---------------------------------------------------------------------------
# Gate scoping: custom providers with static credentials skip the Kimi login
# ---------------------------------------------------------------------------


def test_check_auth_skips_gate_for_static_key_custom_provider(server: ACPServer) -> None:
    """A static-key custom provider needs no Kimi token (#2633)."""
    with patch("kimi_cli.acp.server.load_tokens", return_value=None) as mock_load:
        # Should not raise
        server._check_auth(config=_custom_config())

    mock_load.assert_not_called()


def test_check_auth_fail_closed_for_managed_provider(server: ACPServer) -> None:
    """An OAuth-backed provider without a token still raises AUTH_REQUIRED."""
    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=_managed_config())

        assert exc_info.value.code == -32000


def test_check_auth_fail_closed_for_oauth_ref_on_custom_key(server: ACPServer) -> None:
    """A non-managed key carrying an oauth ref is still gated."""
    config = _custom_config()
    config.providers["zhipu"].oauth = OAuthRef(storage="file", key="oauth/kimi-code")

    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=config)

        assert exc_info.value.code == -32000


def test_check_auth_fail_closed_for_managed_key_without_oauth(server: ACPServer) -> None:
    """A managed: key without an oauth ref is still gated."""
    config = _managed_config()
    config.providers["managed:kimi-code"].oauth = None

    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=config)

        assert exc_info.value.code == -32000


def test_check_auth_fail_closed_for_empty_key_custom_provider(
    server: ACPServer, clean_provider_env: None
) -> None:
    """A custom provider with no key anywhere keeps today's AUTH_REQUIRED."""
    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=_custom_config(api_key=""))

        assert exc_info.value.code == -32000


def test_check_auth_fail_closed_for_unconfigured_session(server: ACPServer) -> None:
    """A fresh session (no default model) must still reach the login bootstrap."""
    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=Config())

        assert exc_info.value.code == -32000


def test_check_auth_fail_closed_for_dangling_default_model(server: ACPServer) -> None:
    """A default_model naming no known model keeps today's AUTH_REQUIRED."""
    config = Config.model_construct(default_model="nope/missing", models={}, providers={})

    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth(config=config)

        assert exc_info.value.code == -32000


def test_check_auth_accepts_env_supplied_key(
    server: ACPServer, clean_provider_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env-supplied keys count: LLM creation applies the same overrides."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-supplied")

    with patch("kimi_cli.acp.server.load_tokens", return_value=None) as mock_load:
        # Should not raise
        server._check_auth(config=_custom_config(api_key=""))

    mock_load.assert_not_called()


def test_check_auth_does_not_mutate_passed_config(
    server: ACPServer, clean_provider_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env overrides are consulted on copies; the caller's config is untouched."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-supplied")
    config = _custom_config(api_key="")

    with patch("kimi_cli.acp.server.load_tokens", return_value=None):
        server._check_auth(config=config)

    assert config.providers["zhipu"].api_key.get_secret_value() == ""


def test_check_auth_loads_default_config_when_none_given(server: ACPServer) -> None:
    """Without an explicit config the gate consults the default config file."""
    with (
        patch("kimi_cli.acp.server.load_config", return_value=_custom_config()) as mock_config,
        patch("kimi_cli.acp.server.load_tokens", return_value=None) as mock_load,
    ):
        # Should not raise
        server._check_auth()

    mock_config.assert_called_once_with()
    mock_load.assert_not_called()
