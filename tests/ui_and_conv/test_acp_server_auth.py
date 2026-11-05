import time
from unittest.mock import patch

import acp
import pytest

from kimi_cli.acp.server import ACPServer
from kimi_cli.auth.oauth import OAuthToken
from kimi_cli.config import Config, load_config_from_string


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


def _managed_config() -> Config:
    """Config whose default model resolves to an OAuth-backed managed provider."""
    return load_config_from_string(
        """
default_model = "kimi-code"

[models.kimi-code]
provider = "managed:kimi-code"
model = "kimi-code"
max_context_size = 262144

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.moonshot.ai/v1"
api_key = ""

[providers."managed:kimi-code".oauth]
storage = "file"
key = "credentials/kimi-code.json"
"""
    )


def test_check_auth_raises_when_no_token(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when no token exists."""
    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32000  # AUTH_REQUIRED error code


def test_check_auth_raises_when_token_has_no_access_token(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when token has no access_token."""
    token = _make_token(access_token="")

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=token),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32000


def test_check_auth_passes_when_valid_token(server: ACPServer) -> None:
    """Test that _check_auth passes when a valid token exists."""
    token = _make_token()

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=token),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        # Should not raise
        server._check_auth()


def test_check_auth_raises_when_token_expired_without_refresh(server: ACPServer) -> None:
    """Test that _check_auth raises AUTH_REQUIRED when token expired and no refresh token."""
    token = _make_token(
        expires_at=time.time() - 100,  # expired 100 seconds ago
        refresh_token="",
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=token),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32000


def test_check_auth_passes_when_token_expired_but_has_refresh(server: ACPServer) -> None:
    """Test that _check_auth passes when token expired but refresh token is available.

    The background refresh mechanism will handle renewal.
    """
    token = _make_token(
        expires_at=time.time() - 100,  # expired
        refresh_token="refresh_123",
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=token),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        # Should not raise — background refresh will handle it
        server._check_auth()


def test_check_auth_passes_when_expires_at_is_zero(server: ACPServer) -> None:
    """Test that expires_at=0 (no expiry info from server) is treated as valid.

    OAuthToken.from_dict() sets expires_at=0.0 when the response has no
    expires_at field. The code uses ``token.expires_at and ...`` so 0.0
    (falsy) skips the expiry check entirely.
    """
    token = _make_token(expires_at=0.0)

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=token),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        # Should not raise
        server._check_auth()


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
# Gate scoping: static-credential providers must not require login (#2633)
# ---------------------------------------------------------------------------


def test_check_auth_skips_gate_for_static_key_custom_provider(server: ACPServer) -> None:
    """Custom provider with a static key passes with no token (I2/D1)."""
    config = load_config_from_string(
        """
default_model = "custom"

[models.custom]
provider = "custom-provider"
model = "gpt-4o"
max_context_size = 128000

[providers.custom-provider]
type = "openai_legacy"
base_url = "https://api.example.com/v1"
api_key = "sk-static-test-key"
"""
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=config),
    ):
        # Should not raise
        server._check_auth()


@pytest.mark.parametrize(
    ("provider_type", "env_var"),
    [("kimi", "KIMI_API_KEY"), ("openai_legacy", "OPENAI_API_KEY")],
)
def test_check_auth_honors_env_supplied_key(
    server: ACPServer, monkeypatch: pytest.MonkeyPatch, provider_type: str, env_var: str
) -> None:
    """Empty config key + env key passes with no token (D1)."""
    monkeypatch.setenv(env_var, "sk-env-test-key")
    config = load_config_from_string(
        f"""
default_model = "custom"

[models.custom]
provider = "custom-provider"
model = "some-model"
max_context_size = 128000

[providers.custom-provider]
type = "{provider_type}"
base_url = "https://api.example.com/v1"
api_key = ""
"""
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=config),
    ):
        # Should not raise
        server._check_auth()


def test_check_auth_still_gates_managed_provider_without_token(server: ACPServer) -> None:
    """Managed provider without a token still raises -32000 with authMethods (I1/I3)."""
    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=_managed_config()),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32000  # AUTH_REQUIRED error code
        data = exc_info.value.data
        assert isinstance(data, dict)
        assert "authMethods" in data


def test_check_auth_rejects_empty_key_custom_provider(
    server: ACPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No oauth, empty key, no env → invalid_params, NOT -32000 (I2 edge/D1)."""
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_config_from_string(
        """
default_model = "custom"

[models.custom]
provider = "custom-provider"
model = "gpt-4o"
max_context_size = 128000

[providers.custom-provider]
type = "openai_legacy"
base_url = "https://api.example.com/v1"
api_key = ""
"""
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=config),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        # acp invalid_params (verified against the installed acp SDK); login can't fix this.
        assert exc_info.value.code == -32602
        assert exc_info.value.code != -32000
        data = exc_info.value.data
        assert isinstance(data, dict)
        assert "custom-provider" in data["api_key"]


def test_check_auth_dangling_default_model_is_client_visible(server: ACPServer) -> None:
    """Unknown default_model → invalid_params, never an unhandled KeyError (I9)."""
    # Config validates default_model at load time, so dangle it post-load.
    config = _managed_config()
    config.default_model = "no-such-model"

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=config),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32602  # acp invalid_params


def test_check_auth_gates_managed_key_without_oauth_ref(server: ACPServer) -> None:
    """managed: key with no oauth ref and no token → -32000 (fail closed)."""
    config = load_config_from_string(
        """
default_model = "kimi-code"

[models.kimi-code]
provider = "managed:kimi-code"
model = "kimi-code"
max_context_size = 262144

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.moonshot.ai/v1"
api_key = ""
"""
    )

    with (
        patch("kimi_cli.acp.server.load_tokens", return_value=None),
        patch("kimi_cli.acp.server.load_config", return_value=config),
    ):
        with pytest.raises(acp.RequestError) as exc_info:
            server._check_auth()

        assert exc_info.value.code == -32000  # AUTH_REQUIRED error code
