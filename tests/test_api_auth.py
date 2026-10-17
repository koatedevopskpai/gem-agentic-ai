"""Authentication: every request maps to a user or is refused.

These tests pin the fail-closed rules: no tokens means no server, no header
means no user, and an unknown token is rejected without a default identity.
"""

import pytest

from gem_agentic_ai.api.auth import AuthError, TokenAuthenticator, parse_tokens


def test_tokens_parse_into_a_mapping() -> None:
    assert parse_tokens("t-alice:alice,t-bob:bob") == {
        "t-alice": "alice",
        "t-bob": "bob",
    }


def test_a_user_may_hold_more_than_one_token() -> None:
    assert parse_tokens("old:alice,new:alice") == {"old": "alice", "new": "alice"}


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "alice", "t-alice:", ":alice", "t-alice:alice,t-alice:bob"],
)
def test_invalid_token_lists_are_refused(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_tokens(raw)


def test_an_authenticator_refuses_to_exist_without_tokens() -> None:
    with pytest.raises(ValueError):
        TokenAuthenticator({})


def test_a_bearer_token_resolves_to_its_user() -> None:
    auth = TokenAuthenticator({"t-alice": "alice"})
    assert auth.user_for("Bearer t-alice") == "alice"


def test_the_scheme_is_case_insensitive() -> None:
    auth = TokenAuthenticator({"t-alice": "alice"})
    assert auth.user_for("bearer t-alice") == "alice"


@pytest.mark.parametrize(
    "header",
    [None, "", "t-alice", "Basic t-alice", "Bearer", "Bearer wrong-token", "Bearer t-ali"],
)
def test_missing_or_wrong_credentials_are_refused(header: str | None) -> None:
    auth = TokenAuthenticator({"t-alice": "alice"})
    with pytest.raises(AuthError):
        auth.user_for(header)


def test_whitespace_around_the_header_is_tolerated() -> None:
    auth = TokenAuthenticator({"t-alice": "alice"})
    assert auth.user_for("  Bearer   t-alice  ") == "alice"
