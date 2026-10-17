"""Bearer-token authentication that yields an authenticated user id.

WHY static tokens rather than OAuth/OIDC
  This module exists to prove one property the API must not fake: every
  session is bound to an *authenticated* user id, not to a value the client
  chooses. A token-to-user map is the smallest mechanism that establishes
  that, and it runs offline. A production deployment would replace
  ``TokenAuthenticator`` with a JWT/OIDC verifier and keep the rest of the
  API unchanged; that replacement is not implemented and is not claimed.

FAIL-CLOSED RULES
  * An empty token map is a configuration error, not a mode that allows
    anonymous access. ``TokenAuthenticator`` refuses to be constructed with
    no tokens.
  * A missing, malformed or unknown credential is rejected. There is no
    default user.
  * Tokens are compared with ``hmac.compare_digest`` so a wrong token does
    not leak its correct prefix through timing.

The token format for the environment variable is ``token:user`` pairs
separated by commas, e.g. ``"t-alice:alice,t-bob:bob"``. One user may hold
several tokens (key rotation); two users cannot share a token.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping

_ENV_VAR = "GEM_API_TOKENS"


class AuthError(Exception):
    """The request carried no usable credential."""


def parse_tokens(raw: str) -> dict[str, str]:
    """Parse ``"token:user,token:user"`` into a token-to-user mapping.

    Raises:
        ValueError: on an empty value, a malformed pair, a duplicate token,
            or an empty token or user.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{_ENV_VAR} must contain at least one token:user pair")
    tokens: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        token, separator, user = entry.partition(":")
        token, user = token.strip(), user.strip()
        if not separator or not token or not user:
            raise ValueError(
                f"invalid token entry {entry!r}; expected 'token:user'"
            )
        if token in tokens:
            raise ValueError("duplicate token in GEM_API_TOKENS")
        tokens[token] = user
    if not tokens:
        raise ValueError(f"{_ENV_VAR} must contain at least one token:user pair")
    return tokens


class TokenAuthenticator:
    """Map a bearer token to a user id, rejecting anything else.

    Example:
        >>> auth = TokenAuthenticator({"t-alice": "alice"})
        >>> auth.user_for("Bearer t-alice")
        'alice'
        >>> auth.user_for(None)
        Traceback (most recent call last):
        ...
        gem_agentic_ai.api.auth.AuthError: missing bearer token
    """

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if not tokens:
            raise ValueError("at least one API token is required; refusing anonymous access")
        for token, user in tokens.items():
            if not isinstance(token, str) or not token:
                raise ValueError("API tokens must be non-empty strings")
            if not isinstance(user, str) or not user.strip():
                raise ValueError("API tokens must map to a non-empty user id")
        self._tokens = {token: user.strip() for token, user in tokens.items()}

    def user_for(self, authorization: str | None) -> str:
        """Return the authenticated user id for an ``Authorization`` header.

        Raises:
            AuthError: when the header is absent or does not name a known
                bearer token.
        """
        if not authorization or not authorization.strip():
            raise AuthError("missing bearer token")
        scheme, _, credential = authorization.strip().partition(" ")
        if scheme.lower() != "bearer" or not credential.strip():
            raise AuthError("expected 'Authorization: Bearer <token>'")
        credential = credential.strip()
        # Compare against every token so the number of comparisons does not
        # depend on which prefix the caller guessed.
        matched: str | None = None
        for token, user in self._tokens.items():
            if hmac.compare_digest(credential, token):
                matched = user
        if matched is None:
            raise AuthError("unknown bearer token")
        return matched
