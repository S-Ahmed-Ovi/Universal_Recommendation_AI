"""Backward-compatible auth exports."""

from backend.api.gateway.auth import (
    SCOPES,
    create_access_token,
    hash_api_key,
    require_auth,
    resolve_api_key,
    token_validator,
    validate_credentials,
    verify_api_key,
    verify_token,
)

__all__ = [
    "SCOPES",
    "create_access_token",
    "hash_api_key",
    "require_auth",
    "resolve_api_key",
    "token_validator",
    "validate_credentials",
    "verify_api_key",
    "verify_token",
]
