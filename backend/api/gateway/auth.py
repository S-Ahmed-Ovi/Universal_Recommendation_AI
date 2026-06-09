"""
Authentication helpers for the gateway layer.

Supports:
- Bearer tokens signed with HMAC-SHA256
- API key auth resolved through TenantManager
- Backward-compatible auth usage as:
  - `Depends(require_auth)`
  - `Depends(require_auth(scopes=[...]))`
  - `@require_auth(scope="...")`
"""

import base64
import hashlib
import hmac
import inspect
import json
import time
from functools import wraps
from typing import Any, Callable, Optional, Sequence

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config.settings import (
    AUTH_PASSWORD,
    AUTH_SECRET_KEY,
    AUTH_TOKEN_TTL_MINUTES,
    AUTH_USERNAME,
)
from backend.core.tenants.manager import TenantConfig, TenantManager
from backend.utils.logger import get_logger

log = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)
_TOKEN_REVOCATION_LIST: set[str] = set()

SCOPES: dict[str, str] = {
    "recommend:read": "Get recommendations, similar items, and explanations",
    "training:write": "Submit training jobs (file or SQL)",
    "models:read": "List trained models",
    "models:write": "Promote models to production",
    "explain:read": "Read recommendation explanations",
    "batch:read": "Read batch recommendation responses",
    "similar:read": "Read similar-item responses",
    "jobs:read": "Read async job status",
    "tenant:manage": "Create and manage the authenticated user's tenant",
    "admin:read": "Read admin dashboard data",
    "admin:write": "Full admin control",
}

_SCOPE_HIERARCHY: dict[str, list[str]] = {
    "admin:write": list(SCOPES.keys()),
    "admin:read": ["recommend:read", "models:read"],
}


def hash_api_key(raw_key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    """Verify a raw API key against its stored SHA-256 hash."""
    if not raw_key or not stored_hash:
        return False
    computed_hash = hash_api_key(raw_key)
    return hmac.compare_digest(computed_hash, stored_hash)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str, secret: str) -> str:
    return _b64url_encode(
        hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    )


def _encode_token(payload: dict[str, Any], secret: str) -> str:
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    signature_b64 = _sign(payload_b64, secret)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


class TokenValidator:
    """Validates signed gateway tokens with expiration and revocation checks."""

    def __init__(self, secret: str, ttl_minutes: int = 120):
        """Initialise the validator with the shared signing secret and token TTL."""
        self.secret = secret
        self.ttl_seconds = ttl_minutes * 60
        self.revocation_list = _TOKEN_REVOCATION_LIST

    def generate_token(
        self,
        tenant_id: str,
        scopes: list[str],
        tier: str = "free",
        subject: Optional[str] = None,
    ) -> str:
        """Create a signed JWT-style token for a tenant-scoped subject."""
        import uuid

        now = time.time()
        payload = {
            "tenant_id": tenant_id,
            "sub": subject or tenant_id,
            "scopes": scopes,
            "tier": tier,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "jti": str(uuid.uuid4()),
        }
        token = _encode_token(payload, self.secret)
        log.info("Token generated | tenant=%s exp=%s", tenant_id, payload["exp"])
        return token

    def validate_token(self, token: str) -> Optional[dict[str, Any]]:
        """Validate structure, signature, expiry, and revocation for a token."""
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
        except ValueError:
            log.warning("Invalid token format")
            return None

        expected_sig = _sign(payload_b64, self.secret)
        if not hmac.compare_digest(sig_b64, expected_sig):
            log.warning("Invalid token signature")
            return None

        try:
            payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        except Exception as exc:
            log.warning("Token payload decode failed: %s", exc)
            return None

        if payload.get("exp", 0) < time.time():
            log.warning("Expired token rejected | subject=%s", payload.get("sub") or payload.get("tenant_id"))
            return None

        jti = payload.get("jti")
        if jti and jti in self.revocation_list:
            log.warning("Revoked token rejected | jti=%s", jti)
            return None

        return payload

    def revoke_token(self, jti: str) -> None:
        """Mark a token identifier as revoked for future validation checks."""
        self.revocation_list.add(jti)
        log.info("Token revoked | jti=%s", jti)


token_validator = TokenValidator(
    secret=AUTH_SECRET_KEY,
    ttl_minutes=AUTH_TOKEN_TTL_MINUTES,
)


def create_access_token(
    subject: str,
    scopes: Optional[Sequence[str]] = None,
    tenant_id: Optional[str] = None,
    tier: str = "free",
    role: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """Backward-compatible token creation helper."""
    import uuid

    now = time.time()
    payload = {
        "sub": subject,
        "tenant_id": tenant_id,
        "scopes": list(scopes or []),
        "tier": tier,
        "iat": now,
        "exp": now + (AUTH_TOKEN_TTL_MINUTES * 60),
        "jti": str(uuid.uuid4()),
    }
    if role:
        payload["role"] = role
    if email:
        payload["email"] = email
    return _encode_token(payload, AUTH_SECRET_KEY)


def verify_token(token: str) -> Optional[dict[str, Any]]:
    """Validate a bearer token and return its payload when it is usable."""
    return token_validator.validate_token(token)


def validate_credentials(username: str, password: str) -> bool:
    """Legacy username/password check used by the older admin surface."""
    return hmac.compare_digest(username or "", AUTH_USERNAME) and hmac.compare_digest(
        password or "",
        AUTH_PASSWORD,
    )


def resolve_api_key(raw_key: str) -> Optional[dict[str, Any]]:
    """Resolve a raw API key to a tenant auth payload without exposing the secret."""
    cfg = TenantManager.resolve_api_key(raw_key)
    if cfg is None:
        return None
    return _tenant_auth_payload(cfg)


def _effective_scopes(granted: Sequence[str]) -> set[str]:
    effective = set(granted)
    for granted_scope in list(granted):
        effective.update(_SCOPE_HIERARCHY.get(granted_scope, []))
    return effective


def _normalize_required_scopes(
    *,
    scope: Optional[str] = None,
    scopes: Optional[Sequence[str] | str] = None,
) -> list[str]:
    if scope is not None and scopes is not None:
        raise ValueError("Use either 'scope=' or 'scopes=', not both.")
    if scope is not None:
        return [scope]
    if scopes is None:
        return []
    if isinstance(scopes, str):
        return [scopes]
    return [value for value in scopes if value]


def _extract_request(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Request:
    request = kwargs.get("request")
    if isinstance(request, Request):
        return request
    for arg in args:
        if isinstance(arg, Request):
            return arg
    raise RuntimeError(
        f"{func.__name__} uses @require_auth but does not expose a FastAPI Request argument."
    )


def _preserve_endpoint_signature(wrapper: Callable[..., Any], func: Callable[..., Any]) -> Callable[..., Any]:
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__annotations__ = dict(getattr(func, "__annotations__", {}))
    wrapper.__globals__.update(func.__globals__)
    return wrapper


def _extract_bearer_credentials(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Optional[HTTPAuthorizationCredentials]:
    if credentials is not None:
        return credentials

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token.strip())


def _tenant_auth_payload(cfg: TenantConfig) -> dict[str, Any]:
    effective = _effective_scopes(cfg.scopes)
    return {
        "tenant_id": cfg.tenant_id,
        "sub": f"apikey:{cfg.tenant_id}",
        "scopes": sorted(effective),
        "tier": cfg.tier,
        "auth_type": "api_key",
    }


def _check_required_scopes(payload: dict[str, Any], required_scopes: Sequence[str]) -> set[str]:
    effective = _effective_scopes(payload.get("scopes", []))
    for required_scope in required_scopes:
        if required_scope not in effective:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing scope: {required_scope}",
            )
    return effective


def _authenticate_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
    required_scopes: Sequence[str],
) -> dict[str, Any]:
    # Path 1: TenantMiddleware already resolved an API key and stamped request.state.
    auth_type = getattr(request.state, "auth_type", "anonymous")
    if auth_type == "api_key" and getattr(request.state, "tenant_id", None):
        payload = {
            "tenant_id": request.state.tenant_id,
            "sub": getattr(request.state, "subject", f"apikey:{request.state.tenant_id}"),
            "scopes": list(getattr(request.state, "scopes", []) or []),
            "tier": getattr(request.state, "tier", None),
            "auth_type": "api_key",
        }
        effective = _check_required_scopes(payload, required_scopes)
        request.state.scopes = sorted(effective)
        return payload

    # Path 2: Direct X-API-Key header fallback — works even if TenantMiddleware
    # is disabled or failed to run (e.g. misconfiguration, middleware exception).
    raw_api_key = request.headers.get("X-API-Key", "").strip()
    if raw_api_key:
        cfg = TenantManager.resolve_api_key(raw_api_key)
        if cfg is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "X-API-Key"},
            )
        payload = _tenant_auth_payload(cfg)
        effective = _check_required_scopes(payload, required_scopes)
        # Stamp request.state so downstream route handlers can read tenant context.
        request.state.tenant_id = cfg.tenant_id
        request.state.subject = payload["sub"]
        request.state.scopes = sorted(effective)
        request.state.auth_type = "api_key"
        request.state.tier = cfg.tier
        log.info("API key auth via direct header fallback | tenant=%s", cfg.tenant_id)
        return payload

    bearer_credentials = _extract_bearer_credentials(request, credentials)
    if bearer_credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = token_validator.validate_token(bearer_credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    effective = _check_required_scopes(payload, required_scopes)

    tenant_id = payload.get("tenant_id") or getattr(request.state, "tenant_id", None)
    request.state.tenant_id = tenant_id
    request.state.subject = payload.get("sub") or tenant_id
    request.state.token_payload = payload
    request.state.scopes = sorted(effective)
    request.state.auth_type = "bearer"
    request.state.tier = payload.get("tier") or getattr(request.state, "tier", None)

    return payload


class _AuthRequirement:
    def __init__(self, required_scopes: Sequence[str]):
        """Capture the normalized scopes that must be present on each request."""
        self.required_scopes = list(required_scopes)

    def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> Any:
        if callable(request) and not isinstance(request, Request):
            return self._decorate(request)
        return _authenticate_request(request, credentials, self.required_scopes)

    def _decorate(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Authenticate the request before invoking an async endpoint."""
                request = _extract_request(func, args, kwargs)
                _authenticate_request(request, credentials=None, required_scopes=self.required_scopes)
                return await func(*args, **kwargs)

            return _preserve_endpoint_signature(async_wrapper, func)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Authenticate the request before invoking a sync endpoint."""
            request = _extract_request(func, args, kwargs)
            _authenticate_request(request, credentials=None, required_scopes=self.required_scopes)
            return func(*args, **kwargs)

        return _preserve_endpoint_signature(sync_wrapper, func)


class _RequireAuthFactory:
    def __call__(
        self,
        request: Request | None = None,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        *,
        scope: Optional[str] = None,
        scopes: Optional[Sequence[str] | str] = None,
    ) -> Any:
        """Act as a FastAPI dependency factory or a direct route decorator helper."""
        required_scopes = _normalize_required_scopes(scope=scope, scopes=scopes)
        requirement = _AuthRequirement(required_scopes)

        if callable(request) and not isinstance(request, Request):
            return requirement._decorate(request)
        if request is None:
            return requirement
        return _authenticate_request(request, credentials, required_scopes)


require_auth = _RequireAuthFactory()


__all__ = [
    "SCOPES",
    "TokenValidator",
    "create_access_token",
    "hash_api_key",
    "require_auth",
    "resolve_api_key",
    "token_validator",
    "validate_credentials",
    "verify_api_key",
    "verify_token",
]
