"""
Starlette middleware for the multi-tenant gateway.

Responsibilities
----------------
TenantMiddleware
    Extracts tenant context from bearer tokens or API keys and stamps
    request.state so routes and auth checks share a single source of truth.

RequestContextMiddleware
    Attaches a request_id to every request for log correlation.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Optional

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from backend.core.tenants.manager import TenantConfig, TenantManager
from backend.utils.logger import get_logger

log = get_logger(__name__)

_PUBLIC_PREFIXES = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/auth/login",
    "/algorithms",
)


def _is_public(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


def _decode_jwt_payload(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except Exception:
        return None


def _attach_tenant_state(
    request: Request,
    cfg: TenantConfig,
    *,
    auth_type: str,
    scopes: Optional[list[str]] = None,
    subject: Optional[str] = None,
) -> Optional[JSONResponse]:
    request.state.tenant_id = cfg.tenant_id
    request.state.subject = subject or request.state.subject
    request.state.scopes = list(scopes or cfg.scopes)
    request.state.auth_type = auth_type
    request.state.tier = cfg.tier
    request.state.tenant_config = cfg
    request.state.rate_limit_rpm = cfg.rate_limit_rpm

    try:
        from backend.core.tenants.context import build_tenant_context

        request.state.tenant = build_tenant_context(
            cfg.tenant_id,
            request,
            extra={"auth_type": auth_type},
        )
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        log.warning("Tenant context attachment skipped | tenant=%s error=%s", cfg.tenant_id, exc)

    return None


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a unique request_id to each request and response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Stamp a request ID on request.state and echo it back in the response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Extract tenant context from each request and stamp request.state.

    Protected routes are not rejected here for missing credentials; auth
    dependencies handle 401 and 403 responses so route behavior stays consistent.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Warm the tenant cache before the middleware starts serving requests."""
        super().__init__(app)
        TenantManager.refresh_cache(force=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Resolve tenant auth context, attach state, and log request outcomes."""
        t_start = time.perf_counter()
        path = request.url.path

        request.state.tenant_id = None
        request.state.subject = None
        request.state.scopes = []
        request.state.auth_type = "anonymous"
        request.state.tier = None
        request.state.tenant = None
        request.state.tenant_config = None
        request.state.rate_limit_rpm = None

        if not _is_public(path):
            auth_header = request.headers.get("Authorization", "")
            api_key = request.headers.get("X-API-Key", "")

            if auth_header.lower().startswith("bearer "):
                token = auth_header.split(" ", 1)[1].strip()
                payload = _decode_jwt_payload(token) or {}
                tenant_id = payload.get("tenant_id")
                if tenant_id:
                    try:
                        cfg = TenantManager.get(tenant_id)
                    except Exception as exc:
                        log.warning("Bearer tenant lookup failed | tenant=%s error=%s", tenant_id, exc)
                    else:
                        error_response = _attach_tenant_state(
                            request,
                            cfg,
                            auth_type="bearer",
                            scopes=list(payload.get("scopes", []) or []),
                            subject=payload.get("sub") or tenant_id,
                        )
                        if error_response is not None:
                            return error_response

            elif api_key:
                cfg = TenantManager.resolve_api_key(api_key)
                if cfg is not None:
                    error_response = _attach_tenant_state(
                        request,
                        cfg,
                        auth_type="api_key",
                        scopes=list(cfg.scopes),
                        subject=f"apikey:{cfg.tenant_id}",
                    )
                    if error_response is not None:
                        return error_response

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error(
                "Unhandled exception path=%s tenant=%s: %s",
                path,
                request.state.tenant_id,
                exc,
                exc_info=True,
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        log.info(
            "%s %s -> %d tenant=%s auth=%s latency=%sms req_id=%s",
            request.method,
            path,
            response.status_code,
            request.state.tenant_id or "anon",
            request.state.auth_type,
            latency_ms,
            getattr(request.state, "request_id", "-"),
        )
        return response


def reload_tenant_index() -> dict[str, int]:
    """Flush and rebuild the in-memory tenant/API-key cache."""
    return TenantManager.refresh_cache(force=True)


__all__ = [
    "RequestContextMiddleware",
    "TenantMiddleware",
    "reload_tenant_index",
]
