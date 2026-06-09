"""
Tenant-aware request helpers.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Request

from backend.api.gateway.auth import require_auth
from backend.core.tenants.manager import TenantManager


def require_scope(scope: str):
    """Return a FastAPI dependency that requires a single scope."""
    return require_auth(scopes=[scope])


def build_tenant_context(
    tenant_id: str,
    request: Optional[Request] = None,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a lightweight tenant context payload for request.state.
    """
    cfg = TenantManager.get(tenant_id)
    context: dict[str, Any] = {
        "tenant_id": cfg.tenant_id,
        "display_name": cfg.display_name,
        "tier": cfg.tier,
        "scopes": list(cfg.scopes),
        "rate_limit_rpm": cfg.rate_limit_rpm,
        "active": cfg.active,
    }
    if request is not None:
        context["request_id"] = getattr(request.state, "request_id", None)
        context["auth_type"] = getattr(request.state, "auth_type", None)
        context["subject"] = getattr(request.state, "subject", None)
    if extra:
        context.update(extra)
    return context
