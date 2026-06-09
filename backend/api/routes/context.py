"""
Tenant context route

GET /tenant/context — resolve the active tenant authenticated by X-API-Key
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.api.gateway.auth import require_auth

router = APIRouter(tags=["Tenant Context"])


@router.get("/tenant/context")
@require_auth(scopes=[])
async def get_tenant_context(request: Request):
    """
    Return the verified tenant context for the current authenticated request.

    This lets frontend clients confirm which tenant the current API key
    actually belongs to, instead of trusting a locally typed tenant label.
    """
    return {
        "tenant_id": getattr(request.state, "tenant_id", None),
        "tier": getattr(request.state, "tier", None),
        "scopes": list(getattr(request.state, "scopes", []) or []),
        "auth_type": getattr(request.state, "auth_type", None),
    }
