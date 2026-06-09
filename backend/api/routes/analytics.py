"""
Analytics route

GET /analytics — get usage statistics and performance metrics for the tenant
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from .shared import _sanitise

router = APIRouter(tags=["Analytics"])

@router.get("/analytics")
@require_auth(scope="models:read")
async def get_analytics(request: Request):
    """
    Return usage statistics for the current tenant.
    """
    tenant_id: str = request.state.tenant_id
    usage = metering.get_tenant_usage(tenant_id)
    
    if not usage:
        return {
            "tenant_id": tenant_id,
            "totals": {},
            "tier": getattr(request.state, "tier", "free"),
            "message": "No usage data recorded yet."
        }
    
    return _sanitise(usage)
