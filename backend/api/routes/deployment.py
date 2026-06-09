"""
Deployment control plane routes.

These endpoints expose the canonical active deployment, deployment history, and
rollback operations for the current tenant.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.core.deployment.control_plane import DeploymentControlPlane
from backend.utils.logger import get_logger
from .shared import _get_loader, _get_registry, _serving_cache, _sanitise

log = get_logger(__name__)

router = APIRouter(tags=["Deployment"])


def _control_plane(tenant_id: str) -> DeploymentControlPlane:
    """Construct a tenant-scoped deployment control plane backed by shared caches."""
    return DeploymentControlPlane(
        tenant_id=tenant_id,
        registry=_get_registry(tenant_id),
        loader=_get_loader(tenant_id),
    )


@router.get("/deployment/active")
@require_auth(scope="models:read")
async def get_active_deployment(request: Request):
    """Return the currently active deployment payload for the tenant."""
    tenant_id: str = request.state.tenant_id
    payload = await asyncio.to_thread(_control_plane(tenant_id).describe_active)
    return _sanitise(payload)


@router.get("/deployment/status")
@require_auth(scope="models:read")
async def get_deployment_status(request: Request):
    """Expose the same active deployment view under the status alias endpoint."""
    tenant_id: str = request.state.tenant_id
    payload = await asyncio.to_thread(_control_plane(tenant_id).describe_active)
    return _sanitise(payload)


@router.get("/deployment/history")
@require_auth(scope="models:read")
async def get_deployment_history(
    request: Request,
    limit: int = Query(12, ge=1, le=100),
):
    """Return recent deployment events for the tenant in newest-first order."""
    tenant_id: str = request.state.tenant_id
    history = await asyncio.to_thread(_control_plane(tenant_id).history, limit)
    return _sanitise(
        {
            "tenant_id": tenant_id,
            "events": history,
            "total": len(history),
        }
    )


@router.post("/deployment/rollback")
@require_auth(scope="models:write")
async def rollback_deployment(request: Request):
    """Rollback the active deployment to the previous promoted model when available."""
    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "model_promote",
        1,
        request_state=request.state,
    )

    try:
        deployment = await asyncio.to_thread(
            _control_plane(tenant_id).rollback,
            actor=getattr(request.state, "subject", tenant_id),
            reason="Manual rollback via API",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    _serving_cache.pop(tenant_id, None)
    log.info("Tenant %s rolled back active deployment", tenant_id)
    return _sanitise(
        {
            "status": "rolled_back",
            **deployment,
        }
    )
