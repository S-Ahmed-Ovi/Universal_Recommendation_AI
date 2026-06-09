"""
Model management routes

GET  /models                    — list trained models for the current tenant
POST /models/{model_id}/promote — promote a model to production

Auth-gated (scopes: models:read / models:write), tenant-aware.
The registry is tenant-scoped; each tenant sees only their own models.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.core.deployment.control_plane import DeploymentControlPlane
from backend.core.tenants.manager import TenantManager
from backend.sdk.webhooks.push import emit_model_promoted
from backend.utils.logger import get_logger
from .shared import _get_loader, _get_registry, _serving_cache, _sanitise

log = get_logger(__name__)

router = APIRouter(tags=["Models"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models")
@require_auth(scope="models:read")
async def list_models(
    request: Request,
    algorithm: Optional[str] = None,
    promoted_only: bool = False,
):
    """
    List all trained models belonging to the current tenant.

    Optionally filter by algorithm name or restrict to promoted models only.
    """
    tenant_id: str = request.state.tenant_id
    registry = _get_registry(tenant_id)
    records = await asyncio.to_thread(
        registry.list_models,
        algorithm=algorithm,
        promoted_only=promoted_only,
    )
    return _sanitise(
        {
            "tenant_id": tenant_id,
            "models": [r.to_dict() for r in records],
            "total": len(records),
        }
    )


@router.post("/models/{model_id}/promote")
@require_auth(scope="models:write")
async def promote_model(request: Request, model_id: str):
    """
    Promote a trained model to production for the current tenant.

    Once promoted the model becomes the default for /recommend, /similar,
    and /explain when no explicit algorithm or model_id is supplied.
    The serving pipeline cache is invalidated so the new model is picked up
    immediately.
    """
    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "model_promote",
        1,
        request_state=request.state,
    )

    registry = _get_registry(tenant_id)
    control_plane = DeploymentControlPlane(
        tenant_id=tenant_id,
        registry=registry,
        loader=_get_loader(tenant_id),
    )
    try:
        deployment = await asyncio.to_thread(
            control_plane.promote,
            model_id,
            actor=getattr(request.state, "subject", tenant_id),
            reason="Manual promotion via API",
            source="manual",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Bust the per-tenant serving cache so the promoted model is loaded fresh
    _serving_cache.pop(tenant_id, None)
    log.info("Tenant %s promoted model %s", tenant_id, model_id)

    # Fire model.promoted webhook if the tenant has one registered
    try:
        record = await asyncio.to_thread(registry.get, model_id)
        algorithm = record.algorithm if record else "unknown"
        cfg = await asyncio.to_thread(TenantManager.get, tenant_id)
        if cfg.webhook_url and cfg.webhook_secret:
            emit_model_promoted(
                tenant_id=tenant_id,
                model_id=model_id,
                algorithm=algorithm,
                webhook_url=cfg.webhook_url,
                webhook_secret=cfg.webhook_secret,
            )
    except Exception as wh_exc:
        log.warning("Webhook emit failed for model promotion %s: %s", model_id, wh_exc)

    return _sanitise(
        {
            "status": "promoted",
            "tenant_id": tenant_id,
            "model_id": model_id,
            **deployment,
        }
    )
