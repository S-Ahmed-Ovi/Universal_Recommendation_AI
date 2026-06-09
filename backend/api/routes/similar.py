"""
Similar-items route

GET /similar/{item_id}  — find items most similar to a given item

Auth-gated (scope: recommend:read), tenant-aware.
Uses the serving pipeline for the requested (or default) algorithm.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.config.settings import TOP_K_ALLOWED
from backend.utils.logger import get_logger
from .shared import (
    _runtime_state,
    _get_serving_pipeline,
    _default_algorithm,
    _sanitise,
    _validate_top_choice,
)

log = get_logger(__name__)

router = APIRouter(tags=["Recommendations"])


@router.get("/similar/{item_id}")
@require_auth(scope="recommend:read")
async def similar_items(
    request: Request,
    item_id: str,
    top_n: int = 10,
    algorithm: Optional[str] = None,
):
    """
    Return the top-N items most similar to `item_id`.

    Falls back to the last-trained best algorithm when `algorithm` is omitted.
    """
    _validate_top_choice("top_n", top_n, TOP_K_ALLOWED)

    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "similar",
        1,
        request_state=request.state,
    )

    algo = algorithm or _default_algorithm(tenant_id, _runtime_state)
    pipeline = _get_serving_pipeline(tenant_id, algo, _runtime_state)

    similar = await asyncio.to_thread(
        pipeline.similar_items,
        item_id,
        top_n=top_n,
        algorithm=algo,
    )
    return _sanitise(
        {
            "tenant_id": tenant_id,
            "algorithm": algo,
            "item_id": item_id,
            "similar": similar,
        }
    )
