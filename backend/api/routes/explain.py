"""
Explanation route

GET /explain/{user_id}/{item_id}  — explain why an item was recommended to a user

Auth-gated (scope: recommend:read), tenant-aware.
Delegates to the serving pipeline's explain() method and returns a rich
human-readable description alongside numeric scores.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Request

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.utils.logger import get_logger
from .shared import (
    _runtime_state,
    _get_serving_pipeline,
    _default_algorithm,
    _sanitise,
)

log = get_logger(__name__)

router = APIRouter(tags=["Explainability"])


@router.get("/explain/{user_id}/{item_id}")
@require_auth(scope="recommend:read")
async def explain(
    request: Request,
    user_id: str,
    item_id: str,
    score: float = 0.0,
    algorithm: Optional[str] = None,
):
    """
    Return a human-readable explanation of why `item_id` was (or would be)
    recommended to `user_id`.

    Includes contributing factors, algorithm reasoning, and the raw numeric
    score. Useful for transparency dashboards and debugging.
    """
    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "explain",
        1,
        request_state=request.state,
    )

    algo = algorithm or _default_algorithm(tenant_id, _runtime_state)
    pipeline = _get_serving_pipeline(tenant_id, algo, _runtime_state)

    explanation = await asyncio.to_thread(
        pipeline.explain,
        user_id=user_id,
        item_id=item_id,
        score=score,
    )
    result = explanation.__dict__ if hasattr(explanation, "__dict__") else dict(explanation)

    return _sanitise(
        {
            "tenant_id": tenant_id,
            "algorithm": algo,
            "user_id": user_id,
            "item_id": item_id,
            **result,
        }
    )
