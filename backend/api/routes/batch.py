"""
Batch recommendation route

POST /recommend/batch  — get recommendations for multiple users in one call

Auth-gated (scope: recommend:read), tenant-aware.
Partial failures are collected and returned alongside successful results
so the caller can handle per-user errors without the whole request failing.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.config.settings import TOP_K_ALLOWED
from backend.core.recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RECOMMEND_STRATEGIES,
    WeightedModelInput,
)
from backend.utils.logger import get_logger
from .shared import (
    _get_last_result,
    _strategy_service,
    _sanitise,
)

log = get_logger(__name__)

router = APIRouter(tags=["Recommendations"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class WeightedModelRequest(BaseModel):
    """Client-supplied model selection entry for weighted batch recommendation requests."""
    algorithm: str
    weight: float
    model_id: Optional[str] = None

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        """Require a non-empty algorithm name for each weighted model entry."""
        v = (value or "").strip()
        if not v:
            raise ValueError("algorithm is required")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        """Require strictly positive model weights for weighted serving."""
        if value <= 0:
            raise ValueError("weight must be positive")
        return value


class BatchRecommendRequest(BaseModel):
    """Request payload for multi-user recommendation inference."""
    user_ids: list[str] = Field(default_factory=list)
    top_n: int = 10
    strategy: Optional[str] = None
    model_id: Optional[str] = None
    algorithm: Optional[str] = None
    models: list[WeightedModelRequest] = Field(default_factory=list)
    auto_normalize_weights: bool = True

    @field_validator("top_n")
    @classmethod
    def _validate_top_n(cls, value: int) -> int:
        """Keep recommendation fan-out within the supported product limits."""
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_n must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("user_ids")
    @classmethod
    def _validate_user_ids(cls, value: list[str]) -> list[str]:
        """Reject empty user identifiers while preserving the original ordering."""
        if not value:
            return value
        cleaned = [v.strip() for v in value if v and v.strip()]
        if len(cleaned) != len(value):
            raise ValueError("user_ids must not contain empty values")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value: Optional[str]) -> Optional[str]:
        """Normalize and validate the requested recommendation strategy."""
        if value is None:
            return None
        v = value.strip().lower()
        if v not in RECOMMEND_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(RECOMMEND_STRATEGIES)}")
        return v


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/recommend/batch")
@require_auth(scope="recommend:read")
async def get_batch_recommendations(request: Request, req: BatchRecommendRequest):
    """
    Get personalised recommendations for multiple users in a single request.

    Partial failures per user are collected in the `errors` field so that
    one bad user_id does not abort results for the rest.
    Returns 422 only if *every* user fails.
    """
    if not req.user_ids:
        raise HTTPException(status_code=400, detail="user_ids cannot be empty")

    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "recommend_batch",
        len(req.user_ids),
        request_state=request.state,
    )

    last_result = _get_last_result(tenant_id)
    svc = _strategy_service(tenant_id)

    model_inputs = [
        WeightedModelInput(
            algorithm=m.algorithm,
            weight=m.weight,
            model_id=m.model_id,
        )
        for m in (req.models or [])
    ]

    def _run_batch() -> tuple[list[dict], list[dict]]:
        """Execute per-user recommendation work and collect partial failures safely."""
        results: list[dict] = []
        errors: list[dict] = []

        for user_id in req.user_ids:
            payload = RecommendationPayload(
                user_id=user_id,
                top_n=req.top_n,
                strategy=req.strategy,
                model_id=req.model_id,
                algorithm=req.algorithm,
                models=model_inputs,
                auto_normalize_weights=req.auto_normalize_weights,
            )
            try:
                results.append(svc.recommend(payload=payload, last_result=last_result))
            except RecommendationStrategyError as exc:
                errors.append({"user_id": user_id, "error": str(exc)})
            except Exception as exc:
                log.warning(
                    "Batch recommend error for tenant=%s user=%s: %s",
                    tenant_id, user_id, exc,
                )
                errors.append({"user_id": user_id, "error": str(exc)})
        return results, errors

    results, errors = await asyncio.to_thread(_run_batch)

    if not results:
        raise HTTPException(
            status_code=422,
            detail="No batch recommendation results could be generated.",
        )

    return _sanitise({"results": results, "errors": errors})
