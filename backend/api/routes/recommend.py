"""
Recommendation route

POST /recommend  — get personalised recommendations for a single user

Auth-gated (scope: recommend:read), tenant-aware.
Was previously a 503 stub — now fully wired to RecommendationStrategyService.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.config.settings import TOP_K_ALLOWED
from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.core.recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RECOMMEND_STRATEGIES,
    WeightedModelInput,
)
from backend.utils.logger import get_logger
from .shared import (
    _runtime_state,
    _strategy_service,
    _get_last_result,
    _sanitise,
    _validate_top_choice,
)

log = get_logger(__name__)

router = APIRouter(tags=["Recommendations"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class WeightedModelRequest(BaseModel):
    """Client-supplied model selection entry for weighted recommendation requests."""
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


class RecommendRequest(BaseModel):
    """Request payload for a single-user recommendation query."""
    user_id: str
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

    @field_validator("user_id")
    @classmethod
    def _validate_user_id(cls, value: str) -> str:
        """Require a non-empty user identifier before inference starts."""
        v = (value or "").strip()
        if not v:
            raise ValueError("user_id is required")
        return v

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
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/recommend/options")
@require_auth(scope="recommend:read")
async def recommendation_options(
    request: Request,
    top_n_models: Optional[int] = Query(
        None,
        ge=1,
        description="How many ranked supported models to return. Omit to return the full benchmark-selected set.",
    ),
):
    """Return available strategies, algorithms, and promoted model options for this tenant."""
    tenant_id: str = request.state.tenant_id
    last_result = _get_last_result(tenant_id)
    svc = _strategy_service(tenant_id)
    options = await asyncio.to_thread(
        svc.recommendation_options,
        last_result=last_result,
        top_n_models=top_n_models,
    )
    options["available_algorithms"] = sorted(ALGORITHM_REGISTRY.keys())
    return _sanitise(options)


@router.post("/recommend")
@require_auth(scope="recommend:read")
async def get_recommendations(request: Request, req: RecommendRequest):
    """
    Get personalised item recommendations for a single user.

    Supports multiple strategies: best_model, specific_model, specific_algorithm,
    weighted_ensemble, and more. See GET /recommend/options for what's available.
    """
    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "recommend",
        1,
        request_state=request.state,
    )

    payload = RecommendationPayload(
        user_id=req.user_id,
        top_n=req.top_n,
        strategy=req.strategy,
        model_id=req.model_id,
        algorithm=req.algorithm,
        models=[
            WeightedModelInput(algorithm=m.algorithm, weight=m.weight, model_id=m.model_id)
            for m in (req.models or [])
        ],
        auto_normalize_weights=req.auto_normalize_weights,
    )

    last_result = _get_last_result(tenant_id)
    svc = _strategy_service(tenant_id)

    try:
        result = await asyncio.to_thread(
            svc.recommend,
            payload=payload,
            last_result=last_result,
        )
        return _sanitise(result)
    except RecommendationStrategyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Recommendation failed for tenant %s: %s", tenant_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc))
