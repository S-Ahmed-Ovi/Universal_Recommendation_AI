"""
Usage tracking and rate limiting for the gateway.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import HTTPException
from starlette.datastructures import State
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TierLimits:
    recommend: int
    recommend_batch: int
    training: int
    similar: int
    explain: int
    model_promote: int
    default: int


TIER_LIMITS: Dict[str, TierLimits] = {
    "free": TierLimits(
        recommend=30,
        recommend_batch=5,
        training=2,
        similar=30,
        explain=20,
        model_promote=5,
        default=60,
    ),
    "pro": TierLimits(
        recommend=300,
        recommend_batch=50,
        training=10,
        similar=300,
        explain=200,
        model_promote=20,
        default=600,
    ),
    "enterprise": TierLimits(
        recommend=10_000,
        recommend_batch=2_000,
        training=100,
        similar=10_000,
        explain=5_000,
        model_promote=200,
        default=20_000,
    ),
}

_DEFAULT_TIER = "free"
_WINDOW_SECONDS = 60


@dataclass
class _WindowCounter:
    window_start: float = field(default_factory=time.time)
    count: int = 0

    def _reset_if_expired(self) -> None:
        now = time.time()
        if now - self.window_start >= _WINDOW_SECONDS:
            self.window_start = now
            self.count = 0

    def projected_count(self, n: int = 1) -> int:
        """Return the next window count without mutating the current counter."""
        self._reset_if_expired()
        return self.count + n

    def increment(self, n: int = 1) -> int:
        """Advance the current window counter and return the new value."""
        self._reset_if_expired()
        self.count += n
        return self.count


@dataclass
class TenantUsage:
    tenant_id: str
    tier: str = _DEFAULT_TIER
    totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated: float = field(default_factory=time.time)


class UsageTracker:
    """Records API usage per tenant and enforces in-process rate limits."""

    def __init__(self) -> None:
        """Initialise the in-memory usage ledger and per-operation windows."""
        self._usage: Dict[str, TenantUsage] = {}
        self._windows: Dict[str, _WindowCounter] = {}
        self._lock = asyncio.Lock()

    async def record_usage(
        self,
        tenant_id: str,
        operation: str,
        count: int = 1,
        tier: str = _DEFAULT_TIER,
        request_state: Optional[State] = None,
        limit_override: Optional[int] = None,
    ) -> None:
        """Reserve and record tenant usage while enforcing the active rate limit."""
        if not tenant_id or count <= 0:
            return

        reservations: Optional[dict[str, int]] = None
        previously_reserved = 0
        if request_state is not None:
            reservations = getattr(request_state, "metering_reservations", None)
            if not isinstance(reservations, dict):
                reservations = {}
            previously_reserved = int(reservations.get(operation, 0) or 0)
            if count <= previously_reserved:
                return

        effective_count = count - previously_reserved
        resolved_tier = tier if tier in TIER_LIMITS else _DEFAULT_TIER

        async with self._lock:
            usage = self._usage.get(tenant_id)
            if usage is None:
                usage = TenantUsage(tenant_id=tenant_id, tier=resolved_tier)
                self._usage[tenant_id] = usage

            usage.tier = resolved_tier
            usage.last_updated = time.time()

            tier_limits = TIER_LIMITS.get(resolved_tier, TIER_LIMITS[_DEFAULT_TIER])
            limit = limit_override or getattr(tier_limits, operation, tier_limits.default)

            window_key = f"{tenant_id}:{operation}"
            counter = self._windows.setdefault(window_key, _WindowCounter())
            projected_count = counter.projected_count(effective_count)
            if projected_count > limit:
                log.warning(
                    "Rate limit exceeded | tenant=%s op=%s count=%d limit=%d",
                    tenant_id,
                    operation,
                    projected_count,
                    limit,
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {projected_count}/{limit} {operation}s per minute",
                )

            current_count = counter.increment(effective_count)
            usage.totals[operation] += effective_count

            if reservations is not None and request_state is not None:
                reservations[operation] = count
                request_state.metering_reservations = reservations

            if current_count > limit * 0.8:
                log.warning(
                    "Near rate limit | tenant=%s op=%s count=%d limit=%d",
                    tenant_id,
                    operation,
                    current_count,
                    limit,
                )
            else:
                log.debug(
                    "Usage recorded | tenant=%s op=%s count=%d tier=%s",
                    tenant_id,
                    operation,
                    effective_count,
                    resolved_tier,
                )

    async def reset_tenant_limits(self, tenant_id: str) -> None:
        """Clear all active rate-limit windows for a specific tenant."""
        async with self._lock:
            keys_to_delete = [key for key in self._windows if key.startswith(f"{tenant_id}:")]
            for key in keys_to_delete:
                del self._windows[key]
            log.debug("Rate limit windows reset | tenant=%s", tenant_id)

    def get_tenant_usage(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Return the accumulated usage snapshot for one tenant, if present."""
        usage = self._usage.get(tenant_id)
        if usage is None:
            return None
        return {
            "tenant_id": tenant_id,
            "tier": usage.tier,
            "totals": dict(usage.totals),
            "last_updated": usage.last_updated,
        }

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """Return usage snapshots for every tenant currently tracked in memory."""
        return {
            tenant_id: {
                "tenant_id": tenant_id,
                "tier": usage.tier,
                "totals": dict(usage.totals),
                "last_updated": usage.last_updated,
            }
            for tenant_id, usage in self._usage.items()
        }


def _operation_for_request(request: Request) -> Optional[str]:
    path = request.url.path
    method = request.method.upper()

    if path.startswith("/health") or path.startswith("/docs") or path.startswith("/openapi.json"):
        return None
    if path.startswith("/redoc") or path.startswith("/auth/login") or path.startswith("/algorithms"):
        return None
    if method == "POST" and path == "/recommend":
        return "recommend"
    if method == "POST" and path == "/recommend/batch":
        return "recommend_batch"
    if method == "POST" and path.startswith("/train/"):
        return "training"
    if method == "GET" and path.startswith("/similar/"):
        return "similar"
    if method == "GET" and path.startswith("/explain/"):
        return "explain"
    if method == "POST" and path.startswith("/models/") and path.endswith("/promote"):
        return "model_promote"
    return "default"


class MeteringMiddleware(BaseHTTPMiddleware):
    """
    Pre-reserve one usage unit for authenticated tenant requests.

    Route handlers can call `metering.record_usage(..., request_state=request.state)`
    to reuse or increase that reservation without double-counting.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the ASGI app with automatic tenant metering reservations."""
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Pre-reserve per-request usage units before the route handler runs."""
        request.state.metering_reservations = {}

        operation = _operation_for_request(request)
        tenant_id = getattr(request.state, "tenant_id", None)
        tier = getattr(request.state, "tier", None) or _DEFAULT_TIER

        if operation is not None and tenant_id:
            limit_override = getattr(request.state, "rate_limit_rpm", None)
            try:
                await metering.record_usage(
                    tenant_id=tenant_id,
                    operation=operation,
                    count=1,
                    tier=tier,
                    request_state=request.state,
                    limit_override=limit_override if operation == "default" else None,
                )
            except HTTPException as exc:
                if exc.status_code == 429:
                    return JSONResponse(status_code=429, content={"detail": exc.detail})
                raise

        return await call_next(request)


metering = UsageTracker()


async def use_redis_for_rate_limiting(redis_url: Optional[str]) -> None:
    """Probe Redis connectivity for future distributed rate-limiting support."""
    if not redis_url:
        log.info("Redis not configured; using in-process rate limiting")
        return

    try:
        import aioredis

        await aioredis.create_redis_pool(redis_url)
        log.info("Redis configured for distributed rate limiting: %s", redis_url)
    except ImportError:
        log.warning("aioredis not installed; cannot use Redis rate limiting")
    except Exception as exc:
        log.warning("Failed to connect to Redis: %s", exc)


__all__ = [
    "MeteringMiddleware",
    "TIER_LIMITS",
    "TierLimits",
    "UsageTracker",
    "metering",
    "use_redis_for_rate_limiting",
]
