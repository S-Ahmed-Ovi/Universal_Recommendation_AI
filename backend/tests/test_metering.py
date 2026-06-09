from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.api.gateway.auth import create_access_token, require_auth
from backend.api.gateway.metering import MeteringMiddleware, TIER_LIMITS, TierLimits, metering
from backend.api.gateway.middleware import TenantMiddleware


def _build_metered_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TenantMiddleware)
    app.add_middleware(MeteringMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/recommend")
    @require_auth(scope="recommend:read")
    async def recommend(request: Request):
        await metering.record_usage(
            tenant_id=request.state.tenant_id,
            operation="recommend",
            count=1,
            request_state=request.state,
        )
        return {"ok": True}

    return app


def test_metering_middleware_does_not_crash_health():
    client = TestClient(_build_metered_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_limiting_returns_429(monkeypatch, write_tenant_config):
    write_tenant_config()
    token = create_access_token(
        subject="tenant-a",
        tenant_id="tenant-a",
        scopes=["recommend:read"],
        tier="free",
    )
    monkeypatch.setitem(
        TIER_LIMITS,
        "free",
        TierLimits(
            recommend=1,
            recommend_batch=5,
            training=2,
            similar=30,
            explain=20,
            model_promote=5,
            default=60,
        ),
    )
    client = TestClient(_build_metered_app())

    first = client.post("/recommend", headers={"Authorization": f"Bearer {token}"})
    second = client.post("/recommend", headers={"Authorization": f"Bearer {token}"})

    assert first.status_code == 200
    assert second.status_code == 429
