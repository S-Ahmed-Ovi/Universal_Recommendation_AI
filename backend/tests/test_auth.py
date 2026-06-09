from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.app_factory import create_app
from backend.api.gateway.auth import create_access_token, hash_api_key, require_auth, verify_api_key
from backend.api.gateway.middleware import TenantMiddleware


def _build_auth_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TenantMiddleware)

    @app.get("/scope")
    @require_auth(scope="recommend:read")
    async def scope_route(request: Request):
        return {"tenant_id": request.state.tenant_id}

    @app.get("/scopes")
    @require_auth(scopes=["recommend:read"])
    async def scopes_route(request: Request):
        return {"tenant_id": request.state.tenant_id}

    @app.get("/api-key")
    @require_auth(scope="recommend:read")
    async def api_key_route(request: Request):
        return {"tenant_id": request.state.tenant_id, "auth_type": request.state.auth_type}

    return app


def test_require_auth_accepts_scope_and_scopes(write_tenant_config):
    write_tenant_config()
    token = create_access_token(
        subject="tenant-a",
        tenant_id="tenant-a",
        scopes=["recommend:read"],
        tier="free",
    )
    client = TestClient(_build_auth_app())

    scope_response = client.get("/scope", headers={"Authorization": f"Bearer {token}"})
    scopes_response = client.get("/scopes", headers={"Authorization": f"Bearer {token}"})

    assert scope_response.status_code == 200
    assert scope_response.json()["tenant_id"] == "tenant-a"
    assert scopes_response.status_code == 200
    assert scopes_response.json()["tenant_id"] == "tenant-a"


def test_missing_and_invalid_auth_are_rejected(write_tenant_config):
    write_tenant_config()
    client = TestClient(_build_auth_app())

    missing_response = client.get("/scope")
    invalid_response = client.get("/scope", headers={"Authorization": "Bearer not-a-real-token"})

    assert missing_response.status_code == 401
    assert invalid_response.status_code == 401


def test_api_key_hashing_and_verification(write_tenant_config):
    raw_key = "secret-key"
    write_tenant_config(raw_api_key=raw_key)
    client = TestClient(_build_auth_app())

    response = client.get("/api-key", headers={"X-API-Key": raw_key})

    assert verify_api_key(raw_key, hash_api_key(raw_key))
    assert response.status_code == 200
    assert response.json() == {"tenant_id": "tenant-a", "auth_type": "api_key"}


def test_public_register_meta_exposes_assignable_roles():
    client = TestClient(create_app())

    response = client.get("/admin/auth/register-meta")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_role"] == "USER"
    assert payload["assignable_on_registration"] == ["USER"]
    assert any(role["role"] == "USER" for role in payload["roles"])
