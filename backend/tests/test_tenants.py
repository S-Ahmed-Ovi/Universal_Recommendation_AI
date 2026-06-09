from __future__ import annotations

import json
from fastapi.testclient import TestClient

from backend.app_factory import create_app
from backend.api.gateway.auth import hash_api_key, verify_token

from backend.core.tenants.manager import TenantManager
from backend.core.users.manager import UserManager


def test_plaintext_api_key_is_migrated_and_cache_refreshes(write_tenant_config):
    config_path = write_tenant_config(raw_api_key="legacy-key", plaintext_api_key=True)

    resolved = TenantManager.resolve_api_key("legacy-key")

    assert resolved is not None
    assert resolved.tenant_id == "tenant-a"

    migrated_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "api_key_hash" in migrated_payload
    assert "api_key" not in migrated_payload

    migrated_payload["tier"] = "pro"
    config_path.write_text(json.dumps(migrated_payload, indent=2), encoding="utf-8")

    refreshed = TenantManager.get("tenant-a")
    assert refreshed.tier == "pro"

    migrated_payload["tier"] = "enterprise"
    config_path.write_text(json.dumps(migrated_payload, indent=2), encoding="utf-8")
    TenantManager.invalidate_cache("tenant-a")

    invalidated = TenantManager.get("tenant-a")
    assert invalidated.tier == "enterprise"


def test_legacy_default_scopes_are_migrated_to_include_models_write(tmp_path):
    tenant_dir = TenantManager._base / "tenant-a"
    tenant_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "tenant_id": "tenant-a",
        "display_name": "tenant-a",
        "tier": "starter",
        "api_key_hash": hash_api_key("secret-key"),
        "scopes": [
            "training:write",
            "recommend:read",
            "models:read",
            "explain:read",
            "batch:read",
            "similar:read",
            "jobs:read",
        ],
        "created_at": 1.0,
        "updated_at": 1.0,
        "algorithm_mode": "auto",
        "top_k": 10,
        "max_recs": 100,
        "rate_limit_rpm": 60,
        "active": True,
        "notes": "",
        "extra": {},
    }
    config_path = tenant_dir / "config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cfg = TenantManager.get("tenant-a")
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert "models:write" in cfg.scopes
    assert "models:write" in saved["scopes"]


def test_registered_user_can_create_exactly_one_tenant():
    client = TestClient(create_app())

    register_response = client.post(
        "/admin/auth/register",
        json={
            "username": "tenant-owner",
            "email": "tenant-owner@example.com",
            "password": "Password123",
            "confirm_password": "Password123",
        },
    )
    assert register_response.status_code == 200
    register_payload = register_response.json()
    assert register_payload["requires_tenant"] is True

    auth_headers = {"Authorization": f"Bearer {register_payload['access_token']}"}
    create_response = client.post(
        "/admin/tenants",
        headers=auth_headers,
        json={
            "tenant_id": "tenant-one",
            "display_name": "Tenant One",
            "tier": "starter",
            "algorithm_mode": "auto",
            "rate_limit_rpm": 60,
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["tenant_id"] == "tenant-one"
    assert created["owner_subject"] == "tenant-owner"
    assert created["api_key"]
    assert UserManager.get("tenant-owner").tenant_id == "tenant-one"

    duplicate_response = client.post(
        "/admin/tenants",
        headers=auth_headers,
        json={
            "tenant_id": "tenant-two",
            "display_name": "Tenant Two",
            "tier": "starter",
            "algorithm_mode": "auto",
            "rate_limit_rpm": 60,
        },
    )
    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["detail"]["tenant"]["tenant_id"] == "tenant-one"

    list_response = client.get("/admin/tenants", headers=auth_headers)
    assert list_response.status_code == 200
    assert [tenant["tenant_id"] for tenant in list_response.json()["tenants"]] == ["tenant-one"]


def test_registration_requires_email_and_confirms_password():
    client = TestClient(create_app())

    response = client.post(
        "/admin/auth/register",
        json={
            "username": "new-user",
            "email": "new-user@example.com",
            "password": "Password123",
            "confirm_password": "Password123",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "USER"
    assert payload["requires_tenant"] is True
    assert payload["email"] == "new-user@example.com"


def test_registration_rejects_duplicate_email():
    client = TestClient(create_app())

    first = client.post(
        "/admin/auth/register",
        json={
            "username": "new-user",
            "email": "dup@example.com",
            "password": "Password123",
            "confirm_password": "Password123",
        },
    )
    assert first.status_code == 200

    duplicate = client.post(
        "/admin/auth/register",
        json={
            "username": "other-user",
            "email": "dup@example.com",
            "password": "Password123",
            "confirm_password": "Password123",
        },
    )

    assert duplicate.status_code == 409
    assert "Email is already registered" in duplicate.json()["detail"]


def test_tenant_api_key_cannot_manage_admin_tenants():
    TenantManager.create(
        tenant_id="tenant-one",
        display_name="Tenant One",
        owner_subject="tenant-owner",
    )
    raw_key = TenantManager.rotate_api_key("tenant-one")
    client = TestClient(create_app())

    response = client.get("/admin/tenants", headers={"X-API-Key": raw_key})

    assert response.status_code == 403


def test_admin_can_activate_tenant_and_receive_session_token():
    client = TestClient(create_app())

    login = client.post(
        "/admin/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    create_response = client.post(
        "/admin/tenants",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "tenant_id": "tenant-one",
            "display_name": "Tenant One",
            "tier": "starter",
            "algorithm_mode": "auto",
            "rate_limit_rpm": 60,
        },
    )
    assert create_response.status_code == 200

    activate = client.post(
        "/admin/tenants/tenant-one/activate",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert activate.status_code == 200
    payload = activate.json()
    decoded = verify_token(payload["access_token"])
    assert payload["tenant"]["tenant_id"] == "tenant-one"
    assert decoded is not None
    assert decoded["tenant_id"] == "tenant-one"
