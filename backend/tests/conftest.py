from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from backend.api.gateway.auth import hash_api_key
from backend.api.gateway.metering import metering
from backend.api.routes.shared import (
    _jobs,
    _loaders,
    _recommenders,
    _registries,
    _runtime_state,
    _serving_cache,
    _strategy_services,
)
from backend.core.tenants.manager import TenantManager
from backend.core.users.manager import UserManager


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()

    monkeypatch.setattr(TenantManager, "_base", tenants_root)
    monkeypatch.setattr(TenantManager, "_cache_ttl_seconds", 0.0)
    monkeypatch.setattr(UserManager, "_path", tmp_path / "users.json")
    TenantManager.invalidate_cache()
    metering._usage.clear()
    metering._windows.clear()
    _jobs.clear()
    _registries.clear()
    _loaders.clear()
    _recommenders.clear()
    _strategy_services.clear()
    _serving_cache.clear()
    _runtime_state.clear()

    yield

    TenantManager.invalidate_cache()
    metering._usage.clear()
    metering._windows.clear()
    _jobs.clear()
    _registries.clear()
    _loaders.clear()
    _recommenders.clear()
    _strategy_services.clear()
    _serving_cache.clear()
    _runtime_state.clear()


@pytest.fixture
def write_tenant_config() -> Callable[..., Path]:
    def _write(
        tenant_id: str = "tenant-a",
        raw_api_key: str = "secret-key",
        *,
        tier: str = "free",
        scopes: list[str] | None = None,
        plaintext_api_key: bool = False,
    ) -> Path:
        tenant_dir = TenantManager._base / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "tenant_id": tenant_id,
            "display_name": tenant_id,
            "tier": tier,
            "scopes": scopes or ["recommend:read"],
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
        if plaintext_api_key:
            payload["api_key"] = raw_api_key
        else:
            payload["api_key_hash"] = hash_api_key(raw_api_key)

        config_path = tenant_dir / "config.json"
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    return _write
