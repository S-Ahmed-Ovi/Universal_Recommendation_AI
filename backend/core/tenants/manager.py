"""
Tenant configuration storage and lookup.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.config.settings import BASE_TENANTS_DIR
from backend.utils.logger import get_logger

log = get_logger(__name__)

TIERS: tuple[str, ...] = ("free", "starter", "pro", "enterprise")
ALL_SCOPES: tuple[str, ...] = (
    "training:write",
    "recommend:read",
    "models:read",
    "models:write",
    "explain:read",
    "batch:read",
    "similar:read",
    "jobs:read",
    "tenant:manage",
    "admin:read",
    "admin:write",
)
DEFAULT_SCOPES: tuple[str, ...] = (
    "training:write",
    "recommend:read",
    "models:read",
    "models:write",
    "explain:read",
    "batch:read",
    "similar:read",
    "jobs:read",
)

_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
_CONFIG_NAME = "config.json"


class TenantError(RuntimeError):
    """Base error for tenant operations."""


class TenantNotFoundError(TenantError):
    """Raised when a tenant cannot be located on disk."""


class TenantAlreadyExistsError(TenantError):
    """Raised when attempting to create a tenant that already exists."""


class TenantValidationError(TenantError):
    """Raised when tenant data fails validation."""


@dataclass
class TenantConfig:
    tenant_id: str
    display_name: str
    tier: str = "starter"
    scopes: list[str] = field(default_factory=list)
    algorithm_mode: str = "auto"
    rate_limit_rpm: int = 60
    active: bool = True
    notes: str = ""
    top_k: int = 10
    max_recs: int = 100
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    last_activated_at: Optional[float] = None
    api_key: str = ""
    api_key_hash: str = ""
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    owner_subject: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the tenant config to a JSON-safe dictionary."""
        payload = {
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "tier": self.tier,
            "scopes": list(self.scopes),
            "algorithm_mode": self.algorithm_mode,
            "rate_limit_rpm": self.rate_limit_rpm,
            "active": self.active,
            "notes": self.notes,
            "top_k": self.top_k,
            "max_recs": self.max_recs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_activated_at": self.last_activated_at,
            "api_key": self.api_key,
            "api_key_hash": self.api_key_hash,
            "webhook_url": self.webhook_url,
            "webhook_secret": self.webhook_secret,
            "owner_subject": self.owner_subject,
            "extra": dict(self.extra),
        }
        return payload


class TenantManager:
    """Filesystem-backed tenant configuration manager."""

    _base: Path = Path(BASE_TENANTS_DIR)
    _cache_ttl_seconds: float = 30.0
    _cache: dict[str, tuple[float, TenantConfig]] = {}
    _lock = threading.RLock()

    @classmethod
    def _base_dir(cls) -> Path:
        base = cls._base
        if isinstance(base, Path):
            return base.expanduser().resolve()
        return Path(base).expanduser().resolve()

    @classmethod
    def _tenant_dir(cls, tenant_id: str) -> Path:
        return cls._base_dir() / tenant_id

    @classmethod
    def _config_path(cls, tenant_id: str) -> Path:
        return cls._tenant_dir(tenant_id) / _CONFIG_NAME

    @staticmethod
    def _hash_api_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def _validate_tenant_id(cls, tenant_id: str) -> str:
        value = (tenant_id or "").strip()
        if not value:
            raise TenantValidationError("tenant_id is required")
        if not _TENANT_ID_RE.fullmatch(value):
            raise TenantValidationError(
                "tenant_id must be 3-63 chars and contain only lowercase letters, digits, and hyphens"
            )
        return value

    @classmethod
    def _validate_choice(cls, name: str, value: str, allowed: tuple[str, ...]) -> str:
        cleaned = (value or "").strip()
        if cleaned not in allowed:
            raise TenantValidationError(f"{name} must be one of {allowed}")
        return cleaned

    @classmethod
    def _validate_scopes(cls, scopes: Optional[list[str]]) -> list[str]:
        if scopes is None:
            return list(DEFAULT_SCOPES)
        cleaned: list[str] = []
        for scope in scopes:
            item = (scope or "").strip()
            if not item:
                raise TenantValidationError("scopes must not contain empty values")
            if item not in ALL_SCOPES:
                raise TenantValidationError(f"Unknown scope: {item}")
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    @classmethod
    def _load_raw_payload(cls, tenant_id: str) -> dict[str, Any]:
        path = cls._config_path(tenant_id)
        if not path.exists():
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found") from exc
        except Exception as exc:
            raise TenantValidationError(f"Failed to read tenant config for '{tenant_id}': {exc}") from exc

        if not isinstance(payload, dict):
            raise TenantValidationError(f"Tenant config for '{tenant_id}' must be a JSON object")
        return payload

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> TenantConfig:
        tenant_id = cls._validate_tenant_id(str(payload.get("tenant_id", "")))
        display_name = str(payload.get("display_name") or tenant_id)
        tier = cls._validate_choice("tier", str(payload.get("tier", "starter")), TIERS)
        algorithm_mode = cls._validate_choice(
            "algorithm_mode",
            str(payload.get("algorithm_mode", "auto")),
            ("auto", "explicit", "implicit", "hybrid"),
        )
        scopes = cls._validate_scopes(payload.get("scopes") if isinstance(payload.get("scopes"), list) else None)
        rate_limit_rpm = int(payload.get("rate_limit_rpm", 60))
        if rate_limit_rpm < 1:
            raise TenantValidationError("rate_limit_rpm must be positive")

        known_keys = {
            "tenant_id",
            "display_name",
            "tier",
            "scopes",
            "algorithm_mode",
            "rate_limit_rpm",
            "active",
            "notes",
            "top_k",
            "max_recs",
            "created_at",
            "updated_at",
            "last_activated_at",
            "api_key",
            "api_key_hash",
            "webhook_url",
            "webhook_secret",
            "owner_subject",
            "extra",
        }

        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        merged_extra = {}
        # 1. Start with what's in 'extra'
        for k, v in extra.items():
            if k not in known_keys:
                merged_extra[k] = v
        # 2. Add unknown top-level keys
        for key, value in payload.items():
            if key not in known_keys:
                merged_extra[key] = value

        cfg = TenantConfig(
            tenant_id=tenant_id,
            display_name=display_name,
            tier=tier,
            scopes=scopes,
            algorithm_mode=algorithm_mode,
            rate_limit_rpm=rate_limit_rpm,
            active=bool(payload.get("active", True)),
            notes=str(payload.get("notes", "")),
            top_k=int(payload.get("top_k", 10)),
            max_recs=int(payload.get("max_recs", 100)),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
            last_activated_at=(
                float(payload.get("last_activated_at")) if payload.get("last_activated_at") else None
            ),
            api_key=str(payload.get("api_key", "")),
            api_key_hash=str(payload.get("api_key_hash", "")),
            webhook_url=(payload.get("webhook_url") or None),
            webhook_secret=(payload.get("webhook_secret") or None),
            owner_subject=(str(payload.get("owner_subject")).strip() if payload.get("owner_subject") else None),
            extra=merged_extra,
        )
        return cfg

    @classmethod
    def _persist(cls, cfg: TenantConfig) -> None:
        path = cls._config_path(cfg.tenant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = cfg.to_dict()
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)

    @classmethod
    def _needs_reload(cls, tenant_id: str) -> bool:
        cached = cls._cache.get(tenant_id)
        if cached is None:
            return True
        if cls._cache_ttl_seconds <= 0:
            return True
        timestamp, _ = cached
        return (time.time() - timestamp) >= cls._cache_ttl_seconds

    @classmethod
    def _cache_config(cls, cfg: TenantConfig) -> TenantConfig:
        cls._cache[cfg.tenant_id] = (time.time(), cfg)
        return cfg

    @classmethod
    def _read_config(cls, tenant_id: str) -> TenantConfig:
        payload = cls._load_raw_payload(tenant_id)
        cfg = cls._from_payload(payload)

        migrated = False
        if "api_key" in payload and not cfg.api_key_hash:
            cfg.api_key_hash = cls._hash_api_key(str(payload.get("api_key", "")))
            migrated = True

        if "models:write" not in cfg.scopes:
            cfg.scopes.append("models:write")
            migrated = True

        if migrated:
            cfg.updated_at = max(cfg.updated_at, time.time())
            cls._persist(cfg)

        return cfg

    @classmethod
    def invalidate_cache(cls, tenant_id: Optional[str] = None) -> None:
        """Drop a cached tenant config or clear the entire cache."""
        with cls._lock:
            if tenant_id is None:
                cls._cache.clear()
                return
            cls._cache.pop(tenant_id, None)

    @classmethod
    def refresh_cache(cls, force: bool = False) -> dict[str, int]:
        """Reload tenant configs from disk into the in-memory cache."""
        with cls._lock:
            if force:
                cls._cache.clear()

            base = cls._base_dir()
            if not base.exists():
                return {"tenants": 0, "cached": 0}

            count = 0
            for path in base.iterdir():
                if not path.is_dir():
                    continue
                tenant_id = path.name
                config_path = path / _CONFIG_NAME
                if not config_path.exists():
                    continue
                try:
                    cfg = cls._read_config(tenant_id)
                except TenantError:
                    continue
                cls._cache_config(cfg)
                count += 1

            return {"tenants": count, "cached": len(cls._cache)}

    @classmethod
    def _store_in_cache(cls, cfg: TenantConfig) -> TenantConfig:
        with cls._lock:
            return cls._cache_config(cfg)

    @classmethod
    def _write_config(cls, tenant_id: str, cfg: TenantConfig) -> None:
        validated = cls._validate_tenant_id(tenant_id)
        if validated != cfg.tenant_id:
            raise TenantValidationError("tenant_id mismatch")
        cfg.updated_at = time.time()
        cls._persist(cfg)

    @classmethod
    def get(cls, tenant_id: str) -> TenantConfig:
        """Load a single tenant configuration."""
        tenant_id = cls._validate_tenant_id(tenant_id)
        with cls._lock:
            if not cls._needs_reload(tenant_id):
                return cls._cache[tenant_id][1]

            cfg = cls._read_config(tenant_id)
            return cls._cache_config(cfg)

    @classmethod
    def list_tenants(
        cls,
        active_only: bool = False,
        tier: Optional[str] = None,
        owner_subject: Optional[str] = None,
    ) -> list[TenantConfig]:
        """Return tenant configs matching the optional filters."""
        tier_value = None
        if tier is not None:
            tier_value = cls._validate_choice("tier", tier, TIERS)
        owner_value = owner_subject.strip() if owner_subject else None

        tenants: list[TenantConfig] = []
        base = cls._base_dir()
        if not base.exists():
            return tenants

        for path in base.iterdir():
            if not path.is_dir():
                continue
            config_path = path / _CONFIG_NAME
            if not config_path.exists():
                continue
            try:
                cfg = cls.get(path.name)
            except TenantError:
                continue
            if active_only and not cfg.active:
                continue
            if tier_value is not None and cfg.tier != tier_value:
                continue
            if owner_value is not None and cfg.owner_subject != owner_value:
                continue
            tenants.append(cfg)
        tenants.sort(key=lambda item: item.tenant_id)
        return tenants

    @classmethod
    def get_by_owner(cls, owner_subject: str) -> Optional[TenantConfig]:
        """Return the tenant owned by a subject, if one exists."""
        owner_value = (owner_subject or "").strip()
        if not owner_value:
            return None
        tenants = cls.list_tenants(owner_subject=owner_value)
        return tenants[0] if tenants else None

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        display_name: str,
        tier: str = "starter",
        scopes: Optional[list[str]] = None,
        algorithm_mode: str = "auto",
        rate_limit_rpm: int = 60,
        notes: str = "",
        owner_subject: Optional[str] = None,
    ) -> TenantConfig:
        """Create a new tenant config on disk."""
        tenant_id = cls._validate_tenant_id(tenant_id)
        tier = cls._validate_choice("tier", tier, TIERS)
        algorithm_mode = cls._validate_choice(
            "algorithm_mode",
            algorithm_mode,
            ("auto", "explicit", "implicit", "hybrid"),
        )
        scopes = cls._validate_scopes(scopes)
        if rate_limit_rpm < 1:
            raise TenantValidationError("rate_limit_rpm must be positive")

        owner_value = owner_subject.strip() if owner_subject else None

        with cls._lock:
            if owner_value and cls.get_by_owner(owner_value):
                raise TenantAlreadyExistsError(f"User '{owner_value}' already has a tenant")

            path = cls._config_path(tenant_id)
            if path.exists():
                raise TenantAlreadyExistsError(f"Tenant '{tenant_id}' already exists")

            raw_key = secrets.token_urlsafe(32)
            cfg = TenantConfig(
                tenant_id=tenant_id,
                display_name=str(display_name or tenant_id).strip() or tenant_id,
                tier=tier,
                scopes=scopes,
                algorithm_mode=algorithm_mode,
                rate_limit_rpm=rate_limit_rpm,
                notes=notes or "",
                active=True,
                owner_subject=owner_value,
                api_key=raw_key,
                api_key_hash=cls._hash_api_key(raw_key),
            )
            cls._persist(cfg)
            return cls._store_in_cache(cfg)

    @classmethod
    def update(
        cls,
        tenant_id: str,
        *,
        display_name: Optional[str] = None,
        tier: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        algorithm_mode: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        active: Optional[bool] = None,
        notes: Optional[str] = None,
        last_activated_at: Optional[float] = None,
    ) -> TenantConfig:
        """Update an existing tenant configuration."""
        cfg = cls.get(tenant_id)
        if display_name is not None:
            cfg.display_name = display_name.strip() or cfg.display_name
        if tier is not None:
            cfg.tier = cls._validate_choice("tier", tier, TIERS)
        if scopes is not None:
            cfg.scopes = cls._validate_scopes(scopes)
        if algorithm_mode is not None:
            cfg.algorithm_mode = cls._validate_choice(
                "algorithm_mode",
                algorithm_mode,
                ("auto", "explicit", "implicit", "hybrid"),
            )
        if rate_limit_rpm is not None:
            if rate_limit_rpm < 1:
                raise TenantValidationError("rate_limit_rpm must be positive")
            cfg.rate_limit_rpm = rate_limit_rpm
        if active is not None:
            cfg.active = bool(active)
        if notes is not None:
            cfg.notes = notes
        if last_activated_at is not None:
            cfg.last_activated_at = last_activated_at

        cls._write_config(tenant_id, cfg)
        return cls._store_in_cache(cfg)

    @classmethod
    def activate_tenant(cls, tenant_id: str, actor: str) -> None:
        """
        Mark a tenant as active and log the activation event.
        Prevents duplicate records if the last activation was by the same actor recently.
        """
        tenant_id = cls._validate_tenant_id(tenant_id)
        now = time.time()

        # 1. Ensure tenant is active and set last_activated_at
        cls.update(tenant_id, active=True, last_activated_at=now)
        log.info("Tenant '%s' automatically activated by '%s'", tenant_id, actor)

        # 2. Log activation to history file
        history_path = cls._tenant_dir(tenant_id) / "activation_history.json"
        
        history = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Failed to read activation history for '%s': %s", tenant_id, exc)

        if not isinstance(history, list):
            history = []

        # Prevent duplicate activation records: 
        # Skip if last record is by the same actor and within the last 5 minutes
        if history:
            last_entry = history[0]
            last_actor = last_entry.get("user")
            last_time = last_entry.get("timestamp", 0)
            if last_actor == actor and (now - last_time) < 300:
                return

        new_entry = {
            "user": actor,
            "timestamp": now,
            "status": "active",
            "source": "login_auto_activation"
        }
        
        history.insert(0, new_entry)
        
        try:
            tmp_path = history_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")
            tmp_path.replace(history_path)
        except Exception as exc:
            log.warning("Failed to save activation history for '%s': %s", tenant_id, exc)

    @classmethod
    def delete(cls, tenant_id: str, delete_data: bool = False) -> None:
        """Delete a tenant config and optionally its data directory."""
        tenant_id = cls._validate_tenant_id(tenant_id)
        tenant_dir = cls._tenant_dir(tenant_id)
        path = cls._config_path(tenant_id)

        if not path.exists() and not tenant_dir.exists():
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")

        if delete_data:
            shutil.rmtree(tenant_dir, ignore_errors=False)
        else:
            if path.exists():
                path.unlink()
        cls.invalidate_cache(tenant_id)

    @classmethod
    def rotate_api_key(cls, tenant_id: str) -> str:
        """Generate and persist a new raw API key for a tenant."""
        cfg = cls.get(tenant_id)
        raw_key = secrets.token_urlsafe(32)
        cfg.api_key_hash = cls._hash_api_key(raw_key)
        cfg.api_key = raw_key
        cls._write_config(tenant_id, cfg)
        cls._store_in_cache(cfg)
        return raw_key

    @classmethod
    def resolve_api_key(cls, raw_key: str) -> Optional[TenantConfig]:
        """Resolve a raw API key to the owning tenant config."""
        if not raw_key:
            return None
        key_hash = cls._hash_api_key(raw_key)
        for cfg in cls.list_tenants():
            if cfg.api_key_hash and secrets.compare_digest(cfg.api_key_hash, key_hash):
                return cfg
        return None
