"""
core/models/model_registry.py – Tenant-scoped model registry and versioned persistence.

Every model artifact and the registry index file live under:

    tenants/{tenant_id}/models/

No model from one tenant is ever visible to, or loadable by, another tenant.

v3 Changes vs v2:
  - __init__ now requires `tenant_id`; `store_dir` is derived automatically as
    tenants/{tenant_id}/models/ relative to BASE_TENANTS_DIR (from settings).
  - All read/write paths are resolved inside the tenant's directory.
  - Path traversal guard added in _resolve_file_path() — a stored absolute path
    that escapes the tenant directory is rejected with a SecurityError.
  - `tenant_id` is recorded inside every ModelRecord for auditability.
  - New `tenant_id` field on ModelRecord (backward-compatible: defaults to "").
  - `_normalise_record` re-anchors legacy absolute paths to the current tenant
    store dir so records remain valid after a directory move / restore.
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.config.settings import BASE_TENANTS_DIR   # e.g. Path("tenants")
from backend.utils.logger import get_logger

log = get_logger(__name__)

_REGISTRY_FILE = "registry.json"
_MODELS_SUBDIR = "models"


class RegistrySecurityError(RuntimeError):
    """Raised when a stored file path attempts to escape the tenant directory."""


@dataclass
class ModelRecord:
    model_id:    str
    algorithm:   str
    version:     int
    metrics:     dict
    params:      dict
    is_implicit: bool
    created_at:  float
    tenant_id:   str  = ""          # v3: which tenant owns this record
    promoted:    bool = False
    file_path:   str  = ""
    notes:       str  = ""
    tags:        list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the model metadata record for JSON persistence."""
        return asdict(self)


class ModelRegistry:
    """
    Versioned, tenant-scoped model registry.

    Stores model .pkl files and a JSON index under:
        tenants/{tenant_id}/models/

    Each ModelRegistry instance is bound to exactly one tenant at construction
    time, so it is impossible to accidentally cross tenant boundaries.

    Usage:
        registry = ModelRegistry(tenant_id="acme-corp")

        model_id = registry.save(
            model=trained_model,
            algorithm="svd",
            metrics={"rmse": 0.91},
            params={"factors": 64},
        )

        registry.promote(model_id)
        record = registry.get_promoted("svd")
        model  = registry.load(record.model_id)
    """

    def __init__(self, tenant_id: str, base_dir: Path | str | None = None):
        """
        Args:
            tenant_id:  The tenant this registry is scoped to. Must be a
                        non-empty, filesystem-safe string (e.g. "acme-corp").
            base_dir:   Override the tenants root directory (useful in tests).
                        Defaults to BASE_TENANTS_DIR from settings.
        """
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string.")

        self.tenant_id = tenant_id

        from backend.core.tenants.manager import TenantManager

        resolved_base_dir = base_dir or getattr(TenantManager, "_base", None) or BASE_TENANTS_DIR
        root = Path(resolved_base_dir).expanduser().resolve()
        # tenants/{tenant_id}/models/
        self.store_dir: Path = (root / tenant_id / _MODELS_SUBDIR).resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.store_dir / _REGISTRY_FILE
        self._index: dict[str, dict] = self._load_index()

        log.debug(
            "ModelRegistry initialised | tenant=%s store=%s",
            self.tenant_id, self.store_dir,
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(
        self,
        model: Any,
        algorithm: str,
        metrics: dict,
        params: dict,
        is_implicit: bool = False,
        notes: str = "",
        tags: list | None = None,
    ) -> str:
        """
        Persist a trained model to disk and register it.

        Returns:
            model_id: Short unique ID for the saved model.
        """
        self.refresh()

        model_id = str(uuid.uuid4())[:12]
        version  = self._next_version(algorithm)
        filename = f"{algorithm.lower().replace(' ', '_')}_v{version}_{model_id}.pkl"
        path     = (self.store_dir / filename).resolve()

        # Safety: ensure destination stays inside tenant directory
        self._assert_within_store(path)

        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        record = ModelRecord(
            model_id=model_id,
            algorithm=algorithm,
            version=version,
            metrics=metrics,
            params=params,
            is_implicit=is_implicit,
            created_at=time.time(),
            tenant_id=self.tenant_id,
            file_path=str(path),
            notes=notes,
            tags=tags or [],
        )
        self._index[model_id] = record.to_dict()
        self._save_index()

        log.info(
            "Model saved | tenant=%s id=%s algo=%s version=%d path=%s",
            self.tenant_id, model_id, algorithm, version, path,
        )
        return model_id

    def promote(self, model_id: str) -> None:
        """
        Mark *model_id* as the canonical production model for this tenant.
        Demotes every other promoted model so only one deployment remains active.
        """
        self.refresh()
        record = self._get_record(model_id)
        algorithm = record["algorithm"]

        for rid, rec in self._index.items():
            if rec.get("promoted"):
                self._index[rid]["promoted"] = False

        self._index[model_id]["promoted"] = True
        self._save_index()

        log.info(
            "Model promoted as active deployment | tenant=%s id=%s algo=%s",
            self.tenant_id, model_id, algorithm,
        )

    def delete(self, model_id: str, delete_file: bool = True) -> None:
        """Remove a model from the registry (and optionally from disk)."""
        self.refresh()
        record = self._get_record(model_id)

        if delete_file:
            path = self._resolve_file_path(record["file_path"])
            if path.exists():
                path.unlink()

        del self._index[model_id]
        self._save_index()
        log.info("Model deleted | tenant=%s id=%s", self.tenant_id, model_id)

    # ── Read ───────────────────────────────────────────────────────────────────

    def load(self, model_id: str) -> Any:
        """Deserialise and return the model object for *model_id*."""
        record = self._get_record(model_id)
        path   = self._resolve_file_path(record["file_path"])

        self._assert_within_store(path)   # path-traversal guard on load too

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found for tenant='{self.tenant_id}': {path}"
            )
        with open(path, "rb") as f:
            model = pickle.load(f)

        log.info(
            "Model loaded | tenant=%s id=%s algo=%s",
            self.tenant_id, model_id, record["algorithm"],
        )
        return model

    def get_promoted(self, algorithm: str) -> Optional[ModelRecord]:
        """Return the promoted ModelRecord for *algorithm*, or None."""
        self.refresh()
        for record in self._index.values():
            if record["algorithm"] == algorithm and record.get("promoted"):
                return ModelRecord(**self._normalise_record(record))
        return None

    def get(self, model_id: str) -> Optional[ModelRecord]:
        """Return the ModelRecord for *model_id*, or None if not found."""
        self.refresh()
        record = self._index.get(model_id)
        if not record:
            return None
        return ModelRecord(**self._normalise_record(record))

    def latest_for_algorithm(self, algorithm: str) -> Optional[ModelRecord]:
        """Return the most-recently saved ModelRecord for *algorithm*."""
        records = self.list_models(algorithm=algorithm)
        return records[0] if records else None

    def list_models(
        self,
        algorithm: Optional[str] = None,
        promoted_only: bool = False,
    ) -> list[ModelRecord]:
        """
        List models in this tenant's registry.

        Args:
            algorithm:     If given, filter to this algorithm only.
            promoted_only: If True, return only promoted models.

        Returns:
            List of ModelRecord sorted newest-first.
        """
        self.refresh()
        records = [
            ModelRecord(**self._normalise_record(v))
            for v in self._index.values()
        ]
        if algorithm:
            records = [r for r in records if r.algorithm == algorithm]
        if promoted_only:
            records = [r for r in records if r.promoted]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def refresh(self) -> None:
        """Re-read the index from disk (picks up external changes)."""
        self._index = self._load_index()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_record(self, model_id: str) -> dict:
        self.refresh()
        record = self._index.get(model_id)
        if not record:
            raise KeyError(
                f"Model ID '{model_id}' not found in registry for "
                f"tenant='{self.tenant_id}'."
            )
        return self._normalise_record(record)

    def _next_version(self, algorithm: str) -> int:
        versions = [
            r["version"]
            for r in self._index.values()
            if r["algorithm"] == algorithm
        ]
        return max(versions, default=0) + 1

    def _load_index(self) -> dict:
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path) as f:
                raw = json.load(f)
            return {
                model_id: self._normalise_record(record)
                for model_id, record in raw.items()
            }
        except Exception as e:
            log.warning(
                "Registry index corrupted for tenant=%s, starting fresh: %s",
                self.tenant_id, e,
            )
            return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _normalise_record(self, record: dict) -> dict:
        """
        Ensure every record field has the right type and that file_path is
        re-anchored to the current store_dir.

        This handles two legacy cases:
          1. Absolute paths stored by v2 (global model_store/) — re-anchored to
             the tenant's own store dir using just the filename.
          2. Missing tenant_id field in older records — back-filled from self.
        """
        out = dict(record)

        # Back-fill tenant_id for records saved before this field existed
        if not out.get("tenant_id"):
            out["tenant_id"] = self.tenant_id

        out["file_path"] = str(self._resolve_file_path(out.get("file_path", "")))
        return out

    def _resolve_file_path(self, raw_path: str) -> Path:
        """
        Turn a stored path string into an absolute Path inside store_dir.

        Rules (in order):
          1. Empty → return store_dir itself (sentinel, not a real file).
          2. Absolute path already under store_dir → use as-is.
          3. Absolute path outside store_dir → raise RegistrySecurityError.
             (v2 re-anchoring removed — migration complete; this branch was
             a path-traversal vulnerability: /etc/passwd would pass through.)
          4. Relative path → resolve relative to store_dir.
        """
        if not raw_path:
            return self.store_dir

        path = Path(raw_path)

        if path.is_absolute():
            try:
                path.relative_to(self.store_dir)
                return path          # already inside tenant dir
            except ValueError:
                # Absolute path outside tenant store — this is a security violation.
                # Previously this branch re-anchored legacy v2 paths by filename,
                # but that silently accepted traversal payloads like /etc/passwd.
                # All v2 migration is complete; any absolute path outside store_dir
                # must now be rejected unconditionally.
                raise RegistrySecurityError(
                    f"Absolute path '{raw_path}' is outside the tenant store directory "
                    f"'{self.store_dir}' (tenant='{self.tenant_id}'). "
                    "Possible path-traversal attempt — record rejected."
                )

        return (self.store_dir / path).resolve()

    def _assert_within_store(self, path: Path) -> None:
        """
        Raise RegistrySecurityError if *path* is not inside store_dir.
        Guards against path-traversal attacks in saved file_path values.
        """
        try:
            path.resolve().relative_to(self.store_dir)
        except ValueError:
            raise RegistrySecurityError(
                f"Path '{path}' is outside the tenant store directory "
                f"'{self.store_dir}' (tenant='{self.tenant_id}'). "
                "Possible path-traversal attempt — record rejected."
            )
