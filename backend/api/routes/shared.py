"""
Shared runtime state and helper utilities for all route modules.

Centralises:
- In-memory job store (_jobs)
- Per-tenant runtime state (_runtime_state)
- Per-tenant serving pipeline cache (_serving_cache)
- Thread-pool job submission (_submit_job)
- Tenant-aware service factories (_strategy_service, _direct_recommender)
- Common serialisation / validation helpers
"""
from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException

from backend.config.settings import TOP_K_ALLOWED, ALGORITHM_MODES, BASE_TENANTS_DIR
from backend.core.deployment.control_plane import DeploymentControlPlane
from backend.core.models.model_loader import ModelLoader
from backend.core.models.model_registry import ModelRegistry
from backend.core.pipeline.serving_pipeline import ServingPipeline
from backend.core.recommendation.recommender_engine import RecommenderEngine
from backend.core.recommendation.strategy_service import RecommendationStrategyService
from backend.core.tenants.manager import TenantManager
from backend.sdk.webhooks.push import emit_job_completed, emit_job_failed
from backend.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global in-memory stores (process lifetime)
# ---------------------------------------------------------------------------

# Async job registry — keyed by job_id (str)
_jobs: dict[str, dict] = {}

# Per-tenant mutable runtime state
# Keys inside each tenant dict: train_df, last_result, serving_im
_runtime_state: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"train_df": None, "last_result": None, "serving_im": None}
)

# Per-tenant serving pipeline cache — keyed by (tenant_id, algorithm)
_serving_cache: dict[str, dict[str, ServingPipeline]] = defaultdict(dict)

_executor = ThreadPoolExecutor(max_workers=4)

# Per-tenant lazy service instances
_registries: dict[str, ModelRegistry] = {}
_loaders: dict[str, ModelLoader] = {}
_recommenders: dict[str, RecommenderEngine] = {}
_strategy_services: dict[str, RecommendationStrategyService] = {}
_job_history_lock = Lock()
_job_history_cache: dict[str, list[dict]] = {}
_job_history_mtimes: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save_last_result(tenant_id: str, result_dict: dict) -> None:
    """Persist the serialised training result to the tenant's directory."""
    try:
        path = _tenant_state_dir(tenant_id) / "last_training_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
        log.debug("Saved last_result for tenant %s to %s", tenant_id, path)
    except Exception as e:
        log.warning("Failed to save last_result for tenant %s: %s", tenant_id, e)


def _load_last_result(tenant_id: str) -> Optional[dict]:
    """Load the serialised training result from the tenant's directory."""
    try:
        path = _tenant_state_dir(tenant_id) / "last_training_result.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.debug("No last_result found for tenant %s: %s", tenant_id, e)
        return None


class _DictWrapper:
    """Wraps a dictionary to allow attribute access (for compatibility with getattr)."""
    def __init__(self, d: dict):
        """Materialise dictionary keys as attributes for legacy callers."""
        self.__dict__.update(d)

    def get(self, key, default=None):
        """Mirror dict.get so wrapped payloads behave like plain mappings."""
        return self.__dict__.get(key, default)


def _get_last_result(tenant_id: str) -> Optional[Any]:
    """Get the last training result for a tenant, from memory or disk."""
    state = _runtime_state[tenant_id]
    if state["last_result"] is not None:
        return state["last_result"]

    data = _load_last_result(tenant_id)
    if data:
        wrapped = _DictWrapper(data)
        state["last_result"] = wrapped
        return wrapped
    return None


def _tenant_state_dir(tenant_id: str) -> Path:
    """Resolve the tenant-specific state directory used by runtime persistence."""
    base_dir = getattr(TenantManager, "_base", None)
    if isinstance(base_dir, Path):
        return base_dir / tenant_id
    if base_dir:
        return Path(base_dir) / tenant_id
    return Path(BASE_TENANTS_DIR) / tenant_id


def _job_history_path(tenant_id: str) -> Path:
    """Return the persisted training-job history file for a tenant."""
    return _tenant_state_dir(tenant_id) / "training_jobs.json"


def _job_history_file_mtime(path: Path) -> float:
    """Read the last-modified timestamp for a history file, defaulting to zero."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _clone_jobs(jobs: list[dict]) -> list[dict]:
    """Copy job payloads so callers cannot mutate the shared cache in place."""
    return [dict(job) for job in jobs]


def _sort_jobs_in_place(jobs: list[dict]) -> list[dict]:
    """Sort job records newest-first using their creation timestamp."""
    jobs.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
    return jobs


def _normalise_job_status(status: Optional[str]) -> str:
    """Collapse legacy and user-facing job states into the canonical API vocabulary."""
    value = str(status or "pending").strip().lower()
    return {
        "done": "completed",
        "error": "failed",
    }.get(value, value if value in {"pending", "running", "completed", "failed"} else "pending")


def _public_job(job: dict) -> dict:
    """Return a sanitised job payload with stable public-facing status values."""
    payload = dict(job)
    payload["status"] = _normalise_job_status(payload.get("status"))
    return _sanitise(payload)


def _read_job_history(path: Path) -> list[dict]:
    """Load persisted job history from disk, ignoring malformed payloads safely."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        log.warning("Failed to read training job history from %s: %s", path, exc)
        return []

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_job_history(path: Path, jobs: list[dict]) -> None:
    """Persist job history atomically so readers never observe partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)
    tmp_path.replace(path)


def _load_job_history_locked(tenant_id: str) -> list[dict]:
    """Read cached job history under the caller-held lock, refreshing on mtime changes."""
    path = _job_history_path(tenant_id)
    current_mtime = _job_history_file_mtime(path)
    cached = _job_history_cache.get(tenant_id)
    if cached is not None and _job_history_mtimes.get(tenant_id) == current_mtime:
        return _clone_jobs(cached)

    jobs = _sort_jobs_in_place(_read_job_history(path))
    _job_history_cache[tenant_id] = _clone_jobs(jobs)
    _job_history_mtimes[tenant_id] = current_mtime
    return _clone_jobs(jobs)


def _store_job_history_locked(tenant_id: str, path: Path, jobs: list[dict]) -> None:
    """Persist and refresh the in-memory job-history cache while holding the lock."""
    ordered = _sort_jobs_in_place(jobs)
    if ordered:
        _write_job_history(path, ordered)
        _job_history_cache[tenant_id] = _clone_jobs(ordered)
        _job_history_mtimes[tenant_id] = _job_history_file_mtime(path)
        return

    path.unlink(missing_ok=True)
    _job_history_cache[tenant_id] = []
    _job_history_mtimes[tenant_id] = 0.0


def _persist_job_record(job: dict) -> None:
    """Upsert a single job into the tenant's persisted job-history log."""
    tenant_id = job.get("tenant_id")
    job_id = job.get("job_id")
    if not tenant_id or not job_id:
        return

    path = _job_history_path(tenant_id)
    public_job = _public_job(job)
    with _job_history_lock:
        jobs = _load_job_history_locked(tenant_id)
        for index, existing in enumerate(jobs):
            if existing.get("job_id") == job_id:
                jobs[index] = public_job
                break
        else:
            jobs.append(public_job)
        _store_job_history_locked(tenant_id, path, jobs)


def _load_job_history(tenant_id: str) -> list[dict]:
    """Return a tenant's job history using the shared cache and lock discipline."""
    with _job_history_lock:
        return _load_job_history_locked(tenant_id)


def _get_job_for_tenant(tenant_id: str, job_id: str) -> Optional[dict]:
    """Fetch a job from live memory first, then fall back to persisted history."""
    live_job = _jobs.get(job_id)
    if live_job and live_job.get("tenant_id") == tenant_id:
        return _public_job(live_job)

    for job in _load_job_history(tenant_id):
        if job.get("job_id") == job_id:
            return job
    return None


def _list_jobs_for_tenant(tenant_id: str, limit: int = 25) -> list[dict]:
    """Merge live and persisted jobs for a tenant and return the newest subset."""
    merged: dict[str, dict] = {
        str(job.get("job_id")): job
        for job in _load_job_history(tenant_id)
        if job.get("job_id")
    }

    for job in _jobs.values():
        if job.get("tenant_id") == tenant_id and job.get("job_id"):
            merged[str(job["job_id"])] = _public_job(job)

    jobs = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("created_at") or 0),
            float(item.get("started_at") or 0),
        ),
        reverse=True,
    )
    return jobs[:limit]


def _delete_job_for_tenant(tenant_id: str, job_id: str) -> Optional[dict]:
    """Delete a completed or failed job while refusing to remove active work."""
    live_job = _jobs.get(job_id)
    if live_job and live_job.get("tenant_id") == tenant_id:
        live_status = _normalise_job_status(live_job.get("status"))
        if live_status in {"pending", "running"}:
            return None

    path = _job_history_path(tenant_id)
    deleted_job = None
    with _job_history_lock:
        jobs = _load_job_history_locked(tenant_id)
        remaining_jobs = []
        for job in jobs:
            if job.get("job_id") == job_id:
                deleted_job = job
                continue
            remaining_jobs.append(job)

        if deleted_job is None:
            return None

        _store_job_history_locked(tenant_id, path, remaining_jobs)

    if live_job and live_job.get("tenant_id") == tenant_id:
        _jobs.pop(job_id, None)

    return deleted_job


# ---------------------------------------------------------------------------
# Per-tenant service factories
# ---------------------------------------------------------------------------


def _get_registry(tenant_id: str) -> ModelRegistry:
    """Return the lazily cached model registry for a tenant."""
    if tenant_id not in _registries:
        _registries[tenant_id] = ModelRegistry(tenant_id=tenant_id)
    return _registries[tenant_id]


def _get_loader(tenant_id: str) -> ModelLoader:
    """Return the shared model loader instance used across tenant requests."""
    if tenant_id not in _loaders:
        _loaders[tenant_id] = ModelLoader()
    return _loaders[tenant_id]


def _deployment_control_plane(tenant_id: str) -> DeploymentControlPlane:
    """Build a tenant-scoped deployment control plane on demand."""
    return DeploymentControlPlane(
        tenant_id=tenant_id,
        registry=_get_registry(tenant_id),
        loader=_get_loader(tenant_id),
    )


def _direct_recommender(tenant_id: str) -> RecommenderEngine:
    """Return the tenant recommender instance backed by the shared model loader."""
    if tenant_id not in _recommenders:
        loader = _get_loader(tenant_id)
        _recommenders[tenant_id] = RecommenderEngine(loader, interaction_matrix=None, tenant_id=tenant_id)
    return _recommenders[tenant_id]


def _strategy_service(tenant_id: str) -> RecommendationStrategyService:
    """Return the tenant's strategy service with shared loader and registry dependencies."""
    if tenant_id not in _strategy_services:
        registry = _get_registry(tenant_id)
        loader = _get_loader(tenant_id)
        recommender = _direct_recommender(tenant_id)
        _strategy_services[tenant_id] = RecommendationStrategyService(
            registry, loader, recommender, tenant_id=tenant_id
        )
    return _strategy_services[tenant_id]


# ---------------------------------------------------------------------------
# Serving pipeline helper
# ---------------------------------------------------------------------------


def _get_serving_pipeline(
    tenant_id: str,
    algorithm: str,
    runtime: dict,
) -> ServingPipeline:
    """Return a fresh or cached serving pipeline aligned with the promoted model."""
    cache = _serving_cache[tenant_id]

    # v3: Check for cache staleness by comparing against promoted model ID
    registry = _get_registry(tenant_id)
    promoted = registry.get_promoted(algorithm)
    promoted_id = promoted.model_id if promoted else None

    if algorithm in cache:
        cached_sp = cache[algorithm]
        if cached_sp.model_id == promoted_id:
            return cached_sp
        log.info("Serving pipeline stale for tenant=%s algo=%s | reloading", tenant_id, algorithm)
        cache.pop(algorithm)

    import pandas as pd

    train_df = runtime.get(tenant_id, {}).get("train_df")
    try:
        if isinstance(train_df, pd.DataFrame):
            sp = ServingPipeline.from_registry(
                algorithm=algorithm,
                train_df=train_df,
                tenant_id=tenant_id,
            )
        else:
            sp = ServingPipeline.from_registry(
                algorithm=algorithm,
                train_df=None,
                tenant_id=tenant_id,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Serving pipeline not ready for algorithm '{algorithm}': {exc}. "
                "Train and promote a model first."
            ),
        )
    cache[algorithm] = sp
    return sp


def _default_algorithm(tenant_id: str, runtime: dict) -> str:
    """Resolve the active deployed algorithm or raise if nothing is deployable yet."""
    active_record = _deployment_control_plane(tenant_id).get_active_record()
    if active_record is None:
        raise HTTPException(
            status_code=404,
            detail="No active deployment found. Train and promote a model first.",
        )
    return active_record.algorithm


# ---------------------------------------------------------------------------
# Async job submission
# ---------------------------------------------------------------------------


def _submit_job(tenant_id: str, fn, *args, job_meta: Optional[dict] = None, **kwargs) -> str:
    """Submit background work, persist lifecycle state, and return the public job id."""
    job_id = str(uuid.uuid4())[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    if job_meta:
        _jobs[job_id].update(_sanitise(job_meta))
    _persist_job_record(_jobs[job_id])

    def _run():
        """Execute the background task and fan out lifecycle webhooks safely."""
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()
        _persist_job_record(_jobs[job_id])
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            _jobs[job_id]["finished_at"] = time.time()
            _persist_job_record(_jobs[job_id])

            # Fire job.completed webhook if the tenant has one registered
            try:
                cfg = TenantManager.get(tenant_id)
                if cfg.webhook_url and cfg.webhook_secret:
                    emit_job_completed(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        result=_sanitise(result) if isinstance(result, dict) else {},
                        webhook_url=cfg.webhook_url,
                        webhook_secret=cfg.webhook_secret,
                    )
            except Exception as wh_exc:
                log.warning("Webhook emit failed for job %s: %s", job_id, wh_exc)

        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = time.time()
            _persist_job_record(_jobs[job_id])
            log.error("Job %s (tenant=%s) failed: %s", job_id, tenant_id, exc, exc_info=True)

            # Fire job.failed webhook if the tenant has one registered
            try:
                cfg = TenantManager.get(tenant_id)
                if cfg.webhook_url and cfg.webhook_secret:
                    emit_job_failed(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        error=str(exc),
                        webhook_url=cfg.webhook_url,
                        webhook_secret=cfg.webhook_secret,
                    )
            except Exception as wh_exc:
                log.warning("Webhook emit failed for job %s: %s", job_id, wh_exc)

    _executor.submit(_run)
    return job_id


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_top_choice(name: str, value: int, allowed: tuple) -> None:
    """Raise a FastAPI validation error when a bounded integer option is invalid."""
    if int(value) not in allowed:
        raise HTTPException(status_code=422, detail=f"{name} must be one of {allowed}")


def _validate_algorithm_mode(value: str) -> None:
    """Validate algorithm mode values against the configured product contract."""
    if (value or "").lower().strip() not in ALGORITHM_MODES:
        raise HTTPException(
            status_code=422, detail=f"algorithm_mode must be one of {ALGORITHM_MODES}"
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _sanitise(obj):
    """Recursively sanitise an object for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass
    return obj


def _serialise_training_result(result, tenant_id: str = "") -> dict:
    """Convert a training result into the stable API payload saved for later reads."""
    lb = result.report.leaderboard()
    rows = lb.to_dict(orient="records") if not lb.empty else []
    svc = _strategy_service(tenant_id) if tenant_id else None
    options = svc.recommendation_options(last_result=result) if svc else {}

    payload = {
        "tenant_id": tenant_id,
        "best_algorithm": result.best_algorithm,
        "best_params": result.best_params,
        "best_model_id": result.best_model_id,
        "elapsed_s": round(result.elapsed_s, 2),
        "all_model_ids": getattr(result, "all_model_ids", []),
        "leaderboard": rows,
        "resolved_mode": result.resolved_mode,
        "feedback_profile": result.feedback_profile,
        "ranking_logic": result.ranking_logic,
        "optuna_note": result.optuna_note,
        "optuna_policy": getattr(result, "optuna_policy", {}),
        "top_model_recommendations": result.top_model_recommendations,
        "model_selection_policy": getattr(result, "model_selection_policy", {}),
        "recommendation_options": options,
        "top_model_candidates": (
            (getattr(result, "model_selection_policy", {}) or {}).get("selected_models")
            or (result.top_model_recommendations or [])
        ),
        "tuning": [
            {
                "algorithm": t.algorithm,
                "best_value": round(t.best_value, 4) if isinstance(t.best_value, (int, float)) else None,
                "metric": t.metric_name,
                "best_params": t.best_params,
                "n_trials": t.n_trials,
                "status": t.status,
                "fallback_reason": t.fallback_reason,
                "trial_budget": t.trial_budget,
            }
            for t in result.tuning_results
        ],
    }
    return _sanitise(payload)
