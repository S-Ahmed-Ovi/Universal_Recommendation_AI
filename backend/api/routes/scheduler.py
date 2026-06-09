"""
Scheduler routes — automated model retraining

POST   /scheduler/schedules          — create a retraining schedule
GET    /scheduler/schedules          — list all schedules for the tenant
GET    /scheduler/schedules/{id}     — get a specific schedule
PATCH  /scheduler/schedules/{id}     — update a schedule (enable/disable, change interval)
DELETE /scheduler/schedules/{id}     — delete a schedule
POST   /scheduler/schedules/{id}/trigger — manually trigger a scheduled retrain now
GET    /scheduler/schedules/{id}/history — retrain job history for a schedule

Schedules are persisted per-tenant in:
    tenants/<tenant_id>/schedules/<schedule_id>.json

A background thread (SchedulerDaemon) wakes every minute, checks all active
schedules across all tenants, and submits a training job when one is due.

Auth scope required: training:write  (create / update / delete / trigger)
                     jobs:read       (list / get / history)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from backend.api.gateway.auth import require_auth
from backend.config.settings import (
    ALGORITHM_MODES,
    BASE_TENANTS_DIR,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_K_ALLOWED,
    TOP_MODEL_ALLOWED,
    TOP_N_MODELS,
)
from backend.core.data_processing.interaction_matrix import InteractionMatrixBuilder
from backend.core.pipeline.training_pipeline import TrainingConfig, TrainingPipeline
from backend.utils.logger import get_logger
from .shared import (
    _direct_recommender,
    _runtime_state,
    _save_last_result,
    _serialise_training_result,
    _serving_cache,
    _submit_job,
    _validate_algorithm_mode,
    _validate_top_choice,
)

log = get_logger(__name__)

router = APIRouter(tags=["Scheduler"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_INTERVAL_SECONDS = 3_600          # 1 hour — floor to prevent abuse
SCHEDULE_STORE_DIR_NAME = "schedules"
HISTORY_MAX_ENTRIES = 50              # kept per schedule

IntervalUnit = Literal["hours", "days", "weeks"]

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _schedules_dir(tenant_id: str) -> Path:
    """Return the tenant schedule directory, creating it lazily when needed."""
    p = Path(BASE_TENANTS_DIR) / tenant_id / SCHEDULE_STORE_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _schedule_path(tenant_id: str, schedule_id: str) -> Path:
    """Resolve the JSON file path for a single tenant schedule."""
    return _schedules_dir(tenant_id) / f"{schedule_id}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Persist JSON atomically so readers never observe a partially written schedule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_schedule_path(path: Path) -> Optional[dict]:
    """Read a schedule file defensively, returning None for missing or invalid payloads."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read schedule file %s: %s", path, exc)
        return None


def _write_schedule(tenant_id: str, schedule: dict) -> None:
    """Persist a schedule document under the tenant's schedule store."""
    path = _schedule_path(tenant_id, schedule["id"])
    _write_json_atomic(path, schedule)


def _read_schedule(tenant_id: str, schedule_id: str) -> Optional[dict]:
    """Load a single schedule by id for the given tenant."""
    return _read_schedule_path(_schedule_path(tenant_id, schedule_id))


def _list_schedules(tenant_id: str) -> list[dict]:
    """Load every persisted schedule for a tenant from disk."""
    d = _schedules_dir(tenant_id)
    schedules = []
    for p in sorted(d.glob("*.json")):
        schedule = _read_schedule_path(p)
        if schedule is not None:
            schedules.append(schedule)
    return schedules


def _delete_schedule_file(tenant_id: str, schedule_id: str) -> bool:
    """Delete a persisted schedule document if it exists."""
    path = _schedule_path(tenant_id, schedule_id)
    if path.exists():
        path.unlink()
        return True
    return False


def _append_history(tenant_id: str, schedule_id: str, entry: dict) -> None:
    """Append a job-run record to the schedule's history (capped at HISTORY_MAX_ENTRIES)."""
    schedule = _read_schedule(tenant_id, schedule_id)
    if schedule is None:
        return
    history: list[dict] = schedule.get("history", [])
    history.append(entry)
    if len(history) > HISTORY_MAX_ENTRIES:
        history = history[-HISTORY_MAX_ENTRIES:]
    schedule["history"] = history
    _write_schedule(tenant_id, schedule)


# ---------------------------------------------------------------------------
# Interval → seconds conversion
# ---------------------------------------------------------------------------


def _interval_to_seconds(value: int, unit: IntervalUnit) -> int:
    """Convert the API interval tuple into the daemon's canonical seconds value."""
    multipliers = {"hours": 3_600, "days": 86_400, "weeks": 604_800}
    return value * multipliers[unit]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DataSourceFile(BaseModel):
    """Re-use a previously uploaded dataset path (server-side path)."""
    type: Literal["file"] = "file"
    path: str = Field(..., description="Absolute server-side path to a CSV/Excel/JSON/Parquet file")
    format: str = Field("auto", description="File format hint (auto, csv, json, parquet, …)")


class DataSourceSQL(BaseModel):
    """SQL data source for scheduled retraining."""
    type: Literal["sql"] = "sql"
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    sql: str


class CreateScheduleRequest(BaseModel):
    """Payload for creating a recurring tenant retraining schedule."""
    name: str = Field(..., min_length=1, max_length=120, description="Human-readable schedule name")
    interval_value: int = Field(..., ge=1, description="Numeric part of the interval (e.g. 6 for '6 hours')")
    interval_unit: IntervalUnit = Field("hours", description="Unit: hours | days | weeks")
    data_source: DataSourceFile | DataSourceSQL = Field(
        ..., discriminator="type", description="Dataset to use for retraining"
    )
    top_k: int = TOP_K
    n_trials: int = OPTUNA_TRIALS_AUTO
    top_models: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE
    enabled: bool = True
    description: Optional[str] = Field(None, max_length=500)

    @field_validator("top_k")
    @classmethod
    def _chk_top_k(cls, v: int) -> int:
        """Reject unsupported recommendation cutoffs at request-validation time."""
        if v not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}")
        return v

    @field_validator("top_models")
    @classmethod
    def _chk_top_models(cls, v: int) -> int:
        """Reject unsupported ranked-model limits before the request reaches business logic."""
        if v not in TOP_MODEL_ALLOWED:
            raise ValueError(f"top_models must be one of {TOP_MODEL_ALLOWED}")
        return v

    @field_validator("algorithm_mode")
    @classmethod
    def _chk_mode(cls, v: str) -> str:
        """Normalize and validate the requested algorithm mode."""
        v = v.lower().strip()
        if v not in ALGORITHM_MODES:
            raise ValueError(f"algorithm_mode must be one of {ALGORITHM_MODES}")
        return v

    @field_validator("interval_value")
    @classmethod
    def _chk_interval(cls, v: int) -> int:
        """Catch obviously invalid interval values before unit-aware validation runs."""
        # Checked again against MIN_INTERVAL_SECONDS after unit is known,
        # but catch absurdly small int values early.
        if v < 1:
            raise ValueError("interval_value must be >= 1")
        return v


class PatchScheduleRequest(BaseModel):
    """Partial update payload for mutating an existing retraining schedule."""
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    interval_value: Optional[int] = Field(None, ge=1)
    interval_unit: Optional[IntervalUnit] = None
    enabled: Optional[bool] = None
    description: Optional[str] = Field(None, max_length=500)
    top_k: Optional[int] = None
    n_trials: Optional[int] = None
    top_models: Optional[int] = None
    algorithm_mode: Optional[str] = None

    @field_validator("top_k")
    @classmethod
    def _chk_top_k(cls, v):
        """Allow omitted values while validating any provided top_k choice."""
        if v is not None and v not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}")
        return v

    @field_validator("top_models")
    @classmethod
    def _chk_top_models(cls, v):
        """Allow omitted values while validating any provided top_models choice."""
        if v is not None and v not in TOP_MODEL_ALLOWED:
            raise ValueError(f"top_models must be one of {TOP_MODEL_ALLOWED}")
        return v

    @field_validator("algorithm_mode")
    @classmethod
    def _chk_mode(cls, v):
        """Allow omitted values while validating any provided algorithm mode."""
        if v is not None:
            v = v.lower().strip()
            if v not in ALGORITHM_MODES:
                raise ValueError(f"algorithm_mode must be one of {ALGORITHM_MODES}")
        return v


# ---------------------------------------------------------------------------
# Core: run a scheduled retraining job
# ---------------------------------------------------------------------------


def _run_scheduled_retrain(tenant_id: str, schedule: dict) -> str:
    """
    Submit an async training job for a schedule.
    Returns the job_id.
    """
    config = TrainingConfig(
        top_k=schedule["top_k"],
        n_tuning_trials=schedule["n_trials"],
        top_model_count=schedule["top_models"],
        algorithm_mode=schedule["algorithm_mode"],
    )

    ds = schedule["data_source"]
    source_type = ds.get("type", "file")

    if source_type == "file":
        file_path = ds["path"]
        file_format = ds.get("format", "auto")

        def _task():
            """Execute the file-backed scheduled retraining flow inside the job runner."""
            pipeline = TrainingPipeline(tenant_id, config=config)
            result = pipeline.run_from_file(file_path, file_format=file_format)
            _post_train(tenant_id, result)
            return _serialise_training_result(result, tenant_id=tenant_id)

        source_label = os.path.basename(file_path)

    else:  # sql
        from backend.config.database import SQLConfig
        sql_cfg = SQLConfig(
            dialect=ds["dialect"],
            host=ds["host"],
            port=ds["port"],
            database=ds["database"],
            username=ds["username"],
            password=ds["password"],
        )
        sql_query = ds["sql"]

        def _task():  # type: ignore[no-redef]
            """Execute the SQL-backed scheduled retraining flow inside the job runner."""
            pipeline = TrainingPipeline(tenant_id, config=config)
            result = pipeline.run_from_sql(sql_cfg, sql_query)
            _post_train(tenant_id, result)
            return _serialise_training_result(result, tenant_id=tenant_id)

        source_label = f"{ds['dialect']} / {ds['database']}"

    job_id = _submit_job(
        tenant_id,
        _task,
        job_meta={
            "algorithm_mode": schedule["algorithm_mode"],
            "n_trials": schedule["n_trials"],
            "source_label": source_label,
            "source_type": source_type,
            "top_k": schedule["top_k"],
            "top_models": schedule["top_models"],
            "triggered_by": "scheduler",
            "schedule_id": schedule["id"],
            "schedule_name": schedule.get("name", ""),
        },
    )
    return job_id


def _post_train(tenant_id: str, result: Any) -> None:
    """Update runtime state after a successful training run."""
    try:
        serialised = _serialise_training_result(result, tenant_id=tenant_id)
        _save_last_result(tenant_id, serialised)
        _runtime_state[tenant_id]["train_df"] = result.report.train.copy()
        _runtime_state[tenant_id]["last_result"] = result
        _runtime_state[tenant_id]["serving_im"] = InteractionMatrixBuilder().build(
            result.report.train
        )
        _direct_recommender(tenant_id).im = _runtime_state[tenant_id]["serving_im"]
        _serving_cache.pop(tenant_id, None)
    except Exception as exc:
        log.warning("_post_train failed for tenant %s: %s", tenant_id, exc)


# ---------------------------------------------------------------------------
# Scheduler daemon (singleton background thread)
# ---------------------------------------------------------------------------


class _SchedulerDaemon(threading.Thread):
    """
    Background thread that polls all tenant schedule files every 60 s and
    fires training jobs when one is due.
    """

    _instance: Optional[_SchedulerDaemon] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialise the daemon thread and its stop event."""
        super().__init__(daemon=True, name="SchedulerDaemon")
        self._stop_event = threading.Event()

    @classmethod
    def ensure_running(cls) -> None:
        """Start the singleton scheduler thread if it is not already alive."""
        with cls._lock:
            if cls._instance is None or not cls._instance.is_alive():
                cls._instance = cls()
                cls._instance.start()
                log.info("SchedulerDaemon started")

    def run(self) -> None:
        """Sleep-wake loop that continuously polls tenant schedule files."""
        while not self._stop_event.wait(timeout=60):
            try:
                self._tick()
            except Exception as exc:
                log.error("SchedulerDaemon tick error: %s", exc)

    def _tick(self) -> None:
        """Scan all tenant schedules once and fire retraining jobs that are due."""
        base = Path(BASE_TENANTS_DIR)
        if not base.exists():
            return

        now = time.time()
        for tenant_dir in base.iterdir():
            if not tenant_dir.is_dir():
                continue
            tenant_id = tenant_dir.name
            sched_dir = tenant_dir / SCHEDULE_STORE_DIR_NAME
            if not sched_dir.exists():
                continue
            for sched_file in sched_dir.glob("*.json"):
                schedule = _read_schedule_path(sched_file)
                if schedule is None:
                    continue

                if not schedule.get("enabled", True):
                    continue

                next_run = schedule.get("next_run_at", 0)
                if now < next_run:
                    continue

                # Due — fire a job
                try:
                    job_id = _run_scheduled_retrain(tenant_id, schedule)
                    log.info(
                        "Scheduler fired retrain for tenant=%s schedule=%s job=%s",
                        tenant_id, schedule["id"], job_id,
                    )
                    history_entry = {
                        "job_id": job_id,
                        "triggered_at": now,
                        "triggered_by": "scheduler",
                    }
                except Exception as exc:
                    log.error(
                        "Scheduler failed to fire retrain for tenant=%s schedule=%s: %s",
                        tenant_id, schedule["id"], exc,
                    )
                    history_entry = {
                        "job_id": None,
                        "triggered_at": now,
                        "triggered_by": "scheduler",
                        "error": str(exc),
                    }

                # Advance next_run_at
                interval_s = schedule.get("interval_seconds", 86_400)
                schedule["next_run_at"] = now + interval_s
                schedule["last_run_at"] = now
                schedule["run_count"] = schedule.get("run_count", 0) + 1
                _write_json_atomic(sched_file, schedule)
                _append_history(tenant_id, schedule["id"], history_entry)


# Start the daemon as soon as this module is imported
_SchedulerDaemon.ensure_running()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/scheduler/schedules", status_code=201)
@require_auth(scope="training:write")
async def create_schedule(request: Request, body: CreateScheduleRequest):
    """
    Create a recurring retraining schedule.

    The model will be retrained automatically at the configured interval
    using the specified data source.  A retraining job is NOT triggered
    immediately — the first run happens after `interval` elapses.

    Use `POST /scheduler/schedules/{id}/trigger` to run immediately.
    """
    tenant_id: str = request.state.tenant_id

    interval_s = _interval_to_seconds(body.interval_value, body.interval_unit)
    if interval_s < MIN_INTERVAL_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum retraining interval is {MIN_INTERVAL_SECONDS // 3600} hour(s).",
        )

    now = time.time()
    schedule_id = str(uuid.uuid4())
    schedule = {
        "id": schedule_id,
        "tenant_id": tenant_id,
        "name": body.name,
        "description": body.description,
        "enabled": body.enabled,
        "interval_value": body.interval_value,
        "interval_unit": body.interval_unit,
        "interval_seconds": interval_s,
        "data_source": body.data_source.model_dump(),
        "top_k": body.top_k,
        "n_trials": body.n_trials,
        "top_models": body.top_models,
        "algorithm_mode": body.algorithm_mode,
        "created_at": now,
        "updated_at": now,
        "next_run_at": now + interval_s,
        "last_run_at": None,
        "run_count": 0,
        "history": [],
    }

    await asyncio.to_thread(_write_schedule, tenant_id, schedule)
    _SchedulerDaemon.ensure_running()

    log.info("Created schedule %s for tenant %s (interval=%ss)", schedule_id, tenant_id, interval_s)
    return _public_schedule(schedule)


@router.get("/scheduler/schedules")
@require_auth(scope="jobs:read")
async def list_schedules(request: Request):
    """List all retraining schedules for the current tenant."""
    tenant_id: str = request.state.tenant_id
    schedules = await asyncio.to_thread(_list_schedules, tenant_id)
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "schedules": [_public_schedule(s) for s in schedules],
        "total": len(schedules),
    })


@router.get("/scheduler/schedules/{schedule_id}")
@require_auth(scope="jobs:read")
async def get_schedule(request: Request, schedule_id: str):
    """Retrieve a specific schedule by ID."""
    tenant_id: str = request.state.tenant_id
    schedule = await asyncio.to_thread(_read_schedule, tenant_id, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")
    return _public_schedule(schedule)


@router.patch("/scheduler/schedules/{schedule_id}")
@require_auth(scope="training:write")
async def update_schedule(request: Request, schedule_id: str, body: PatchScheduleRequest):
    """
    Partially update a schedule.

    You can enable/disable it, change the interval, name, description,
    or training parameters.  If the interval changes, `next_run_at` is
    recalculated from now.
    """
    tenant_id: str = request.state.tenant_id
    schedule = await asyncio.to_thread(_read_schedule, tenant_id, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")

    changed_interval = False

    if body.name is not None:
        schedule["name"] = body.name
    if body.description is not None:
        schedule["description"] = body.description
    if body.enabled is not None:
        schedule["enabled"] = body.enabled
    if body.top_k is not None:
        schedule["top_k"] = body.top_k
    if body.n_trials is not None:
        schedule["n_trials"] = body.n_trials
    if body.top_models is not None:
        schedule["top_models"] = body.top_models
    if body.algorithm_mode is not None:
        schedule["algorithm_mode"] = body.algorithm_mode

    if body.interval_value is not None:
        schedule["interval_value"] = body.interval_value
        changed_interval = True
    if body.interval_unit is not None:
        schedule["interval_unit"] = body.interval_unit
        changed_interval = True

    if changed_interval:
        new_iv = schedule["interval_value"]
        new_iu: IntervalUnit = schedule["interval_unit"]
        interval_s = _interval_to_seconds(new_iv, new_iu)
        if interval_s < MIN_INTERVAL_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum retraining interval is {MIN_INTERVAL_SECONDS // 3600} hour(s).",
            )
        schedule["interval_seconds"] = interval_s
        schedule["next_run_at"] = time.time() + interval_s

    schedule["updated_at"] = time.time()
    await asyncio.to_thread(_write_schedule, tenant_id, schedule)
    return _public_schedule(schedule)


@router.delete("/scheduler/schedules/{schedule_id}", status_code=200)
@require_auth(scope="training:write")
async def delete_schedule(request: Request, schedule_id: str):
    """Permanently delete a schedule. In-flight jobs are not cancelled."""
    tenant_id: str = request.state.tenant_id
    schedule = await asyncio.to_thread(_read_schedule, tenant_id, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")

    await asyncio.to_thread(_delete_schedule_file, tenant_id, schedule_id)
    log.info("Deleted schedule %s for tenant %s", schedule_id, tenant_id)
    return {"deleted": True, "schedule_id": schedule_id}


@router.post("/scheduler/schedules/{schedule_id}/trigger", status_code=202)
@require_auth(scope="training:write")
async def trigger_schedule(request: Request, schedule_id: str):
    """
    Manually trigger an immediate retraining run for this schedule,
    regardless of the next_run_at time.

    Returns the same `job_id` / `poll_url` structure as `/train/file`.
    `next_run_at` is NOT reset — the next automatic run still happens
    at the originally scheduled time.
    """
    tenant_id: str = request.state.tenant_id
    schedule = await asyncio.to_thread(_read_schedule, tenant_id, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")

    try:
        job_id = _run_scheduled_retrain(tenant_id, schedule)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retrain: {exc}")

    # Record manual trigger in history
    await asyncio.to_thread(
        _append_history,
        tenant_id,
        schedule_id,
        {
            "job_id": job_id,
            "triggered_at": time.time(),
            "triggered_by": "manual",
        },
    )

    log.info("Manual trigger: tenant=%s schedule=%s job=%s", tenant_id, schedule_id, job_id)
    return JSONResponse(content={
        "job_id": job_id,
        "poll_url": f"/jobs/{job_id}",
        "schedule_id": schedule_id,
        "triggered_by": "manual",
    })


@router.get("/scheduler/schedules/{schedule_id}/history")
@require_auth(scope="jobs:read")
async def schedule_history(request: Request, schedule_id: str):
    """
    Return the last N job runs triggered by this schedule
    (both automatic and manual triggers).
    """
    tenant_id: str = request.state.tenant_id
    schedule = await asyncio.to_thread(_read_schedule, tenant_id, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found.")

    history = schedule.get("history", [])
    return JSONResponse(content={
        "schedule_id": schedule_id,
        "schedule_name": schedule.get("name"),
        "total": len(history),
        "history": list(reversed(history)),  # most recent first
    })


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _public_schedule(s: dict) -> dict:
    """Return a sanitised, user-facing schedule dict (strip internal cruft)."""
    # Mask SQL passwords before returning
    ds = dict(s.get("data_source", {}))
    if ds.get("type") == "sql" and "password" in ds:
        ds = {**ds, "password": "***"}

    return {
        "id": s["id"],
        "tenant_id": s.get("tenant_id"),
        "name": s.get("name"),
        "description": s.get("description"),
        "enabled": s.get("enabled", True),
        "interval_value": s.get("interval_value"),
        "interval_unit": s.get("interval_unit"),
        "interval_seconds": s.get("interval_seconds"),
        "data_source": ds,
        "top_k": s.get("top_k"),
        "n_trials": s.get("n_trials"),
        "top_models": s.get("top_models"),
        "algorithm_mode": s.get("algorithm_mode"),
        "created_at": s.get("created_at"),
        "updated_at": s.get("updated_at"),
        "next_run_at": s.get("next_run_at"),
        "last_run_at": s.get("last_run_at"),
        "run_count": s.get("run_count", 0),
        "next_run_iso": _ts_to_iso(s.get("next_run_at")),
        "last_run_iso": _ts_to_iso(s.get("last_run_at")),
    }


def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    """Convert a Unix timestamp into an ISO-8601 UTC string for API responses."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
