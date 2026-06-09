"""
Training routes

POST /train/file  — upload a CSV/Excel/JSON/Parquet file and kick off training
POST /train/sql   — supply DB credentials + SQL query and kick off training

Both endpoints are auth-gated (scope: training:write) and tenant-aware.
Training is executed asynchronously; callers poll /jobs/{job_id} for status.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from backend.api.gateway.auth import require_auth
from backend.api.gateway.metering import metering
from backend.config.settings import (
    ALGORITHM_MODES,
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
    _runtime_state,
    _serving_cache,
    _direct_recommender,
    _submit_job,
    _serialise_training_result,
    _save_last_result,
    _validate_algorithm_mode,
    _validate_top_choice,
)

log = get_logger(__name__)

router = APIRouter(tags=["Training"])

_MAX_UPLOAD_FILE_SIZE = 100 * 1024 * 1024  # 100MB
_UPLOAD_COPY_BUFFER_SIZE = 1024 * 1024
_ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SQLTrainRequest(BaseModel):
    """Request body for tenant-scoped SQL training jobs."""
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    sql: str
    top_k: int = TOP_K
    n_trials: int = OPTUNA_TRIALS_AUTO
    top_models: int = TOP_N_MODELS
    algorithm_mode: str = DEFAULT_ALGORITHM_MODE

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        """Reject unsupported recommendation cutoffs before job submission."""
        if value not in TOP_K_ALLOWED:
            raise ValueError(f"top_k must be one of {TOP_K_ALLOWED}")
        return value

    @field_validator("top_models")
    @classmethod
    def _validate_top_models(cls, value: int) -> int:
        """Reject unsupported ranked-model counts before job submission."""
        if value not in TOP_MODEL_ALLOWED:
            raise ValueError(f"top_models must be one of {TOP_MODEL_ALLOWED}")
        return value

    @field_validator("algorithm_mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        """Normalize and validate the requested training mode."""
        v = (value or "").lower().strip()
        if v not in ALGORITHM_MODES:
            raise ValueError(f"algorithm_mode must be one of {ALGORITHM_MODES}")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _persist_upload_to_path(source_file, destination: Path) -> None:
    """Copy the uploaded dataset stream to a temporary path off the event loop."""
    source_file.seek(0)
    with open(destination, "wb") as out:
        shutil.copyfileobj(source_file, out, length=_UPLOAD_COPY_BUFFER_SIZE)


def _finalise_training_state(tenant_id: str, result, serialised: dict) -> dict:
    """Refresh cached runtime artifacts after a successful training run."""
    _save_last_result(tenant_id, serialised)

    runtime = _runtime_state[tenant_id]
    runtime["train_df"] = result.report.train.copy()
    runtime["last_result"] = result
    runtime["serving_im"] = InteractionMatrixBuilder().build(result.report.train)

    _direct_recommender(tenant_id).im = runtime["serving_im"]
    _serving_cache.pop(tenant_id, None)
    return serialised


@router.post("/train/file")
@require_auth(scope="training:write")
async def train_from_file(
    request: Request,
    file: UploadFile = File(..., description="CSV/Excel/JSON/Parquet file"),
    top_k: int = Form(TOP_K),
    n_trials: int = Form(OPTUNA_TRIALS_AUTO),
    top_models: int = Form(TOP_N_MODELS),
    algorithm_mode: str = Form(DEFAULT_ALGORITHM_MODE),
    format: str = Form("auto"),
):
    """
    Upload a dataset file and start an async training job.

    Returns a job_id that can be polled via GET /jobs/{job_id}.
    """
    _validate_top_choice("top_k", top_k, TOP_K_ALLOWED)
    _validate_top_choice("top_models", top_models, TOP_MODEL_ALLOWED)
    _validate_algorithm_mode(algorithm_mode)

    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "training",
        1,
        request_state=request.state,
    )

    # ── Security & Validation ───────────────────────────────────────────────
    # Check file size (if available from headers or already read)
    if file.size and file.size > _MAX_UPLOAD_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file.size} bytes). Maximum allowed is {_MAX_UPLOAD_FILE_SIZE} bytes.",
        )

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in _ALLOWED_UPLOAD_EXTENSIONS and format == "auto":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {_ALLOWED_UPLOAD_EXTENSIONS}",
        )

    suffix = Path(file.filename).suffix if file.filename else ""
    fd, temp_name = tempfile.mkstemp(prefix="proactive_upload_", suffix=suffix)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        await asyncio.to_thread(_persist_upload_to_path, file.file, temp_path)
    finally:
        await file.close()

    config = TrainingConfig(
        top_k=top_k,
        n_tuning_trials=n_trials,
        top_model_count=top_models,
        algorithm_mode=algorithm_mode.lower(),
    )

    def _task():
        """Run file-based training and always clean up the temporary upload artifact."""
        try:
            pipeline = TrainingPipeline(tenant_id, config=config)
            result = pipeline.run_from_file(str(temp_path), file_format=format)

            serialised = _serialise_training_result(result, tenant_id=tenant_id)
            return _finalise_training_state(tenant_id, result, serialised)
        finally:
            try:
                if temp_path.exists():
                    os.remove(temp_path)
            except Exception:
                pass

    job_id = _submit_job(
        tenant_id,
        _task,
        job_meta={
            "algorithm_mode": algorithm_mode.lower(),
            "n_trials": n_trials,
            "source_label": file.filename or "uploaded dataset",
            "source_type": "file",
            "top_k": top_k,
            "top_models": top_models,
        },
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "poll_url": f"/jobs/{job_id}"},
    )


@router.post("/train/sql")
@require_auth(scope="training:write")
async def train_from_sql(request: Request, req: SQLTrainRequest):
    """
    Provide DB credentials and a SQL query to start an async training job.

    Returns a job_id that can be polled via GET /jobs/{job_id}.
    """
    from backend.config.database import SQLConfig

    tenant_id: str = request.state.tenant_id
    await metering.record_usage(
        tenant_id,
        "training",
        1,
        request_state=request.state,
    )

    sql_config = SQLConfig(
        dialect=req.dialect,
        host=req.host,
        port=req.port,
        database=req.database,
        username=req.username,
        password=req.password,
    )
    config = TrainingConfig(
        top_k=req.top_k,
        n_tuning_trials=req.n_trials,
        top_model_count=req.top_models,
        algorithm_mode=req.algorithm_mode,
    )

    def _task():
        """Run SQL-based training and publish the updated runtime state."""
        pipeline = TrainingPipeline(tenant_id, config=config)
        result = pipeline.run_from_sql(sql_config, req.sql)

        serialised = _serialise_training_result(result, tenant_id=tenant_id)
        return _finalise_training_state(tenant_id, result, serialised)

    job_id = _submit_job(
        tenant_id,
        _task,
        job_meta={
            "algorithm_mode": req.algorithm_mode,
            "n_trials": req.n_trials,
            "source_label": f"{req.dialect} / {req.database}",
            "source_type": "sql",
            "top_k": req.top_k,
            "top_models": req.top_models,
        },
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "poll_url": f"/jobs/{job_id}"},
    )
