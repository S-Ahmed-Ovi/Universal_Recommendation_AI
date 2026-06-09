"""
smart_db_csv_builder/api/routers/build.py

Endpoints
---------
POST /smart-db-csv/api/build        — submit a build job (LLM-planned or manual)
GET  /smart-db-csv/api/build/{conn_id}/preview — quick data preview (no job)

A build job:
  1. Uses a registered conn_id to open the database.
  2. Either runs LLMPlanner to auto-detect the interaction query, or uses the
     caller-supplied query_spec / custom_sql.
  3. Streams the result to a CSV file under tenants/{tenant_id}/data/.
  4. Returns a job_id that can be polled via GET /smart-db-csv/api/jobs/{job_id}.

Jobs run synchronously in this implementation (background thread for long
queries).  Swap the thread pool for Celery / ARQ for true async production use.
"""
from __future__ import annotations

import csv
import io
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.config.database import SQLConfig
from backend.smart_db_csv_builder.api.routers.connections import get_connection_entry
from backend.smart_db_csv_builder.models.schemas import (
    AlgorithmMode,
    InteractionQuerySpec,
)
from backend.smart_db_csv_builder.services.executor import SQLExecutor
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# ── Job registry (in-process; swap for Redis/DB in production) ─────────────────
# Maps job_id → job dict (see _make_job)
_jobs: dict[str, dict[str, Any]] = {}

TENANTS_ROOT = Path("tenants")


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


# ── Request / response models ─────────────────────────────────────────────────

class BuildRequest(BaseModel):
    conn_id: str = Field(..., description="UUID from POST /connections")
    use_llm_planner: bool = Field(
        False,
        description="Auto-detect interaction columns via LLM schema analysis.",
    )
    query_spec: Optional[InteractionQuerySpec] = Field(
        None,
        description="Manual column mapping (ignored when use_llm_planner=true).",
    )
    custom_sql: Optional[str] = Field(
        None,
        description="Raw SELECT statement (ignored when use_llm_planner=true).",
    )
    algorithm_mode: AlgorithmMode = AlgorithmMode.implicit
    output_filename: Optional[str] = Field(
        None,
        description="CSV filename under tenants/{tenant}/data/. Defaults to <job_id>.csv",
    )
    row_limit: Optional[int] = Field(None, gt=0, description="Cap the number of rows fetched.")

    model_config = ConfigDict(use_enum_values=True)


# ── Job helpers ───────────────────────────────────────────────────────────────

def _make_job(job_id: str, tenant_id: str, conn_id: str) -> dict:
    return {
        "job_id":     job_id,
        "tenant_id":  tenant_id,
        "conn_id":    conn_id,
        "status":     "pending",   # pending | running | done | failed
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "output_path": None,
        "rows":       None,
        "columns":    None,
        "sql_used":   None,
        "error":      None,
    }


def _run_build(job_id: str, body: BuildRequest, tenant_id: str, entry: dict) -> None:
    """Executed in a background thread. Updates _jobs[job_id] in-place."""
    job = _jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = time.time()

    try:
        cfg = entry["config"]
        sql_cfg = SQLConfig(
            dialect=cfg["dialect"],
            host=cfg["host"],
            port=cfg["port"],
            database=cfg["database"],
            username=cfg["username"],
            password=cfg["password"],
            schema=cfg.get("schema_name"),
            ssl=cfg.get("ssl", True),
            connect_timeout=cfg.get("connect_timeout", 30),
        )

        if body.use_llm_planner:
            # ── LLM-assisted path ──────────────────────────────────────────
            from backend.smart_db_csv_builder.services.builder import ConnectionBuilder
            from backend.smart_db_csv_builder.services.llm_planner import LLMPlanner

            builder = ConnectionBuilder(sql_cfg, tenant_id=tenant_id)
            planner = LLMPlanner(builder, tenant_id=tenant_id)
            result  = planner.plan(algorithm_mode_hint=body.algorithm_mode)

            if not result.success:
                raise RuntimeError(f"LLM planner failed: {result.error}")

            executor = SQLExecutor(sql_cfg, tenant_id=tenant_id)
            sql      = result.sql
            if body.row_limit:
                sql += f" LIMIT {body.row_limit}"
            qr = executor.execute_custom(sql, algorithm_mode=result.algorithm_mode)

        elif body.custom_sql:
            # ── Raw SQL path ───────────────────────────────────────────────
            executor = SQLExecutor(sql_cfg, tenant_id=tenant_id)
            sql      = body.custom_sql
            if body.row_limit:
                sql = f"SELECT * FROM ({sql}) AS _sub LIMIT {body.row_limit}"
            qr = executor.execute_custom(sql, algorithm_mode=body.algorithm_mode)

        elif body.query_spec:
            # ── Manual spec path ───────────────────────────────────────────
            spec     = body.query_spec
            executor = SQLExecutor(sql_cfg, tenant_id=tenant_id)
            qr = executor.execute_interaction_query(
                user_col=spec.user_col,
                item_col=spec.item_col,
                rating_col=spec.rating_col,
                table=spec.table,
                timestamp_col=spec.timestamp_col,
                extra_filters=spec.extra_filters,
                limit=body.row_limit or spec.limit,
                algorithm_mode=body.algorithm_mode,
            )
        else:
            raise ValueError(
                "Provide one of: use_llm_planner=true, custom_sql, or query_spec."
            )

        # ── Write CSV ──────────────────────────────────────────────────────
        out_dir  = TENANTS_ROOT / tenant_id / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = body.output_filename or f"{job_id}.csv"
        out_path = out_dir / filename

        qr.df.to_csv(out_path, index=False)

        job["status"]      = "done"
        job["finished_at"] = time.time()
        job["output_path"] = str(out_path)
        job["rows"]        = qr.rows
        job["columns"]     = qr.columns
        job["sql_used"]    = qr.sql_used

        log.info(
            "Build job done | tenant=%s job_id=%s rows=%d path=%s",
            tenant_id, job_id, qr.rows, out_path,
        )

    except Exception as exc:
        job["status"]      = "failed"
        job["finished_at"] = time.time()
        job["error"]       = str(exc)
        log.error("Build job failed | tenant=%s job_id=%s error=%s", tenant_id, job_id, exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    status_code=202,
    summary="Submit a CSV build job",
)
def submit_build(body: BuildRequest, request: Request):
    """
    Kick off a CSV build job from a registered database connection.

    Choose **one** data source strategy per request:
    - `use_llm_planner: true` — let Claude analyse the schema and write the query
    - `custom_sql` — provide a raw SELECT statement
    - `query_spec` — specify table + column mapping

    The job runs asynchronously. Poll `GET /smart-db-csv/api/jobs/{job_id}`
    for status and the output CSV path when done.
    """
    tenant_id = _tenant_id(request)
    entry     = get_connection_entry(body.conn_id, tenant_id)

    if entry["type"] != "sql":
        raise HTTPException(status_code=400, detail="Build jobs only support SQL connections.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = _make_job(job_id, tenant_id, body.conn_id)

    t = threading.Thread(
        target=_run_build,
        args=(job_id, body, tenant_id, entry),
        daemon=True,
    )
    t.start()

    log.info("Build job submitted | tenant=%s job_id=%s", tenant_id, job_id)
    return {
        "job_id":  job_id,
        "status":  "pending",
        "message": "Job submitted. Poll /smart-db-csv/api/jobs/{job_id} for status.",
    }


@router.get(
    "/{conn_id}/preview",
    summary="Quick data preview (no job)",
)
def preview_data(
    conn_id: str,
    request: Request,
    table: str = "",
    limit: int = 20,
    custom_sql: Optional[str] = None,
):
    """
    Fetch a small sample of data directly — no job created, no CSV written.

    Useful for validating a connection and exploring data before committing to
    a full build job. Maximum 200 rows.
    """
    tenant_id = _tenant_id(request)
    entry     = get_connection_entry(conn_id, tenant_id)
    limit     = max(1, min(int(limit), 200))

    cfg = entry["config"]
    sql_cfg  = SQLConfig(
        dialect=cfg["dialect"],
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        username=cfg["username"],
        password=cfg["password"],
        schema=cfg.get("schema_name"),
        ssl=cfg.get("ssl", True),
        connect_timeout=cfg.get("connect_timeout", 30),
    )
    executor = SQLExecutor(sql_cfg, tenant_id=tenant_id)

    try:
        if custom_sql:
            sql = f"SELECT * FROM ({custom_sql}) AS _sub LIMIT {limit}"
            qr  = executor.execute_custom(sql)
        elif table:
            qr = executor.execute_custom(f'SELECT * FROM "{table}" LIMIT {limit}')
        else:
            # Fall back to listing tables
            tables = executor.list_tables()
            return {"tables": tables, "hint": "Pass ?table=<name> to preview a table."}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Preview failed: {exc}")

    return {
        "rows":    qr.rows,
        "columns": qr.columns,
        "data":    qr.df.head(limit).to_dict(orient="records"),
    }


# ── Expose _jobs for jobs.py ──────────────────────────────────────────────────

def get_job_store() -> dict[str, dict]:
    """Expose the shared job store used by the companion jobs router."""
    return _jobs
