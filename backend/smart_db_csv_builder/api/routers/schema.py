"""
smart_db_csv_builder/api/routers/schema.py

Endpoints
---------
GET  /smart-db-csv/api/schema/{conn_id}           — full schema discovery
GET  /smart-db-csv/api/schema/{conn_id}/{table}   — single table columns + sample
POST /smart-db-csv/api/schema/{conn_id}/plan      — LLM-assisted query planning
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.config.database import NoSQLConfig, SQLConfig
from backend.smart_db_csv_builder.api.routers.connections import get_connection_entry
from backend.smart_db_csv_builder.models.schemas import (
    ColumnMeta,
    DatabaseSchemaResponse,
    LLMPlannerResponse,
    TableSchema,
)
from backend.smart_db_csv_builder.services.builder import ConnectionBuilder
from backend.smart_db_csv_builder.services.executor import SQLExecutor
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


def _build_sql_config(cfg: dict) -> SQLConfig:
    return SQLConfig(
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


def _build_nosql_config(cfg: dict) -> NoSQLConfig:
    return NoSQLConfig(
        engine=cfg["engine"],
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        username=cfg.get("username"),
        password=cfg.get("password"),
        region=cfg.get("region"),
        ssl=cfg.get("ssl", True),
        connect_timeout=cfg.get("connect_timeout", 30),
    )


@router.get(
    "/{conn_id}",
    response_model=DatabaseSchemaResponse,
    summary="Discover full database schema",
)
def get_schema(conn_id: str, request: Request, sample_rows: int = 0):
    """
    Introspect all tables and columns in the registered database connection.

    Set **sample_rows** > 0 (max 10) to include example rows in the response —
    useful for understanding data shape before building a query.
    """
    tenant_id = _tenant_id(request)
    entry = get_connection_entry(conn_id, tenant_id)

    if entry["type"] != "sql":
        raise HTTPException(
            status_code=400,
            detail="Schema discovery is only supported for SQL connections.",
        )

    cfg       = entry["config"]
    sql_cfg   = _build_sql_config(cfg)
    executor  = SQLExecutor(sql_cfg, tenant_id=tenant_id)
    sample_n  = max(0, min(int(sample_rows), 10))

    try:
        tables = executor.list_tables()
    except Exception as exc:
        log.error("Schema discovery failed | tenant=%s conn_id=%s error=%s", tenant_id, conn_id, exc)
        raise HTTPException(status_code=502, detail=f"Database connection failed: {exc}")

    table_schemas: list[TableSchema] = []
    for table in tables:
        try:
            raw_cols = executor.describe_table(table)
            cols     = [ColumnMeta(**c) for c in raw_cols]
            samples  = []
            if sample_n:
                df      = executor.sample_table(table, sample_n)
                samples = df.to_dict(orient="records")
            table_schemas.append(TableSchema(table=table, columns=cols, sample_rows=samples))
        except Exception as exc:
            log.warning("Could not describe table '%s' | tenant=%s: %s", table, tenant_id, exc)

    return DatabaseSchemaResponse(
        tenant_id=tenant_id,
        dialect=sql_cfg.dialect,
        tables=table_schemas,
        total_tables=len(table_schemas),
    )


@router.get(
    "/{conn_id}/{table}",
    summary="Describe a single table",
)
def get_table_schema(conn_id: str, table: str, request: Request, sample_rows: int = 5):
    """Return column metadata and optional sample rows for a single table."""
    tenant_id = _tenant_id(request)
    entry     = get_connection_entry(conn_id, tenant_id)

    if entry["type"] != "sql":
        raise HTTPException(status_code=400, detail="Only SQL connections support table introspection.")

    sql_cfg  = _build_sql_config(entry["config"])
    executor = SQLExecutor(sql_cfg, tenant_id=tenant_id)
    sample_n = max(0, min(int(sample_rows), 10))

    try:
        raw_cols = executor.describe_table(table)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to describe table '{table}': {exc}")

    cols    = [ColumnMeta(**c) for c in raw_cols]
    samples = []
    if sample_n:
        try:
            df      = executor.sample_table(table, sample_n)
            samples = df.to_dict(orient="records")
        except Exception as exc:
            log.warning("Sample fetch failed for '%s' | tenant=%s: %s", table, tenant_id, exc)

    return TableSchema(table=table, columns=cols, sample_rows=samples)


@router.post(
    "/{conn_id}/plan",
    response_model=LLMPlannerResponse,
    summary="LLM-assisted interaction query planning",
)
def llm_plan(
    conn_id: str,
    request: Request,
    algorithm_mode_hint: str = "",
    include_samples: bool = True,
):
    """
    Use Claude to analyse the schema and propose the best user-item interaction
    query for training a recommendation model.

    Requires **ANTHROPIC_API_KEY** to be set in the environment.
    Returns the suggested table, columns, and a ready-to-run SQL statement.
    """
    tenant_id = _tenant_id(request)
    entry     = get_connection_entry(conn_id, tenant_id)

    if entry["type"] != "sql":
        raise HTTPException(status_code=400, detail="LLM planning only supports SQL connections.")

    sql_cfg = _build_sql_config(entry["config"])
    builder = ConnectionBuilder(sql_cfg, tenant_id=tenant_id)

    try:
        from backend.smart_db_csv_builder.services.llm_planner import LLMPlanner
        planner = LLMPlanner(builder, tenant_id=tenant_id, include_samples=include_samples)
        result  = planner.plan(algorithm_mode_hint=algorithm_mode_hint)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.error("LLM planning failed | tenant=%s conn_id=%s error=%s", tenant_id, conn_id, exc)
        raise HTTPException(status_code=500, detail=f"LLM planning error: {exc}")

    return LLMPlannerResponse(
        tenant_id=result.tenant_id,
        success=result.success,
        table=result.table,
        user_col=result.user_col,
        item_col=result.item_col,
        rating_col=result.rating_col,
        timestamp_col=result.timestamp_col,
        algorithm_mode=result.algorithm_mode,
        sql=result.sql,
        reasoning=result.reasoning,
        error=result.error,
    )
