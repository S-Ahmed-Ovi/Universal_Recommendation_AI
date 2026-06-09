"""
smart_db_csv_builder/models/schemas.py — Pydantic request / response models

New in v3.

Covers all data shapes flowing into and out of the smart_db_csv_builder
services — both the API layer (FastAPI request bodies / responses) and
internal service boundaries (executor results, planner results, file
upload descriptors).

Design notes
------------
- All models inherit from pydantic.BaseModel with model_config set to
  populate_by_name=True so both alias and field name work in JSON.
- Optional fields default to None; callers should check before using.
- Enum types are defined for fixed vocabularies (dialect, algorithm mode)
  to catch typos early at the validation boundary.
- "Tenant-scoped" models carry tenant_id as a required field so they
  can never be used accidentally without it.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class SQLDialect(str, Enum):
    postgres   = "postgres"
    postgresql = "postgresql"
    mysql      = "mysql"
    mariadb    = "mariadb"
    sqlite     = "sqlite"


class NoSQLEngine(str, Enum):
    mongodb   = "mongodb"
    cassandra = "cassandra"
    dynamodb  = "dynamodb"


class AlgorithmMode(str, Enum):
    explicit = "explicit"
    implicit = "implicit"


class FileFormat(str, Enum):
    csv     = "csv"
    excel   = "excel"
    json    = "json"
    parquet = "parquet"
    auto    = "auto"


# ── SQL connection request ─────────────────────────────────────────────────────

class SQLConnectionRequest(BaseModel):
    """Request body for POST /train/sql — establishes a DB connection."""

    tenant_id: str = Field(..., description="Owning tenant identifier")
    dialect: SQLDialect
    host: str
    port: int = Field(..., gt=0, le=65535)
    database: str
    username: str
    password: str
    schema_name: Optional[str] = Field(None, alias="schema")
    ssl: bool = True
    connect_timeout: int = Field(30, ge=5, le=120)

    model_config = {"populate_by_name": True}


class NoSQLConnectionRequest(BaseModel):
    """Request body for NoSQL ingestion endpoint."""

    tenant_id: str
    engine: NoSQLEngine
    host: str
    port: int = Field(..., gt=0, le=65535)
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    region: Optional[str] = None       # DynamoDB
    ssl: bool = True
    connect_timeout: int = Field(30, ge=5, le=120)


# ── Column / table descriptor ──────────────────────────────────────────────────

class ColumnMeta(BaseModel):
    """Metadata for a single database column (from introspection)."""

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


class TableSchema(BaseModel):
    """Schema for a single table — returned by /schema endpoints."""

    table: str
    columns: list[ColumnMeta]
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class DatabaseSchemaResponse(BaseModel):
    """Full schema snapshot returned by schema-discovery endpoints."""

    tenant_id: str
    dialect: str
    tables: list[TableSchema]
    total_tables: int


# ── Interaction query specification ───────────────────────────────────────────

class InteractionQuerySpec(BaseModel):
    """
    Column mapping for building a user-item interaction query.
    Used by SQLExecutor.execute_interaction_query() and the LLM planner.
    """

    table: str
    user_col: str
    item_col: str
    rating_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    extra_filters: Optional[str] = None
    limit: Optional[int] = Field(None, gt=0)
    algorithm_mode: AlgorithmMode = AlgorithmMode.implicit

    @field_validator("extra_filters")
    @classmethod
    def no_semicolons(cls, v: Optional[str]) -> Optional[str]:
        """Prevent filter injection via semicolons."""
        if v and ";" in v:
            raise ValueError("extra_filters must not contain semicolons.")
        return v


# ── SQL training request ───────────────────────────────────────────────────────

class SQLTrainRequest(BaseModel):
    """
    Combined request body for POST /train/sql.

    Bundles the connection config, the column mapping, and optional LLM
    planning flag so the endpoint can handle both manual and auto-planned flows.
    """

    connection: SQLConnectionRequest
    query_spec: Optional[InteractionQuerySpec] = None
    custom_sql: Optional[str] = None
    use_llm_planner: bool = Field(
        False,
        description="If True, ignore query_spec/custom_sql and use LLMPlanner.",
    )
    algorithm_mode_hint: AlgorithmMode = AlgorithmMode.implicit

    @model_validator(mode="after")
    def require_spec_or_sql_or_planner(self) -> "SQLTrainRequest":
        """Require at least one SQL planning source before request validation succeeds."""
        has_spec   = self.query_spec is not None
        has_sql    = self.custom_sql is not None
        has_planner = self.use_llm_planner
        if not (has_spec or has_sql or has_planner):
            raise ValueError(
                "Provide one of: query_spec, custom_sql, or use_llm_planner=true."
            )
        return self


# ── File upload request ────────────────────────────────────────────────────────

class FileUploadMeta(BaseModel):
    """
    Metadata accompanying a file upload for POST /train/file.

    The file bytes are sent as multipart/form-data; this model covers the
    JSON metadata fields in the same request.
    """

    tenant_id: str
    filename: str
    file_format: FileFormat = FileFormat.auto
    algorithm_mode: AlgorithmMode = AlgorithmMode.implicit
    persist: bool = Field(
        True,
        description="Save the file to tenants/{tenant_id}/data/ after loading.",
    )


# ── Query result response ──────────────────────────────────────────────────────

class QueryResultResponse(BaseModel):
    """
    Serialisable summary of a QueryResult — returned by training endpoints
    before kicking off the async training job.
    """

    tenant_id: str
    sql_used: str
    rows: int
    columns: list[str]
    source_dialect: str
    algorithm_mode: str
    preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="First 5 rows of the loaded data for UI preview.",
    )


# ── LLM planner response ───────────────────────────────────────────────────────

class LLMPlannerResponse(BaseModel):
    """Response schema for POST /schema/plan (LLM-assisted query planning)."""

    tenant_id: str
    success: bool
    table: str = ""
    user_col: str = ""
    item_col: str = ""
    rating_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    algorithm_mode: str = ""
    sql: str = ""
    reasoning: str = ""
    error: str = ""


# ── Generic API responses ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "3"


class ErrorResponse(BaseModel):
    detail: str
    tenant_id: Optional[str] = None
    code: Optional[str] = None
