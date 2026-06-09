"""
smart_db_csv_builder/services/executor.py — Tenant-aware SQL query executor

Refactored from ingestion/sql_query_executor.py (v2).

v3 changes
----------
- tenant_id threaded through constructor and all log lines.
- Connector is resolved at runtime via ConnectionBuilder (not hard-coded to DBConnector).
- execute_interaction_query() accepts algorithm_mode hint for downstream pipeline.
- list_tables() / describe_table() delegate to the active connector's introspection API.
- SQL validation logic preserved verbatim from v2.
- QueryResult carries tenant_id and algorithm_mode for pipeline handoff.

Usage
-----
    from backend.config.database import SQLConfig
    from backend.smart_db_csv_builder.services.executor import SQLExecutor, QueryResult

    cfg = SQLConfig(dialect="postgres", host="db.example.com", ...)
    executor = SQLExecutor(cfg, tenant_id="acme-corp")

    result = executor.execute_interaction_query(
        user_col="user_id", item_col="item_id", rating_col="rating",
        table="interactions", limit=500_000,
    )
    print(result.df.head())
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.config.database import SQLConfig
from backend.utils.logger import get_logger
from backend.smart_db_csv_builder.services.builder import ConnectionBuilder

log = get_logger(__name__)


@dataclass
class QueryResult:
    """Carries the DataFrame and rich metadata for downstream pipeline stages."""

    df: pd.DataFrame
    sql_used: str
    rows: int
    columns: list[str]
    source_dialect: str
    tenant_id: str = ""
    algorithm_mode: str = ""          # e.g. "explicit" | "implicit" — set by caller
    extra: dict = field(default_factory=dict)


class SQLExecutor:
    """
    High-level, tenant-scoped SQL executor.

    Sits on top of the dialect-specific connectors returned by ConnectionBuilder.
    Supports:
      - Raw custom SQL (with safety validation)
      - Auto-generated interaction queries from column/table metadata
      - Schema discovery helpers (list_tables, describe_table)
    """

    # SQL injection blocklist (preserved from v2)
    _FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
        {"drop", "delete", "truncate", "alter", "insert", "update", "exec", "execute"}
    )

    def __init__(self, config: SQLConfig, tenant_id: str = "") -> None:
        """Initialise the SQL executor and its connector builder for one tenant."""
        self.config    = config
        self.tenant_id = tenant_id
        self._builder  = ConnectionBuilder(config, tenant_id=tenant_id)

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute_custom(
        self,
        sql: str,
        params: Optional[dict] = None,
        algorithm_mode: str = "",
    ) -> QueryResult:
        """
        Execute a user-provided SQL query with safety validation.

        Parameters
        ----------
        sql            : raw SQL string
        params         : named parameters dict (recommended for user-supplied values)
        algorithm_mode : optional hint ("explicit" | "implicit") stored in result
        """
        self._validate_sql(sql)
        log.info("Executing custom SQL | tenant=%s", self.tenant_id)

        with self._builder.get_connector() as conn:
            df = conn.query(sql, params)

        return QueryResult(
            df=df,
            sql_used=sql,
            rows=len(df),
            columns=list(df.columns),
            source_dialect=self.config.dialect,
            tenant_id=self.tenant_id,
            algorithm_mode=algorithm_mode,
        )

    def execute_interaction_query(
        self,
        user_col: str,
        item_col: str,
        rating_col: Optional[str],
        table: str,
        timestamp_col: Optional[str] = None,
        extra_filters: Optional[str] = None,
        limit: Optional[int] = None,
        algorithm_mode: str = "",
    ) -> QueryResult:
        """
        Auto-build and execute a standard interaction query.

        Produces:
            SELECT user_col, item_col, [rating_col], [timestamp_col]
            FROM table
            [WHERE extra_filters]
            [LIMIT limit]

        Parameters
        ----------
        user_col       : column name for user identifiers
        item_col       : column name for item identifiers
        rating_col     : optional explicit rating/score column
        table          : source table name
        timestamp_col  : optional event timestamp column
        extra_filters  : raw SQL WHERE clause fragment (no WHERE keyword)
        limit          : optional row cap
        algorithm_mode : "explicit" | "implicit" — carried into QueryResult
        """
        select_cols = [user_col, item_col]
        if rating_col:
            select_cols.append(rating_col)
        if timestamp_col:
            select_cols.append(timestamp_col)

        cols_str = ", ".join(select_cols)
        sql = f"SELECT {cols_str} FROM {table}"
        if extra_filters:
            sql += f" WHERE {extra_filters}"
        if limit:
            sql += f" LIMIT {limit}"

        log.info(
            "Auto-built interaction query | tenant=%s algorithm_mode=%s sql=%.120s",
            self.tenant_id, algorithm_mode or "unset", sql,
        )
        return self.execute_custom(sql, algorithm_mode=algorithm_mode)

    def list_tables(self) -> list[str]:
        """Return all table names visible in the connected database."""
        with self._builder.get_connector() as conn:
            tables = conn.get_table_names()
        log.info("Listed %d tables | tenant=%s", len(tables), self.tenant_id)
        return tables

    def describe_table(self, table: str) -> list[dict]:
        """Return column metadata for a given table (name, type, nullable, primary_key)."""
        with self._builder.get_connector() as conn:
            cols = conn.get_columns(table)
        log.info(
            "Described table '%s' (%d cols) | tenant=%s",
            table, len(cols), self.tenant_id,
        )
        return cols

    def sample_table(self, table: str, n: int = 5) -> pd.DataFrame:
        """
        Return the first *n* rows of *table*.

        Delegates to the connector's get_sample() method when available,
        otherwise falls back to a plain LIMIT query.
        """
        with self._builder.get_connector() as conn:
            if hasattr(conn, "get_sample"):
                return conn.get_sample(table, n)
            return conn.query(f"SELECT * FROM {table} LIMIT :n", {"n": n})

    # ── SQL validation (preserved from v2) ────────────────────────────────────

    def _validate_sql(self, sql: str) -> None:
        """
        Basic SQL injection guard.

        Raises ValueError if forbidden keywords are detected or if the query
        is not a SELECT / CTE statement.
        """
        lowered = sql.lower()

        # Strip string literals to avoid false positives on quoted values
        cleaned = re.sub(r"'[^']*'", "", lowered)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)

        found = [
            kw for kw in self._FORBIDDEN_KEYWORDS
            if f" {kw} " in f" {cleaned} "
        ]
        if found:
            raise ValueError(
                f"SQL contains forbidden keyword(s): {found}. "
                f"Only SELECT queries are permitted. | tenant={self.tenant_id}"
            )

        stripped = cleaned.strip()
        if not stripped.startswith("select") and "with" not in stripped[:10]:
            raise ValueError(
                f"Only SELECT (or WITH … SELECT) queries are permitted. "
                f"| tenant={self.tenant_id}"
            )

        log.debug("SQL validation passed | tenant=%s", self.tenant_id)
