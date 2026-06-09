"""
smart_db_csv_builder/connectors/postgres.py — PostgreSQL connector

Split from ingestion/db_connector.py (v2 supported all dialects in one class).

v3 changes
----------
- Dialect is hardcoded to PostgreSQL; no dialect-dispatch branching.
- tenant_id is carried on the instance for audit logging.
- SSL mode respected from SQLConfig.ssl flag (sslmode=require / prefer).
- Schema search_path set on connect when SQLConfig.schema is provided.
- get_sample() convenience method for schema discovery / LLM planner.
- All core logic (connect/query/disconnect/introspect) preserved from v2.

Usage
-----
    from backend.config.database import SQLConfig

    cfg = SQLConfig(
        dialect="postgres", host="db.example.com", port=5432,
        database="warehouse", username="svc", password="...",
    )
    with PostgresConnector(cfg, tenant_id="acme-corp") as conn:
        df = conn.query("SELECT user_id, item_id, rating FROM events LIMIT 1000")
        tables = conn.get_table_names()
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.config.database import SQLConfig
from backend.utils.logger import get_logger

log = get_logger(__name__)

_DRIVER = "postgresql+psycopg2"


class PostgresConnector:
    """
    Manages a SQLAlchemy connection pool to a PostgreSQL server.

    Tenant-scoped: tenant_id is recorded on every log line for tracing.
    """

    def __init__(self, config: SQLConfig, tenant_id: str = "") -> None:
        """Initialise a tenant-aware PostgreSQL connector from validated config."""
        if config.dialect not in ("postgres", "postgresql"):
            raise ValueError(
                f"PostgresConnector requires dialect 'postgres', got '{config.dialect}'"
            )
        self.config    = config
        self.tenant_id = tenant_id
        self._engine   = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "PostgresConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> None:
        """Open and verify a connection pool to PostgreSQL."""
        try:
            from sqlalchemy import create_engine, event, text
        except ImportError:
            raise ImportError("Install sqlalchemy + psycopg2: pip install sqlalchemy psycopg2-binary")

        connect_args: dict = {
            "connect_timeout": self.config.connect_timeout,
            "sslmode": "require" if self.config.ssl else "prefer",
        }

        # FIX: use config.connection_url() so that username and password are
        # percent-encoded via quote_plus before being embedded in the URL.
        # The previous raw f-string interpolation caused passwords containing
        # special characters (e.g. '@', '#') to corrupt the host portion of
        # the URL, producing bogus hostnames like "885d#d@arl-postgresql...".
        url = self.config.connection_url()

        self._engine = create_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

        # Set search_path if a schema is configured
        if self.config.schema:
            schema = self.config.schema

            @event.listens_for(self._engine, "connect")
            def set_search_path(dbapi_conn, _record):
                """Apply the configured tenant schema on each new DBAPI connection."""
                cursor = dbapi_conn.cursor()
                cursor.execute(f"SET search_path TO {schema}, public")
                cursor.close()

        # Verify connectivity
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        log.info(
            "PostgreSQL connected | tenant=%s host=%s db=%s schema=%s",
            self.tenant_id, self.config.host, self.config.database, self.config.schema or "public",
        )

    def disconnect(self) -> None:
        """Dispose the connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            log.info("PostgreSQL disconnected | tenant=%s", self.tenant_id)

    # ── Query execution ────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return a DataFrame.

        Parameters
        ----------
        sql    : SQL string (use :name for named placeholders)
        params : optional dict of bind parameters
        """
        self._require_connected()
        log.info(
            "PG query | tenant=%s sql_preview=%.120s",
            self.tenant_id, sql.replace("\n", " "),
        )
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
            log.info(
                "PG result | tenant=%s rows=%d cols=%d",
                self.tenant_id, *df.shape,
            )
            return df
        except Exception as exc:
            log.error("PG query failed | tenant=%s error=%s", self.tenant_id, exc)
            raise RuntimeError(f"PostgreSQL query failed: {exc}") from exc

    # ── Schema introspection ───────────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        """List all tables visible in the connected schema."""
        self._require_connected()
        from sqlalchemy import inspect
        inspector = inspect(self._engine)
        return inspector.get_table_names(schema=self.config.schema)

    def get_columns(self, table: str) -> list[dict]:
        """
        Return column metadata for *table*.

        Each dict: {name, type, nullable, primary_key}
        """
        self._require_connected()
        from sqlalchemy import inspect
        inspector = inspect(self._engine)
        schema    = self.config.schema
        cols = inspector.get_columns(table, schema=schema)
        pks  = {
            c["name"]
            for c in inspector.get_pk_constraint(table, schema=schema).get("constrained_columns", [])
        }
        return [
            {
                "name":        c["name"],
                "type":        str(c["type"]),
                "nullable":    c.get("nullable", True),
                "primary_key": c["name"] in pks,
            }
            for c in cols
        ]

    def get_sample(self, table: str, n: int = 5) -> pd.DataFrame:
        """
        Return the first *n* rows of *table* — useful for LLM schema planning.
        """
        return self.query(f'SELECT * FROM "{table}" LIMIT :n', params={"n": n})

    # ── Internal ───────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "Not connected. Call connect() or use the context manager."
            )
