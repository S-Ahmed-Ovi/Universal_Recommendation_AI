"""
smart_db_csv_builder/connectors/sqlite.py — SQLite connector

New in v3 (was not a separate connector in v2).

SQLite is zero-config and needs no host/port/credentials.  It's useful for:
  - Local development without a running database server
  - Tenant onboarding demos and integration tests
  - Embedded analytics on small datasets

Tenant-awareness: the default database file is resolved relative to
tenants/{tenant_id}/data/ so each tenant has its own SQLite file by default.
An explicit absolute path overrides this.

Usage
-----
    # Default path  →  tenants/acme-corp/data/interactions.db
    with SQLiteConnector("interactions.db", tenant_id="acme-corp") as conn:
        df = conn.query("SELECT user_id, item_id, rating FROM events")
        tables = conn.get_table_names()

    # Absolute path (e.g. shared test fixture)
    with SQLiteConnector("/tmp/test.db") as conn:
        df = conn.query("SELECT * FROM ratings LIMIT 100")

    # In-memory (no persistence)
    with SQLiteConnector(":memory:") as conn:
        conn.query("CREATE TABLE t (a INT)")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from backend.utils.logger import get_logger

log = get_logger(__name__)

_TENANTS_ROOT = Path("tenants")


class SQLiteConnector:
    """
    Lightweight SQLite connector scoped to a tenant.

    Zero-dependency on external DB drivers — SQLite ships with Python.
    Uses SQLAlchemy for the same query/introspect API as the other connectors.
    """

    def __init__(self, db_path: str = ":memory:", tenant_id: str = "") -> None:
        """
        Parameters
        ----------
        db_path   : filename or absolute path to the .db file.
                    Relative filenames are resolved to tenants/{tenant_id}/data/.
                    Use ":memory:" for an in-process ephemeral database.
        tenant_id : owning tenant — used for path resolution and logging.
        """
        self.tenant_id = tenant_id
        self.db_path   = self._resolve_path(db_path, tenant_id)
        self._engine   = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "SQLiteConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> None:
        """Open a SQLAlchemy engine backed by the resolved SQLite file."""
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError("Install sqlalchemy: pip install sqlalchemy")

        url = f"sqlite:///{self.db_path}" if self.db_path != ":memory:" else "sqlite://"

        self._engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        log.info(
            "SQLite connected | tenant=%s path=%s",
            self.tenant_id or "–", self.db_path,
        )

    def disconnect(self) -> None:
        """Dispose the SQLite engine and clear the cached connection handle."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            log.info("SQLite disconnected | tenant=%s", self.tenant_id or "–")

    # ── Query execution ────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return a DataFrame.

        Parameters
        ----------
        sql    : SQL string (use :name for named placeholders)
        params : optional bind parameters dict
        """
        self._require_connected()
        log.info(
            "SQLite query | tenant=%s sql_preview=%.120s",
            self.tenant_id or "–", sql.replace("\n", " "),
        )
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
            log.info(
                "SQLite result | tenant=%s rows=%d cols=%d",
                self.tenant_id or "–", *df.shape,
            )
            return df
        except Exception as exc:
            log.error("SQLite query failed | tenant=%s error=%s", self.tenant_id, exc)
            raise RuntimeError(f"SQLite query failed: {exc}") from exc

    def execute(self, sql: str, params: Optional[dict] = None) -> None:
        """
        Execute a DDL or DML statement (CREATE, INSERT, etc.).

        Only intended for test fixtures and local dev — production ingestion
        should use query() for SELECT only.
        """
        self._require_connected()
        from sqlalchemy import text
        with self._engine.begin() as conn:
            conn.execute(text(sql), params or {})

    # ── Schema introspection ───────────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        """List all user-created tables in the database."""
        self._require_connected()
        from sqlalchemy import inspect
        return inspect(self._engine).get_table_names()

    def get_columns(self, table: str) -> list[dict]:
        """
        Return column metadata for *table*.

        Each dict: {name, type, nullable, primary_key}
        """
        self._require_connected()
        from sqlalchemy import inspect
        inspector = inspect(self._engine)
        cols = inspector.get_columns(table)
        pks  = {
            c["name"]
            for c in inspector.get_pk_constraint(table).get("constrained_columns", [])
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
        """Return the first *n* rows of *table* — useful for LLM schema planning."""
        return self.query(f'SELECT * FROM "{table}" LIMIT :n', params={"n": n})

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(db_path: str, tenant_id: str) -> str:
        if db_path == ":memory:":
            return ":memory:"
        p = Path(db_path)
        if p.is_absolute():
            return str(p)
        if tenant_id:
            tenant_data = _TENANTS_ROOT / tenant_id / "data"
            tenant_data.mkdir(parents=True, exist_ok=True)
            return str(tenant_data / db_path)
        return str(p)

    def _require_connected(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "Not connected. Call connect() or use the context manager."
            )
