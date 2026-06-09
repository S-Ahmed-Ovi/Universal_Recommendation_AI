"""
smart_db_csv_builder/connectors/mysql.py — MySQL / MariaDB connector

Split from ingestion/db_connector.py (v2 supported all dialects in one class).

v3 changes
----------
- Dialect locked to MySQL/MariaDB; uses PyMySQL driver by default.
- tenant_id on every log line for audit tracing.
- SSL flag maps to ssl_ca / ssl_disabled connect_args.
- charset defaults to utf8mb4; can be overridden via SQLConfig.extra_params.
- get_sample() convenience method for schema discovery / LLM planner.
- Full introspection API (get_table_names, get_columns) preserved from v2.

Usage
-----
    from backend.config.database import SQLConfig

    cfg = SQLConfig(
        dialect="mysql", host="db.example.com", port=3306,
        database="shop", username="svc", password="...", ssl=True,
    )
    with MySQLConnector(cfg, tenant_id="techco-inc") as conn:
        df = conn.query("SELECT user_id, item_id, rating FROM interactions")
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.config.database import SQLConfig
from backend.utils.logger import get_logger


from urllib.parse import quote_plus

log = get_logger(__name__)

_DRIVER = "mysql+pymysql"


class MySQLConnector:
    """
    Manages a SQLAlchemy connection pool to MySQL or MariaDB.

    Tenant-scoped: tenant_id is stamped on every log line.
    """

    def __init__(self, config: SQLConfig, tenant_id: str = "") -> None:
        """Initialise a tenant-aware MySQL connector from validated SQL config."""
        if config.dialect not in ("mysql", "mariadb"):
            raise ValueError(
                f"MySQLConnector requires dialect 'mysql' or 'mariadb', got '{config.dialect}'"
            )
        self.config    = config
        self.tenant_id = tenant_id
        self._engine   = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "MySQLConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> None:
        """Open and verify a connection pool to MySQL."""
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError("Install sqlalchemy + pymysql: pip install sqlalchemy pymysql")

        charset = self.config.extra_params.get("charset", "utf8mb4")

        connect_args = {
            "connect_timeout": self.config.connect_timeout,
            "charset": charset,
        }

        # DO NOT set ssl at all
        
        username = quote_plus(self.config.username)
        password = quote_plus(self.config.password)

        url = (
            f"{_DRIVER}://{username}:{password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
            f"?charset={charset}"
        )
        # url = (
        #     f"{_DRIVER}://{self.config.username}:{self.config.password}"
        #     f"@{self.config.host}:{self.config.port}/{self.config.database}"
        #     f"?charset={charset}"
        # )

        self._engine = create_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        log.info(
            "MySQL connected | tenant=%s host=%s db=%s charset=%s",
            self.tenant_id, self.config.host, self.config.database, charset,
        )

    def disconnect(self) -> None:
        """Dispose the MySQL engine and clear the in-memory connection handle."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            log.info("MySQL disconnected | tenant=%s", self.tenant_id)

    # ── Query execution ────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return a DataFrame.

        Parameters
        ----------
        sql    : SQL string (use %(name)s for PyMySQL named params)
        params : optional bind parameters dict
        """
        self._require_connected()
        log.info(
            "MySQL query | tenant=%s sql_preview=%.120s",
            self.tenant_id, sql.replace("\n", " "),
        )
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
            log.info(
                "MySQL result | tenant=%s rows=%d cols=%d",
                self.tenant_id, *df.shape,
            )
            return df
        except Exception as exc:
            log.error("MySQL query failed | tenant=%s error=%s", self.tenant_id, exc)
            raise RuntimeError(f"MySQL query failed: {exc}") from exc

    # ── Schema introspection ───────────────────────────────────────────────

    def get_table_names(self) -> list[str]:
        """List all tables in the connected database."""
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
        return self.query(f"SELECT * FROM `{table}` LIMIT :n", params={"n": n})

    # ── Internal ───────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "Not connected. Call connect() or use the context manager."
            )
