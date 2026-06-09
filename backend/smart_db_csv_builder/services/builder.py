"""
smart_db_csv_builder/services/builder.py — Connection builder

New in v3.

Responsible for instantiating the correct dialect-specific connector from a
SQLConfig or NoSQLConfig object.  Acts as a single dispatch point so that
callers (SQLExecutor, LLMPlanner, API routes) never import individual
connector classes directly.

Supported SQL dialects  : postgres, postgresql, mysql, mariadb, sqlite
Supported NoSQL engines : mongodb, cassandra, dynamodb  (via NoSQLConnector)

Usage
-----
    from backend.config.database import SQLConfig
    from backend.smart_db_csv_builder.services.builder import ConnectionBuilder

    cfg = SQLConfig(dialect="postgres", host="...", ...)
    builder = ConnectionBuilder(cfg, tenant_id="acme-corp")

    with builder.get_connector() as conn:
        df = conn.query("SELECT user_id, item_id FROM interactions LIMIT 100")

    # Introspect available tables
    with builder.get_connector() as conn:
        tables = conn.get_table_names()
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

from backend.config.database import NoSQLConfig, SQLConfig
from backend.utils.logger import get_logger

from backend.smart_db_csv_builder.connectors.mysql import MySQLConnector
from backend.smart_db_csv_builder.connectors.nosql import NoSQLConnector
from backend.smart_db_csv_builder.connectors.postgres import PostgresConnector
from backend.smart_db_csv_builder.connectors.sqlite import SQLiteConnector

log = get_logger(__name__)

# Union type for all supported connector instances
AnyConnector = Union[
    PostgresConnector,
    MySQLConnector,
    SQLiteConnector,
    NoSQLConnector,
]


class ConnectionBuilder:
    """
    Factory that maps a database config to the appropriate connector class.

    Tenant-scoped: tenant_id is forwarded to every connector for audit logging
    and path resolution (SQLite auto-resolves db files under tenants/{id}/data/).
    """

    # SQL dialect → connector class
    _SQL_REGISTRY: dict[str, type] = {
        "postgres":   PostgresConnector,
        "postgresql": PostgresConnector,
        "mysql":      MySQLConnector,
        "mariadb":    MySQLConnector,
        "sqlite":     SQLiteConnector,
    }

    def __init__(
        self,
        config: Union[SQLConfig, NoSQLConfig],
        tenant_id: str = "",
    ) -> None:
        """Capture connection config and tenant context for later connector creation."""
        self.config    = config
        self.tenant_id = tenant_id

    # ── Public API ─────────────────────────────────────────────────────────────

    @contextmanager
    def get_connector(self) -> Iterator[AnyConnector]:
        """
        Context manager that yields an open, ready-to-use connector.

        The connector is disconnected automatically on exit — whether normal
        or exceptional.

        Yields
        ------
        AnyConnector
            An open dialect-specific connector instance.

        Raises
        ------
        ValueError
            If the dialect / engine is not registered.
        """
        connector = self._build()
        log.debug(
            "Opening connector | tenant=%s type=%s",
            self.tenant_id, type(connector).__name__,
        )
        with connector:
            yield connector

    def build_unmanaged(self) -> AnyConnector:
        """
        Return an unconnected connector instance.

        The caller is responsible for calling connect() / disconnect()
        (or using it as a context manager).  Prefer get_connector() for
        most use-cases.
        """
        return self._build()

    def supported_dialects(self) -> list[str]:
        """Return the list of supported SQL dialects."""
        return list(self._SQL_REGISTRY)

    # ── Internal factory ───────────────────────────────────────────────────────

    def _build(self) -> AnyConnector:
        if isinstance(self.config, NoSQLConfig):
            return self._build_nosql()
        return self._build_sql()

    def _build_sql(self) -> Union[PostgresConnector, MySQLConnector, SQLiteConnector]:
        config: SQLConfig = self.config  # type: ignore[assignment]
        dialect = config.dialect.lower()

        connector_cls = self._SQL_REGISTRY.get(dialect)
        if connector_cls is None:
            raise ValueError(
                f"Unsupported SQL dialect '{dialect}'. "
                f"Registered dialects: {list(self._SQL_REGISTRY)}"
            )

        log.info(
            "Building SQL connector | tenant=%s dialect=%s class=%s",
            self.tenant_id, dialect, connector_cls.__name__,
        )

        if connector_cls is SQLiteConnector:
            # SQLite takes a file path, not a full SQLConfig
            db_path = getattr(config, "database", ":memory:") or ":memory:"
            return SQLiteConnector(db_path=db_path, tenant_id=self.tenant_id)

        return connector_cls(config, tenant_id=self.tenant_id)

    def _build_nosql(self) -> NoSQLConnector:
        config: NoSQLConfig = self.config  # type: ignore[assignment]
        log.info(
            "Building NoSQL connector | tenant=%s engine=%s",
            self.tenant_id, config.engine,
        )
        return NoSQLConnector(config, tenant_id=self.tenant_id)
