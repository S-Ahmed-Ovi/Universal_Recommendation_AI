"""
smart_db_csv_builder/connectors package

Provides a dialect-dispatch factory so callers never need to import
individual connector classes directly.

    from backend.smart_db_csv_builder.connectors import get_sql_connector, TenantFileLoader

    # SQL — dialect dispatched automatically from SQLConfig.dialect
    with get_sql_connector(sql_cfg, tenant_id="acme-corp") as conn:
        df = conn.query("SELECT user_id, item_id, rating FROM events")

    # File
    loader = TenantFileLoader(tenant_id="acme-corp")
    df = loader.load(uploaded_bytes, filename="ratings.csv", persist=True)

    # NoSQL
    with NoSQLConnector(nosql_cfg, tenant_id="acme-corp") as conn:
        df = conn.fetch_collection("interactions")
"""
from __future__ import annotations

from backend.config.database import SQLConfig

from .csv    import TenantFileLoader
from .nosql  import NoSQLConnector
from .postgres import PostgresConnector
from .mysql    import MySQLConnector
from .sqlite   import SQLiteConnector

__all__ = [
    "TenantFileLoader",
    "NoSQLConnector",
    "PostgresConnector",
    "MySQLConnector",
    "SQLiteConnector",
    "get_sql_connector",
]

# Dialect → connector class mapping
_SQL_CONNECTORS = {
    "postgres":   PostgresConnector,
    "postgresql": PostgresConnector,
    "mysql":      MySQLConnector,
    "mariadb":    MySQLConnector,
    "sqlite":     SQLiteConnector,
}


def get_sql_connector(
    config: SQLConfig,
    tenant_id: str = "",
):
    """
    Return the correct SQL connector instance for config.dialect.

    Raises ValueError for unsupported dialects.
    """
    cls = _SQL_CONNECTORS.get(config.dialect.lower())
    if cls is None:
        supported = sorted(_SQL_CONNECTORS.keys())
        raise ValueError(
            f"Unsupported SQL dialect '{config.dialect}'. "
            f"Supported: {supported}. "
            "For Snowflake / Redshift / MSSQL use the generic DBConnector "
            "from ingestion.db_connector (v2 compatibility shim) until "
            "dedicated connectors are added."
        )
    # SQLiteConnector takes db_path as first arg, not SQLConfig
    if cls is SQLiteConnector:
        db_path = config.database or ":memory:"
        return cls(db_path=db_path, tenant_id=tenant_id)
    return cls(config, tenant_id=tenant_id)