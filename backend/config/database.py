"""
config/database.py – Database connection profiles (v3)

Changes from v2:
  - TenantSQLConfig / TenantNoSQLConfig: scoped to a single tenant_id
  - ConnectionPool: per-tenant pool registry (create once, reuse)
  - load_sql_config_for_tenant() / load_nosql_config_for_tenant():
      load from tenants/{tid}/config.json, falling back to env vars
  - Global loaders kept for v2 back-compat
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional
import re
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlparse, unquote_plus
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
# ---------------------------------------------------------------------------
# Base dataclasses (unchanged from v2, kept for back-compat)
# ---------------------------------------------------------------------------

@dataclass
class SQLConfig:
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: Optional[str] = None
    ssl: bool = True
    connect_timeout: int = 30
    extra_params: dict = field(default_factory=dict)

    def __post_init__(self):
        # If someone passed a full connection URL as the host, parse it out
        if "://" in self.host:
            raise ValueError(
                f"'host' looks like a full connection URL: {self.host!r}. "
                "Pass the hostname only, e.g. 'myserver.postgres.database.azure.com'"
            )

        # Reject hosts containing URL-structural characters that should never appear
        forbidden = set('@#/?')
        found = forbidden & set(self.host)
        if found:
            raise ValueError(
                f"'host' contains invalid characters {found}: {self.host!r}. "
                "The host field must be a plain hostname or IP address."
            )

        # Validate it looks like a real hostname or IP
        hostname_re = re.compile(
            r'^(?:[a-zA-Z0-9]'
            r'(?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
            r'\.)*[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$'
        )
        if not hostname_re.match(self.host):
            raise ValueError(
                f"'host' does not look like a valid hostname: {self.host!r}"
            )

    def connection_url(self) -> str:
        """Build a percent-safe SQLAlchemy connection URL for the current config."""
        dialect_map = {
            "postgres":  "postgresql+psycopg2",
            "mysql":     "mysql+pymysql",
            "mssql":     "mssql+pyodbc",
            "snowflake": "snowflake",
            "redshift":  "redshift+redshift_connector",
            "sqlite":    "sqlite",
        }
        driver = dialect_map.get(self.dialect, self.dialect)
        if self.dialect == "sqlite":
            return f"sqlite:///{self.database}"

        username = quote_plus(self.username)
        password = quote_plus(self.password)

        return (
            f"{driver}://{username}:{password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def create_engine(self, **engine_kwargs) -> Engine:
        """
        Single authoritative way to get a SQLAlchemy engine from this config.
        Never call create_engine() directly elsewhere in the codebase.
        """
        url = self.connection_url()  # already encodes username/password safely

        # Merge SSL + timeout into connect_args
        connect_args = engine_kwargs.pop("connect_args", {})
        if self.dialect in ("postgres", "redshift"):
            connect_args.setdefault("connect_timeout", self.connect_timeout)
            if self.ssl:
                connect_args.setdefault("sslmode", "require")
        elif self.dialect == "mysql":
            connect_args.setdefault("connect_timeout", self.connect_timeout)
            if self.ssl:
                connect_args.setdefault("ssl", {"ssl_disabled": False})

        return create_engine(
            url,
            connect_args=connect_args,
            **engine_kwargs,
        )

@dataclass
class NoSQLConfig:
    """Configuration for NoSQL database connections."""
    engine: str           # mongodb | cassandra | dynamodb
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    region: Optional[str] = None         # for DynamoDB
    ssl: bool = True
    connect_timeout: int = 30


# ---------------------------------------------------------------------------
# v3 – Tenant-scoped variants
# ---------------------------------------------------------------------------

@dataclass
class TenantSQLConfig(SQLConfig):
    """SQL config bound to a specific tenant."""
    tenant_id: str = ""

    @property
    def tenant_schema(self) -> str:
        """Default schema name derived from tenant when not explicit."""
        return self.schema or f"tenant_{self.tenant_id}"


@dataclass
class TenantNoSQLConfig(NoSQLConfig):
    """NoSQL config bound to a specific tenant."""
    tenant_id: str = ""

    @property
    def tenant_collection_prefix(self) -> str:
        """Return the default collection/table prefix derived from the tenant ID."""
        return f"{self.tenant_id}_"


# ---------------------------------------------------------------------------
# Per-tenant config loader
# ---------------------------------------------------------------------------

def _tenant_config_path(tenant_id: str) -> Optional[Path]:
    """Return the path to tenants/{tid}/config.json if it exists."""
    from backend.config.settings import TENANTS_DIR  # lazy import avoids circular
    p = Path(TENANTS_DIR) / tenant_id / "config.json"
    return p if p.exists() else None


def _parse_host_field(raw: dict) -> dict:
    """
    FIX: If the 'host' field in config.json contains a full connection string
    (e.g. 'postgresql://user:p%40ss@host/db' or 'user:pass@host'), parse it
    via urlparse so that password special characters never corrupt the hostname.
    All other fields in *raw* take precedence over anything extracted from the URL.
    Returns a clean dict with host/port/database/username/password resolved.
    """
    host_val = raw.get("host", "localhost")

    # Only attempt parsing when the value looks like a connection string
    if "://" not in host_val and "@" not in host_val:
        return raw  # nothing to fix, return as-is

    # Ensure urlparse can handle strings that have @-sign but no scheme
    parse_target = host_val if "://" in host_val else f"postgresql://{host_val}"
    parsed = urlparse(parse_target)

    # Build a merged dict: explicit raw keys always win over parsed values
    resolved = dict(raw)
    if parsed.hostname and "host" not in raw or raw.get("host") == host_val:
        resolved["host"] = parsed.hostname or host_val
    if parsed.port and "port" not in raw:
        resolved["port"] = parsed.port
    if parsed.username and not raw.get("username"):
        resolved["username"] = unquote_plus(parsed.username)
    if parsed.password and not raw.get("password"):
        resolved["password"] = unquote_plus(parsed.password)
    if parsed.path and not raw.get("database"):
        resolved["database"] = parsed.path.lstrip("/")

    return resolved


def load_sql_config_for_tenant(tenant_id: str) -> Optional[TenantSQLConfig]:
    """
    Load SQL config for *tenant_id*.

    Resolution order:
      1. tenants/{tenant_id}/config.json  →  "database.sql" key
      2. Environment variables (same as v2 global loader)
    """
    cfg_path = _tenant_config_path(tenant_id)
    if cfg_path:
        raw = json.loads(cfg_path.read_text()).get("database", {}).get("sql", {})
        if raw:
            raw = _parse_host_field(raw)  # FIX: safely resolve host before constructing
            return TenantSQLConfig(
                tenant_id=tenant_id,
                dialect=raw.get("dialect", "postgres"),
                host=raw.get("host", "localhost"),
                port=int(raw.get("port", 5432)),
                database=raw.get("database", ""),
                username=raw.get("username", ""),
                password=raw.get("password", ""),
                schema=raw.get("schema"),
                ssl=raw.get("ssl", True),
                connect_timeout=int(raw.get("connect_timeout", 30)),
            )

    # Fall back to env vars
    global_cfg = load_sql_config_from_env()
    if global_cfg is None:
        return None
    return TenantSQLConfig(
        tenant_id=tenant_id,
        **{k: getattr(global_cfg, k) for k in global_cfg.__dataclass_fields__},
    )


def load_nosql_config_for_tenant(tenant_id: str) -> Optional[TenantNoSQLConfig]:
    """
    Load NoSQL config for *tenant_id*.

    Resolution order:
      1. tenants/{tenant_id}/config.json  →  "database.nosql" key
      2. Environment variables (same as v2 global loader)
    """
    cfg_path = _tenant_config_path(tenant_id)
    if cfg_path:
        raw = json.loads(cfg_path.read_text()).get("database", {}).get("nosql", {})
        if raw:
            return TenantNoSQLConfig(
                tenant_id=tenant_id,
                engine=raw.get("engine", "mongodb"),
                host=raw.get("host", "localhost"),
                port=int(raw.get("port", 27017)),
                database=raw.get("database", ""),
                username=raw.get("username"),
                password=raw.get("password"),
                region=raw.get("region"),
                ssl=raw.get("ssl", True),
            )

    global_cfg = load_nosql_config_from_env()
    if global_cfg is None:
        return None
    return TenantNoSQLConfig(
        tenant_id=tenant_id,
        **{k: getattr(global_cfg, k) for k in global_cfg.__dataclass_fields__},
    )


# ---------------------------------------------------------------------------
# Multi-tenant connection pool registry
# ---------------------------------------------------------------------------

class ConnectionPool:
    """
    Lightweight per-tenant SQLAlchemy engine registry.

    Usage:
        engine = ConnectionPool.get("acme-corp")
        with engine.connect() as conn:
            ...
    """

    _pools: Dict[str, "sqlalchemy.engine.Engine"] = {}

    @classmethod
    def get(cls, tenant_id: str) -> "sqlalchemy.engine.Engine":
        """Return a cached engine for *tenant_id*, creating it if needed."""
        if tenant_id not in cls._pools:
            cls._pools[tenant_id] = cls._create(tenant_id)
        return cls._pools[tenant_id]

    @classmethod
    def _create(cls, tenant_id: str) -> "sqlalchemy.engine.Engine":
        cfg = load_sql_config_for_tenant(tenant_id)
        if cfg is None:
            raise ValueError(
                f"No SQL configuration found for tenant '{tenant_id}'. "
                "Add a 'database.sql' block to tenants/{tenant_id}/config.json "
                "or set DB_* environment variables."
            )

        # FIX: route through cfg.create_engine() so SSL/timeout connect_args
        # are applied consistently and quote_plus encoding is guaranteed.
        return cfg.create_engine(
            pool_size=5,
            max_overflow=10,
            pool_timeout=cfg.connect_timeout,
            pool_pre_ping=True,
        )

    @classmethod
    def invalidate(cls, tenant_id: str) -> None:
        """Remove a cached engine (forces re-creation on next access)."""
        engine = cls._pools.pop(tenant_id, None)
        if engine is not None:
            engine.dispose()

    @classmethod
    def invalidate_all(cls) -> None:
        """Dispose all engines – call on shutdown."""
        for engine in cls._pools.values():
            engine.dispose()
        cls._pools.clear()


# ---------------------------------------------------------------------------
# v2 back-compat – global env-var loaders (unchanged)
# ---------------------------------------------------------------------------

def load_sql_config_from_env() -> Optional[SQLConfig]:
    """Load SQL config from environment variables."""
    dialect = os.getenv("DB_DIALECT")
    if not dialect:
        return None
    return SQLConfig(
        dialect=dialect,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", ""),
        username=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        schema=os.getenv("DB_SCHEMA"),
        ssl=os.getenv("DB_SSL", "true").lower() == "true",
    )


def load_nosql_config_from_env() -> Optional[NoSQLConfig]:
    """Load NoSQL config from environment variables."""
    engine = os.getenv("NOSQL_ENGINE")
    if not engine:
        return None
    return NoSQLConfig(
        engine=engine,
        host=os.getenv("NOSQL_HOST", "localhost"),
        port=int(os.getenv("NOSQL_PORT", "27017")),
        database=os.getenv("NOSQL_DB", ""),
        username=os.getenv("NOSQL_USER"),
        password=os.getenv("NOSQL_PASSWORD"),
        region=os.getenv("AWS_REGION"),
        ssl=os.getenv("NOSQL_SSL", "true").lower() == "true",
    )
